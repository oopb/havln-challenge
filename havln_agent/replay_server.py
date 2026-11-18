"""Read-only offline HA-VLN replay API, supporting run directories and ZIPs.

No simulator, credentials, model API, mutable files or background job.
Run with: python scripts/serve_agent_replay.py --output-dir /workspace/agent-runs/eval-batch4-30
"""
import base64
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from zipfile import ZipFile

import numpy as np
from PIL import Image

from .perception import depth_image_url
from .competition_scoring import official_formula, summarize_run

_SAFE = re.compile(r'^[A-Za-z0-9_.-]+$')
_HASH = re.compile(r'^[a-f0-9]{64}$')


def _read_json(raw, fallback=None):
    try:
        return json.loads(raw.decode('utf-8')) if raw else fallback
    except (ValueError, UnicodeError):
        return fallback


def _jsonl(raw):
    if not raw:
        return []
    results = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                results.append(row)
        except (ValueError, UnicodeError):
            continue
    return results


class ReplaySource:
    """Read from a selected output-dir or a diagnostic ZIP, never write there."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.zip = ZipFile(self.path) if self.path.is_file() else None
        self.names = set(self.zip.namelist()) if self.zip else None
        if self.zip:
            matches = [n.split('/traces/', 1)[0] + '/' for n in self.names
                       if '/traces/' in n]
            if not matches:
                matches = [n.split('/codex/', 1)[0] + '/' for n in self.names
                           if n.endswith('/codex/codex_events.jsonl')]
            if not matches:
                raise ValueError('Archive contains neither traces/ nor a Codex startup audit')
            self.prefix = matches[0]
        elif not ((self.path / 'traces').is_dir() or
                  (self.path / 'codex' / 'codex_events.jsonl').is_file()):
            raise ValueError('Expected traces/ or Codex startup audit')
        else:
            self.prefix = ''

    def files(self):
        if self.zip:
            return [n[len(self.prefix):] for n in self.names
                    if n.startswith(self.prefix)]
        if (self.path / 'traces').is_dir():
            return [str(p.relative_to(self.path)) for p in
                    (self.path / 'traces').rglob('*') if p.is_file()]
        return ['codex/codex_events.jsonl'] if (
            self.path / 'codex' / 'codex_events.jsonl').is_file() else []

    def read(self, name):
        if name.startswith('/') or '..' in Path(name).parts:
            return None
        if self.zip:
            filename = self.prefix + name
            if filename not in self.names:
                # A packed archive may omit byte-identical RGB-D snapshots.
                # Resolve only a same-episode, allowlisted observation alias.
                parts = name.split('/')
                if (len(parts) != 4 or parts[0] != 'traces' or
                        parts[2] != 'observations' or
                        not re.fullmatch(r'obs_[0-9]+\\.npz', parts[3])):
                    return None
                manifest_path = self.prefix + 'diagnostic_manifest.json'
                if manifest_path not in self.names:
                    return None
                info = self.zip.getinfo(manifest_path)
                if info.file_size > 16 * 1024 * 1024:
                    return None
                manifest = _read_json(self.zip.read(manifest_path), {})
                alias = ((manifest.get('frame_aliases') or {})
                         .get(parts[1], {}).get(parts[3]))
                if not isinstance(alias, str) or not re.fullmatch(
                        r'obs_[0-9]+\\.npz', alias):
                    return None
                filename = (self.prefix + 'traces/' + parts[1] +
                            '/observations/' + alias)
                if filename not in self.names:
                    return None
            info = self.zip.getinfo(filename)
            if info.file_size > 256 * 1024 * 1024:
                return None
            return self.zip.read(filename)
        file = self.path / name
        if not file.is_file():
            return None
        return file.read_bytes()

    @lru_cache(maxsize=1)
    def episodes(self):
        # Avoid enumerating every saved RGB-D frame in large batch runs.
        if not self.zip:
            traces = self.path / 'traces'
            names = sorted(p.name for p in traces.iterdir()
                           if p.is_dir() and _SAFE.fullmatch(p.name)) if traces.is_dir() else []
        else:
            names = sorted({p[1] for name in self.names
                            if (p := name[len(self.prefix):].split('/')) and
                            len(p) >= 3 and p[0] == 'traces' and _SAFE.fullmatch(p[1])
                            and name.startswith(self.prefix)})
        if not names and self.read('codex/codex_events.jsonl') is not None:
            return ['codex-startup']
        return names

    def trace(self, episode, name):
        if not _SAFE.fullmatch(episode):
            raise KeyError('Invalid episode')
        if episode == 'codex-startup':
            return self.read(name)
        return self.read('traces/' + episode + '/' + name)

    def json(self, episode, name):
        return _read_json(self.trace(episode, name), {})

    def lines(self, episode, name):
        return _jsonl(self.trace(episode, name))


def _codex_event_for_view(row):
    """Hide base64 image bytes in HTTP JSON; original CLI log remains lossless."""
    if not isinstance(row, dict):
        return row
    event = dict(row)
    item = event.get('item')
    if not isinstance(item, dict):
        return event
    item = dict(item)
    result = item.get('result')
    if isinstance(result, dict) and isinstance(result.get('content'), list):
        result = dict(result)
        parts = []
        for part in result['content']:
            if isinstance(part, dict) and part.get('type') == 'image':
                visual = {k: v for k, v in part.items() if k != 'data'}
                visual['data_omitted_from_viewer'] = True
                visual['base64_characters'] = len(part.get('data') or '')
                parts.append(visual)
            else:
                parts.append(part)
        result['content'] = parts
        item['result'] = result
    event['item'] = item
    return event


class EpisodeReplay:
    def __init__(self, source, episode):
        if episode not in source.episodes():
            raise KeyError('Unknown episode')
        self.source, self.episode = source, episode
        self.api = source.lines(episode, 'api.jsonl')
        self.steps = source.lines(episode, 'steps.jsonl')
        self.tools = source.lines(episode, 'tools.jsonl')
        self.events_log = source.lines(episode, 'events.jsonl')
        # Codex CLI JSON events are a separate, ordered model/tool stream.
        # Do not fabricate a cross-stream clock with Habitat action events.
        self.codex = source.lines(episode, 'codex/codex_events.jsonl')
        self.codex_run_status = source.json(episode, 'codex/run_status.json')
        self.codex_usage = source.json(episode, 'codex/codex_usage.json')
        prompt = source.trace(episode, 'codex/prompt.txt')
        self.codex_prompt = (prompt.decode('utf-8', errors='replace')
                             if prompt is not None else None)
        self.meta = source.json(episode, 'episode.json')
        self.summary = source.json(episode, 'summary.json')
        self.failure = source.json(episode, 'failure.json')
        if episode == 'codex-startup':
            self.failure = _read_json(source.read('run_status.json'), {})
        self.memory = source.json(episode, 'memory.json')
        self.camera = self.meta.get('camera')
        if not self.camera:
            self.camera = next((row.get('geometry', {}).get('camera')
                                for row in self.steps
                                if isinstance(row.get('geometry'), dict)
                                and row['geometry'].get('camera')), None)
        self.events = source.lines(episode, 'replay/events.jsonl')
        self.precise_order = bool(self.events)
        if not self.events:
            self.events = self._legacy_events()
        self.habitat_event_count = len(self.events)
        self.codex_links = self._match_codex_tool_receipts()
        # Appended as a clearly separate event group, never asserted to be
        # interleaved in causal order with Habitat's recorded action sequence.
        self.events.extend(dict(kind='codex', codex_index=i)
                           for i in range(len(self.codex)))
        self.events = [dict(sequence=i, **{k: v for k, v in ev.items()
                                           if k != 'sequence'})
                       for i, ev in enumerate(self.events)]

    def _match_codex_tool_receipts(self):
        """Join Codex MCP calls to Habitat receipts only when arguments agree.

        Codex item IDs differ from the simulator call IDs. The defensible
        join in v1 is ordered tool sequence, verified by names and arguments.
        Stop linking on discrepancy instead of showing the wrong camera.
        """
        started = []
        for event in self.codex:
            item = event.get('item')
            if (event.get('type') == 'item.started' and
                    isinstance(item, dict) and
                    item.get('type') == 'mcp_tool_call' and
                    item.get('server') == 'havln'):
                started.append(item)
        steps_by_call = {}
        for step in self.steps:
            ident = step.get('tool_call_id')
            if isinstance(ident, str):
                steps_by_call.setdefault(ident, []).append(step)
        links = {}
        for item, record in zip(started, self.tools):
            call = record.get('call')
            if not isinstance(call, dict):
                break
            name = item.get('tool')
            arguments = item.get('arguments') or {}
            if (not isinstance(arguments, dict) or
                    name not in ('observe', 'step') or
                    name != call.get('tool') or
                    arguments != (call.get('args') or {})):
                break
            item_id, call_id = item.get('id'), record.get('call_id')
            if not isinstance(item_id, str) or not isinstance(call_id, str):
                break
            result = record.get('result') or {}
            if not isinstance(result, dict):
                result = {}
            linked = dict(linked_tool=name, tool_call_id=call_id)
            if name == 'observe':
                observation_id = result.get('observation_id')
                if isinstance(observation_id, str) and re.fullmatch(
                        r'obs_[0-9]+', observation_id):
                    linked['observation_id'] = observation_id
            else:
                actions = steps_by_call.get(call_id) or []
                if actions:
                    before = Path(actions[0].get('input_file') or '').stem
                    after = Path(actions[-1].get('post_file') or '').stem
                    if re.fullmatch(r'obs_[0-9]+', before):
                        linked['observation_id'] = before
                    if re.fullmatch(r'obs_[0-9]+', after):
                        linked['after_id'] = after
            links[item_id] = linked
        return links

    def _legacy_events(self):
        """Best-effort old-log view: no fabricated cross-stream order."""
        events = []
        # The legacy trace has no shared event clock. Keep within-stream order
        # and label the resulting merged view as approximate.
        for i, row in enumerate(self.api):
            events.append(dict(kind='api', api_index=i,
                               role=row.get('role'), purpose=row.get('purpose'),
                               decision_id=row.get('decision_id')))
        for i, row in enumerate(self.steps):
            events.append(dict(kind='action', step=i, action=row.get('action'),
                               tool_call_id=row.get('tool_call_id')))
        for i, row in enumerate(self.tools):
            events.append(dict(kind='tool', tool_index=i,
                               tool_call_id=row.get('call_id'),
                               tool=(row.get('call') or {}).get('tool')))
        return events

    def _api_observation(self, row):
        messages = row.get('messages') or []
        if len(messages) > 1:
            try:
                ctx = json.loads(messages[1].get('content', ''))
                for key in ('source_observation_id', 'observation_id'):
                    if isinstance(ctx.get(key), str):
                        return ctx[key]
                current = ctx.get('current_rgb') or {}
                if current.get('observation_id'):
                    return current['observation_id']
                manifest = (ctx.get('history_media') or {}).get('manifest') or []
                if manifest:
                    return manifest[0].get('observation_id')
            except (TypeError, ValueError, AttributeError):
                pass
        return None

    def overview(self):
        previews = []
        for ev in self.events:
            p = dict(ev)
            kind = ev.get('kind')
            if kind == 'api':
                idx = ev.get('api_index')
                if isinstance(idx, int) and 0 <= idx < len(self.api):
                    row = self.api[idx]
                    p.update(model=row.get('model'), success=row.get('success'),
                             error=row.get('error'),
                             observation_id=self._api_observation(row),
                             attempt_number=row.get('attempt_number'),
                             cache_hit=row.get('cache_hit', False))
            elif kind == 'action':
                idx = ev.get('step')
                if isinstance(idx, int) and 0 <= idx < len(self.steps):
                    row = self.steps[idx]
                    p.update(action=row.get('action'),
                             observation_id=Path(row.get('input_file') or '').stem or None,
                             after_id=Path(row.get('post_file') or '').stem or None)
            elif kind == 'tool':
                idx = ev.get('tool_index')
                if isinstance(idx, int) and 0 <= idx < len(self.tools):
                    row = self.tools[idx]
                    p.update(status=(row.get('result') or {}).get('status'),
                             tool=(row.get('call') or {}).get('tool'))
            elif kind == 'codex':
                idx = ev.get('codex_index')
                if isinstance(idx, int) and 0 <= idx < len(self.codex):
                    row = self.codex[idx]
                    item = row.get('item') or {}
                    if not isinstance(item, dict):
                        item = {}
                    response = item.get('result') or {}
                    if not isinstance(response, dict):
                        response = {}
                    p.update(event_type=row.get('type', 'event'),
                             item_type=item.get('type'),
                             status=item.get('status', row.get('status')),
                             error=bool(row.get('error') or item.get('error') or
                                        item.get('is_error') or
                                        response.get('isError') or
                                        response.get('is_error')),
                             item_id=item.get('id'))
                    linked = self.codex_links.get(item.get('id'))
                    if linked:
                        # observe() images exist only upon tool completion.
                        if (linked['linked_tool'] == 'observe' and
                                row.get('type') != 'item.completed'):
                            linked = {k: v for k, v in linked.items()
                                      if k != 'observation_id'}
                        # step() returns receipts, not a new image.
                        if (linked['linked_tool'] == 'step' and
                                row.get('type') != 'item.completed'):
                            linked = {k: v for k, v in linked.items()
                                      if k != 'after_id'}
                        p.update(linked)
            previews.append(p)
        return dict(episode=self.episode, metadata=self.meta,
                    summary=self.summary, failure=self.failure,
                    counts=dict(events=len(previews), api=len(self.api),
                                actions=len(self.steps), tools=len(self.tools),
                                codex=len(self.codex)),
                    precise_order=self.precise_order,
                    codex_independent_timeline=bool(self.codex),
                    codex_prompt=self.codex_prompt,
                    codex_run_status=self.codex_run_status,
                    codex_usage=self.codex_usage,
                    events=previews)

    def detail(self, index):
        if not 0 <= index < len(self.events):
            raise KeyError('Unknown event')
        ev = self.events[index]
        kind = ev.get('kind')
        row = None
        if kind == 'api':
            idx = ev.get('api_index')
            if isinstance(idx, int) and 0 <= idx < len(self.api):
                row = self.api[idx]
        elif kind == 'action':
            idx = ev.get('step')
            if isinstance(idx, int) and 0 <= idx < len(self.steps):
                row = self.steps[idx]
        elif kind == 'tool':
            idx = ev.get('tool_index')
            if isinstance(idx, int) and 0 <= idx < len(self.tools):
                row = self.tools[idx]
        elif kind == 'codex':
            idx = ev.get('codex_index')
            if isinstance(idx, int) and 0 <= idx < len(self.codex):
                row = _codex_event_for_view(self.codex[idx])
        elif kind == 'runtime_event':
            idx = ev.get('event_index')
            if isinstance(idx, int) and 0 <= idx < len(self.events_log):
                row = self.events_log[idx]
        return dict(event=ev, record=row, replay_supported=bool(
            row and isinstance(row.get('replay'), dict)) if kind == 'api' else None)

    @lru_cache(maxsize=36)
    def frame(self, identifier, channel='rgb', max_edge=0):
        if not re.fullmatch(r'obs_[0-9]+', identifier):
            raise KeyError('Invalid frame')
        raw = self.source.trace(self.episode, 'observations/' + identifier + '.npz')
        if raw is None:
            raise FileNotFoundError('Frame not saved or omitted from compact archive')
        with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
            rgb, depth = npz['rgb'], npz['depth']
        if channel == 'rgb':
            image = Image.fromarray(rgb)
            if max_edge and max(image.size) > max_edge:
                image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            return buffer.getvalue()
        if channel != 'depth':
            raise KeyError('Invalid channel')
        if self.camera:
            from types import SimpleNamespace
            frame = SimpleNamespace(rgb=rgb, depth=depth)
            from .perception import Camera
            camera = Camera(**self.camera)
            url = depth_image_url(frame, camera, max_edge=max_edge or 640,
                                  display_max_m=5.0)
            return base64.b64decode(url.split(',', 1)[1])
        # Older sparse trace without calibration: explicitly uncalibrated.
        depth = np.asarray(depth, dtype=np.float64).squeeze()
        ok = np.isfinite(depth)
        gray = np.zeros(depth.shape, dtype=np.uint8)
        if ok.any():
            near, far = np.percentile(depth[ok], [2, 98])
            gray[ok] = np.clip((depth[ok] - near) /
                              max(far - near, 1e-6) * 255, 0, 255).astype('uint8')
        image = Image.fromarray(gray)
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        return buf.getvalue()

    def stored_image(self, digest):
        if not _HASH.fullmatch(digest):
            raise KeyError('Invalid digest')
        raw = self.source.trace(self.episode, 'replay/media/' + digest + '.png')
        if not raw or hashlib.sha256(raw).hexdigest() != digest:
            raise FileNotFoundError('Referenced image unavailable')
        return raw


def _split_for_episode(episode):
    for split in ('val_unseen', 'val_seen', 'train', 'test'):
        if episode.startswith(split + '-'):
            return split
    return None


def _load_scoring_report(source, metrics_path=None):
    if metrics_path is not None:
        file = Path(metrics_path).resolve()
        if not file.is_file() or file.stat().st_size > 32 * 1024 * 1024:
            raise ValueError('Metrics file missing or exceeds 32 MiB')
        raw = file.read_bytes()
        source_name = file.name
    else:
        # The only automatically discovered metric report is explicit and
        # separate from model-generated summaries or action diagnostics.
        raw = source.read('score_metrics.json')
        source_name = 'score_metrics.json' if raw is not None else None
    if raw is None:
        return None, None
    report = _read_json(raw)
    if not isinstance(report, dict):
        raise ValueError('Metrics file must contain a valid JSON object')
    return report, source_name


def serve(source_path, host='127.0.0.1', port=8765, metrics_path=None):
    source = ReplaySource(source_path)
    episodes = source.episodes()
    if not episodes:
        raise ValueError('No episode traces found')
    report, source_name = _load_scoring_report(source, metrics_path)
    grouped = {}
    for episode_id in episodes:
        split = _split_for_episode(episode_id)
        if split:
            # Episode id in traces is split-episode_id, preserve entire suffix.
            grouped.setdefault(split, []).append(episode_id[len(split)+1:])
    scoring = summarize_run(grouped, report, source_name)
    cache = {}
    assets = Path(__file__).resolve().parents[1] / 'replay_ui'

    def current(episode):
        if episode not in cache:
            cache[episode] = EpisodeReplay(source, episode)
        return cache[episode]

    class Handler(BaseHTTPRequestHandler):
        def _send(self, data, mime='application/json', status=200):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def _json(self, value):
            self._send(json.dumps(value, ensure_ascii=False,
                                  allow_nan=False).encode('utf-8'))

        def do_GET(self):
            try:
                url = urlsplit(self.path)
                parts = [p for p in url.path.split('/') if p]
                query = parse_qs(url.query)
                if parts == ['api', 'episodes']:
                    return self._json(dict(episodes=episodes,
                                           source=str(source.path)))
                if parts == ['api', 'scoring']:
                    return self._json(scoring)
                if parts == ['api', 'score', 'calculate']:
                    values = {}
                    for field in ('sr', 'ne', 'cr', 'tcr'):
                        raw = query.get(field, [])
                        if len(raw) != 1:
                            raise ValueError('Missing %s' % field.upper())
                        values[field] = float(raw[0])
                    return self._json(dict(
                        status='formula_preview',
                        official=False,
                        calculation=official_formula(**values)))
                if len(parts) >= 3 and parts[:2] == ['api', 'episode']:
                    episode = parts[2]
                    data = current(episode)
                    if len(parts) == 4 and parts[3] == 'timeline':
                        overview = data.overview()
                        split = _split_for_episode(episode)
                        overview['scoring_split'] = split
                        if split:
                            summary = scoring['by_split'].get(split)
                            if summary:
                                # Only return this episode's evidence, not an
                                # unrelated full split record for every visit.
                                scoped = dict(summary)
                                scoped['per_episode'] = (
                                    summary.get('per_episode', {}).get(
                                        episode[len(split)+1:]))
                                overview['scoring'] = scoped
                        return self._json(overview)
                    if len(parts) == 5 and parts[3] == 'event':
                        return self._json(data.detail(int(parts[4])))
                    if len(parts) == 5 and parts[3] == 'frame':
                        channel = query.get('channel', ['rgb'])[0]
                        edge = int(query.get('max_edge', ['0'])[0])
                        if edge < 0 or edge > 4096:
                            raise ValueError('Invalid max_edge')
                        return self._send(data.frame(parts[4], channel, edge),
                                          'image/png')
                    if len(parts) == 5 and parts[3] == 'media':
                        return self._send(data.stored_image(parts[4]), 'image/png')
                if not parts:
                    filename, mime = 'index.html', 'text/html; charset=utf-8'
                elif len(parts) == 1 and parts[0] in ('app.js', 'style.css'):
                    filename = parts[0]
                    mime = 'text/javascript; charset=utf-8' if filename.endswith('.js') else 'text/css; charset=utf-8'
                else:
                    raise KeyError('Not found')
                return self._send((assets / filename).read_bytes(), mime)
            except (KeyError, ValueError, FileNotFoundError, TypeError) as error:
                self._json_error(404 if isinstance(error, (KeyError, FileNotFoundError)) else 400,
                                 str(error))
            except (OSError, RuntimeError) as error:
                self._json_error(500, str(error))

        def _json_error(self, status, message):
            value = json.dumps(dict(error=message)).encode('utf-8')
            self._send(value, status=status)

    server = ThreadingHTTPServer((host, port), Handler)
    print('HA-VLN Replay Studio: http://%s:%d (episodes: %d)' %
          (host, server.server_port, len(episodes)), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if source.zip:
            source.zip.close()
