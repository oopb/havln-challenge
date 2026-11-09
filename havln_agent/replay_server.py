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
                raise ValueError('Archive contains no traces/')
            self.prefix = matches[0]
        elif not (self.path / 'traces').is_dir():
            raise ValueError('Expected run output with traces/ directory')
        else:
            self.prefix = ''

    def files(self):
        if self.zip:
            return [n[len(self.prefix):] for n in self.names
                    if n.startswith(self.prefix)]
        return [str(p.relative_to(self.path)) for p in
                (self.path / 'traces').rglob('*') if p.is_file()]

    def read(self, name):
        if name.startswith('/') or '..' in Path(name).parts:
            return None
        if self.zip:
            filename = self.prefix + name
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

    def episodes(self):
        names = set()
        for file in self.files():
            parts = file.split('/')
            if len(parts) >= 3 and parts[0] == 'traces' and _SAFE.fullmatch(parts[1]):
                names.add(parts[1])
        return sorted(names)

    def trace(self, episode, name):
        if not _SAFE.fullmatch(episode):
            raise KeyError('Invalid episode')
        return self.read('traces/' + episode + '/' + name)

    def json(self, episode, name):
        return _read_json(self.trace(episode, name), {})

    def lines(self, episode, name):
        return _jsonl(self.trace(episode, name))


class EpisodeReplay:
    def __init__(self, source, episode):
        if episode not in source.episodes():
            raise KeyError('Unknown episode')
        self.source, self.episode = source, episode
        self.api = source.lines(episode, 'api.jsonl')
        self.steps = source.lines(episode, 'steps.jsonl')
        self.tools = source.lines(episode, 'tools.jsonl')
        self.events_log = source.lines(episode, 'events.jsonl')
        self.meta = source.json(episode, 'episode.json')
        self.summary = source.json(episode, 'summary.json')
        self.failure = source.json(episode, 'failure.json')
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
        self.events = [dict(sequence=i, **{k: v for k, v in ev.items()
                                           if k != 'sequence'})
                       for i, ev in enumerate(self.events)]

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
            previews.append(p)
        return dict(episode=self.episode, metadata=self.meta,
                    summary=self.summary, failure=self.failure,
                    counts=dict(events=len(previews), api=len(self.api),
                                actions=len(self.steps), tools=len(self.tools)),
                    precise_order=self.precise_order, events=previews)

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


def serve(source_path, host='127.0.0.1', port=8765):
    source = ReplaySource(source_path)
    episodes = source.episodes()
    if not episodes:
        raise ValueError('No episode traces found')
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
                if len(parts) >= 3 and parts[:2] == ['api', 'episode']:
                    episode = parts[2]
                    data = current(episode)
                    if len(parts) == 4 and parts[3] == 'timeline':
                        return self._json(data.overview())
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
