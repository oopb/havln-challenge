"""Lossless, low-duplication replay metadata for one HA-VLN episode.

The existing api.jsonl remains the only API ledger. Images normally point to
canonical observations/obs_XXXX.npz plus a verified rendering recipe; only
images that cannot be reproduced from a saved sensor frame are stored once,
by content digest, under replay/media/. No repeated request/base64 dumps.
"""
import base64
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .perception import depth_image_url, image_url


def _ids(value):
    """Prefer explicit source-frame references, preserving their order."""
    seen = set()
    found = []
    def walk(item):
        if isinstance(item, dict):
            for key in ('source_observation_id', 'observation_id',
                        'start_observation_id'):
                ident = item.get(key)
                if isinstance(ident, str) and ident.startswith('obs_') and ident not in seen:
                    seen.add(ident)
                    found.append(ident)
            for v in item.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(item, list):
            for v in item:
                walk(v)
    walk(value)
    return found


def _text_ids(message):
    content = message.get('content', '')
    texts = [content] if isinstance(content, str) else [
        p.get('text', '') for p in content if isinstance(p, dict)
    ] if isinstance(content, list) else []
    found = []
    for text in texts:
        try:
            found.extend(_ids(json.loads(text)))
        except (ValueError, TypeError):
            continue
    return list(dict.fromkeys(found))


def _image_parts(data_url):
    if not isinstance(data_url, str) or not data_url.startswith('data:image/'):
        raise ValueError('unsupported_replay_image_url')
    header, encoded = data_url.split(',', 1)
    if ';base64' not in header or len(encoded) > 45_000_000:
        raise ValueError('unsupported_replay_image_encoding')
    raw = base64.b64decode(encoded, validate=True)
    return raw, hashlib.sha256(raw).hexdigest()


class ReplayRecorder:
    """Append tiny event references and compact complete wire-input metadata.

    An episode recorder is instantiated once. All emitted events refer to the
    canonical api/steps/tools logs; they never duplicate the corresponding row.
    """
    def __init__(self, directory, camera):
        self.directory = Path(directory)
        self.camera = camera
        self.path = self.directory / 'replay' / 'events.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()
        self.media_dir = self.path.parent / 'media'
        self.by_hash = {}
        self._renditions = {}
        self.api_count = 0
        self.tool_count = 0
        self.event_count = 0

    def event(self, kind, **refs):
        row = dict(sequence=self.event_count, kind=kind, **refs)
        self.event_count += 1
        with self.path.open('a', encoding='utf-8') as out:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

    def observation(self, identifier):
        self.event('observation', observation_id=identifier)

    def action(self, record):
        self.event('action', step=record.get('step'),
                   action=record.get('action'),
                   tool_call_id=record.get('tool_call_id'),
                   observation_id=Path(record.get('input_file', '')).stem,
                   after_id=Path(record.get('post_file', '')).stem)

    def tool(self, record):
        self.event('tool', tool_index=self.tool_count,
                   tool_call_id=record.get('call_id'),
                   decision_id=(record.get('request_context') or {}).get('decision_id'),
                   tool=(record.get('call') or {}).get('tool'))
        self.tool_count += 1

    def api_logged(self, record):
        self.event('api', api_index=self.api_count,
                   role=record.get('role'), purpose=record.get('purpose'),
                   completion_request_id=record.get('completion_request_id'),
                   request_number=record.get('request_number'),
                   decision_id=record.get('decision_id'),
                   tool_call_id=record.get('tool_call_id'),
                   success=record.get('success'),
                   cache_hit=record.get('cache_hit', False))
        self.api_count += 1

    def _observation_renditions(self, identifier):
        if identifier in self._renditions:
            return self._renditions[identifier]
        file = self.directory / 'observations' / (identifier + '.npz')
        if not file.is_file():
            return []
        try:
            with np.load(file, allow_pickle=False) as data:
                frame = SimpleNamespace(rgb=data['rgb'], depth=data['depth'])
                variants = []
                for sensor, max_edge in (('rgb', 640), ('rgb', None), ('depth', 640)):
                    url = (depth_image_url(frame, self.camera, max_edge=640,
                                           display_max_m=5.0)
                           if sensor == 'depth' else
                           image_url(frame, max_edge=max_edge))
                    png, digest = _image_parts(url)
                    variants.append((digest, dict(
                        kind='observation', observation_id=identifier,
                        sensor=sensor, max_edge=max_edge, sha256=digest)))
                self._renditions[identifier] = variants
                return variants
        except (OSError, ValueError, KeyError, TypeError):
            return []

    def media(self, url, candidates):
        raw, digest = _image_parts(url)
        previous = self.by_hash.get(digest)
        if previous is not None:
            return previous
        # Only consider IDs actually mentioned by this request. The model
        # never receives an invented frame association.
        for ident in candidates[:48]:
            for expected, ref in self._observation_renditions(ident):
                self.by_hash.setdefault(expected, ref)
            if digest in self.by_hash:
                return self.by_hash[digest]
        # Unmatched media is saved exactly once; never duplicate it per call.
        self.media_dir.mkdir(exist_ok=True)
        filename = digest + '.png'
        dest = self.media_dir / filename
        if not dest.exists():
            dest.write_bytes(raw)
        ref = dict(kind='stored', path='replay/media/' + filename, sha256=digest)
        self.by_hash[digest] = ref
        return ref

    def request_snapshot(self, messages, context, body):
        """Capture actual assembled messages, including in-session dialogue.

        Existing api.jsonl.messages already stores system and current user text;
        here only prior dialogue, media refs, and protocol parameters are added.
        """
        preferred = []
        if isinstance(context, dict):
            for key in ('source_observation_id', 'observation_id'):
                if isinstance(context.get(key), str):
                    preferred.append(context[key])
            current = context.get('current_rgb') or {}
            if isinstance(current, dict) and current.get('observation_id'):
                preferred.append(current['observation_id'])
            preferred.extend(_ids((context.get('history_media') or {}).get('manifest', [])))
            preferred.extend(_ids(context.get('historical_images', [])))
        context_candidates = list(dict.fromkeys(preferred + _ids(context)))
        prior = []
        current_media = []
        for index, message in enumerate(messages):
            if index == 0:
                continue
            copy_message = copy.deepcopy(message)
            local_ids = _text_ids(message)
            refs = []
            content = copy_message.get('content')
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict) or part.get('type') != 'image_url':
                        continue
                    value = (part.get('image_url') or {}).get('url')
                    try:
                        ref = self.media(value, local_ids + context_candidates)
                    except (ValueError, OSError, TypeError) as error:
                        ref = dict(kind='unavailable', reason=type(error).__name__)
                    part['image_url'] = dict(replay_media=ref)
                    refs.append(ref)
            if index == len(messages) - 1:
                current_media = refs
            else:
                prior.append(copy_message)
        return dict(version=1, prior_messages=prior, current_media=current_media,
                    request_parameters={k: copy.deepcopy(body[k]) for k in (
                        'model', 'response_format', 'temperature', 'max_tokens',
                        'thinking') if k in body},
                    input_message_order='system_then_prior_messages_then_current_user',
                    media_policy='canonical_observation_verified_or_content_addressed')
