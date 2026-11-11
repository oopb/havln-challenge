"""Submission format helpers shared by the API Agent and its Phase 2 merger.

These contract checks are independent of the released CMA checkpoint, source
bootstrap and Habitat runtime. Keep the public six-action vocabulary unchanged.
"""
import hashlib
import json
import os
from pathlib import Path

SPLITS = ('val_seen', 'val_unseen')
ACTIONS = ('STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def check_rows(rows, expected):
    ids = [row['episode_id'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('Export episode coverage is not exact')
    for row in rows:
        actions = row['actions']
        if (not 1 <= len(actions) <= 500
                or any(action not in ACTIONS for action in actions)
                or (len(actions) < 500 and actions[-1] != 'STOP')
                or 'STOP' in actions[:-1]):
            raise ValueError('Exported action sequence violates the task contract')
