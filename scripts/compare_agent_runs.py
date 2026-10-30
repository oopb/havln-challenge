"""Read-only RGB-D audit and same-episode comparison. Does not call an API."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from havln_agent.observation import ObservationAudit, frame_info
from havln_agent.perception import Camera, Observation, Geometry
from havln_agent.waypoint import generate_waypoints
from scripts.export_cma_submission import ACTIONS, atomic_json


def inspect_run(root, episode):
    directory = root / 'traces' / episode
    summary = json.loads((directory / 'summary.json').read_text())
    records = [json.loads(line) for line in (directory / 'steps.jsonl').read_text().splitlines()]
    api = [json.loads(line) for line in (directory / 'api.jsonl').read_text().splitlines()]
    camera = Camera(**records[0]['geometry']['camera'])
    audit, frames, rejections = ObservationAudit(camera), [], Counter()
    identity_path = root / 'export_metadata.json'
    identity = json.loads(identity_path.read_text())['identity'] if identity_path.exists() else {}
    nav = identity.get('config', {}).get('runtime', {})
    previous_action = None
    turn_anomalies, saved_hash_mismatches = [], []
    for index, record in enumerate(records):
        with np.load(directory / record['input_file']) as data:
            observation = Observation(data['rgb'], data['depth'], '', 0,
                float(data['pitch_deg']) if 'pitch_deg' in data else 0)
        frame = audit.update(observation, record['pose'], previous_action)
        frame.update(step=record['step'], file=record['input_file'])
        for name in ('rgb_hash', 'depth_hash'):
            if record.get('observation', {}).get(name, frame[name]) != frame[name]:
                saved_hash_mismatches.append(dict(step=record['step'], field=name))
        geometry = Geometry(observation, camera, record['geometry'].get('safety_margin', nav.get('safety_margin', 0.15)))
        diagnostics = []
        generate_waypoints(geometry, nav.get('waypoint_distance', 1.25), diagnostics)
        frame['geometry_recomputed_without_semantics'] = diagnostics
        rejections.update(reason for item in diagnostics for reason in item['rejection_reasons'])
        next_file = record.get('post_file') or (records[index + 1]['input_file'] if index + 1 < len(records) else 'terminal.npz')
        with np.load(directory / next_file) as data:
            after = frame_info(Observation(data['rgb'], data['depth'], '', 0), camera)
        for name in ('rgb_hash', 'depth_hash'):
            if record.get('post_observation', {}).get(name, after[name]) != after[name]:
                saved_hash_mismatches.append(dict(step=record['step'], field='post_' + name))
        unchanged = all(frame[name] == after[name] for name in ('rgb_hash', 'depth_hash'))
        if record['action'] in ('TURN_LEFT', 'TURN_RIGHT') and unchanged:
            turn_anomalies.append(record['step'])
        frames.append(frame)
        previous_action = record['action']
    scenes = [item for item in api if item['purpose'] == 'scene' and 'result' in item]
    hits = sum(bool(item.get('cache_hit')) for item in api)
    scene_hits = sum(bool(item.get('cache_hit')) for item in scenes)
    metrics = dict(
        validation_scope=identity.get('validation_scope', 'UNLABELED_SAVED_TRACE'),
        actions=len(records), action_counts=dict(Counter(r['action'] for r in records)),
        move_forward=sum(r['action'] == 'MOVE_FORWARD' for r in records),
        unique_pose_directions=audit.stats()['unique_pose_directions'],
        repeated_direction_observations=audit.repeated_directions,
        repeated_identical_views=audit.repeated_views,
        waypoints_generated=sum(len(r['candidates']) for r in records),
        completed_subgoals=summary['completed_subgoals'],
        human_detections=sum(len(item['result']['humans']) for item in scenes),
        uncertain_humanoid_detections=sum(len(item['result'].get('uncertain_humanoids', [])) for item in scenes),
        scene_results=len(scenes), api_requests=summary['api']['requests'], cache_hits=hits,
        scene_cache_hits=scene_hits, scene_cache_hit_rate=scene_hits / max(1, len(scenes)),
        api_calls_saved=hits, stop_reason=summary['stop_reason'], seconds=summary['seconds'],
        unchanged_turn_steps=turn_anomalies, saved_hash_mismatches=saved_hash_mismatches,
        illegal_actions=[r['action'] for r in records if r['action'] not in ACTIONS],
        recomputed_rejection_reasons=dict(rejections))
    metrics['memory'] = summary.get('memory')
    metrics['api_budget'] = summary['api'].get('budget')
    for field in ('finish_reason','first_forward_step','net_displacement_m_est','task_reviews'):
        metrics[field] = summary.get(field)
    metrics['action_conversion_reasons'] = dict(Counter(
        c['reason'] for r in records for c in r.get('action_changes', []) if c['from_action'] in ACTIONS))
    metrics['executor_action_mismatches'] = [r['step'] for r in records if r.get('executed_action', r['action']) != r['action']]
    metrics['action_pipeline_errors'] = [r['step'] for r in records if 'action_pipeline' in r and (
        not r['action_pipeline'] or r['action_pipeline'][-1]['action'] != r['action'] or
        any(not stage.get('reason') for stage in r['action_pipeline']) or
        r['proposed_action'] != r['action'] and not r.get('action_changes'))]
    metrics['action_pipeline_audit_available'] = all('action_pipeline' in r for r in records)
    ledger_path = directory / 'tool_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    metrics['tool_ledger_available'] = ledger_path.exists()
    metrics['unresolved_tool_calls'] = sum(not r['resolved'] for r in ledger) if ledger_path.exists() else None
    metrics['tool_failure_kinds'] = dict(Counter(r['result'].get('failure_kind', 'unrecorded')
                                                for r in ledger if not r['resolved']))
    metrics['path_revalidation_calls'] = sum('validation' in r['result'] for r in ledger) if ledger_path.exists() else None
    metrics['rgbd_motion_estimates'] = dict(Counter((r.get('motion_evidence') or {}).get('status', 'UNKNOWN')
                                                  if 'motion_evidence' in r else 'UNRECORDED'
                                                  for r in records if r['action'] == 'MOVE_FORWARD'))
    world_path = directory / 'state.jsonl'
    worlds = [json.loads(s) for s in world_path.read_text().splitlines()] if world_path.exists() else []
    metrics['state_snapshots'] = len(worlds)
    metrics['scene_validity_states'] = dict(Counter(w['scene_state']['validity']['status'] for w in worlds))
    metrics['progress_state'] = (json.loads((directory / 'final_state.json').read_text())['progress_state']
                                 if (directory / 'final_state.json').exists() else None)
    return dict(root=str(root), episode=episode, identity=identity, metrics=metrics, frames=frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path, nargs='?')
    parser.add_argument('--episode', default='val_seen-1')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--details', type=Path)
    args = parser.parse_args()
    before = inspect_run(args.before, args.episode)
    result = dict(episode=args.episode, before=before['metrics'])
    details = dict(before=before)
    if args.after:
        after = inspect_run(args.after, args.episode)
        details['after'] = after
        result['after'] = after['metrics']
        keys = ('seed', 'datasets', 'config', 'splits', 'episode_limit', 'runtime_config',
                'resolved_runtime_config', 'human_annotations', 'simulator', 'environment')
        result['identity_matches'] = {key: before['identity'].get(key) == after['identity'].get(key)
                                      for key in keys if key in before['identity'] or key in after['identity']}
        result['caution'] = 'Human animation uses wall-clock time; matching episode/seed is not identical dynamic state. More actions alone do not prove improvement.'
    if args.report:
        atomic_json(args.report, result)
    if args.details:
        atomic_json(args.details, details)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
