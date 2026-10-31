"""Bounded, goal-driven camera scanning.

Only the Navigation Agent chooses the sweep goal and area. This controller owns
the atomic TURN/LOOK sequence and checkpoint scheduling; the VLM only supplies
visual hypotheses. No MOVE_FORWARD, semantic completion, or safety override is
possible here. Each atomic action is charged to the normal Habitat budget.
"""
import copy
import math
import time

from .api_client import APIError, BudgetExceeded
from .local_exploration import (covered_view, last_block, local_coverage,
                                predicted_scan, bearing_from_pose)


TARGET_TYPES = ('stairs_down', 'stair_entry', 'walkable_approach', 'corridor',
                'door', 'goal_landmark')


def meaningful_candidate(result, targets):
    """A visible, localized, confident hypothesis worth escalating upstairs."""
    return (result.get('target_visible') is True
            and result.get('target_type') in targets
            and result.get('confidence') in ('medium', 'high')
            and result.get('box') is not None
            and bool((result.get('evidence') or '').strip()))


def comparable_scan_pose(a, b):
    """Tolerate floating-point yaw wrap; reject translated or re-anchored views."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if a.get('motion_epoch') != b.get('motion_epoch'):
        return False
    if math.hypot(a.get('x',1e6)-b.get('x',-1e6),
                  a.get('z',1e6)-b.get('z',-1e6)) > .03:
        return False
    delta=(a.get('heading',0)-b.get('heading',0)+math.pi)%(2*math.pi)-math.pi
    return abs(delta)<math.radians(1) and abs(a.get('pitch_deg',0)-b.get('pitch_deg',0))<1


class ObservationController:
    """Synchronously scan in one direction; each VLM checkpoint is optional."""

    def __init__(self, runtime):
        self.runtime = runtime

    def _existing_result(self, target_types, objective):
        r = self.runtime
        matches = set(r.memory.current.get('revisited_view_ids', ()))
        if not matches:
            return None
        for record in reversed(r.memory.observation_searches):
            if record['subgoal'] != r.memory.task.current_subgoal_index:
                continue
            if (tuple(record['target_types']) != tuple(target_types)
                    or record['objective'].strip().casefold() != objective.strip().casefold()):
                continue
            for check in reversed(record['checkpoints']):
                if (check.get('evidence_origin') == 'vlm_current_frame'
                        and check['observation_id'] in matches
                        and isinstance(check.get('visual'), dict)):
                    return copy.deepcopy(check['visual']), check['observation_id']
        return None

    def scan(self, objective, target_types, direction, sweep_degrees,
             checkpoint_degrees=30, look='none'):
        r = self.runtime
        # A large scan while a local path is in progress can undo its heading
        # alignment. Require a separate, explicit route cancellation first.
        if r.motion_controller.target is not None:
            return dict(status='INVALID_ARGUMENTS',
                reason='active_local_navigation_requires_explicit_cancel_before_scan',
                suggestion='continue_navigate_local_or_cancel_local_navigation_first')
        if not target_types or any(t not in TARGET_TYPES for t in target_types):
            return dict(status='INVALID_ARGUMENTS', reason='unsupported_static_visual_target')
        turn = r.camera.turn_angle
        total = round(sweep_degrees / turn)
        interval = round(checkpoint_degrees / turn)
        if (direction not in ('left', 'right') or look not in ('none', 'up', 'down')
                or not 1 <= total <= 8 or not 1 <= interval <= total
                or abs(total*turn-sweep_degrees) > .001
                or abs(interval*turn-checkpoint_degrees) > .001):
            return dict(status='INVALID_ARGUMENTS',reason='invalid_observation_sweep_calibration')
        if look != 'none':
            target_pitch = r.pitch_deg + (r.camera.tilt_angle if look == 'up' else -r.camera.tilt_angle)
            if abs(target_pitch) > 60:
                return dict(status='INVALID_ARGUMENTS',reason='camera_pitch_outside_calibrated_projection')

        start = r.motion_controller.pose()
        start_view=r.memory.current
        if last_block(r.memory,start):
            look_pitch = (start['pitch_deg'] + (
                r.camera.tilt_angle if look=='up' else -r.camera.tilt_angle)
                if look!='none' else start['pitch_deg'])
            planned_pose=dict(start,pitch_deg=look_pitch)
            signed_sweep=sweep_degrees if direction=='left' else -sweep_degrees
            planned=predicted_scan(r.memory,planned_pose,signed_sweep,'none',
                r.camera.turn_angle,r.camera.tilt_angle)
            initial_view=(covered_view(r.memory.observations,start,
                r.memory.task.current_subgoal_index,bearing_from_pose(start),
                look_pitch,r.camera.turn_angle) if look!='none' else True)
            if not planned['new_view_expected'] and initial_view:
                return dict(status='NO_NEW_INFORMATION',
                    outcome='REPEATED_SEARCH_NO_NEW_VIEW',
                    reason='all_scan_camera_headings_previously_observed_since_block',
                    covered_turn_deg=0,planned_sweep_deg=sweep_degrees,
                    checkpoints=[],best_candidate=None,
                    new_information_ids=[],revisited_view_ids=[],
                    usable_for_live_safety=False,
                    exploration=local_coverage(r.memory,start,r.camera.turn_angle,
                        round(360/r.camera.turn_angle)),
                    next_step='change_absolute_heading_or_strategy_not_repeat_scan')
        now=time.monotonic()
        # Exact duplicate observation task from the same pose and frame is
        # idempotent. No repeated TURN sweep for a previously covered static
        # sector. This expires quickly and never authorizes live human safety.
        for previous in reversed(r.memory.observation_searches):
            if (previous['outcome'] not in (
                    'SEARCH_REGION_COVERED_NO_MATCH','TARGET_CANDIDATE_FOUND')
                    or previous.get('subgoal')!=r.memory.task.current_subgoal_index
                    or previous.get('objective','').strip().casefold()!=objective.strip().casefold()
                    or previous.get('target_types')!=list(target_types)
                    or previous.get('direction')!=direction
                    or previous.get('planned_sweep_deg')!=sweep_degrees
                    or previous.get('checkpoint_degrees')!=checkpoint_degrees
                    or previous.get('look')!=look
                    or not comparable_scan_pose(previous.get('start_pose'),start)
                    or previous.get('start_rgb_hash')!=start_view.get('rgb_hash')
                    or previous.get('start_depth_hash')!=start_view.get('depth_hash')
                    or now-previous.get('started_monotonic',-1e9)>20):
                continue
            return dict(status='OK',outcome='REPEATED_SEARCH_NO_NEW_VIEW',
                reason='identical_static_scan_recently_completed',
                covered_turn_deg=0,planned_sweep_deg=sweep_degrees,
                previous_outcome=previous['outcome'],
                best_candidate=copy.deepcopy(previous.get('candidate')),
                previous_end_observation_id=previous.get('end_observation_id'),
                checkpoints=[],new_information_ids=[],revisited_view_ids=[],
                usable_for_live_safety=False,
                next_step='choose_a_new_search_sector_or_local_geometry_not_same_scan')
        record = dict(objective=objective, target_types=list(target_types),
            subgoal=r.memory.task.current_subgoal_index,
            start_observation_id=start_view['observation_id'],
            start_heading_deg=math.degrees(start['heading']),
            start_pose=dict(start),start_rgb_hash=start_view.get('rgb_hash'),
            start_depth_hash=start_view.get('depth_hash'),
            started_monotonic=now,look=look,
            checkpoint_degrees=checkpoint_degrees,
            direction=direction, planned_sweep_deg=sweep_degrees,
            checkpoints=[], actions=[], outcome='IN_PROGRESS')
        r.memory.observation_searches.append(record)

        def finish(outcome, status='OK', reason=None, candidate=None):
            record['outcome'] = outcome
            record['end_observation_id'] = r.memory.current['observation_id']
            record['covered_turn_deg'] = sum(
                r.camera.turn_angle for action in record['actions']
                if action in ('TURN_LEFT', 'TURN_RIGHT'))
            record['candidate'] = copy.deepcopy(candidate)
            value = dict(status=status, outcome=outcome, reason=reason,
                objective=objective, target_types=list(target_types),
                covered_turn_deg=record['covered_turn_deg'],
                planned_sweep_deg=sweep_degrees,
                checkpoints=[{key:point[key] for key in (
                    'observation_id','heading_deg','pitch_deg',
                    'evidence_origin','reused_from_observation_id','visual',
                    'geometry') if key in point} for point in record['checkpoints']],
                best_candidate=candidate,
                new_information_ids=[p['observation_id'] for p in record['checkpoints']
                    if p.get('new_static_information')],
                revisited_view_ids=[p['observation_id'] for p in record['checkpoints']
                    if not p.get('new_static_information')],
                next_step='navigation_agent_review_evidence_not_motion_authorization',
                usable_for_live_safety=False)
            r.event('observation_task_finished', source='observation_controller',
                outcome=outcome, covered_turn_deg=record['covered_turn_deg'],
                checkpoint_count=len(record['checkpoints']))
            return value

        def step(action):
            result = r._step(action)
            if result['status'] != 'EXECUTED':
                return result
            record['actions'].append(action)
            return None

        def checkpoint():
            source = r.source()
            observation = r.memory.current
            reused = self._existing_result(target_types, objective)
            if reused:
                visual, previous_id = reused
                origin = 'reused_static_visual_hypothesis'
            else:
                visual = r.visual_perception.search_view(
                    r.observation, source, objective, target_types)
                previous_id = None
                origin = 'vlm_current_frame'
            # Current RGB-D is a *measurement*; VLM box/identity is a hypothesis.
            # Diagnostic geometry does not grant forward authorization.
            from .safety import forward_probe
            check = forward_probe(r.geometry())
            point = dict(observation_id=source['observation_id'],
                heading_deg=round(math.degrees(source['pose']['heading']),2),
                pitch_deg=r.pitch_deg, new_static_information=bool(
                    observation.get('new_visual_information',True)),
                evidence_origin=origin, visual=visual,
                geometry=dict(forward_status=check['state'],
                    reason=check['reason'],
                    clearance_m=round(check['depth']['clearance'],3),
                    terrain_status=check['support']['status'],
                    motion_authorized=False))
            if previous_id:
                point['reused_from_observation_id'] = previous_id
            record['checkpoints'].append(point)
            if meaningful_candidate(visual, target_types):
                if origin == 'vlm_current_frame':
                    r.memory.note_semantics(source, [
                        visual['target_type']+': '+visual['evidence']])
                return dict(target_type=visual['target_type'],
                    confidence=visual['confidence'], box=visual['box'],
                    evidence=visual['evidence'],
                    uncertainty=visual.get('uncertainty',''),
                    observation_id=source['observation_id'],
                    semantic_source_observation_id=previous_id or source['observation_id'],
                    heading_deg=point['heading_deg'], pitch_deg=point['pitch_deg'],
                    evidence_origin=origin, geometry=point['geometry'],
                    usable_for_live_safety=False)
            return None

        if look != 'none':
            failed = step('LOOK_UP' if look == 'up' else 'LOOK_DOWN')
            if failed:
                return finish('INTERRUPTED', status='PARTIAL', reason=failed.get('reason'))

        turn_action = 'TURN_LEFT' if direction == 'left' else 'TURN_RIGHT'
        refined = False
        for index in range(1,total+1):
            failed = step(turn_action)
            if failed:
                return finish('INTERRUPTED', status='PARTIAL', reason=failed.get('reason'))
            if index % interval != 0 and index != total:
                continue
            try:
                found = checkpoint()
            except (APIError, BudgetExceeded) as exc:
                return finish('VISION_UNAVAILABLE',status='PARTIAL',
                    reason=type(exc).__name__+'_at_visual_checkpoint')
            if found:
                return finish('TARGET_CANDIDATE_FOUND',candidate=found)
            # At most one VLM-guided pitch refinement in the entire subtask.
            last = record['checkpoints'][-1]['visual']
            suggestion = last.get('suggested_next_view')
            if not refined and suggestion in ('look_up','look_down'):
                next_pitch = r.pitch_deg + (
                    r.camera.tilt_angle if suggestion == 'look_up' else -r.camera.tilt_angle)
                if abs(next_pitch) <= 60:
                    refined = True
                    failed = step('LOOK_UP' if suggestion == 'look_up' else 'LOOK_DOWN')
                    if failed:
                        return finish('INTERRUPTED',status='PARTIAL',reason=failed.get('reason'))
                    try:
                        found = checkpoint()
                    except (APIError, BudgetExceeded) as exc:
                        return finish('VISION_UNAVAILABLE',status='PARTIAL',
                            reason=type(exc).__name__+'_at_visual_checkpoint')
                    if found:
                        return finish('TARGET_CANDIDATE_FOUND',candidate=found)
            if r.done:
                return finish('INTERRUPTED',status='PARTIAL',reason='environment_done')
        return finish('SEARCH_REGION_COVERED_NO_MATCH')
