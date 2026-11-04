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


# Open-vocabulary target names come from the model's explicit observation goal.


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
    if (a.get('motion_epoch') != b.get('motion_epoch')
            or a.get('viewpoint_generation',0) != b.get('viewpoint_generation',0)):
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
             checkpoint_degrees=30, look='none', force_recheck=False):
        r = self.runtime
        # The agent may pause any local route for observation. The relative
        # goal remains pending and is revalidated after the scan; it is never
        # treated as pre-authorized movement from the previous heading.
        if (not target_types or any(not isinstance(t,str) or not t.strip() or
                len(t)>80 for t in target_types)):
            return dict(status='INVALID_ARGUMENTS', reason='invalid_visual_target_description')
        turn = r.camera.turn_angle
        total = round(sweep_degrees / turn)
        interval = round(checkpoint_degrees / turn)
        if (direction not in ('left', 'right') or look not in ('none', 'up', 'down')
                or not 1 <= total <= round(360/turn) or not 1 <= interval <= total
                or abs(total*turn-sweep_degrees) > .001
                or abs(interval*turn-checkpoint_degrees) > .001):
            return dict(status='INVALID_ARGUMENTS',reason='invalid_observation_sweep_calibration')
        if look != 'none':
            target_pitch = r.pitch_deg + (r.camera.tilt_angle if look == 'up' else -r.camera.tilt_angle)
            if abs(target_pitch) > 60:
                return dict(status='INVALID_ARGUMENTS',reason='camera_pitch_outside_calibrated_projection')

        start = r.motion_controller.pose()
        start_view=r.memory.current
        if last_block(r.memory,start) and not force_recheck:
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
        for previous in ([] if force_recheck else reversed(r.memory.observation_searches)):
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
            return dict(status='NO_NEW_INFORMATION',outcome='REPEATED_SEARCH_NO_NEW_VIEW',
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
                paused_waypoint_preserved=bool(r.motion_controller.target is not None),
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
            reused = None if force_recheck else self._existing_result(target_types, objective)
            if reused:
                visual, previous_id = reused
                origin = 'reused_static_visual_hypothesis'
            else:
                # Explicit fresh observation bypasses both the saved-view
                # semantic lookup AND the visual model's image/question cache.
                if force_recheck:
                    visual = r.visual_perception.search_view(
                        r.observation, source, objective, target_types,
                        force_recheck=True)
                else:
                    visual = r.visual_perception.search_view(
                        r.observation, source, objective, target_types)
                previous_id = None
                origin = 'vlm_current_frame'
            # A goal-directed scan is a visual tool, not an implicit
            # geometry/obstacle inspection. VLM evidence remains semantic.
            point = dict(observation_id=source['observation_id'],
                heading_deg=round(math.degrees(source['pose']['heading']),2),
                pitch_deg=r.pitch_deg, new_static_information=bool(
                    observation.get('new_visual_information',True)),
                evidence_origin=origin, visual=visual,
                geometry=dict(forward_status='NOT_CHECKED',
                    reason='geometry_requires_explicit_model_tool',
                    clearance_m=None,terrain_status='NOT_CHECKED',
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


    def scan_vlm(self, objective, target_types, sweep_degrees=360,
                 look='none', force_recheck=False):
        """VLM owns the complete camera search: observe -> choose view -> observe.

        NavigationAgent only specifies the objective. No main-model inference
        occurs between VLM-selected camera turns; the controller only enforces
        actuator legality, observation coverage and Habitat's action budget.
        """
        r=self.runtime
        if (not objective or not isinstance(objective,str)
                or not target_types or any(not isinstance(t,str) or not t.strip()
                    or len(t)>80 for t in target_types)):
            return dict(status='INVALID_ARGUMENTS',reason='invalid_vlm_observation_objective')
        turn=r.camera.turn_angle
        if (look not in ('none','up','down')
                or sweep_degrees<30 or sweep_degrees>360
                or abs(sweep_degrees/turn-round(sweep_degrees/turn))>1e-6):
            return dict(status='INVALID_ARGUMENTS',reason='invalid_vlm_observation_sweep')
        if look!='none' and abs(r.pitch_deg+(
                r.camera.tilt_angle if look=='up' else -r.camera.tilt_angle))>60:
            return dict(status='INVALID_ARGUMENTS',reason='camera_pitch_outside_calibrated_projection')
        start=r.motion_controller.pose()
        initial=r.memory.current
        now=time.monotonic()
        for prev in ([] if force_recheck else reversed(r.memory.observation_searches)):
            if (prev.get('mode')!='vlm_led'
                    or prev.get('outcome') not in (
                        'TARGET_CANDIDATE_FOUND','VLM_SEARCH_FINISHED',
                        'SEARCH_REGION_COVERED_NO_MATCH','CUE_ONLY')
                    or prev.get('objective','').strip().casefold()!=objective.strip().casefold()
                    or prev.get('target_types')!=list(target_types)
                    or prev.get('subgoal')!=r.memory.task.current_subgoal_index
                    or prev.get('start_rgb_hash')!=initial.get('rgb_hash')
                    or not comparable_scan_pose(prev.get('start_pose'),start)
                    or now-prev.get('started_monotonic',-1e9)>30):
                continue
            return dict(status='NO_NEW_INFORMATION',
                outcome='REPEATED_SEARCH_NO_NEW_VIEW',
                reason='same_vlm_observation_task_recently_completed',
                covered_turn_deg=0,planned_sweep_deg=sweep_degrees,
                checkpoints=[],best_candidate=copy.deepcopy(prev.get('candidate')),
                best_cue=copy.deepcopy(prev.get('best_cue')),
                next_step='navigation_agent_use_existing_evidence_or_choose_new_goal')
        record=dict(mode='vlm_led',objective=objective,target_types=list(target_types),
            subgoal=r.memory.task.current_subgoal_index,
            start_observation_id=initial['observation_id'],start_pose=dict(start),
            start_rgb_hash=initial.get('rgb_hash'),start_depth_hash=initial.get('depth_hash'),
            started_monotonic=now,direction='vlm_selected',look=look,
            planned_sweep_deg=sweep_degrees,checkpoint_degrees='vlm_selected',
            checkpoints=[],actions=[],outcome='IN_PROGRESS')
        r.memory.observation_searches.append(record)
        turned=0
        best_cue=None
        best_candidate=None
        seen=set()
        # A VLM decision always refers to ONE real current-frame observation.
        # Metadata contains no synthesized, unseen or predicted RGB frames.
        def view_key():
            pose=r.motion_controller.pose()
            return (round(math.degrees(pose['heading'])/turn)%round(360/turn),
                    round(r.pitch_deg/r.camera.tilt_angle))
        def finish(outcome,status='OK',reason=None):
            # Saved absolute view bearings are rebound to the CURRENT heading.
            # Both the return receipt and persisted observation task agree.
            live_heading=math.degrees(r.motion_controller.pose()['heading'])
            for hypothesis in (best_candidate,best_cue):
                if hypothesis:
                    hypothesis['relative_turn_deg_from_current']=round(
                        (hypothesis['heading_deg']-live_heading+180)%360-180,2)
            record.update(outcome=outcome,covered_turn_deg=turned,
                end_observation_id=r.memory.current['observation_id'],
                candidate=copy.deepcopy(best_candidate),
                best_cue=copy.deepcopy(best_cue))
            return dict(status=status,outcome=outcome,reason=reason,
                objective=objective,target_types=list(target_types),
                observation_owner='visual_perception_vlm',
                planned_sweep_deg=sweep_degrees,covered_turn_deg=turned,
                checkpoints=[dict(observation_id=p['observation_id'],
                    heading_deg=p['heading_deg'],pitch_deg=p['pitch_deg'],
                    evidence_origin='vlm_current_frame',
                    visual=p['visual']) for p in record['checkpoints']],
                best_candidate=copy.deepcopy(best_candidate),
                best_cue=copy.deepcopy(best_cue),
                view_history=[dict(observation_id=p['observation_id'],
                    heading_deg=p['heading_deg'],pitch_deg=p['pitch_deg'],
                    target_kind=p['visual'].get('target_kind','UNKNOWN'))
                    for p in record['checkpoints']],
                new_information_ids=[p['observation_id'] for p in record['checkpoints']
                    if p.get('new_static_information')],
                paused_waypoint_preserved=bool(r.motion_controller.target is not None),
                usable_for_live_safety=False,
                next_step='navigation_agent_choose_motion_or_specific_new_visual_goal')
        def do_action(action):
            result=r._step(action)
            if result['status']!='EXECUTED':
                return result
            record['actions'].append(action)
            return None
        if look!='none':
            failed=do_action('LOOK_DOWN' if look=='down' else 'LOOK_UP')
            if failed:return finish('INTERRUPTED','PARTIAL',failed.get('reason'))
        max_checks=min(14,max(3,math.ceil(sweep_degrees/30)+3))
        for _ in range(max_checks):
            if r.done:
                return finish('INTERRUPTED','PARTIAL','environment_done')
            key=view_key()
            if key in seen:
                return finish('REPEATED_VIEW_REQUEST','NO_NEW_INFORMATION',
                    'vlm_requested_already_inspected_heading_pitch')
            seen.add(key)
            source=r.source()
            history=[dict(observation_id=p['observation_id'],
                heading_deg=p['heading_deg'],pitch_deg=p['pitch_deg'],
                target_kind=p['visual'].get('target_kind','UNKNOWN'),
                evidence=(p['visual'].get('evidence') or '')[:110])
                for p in record['checkpoints'][-5:]]
            prior=[dict(observation_id=t.get('selected_observation_id'),
                target_status=t.get('status'),
                evidence=(t.get('visible_evidence') or '')[:100])
                for t in getattr(r.memory,'visual_targets',())[-2:]]
            try:
                visual=r.visual_perception.guide_scan_view(
                    r.observation,source,objective,target_types,history,
                    max(0,sweep_degrees-turned),prior_targets=prior)
            except (APIError,BudgetExceeded) as error:
                return finish('VISION_UNAVAILABLE','PARTIAL',type(error).__name__)
            position=r.motion_controller.pose()
            point=dict(observation_id=source['observation_id'],
                heading_deg=round(math.degrees(position['heading']),2),
                pitch_deg=r.pitch_deg,
                new_static_information=bool(r.memory.current.get(
                    'new_visual_information',True)),visual=visual)
            record['checkpoints'].append(point)
            if meaningful_candidate(visual,target_types):
                candidate=dict(target_type=visual['target_type'],
                    confidence=visual['confidence'],box=visual['box'],
                    evidence=visual['evidence'],uncertainty=visual.get('uncertainty',''),
                    target_kind=visual['target_kind'],
                    observation_id=source['observation_id'],
                    semantic_source_observation_id=source['observation_id'],
                    heading_deg=point['heading_deg'],pitch_deg=point['pitch_deg'],
                    evidence_origin='vlm_current_frame',
                    geometry=dict(forward_status='NOT_CHECKED',
                        motion_authorized=False),
                    usable_for_live_safety=False)
                if visual['target_kind'] in ('ENTRY','APPROACH','TARGET'):
                    best_candidate=candidate
                    r.memory.note_semantics(source,[
                        visual['target_type']+': '+visual['evidence']])
                    return finish('TARGET_CANDIDATE_FOUND')
                if best_cue is None or visual['confidence']=='high':
                    best_cue=candidate
            action=visual['next_action']
            if action=='finish':
                return finish('CUE_ONLY' if best_cue else 'VLM_SEARCH_FINISHED',
                    reason='vlm_concluded_visual_search')
            if action in ('look_up','look_down'):
                new_pitch=r.pitch_deg+(r.camera.tilt_angle if action=='look_up'
                                      else -r.camera.tilt_angle)
                if abs(new_pitch)>60:
                    return finish('CUE_ONLY' if best_cue else 'VLM_SEARCH_FINISHED',
                        reason='vlm_requested_out_of_range_pitch')
                failed=do_action('LOOK_UP' if action=='look_up' else 'LOOK_DOWN')
                if failed:return finish('INTERRUPTED','PARTIAL',failed.get('reason'))
                continue
            degrees=visual['turn_deg']
            if degrees<=0 or degrees%turn or turned+degrees>sweep_degrees:
                return finish('CUE_ONLY' if best_cue else 'SEARCH_REGION_COVERED_NO_MATCH',
                    reason='vlm_turn_exceeds_remaining_sweep')
            direction='TURN_LEFT' if action=='turn_left' else 'TURN_RIGHT'
            target_heading=(key[0]+(
                degrees/turn if direction=='TURN_LEFT' else -degrees/turn))%round(360/turn)
            if (int(round(target_heading)),key[1]) in seen:
                return finish('REPEATED_VIEW_REQUEST','NO_NEW_INFORMATION',
                    'vlm_requested_already_inspected_heading_pitch')
            for _turn in range(round(degrees/turn)):
                failed=do_action(direction)
                if failed:return finish('INTERRUPTED','PARTIAL',failed.get('reason'))
                turned+=turn
            if turned>=sweep_degrees:
                # The VLM gets a final look at the last real direction if a
                # checkpoint remains; otherwise do not issue an extra turn.
                continue
        return finish('CUE_ONLY' if best_cue else 'SEARCH_REGION_COVERED_NO_MATCH',
            reason='bounded_vlm_checkpoints_exhausted')

