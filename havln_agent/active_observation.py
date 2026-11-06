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
    """Legacy fixed-direction scan predicate, NOT a gate for VLM-led scans."""
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


def recent_authentic_motion_frames(runtime, current_id, limit=3):
    """Select a few real motion-relevant RGB-D views from earlier tasks.

    Old frames are historical geometry and cannot authorize current MOVE.
    All missing/evicted frames are skipped; never invent images.
    """
    frame_reader=getattr(runtime,'_vision_frame',None)
    route_reader=getattr(runtime.memory,'route_progress_summary',None)
    if not callable(frame_reader) or not callable(route_reader):
        return []
    progress=route_reader()
    subgoal=runtime.memory.task.current_subgoal_index
    segments=[v for v in progress.get('successful_segments',[])
        if v.get('subgoal')==subgoal]
    blocks=[v for v in progress.get('blocked_at_prior_stances',[])
        if v.get('subgoal') in (None,'sg_%d'%subgoal)]
    ids=[]
    if segments:
        ids.extend([segments[-1].get('start_observation_id'),
                    segments[-1].get('end_observation_id')])
    if blocks:
        ids.append(blocks[-1].get('source_observation_id'))
    frames=[]
    seen=set()
    for observation_id in ids:
        if not observation_id or observation_id==current_id or observation_id in seen:
            continue
        seen.add(observation_id)
        try:
            observation,source=frame_reader(observation_id)
        except (ValueError,KeyError,OSError,RuntimeError):
            continue
        if (source.get('observation_id')!=observation_id or
                getattr(observation,'rgb',None) is None or
                getattr(observation,'depth',None) is None):
            continue
        pose=source.get('pose') or {}
        frames.append((observation,dict(
            observation_id=observation_id,
            heading_deg=round(math.degrees(pose.get('heading',0)),2),
            pitch_deg=round(pose.get('pitch_deg',0),2),
            motion_epoch=pose.get('motion_epoch'),
            historical_only=True)))
        if len(frames)>=limit:
            break
    return frames


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
                 look='none', force_recheck=False, mode='APPROACH'):
        """VLM owns the complete camera search: observe -> choose view -> observe.

        NavigationAgent only specifies the objective. No main-model inference
        occurs between VLM-selected camera turns; the controller only enforces
        actuator legality, observation coverage and Habitat's action budget.
        """
        r=self.runtime
        if mode not in ('SEARCH','APPROACH','OBSTACLE_REVIEW','VERIFY'):
            return dict(status='INVALID_ARGUMENTS',reason='unknown_observation_mode')
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
                        'SEARCH_REGION_COVERED_NO_MATCH','CUE_ONLY',
                        'APPROACH_CANDIDATE_FOUND','NO_RELIABLE_CANDIDATE',
                        'SUBGOAL_NOT_YET','SUBGOAL_UNCERTAIN',
                        'VISION_UNAVAILABLE')
                    or prev.get('intent','APPROACH')!=mode
                    or prev.get('target_types')!=list(target_types)
                    # VERIFY always assesses the current subgoal's completion.
                    # Rewording that question must not trigger another expensive
                    # panorama from identical RGB-D at the same physical stance.
                    or (mode!='VERIFY' and
                        ' '.join(prev.get('objective','').casefold().split())
                        !=' '.join(objective.casefold().split()))
                    or (mode=='VERIFY' and prev.get('outcome')=='VISION_UNAVAILABLE'
                        and now-prev.get('started_monotonic',-1e9)>20)
                    or prev.get('subgoal')!=r.memory.task.current_subgoal_index
                    or prev.get('start_rgb_hash')!=initial.get('rgb_hash')
                    or prev.get('start_depth_hash')!=initial.get('depth_hash')
                    or not comparable_scan_pose(prev.get('start_pose'),start)):
                continue
            current_heading=math.degrees(start['heading'])
            def rebound(hypothesis):
                value=copy.deepcopy(hypothesis)
                if value is not None:
                    value['relative_turn_deg_from_current']=round(
                        (value['heading_deg']-current_heading+180)%360-180,2)
                return value
            return dict(status='NO_NEW_INFORMATION',
                outcome='REPEATED_SEARCH_NO_NEW_VIEW',
                reason='unchanged_RGBD_stance_and_observation_intent_reuse',
                verification=copy.deepcopy(prev.get('verification')),
                navigation_feedback=copy.deepcopy(prev.get('navigation_feedback')),
                observation_mode=mode,
                covered_turn_deg=0,planned_sweep_deg=sweep_degrees,
                checkpoints=[],best_candidate=rebound(prev.get('candidate')),
                best_cue=rebound(prev.get('best_cue')),
                best_approach=rebound(prev.get('best_approach')),
                approach_candidates=[rebound(v) for v in prev.get('approach_candidates',[])],
                blocked_approaches=[rebound(v) for v in prev.get('blocked_approaches',[])],
                next_step='navigation_agent_use_existing_evidence_or_choose_new_goal')
        record=dict(mode='vlm_led',intent=mode,objective=objective,target_types=list(target_types),
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
        best_approach=None
        approach_candidates=[]
        blocked_approaches=[]
        session=[]  # Full RGB-D and assistant replies for one observation task.
        historical_frames=recent_authentic_motion_frames(
            r,initial['observation_id'])
        record['historical_context_observation_ids']=[
            info['observation_id'] for _,info in historical_frames]
        seen=set()
        # Physical yaw visits help detect costly 180-degree undo sweeps even
        # when a LOOK changed pitch. Only VLM checkpoints count as SEMANTIC
        # observed views: a passing camera TURN is not visual evidence.
        yaw_traversed=set()
        last_turn_direction=None
        candidate_found_checkpoint=None
        max_candidate_refinement_views=3
        rejected_views=[]
        pending_view_correction=None
        consecutive_replans=0
        max_replans=2
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
            for hypothesis in (best_candidate,best_cue,best_approach):
                if hypothesis:
                    hypothesis['relative_turn_deg_from_current']=round(
                        (hypothesis['heading_deg']-live_heading+180)%360-180,2)
            for hypothesis in approach_candidates+blocked_approaches:
                hypothesis['relative_turn_deg_from_current']=round(
                    (hypothesis['heading_deg']-live_heading+180)%360-180,2)
            # Relay all inspected RGB-frame VLM reports without applying any
            # target-name match, confidence threshold or semantic relevance
            # predicate. Real frame references let NavigationAgent judge.
            reports=[dict(
                observation_id=p['observation_id'],
                heading_deg=p['heading_deg'],pitch_deg=p['pitch_deg'],
                target_visible=p['visual'].get('target_visible'),
                target_type=p['visual'].get('target_type'),
                target_kind=p['visual'].get('target_kind'),
                confidence=p['visual'].get('confidence'),
                box=copy.deepcopy(p['visual'].get('box')),
                evidence=p['visual'].get('evidence',''),
                uncertainty=p['visual'].get('uncertainty',''),
                next_action=p['visual'].get('next_action'),
                turn_deg=p['visual'].get('turn_deg'),
                reason=p['visual'].get('reason',''),
                evidence_origin='vlm_current_rgbd_frame',
                approach_probe=copy.deepcopy(p.get('approach_probe')),
                candidate_approach=bool(p['visual'].get('candidate_approach',False)),
                usable_for_live_safety=False)
                for p in record['checkpoints']]
            last_visual=(record['checkpoints'][-1]['visual']
                         if record['checkpoints'] else {})
            final_status={'SUBGOAL_CONFIRMED':'CONFIRMED',
                'SUBGOAL_NOT_YET':'NOT_YET','SUBGOAL_UNCERTAIN':'UNCERTAIN'
                }.get(outcome,'UNCERTAIN')
            verification=(dict(
                subgoal_index=record['subgoal'],
                status=final_status,
                evidence=(last_visual.get('verification_evidence','')
                    if outcome in ('SUBGOAL_CONFIRMED','SUBGOAL_NOT_YET') else ''),
                final_verdict=outcome in ('SUBGOAL_CONFIRMED',
                    'SUBGOAL_NOT_YET','SUBGOAL_UNCERTAIN'),
                uncertainty=last_visual.get('uncertainty',''),
                source_observation_id=(record['checkpoints'][-1]['observation_id']
                    if record['checkpoints'] else None),
                observation_ids=list(dict.fromkeys(
                    p['observation_id'] for p in record['checkpoints'])),
                cited_observation_ids=list(last_visual.get(
                    'evidence_observation_ids',[])),
                missing_evidence=last_visual.get('missing_evidence',''),
                source='vlm_real_rgbd_history') if mode=='VERIFY' else None)
            # A concise narrative is useful to NavigationAgent, but it must
            # never replace measurement provenance, clearance, or the actual
            # final VERIFY result. Conflicting suggested transitions are
            # discarded while keeping the original observation evidence.
            advice_status=last_visual.get('recommended_next','UNKNOWN')
            if (mode=='VERIFY' and final_status!='CONFIRMED'
                    and advice_status=='PROCEED_NEXT_TASK'):
                advice_status='UNKNOWN'
            summary=(last_visual.get('observation_summary') or
                last_visual.get('verification_evidence') or
                last_visual.get('evidence') or '')
            narrative=dict(
                observation_summary=summary[:750],
                navigation_suggestion=(last_visual.get('navigation_suggestion')
                    or '')[:600],
                recommended_next=advice_status,
                observation_mode=mode,
                outcome=outcome,
                evidence_observation_ids=list(dict.fromkeys(
                    p['observation_id'] for p in record['checkpoints'])),
                measured_motion_is_distinct_from_visual_hypothesis=True,
                advice_only_not_motion_authorization=True,
                evidence_scope='real_RGBD_and_measurement_plus_VLM_hypothesis')
            record.update(verification=copy.deepcopy(verification),
                navigation_feedback=copy.deepcopy(narrative),
                session_rounds=len(session)//2,
                outcome=outcome,covered_turn_deg=turned,
                visual_reports=copy.deepcopy(reports),
                end_observation_id=r.memory.current['observation_id'],
                candidate_refinement_views=(max(0,len(record['checkpoints'])-1
                    - candidate_found_checkpoint) if candidate_found_checkpoint is not None else 0),
                candidate=copy.deepcopy(best_candidate),
                best_cue=copy.deepcopy(best_cue),
                best_approach=copy.deepcopy(best_approach),
                approach_candidates=copy.deepcopy(approach_candidates),
                blocked_approaches=copy.deepcopy(blocked_approaches),
                rejected_view_requests=copy.deepcopy(rejected_views))
            store=getattr(r.memory,'note_active_observation_target',None)
            # Persist the selected APPROACH with its progress classification,
            # not an older visually salient but currently blocked landmark.
            hypothesis=(best_approach if best_approach and
                best_approach.get('target_kind') in ('ENTRY','APPROACH','TARGET','CUE')
                else best_candidate or best_cue)
            if hypothesis and callable(store):
                store(hypothesis,objective,target_types,
                    goal=getattr(r,'active_subgoal',None))
            return dict(status=status,outcome=outcome,reason=reason,
                objective=objective,target_types=list(target_types),
                observation_mode=mode,verification=copy.deepcopy(verification),
                navigation_feedback=copy.deepcopy(narrative),
                session_rounds=len(session)//2,
                historical_context_observation_ids=list(
                    record['historical_context_observation_ids']),
                observation_owner='visual_perception_vlm',
                view_replan_attempts=len(rejected_views),
                candidate_refinement_views=record['candidate_refinement_views'],
                planned_sweep_deg=sweep_degrees,covered_turn_deg=turned,
                checkpoints=[dict(observation_id=p['observation_id'],
                    heading_deg=p['heading_deg'],pitch_deg=p['pitch_deg'],
                    evidence_origin='vlm_current_frame',
                    visual=p['visual']) for p in record['checkpoints']],
                visual_reports=copy.deepcopy(reports),
                best_candidate=copy.deepcopy(best_candidate),
                best_cue=copy.deepcopy(best_cue),
                best_approach=copy.deepcopy(best_approach),
                approach_candidates=copy.deepcopy(approach_candidates),
                blocked_approaches=copy.deepcopy(blocked_approaches),
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
        # Verification is a bounded assessment, not another full search.
        # Search and obstacle exploration retain their existing view budget.
        max_checks=min(5 if mode=='VERIFY' else 16,
                       max(3,math.ceil(sweep_degrees/30)+5))
        for _ in range(max_checks):
            if r.done:
                return finish('INTERRUPTED','PARTIAL','environment_done')
            key=view_key()
            retrying=pending_view_correction is not None
            if key in seen and not retrying:
                return finish('REPEATED_VIEW_REQUEST','NO_NEW_INFORMATION',
                    'camera_returned_to_an_already_inspected_pose')
            if not retrying:
                seen.add(key)
            yaw_traversed.add(key[0])
            source=r.source()
            history=[dict(observation_id=p['observation_id'],
                heading_deg=p['heading_deg'],pitch_deg=p['pitch_deg'],
                target_kind=p['visual'].get('target_kind','UNKNOWN'),
                evidence=(p['visual'].get('evidence') or '')[:110],
                approach_probe_status=(p.get('approach_probe') or {}).get('status'))
                for p in record['checkpoints'][-12:]]
            prior=[dict(observation_id=t.get('selected_observation_id'),
                target_status=t.get('status'),
                evidence=(t.get('visible_evidence') or '')[:100])
                for t in getattr(r.memory,'visual_targets',())[-2:]]
            correction=pending_view_correction
            pending_view_correction=None
            try:
                options=dict(prior_targets=prior)
                if correction is not None:
                    options['view_correction']=correction
                options['observed_yaw_pitch']=[
                    dict(heading_deg=p['heading_deg'],pitch_deg=p['pitch_deg'])
                    for p in record['checkpoints'][-12:]]
                geometry_facts={}
                try:
                    geom=r.geometry()
                    center_probe=getattr(geom,'central_forward_obstruction',None)
                    wide_probe=getattr(geom,'movement_probe',None)
                    center=center_probe() if callable(center_probe) else {}
                    wide=wide_probe(0.) if callable(wide_probe) else {}
                    swept=wide.get('depth',wide)
                    geometry_facts=dict(source_observation_id=source['observation_id'],
                        heading_deg=round(math.degrees(
                            r.motion_controller.pose()['heading']),2),
                        center_obstruction=center,
                        swept_body_state=wide.get('state'),
                        swept_body_reason=wide.get('reason'),
                        swept_body_depth={k:swept[k] for k in (
                            'path_state','clearance','unknown_sweep_pixels',
                            'nearest_collision_point','rejection_reasons',
                            'human_occupancy') if k in swept},
                        scope='current_public_depth_only_not_route_certification')
                except (AttributeError,RuntimeError,ValueError,TypeError):
                    geometry_facts=dict(status='UNAVAILABLE')
                progress=getattr(r.memory,'route_progress_summary',None)
                route_history=(progress() if callable(progress) else {})
                visual=r.visual_perception.guide_scan_view(
                    r.observation,source,objective,target_types,history,
                    max(0,sweep_degrees-turned),session=session,mode=mode,
                    route_history=route_history,
                    geometry_evidence=geometry_facts,
                    historical_frames=historical_frames,**options)
            except (APIError,BudgetExceeded) as error:
                return finish('VISION_UNAVAILABLE','PARTIAL',type(error).__name__)
            position=r.motion_controller.pose()
            point=dict(observation_id=source['observation_id'],
                heading_deg=round(math.degrees(position['heading']),2),
                pitch_deg=r.pitch_deg,
                new_static_information=bool(r.memory.current.get(
                    'new_visual_information',True)),visual=visual)
            # A rejected camera micro-action can replace its same-frame
            # proposal. But a BLOCKED approach has genuine Depth evidence,
            # so preserve its own checkpoint and attach the next VLM
            # interpretation as a separate same-frame report.
            replace_rejected_view=(retrying and
                (correction or {}).get('reason')
                    != 'candidate_forward_measured_center_obstacle')
            if replace_rejected_view and record['checkpoints'] and (
                    record['checkpoints'][-1]['observation_id']==source['observation_id']):
                record['checkpoints'][-1]['visual']=visual
            else:
                record['checkpoints'].append(point)
            # VLM target_type is an OPEN VOCABULARY label, not an enum of
            # NavigationAgent's target_types. Relay even low confidence and
            # missing boxes as hypotheses; only the model decides relevance.
            # The structural API schema still validates values/coordinates.
            if visual.get('target_visible') is True:
                candidate=dict(target_type=visual['target_type'],
                    confidence=visual['confidence'],box=visual['box'],
                    evidence=visual['evidence'],uncertainty=visual.get('uncertainty',''),
                    target_kind=visual['target_kind'],
                    expected_progress=visual.get('expected_progress','UNKNOWN'),
                    observation_id=source['observation_id'],
                    semantic_source_observation_id=source['observation_id'],
                    heading_deg=point['heading_deg'],pitch_deg=point['pitch_deg'],
                    evidence_origin='vlm_current_frame',
                    geometry=dict(forward_status='NOT_CHECKED',
                        motion_authorized=False),
                    usable_for_live_safety=False)
                if visual['target_kind'] in ('ENTRY','APPROACH','TARGET'):
                    priority={'ENTRY':3,'APPROACH':2,'TARGET':1}
                    current_rank=(priority.get((best_candidate or {}).get('target_kind'),0),
                                  {'low':0,'medium':1,'high':2}.get(
                                      (best_candidate or {}).get('confidence'),0))
                    new_rank=(priority[visual['target_kind']],
                              {'low':0,'medium':1,'high':2}[visual['confidence']])
                    if best_candidate is None or new_rank>=current_rank:
                        best_candidate=candidate
                    if visual.get('evidence','').strip():
                        r.memory.note_semantics(source,[
                            visual['target_type']+': '+visual['evidence']])
                    if candidate_found_checkpoint is None:
                        candidate_found_checkpoint=len(record['checkpoints'])-1
                elif best_cue is None or visual['confidence']=='high':
                    # UNKNOWN is still available as a visual hypothesis,
                    # never silently erased because it lacks an exact label.
                    best_cue=candidate
            action=visual['next_action']
            # A semantic verdict is FINAL only when the VLM itself chooses
            # finish; a tentative CONFIRMED with a requested look/turn must
            # obtain that fresh RGB-D checkpoint first.
            if mode=='VERIFY' and action=='finish':
                verdict=visual.get('verification_status','UNCERTAIN')
                return finish({'CONFIRMED':'SUBGOAL_CONFIRMED',
                    'NOT_YET':'SUBGOAL_NOT_YET',
                    'UNCERTAIN':'SUBGOAL_UNCERTAIN'}[verdict],
                    reason='vlm_semantic_subgoal_observation')
            # An observation agent may propose the *current* facing as a
            # short movement hypothesis. Check only conservative live Depth
            # center occupancy, without full terrain review or moving.
            # Reliable blockers are fed to the SAME VLM observation loop.
            if mode!='VERIFY' and visual.get('candidate_approach',False):
                heading=point['heading_deg']
                # The SAME measured Depth snapshot was already supplied to
                # this VLM checkpoint; do not recompute a contradictory probe.
                center=geometry_facts.get('center_obstruction') or {}
                blocked=center.get('confirmed') is True
                probe=dict(status=('BLOCKED' if blocked else 'UNCONFIRMED'),
                    reason=(center.get('reason') or 'candidate_live_depth_not_confirmed'),
                    heading_deg=heading,
                    observation_id=source['observation_id'],
                    depth_hash=source.get('depth_hash'),
                    nearest_along_m=center.get('nearest_along_m'),
                    central_support_pixels=center.get('central_support_pixels'),
                    source_pixel_xy=center.get('source_pixel_xy'),
                    no_confirmed_obstacle_is_not_safe_path=True,
                    full_terrain_not_evaluated=True)
                point['approach_probe']=copy.deepcopy(probe)
                proposal=dict(
                    observation_id=source['observation_id'],
                    heading_deg=heading,pitch_deg=point['pitch_deg'],
                    target_type=visual.get('target_type'),
                    target_kind=visual.get('target_kind'),
                    evidence=(visual.get('evidence') or '')[:250],
                    uncertainty=(visual.get('uncertainty') or '')[:160],
                    expected_progress=visual.get('expected_progress','UNKNOWN'),
                    probe=probe,
                    direction_is_a_hypothesis_not_motion_authorization=True)
                if blocked:
                    repeated=any(
                        b['heading_deg']==heading and
                        b['probe']['depth_hash']==source.get('depth_hash')
                        for b in blocked_approaches)
                    if not repeated:
                        blocked_approaches.append(proposal)
                    # Same-frame requery is explicitly permitted once to let
                    # the VLM choose a DIFFERENT camera view / bearing.
                    if repeated:
                        return finish('NO_RELIABLE_CANDIDATE',
                            reason='same_blocked_bearing_reproposed_without_new_depth')
                    pending_view_correction=dict(
                        reason='candidate_forward_measured_center_obstacle',
                        rejected_heading_deg=heading,
                        rejection_observation_id=source['observation_id'],
                        current_depth_probe=probe,
                        required_next='look_or_turn_to_another_bearing_or_finish',
                        note='VLM_choose_a_new_view; NO_forward_was_executed')
                    continue
                if not any(p['observation_id']==proposal['observation_id'] and
                           p['heading_deg']==proposal['heading_deg']
                           for p in approach_candidates):
                    approach_candidates.append(proposal)
                # Keep the VLM's final chosen hypothesis as the summary.
                # Do not impose Python semantic priority between ADVANCE,
                # DETOUR and REVERSE; NavigationAgent evaluates ALL options.
                best_approach=proposal
                if candidate_found_checkpoint is None:
                    candidate_found_checkpoint=len(record['checkpoints'])-1
                # Persist the accepted bearing's *visual* evidence rather
                # than retaining a higher-confidence but blocked landmark
                # from an earlier viewing direction as the top approach.
                if (visual.get('target_visible') is True and
                        visual.get('target_kind') in ('ENTRY','APPROACH')):
                    best_candidate=candidate
                if action=='finish':
                    return finish('APPROACH_CANDIDATE_FOUND',
                        reason='vlm_finished_with_measured_candidate')
            # Rejected VLM camera choices are corrected with another VLM
            # decision on the SAME genuine image. No main LLM call and no
            # physical TURN is spent on a duplicate heading/pitch pair.
            def request_new_view(reason,requested_heading=None,requested_pitch=None):
                nonlocal pending_view_correction,consecutive_replans
                rejected_views.append(dict(
                    observation_id=source['observation_id'],
                    action=action,turn_deg=visual['turn_deg'],
                    reason=reason,requested_heading_deg=requested_heading,
                    requested_pitch_deg=requested_pitch))
                consecutive_replans+=1
                if consecutive_replans>max_replans:
                    if best_approach:
                        return finish('APPROACH_CANDIDATE_FOUND',
                            reason='vlm_refinement_exhausted_same_frame_corrections')
                    if best_candidate:
                        return finish('TARGET_CANDIDATE_FOUND',
                            reason='vlm_refinement_exhausted_same_frame_corrections')
                    return finish('REPEATED_VIEW_REQUEST','NO_NEW_INFORMATION',
                        'vlm_exhausted_bounded_same_frame_view_corrections')
                pending_view_correction=dict(
                    rejected_action=action,rejected_turn_deg=visual['turn_deg'],
                    reason=reason,
                    required_next='choose_unseen_yaw_or_pitch_or_finish',
                    already_inspected=[
                        dict(heading_deg=p['heading_deg'],pitch_deg=p['pitch_deg'])
                        for p in record['checkpoints'][-12:]])
                return None
            if action=='finish':
                if best_approach is not None:
                    return finish('APPROACH_CANDIDATE_FOUND',
                        reason='vlm_finished_after_candidate_comparison')
                if blocked_approaches:
                    return finish('NO_RELIABLE_CANDIDATE',
                        reason='vlm_finished_after_measured_candidate_obstacle')
                return finish('TARGET_CANDIDATE_FOUND' if best_candidate else
                              'CUE_ONLY' if best_cue else 'VLM_SEARCH_FINISHED',
                    reason='vlm_concluded_visual_search')
            # A candidate is a hypothesis, not an observation-stop signal.
            # Honor VLM's requested follow-up views, then bound the cost of
            # continued refinement: after three additional genuine visual
            # checkpoints hand the retained candidate to NavigationAgent.
            if (candidate_found_checkpoint is not None
                    and len(record['checkpoints'])-1-candidate_found_checkpoint
                        >=max_candidate_refinement_views):
                return finish('APPROACH_CANDIDATE_FOUND' if best_approach else
                              'NO_RELIABLE_CANDIDATE' if blocked_approaches else
                              'TARGET_CANDIDATE_FOUND',
                    reason='bounded_vlm_candidate_refinement_completed')
            if action in ('look_up','look_down'):
                new_pitch=r.pitch_deg+(r.camera.tilt_angle if action=='look_up'
                                      else -r.camera.tilt_angle)
                if abs(new_pitch)>60:
                    result=request_new_view('camera_pitch_outside_allowed_range',
                        requested_pitch=new_pitch)
                    if result is not None:return result
                    continue
                proposed_key=(key[0],round(new_pitch/r.camera.tilt_angle))
                if proposed_key in seen:
                    result=request_new_view('already_inspected_heading_pitch',
                        requested_heading=round(math.degrees(
                            r.motion_controller.pose()['heading']),2),
                        requested_pitch=new_pitch)
                    if result is not None:return result
                    continue
                consecutive_replans=0
                failed=do_action('LOOK_UP' if action=='look_up' else 'LOOK_DOWN')
                if failed:return finish('INTERRUPTED','PARTIAL',failed.get('reason'))
                continue
            degrees=visual['turn_deg']
            if degrees<=0 or degrees%turn or turned+degrees>sweep_degrees:
                result=request_new_view('turn_exceeds_remaining_observation_budget')
                if result is not None:return result
                continue
            direction='TURN_LEFT' if action=='turn_left' else 'TURN_RIGHT'
            target_heading=(key[0]+(
                degrees/turn if direction=='TURN_LEFT' else -degrees/turn))%round(360/turn)
            if (int(round(target_heading)),key[1]) in seen:
                result=request_new_view('already_inspected_heading_pitch',
                    requested_heading=round(target_heading*turn,2),
                    requested_pitch=r.pitch_deg)
                if result is not None:return result
                continue
            # An expensive full undo of a prior sweep is not a new camera
            # search: ask VLM to choose an alternative. It may still view the
            # same yaw at a new pitch through shorter, deliberate motions.
            # Do not confuse passed-through yaw with VLM-inspected RGB.
            if (last_turn_direction is not None and last_turn_direction!=direction
                    and degrees>=90 and turned+degrees>=180
                    and int(round(target_heading)) in yaw_traversed):
                result=request_new_view('large_reverse_sweep_to_previously_visited_yaw',
                    requested_heading=round(target_heading*turn,2),
                    requested_pitch=r.pitch_deg)
                if result is not None:return result
                continue
            consecutive_replans=0
            for _turn in range(round(degrees/turn)):
                failed=do_action(direction)
                if failed:return finish('INTERRUPTED','PARTIAL',failed.get('reason'))
                turned+=turn
                yaw_traversed.add(view_key()[0])
            last_turn_direction=direction
            if turned>=sweep_degrees:
                # The VLM gets a final look at the last real direction if a
                # checkpoint remains; otherwise do not issue an extra turn.
                continue
        return finish('SUBGOAL_UNCERTAIN' if mode=='VERIFY' else
                      'APPROACH_CANDIDATE_FOUND' if best_approach else
                      'NO_RELIABLE_CANDIDATE' if blocked_approaches else
                      'TARGET_CANDIDATE_FOUND' if best_candidate else
                      'CUE_ONLY' if best_cue else 'SEARCH_REGION_COVERED_NO_MATCH',
            reason='bounded_vlm_checkpoints_exhausted')

