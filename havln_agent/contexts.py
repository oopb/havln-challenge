"""Decision-sized projections; raw records and RGB-D stay in Memory Service."""
import math
import time

from .progress import turn_evidence, current_visual_evidence, registered_motion_history
from .memory_records import task_tokens
from .local_exploration import (last_block, local_coverage,
    multi_pitch_coverage, panorama_coverage, observation_stagnation,
    recent_forward_stall_evidence)


def latest_observation_for_subgoal(memory,index):
    """Only current milestone observations can define the active objective.

    Previous milestone reports remain available as historical evidence, not
    as unscoped instructions for the NavigationAgent.
    """
    generation=getattr(memory.task,'subgoal_generations',{}).get(str(index),0)
    return next((item for item in reversed(memory.observation_searches)
                 if item.get('subgoal')==index
                 and item.get('subgoal_generation',0)==generation),None)


def last_tool_result_for_subgoal(memory,index):
    """Ignore the previous milestone's last tool receipt after transition."""
    if not memory.tool_ledger:
        return None
    entry=memory.tool_ledger[-1]
    if entry.get('subgoal_id') not in (None,'sg_%d'%index):
        return None
    return entry.get('result')


def fields(value,names):
    return {key:value[key] for key in names if key in value}



 
def local_geometry_measurement_scope(directions, turn_step_deg):
    """Describe the *sampled* current RGB-D directions, not a 360-degree map.

    Other headings are UNMEASURED; a missing executable direction is never
    evidence that every other bearing is blocked. Historical view coverage is
    kept separately and never authorizes new motion.
    """
    bins = max(1, round(360.0 / float(turn_step_deg)))
    step = 360.0 / bins
    sampled = []
    for direction in directions:
        try:
            angle = float(direction['angle_degrees'])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(angle):
            sampled.append((angle, direction))
    checked = {round((angle + 180.0) / step) % bins for angle, _ in sampled}
    unchecked = [index for index in range(bins) if index not in checked]
    runs = []
    for index in unchecked:
        if runs and index == runs[-1][1] + 1:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    return dict(
        evidence_scope='current_RGBD_snapshot_relative_to_current_heading',
        sampled_relative_bearings_deg=sorted(set(round(a, 2) for a, _ in sampled)),
        sampled_count=len(checked),
        yaw_bin_width_deg=round(step, 2),
        unmeasured_yaw_bin_count=len(unchecked),
        unmeasured_relative_bearing_ranges_deg=[
            [round(-180.0 + lo * step, 2), round(-180.0 + hi * step, 2)]
            for lo, hi in runs],
        ranges_are_inclusive_yaw_bins=True,
        confirmed_executable_relative_bearings_deg=[
            round(a, 2) for a, d in sampled if d.get('motion_state') == 'EXECUTABLE'],
        unconfirmed_but_not_blocked_relative_bearings_deg=[
            round(a, 2) for a, d in sampled if d.get('motion_state') == 'UNCONFIRMED'],
        explicitly_blocked_relative_bearings_deg=[
            round(a, 2) for a, d in sampled if d.get('motion_state') == 'BLOCKED'],
        sampled_unconfirmed=[
            dict(relative_bearing_deg=round(a, 2),
                 motion_state=d.get('motion_state', 'UNKNOWN'),
                 reason=d.get('reason'))
            for a, d in sampled if d.get('motion_state') not in ('EXECUTABLE', 'BLOCKED')],
        full_circle_sampled=len(checked) == bins,
        sampled_evidence_is_not_future_motion_permission=True,
        interpretation=('Only confirmed measured obstacles veto movement. Missing '
                        'depth, incomplete FOV and unlocalized semantics are '
                        'UNCONFIRMED, not BLOCKED. Every forward uses fresh RGB-D.'))


def collision_summary(result):
    """Keep decisive measured data, not a VLM guess about obstacle identity."""
    safety=result.get('safety') or {}
    validation=result.get('validation') or {}
    depth=(safety if 'clearance' in safety else validation.get('geometry') or {})
    if not depth and isinstance(result.get('geometry'),dict):
        depth=result['geometry'].get('front') or {}
    if not depth:return None
    people=depth.get('human_occupancy') or depth.get('human_evidence') or {}
    terrain=safety.get('terrain_evidence') or safety.get('terrain_support') or depth.get('terrain_support') or {}
    if not isinstance(terrain,dict):terrain=dict(status=str(terrain))
    values=fields(depth,('path_state','clearance','required_forward_step',
        'nearest_obstacle_distance','nearest_collision_travel','rejection_reasons',
        'unknown_sweep_pixels','collision_points','nearest_collision_point'))
    values['people']=fields(people,('confirmed','uncertain','unlocalized',
        'corridor_points','nearest_distance'))
    values['terrain']=fields(terrain,('status','reason','direction',
        'initial_height_m','end_height_m','max_step_height_m'))
    if terrain.get('samples'):
        values['terrain']['samples']=[
            fields(v,('along_m','height_m','height_spread_m','points','required_half_width_m'))
            for v in terrain['samples'][-8:]]
    if values.get('path_state')=='OCCUPIED' and not values['people'].get('corridor_points'):
        values['identity_note']='measured_depth_obstacle; semantic identity unverified; do not assume human'
    return values


def waypoint_summary(path):
    """Keep semantic target alignment in compact waypoint context."""
    value = fields(path, ('id', 'type', 'bearing_deg', 'range_m', 'first_step',
                          'goal_status', 'source_observation_id', 'objective',
                          'reason', 'first_atomic_action',
                          'requires_post_turn_recheck', 'execution_status'))
    associations = path.get('goal_association') or []
    if associations:
        best = max(associations, key=lambda item:
                   item.get('distance_before_m', 0) - item.get('distance_after_m', 0))
        value['goal_alignment'] = fields(best, (
            'grounding_id', 'target', 'distance_before_m', 'distance_after_m',
            'bearing_error_degrees', 'direction_compatibility',
            'rejection_reasons', 'stair_direction_hypothesis'))
    failures = path.get('failure_history') or []
    if failures:
        value['failed_approaches'] = [fields(item, ('reason', 'basis', 'confidence'))
                                      for item in failures[-2:]]
    return value


def compact_vlm_reports(reports, limit=8):
    """Retain unfiltered VLM statements with real frame/yaw/pitch citations.

    Truncate prose only to keep the cheap NavigationAgent's context bounded;
    never discard a report for mismatched target vocabulary or low confidence.
    Full structured outputs are kept in observation records/diagnostics.
    """
    results=[]
    for report in list(reports or [])[-limit:]:
        item=fields(report,(
            'observation_id','heading_deg','pitch_deg','target_visible',
            'target_type','target_kind','confidence','box','evidence',
            'uncertainty','next_action','reason','relative_turn_deg_from_current',
            'candidate_approach','approach_probe','usable_for_live_safety'))
        for name,max_chars in (('evidence',180),('uncertainty',120),('reason',90)):
            if isinstance(item.get(name),str):
                item[name]=item[name][:max_chars]
        results.append(item)
    return results


def tool_summary(result):
    value=fields(result,('tool','status','reason','suggestion','missing_evidence','recoverable','interrupted_status',
        'requested_actions','executed_actions','executed_motion','end_observation_id',
        'observation_owner','best_candidate','best_cue',
        'movement_intent_feedback',
        'remaining_distance','endpoint_estimate_valid','reach_basis',
        'action_intent_completed','remaining_commanded_forward_actions',
        'nominal_remaining_command_distance_m','commanded_forward_actions',
        'commanded_forward_horizon','odometry_unknown_during_route',
        'vertical_motion','vertical_translation_m_est','semantic_direction_mismatch','waypoint_type','waypoint_arrival_completes_task','completion_confirmed','criteria_verdicts','missing','supported','information_options'))
    if result.get('status') in ('BLOCKED','NEEDS_VIEW','REQUIRES_ATTENTION') or result.get('safety'):
        collision=collision_summary(result)
        if collision:value['measured_collision']=collision
    if 'paths' in result:
        value['paths']=[waypoint_summary(p) for p in result['paths']]
    if 'semantic' in result:
        value['grounding']=dict(fields(result,('grounding_id','target','status','localization',
            'entry_status','stair_direction','traversable_access')),
            uncertainty=result['semantic'].get('uncertainty'),
            position=fields(result.get('position') or {},('status','reason','bearing_degrees',
                'x_right','z_forward','depth_spread_metres','uncertainty_m')),
            entry=fields(result.get('entry') or {},('status','reason','bearing_degrees',
                'x_right','z_forward','depth_spread_metres')),
            interpretation='observation_only_not_motion_permission')
    if 'assessment' in result:value['assessment']=result['assessment']
    if 'memory' in result:value['memory']=result['memory']
    if 'waypoint_id' in result:value['waypoint_id']=result['waypoint_id']
    if 'requested_motion' in result:value['requested_motion']=result['requested_motion']
    if 'pixel_target' in result:value['pixel_target']=result['pixel_target']
    if 'geometry' in result:
        value['geometry']=fields(result['geometry']['front'],('motion_state','depth_state','clearance',
            'nearest_obstacle_distance','nearest_collision_point','unknown_fraction',
            'rejection_reasons','terrain_support','human_evidence','human_occupancy'))
    if 'geometry' in value and 'terrain_support' in value['geometry']:
        value['geometry']['terrain_support']=fields(value['geometry']['terrain_support'],
            ('status','reason','direction','initial_height_m','end_height_m','max_step_height_m','scope'))
    if 'directions' in result:
        value['directions']=result['directions']
    if 'outcome' in result and result.get('tool')=='observe_for_goal':
        value['observation_task']=dict(
            outcome=result['outcome'],
            covered_turn_deg=result.get('covered_turn_deg'),
            planned_sweep_deg=result.get('planned_sweep_deg'),
            best_candidate=result.get('best_candidate'),
            best_cue=result.get('best_cue'),
            best_approach=result.get('best_approach'),
            navigation_feedback=result.get('navigation_feedback'),
            observation_mode=result.get('observation_mode'),
            verification=result.get('verification'),
            subgoal_transition=result.get('subgoal_transition'),
            approach_candidates=result.get('approach_candidates',[])[:3],
            blocked_approaches=result.get('blocked_approaches',[])[:6],
            visual_reports=compact_vlm_reports(result.get('visual_reports')),
            observation_owner=result.get('observation_owner'),
            checked_views=[fields(p,('observation_id','heading_deg','pitch_deg',
                'evidence_origin')) for p in result.get('checkpoints',[])],
            note='approach_candidates_are_hypotheses; only_measured_central_obstacles_are_rejected_inside_observation; navigation_agent_owns_motion')
    if 'obstacle_observation' in result:
        item=result['obstacle_observation']
        value['obstacle_observation']=dict(
            source_observation_id=item.get('source_observation_id'),
            depth=fields(item.get('depth') or {},(
                'confirmed','central_support_pixels','nearest_along_m',
                'nearest_depth_m','nearest_height_m',
                'focus_point_xy_norm','focus_region_xyxy_norm',
                'clearance_m','body_step_m','identity','reason')),
            visual=fields(item.get('visual') or {},(
                'status','visible','description','visual_relation',
                'uncertainty','evidence_origin','error_type')),
            visual_reused_for_identical_frame=item.get('visual_reused_for_identical_frame'),
            identity_is_visual_hypothesis_only=True,
            next_action_selected_by='navigation_agent')
    if 'alternatives' in result:value['alternatives']=result['alternatives']
    if 'view_directions' in result:value['view_directions']=result['view_directions'][:6]
    if 'scan' in result:
        value['observations']=result['scan']['frames']
    if result.get('motion_alert'):
        value['motion_alert']=fields(result['motion_alert'],('reason','observation_before','observation_after'))
    if result.get('recovery'):
        value['recovery']=result['recovery']
    if result.get('interrupted_status'):
        value['interrupted_status']=result['interrupted_status']
    if result.get('exploration'):
        value['exploration']=result['exploration']
    if result.get('no_new_information_reason'):
        value['no_new_information_reason']=result['no_new_information_reason']
    return value


def planner_context(runtime):
    task=runtime.memory.task
    index=task.current_subgoal_index
    return dict(current_subgoal=fields(task.subgoals[index],(
        'action','target','landmark','completion_description','semantic_description')),
        verified_milestones=[dict(subgoal=m['subgoal'],source_observation_ids=m.get('observation_ids',[])[-2:],
            executed_action_count=len(m.get('executed_steps',[])),supported=m['verification']['supported']) for m in task.completion_evidence[-3:]],
        new_evidence=runtime.memory.recall_observation(runtime.memory.current_place_id,
            landmark=task.subgoals[index]['target'],top_k=3)['observations'],
        review_reason=runtime.review_reason,
        actions_remaining=runtime.config['max_steps']-len(runtime.memory.actions))


def navigation_progress_review(memory):
    """Compact, chronological evidence review; never a completion verdict.

    It summarizes *registered* local RGB-D motion and VLM visual hypotheses.
    Relative y measurements from different pose epochs are not added into a
    global position. The NavigationAgent alone may request verification or
    select the next UNVERIFIED semantic goal.
    """
    task=memory.task
    index=task.current_subgoal_index
    goals=getattr(task,'subgoals',[])
    history=registered_motion_history(memory)
    rows=history.get('rows',[])
    directed=[]
    for row in rows:
        if (row.get('status')!='ESTIMATED'
                or row.get('delta_y_m') is None
                or row.get('uncertainty_m') is None):
            continue
        displacement=row['delta_y_m']
        uncertainty=row['uncertainty_m']
        direction=('UP' if displacement>uncertainty else
                   'DOWN' if displacement<-uncertainty else 'LEVEL_OR_UNCERTAIN')
        if direction in ('UP','DOWN'):
            directed.append(dict(step=row['step'],direction=direction,
                observation_id=row.get('observation_after'),
                heading_deg=row.get('heading_deg'),
                delta_y_m_est=round(displacement,3),
                uncertainty_m_est=round(uncertainty,3),
                motion_epoch=row.get('motion_epoch')))
    # Only actual chronological, registered measurement events are described.
    segments=[]
    for row in directed:
        if segments and segments[-1]['direction']==row['direction']:
            segments[-1]['count']+=1
            segments[-1]['last_step']=row['step']
            segments[-1]['last_observation_id']=row['observation_id']
        else:
            segments.append(dict(direction=row['direction'],
                first_step=row['step'],last_step=row['step'],count=1,
                last_observation_id=row['observation_id']))
    latest_targets=[]
    seen=set()
    for target in reversed(getattr(memory,'visual_targets',[])):
        if (target.get('subgoal_index')!=index or
                target.get('subgoal_generation',0)!=
                task.subgoal_generations.get(str(index),0)):
            continue
        key=(target.get('target_type'),target.get('target_kind'),
             target.get('selected_observation_id'))
        if key in seen:
            continue
        seen.add(key)
        latest_targets.append(dict(
            observation_id=target.get('selected_observation_id'),
            target_type=target.get('target_type'),
            kind=target.get('target_kind'),
            status=target.get('status'),
            heading_deg=target.get('absolute_heading_deg'),
            candidate_id=target.get('id'),
            pitch_deg=target.get('pitch_deg'),
            evidence=(target.get('visual_evidence') or '')[:130],
            expected_progress=target.get('expected_progress','UNKNOWN'),
            evidence_class='VLM_HYPOTHESIS_NOT_NAVIGATION_PROOF'))
        if len(latest_targets)==3:
            break
    latest_targets.reverse()
    changed_direction=any(a['direction']!=b['direction']
                          for a,b in zip(segments,segments[1:]))
    last_verifier=next((
        item.get('verification') for item in reversed(
            getattr(memory,'observation_searches',[]))
        if item.get('intent')=='VERIFY' and
        item.get('subgoal')==index and
        item.get('subgoal_generation',0)==task.subgoal_generations.get(
            str(index),0) and item.get('verification')
    ),None)
    return dict(
        current_subgoal_index=index,
        next_subgoal=(fields(goals[index+1],
            ('action','target','landmark','semantic_description'))
            if index+1<len(goals) else None),
        registered_forward_steps=history.get('registered_forward_steps',0),
        unknown_forward_steps=len(history.get('unknown_forward_steps',[])),
        observed_downward_m_est=history.get('observed_downward_m_est',0),
        observed_upward_m_est=history.get('observed_upward_m_est',0),
        directional_history=segments[-5:],
        observed_direction_reversal=changed_direction,
        latest_directional_step=(directed[-1] if directed else None),
        latest_visual_hypotheses=latest_targets,
        route_prediction_feedback=copy.deepcopy(memory.evidence.route_outcomes[-5:]),
        recent_surface_measurements=copy.deepcopy(
            memory.evidence.surface_measurements[-3:]),
        recent_verification_evidence=copy.deepcopy(
            memory.evidence.verifications[-3:]),
        latest_blocked_stances=copy.deepcopy(memory.evidence.obstacle_blocks[-4:]),
        last_verification=(fields(last_verifier,('status','evidence',
            'uncertainty','source_observation_id')) if last_verifier else None),
        interpretation='local_measured_segments_are_historical_not_global_pose; '
            'visual_targets_are_unverified_hypotheses; no_subgoal_auto_completion',
        model_choices='If evidence suggests a phase change, evaluate '
            'observe_for_goal(mode=VERIFY), targeted ground_landmark, or model-owned '
            'continue the current milestone; only VLM VERIFY can advance it.')


def navigation_context(runtime):
    pose=runtime.motion_controller.pose()
    memory=runtime.memory
    index=memory.task.current_subgoal_index
    language=memory.task.subgoals[index] if index<len(memory.task.subgoals) else {}
    view=dict(observation_id=memory.current['observation_id'],semantic_status='NOT_ANALYZED')
    scene=memory.scene_memory
    historical_hint=None
    if scene:
        wanted=task_tokens(language.get('landmark','')+' '+language.get('target',''))
        landmarks=sorted(scene['scene'].get('landmarks',[]),
            key=lambda item:-len(wanted & task_tokens(item['name'])))[:4]
        if current_visual_evidence(scene['source'],memory,time.monotonic(),runtime.config['max_scene_age_seconds']):
            view.update(fields(scene['scene'],('description','uncertainty')),landmarks=landmarks,
                source_observation_id=scene['source']['observation_id'],semantic_status='CURRENT')
            view['people']=[dict(fields(p,('activity','description')),evidence_state='model_hypothesis')
                for p in (scene['scene'].get('humans',[])+scene['scene'].get('uncertain_humanoids',[]))[:2]]
        else:
            historical_hint=dict(observation_id=scene['source']['observation_id'],
                description=scene['scene'].get('description',''),landmarks=landmarks[:2],
                evidence_state='model_hypothesis',temporal_status='HISTORICAL',usable_for_current_view=False)
    hints=memory.recall_observation(memory.current_place_id,
        landmark=language.get('landmark','')+' '+language.get('target',''),top_k=3)['observations']
    # A verified sg_0 report is historical evidence, not the sg_1 tool
    # feedback. Never re-feed the prior goal's prose as the current one.
    last=last_tool_result_for_subgoal(memory,index)
    navigation_memory = (memory.navigation_summary() if hasattr(memory, 'navigation_summary') else {})
    grounding_hints = []
    for item in reversed(memory.groundings):
        if item['source'].get('subgoal') != index or len(grounding_hints) >= 3:
            continue
        grounding_hints.append(dict(grounding_id=item.get('grounding_id'),
            target=item.get('target'),status=item.get('status'),
            source_observation_id=item['source'].get('observation_id'),
            position=fields(item.get('position') or {},('status','reason','bearing_degrees',
                'x_right','z_forward','depth_spread_metres')),
            entry=fields(item.get('entry') or {},('status','reason','bearing_degrees')),
            evidence_uncertainty=(item.get('semantic') or {}).get('uncertainty'),
            is_current_view=(item['source'].get('rgb_hash') == memory.current.get('rgb_hash')
                and item['source'].get('depth_hash') == memory.current.get('depth_hash')
                and item['source'].get('pose') == pose),
            interpretation='visual_location_hypothesis_not_motion_permission'))
    prior_motion_evidence=[]
    for transition in reversed(getattr(memory.task,'provisional_transitions',[])):
        idx=transition['from_subgoal']
        if idx==index or any(x['subgoal_index']==idx for x in prior_motion_evidence):
            continue
        prior_motion_evidence.append(dict(subgoal_index=idx,
            **{k:v for k,v in registered_motion_history(memory,subgoal=idx).items()
               if k!='rows'}))
        if len(prior_motion_evidence)>=2:
            break
    result=dict(goal=runtime.active_subgoal,
        progress_review=navigation_progress_review(memory),
        prior_subgoal_motion_evidence=prior_motion_evidence,
        task_selection=dict(
            current_subgoal_index=index,
            subgoal_generation=memory.task.subgoal_generations.get(str(index),0),
            total_subgoals=len(memory.task.subgoals),
            verified_subgoals=list(getattr(memory.task,'completed_subgoals',[])),
            provisional_transitions=getattr(memory.task,'provisional_transitions',[])[-3:],
            unverified_subgoal_indices=[i for i in range(len(memory.task.subgoals))
                if i not in getattr(memory.task,'completed_subgoals',[])],
            select_subgoal_is_navigation_decision_not_verified_completion=True),
        historical_motion_evidence={key:value for key,value in registered_motion_history(memory).items()
            if key!='rows'},
        route_progress=memory.route_progress_summary(),
        action_calibration=dict(turn_step_deg=runtime.camera.turn_angle,
            forward_step_m=getattr(runtime.camera,'forward_step',None),
            turn_and_observe_max_deg=runtime.camera.turn_angle *
                min(runtime.config.get('tool_max_actions',3),
                    max(1,runtime.config['max_steps']-1-len(memory.actions))),
            unknown_depth_is_obstacle=False,
            geometry_policy='measured_Depth_is_Observation_Agent_input; only_fresh_centerline_probe_before_MOVE',
            geometry_autorun_during_move='central_obstacle_only',
            motion_veto='measured_centerline_occupancy_or_simulator_action_legality',
            direct_segment_max_m=3.0,
            navigation_execution_policy='continuous_until_route_or_pose_or_episode_budget',
            per_tool_atomic_action_cap=None,
            remaining_episode_atomic_actions=max(
                0,runtime.config['max_steps']-1-len(memory.actions))),
        grounding_evidence=grounding_hints,
        navigation_mode=getattr(runtime, 'navigation_mode', 'OBSERVE'),
        navigation_memory=navigation_memory,
        pose=dict(place_id=memory.current_place_id,heading_deg=math.degrees(pose['heading']),
            pitch_deg=runtime.pitch_deg,
            position_xz_m_est=([pose['x'],pose['z']]
                if pose.get('position_coordinates_valid',True) else None),
            height_y_m_est=pose['y'],
            confidence=('estimated_relative' if pose.get('global_chain_continuous',True)
                        else 'action_only_after_unknown_translation'),
            position_uncertainty_m=(pose['position_uncertainty']
                if pose.get('position_coordinates_valid',True) else None),
            viewpoint_generation=pose.get('viewpoint_generation',0),
            unknown_forward_actions=pose.get('unknown_forward_actions',0),
            position_coordinates_valid=pose.get('position_coordinates_valid',True),
            motion_epoch=pose.get('motion_epoch',0),
            coordinate_scope=pose.get('coordinate_scope','episode_relative'),
            global_chain_continuous=pose.get('global_chain_continuous',True),
            local_anchor_observation_id=pose.get('local_anchor_observation_id')),
        current_view=view,legal_look=[look for look,delta in [('up',runtime.camera.tilt_angle),('down',-runtime.camera.tilt_angle)]
            if abs(runtime.pitch_deg+delta)<=60],
        local_geometry=[dict(fields(d,('angle_degrees','motion_state','reason')),
            clearance_m=(d.get('clearance') if d.get('motion_state') in
                ('EXECUTABLE','BLOCKED') else None),
            clearance_is_measured=d.get('motion_state') in ('EXECUTABLE','BLOCKED'))
            for d in sorted(memory.current.get('directions',[]),
                key=lambda d:abs(d['angle_degrees']))[:7]],
        local_geometry_scope=local_geometry_measurement_scope(
            memory.current.get('directions', []), runtime.camera.turn_angle),
        memory_hints=hints,dynamic_hints=memory.dynamic_summaries(memory.current_place_id)[-2:],
        last_tool=tool_summary(last) if last else None,
        actions_remaining=runtime.config['max_steps']-len(memory.actions))
    orientation=turn_evidence(memory)
    if orientation['requirement']:
        result['orientation']=fields(orientation,('requirement','remaining_deg','allowed_directions','completed'))
    # Make actionable candidates survive the next LLM call. A prior tool's
    # waypoints otherwise disappear from last_tool after a single scan/query,
    # driving repeated sensing rather than the chosen local movement.
    current=memory.current
    recent=[]
    for path in reversed(list(runtime.paths.values())):
        source=path['source']
        if (source['subgoal']!=memory.task.current_subgoal_index or
            source['pose']!=pose or source['rgb_hash']!=current['rgb_hash'] or
            path.get('execution_status') in ('CANCELLED','BLOCKED','REQUIRES_ATTENTION','TARGET_REACHED')):
            continue
        if path.get('first_step')=='BLOCKED':
            continue
        recent.append(waypoint_summary(path))
        if len(recent)==9:break
    if recent:
        result['candidate_waypoints']=recent
        result['candidate_note']='old_preview_only; navigate_local rechecks current RGB-D before each movement'
    result['motion_tools_note']=('Only fresh coherent central Depth occupancy interrupts '
        'atomic MOVE; then Observation Agent reviews current RGB-D and measured '
        'geometry under OBSTACLE_REVIEW. NavigationAgent has no independent '
        'Geometry tool. RGB-D estimates movement after each executed action.')
    if runtime.active_waypoint_id and runtime.motion_controller.target is not None:
        target=runtime.motion_controller.target
        attempt=memory.active_attempt or {}
        unverified=bool(attempt.get('unknown_motion_seen'))
        # Pending means the finite target has not been consumed. It does NOT
        # imply fresh clearance, nor that a stale endpoint can be re-used.
        recent_attempt=next((
            entry.get('result') or {}
            for entry in reversed(memory.tool_ledger)
            if (entry.get('call') or {}).get('tool') in
                ('navigate_local','navigate_relative','navigate_to_pixel')
            and ((entry.get('result') or {}).get('waypoint_id') or
                 (entry.get('call') or {}).get('args',{}).get('waypoint_id'))
                   ==runtime.active_waypoint_id), {})
        last_status=recent_attempt.get('interrupted_status') or recent_attempt.get('status')
        pending_valid=last_status not in (
            'NEEDS_VIEW','BLOCKED','REQUIRES_ATTENTION','STALE_TARGET')
        result['local_navigation']=dict(waypoint_id=runtime.active_waypoint_id,
            status='IN_PROGRESS',
            route_status='IN_PROGRESS',
            execution_feasibility=('MODEL_DECIDES_CONTINUE_OR_REPLAN'
                if not pending_valid else 'FINITE_ACTION_PLAN_PENDING'),
            last_execution_status=last_status,
            last_failure_reason=(recent_attempt.get('reason') if not pending_valid else None),
            next_action=runtime.motion_controller.route_action(
                attempt['target'],attempt.get('forward_actions',0),
                attempt.get('forward_action_horizon',1)) if attempt.get('target') else None,
            remaining_distance_m=None,
            endpoint_estimate_valid=False,
            relative_motion_only=True,
            nominal_remaining_command_distance_m=round(max(0,
                attempt.get('forward_action_horizon',0)-attempt.get('forward_actions',0))
                *getattr(runtime.camera,'forward_step',0.25),3),
            unknown_odometry_is_diagnostic_only=True,
            forward_actions=attempt.get('forward_actions',0),
            forward_action_horizon=attempt.get('forward_action_horizon'),
            suggestion=('old_waypoint_is_not_a_currently_executable_direction; '
                        'obtain specific new evidence or choose a different model-led route'
                if not pending_valid else
                'waypoint_is_pending_only; it resumes finite model action counts without metric pose gating'))
    # Persistent physical obstruction: use the END pose, and recognize
    # PARTIAL + interrupted_status=BLOCKED from a waypoint that moved first.
    # Historical depth is evidence of failure, never fresh motion permission.
    block=last_block(memory,pose)
    if block:
        for item in reversed(memory.tool_ledger):
            if item.get('call_id')==block['tool_call_id']:
                measured=collision_summary(item['result'])
                if measured:block['measured_collision']=measured
                break
        result['recent_navigation_block']=block
    # A spatially coherent yaw/pitch map avoids treating LOOK_UP -> LOOK_DOWN
    # revisits as novel exploration merely because the camera pitch changed.
    coverage_pose=dict(pose,pitch_deg=runtime.pitch_deg)
    result['local_exploration']=local_coverage(memory,coverage_pose,runtime.camera.turn_angle,
        round(360/runtime.camera.turn_angle))
    result['observation_coverage']=multi_pitch_coverage(
        memory,coverage_pose,runtime.camera.turn_angle,
        runtime.camera.tilt_angle)
    panorama=panorama_coverage(memory, coverage_pose,
        runtime.camera.turn_angle,
        getattr(runtime.camera,'hfov',runtime.camera.turn_angle))
    # Only surface detailed historical bearings after the panoramic threshold.
    # These are proposals, NOT current walking clearance or live human evidence.
    result['panoramic_observation']=(
        panorama if panorama['all_azimuths_observed'] else
        fields(panorama, ('reference','covered_yaw_bins','total_yaw_bins',
                         'coverage_fraction','all_azimuths_observed',
                         'motion_authorized')))
    stagnation=observation_stagnation(memory,coverage_pose)
    plausible_directions=[dict(relative_turn_deg=d.get('angle_degrees'),
                               current_snapshot_status=d.get('motion_state'),
                               clearance_m=(d.get('clearance') if
                                   d.get('motion_state')=='EXECUTABLE' else None))
        for d in memory.current.get('directions',())
        if d.get('motion_state')!='BLOCKED'][:7]
    stagnation['currently_plausible_local_bearings']=plausible_directions
    stagnation['current_bearings_are_not_movement_authorization']=True
    # The latest motion intent can be obscured by several read-only scans.
    # Keep its physically executed receipt, not just the requested tool name.
    recent_move=next((entry for entry in reversed(memory.tool_ledger)
        if (entry.get('call') or {}).get('tool') in
            ('navigate_relative','navigate_to_pixel','navigate_local')
        and entry.get('subgoal_id') in (None,'sg_%d' % index)),None)
    if recent_move:
        move_result=recent_move.get('result') or {}
        receipt=move_result.get('movement_intent_feedback') or {}
        result['recent_motion_intent']=dict(
            tool_call_id=recent_move.get('call_id'),
            tool=(recent_move.get('call') or {}).get('tool'),
            status=move_result.get('interrupted_status') or
                move_result.get('status'),
            reason=move_result.get('reason'),
            **receipt)
    stall=recent_forward_stall_evidence(memory,pose,runtime.camera.turn_angle)
    if stall:
        result['forward_stall_evidence']=stall
    if stagnation['repeated_low_information']:
        stagnation['intervention']=(
            'DIRECTIONAL_ACTION_DECISION_REQUIRED'
            if plausible_directions else 'BLOCKED_ACTIONABILITY_REASSESSMENT')
        stagnation['no_unconditional_forward']=True
        stagnation['model_must_propose']=(
            'Choose a task-directed finite route, executed continuously until arrival or a live safety/pose interruption.'
            if plausible_directions else
            'No measured non-blocked bearing exists in this snapshot; unmeasured headings remain unknown, not blocked. Choose a purposeful route or new observation.')
    result['observation_stagnation']=stagnation
    if current.get('revisited_static_view'):
        result['revisited_static_view']=dict(
            matched_observation_ids=current.get('revisited_view_ids',[])[-3:],
            evidence_type='historical_static_appearance_only',
            live_persons_and_motion_authorized=False)
        # A historical image is only a static semantic hint. It never supplies
        # a live human box, free space, or current geometric safety.
        matching=set(current.get('revisited_view_ids',[]))
        for item in reversed(memory.scene_history):
            source=item.get('source',{})
            if source.get('observation_id') in matching:
                scene_hint=item['scene']
                result['historical_static_scene']=dict(
                    source_observation_id=source.get('observation_id'),
                    description=scene_hint.get('description','')[:350],
                    landmarks=[fields(l,('name','direction')) for l in scene_hint.get('landmarks',[])[:4]],
                    evidence_state='historical_hypothesis_not_live_safety')
                break
    recent_search=latest_observation_for_subgoal(memory,index)
    if recent_search:
        result['latest_observation_task']=dict(
            objective=recent_search['objective'],outcome=recent_search['outcome'],
            subgoal_index=index,
            mode=recent_search.get('mode','legacy_fixed_direction'),
            direction=recent_search['direction'],
            covered_turn_deg=recent_search.get('covered_turn_deg',0),
            candidate=recent_search.get('candidate'),
            best_cue=recent_search.get('best_cue'),
            best_approach=recent_search.get('best_approach'),
            navigation_feedback=recent_search.get('navigation_feedback'),
            observation_mode=recent_search.get('intent'),
            verification=recent_search.get('verification'),
            approach_candidates=recent_search.get('approach_candidates',[])[-3:],
            blocked_approaches=recent_search.get('blocked_approaches',[])[-6:],
            visual_reports=[dict(report,
                relative_turn_deg_from_current=round(
                    (report['heading_deg']-math.degrees(pose['heading'])+180)%360-180,2))
                for report in compact_vlm_reports(
                    recent_search.get('visual_reports'))],
            searched_headings_deg=[p['heading_deg'] for p in recent_search['checkpoints']][-5:])
    if historical_hint:result['historical_hint']=historical_hint
    liveness=getattr(runtime,'navigation_liveness',None) or {}
    if liveness.get('consecutive_unchanged_decisions',0)>=2:
        result['decision_liveness']=fields(liveness,(
            'consecutive_unchanged_decisions',
            'max_unchanged_evidence_decisions','review_requested','reason'))
    return result

def compact_navigation_context(state):
    """Small decision payload; full audit/memory remains available by tool."""
    if not state:return {}
    def short(x,n=160):return x[:n] if isinstance(x,str) else x
    out={}
    for key in ('instruction','goal','orientation','legal_look','actions_remaining',
                'decision_liveness','format_feedback'):
        if key in state:out[key]=state[key]
    for key,names in (
        ('task_selection',('current_subgoal_index','total_subgoals',
            'verified_subgoals','unverified_subgoal_indices')),
        ('pose',('heading_deg','pitch_deg','position_coordinates_valid',
            'position_xz_m_est','confidence','place_id','viewpoint_generation')),
        ('current_view',('observation_id','semantic_status','description',
            'uncertainty','source_observation_id')),
        ('action_calibration',('turn_step_deg','forward_step_m',
            'direct_segment_max_m','remaining_episode_atomic_actions')),
        ('local_navigation',('waypoint_id','status','last_failure_reason',
            'next_action','forward_actions','forward_action_horizon')),
        ('recent_motion_intent',('status','reason','executed_turn_actions',
            'executed_forward_actions','actual_forward_attempted')),
        ('recent_navigation_block',('reason','status','tool_call_id')),
        ('observation_stagnation',('intervention','repeated_low_information',
            'same_stance_observation_calls','low_information_calls')),
        ('observation_coverage',('unique_heading_pitch_cells','coverage_fraction')),
        ('panoramic_observation',('all_azimuths_observed','covered_yaw_bins',
            'total_yaw_bins','coverage_fraction')),
        ('historical_motion_evidence',('recorded_forward_steps',
            'registered_forward_steps','observed_downward_m_est',
            'observed_upward_m_est'))):
        if state.get(key):out[key]=fields(state[key],names)
    if 'current_view' in out:
        for k in ('description','uncertainty'):
            if k in out['current_view']:out['current_view'][k]=short(out['current_view'][k],220)
    if state.get('progress_review'):
        p=state['progress_review']
        out['progress_review']=fields(p,(
            'current_subgoal_index','registered_forward_steps',
            'unknown_forward_steps','observed_downward_m_est',
            'observed_upward_m_est','directional_history',
            'observed_direction_reversal','latest_directional_step',
            'latest_visual_hypotheses','route_prediction_feedback',
            'latest_blocked_stances','recent_surface_measurements',
            'recent_verification_evidence',
            'last_verification','interpretation','model_choices'))
    if state.get('route_progress'):
        out['route_progress']=state['route_progress']
    if state.get('forward_stall_evidence'):
        s=state['forward_stall_evidence']
        out['forward_stall_evidence']=fields(s,(
            'total_failed_forward_attempts','current_heading_deg',
            'current_heading_failed_forward_attempts','failed_bearings',
            'last_failed_step','last_failed_observation_id','evidence',
            'collision_proven','depth_obstacle_inspected',
            'interpretation','model_options'))
    if state.get('navigation_memory'):
        m=state['navigation_memory']
        n=fields(m,('expected_vertical_motion','total_forward_actions',
            'last_motion','last_progress_motion'))
        n['recent_forward_motion']=[fields(v,('step','heading_deg',
            'vertical_motion','vertical_change_m_est')) for v in
            m.get('recent_forward_motion',[])[-3:]]
        n['recent_failed_waypoints']=[fields(v,('reason','status',
            'requested_motion')) for v in m.get('recent_failed_waypoints',[])[-2:]]
        n['visual_targets']=[]
        current_index=(state.get('task_selection') or {}).get(
            'current_subgoal_index')
        current_generation=(state.get('task_selection') or {}).get(
            'subgoal_generation',0)
        for v in [x for x in m.get('visual_target_memory',[])
                  if x.get('subgoal_index')==current_index
                  and x.get('subgoal_generation',0)==current_generation][-2:]:
            t=fields(v,('id','status','target_type','target_kind',
                'selected_observation_id','absolute_heading_deg',
                'pitch_deg','target_region','expected_progress'))
            t['visual_evidence']=short(v.get('visual_evidence',''),150)
            t['missing_evidence']=short(v.get('missing_evidence',''),100)
            t['approach_candidates']=[dict(observation_id=a.get('observation_id'),
                kind=a.get('kind'),absolute_heading_deg=a.get('absolute_heading_deg'),
                evidence=short(a.get('evidence',''),90))
                for a in v.get('approach_candidates',[])[:2]]
            n['visual_targets'].append(t)
        out['navigation_memory']=n
    if state.get('grounding_evidence'):
        out['grounding_evidence']=[fields(v,('target','status',
            'source_observation_id','position','entry')) for v in
            state['grounding_evidence'][:2]]
    if state.get('candidate_waypoints'):
        out['candidate_waypoints']=[fields(v,('id','bearing_deg',
            'range_m','goal_alignment')) for v in state['candidate_waypoints'][:2]]
    if state.get('last_tool'):
        t=state['last_tool']
        out['last_tool']=fields(t,('tool','status','reason','interrupted_status',
            'executed_actions','end_observation_id','action_intent_completed',
            'remaining_commanded_forward_actions','waypoint_id'))
        if t.get('observation_task'):
            v=t['observation_task']
            out['last_tool']['observation_task']=dict(outcome=v.get('outcome'),
                best_candidate=v.get('best_candidate'),
                best_cue=v.get('best_cue'),
                best_approach=v.get('best_approach'),
                observation_mode=v.get('observation_mode'),
                verification=v.get('verification'),
                subgoal_transition=v.get('subgoal_transition'))
            if not state.get('latest_observation_task'):
                out['last_tool']['observation_task'].update(
                    navigation_feedback=v.get('navigation_feedback'),
                    approach_candidates=v.get('approach_candidates',[])[-3:],
                    blocked_approaches=v.get('blocked_approaches',[])[-4:],
                    checked_views=v.get('checked_views',[])[-3:])
            # Older or synthetic tool receipts may not have an observation
            # search record. Otherwise latest_observation_task is the single
            # canonical location for direct VLM evidence (avoid duplication).
            if not (state.get('latest_observation_task') or {}).get('visual_reports'):
                out['last_tool']['observation_task']['visual_reports']=compact_vlm_reports(
                    v.get('visual_reports'))
        if t.get('measured_collision'):
            out['last_tool']['measured_collision']=fields(t['measured_collision'],
                ('reason','clearance','motion_state','path_state',
                'nearest_obstacle_distance','nearest_collision_travel',
                'nearest_collision_point','people','identity_note'))
        if t.get('obstacle_observation'):
            v=t['obstacle_observation']
            visual=fields(v.get('visual') or {},(
                'status','visible','description','visual_relation',
                'uncertainty','evidence_origin','error_type'))
            for name in ('description','visual_relation','uncertainty'):
                if isinstance(visual.get(name),str):
                    visual[name]=visual[name][:200]
            out['last_tool']['obstacle_observation']=dict(
                source_observation_id=v.get('source_observation_id'),
                depth=v.get('depth'),visual=visual,
                visual_reused_for_identical_frame=v.get(
                    'visual_reused_for_identical_frame'),
                identity_is_visual_hypothesis_only=True,
                next_action_selected_by='navigation_agent')
        # Explicit model-requested depth inspection must actually reach the
        # next model decision, not be silently dropped by compaction.
        if t.get('tool')=='inspect_geometry':
            if t.get('geometry'):
                out['last_tool']['geometry']=fields(t['geometry'],(
                    'motion_state','depth_state','clearance',
                    'nearest_obstacle_distance','nearest_collision_point',
                    'unknown_fraction','rejection_reasons','human_occupancy',
                    'human_evidence','terrain_support'))
            if t.get('directions'):
                out['last_tool']['directions']=[
                    fields(d,('bearing_deg','status','reason','clearance_m',
                              'support')) for d in t['directions'][:7]]
            if t.get('alternatives'):
                out['last_tool']['alternatives']=[
                    fields(d,('bearing_deg','status','reason','clearance_m'))
                    for d in t['alternatives'][:5]]
    if state.get('latest_observation_task'):
        t=state['latest_observation_task']
        out['latest_observation_task']=dict(mode=t.get('mode'),
            subgoal_index=t.get('subgoal_index'),
            objective=short(t.get('objective',''),120),outcome=t.get('outcome'),
            candidate=t.get('candidate'),best_cue=t.get('best_cue'),
            best_approach=t.get('best_approach'),
            navigation_feedback=t.get('navigation_feedback'),
            observation_mode=t.get('observation_mode'),
            verification=t.get('verification'),
            approach_candidates=t.get('approach_candidates',[])[-3:],
            blocked_approaches=t.get('blocked_approaches',[])[-4:],
            visual_reports=compact_vlm_reports(t.get('visual_reports')),
            searched_headings_deg=t.get('searched_headings_deg',[])[-4:])
    if state.get('local_geometry'):out['local_geometry']=state['local_geometry'][:3]
    if state.get('historical_hint'):
        t=state['historical_hint']
        out['historical_hint']=dict(observation_id=t.get('observation_id'),
            description=short(t.get('description',''),150))
    return out
