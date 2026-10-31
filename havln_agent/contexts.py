"""Decision-sized projections; raw records and RGB-D stay in Memory Service."""
import math
import time

from .progress import turn_evidence, current_visual_evidence
from .memory_records import task_tokens
from .local_exploration import last_block, local_coverage


def fields(value,names):
    return {key:value[key] for key in names if key in value}



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


def tool_summary(result):
    value=fields(result,('tool','status','reason','suggestion','missing_evidence','recoverable','interrupted_status',
        'requested_actions','executed_actions','executed_motion','end_observation_id',
        'remaining_distance','endpoint_estimate_valid','reach_basis',
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
            'rejection_reasons','terrain_support','human_evidence'))
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
            checked_views=[fields(p,('observation_id','heading_deg','pitch_deg',
                'evidence_origin')) for p in result.get('checkpoints',[])],
            note='visual_candidates_are_hypotheses; live_geometry_required_for_motion')
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
    return dict(current_subgoal=fields(task.subgoals[index],('action','target','landmark','success_criteria')),
        verified_milestones=[dict(subgoal=m['subgoal'],source_observation_ids=m.get('observation_ids',[])[-2:],
            executed_action_count=len(m.get('executed_steps',[])),supported=m['verification']['supported']) for m in task.completion_evidence[-3:]],
        new_evidence=runtime.memory.recall_observation(runtime.memory.current_place_id,
            landmark=task.subgoals[index]['target'],top_k=3)['observations'],
        review_reason=runtime.review_reason,
        actions_remaining=runtime.config['max_steps']-len(runtime.memory.actions))


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
    last=memory.tool_ledger[-1]['result'] if memory.tool_ledger else None
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
    result=dict(goal=runtime.active_subgoal,
        grounding_evidence=grounding_hints,
        navigation_mode=getattr(runtime, 'navigation_mode', 'OBSERVE'),
        navigation_memory=navigation_memory,
        pose=dict(place_id=memory.current_place_id,heading_deg=math.degrees(pose['heading']),
            pitch_deg=runtime.pitch_deg,position_xz_m_est=[pose['x'],pose['z']],
            height_y_m_est=pose['y'],
            confidence=('estimated_relative' if pose.get('global_chain_continuous',True)
                        else 'local_relative_only' if pose.get('local_anchor_observation_id')
                        else 'unknown_after_unregistered_motion'),
            position_uncertainty_m=pose['position_uncertainty'],
            motion_epoch=pose.get('motion_epoch',0),
            coordinate_scope=pose.get('coordinate_scope','episode_relative'),
            global_chain_continuous=pose.get('global_chain_continuous',True),
            local_anchor_observation_id=pose.get('local_anchor_observation_id')),
        current_view=view,legal_look=[look for look,delta in [('up',runtime.camera.tilt_angle),('down',-runtime.camera.tilt_angle)]
            if abs(runtime.pitch_deg+delta)<=60],
        local_geometry=[fields(d,('angle_degrees','motion_state','reason','clearance')) for d in
            sorted(memory.current.get('directions',[]),key=lambda d:abs(d['angle_degrees']))[:7]],
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
        if path.get('first_step')!='EXECUTABLE':
            continue
        recent.append(waypoint_summary(path))
        if len(recent)==9:break
    if recent:
        result['candidate_waypoints']=recent
        result['candidate_note']='old_preview_only; navigate_local rechecks current RGB-D before each movement'
    result['motion_tools_note']='navigate_relative and navigate_to_pixel do not require a pre-generated waypoint; all forward actions use live collision probes'
    if runtime.active_waypoint_id and runtime.motion_controller.target is not None:
        target=runtime.motion_controller.target
        attempt=memory.active_attempt or {}
        unverified=bool(attempt.get('unknown_motion_seen'))
        result['local_navigation']=dict(waypoint_id=runtime.active_waypoint_id,
            status='IN_PROGRESS',next_action=(
                'MOVE_FORWARD_PENDING_LIVE_SAFETY' if unverified else
                runtime.motion_controller.next_action()),
            remaining_distance_m=(None if unverified else
                round(math.hypot(target[0]-pose['x'],target[1]-pose['z']),3)),
            endpoint_estimate_valid=not unverified,
            relative_motion_only=unverified,
            forward_actions=attempt.get('forward_actions',0),
            forward_action_horizon=attempt.get('forward_action_horizon'),
            suggestion=('continue_same_bounded_waypoint_with_fresh_rgbd_checks'
                if unverified else
                'prefer_continuing_same_waypoint_if_still_relevant; each forward step is rechecked'))
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
    result['local_exploration']=local_coverage(memory,pose,runtime.camera.turn_angle,
        round(360/runtime.camera.turn_angle))
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
    if memory.observation_searches:
        recent_search=memory.observation_searches[-1]
        result['latest_observation_task']=dict(
            objective=recent_search['objective'],outcome=recent_search['outcome'],
            direction=recent_search['direction'],
            covered_turn_deg=recent_search.get('covered_turn_deg',0),
            candidate=recent_search.get('candidate'),
            searched_headings_deg=[p['heading_deg'] for p in recent_search['checkpoints']][-5:])
    if historical_hint:result['historical_hint']=historical_hint
    liveness=getattr(runtime,'navigation_liveness',None) or {}
    if liveness.get('consecutive_unchanged_decisions',0)>=2:
        result['decision_liveness']=fields(liveness,(
            'consecutive_unchanged_decisions',
            'max_unchanged_evidence_decisions','review_requested','reason'))
    return result
