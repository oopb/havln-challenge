"""Decision-sized projections; raw records and RGB-D stay in Memory Service."""
import math
import time

from .progress import turn_evidence, current_visual_evidence
from .memory_records import task_tokens


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
    terrain=safety.get('terrain_support') or depth.get('terrain_support') or {}
    if not isinstance(terrain,dict):terrain=dict(status=str(terrain))
    values=fields(depth,('path_state','clearance','required_forward_step',
        'nearest_obstacle_distance','nearest_collision_travel','rejection_reasons',
        'unknown_sweep_pixels','collision_points','nearest_collision_point'))
    values['people']=fields(people,('confirmed','uncertain','unlocalized',
        'corridor_points','nearest_distance'))
    values['terrain']=fields(terrain,('status','reason','direction'))
    if values.get('path_state')=='OCCUPIED' and not values['people'].get('corridor_points'):
        values['identity_note']='measured_depth_obstacle; semantic identity unverified; do not assume human'
    return values


def tool_summary(result):
    value=fields(result,('tool','status','reason','suggestion','missing_evidence','recoverable','interrupted_status',
        'requested_actions','executed_actions','executed_motion','end_observation_id',
        'remaining_distance','vertical_motion','vertical_translation_m_est','semantic_direction_mismatch','waypoint_type','waypoint_arrival_completes_task','completion_confirmed','criteria_verdicts','missing','supported','information_options'))
    if result.get('status') in ('BLOCKED','NEEDS_VIEW','REQUIRES_ATTENTION') or result.get('safety'):
        collision=collision_summary(result)
        if collision:value['measured_collision']=collision
    if 'paths' in result:
        value['paths']=[fields(p,('id','type','bearing_deg','range_m','first_step','goal_status',
            'source_observation_id','objective','reason','first_atomic_action',
            'requires_post_turn_recheck')) for p in result['paths']]
    if 'semantic' in result:
        value['grounding']=dict(fields(result,('grounding_id','target','localization','entry_status',
            'stair_direction','traversable_access')),uncertainty=result['semantic']['uncertainty'])
    if 'assessment' in result:value['assessment']=result['assessment']
    if 'memory' in result:value['memory']=result['memory']
    if 'geometry' in result:
        value['geometry']=fields(result['geometry']['front'],('motion_state','depth_state','clearance',
            'rejection_reasons','terrain_support','human_evidence'))
    if 'geometry' in value and 'terrain_support' in value['geometry']:
        value['geometry']['terrain_support']=fields(value['geometry']['terrain_support'],
            ('status','reason','direction','initial_height_m','end_height_m','max_step_height_m','scope'))
    if 'directions' in result:
        value['directions']=result['directions']
    if 'alternatives' in result:value['alternatives']=result['alternatives']
    if 'scan' in result:
        value['observations']=result['scan']['frames']
    if result.get('motion_alert'):
        value['motion_alert']=fields(result['motion_alert'],('reason','observation_before','observation_after'))
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
    result=dict(goal=runtime.active_subgoal,
        pose=dict(place_id=memory.current_place_id,heading_deg=math.degrees(pose['heading']),
            pitch_deg=runtime.pitch_deg,position_xz_m_est=[pose['x'],pose['z']],
            height_y_m_est=pose['y'],confidence='estimated'),
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
        recent.append(fields(path,('id','type','bearing_deg','range_m',
            'first_atomic_action','requires_post_turn_recheck','execution_status')))
        if len(recent)==3:break
    if recent:
        result['candidate_waypoints']=recent
        result['candidate_note']='old_preview_only; navigate_local rechecks current RGB-D before each movement'
    if runtime.active_waypoint_id and runtime.motion_controller.target is not None:
        target=runtime.motion_controller.target
        result['local_navigation']=dict(waypoint_id=runtime.active_waypoint_id,
            status='IN_PROGRESS',next_action=runtime.motion_controller.next_action(),
            remaining_distance_m=round(math.hypot(target[0]-pose['x'],target[1]-pose['z']),3),
            suggestion='prefer_continuing_same_waypoint_if_still_relevant; each forward step is rechecked')
    # The latest tool is often a harmless scan. Keep the most recent physical
    # navigation failure available until a new route is tested.
    for item in reversed(memory.tool_ledger[-40:]):
        prior=item['result']
        if item['call'].get('tool')=='navigate_local' and prior.get('status')=='BLOCKED':
            result['recent_navigation_block']=dict(tool_call_id=item['call_id'],
                observation_id=prior.get('end_observation_id'),
                **fields(tool_summary(prior),('reason','measured_collision')))
            break
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
    if historical_hint:result['historical_hint']=historical_hint
    return result
