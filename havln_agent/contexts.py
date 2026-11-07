"""Bounded, source-aware context for whole-mission navigation decisions.

No milestones, Python progress predicates, or completion verification gates.
"""
import copy
import math

def fields(value,names):
    return {k:copy.deepcopy(value[k]) for k in names if isinstance(value,dict) and k in value}

def navigation_context(runtime):
    m=runtime.memory
    task=m.task
    pose=runtime.motion_controller.pose()
    view=m.current or {}
    last=m.tool_ledger[-1] if m.tool_ledger else {}
    last_result=copy.deepcopy(last.get('result') or {})
    recent_motion=[]
    for a in m.actions[-24:]:
        if a.get('action')!='MOVE_FORWARD':
            continue
        motion=(a.get('result') or {}).get('rgbd_motion') or {}
        displacement=motion.get('translation_m') or {}
        recent_motion.append(dict(step=a['step'],
            heading_deg=round(math.degrees(a.get('pose',{}).get('heading',0))%360,1),
            registration=motion.get('status','UNKNOWN'),
            vertical_change_m_est=displacement.get('y_up') if motion.get('status')=='ESTIMATED' else None,
            uncertainty_m=motion.get('uncertainty_m') if motion.get('status')=='ESTIMATED' else None,
            observed_frame_change=motion.get('observed_frame_change'),
            observation_id=(a.get('result') or {}).get('observation_after')))
    observations=[]
    for task_result in m.observation_searches[-2:]:
        if task_result.get('mode')!='mission_vlm':
            continue
        observations.append(fields(task_result,('objective','observation_summary',
            'navigation_suggestion','waypoints','outcome','evidence_observation_ids',
            'end_observation_id')))
    last_tool=fields(last_result,('tool','status','reason','interrupted_status',
        'executed_actions','end_observation_id','movement_intent_feedback',
        'observation_summary','navigation_suggestion','waypoints','blocked_at',
        'vertical_translation_m_est','vertical_motion'))
    if last_result.get('obstacle_observation'):
        last_tool['obstacle_observation']=fields(last_result['obstacle_observation'],
            ('source_observation_id','visual','depth'))
    return dict(
        instruction=task.instruction,
        progress_narrative=task.progress_narrative,
        navigation_intent=task.navigation_intent,
        uncertainties=copy.deepcopy(task.uncertainties[-6:]),
        progress_history=copy.deepcopy(task.progress_history[-3:]),
        current_view=dict(observation_id=view.get('observation_id'),
            rgb_hash=view.get('rgb_hash'),depth_hash=view.get('depth_hash'),
            place_id=view.get('place_id'),new_visual_information=view.get('new_visual_information')),
        pose=dict(heading_deg=round(math.degrees(pose.get('heading',0))%360,2),
            pitch_deg=pose.get('pitch_deg'),position_xz_m_est=[pose.get('x'),pose.get('z')],
            position_coordinates_valid=pose.get('position_coordinates_valid',True),
            motion_epoch=pose.get('motion_epoch'),
            position_uncertainty_m=pose.get('position_uncertainty')),
        action_calibration=dict(forward_step_m=runtime.camera.forward_step,
            turn_step_deg=runtime.camera.turn_angle,direct_segment_max_m=3.0),
        actions_remaining=max(0,runtime.config['max_steps']-len(m.actions)-1),
        recent_motion=recent_motion[-8:],
        route_history=m.route_progress_summary(),
        recent_observation_tasks=observations,
        recent_blocked_directions=copy.deepcopy(m.evidence.obstacle_blocks[-4:]),
        last_tool=last_tool,
        evidence_refs=copy.deepcopy(task.evidence_observation_ids[-12:]),
        navigation_mode=getattr(runtime,'navigation_mode','OBSERVE'),
        decision_liveness=copy.deepcopy(getattr(runtime,'navigation_liveness',None)))

def compact_navigation_context(state):
    # Context is already deliberately bounded at its source.
    return copy.deepcopy(state)

def planner_context(runtime):
    # Deprecated TaskPlanner is not in the runtime; retain only a benign
    # import-compatible function for archived offline tooling.
    return dict(instruction=runtime.memory.task.instruction)
