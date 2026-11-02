"""Read-only navigation evidence summaries; NEVER certify semantic completion.

Semantic milestone verification lives exclusively in verification_agent.py.
These helpers only classify physical motion policy and summarize recorded facts.
"""
import math
import re

from .motion import registered_action


def stair_motion_direction(goal):
    """Identify explicit requested vertical traversal for local terrain safety."""
    action=str(goal.get('action',''))
    if re.search(r'\b(?:descend\w*|descent|downstairs)\b|\b(?:go|walk|move|head)\s+down\s+(?:the\s+)?stairs\b|下楼|下台阶',action,re.I):
        return 'DOWN'
    if re.search(r'\b(?:ascend\w*|ascent|upstairs)\b|\b(?:go|walk|move|head)\s+up\s+(?:the\s+)?stairs\b|上楼|上台阶',action,re.I):
        return 'UP'
    return None


def turn_requirement(goal):
    """Optional directional hint for navigation; NOT a completion predicate."""
    text=str(goal.get('action','')).lower()
    text=re.sub(r'\brightward\b','right',text)
    text=re.sub(r'\bleftward\b','left',text)
    if not re.search(r'\b(turn\w*|rotat\w*|orient\w*|reorient\w*|face|facing|heading)\b|转身|掉头|左转|右转|调整朝向',text):
        return None
    if re.search(r'\b(walk\w*|proceed\w*|enter\w*|exit\w*|pass\w*|continue\w*|descend\w*|ascend\w*|climb\w*|advance\w*|go|move\w*|follow\w*)\b|然后.*走',text):
        return None
    around=bool(re.search(r'\baround\b|opposite|reverse|转身|掉头',text))
    explicit=re.search(r'(\d+(?:\.\d+)?)\s*(?:degrees?|°|度)',text)
    if not around and not explicit and not re.search(r'\bleft\b|\bright\b|左|右',text):
        return None
    angle=float(explicit.group(1)) if explicit else 180 if around else 90
    if not 0<angle<=180:return None
    direction='right' if re.search(r'\bright\b|右',text) else 'left' if re.search(r'\bleft\b|左',text) else 'either'
    return dict(angle_deg=angle,direction=direction)


def turn_evidence(memory):
    """Estimated relative turn feedback, explicitly never ground-truth yaw."""
    task=memory.task
    index=task.current_subgoal_index
    goal=task.subgoals[index] if index<len(task.subgoals) else {}
    requirement=turn_requirement(goal)
    actions=[a for a in memory.actions if a.get('step',-1)>=task.subgoal_started
        and a.get('subgoal',index)==index]
    signed=sum((1 if a['action']=='TURN_LEFT' else -1 if a['action']=='TURN_RIGHT' else 0)
        *memory.camera.turn_angle for a in actions)
    degrees=(signed+180)%360-180
    complete=False
    remaining,directions=None,[]
    if requirement:
        expected=requirement['angle_deg']
        desired=-expected if requirement['direction']=='right' else expected
        delta=(desired-degrees+180)%360-180
        remaining=abs(delta)
        directions=['left','right'] if math.isclose(remaining,180) else [
            'left' if delta>0 else 'right']
        error=(min(abs(degrees-expected),abs(degrees+expected))
            if requirement['direction']=='either' else
            abs((degrees-(expected if requirement['direction']=='left' else -expected)+180)%360-180))
        complete=error<memory.camera.turn_angle/2 and any(
            a['action'].startswith('TURN_') and a['result'].get('visual_change')
            for a in actions)
    return dict(requirement=requirement,net_turn_deg=degrees,completed=complete,
        remaining_deg=remaining,allowed_directions=directions,
        turn_steps=[a['step'] for a in actions if a['action'].startswith('TURN_')],
        source='accepted_turn_receipts_and_public_view_change',
        confidence='estimated_not_gt_yaw')


def current_visual_evidence(source,memory,now,max_age):
    """Check an unmodified public RGB-D view is still current."""
    current=memory.current or {}
    return bool(source.get('image_scope','current')=='current'
        and source.get('pose')==current.get('pose')
        and source.get('rgb_hash')==current.get('rgb_hash')
        and source.get('depth_hash')==current.get('depth_hash')
        and now-current.get('last_read_at',float('-inf'))<=max_age)


def registered_motion_history(memory,subgoal=None):
    """List authentic local RGB-D forward receipts without aggregating poses.

    Preserve ALL receipts explicitly attributed to this milestone across
    provisional revisits; a later task selection cannot erase earlier facts.
    Old coordinates across different motion epochs are never composable.
    """
    index=memory.task.current_subgoal_index if subgoal is None else subgoal
    observed={o['observation_id']:o for o in getattr(memory,'observations',[])}
    rows=[]
    for action in memory.actions:
        if (action.get('action')!='MOVE_FORWARD' or action.get('subgoal',index)!=index
                or not isinstance(action.get('step'),int)):
            continue
        result=action.get('result') or {}
        motion=result.get('rgbd_motion') or {}
        valid=registered_action(action,observed)
        translation=motion.get('translation_m') or {}
        row=dict(step=action['step'],motion_epoch=action['pose'].get('motion_epoch',0),
            status='ESTIMATED' if valid else 'UNKNOWN',
            vertical_motion=motion.get('vertical_motion','UNKNOWN') if valid else 'UNKNOWN',
            delta_y_m=translation.get('y_up') if valid else None,
            uncertainty_m=motion.get('uncertainty_m') if valid else None,
            observation_before=result.get('observation_before'),
            observation_after=result.get('observation_after'),
            unknown_reason=motion.get('reason') if not valid else None)
        rows.append(row)
    down=[r for r in rows if r['status']=='ESTIMATED'
          and r['delta_y_m'] is not None and r['uncertainty_m'] is not None
          and r['delta_y_m'] < -r['uncertainty_m']]
    up=[r for r in rows if r['status']=='ESTIMATED'
        and r['delta_y_m'] is not None and r['uncertainty_m'] is not None
        and r['delta_y_m'] > r['uncertainty_m']]
    return dict(recorded_forward_steps=len(rows),
        registered_forward_steps=sum(r['status']=='ESTIMATED' for r in rows),
        unknown_forward_steps=[r['step'] for r in rows if r['status']=='UNKNOWN'],
        registered_down_steps=[r['step'] for r in down],
        registered_up_steps=[r['step'] for r in up],
        observed_downward_m_est=round(-sum(r['delta_y_m'] for r in down),4),
        observed_upward_m_est=round(sum(r['delta_y_m'] for r in up),4),
        motion_epochs=sorted({r['motion_epoch'] for r in rows}),
        rows=rows,
        caveat='separate_local_measurements_not_verified_route_or_global_pose')


def stair_replay_hint(memory,subgoal=None):
    """Read-only stair history discovery, not a task-specific success test."""
    index=memory.task.current_subgoal_index if subgoal is None else subgoal
    if not 0<=index<len(memory.task.subgoals):
        return None
    direction=stair_motion_direction(memory.task.subgoals[index])
    if direction is None:return None
    hist=registered_motion_history(memory,index)
    relevant=hist['registered_down_steps' if direction=='DOWN' else 'registered_up_steps']
    observed={o['observation_id']:o for o in memory.observations}
    first=min(relevant) if relevant else None
    scenes=[]
    for item in getattr(memory,'scene_history',[]):
        source=item.get('source') or {}
        obs=observed.get(source.get('observation_id'))
        if (not obs or source.get('subgoal')!=index or
                first is None or obs['step']>first):
            continue
        landmarks=[str(l.get('name','')) for l in
                   (item.get('scene') or {}).get('landmarks',[])]
        if not any(re.search(r'\b(stair|stairs|staircase|steps|treads)\b',v,re.I)
                   for v in landmarks):continue
        # A scene can explain a measured segment's setting only when the
        # camera has not translated between that frame and the segment.
        if any(a.get('action')=='MOVE_FORWARD' and obs['step']<=a['step']<first
               for a in memory.actions):continue
        scenes.append(obs)
    earlier=max(scenes,key=lambda o:o['step']) if scenes else None
    grounded=[g for g in memory.groundings if g.get('source',{}).get('subgoal')==index
        and g.get('source',{}).get('observation_id') in observed
        and g.get('status') in ('LOCALIZED','PARTIAL')]
    landing=[]
    for g in grounded:
        source=g['source']
        obs=observed[source['observation_id']]
        if (first is None or obs['step']<=first or
            (g.get('landing_geometry') or {}).get('status')!='SUPPORTED'
            or (g.get('semantic') or {}).get('landing_box') is None
            or source.get('image_scope')!='current'):
            continue
        if all(source.get(k) and source[k]==obs.get(k)
               for k in ('rgb_hash','depth_hash','pose')):
            landing.append(source['observation_id'])
    return dict(subgoal=index,stair_direction=direction,
        registered_vertical_steps=relevant[-8:],
        historical_motion_m_est=(hist['observed_downward_m_est'] if direction=='DOWN'
                                 else hist['observed_upward_m_est']),
        preceding_scene_observation_id=(earlier['observation_id'] if earlier else None),
        landmark_grounding_observation_ids=[
            g['source']['observation_id'] for g in grounded][-3:],
        captured_landing_observation_ids=landing[-3:],
        unknown_forward_steps=hist['unknown_forward_steps'][-8:],
        proof_status='HISTORICAL_COMPONENTS_NOT_COMPLETION',
        note='historical factual hints; semantic success only through VerificationAgent')
