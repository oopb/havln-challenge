"""Factual progress evidence from public sensor and action receipts.

No named-stage completion predicates, special stair rules or VERIFY handlers.
The navigation model alone interprets these measurements as overall mission
progress; the official challenge scorer owns objective success measurement.
"""
import math
import time

def current_visual_evidence(source,memory,now,max_age):
    if not source or source.get('image_scope','current')!='current':
        return False
    current=memory.current or {}
    return bool(current) and all(source.get(k)==current.get(k)
        for k in ('pose','rgb_hash','depth_hash')) and (
            0<=now-current.get('last_read_at',-math.inf)<=max_age)

def registered_motion_history(memory):
    observations={o['observation_id'] for o in memory.observations}
    rows=[]
    for action in memory.actions:
        if action.get('action')!='MOVE_FORWARD':continue
        feedback=action.get('result') or {}
        motion=feedback.get('rgbd_motion') or {}
        valid=(motion.get('status')=='ESTIMATED'
            and feedback.get('observation_before') in observations
            and feedback.get('observation_after') in observations)
        vector=motion.get('translation_m') or {}
        rows.append(dict(step=action.get('step'),registered=valid,
            observed_vertical_change_m_est=vector.get('y_up') if valid else None,
            uncertainty_m=motion.get('uncertainty_m') if valid else None,
            observation_id=feedback.get('observation_after')))
    registered=[v for v in rows if v['registered']]
    return dict(recorded_forward_steps=len(rows),
        registered_forward_steps=len(registered),
        observed_downward_m_est=sum(max(0,-v.get('observed_vertical_change_m_est',0))
                                    for v in registered),
        observed_upward_m_est=sum(max(0,v.get('observed_vertical_change_m_est',0))
                                  for v in registered),
        recent_motion=rows[-12:],
        measurement_scope='uncertain_public_RGBD_not_ground_truth')
