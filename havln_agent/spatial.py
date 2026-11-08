"""Relative RGB-D pose estimates and source validity; no hidden environment state."""

import math
import re


def angle_delta(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def validity(source, current, now, max_age):
    if not source or not current:
        return dict(status='UNKNOWN', current_view=False, usable_for_navigation=False,
                    reasons=['missing_observation_source'])
    age = max(0, now - source['captured_at'])
    same_view = all(source['pose'].get(k) == current['pose'].get(k) for k in ('x', 'z', 'heading', 'pitch_deg', 'motion_epoch', 'viewpoint_generation'))
    same_image = source['rgb_hash'] == current['rgb_hash']
    if same_view and same_image and source.get('image_scope', 'current') == 'current':
        age = max(0, now - source.get('revalidated_at', source['captured_at']))
    reasons = []
    if source.get('image_scope', 'current') != 'current':
        reasons.append('historical_keyframe_not_live_evidence')
    if not same_view or not same_image:
        reasons.append('different_view_or_image')
    if age > max_age:
        reasons.append('age_limit_exceeded')
    return dict(status='STALE' if age > max_age else 'CURRENT' if not reasons else 'HISTORICAL',
                age_seconds=age, current_view=same_view and same_image,
                usable_for_navigation=not reasons, reasons=reasons)


def relative_pose_error(source, pose):
    """Metric odometry uncertainty is unavailable in action-stance mode.

    Relative object geometry can still be transformed by commanded yaw
    when no forward action occurred between the source and current view.
    """
    if (source.get('position_coordinates_valid') is False or
            pose.get('position_coordinates_valid') is False):
        if (source.get('motion_epoch')==pose.get('motion_epoch')
                and source.get('viewpoint_generation')==
                    pose.get('viewpoint_generation')
                and 'viewpoint_generation' in source
                and 'viewpoint_generation' in pose):
            return 0.  # no *relative translation* occurred in this stance
        return float('inf')
    if all(isinstance(p.get('motion_error_total'),(int,float))
           and 'motion_epoch' in p for p in (source,pose)):
        if (source['motion_epoch']!=pose['motion_epoch'] or
                source.get('unknown_forward_actions',0)!=
                    pose.get('unknown_forward_actions',0)):
            return float('inf')
        return abs(pose['motion_error_total']-source['motion_error_total'])
    return (source.get('position_uncertainty') or .05) + (
        pose.get('position_uncertainty') or .05)


def relative_goal(grounding, pose):
    """Project a grounded point at a SAME STANCE; no fake x/z delta.

    In action-only mode, camera yaw and source-local RGB-D target geometry
    are sufficient for a local candidate bearing if no MOVE_FORWARD was
    executed since grounding. Across generations the target MUST be
    re-grounded; stale object coordinates cannot authorize a new route.
    """
    point=grounding['position']
    source=grounding['source']['pose']
    entry=grounding.get('entry',{})
    region='target_point'
    if point['status']!='LOCALIZED':
        return dict(status='UNKNOWN',requires_reconfirmation=True,
                    reasons=['unlocalized_goal'])
    error=relative_pose_error(source,pose)
    if not math.isfinite(error):
        return dict(status='UNKNOWN',requires_reconfirmation=True,
                    reasons=['executed_forward_since_target_source'],
                    source_stance_reusable=False)
    hs, hc=source['heading'],pose['heading']
    if (source.get('position_coordinates_valid') is False or
            pose.get('position_coordinates_valid') is False):
        # Rotate target vector from its original RGB-D camera bearing into
        # the current commanded heading, with NO translation estimate.
        dx=(point['x_right']*math.cos(hs)
            - point['z_forward']*math.sin(hs))
        dz=(point['x_right']*math.sin(hs)
            + point['z_forward']*math.cos(hs))
        x=dx*math.cos(hc)+dz*math.sin(hc)
        z=-dx*math.sin(hc)+dz*math.cos(hc)
        uncertainty=max(.05,point.get('depth_spread_metres',0)/2)
        source_label='same_action_stance_RGBD_target_and_commanded_yaw'
    else:
        dx=(source['x']+point['x_right']*math.cos(hs)
            -point['z_forward']*math.sin(hs)-pose['x'])
        dz=(source['z']+point['x_right']*math.sin(hs)
            +point['z_forward']*math.cos(hs)-pose['z'])
        x=dx*math.cos(hc)+dz*math.sin(hc)
        z=-dx*math.sin(hc)+dz*math.cos(hc)
        uncertainty=error+max(.05,point.get('depth_spread_metres',0)/2)
        source_label='rgb_depth_and_valid_relative_odometry'
    reasons=[]
    if uncertainty>.5:
        reasons.append('spatial_uncertainty_too_large')
    if grounding['semantic']['confidence']=='low':
        reasons.append('low_semantic_confidence')
    return dict(
        status='UNCERTAIN' if reasons else 'ESTIMATED',
        x_right_m=x,z_forward_m=z,region=region,
        bearing_deg=math.degrees(math.atan2(-x,z)),
        uncertainty_m=uncertainty,
        uncertainty_kind='target_depth_spread_not_global_position',
        heading_uncertainty='NOT_CALIBRATED',
        frame='current_robot_camera',
        source=source_label,
        metric_robot_position_known=False
            if source_label.startswith('same_action') else True,
        requires_reconfirmation=bool(reasons),reasons=reasons)


def goal_estimate(grounding, memory, pose, now, max_age, static_max_age=300, human_max_age=2):
    estimate = relative_goal(grounding, pose)
    if estimate['status'] == 'UNKNOWN':
        return estimate
    if (re.search(r'\b(person|human|dancer|man|woman|people|someone|individual|pedestrian)\b|舞者|行人',
                  grounding['target'], re.I) or
            any(c['kind'] == 'person' for c in grounding['semantic'].get('components', []))) \
            and now - grounding['source']['captured_at'] > human_max_age:
        estimate['reasons'].append('dynamic_landmark_requires_current_confirmation')
    if now - grounding['source']['captured_at'] > static_max_age:
        estimate['reasons'].append('static_landmark_age_requires_reconfirmation')
    index = next((i for i, g in enumerate(memory.groundings) if g is grounding or
                  g.get('grounding_id') == grounding.get('grounding_id')), -1)
    later = [g for i, g in enumerate(memory.groundings) if g['target'].lower() == grounding['target'].lower()
             and i > index and g['source']['captured_at'] >= grounding['source']['captured_at']]
    if later:
        latest = relative_goal(later[-1], pose)
        direction_conflict = (grounding['semantic']['stair_direction'] in ('up', 'down') and
                              later[-1]['semantic']['stair_direction'] in ('up', 'down') and
                              grounding['semantic']['stair_direction'] != later[-1]['semantic']['stair_direction'])
        if latest['status'] == 'UNKNOWN' or direction_conflict or math.hypot(
                latest['x_right_m'] - estimate['x_right_m'],
                latest['z_forward_m'] - estimate['z_forward_m']) > max(
                    0.5, latest.get('uncertainty_m', 0) + estimate['uncertainty_m']):
            estimate['reasons'].append('newer_goal_evidence_conflicts')
    estimate['requires_reconfirmation'] = bool(estimate['reasons'])
    if estimate['requires_reconfirmation']:
        estimate['status'] = 'UNCERTAIN'
    return estimate

