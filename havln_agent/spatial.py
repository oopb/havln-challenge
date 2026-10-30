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
    same_view = all(source['pose'].get(k) == current['pose'].get(k) for k in ('x', 'z', 'heading', 'pitch_deg', 'motion_epoch'))
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
    """Shared odometry prefixes cancel only within one explicitly known chain."""
    if all('motion_error_total' in p and 'motion_epoch' in p for p in (source, pose)):
        if source['motion_epoch'] != pose['motion_epoch']:
            return float('inf')
        return abs(pose['motion_error_total'] - source['motion_error_total'])
    return source.get('position_uncertainty', 0.05) + pose.get('position_uncertainty', 0.05)


def relative_goal(grounding, pose):
    """Transform a visible point estimate; never turn it into certified terrain."""
    point, source = grounding['position'], grounding['source']['pose']
    region = 'target_point'
    entry = grounding.get('entry', {})
    if grounding['semantic']['stair_direction'] in ('up', 'down') and entry.get('status') == 'LOCALIZED' \
            and (point['status'] != 'LOCALIZED' or entry.get('depth_spread_metres', 99) <
                 point.get('depth_spread_metres', 99)):
        point, region = entry, 'stair_entry'
    if point['status'] != 'LOCALIZED':
        return dict(status='UNKNOWN', requires_reconfirmation=True, reasons=['unlocalized_goal'])
    h = source['heading']
    dx = source['x'] + point['x_right'] * math.cos(h) - point['z_forward'] * math.sin(h) - pose['x']
    dz = source['z'] + point['x_right'] * math.sin(h) + point['z_forward'] * math.cos(h) - pose['z']
    x = dx * math.cos(pose['heading']) + dz * math.sin(pose['heading'])
    z = -dx * math.sin(pose['heading']) + dz * math.cos(pose['heading'])
    error = relative_pose_error(source, pose)
    # An unknown motion interval invalidates the old anchor. Keep exported
    # uncertainty finite; the reason below prevents its use for execution.
    uncertainty = (error if math.isfinite(error) else source.get('position_uncertainty', .05)
                   + pose.get('position_uncertainty', .05)) + max(.05, point.get('depth_spread_metres', 0)/2)
    reasons = []
    if not math.isfinite(error):
        reasons.append('unknown_motion_since_anchor')
    if uncertainty > 0.5:
        reasons.append('spatial_uncertainty_too_large')
    if grounding['semantic']['confidence'] == 'low':
        reasons.append('low_semantic_confidence')
    return dict(status='UNCERTAIN' if reasons else 'ESTIMATED', x_right_m=x, z_forward_m=z,
                region=region,
                bearing_deg=math.degrees(math.atan2(-x, z)), uncertainty_m=uncertainty,
                uncertainty_kind='relative_odometry_chain_and_depth_spread_not_calibrated',
                heading_uncertainty='NOT_CALIBRATED',
                frame='current_robot_camera', source='rgb_depth_and_action_odometry',
                requires_reconfirmation=bool(reasons), reasons=reasons)


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

