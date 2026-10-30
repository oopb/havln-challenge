"""Observed short corridors; no task policy or failure blacklist."""

import math
from .progress import stair_motion_direction


def goal_relation(candidate, grounding, estimate, language_goal, turn_angle):
    """Describe goal relevance without choosing a route or certifying terrain."""
    reasons = []
    stairs = grounding['semantic']['stair_direction'] != 'not_stairs'
    wanted = stair_motion_direction(language_goal)
    direction = grounding.get('stair_direction', 'UNKNOWN')
    if wanted and direction not in (wanted,'UNKNOWN'):
        reasons.append('stair_direction_unknown' if direction == 'UNKNOWN' else 'stair_direction_opposes_language_goal')
    error = abs(math.degrees(candidate['angle']) - estimate['bearing_deg'])
    x, z = estimate['x_right_m'], estimate['z_forward_m']
    end_x, end_z = -math.sin(candidate['angle']) * candidate['distance'], math.cos(candidate['angle']) * candidate['distance']
    before, after = math.hypot(x, z), math.hypot(x - end_x, z - end_z)
    if error > turn_angle:
        reasons.append('bearing_not_toward_goal')
    if before - after <= 0:
        reasons.append('candidate_does_not_reduce_goal_distance')
    return dict(grounding_id=grounding['grounding_id'], target=grounding['target'],
                bearing_error_degrees=error, distance_before_m=before, distance_after_m=after,
                relative_pose_uncertainty_m=estimate['uncertainty_m'],
                relation='possibly_approaches_entry' if stairs else 'possibly_approaches_visible_target',
                direction_compatibility=('CONFIRMED' if direction == wanted else 'UNKNOWN') if wanted else 'NOT_REQUIRED',
                terrain_support='not_certified', rejection_reasons=reasons,
                relevant=not reasons, vertical_motion_verified=False)


def generate_waypoints(geometry, max_distance, diagnostics=None, angles=None):
    # Only offer directions reachable by the actual discrete TURN action.
    # A 7.5-degree candidate with a 15-degree robot TURN was falsely labelled
    # EXECUTABLE even though its first MOVE_FORWARD could be along 0 degrees.
    stride = geometry.camera.turn_angle
    extent = geometry.camera.hfov / 2
    requested = (angles if angles is not None else
                 [i * stride for i in range(-int(extent / stride), int(extent / stride) + 1)])
    angles = sorted(set(round(float(deg) / stride) * stride for deg in requested
                        if abs(round(float(deg) / stride) * stride) < extent))
    candidates = []
    for degrees in angles:
        angle = math.radians(degrees)
        check = geometry.assess_step(angle)
        diagnostic = dict(check['depth'],motion_state=check['state'],reason=check['reason'],terrain_support=check['support'])
        # Offer up to two short steps; only the FIRST step is certified here.
        # Every later MOVE_FORWARD must use a newly acquired RGB-D frame.
        distance = min(max_distance, 2 * geometry.camera.forward_step)
        diagnostic.update(waypoint_id='wp_%g' % degrees, waypoint_distance=distance,
                          first_atomic_action=('MOVE_FORWARD' if degrees == 0 else
                              'TURN_LEFT' if degrees > 0 else 'TURN_RIGHT'),
                          requires_post_turn_recheck=(degrees != 0))
        if diagnostics is not None:
            diagnostics.append(diagnostic)
        if check['state'] in ('EXECUTABLE','NEEDS_VIEW'):
            candidates.append(
                dict(
                    id=diagnostic['waypoint_id'],
                    angle=angle,
                    distance=distance,
                    first_step=check['state'],reason=check['reason'],
                    first_atomic_action=diagnostic['first_atomic_action'],
                    requires_post_turn_recheck=(degrees != 0),
                    clearance=diagnostic['clearance'],
                )
            )
    return candidates
