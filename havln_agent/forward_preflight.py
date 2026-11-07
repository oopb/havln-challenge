"""One Depth preflight protocol shared by model context and atomic MOVE.

This is a measured, instantaneous corridor test, not a terrain / navigation
policy. UNKNOWN is not CLEAR. The same function is used before NavigationAgent
inference and immediately before each MOVE_FORWARD, on the relevant live
RGB-D frame after camera/heading adjustments.
"""
import math
from .safety import forward_probe


def sense_forward(geometry, source):
    probe=forward_probe(geometry)
    center=(geometry.central_forward_obstruction()
            if probe['state']=='BLOCKED' else None)
    confirmed=bool(center and center.get('confirmed'))
    state=('BLOCKED' if confirmed else
           'CLEAR' if probe['state']=='EXECUTABLE' else 'UNKNOWN')
    reason=('measured_centerline_obstacle' if confirmed else
            'noncentral_depth_obstacle_not_verified_as_forward_block'
            if probe['state']=='BLOCKED' else
            probe.get('reason') or 'no_confirmed_depth_obstacle')
    depth=probe.get('depth') or {}
    pose=source['pose']
    blocker=None
    if confirmed:
        blocker={key:center.get(key) for key in (
            'central_support_pixels','required_support_pixels',
            'nearest_along_m','nearest_depth_m','nearest_height_m',
            'source_pixel_xy','focus_point_xy_norm',
            'focus_region_xyxy_norm','reason','identity')}
    return dict(
        status=state,
        reason=reason,
        source_observation_id=source['observation_id'],
        rgb_hash=source.get('rgb_hash'),
        depth_hash=source.get('depth_hash'),
        heading_deg=round(math.degrees(pose['heading'])%360,2),
        pitch_deg=pose.get('pitch_deg'),
        motion_epoch=pose.get('motion_epoch'),
        viewpoint_generation=pose.get('viewpoint_generation'),
        local_position_xz_m_est=[pose.get('x'),pose.get('z')],
        raw_probe_state=probe['state'],
        raw_probe_reason=probe.get('reason'),
        center_confirmed=confirmed,
        blocker=blocker,
        clearance_m=depth.get('clearance'),
        required_forward_step_m=depth.get('required_forward_step'),
        path_state=depth.get('path_state'),
        rejection_reasons=depth.get('rejection_reasons',[]),
        unknown_sweep_pixels=depth.get('unknown_sweep_pixels',0),
        geometry_check='same_calibrated_depth_for_decision_and_atomic_move',
        current_view_only=True,
        clear_means_guaranteed_habitat_motion=False,
        historical_depth_authorizes_motion=False)
