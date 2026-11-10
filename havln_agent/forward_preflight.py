"""One Depth preflight protocol shared by model context and atomic MOVE.

This is a measured, instantaneous corridor test, not a terrain / navigation
policy. UNKNOWN is not CLEAR. The same function is used before NavigationAgent
inference and immediately before each MOVE_FORWARD, on the relevant live
RGB-D frame after camera/heading adjustments.
"""
import math
from .safety import forward_probe


def sense_forward(geometry, source, human_margin_m=.45):
    probe=forward_probe(geometry)
    center=(geometry.central_forward_obstruction()
            if probe['state']=='BLOCKED' else None)
    confirmed=bool(center and center.get('confirmed'))
    # Only SAME-SOURCE RGB person regions + measured Depth may override UNKNOWN.
    # Inconclusive side geometry alone never becomes a blanket veto.
    person=(geometry.human_forward_clearance(buffer_m=human_margin_m)
            if callable(getattr(type(geometry),'human_forward_clearance',None))
            else dict(status='NO_CURRENT_PERSON_REGIONS',
                      human_buffer_confirmed=False))
    human_confirmed=bool(person.get('human_buffer_confirmed'))
    near=(geometry.near_field_proximity()
          if callable(getattr(type(geometry),'near_field_proximity',None))
          else dict(status='NO_MEASURED_NEAR_OCCUPANCY'))
    state=('BLOCKED' if confirmed or human_confirmed else
           'CLEAR' if probe['state']=='EXECUTABLE' else 'UNKNOWN')
    reason=('measured_current_RGB_person_buffer_intrusion' if human_confirmed
            else 'measured_centerline_obstacle' if confirmed else
            'noncentral_depth_obstacle_not_verified_as_forward_block'
            if probe['state']=='BLOCKED' else
            probe.get('reason') or 'no_confirmed_depth_obstacle')
    depth=probe.get('depth') or {}
    pose=source['pose']
    blocker=None
    if human_confirmed:
        blocker=dict(
            blocker_kind='current_RGB_person_depth_buffer',
            central_support_pixels=0,
            support_pixels=person.get('buffer_support_pixels',0),
            required_support_pixels=person.get('required_support_pixels',0),
            nearest_along_m=person.get('nearest_along_m'),
            nearest_depth_m=person.get('nearest_person_depth_m'),
            nearest_height_m=None,
            source_pixel_xy=person.get('source_pixel_xy'),
            focus_point_xy_norm=person.get('focus_point_xy_norm'),
            focus_region_xyxy_norm=person.get('focus_region_xyxy_norm'),
            reason=reason,
            identity='current_model_person_box_correlated_with_live_depth')
    elif confirmed:
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
        local_position_xz_m_est=(
            [pose.get('x'),pose.get('z')]
            if pose.get('position_coordinates_valid') is not False
            else None),
        stance_id=pose.get('stance_id'),
        metric_pose_available=pose.get('position_coordinates_valid',True),
        raw_probe_state=probe['state'],
        raw_probe_reason=probe.get('reason'),
        center_confirmed=confirmed,
        human_buffer_confirmed=human_confirmed,
        human_forward_clearance=person,
        near_field_proximity=near,
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
