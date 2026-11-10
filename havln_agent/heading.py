"""Canonical robot bearings and unambiguous model turn commands.

Habitat yaw uses LEFT as positive; this is an internal coordinate convention.
Navigation and observation models use explicit LEFT/RIGHT/NONE directions.
"""
import math


def signed_turn(direction, degrees):
    if direction not in ('LEFT','RIGHT','NONE'):
        raise ValueError('turn_direction_must_be_LEFT_RIGHT_or_NONE')
    if (isinstance(degrees,bool) or not isinstance(degrees,(int,float))
            or not math.isfinite(degrees) or not 0<=degrees<=180):
        raise ValueError('invalid_nonnegative_turn_degrees')
    if (direction=='NONE') != (degrees==0):
        raise ValueError('NONE_requires_zero_and_turn_requires_positive_degrees')
    return degrees if direction=='LEFT' else -degrees if direction=='RIGHT' else 0.


def relative_bearing(absolute_deg, current_deg):
    """Signed delta only for internal motion geometry, not model tool args."""
    return round((absolute_deg-current_deg+180)%360-180,2)


def direction_from_bearing(bearing_deg):
    if not isinstance(bearing_deg,(int,float)) or not math.isfinite(bearing_deg):
        raise ValueError('invalid_bearing')
    return ('LEFT' if bearing_deg>0 else 'RIGHT' if bearing_deg<0
            else 'NONE'),abs(bearing_deg)



def resolve_target_heading(target_heading_deg, current_heading_deg, step_deg):
    """Map a model-chosen ABSOLUTE yaw onto the shortest calibrated turn.

    The model owns the heading choice. Only the executor-side arithmetic
    chooses LEFT/RIGHT/NONE and rounds to legal Habitat turn increments.
    This is neither semantic route validation nor collision authorization.
    """
    for name, value in (('target_heading_deg', target_heading_deg),
                        ('current_heading_deg', current_heading_deg),
                        ('turn_step_deg', step_deg)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError('invalid_' + name)
    if not 0 <= target_heading_deg <= 360:
        raise ValueError('target_heading_deg_out_of_range')
    if not 0 < step_deg <= 180:
        raise ValueError('invalid_turn_step_deg')
    relative = relative_bearing(target_heading_deg % 360,
                                current_heading_deg % 360)
    count = math.floor(abs(relative) / step_deg + 0.5 + 1e-9)
    # Round to a legal number of atomic TURNs; never request >180 degrees.
    count = min(count, math.floor(180 / step_deg + 1e-9))
    degrees = round(count * step_deg, 8)
    if not count:
        return 'NONE', 0.
    return ('LEFT' if relative > 0 else 'RIGHT'), degrees

def box_center_bearing(box, horizontal_fov_deg):
    """Calibrated camera-horizontal ray to target box center (positive LEFT).

    A box is a *visual* target hypothesis; this is not a traversable direction
    or a metric way-point. No box means no measured target bearing.
    """
    if box is None:
        return None
    if (not isinstance(box,(list,tuple)) or len(box)!=4
            or any(not isinstance(v,(int,float)) or not math.isfinite(v)
                   for v in box)
            or not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1)
            or not 0<horizontal_fov_deg<180):
        raise ValueError('invalid_visual_box_or_camera_fov')
    u=(box[0]+box[2])/2
    return round(-math.degrees(math.atan(
        (2*u-1)*math.tan(math.radians(horizontal_fov_deg)/2))),2)


def same_local_stance(source_pose, current_pose, tolerance_m=.08):
    """Conservative stance identity, independent of metric VO when disabled.

    When poses explicitly lack metric coordinates, an identical *executed
    forward action generation* is the only same-stance evidence. A successful
    TURN or LOOK does not advance that generation. Any executed forward
    advances it, even if the RGB-D does not change; this errs toward not
    reusing stale historical views. Legacy metric fixtures with valid
    coordinates retain their old numeric distance comparison.
    """
    if not isinstance(source_pose,dict) or not isinstance(current_pose,dict):
        return False
    for name in ('motion_epoch','viewpoint_generation'):
        if source_pose.get(name)!=current_pose.get(name):
            return False
    invalid=(source_pose.get('position_coordinates_valid') is False or
             current_pose.get('position_coordinates_valid') is False)
    if invalid:
        # Never interpret two missing generations as the same location.
        return all(name in source_pose and name in current_pose
                   for name in ('motion_epoch','viewpoint_generation'))
    try:
        return math.hypot(source_pose['x']-current_pose['x'],
                          source_pose['z']-current_pose['z'])<=tolerance_m
    except (KeyError,TypeError,ValueError):
        return False
