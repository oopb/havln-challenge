"""Local observation coverage; history is for exploration, never motion authorization.

The key is an estimated physical stance (x/y/z, motion epoch, subgoal) plus
ABSOLUTE camera heading and pitch. A relative +/-30-degree request cannot
pretend to discover a new direction after the robot turns back.
"""
import math


def wrap_heading(degrees):
    return round(degrees % 360.0, 4)


def angular_separation(a, b):
    return abs((a - b + 180) % 360 - 180)


def same_stance(a, b, horizontal_tolerance=.12, height_tolerance=.12):
    """Unknown translation / new odometry epoch invalidates old exploration."""
    return (bool(a) and bool(b)
            and a.get('motion_epoch', 0) == b.get('motion_epoch', 0)
            and math.hypot(a.get('x', 1e5)-b.get('x', -1e5),
                           a.get('z', 1e5)-b.get('z', -1e5)) <= horizontal_tolerance
            and abs(a.get('y', 1e5)-b.get('y', -1e5)) <= height_tolerance)


def bearing_from_pose(pose):
    return wrap_heading(math.degrees(pose['heading']))


def covered_view(records, pose, subgoal, bearing, pitch, turn_degrees=15):
    """Newest comparable exact camera direction; NEVER a live safety check."""
    for record in reversed(records):
        prior = record.get('pose') or {}
        if (record.get('subgoal') != subgoal or not same_stance(prior, pose)
                or angular_separation(bearing_from_pose(prior), bearing) > turn_degrees / 3
                or abs(prior.get('pitch_deg', 0)-pitch) > 1.0):
            continue
        return record
    return None


def local_coverage(memory, pose, turn_degrees=15, max_headings=24):
    """Describe search history near the current stance; never blacklist occupancy."""
    subgoal = memory.task.current_subgoal_index
    heading = bearing_from_pose(pose)
    pitch = pose.get('pitch_deg', 0)
    visited = {}
    for record in memory.observations:
        prior = record.get('pose') or {}
        if record.get('subgoal') != subgoal or not same_stance(prior, pose):
            continue
        key = round(bearing_from_pose(prior) / turn_degrees) % max_headings
        if abs(prior.get('pitch_deg', 0)-pitch) <= 1:
            visited[key] = record
    current_bin = round(heading / turn_degrees) % max_headings
    unchecked = [i for i in range(max_headings) if i not in visited]
    unchecked.sort(key=lambda i:(min((i-current_bin)%max_headings,
                                     (current_bin-i)%max_headings),i))
    # The latest occupancy status at a previously seen bearing is
    # exploration history, not a permanent block and never a safety permit.
    historical_blocked=[]
    historical_free=[]
    for i,record in visited.items():
        front=next((d for d in record.get('directions',[])
                    if abs(d.get('angle_degrees',1e5))<0.01),None)
        if not front:continue
        if front.get('motion_state')=='BLOCKED':
            historical_blocked.append(wrap_heading(i*turn_degrees))
        elif front.get('motion_state')=='EXECUTABLE':
            historical_free.append(wrap_heading(i*turn_degrees))
    targets = []
    for i in unchecked[:5]:
        target = wrap_heading(i*turn_degrees)
        delta = (target-heading+180)%360-180
        targets.append(dict(absolute_heading_deg=target,
            turn_deg=round(delta, 2),
            pitch_deg=pitch, purpose='acquire_unseen_camera_heading',
            route_authorized=False))
    return dict(reference='estimated_same_stance_and_motion_epoch',
        current_absolute_heading_deg=round(heading, 2), pitch_deg=pitch,
        checked_absolute_headings_deg=sorted(
            wrap_heading(k*turn_degrees) for k in visited),
        checked_count=len(visited), possible_heading_count=max_headings,
        historical_blocked_headings_deg=sorted(historical_blocked),
        historical_executable_headings_deg=sorted(historical_free),
        remaining_unchecked_count=len(unchecked),
        unexplored_directions=targets,
        all_headings_checked=len(unchecked)==0,
        historical_only=True, motion_authorized=False,
        instruction=('choose an unobserved absolute heading, not another repeated relative turn'
                     if unchecked else
                     'all headings at current pitch have been checked; change observation height or route strategy, not repeat sweep'))


def predicted_scan(memory, pose, delta_degrees, look, turn_degrees=15, tilt_degrees=30):
    """Classify intended TURN/LOOK camera frames before spending atomic actions."""
    steps = round(abs(delta_degrees)/turn_degrees)
    direction = 1 if delta_degrees > 0 else -1
    heading = bearing_from_pose(pose)
    pitch = pose.get('pitch_deg', 0)
    target_views = [(wrap_heading(heading+direction*i*turn_degrees),pitch)
                    for i in range(1,steps+1)]
    final_heading = target_views[-1][0] if target_views else heading
    if look != 'none':
        target_views.append((final_heading,pitch+(tilt_degrees if look=='up' else -tilt_degrees)))
    prior = [covered_view(memory.observations,pose,
        memory.task.current_subgoal_index,h,p,turn_degrees) for h,p in target_views]
    return dict(new_view_expected=any(record is None for record in prior),
        already_visited=[dict(heading_deg=h,pitch_deg=p,
            observation_id=record['observation_id']) for (h,p),record
            in zip(target_views,prior) if record is not None],
        unseen_views=[dict(heading_deg=h,pitch_deg=p) for (h,p),record
            in zip(target_views,prior) if record is None],
        note='prediction_only_not_sensor_confirmation')


def last_block(memory, pose, *, max_recent_actions=150):
    """Persist BLOCKED inside PARTIAL, anchored to post-action location.

    Previous code compared PRE-tool pose and checked only status==BLOCKED.
    Both fail when a tool moves 0.25m and then the next step is BLOCKED.
    """
    records = {o['observation_id']:o for o in memory.observations}
    for item in reversed(memory.tool_ledger):
        if (item.get('call') or {}).get('tool') != 'navigate_local':
            continue
        result = item.get('result') or {}
        if result.get('status') == 'TARGET_REACHED':
            return None
        if result.get('status') != 'BLOCKED' and result.get('interrupted_status') != 'BLOCKED':
            if (not result.get('executed_motion') and result.get('status') in (
                    'NEEDS_VIEW','INVALID_TARGET','STALE_TARGET','NO_NEW_INFORMATION')):
                # A rejected pre-execution waypoint selection did not resolve
                # the previous physical collision; keep the last real block.
                continue
            return None
        reference = records.get(result.get('end_observation_id'))
        if not reference or not same_stance(reference.get('pose'),pose):
            return None
        if len(memory.actions)-reference.get('step',len(memory.actions)) > max_recent_actions:
            return None
        return dict(tool_call_id=item.get('call_id'),
            observation_id=reference['observation_id'],
            absolute_heading_deg=round(bearing_from_pose(reference['pose']),2),
            status='BLOCKED',outer_status=result.get('status'),
            interrupted_status=result.get('interrupted_status'),
            reason=result.get('reason'),
            temporal_status='historical_depth_obstacle_recheck_live_before_action',
            requires_live_revalidation=True)
    return None
