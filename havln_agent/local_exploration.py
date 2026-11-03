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
    for record in getattr(memory,'observations',()):
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



def multi_pitch_coverage(memory, pose, turn_degrees=15, pitch_degrees=30):
    """Compact camera coverage from AUTHENTIC public observations at this stance.

    Stored observations are the source of truth: no second mutable heading map.
    Spatial identity uses the measured x/y/z and motion_epoch, not a static
    room label. Every yaw/pitch is absolute in the current odometry frame.
    Historical visibility NEVER grants fresh collision or human clearance.
    """
    observations = getattr(memory, 'observations', ())
    bins = {}
    for record in observations:
        prior = record.get('pose') or {}
        if not same_stance(prior, pose):
            continue
        yaw = bearing_from_pose(prior)
        pitch = float(record.get('pitch_deg', prior.get('pitch_deg', 0)))
        yaw_bin = round(yaw / turn_degrees) % max(1, round(360 / turn_degrees))
        pitch_bin = round(pitch / pitch_degrees) * pitch_degrees
        key = (pitch_bin, yaw_bin)
        cell = bins.setdefault(key, dict(
            pitch_deg=pitch_bin,
            heading_deg=wrap_heading(yaw_bin * turn_degrees),
            samples=0, revisits=0, last_observation_id=None,
            newest_static_information=False))
        cell['samples'] += 1
        if record.get('revisited_static_view', False):
            cell['revisits'] += 1
        cell['last_observation_id'] = record.get('observation_id')
        cell['newest_static_information'] = bool(
            record.get('new_visual_information', True))
    by_pitch = {}
    for cell in bins.values():
        by_pitch.setdefault(cell['pitch_deg'], []).append(cell)
    levels = []
    for pitch, cells in sorted(by_pitch.items()):
        cells.sort(key=lambda c: c['heading_deg'])
        levels.append(dict(
            pitch_deg=pitch,
            observed_headings_deg=[c['heading_deg'] for c in cells],
            distinct_headings=len(cells),
            repeated_observation_samples=sum(max(0, c['samples'] - 1)
                                             for c in cells),
            recent_revisited_headings_deg=[
                c['heading_deg'] for c in cells if c['revisits']],
            recent_sample_observation_ids=[
                c['last_observation_id'] for c in cells[-3:]]))
    total = sum(c['samples'] for c in bins.values())
    current_pitch = round(float(pose.get('pitch_deg', 0)) /
                          pitch_degrees) * pitch_degrees
    seen_current = {c['heading_deg']
                    for c in by_pitch.get(current_pitch, [])}
    return dict(
        reference='estimated_same_stance_and_motion_epoch',
        absolute_yaw_convention='degrees_0_to_360',
        pose_epoch=pose.get('motion_epoch', 0),
        current_absolute_heading_deg=bearing_from_pose(pose),
        current_pitch_deg=pose.get('pitch_deg', 0),
        heading_bin_deg=turn_degrees, pitch_bin_deg=pitch_degrees,
        observed_pitch_levels=levels,
        total_observation_samples=total,
        unique_heading_pitch_cells=len(bins),
        repeated_samples=max(0, total - len(bins)),
        current_pitch_checked_headings_deg=sorted(seen_current),
        measurement='historical_public_RGBD_only_not_live_path_clearance')


def observation_stagnation(memory, pose, window=24):
    """Summarize repeated information-poor observations; no forbidden actions.

    Unlike the action/observation-ID liveness guard, a LOOK_UP followed by
    LOOK_DOWN at a known stance is not considered useful progress merely
    because it consumed two atomic actions and produced new IDs.
    """
    observations = {o['observation_id']: o
                    for o in getattr(memory, 'observations', ())}
    actions = getattr(memory, 'actions', ())
    last_forward = max((a.get('step', i) for i, a in enumerate(actions)
                        if a.get('action') == 'MOVE_FORWARD'), default=-1)
    observation_tools = {
        'turn_and_observe', 'observe_for_goal',
        'get_waypoint_candidates', 'inspect_geometry',
        'analyze_scene', 'ground_landmark', 'compare_views'}
    sampled = []
    for entry in reversed(getattr(memory, 'tool_ledger', ())):
        if len(sampled) >= window:
            break
        result = entry.get('result') or {}
        tool = (entry.get('call') or {}).get('tool')
        if tool not in observation_tools:
            continue
        final = observations.get(result.get('end_observation_id'))
        if not final or not same_stance(final.get('pose'), pose):
            continue
        # Coverage remains useful across goals, but an observation-loop
        # intervention belongs to the CURRENT semantic milestone only.
        task = getattr(memory, 'task', None)
        if (task is not None and final.get('subgoal') !=
                getattr(task, 'current_subgoal_index', None)):
            continue
        # Do not interpret previous physical locations as current stagnation.
        if final.get('step', -1) <= last_forward:
            continue
        status = result.get('status')
        low_gain = (status == 'NO_NEW_INFORMATION'
                    or result.get('outcome') == 'REPEATED_SEARCH_NO_NEW_VIEW'
                    or result.get('reason') in (
                        'scan_completed_without_new_static_view',
                        'local_view_coverage_exhausted_with_unknown_geometry',
                        'no_new_relative_view_candidates'))
        if tool in ('turn_and_observe', 'observe_for_goal'):
            low_gain = low_gain or bool(
                final.get('revisited_static_view', False))
        elif tool == 'get_waypoint_candidates':
            # Repeated NEEDS_VIEW and no executable candidate is not a new
            # physically usable route; still not a safety prohibition.
            low_gain = low_gain or (status == 'NEEDS_VIEW'
                and not any(p.get('first_step') == 'EXECUTABLE'
                            for p in result.get('paths', ())))
        sampled.append(dict(
            tool=tool, status=status, low_information=bool(low_gain),
            observation_id=final['observation_id'],
            heading_deg=bearing_from_pose(final['pose']),
            pitch_deg=final.get('pitch_deg',
                                final['pose'].get('pitch_deg', 0))))
    low = sum(x['low_information'] for x in sampled)
    repeatedly_unproductive = len(sampled) >= 8 and low >= 6
    return dict(
        same_stance_observation_calls=len(sampled),
        low_information_calls=low,
        observations_since_last_forward=len(sampled),
        repeated_low_information=bool(repeatedly_unproductive),
        recent_attempts=list(reversed(sampled[:6])),
        no_new_route_from_repeated_scans=bool(repeatedly_unproductive),
        interpretation=('same_stance_observation_loop_no_new_actionable_evidence'
                        if repeatedly_unproductive else 'normal_observation'),
        decision_authority='NavigationAgent',
        no_forced_physical_action=True)



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
    prior = [covered_view(getattr(memory,'observations',()),pose,
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
    records = {o['observation_id']:o for o in getattr(memory,'observations',())}
    for item in reversed(getattr(memory,'tool_ledger',())):
        if (item.get('call') or {}).get('tool') not in (
                'navigate_local','navigate_relative','navigate_to_pixel'):
            continue
        result = item.get('result') or {}
        if result.get('status') == 'TARGET_REACHED':
            return None
        if result.get('status') != 'BLOCKED' and result.get('interrupted_status') != 'BLOCKED':
            if ('MOVE_FORWARD' not in (result.get('executed_motion') or [])
                    and result.get('status') in ('PARTIAL','NEEDS_VIEW',
                        'INVALID_TARGET','STALE_TARGET','NO_NEW_INFORMATION')):
                # Neither a rejected waypoint nor a turn-only continuation
                # invalidates the previous physical collision at this stance.
                # A newer verified MOVE_FORWARD or blocked event supersedes it.
                continue
            return None
        reference = records.get(result.get('end_observation_id'))
        if not reference or not same_stance(reference.get('pose'),pose):
            return None
        actions=getattr(memory,'actions',())
        if len(actions)-reference.get('step',len(actions)) > max_recent_actions:
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
