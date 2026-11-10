"""One read-only evidence projection shared by navigation and visual agents.

MemoryRecords/MemoryService alone own facts. This module makes compact,
source-linked views; it never interprets successful task completion, steers
the robot or treats a historical Depth frame as current clearance.
"""
import copy
import json
import math
from .heading import same_local_stance, relative_bearing
from .evidence_ledger import physical_view_key

def _view(record, observation):
    pose=observation.get('pose') or {}
    visual=record.get('visual') or {}
    return dict(
        observation_id=observation['observation_id'],
        heading_deg=round(record.get('heading_deg',
                          math.degrees(pose.get('heading',0))%360),2),
        pitch_deg=record.get('pitch_deg',pose.get('pitch_deg')),
        motion_epoch=pose.get('motion_epoch'),
        position_xz_m_est=(
            [pose.get('x'),pose.get('z')]
            if pose.get('position_coordinates_valid') is not False
            else None),
        observation_summary=(visual.get('observation_summary') or
                             visual.get('evidence') or '')[:300],
        navigation_suggestion=(visual.get('navigation_suggestion') or
                               visual.get('reason') or '')[:180],
        target_type=visual.get('target_type'),
        confidence=visual.get('confidence'),
        frame_scope='historical_sensor_frame_with_model_visual_hypothesis')

def selected_history_frames(memory, current_id=None, max_frames=3):
    """Select real, diverse, source-anchored frames across observation tasks.

    Previous failed/successful exploration receives priority over simply
    choosing the three most recent frames (often near-identical wall views).
    Every returned observation ID must exist in Memory and be loadable by
    the caller. Pose is retained for same-stance and epoch checks.
    """
    by_id={o['observation_id']:o for o in memory.observations}
    candidates=[]
    for search in reversed(memory.observation_searches[-8:]):
        checkpoints=search.get('checkpoints') or []
        for checkpoint in reversed(checkpoints):
            identifier=checkpoint.get('observation_id')
            if identifier not in by_id or identifier==current_id:
                continue
            visual=checkpoint.get('visual') or {}
            score=(2 if visual.get('candidate_approach')
                   and visual.get('passage_box') is not None else 1)
            candidates.append((score,by_id[identifier],checkpoint))
    # Store references to real action endpoints as a fallback when the model
    # has made progress without opening a full visual session.
    for a in reversed(memory.actions[-30:]):
        if a.get('action')!='MOVE_FORWARD':continue
        f=a.get('result') or {}
        identifier=f.get('observation_after')
        if identifier in by_id and identifier!=current_id:
            candidates.append((1,by_id[identifier],
                               dict(observation_id=identifier)))
            break
    selected=[]
    for rank in (2,1):
        for priority,observation,checkpoint in candidates:
            if priority!=rank:continue
            identifier=observation['observation_id']
            if any(v['observation_id']==identifier for v in selected):
                continue
            yaw=math.degrees(observation['pose'].get('heading',0))%360
            place=observation.get('place_id')
            if any(v.get('place_id')==place and
                   abs(relative_bearing(yaw,v['heading_deg']))<15 and
                   abs(v.get('pitch_deg',0)-observation['pose'].get('pitch_deg',0))<15
                   for v in selected):
                continue
            item=_view(checkpoint,observation)
            item['place_id']=place
            selected.append(item)
            if len(selected)==max_frames:
                return selected
    return selected


def recent_views_at_stance(memory, pose, max_views=16, window=80):
    """Complete, compact yaw/pitch coverage for the current action stance.

    All recorded views at this stance contribute, including views older than
    the latest 80 samples. Only the source-labelled detailed examples are
    truncated. No view becomes forbidden and no historical depth is current.
    """
    camera=memory.camera
    yaw_step=max(1,float(camera.turn_angle))
    tilt_step=max(1,float(camera.tilt_angle))
    unique={}
    pitch_bins={}
    samples=0
    repeat_rgbd=0
    seen_image_fingerprints=set()
    for observation in reversed(memory.observations):
        measured_pose=observation.get('pose') or {}
        if not same_local_stance(measured_pose,pose):
            continue
        samples+=1
        yaw=math.degrees(measured_pose.get('heading',0))%360
        pitch=measured_pose.get('pitch_deg',
                                observation.get('pitch_deg',0))
        qyaw=(round(yaw/yaw_step)*yaw_step)%360
        qpitch=round(pitch/tilt_step)*tilt_step
        key=(qyaw,qpitch)
        fingerprint=(key,observation.get('rgb_hash'),
                     observation.get('depth_hash'))
        if fingerprint in seen_image_fingerprints:
            repeat_rgbd+=1
        seen_image_fingerprints.add(fingerprint)
        pitch_bins.setdefault(qpitch,set()).add(qyaw)
        if key in unique:
            continue
        unique[key]=dict(
            source_observation_id=observation.get('observation_id'),
            heading_deg=round(yaw,2),
            pitch_deg=round(pitch,2),
            relative_bearing_deg_from_current=relative_bearing(
                yaw,math.degrees(pose.get('heading',0))%360),
            observed_only=True)
    # Factual scan-cost information at the current executed-forward stance.
    # There is deliberately no Python intervention or chosen heading.
    actions_without_forward=[]
    for action in reversed(memory.actions):
        if action.get('action')=='MOVE_FORWARD':
            break
        actions_without_forward.append(action.get('action'))
    actions_without_forward.reverse()
    yaws={key[0] for key in unique}
    full_horizontal_sweep=(
        len(yaws)>=math.ceil(360/yaw_step)
        if yaw_step<=180 else False)
    # 24 bins at a 15-degree actuator step, independent of episode duration.
    full_coverage=[dict(
        pitch_deg=round(pitch,2),
        observed_heading_bins_deg=[round(a,2) for a in sorted(headings)],
        distinct_headings=len(headings))
        for pitch,headings in sorted(pitch_bins.items())]
    return dict(
        scope='same_executed_forward_stance_not_metric_position',
        sampled_observations=samples,
        distinct_heading_pitch_views=len(unique),
        latest_distinct_views=list(unique.values())[:max_views],
        older_views_may_be_omitted=len(unique)>max_views,
        full_coverage_by_pitch=full_coverage,
        full_coverage_includes_old_views=True,
        repeated_same_RGBD_heading_pitch_views=repeat_rgbd,
        turn_actions_since_last_forward=sum(
            a in ('TURN_LEFT','TURN_RIGHT') for a in actions_without_forward),
        look_actions_since_last_forward=sum(
            a in ('LOOK_UP','LOOK_DOWN') for a in actions_without_forward),
        distinct_horizontal_heading_bins=len(yaws),
        horizontal_360_coverage_observed=full_horizontal_sweep,
        coverage_is_information_not_progress_or_stall_proof=True,
        action_guidance_only_no_hard_view_ban=True,
        historical_depth_does_not_authorize_move=True)


def mission_evidence(memory, pose, max_observation_tasks=4):
    """The common structured truth base for NavigationAgent and VLM."""
    task=memory.task
    current=memory.current or {}
    current_id=current.get('observation_id')
    by_id={o['observation_id']:o for o in memory.observations}
    searches=[]
    for search in memory.observation_searches[-max_observation_tasks:]:
        checkpoints=search.get('checkpoints') or []
        actual=[_view(v,by_id[v['observation_id']])
                for v in checkpoints
                if v.get('observation_id') in by_id]
        waypoints=[]
        for proposal in (search.get('waypoints') or [])[-3:]:
            item={k:copy.deepcopy(proposal[k]) for k in (
                'source_observation_id','target_type','confidence',
                'absolute_heading_deg','relative_bearing_deg_from_current',
                'bearing_source','distance_m_hint','traversability',
                'source_stance_matches_current') if k in proposal}
            if item.get('source_observation_id') not in by_id:continue
            source=by_id[item['source_observation_id']]
            current_stance=same_local_stance(source.get('pose'),pose)
            item['source_stance_matches_current']=current_stance
            # Do not carry forward an old relative bearing computed at the
            # *end* of a previous scan after the robot moved or rotated.
            if current_stance and item.get('absolute_heading_deg') is not None:
                item['relative_bearing_deg_from_current']=relative_bearing(
                    item['absolute_heading_deg'],math.degrees(pose['heading'])%360)
            else:
                # A past camera heading was never a current directional
                # instruction after an executed translation.
                item['historical_bearing_in_source_record_only']=bool(
                    item.get('absolute_heading_deg') is not None)
                item['absolute_heading_deg']=None
                item['relative_bearing_deg_from_current']=None
            waypoints.append(item)
        searches.append(dict(
            observation_task_id=search.get('observation_task_id'),
            objective=str(search.get('objective',''))[:240],
            outcome=search.get('outcome'),
            start_observation_id=search.get('start_observation_id'),
            end_observation_id=search.get('end_observation_id'),
            observation_summary=str(search.get('observation_summary') or '')[:400],
            navigation_suggestion=str(search.get('navigation_suggestion') or '')[:280],
            selected_views=actual[-3:],
            waypoints=waypoints))
    blocked=[]
    # Dedup a *physical* view (RGB/Depth hashes and local pose), not an
    # observation_id assigned by the record writer on each sensor refresh.
    block_counts={}
    for item in memory.evidence.obstacle_blocks:
        key=physical_view_key(item)
        block_counts[key]=block_counts.get(key,0)+1
    block_keys=set()
    for item in reversed(memory.evidence.obstacle_blocks):
        key=physical_view_key(item)
        if key in block_keys:continue
        block_keys.add(key)
        source=by_id.get(item.get('observation_id'))
        blocked.append(dict(
            kind='BLOCKED_BEFORE_FORWARD',
            source_observation_id=item.get('observation_id'),
            rgb_hash=item.get('rgb_hash'),
            depth_hash=item.get('depth_hash'),
            heading_deg=item.get('heading_deg'),
            pitch_deg=item.get('pitch_deg'),
            motion_epoch=item.get('motion_epoch'),
            source_stance_matches_current=(
                same_local_stance(source.get('pose'),pose) if source else False),
            repeated_same_frame_heading_attempts=block_counts[key],
            central_support_pixels=item.get('central_support_pixels'),
            nearest_along_m=item.get('nearest_along_m'),
            focus_region_xyxy_norm=item.get('focus_region_xyxy_norm'),
            source_pixel_xy=item.get('source_pixel_xy'),
            reason=item.get('reason'),
            forward_atomic_action_executed=False,
            historical_depth_does_not_prove_current_obstacle=True))
        if len(blocked)>=5:break
    stall_counts={}
    for item in memory.evidence.motion_stalls:
        key=physical_view_key(item)
        stall_counts[key]=stall_counts.get(key,0)+1
    stall_keys=set()
    stalls=[]
    for item in reversed(memory.evidence.motion_stalls):
        key=physical_view_key(item)
        if key in stall_keys:continue
        stall_keys.add(key)
        source=by_id.get(item.get('observation_id'))
        stalls.append(dict(
            kind='STALLED_AFTER_FORWARD',
            source_observation_id=item.get('observation_id'),
            after_observation_id=item.get('after_observation_id'),
            rgb_hash=item.get('rgb_hash'),depth_hash=item.get('depth_hash'),
            heading_deg=item.get('heading_deg'),pitch_deg=item.get('pitch_deg'),
            motion_epoch=item.get('motion_epoch'),
            source_stance_matches_current=(
                same_local_stance(source.get('pose'),pose) if source else False),
            repeated_same_frame_heading_attempts=stall_counts[key],
            last_action_step=item.get('action_step'),
            forward_atomic_action_executed=True,
            visual_change=False,collision_proven=False,
            displacement_verified=False,
            reason='executed_forward_same_RGBD_before_and_after'))
        if len(stalls)>=5:break
    observed_front_blocks=[]
    for item in reversed(memory.evidence.predecision_front_obstacles[-6:]):
        source=by_id.get(item.get('observation_id'))
        observed_front_blocks.append(dict(
            kind='OBSERVED_FRONT_DEPTH_BLOCK_NO_MOVE_ATTEMPT',
            source_observation_id=item.get('observation_id'),
            last_observation_id=item.get('last_observation_id'),
            heading_deg=item.get('heading_deg'),
            pitch_deg=item.get('pitch_deg'),
            motion_epoch=item.get('motion_epoch'),
            source_stance_matches_current=(
                same_local_stance(source.get('pose'),pose) if source else False),
            repeat_observation_count=item.get('seen_count',1),
            blocker=copy.deepcopy(item.get('blocker')),
            attempted_forward=False,
            historical_depth_does_not_certify_current_block=True))
        if len(observed_front_blocks)>=4:break
    tool_feedback=[]
    receipt_keys=set()
    for receipt in reversed(memory.tool_ledger[-12:]):
        result=receipt.get('result') or {}
        call=receipt.get('call') or {}
        status=result.get('interrupted_status') or result.get('status')
        args=call.get('args') or {}
        key=(call.get('tool'),json.dumps(args,sort_keys=True),
             result.get('end_observation_id'),status)
        if result.get('executed_actions')==0 and status in ('BLOCKED','PARTIAL'):
            if key in receipt_keys:continue
            receipt_keys.add(key)
        obstacle=result.get('obstacle_observation') or {}
        depth=obstacle.get('depth') or {}
        # v8 incorrectly assumed depth.center nesting. The actual source
        # is obstacle_observation.depth.focus_region_xyxy_norm.
        center=depth.get('center') or depth
        tool_feedback.append(dict(
            tool_call_id=receipt.get('call_id'),tool=call.get('tool'),
            args=copy.deepcopy(args),
            status=status,reason=result.get('reason'),
            executed_actions=result.get('executed_actions'),
            start_observation_id=result.get('start_observation_id'),
            end_observation_id=result.get('end_observation_id'),
            repeated_same_RGBD_stance_heading=result.get(
                'repeated_same_RGBD_stance_heading'),
            camera_pitch_limit=(
                dict(requested_look=result.get('requested_look'),
                     current_pitch_deg=result.get('current_pitch_deg'),
                     requested_pitch_deg=result.get('requested_pitch_deg'),
                     pitch_limits_deg=copy.deepcopy(result.get('pitch_limits_deg')),
                     tilt_step_deg=result.get('tilt_step_deg'),
                     allowed_look=copy.deepcopy(result.get('allowed_look')),
                     recoverable=result.get('recoverable'),
                     executed_camera_actions=copy.deepcopy(
                         result.get('executed_camera_actions',[])))
                if status=='CAMERA_PITCH_LIMIT' else None),
            blocked_at=copy.deepcopy(result.get('blocked_at')),
            blocker=dict(source_observation_id=obstacle.get('source_observation_id'),
                central_support_pixels=center.get('central_support_pixels'),
                source_pixel_xy=center.get('source_pixel_xy'),
                focus_region_xyxy_norm=center.get('focus_region_xyxy_norm'),
                nearest_along_m=center.get('nearest_along_m'),
                nearest_height_m=center.get('nearest_height_m'))
                    if center else None))
        if len(tool_feedback)>=4:break
    tool_feedback.reverse()
    camera_limits=[]
    seen_pitch_rejections=set()
    # A non-executed camera-boundary error is valid navigation feedback.
    # Collapse repetitions so the model sees what failed and what remains
    # physically possible instead of a list of identical error messages.
    for receipt in reversed(memory.tool_ledger[-24:]):
        result=receipt.get('result') or {}
        if (result.get('interrupted_status') or
                result.get('status'))!='CAMERA_PITCH_LIMIT':
            continue
        key=(result.get('current_pitch_deg'),result.get('requested_look'))
        if key in seen_pitch_rejections:
            continue
        seen_pitch_rejections.add(key)
        count=sum(
            (v.get('result') or {}).get('status')=='CAMERA_PITCH_LIMIT'
            and (v.get('result') or {}).get('current_pitch_deg')==key[0]
            and (v.get('result') or {}).get('requested_look')==key[1]
            for v in memory.tool_ledger[-24:])
        camera_limits.append(dict(
            current_pitch_deg=key[0],
            rejected_look=key[1],
            allowed_look=copy.deepcopy(result.get('allowed_look',[])),
            last_tool_call_id=receipt.get('call_id'),
            repeated_same_invalid_look=count,
            recovered_without_actions=True,
            suggested_next_step='model_selects_another_valid_action'))
        if len(camera_limits)>=3:
            break
    route=memory.route_progress_summary()
    # Keep a compact, source-linked route candidate pool independent of the
    # last three observation sessions. A missing box never becomes a bearing.
    route_candidates=[]
    repeated_unlocalized={}
    for candidate in reversed(getattr(memory,'route_candidates',[])):
        original=by_id.get(candidate.get('source_observation_id'))
        if not original:
            continue
        # Several scans of the same blocked person at one action stance
        # are distinct source facts, not three independent ROUTES. Keep
        # complete records queryable; summarize repeated unboxed cues once.
        unlocalized=not candidate.get('angular_bearing_grounded')
        cue_key=(
            str(candidate.get('target_type') or '').casefold(),
            candidate.get('target_kind'),
            candidate.get('source_stance_id') or
                candidate.get('source_observation_id'),
            candidate.get('objective')) if unlocalized else None
        if cue_key in repeated_unlocalized:
            repeated_unlocalized[cue_key][
                'other_source_observation_ids'].append(
                    candidate.get('source_observation_id'))
            continue
        fresh=same_local_stance(original.get('pose') or {},pose)
        absolute=candidate.get('absolute_heading_deg')
        item={key:copy.deepcopy(candidate.get(key)) for key in (
            'candidate_id','source_observation_id','target_type',
            'target_kind','confidence','evidence','navigation_suggestion',
            'uncertainty','target_visible','absolute_heading_deg',
            'angular_bearing_grounded','route_status',
            'bearing_region_role')}
        if unlocalized:
            item['other_source_observation_ids']=[]
            item['compact_unlocalized_cue_not_independent_route']=True
            repeated_unlocalized[cue_key]=item
        item['source_stance_matches_current']=fresh
        item['historical_bearing_in_source_record_only']=bool(
            not fresh and isinstance(absolute,(int,float)))
        if not fresh:
            # Keep the source record for retrieval, but do NOT invite an
            # old numerical heading to masquerade as today's target.
            item['absolute_heading_deg']=None
        item['relative_bearing_deg_from_current']=(
            round(relative_bearing(absolute,
                 math.degrees(pose.get('heading',0))%360),2)
            if fresh and isinstance(absolute,(int,float)) else None)
        item['historical_only_not_live_clearance']=True
        route_candidates.append(item)
        if len(route_candidates)>=6:
            break
    # Compact source-linked receipt rather than repeating raw action logs.
    raw_motion=getattr(memory,'latest_motion_receipt',None)
    last_motion=None
    if raw_motion is not None:
        last_motion={k:copy.deepcopy(raw_motion.get(k)) for k in (
            'decision_id','tool','source_observation_id',
            'end_observation_id','last_heading_deg',
            'last_stance_id','status','reason',
            'executed_forward_actions','forward_actions_with_RGBD_change',
            'last_motion_was_uninterrupted','continuation_is_model_choice')}
        last_motion['same_camera_as_last_motion']=(
            (memory.current or {}).get('observation_id')==
            raw_motion.get('end_observation_id'))
        last_motion['motion_is_evidence_not_command']=True
        last_motion['atomic_actions_since_motion']=max(
            0,len(memory.actions)-raw_motion.get('action_step_finished',0))
        last_motion['current_heading_deg']=round(
            math.degrees(pose.get('heading',0))%360,2)
    anchor=copy.deepcopy(getattr(task,'navigation_intent_anchor',{}) or {})
    anchored_source=by_id.get(anchor.get('source_observation_id'))
    source_pose=(anchored_source or {}).get('pose') or {}
    # SAME stance is not sufficient for camera-relative left/right:
    # a successful TURN changes the relative bearing without moving.
    source_yaw=anchor.get('source_heading_deg')
    current_yaw=math.degrees(pose.get('heading',0))%360
    intent_current=bool(
        anchored_source and same_local_stance(source_pose,pose)
        and isinstance(source_yaw,(int,float))
        and abs(relative_bearing(source_yaw,current_yaw))<1.0)
    prior_intent=dict(
        text=task.navigation_intent,
        source=anchor or None,
        relative_direction_valid_for_current_camera=intent_current,
        historical_only=bool(anchor and not intent_current),
        recheck_against_current_RGBD_before_reusing_direction=True)
    return dict(
        contract='mission_evidence_v1',
        instruction=task.instruction,
        progress=dict(
            narrative=task.progress_narrative,
            navigation_intent=(task.navigation_intent
                if not anchor or intent_current else None),
            prior_navigation_intent=prior_intent,
            uncertainties=list(task.uncertainties[-6:]),
            model_revision_count=len(task.progress_history),
            source_observation_ids=list(task.evidence_observation_ids[-12:]),
            evidence_interpretation='VLM/model_assessments_not_verified_task_completion'),
        current=dict(observation_id=current_id,
            place_id=current.get('place_id'),
            heading_deg=round(math.degrees(pose.get('heading',0))%360,2),
            pitch_deg=pose.get('pitch_deg'),
            motion_epoch=pose.get('motion_epoch'),
            position_xz_m_est=(
                [pose.get('x'),pose.get('z')]
                if pose.get('position_coordinates_valid') is not False
                else None),
            stance_id=pose.get('stance_id'),
            position_coordinates_valid=pose.get('position_coordinates_valid',True)),
        observation_tasks=searches,
        candidate_route_hypotheses=route_candidates,
        last_motion_decision=last_motion,
        historical_frames=selected_history_frames(memory,current_id),
        blocked_at_source_stances=blocked,
        stalled_forward_at_source_stances=stalls,
        observed_front_blocks=observed_front_blocks,
        # Preserve the two distinct evidence modalities. Never infer a
        # measured wall from an executed MOVE with unchanged RGB-D.
        recent_forward_failures=(blocked[:3]+stalls[:3]),
        recent_tool_receipts=tool_feedback,
        recent_camera_pitch_limits=camera_limits,
        recent_views_at_current_stance=recent_views_at_stance(memory,pose),
        observed_forward_action_segments=(route.get('successful_segments') or [])[-5:],
        forward_action_segments_are_not_verified_translation=True,
        metric_pose_available=False,
        bearing_convention=dict(
            model_turn_commands='LEFT/RIGHT/NONE + nonnegative turn_degrees',
            camera_tilt_commands='look=up/down/none',
            actuator_conversion_is_executor_only=True,
            source_bearings_are_historical_references_only=True,
            coordinate_scope='executed_forward_stance_yaw_pitch_only_no_metric_xyz'),
        historical_clearance_authorizes_move=False)
