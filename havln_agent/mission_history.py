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
        position_xz_m_est=[pose.get('x'),pose.get('z')],
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
            score=2 if (checkpoint.get('visual') or {}).get('candidate_approach') else 1
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
    route=memory.route_progress_summary()
    return dict(
        contract='mission_evidence_v1',
        instruction=task.instruction,
        progress=dict(
            narrative=task.progress_narrative,
            navigation_intent=task.navigation_intent,
            uncertainties=list(task.uncertainties[-6:]),
            model_revision_count=len(task.progress_history),
            source_observation_ids=list(task.evidence_observation_ids[-12:]),
            evidence_interpretation='VLM/model_assessments_not_verified_task_completion'),
        current=dict(observation_id=current_id,
            place_id=current.get('place_id'),
            heading_deg=round(math.degrees(pose.get('heading',0))%360,2),
            pitch_deg=pose.get('pitch_deg'),
            motion_epoch=pose.get('motion_epoch'),
            position_xz_m_est=[pose.get('x'),pose.get('z')],
            position_coordinates_valid=pose.get('position_coordinates_valid',True)),
        observation_tasks=searches,
        historical_frames=selected_history_frames(memory,current_id),
        blocked_at_source_stances=blocked,
        stalled_forward_at_source_stances=stalls,
        # Preserve the two distinct evidence modalities. Never infer a
        # measured wall from an executed MOVE with unchanged RGB-D.
        recent_forward_failures=(blocked[:3]+stalls[:3]),
        recent_tool_receipts=tool_feedback,
        successful_motion_segments=(route.get('successful_segments') or [])[-5:],
        bearing_convention=dict(
            absolute_heading_deg='0..360, increases on LEFT turn',
            model_turn_commands='LEFT or RIGHT + nonnegative degrees; NONE + 0',
            stored_relative_bearing='positive LEFT, negative RIGHT; internal evidence only',
            coordinate_scope='local_pose_and_motion_epoch_not_global_localization'),
        historical_clearance_authorizes_move=False)
