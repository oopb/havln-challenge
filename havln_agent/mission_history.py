"""One read-only evidence projection shared by navigation and visual agents.

MemoryRecords/MemoryService alone own facts. This module makes compact,
source-linked views; it never interprets successful task completion, steers
the robot or treats a historical Depth frame as current clearance.
"""
import copy
import math
from .heading import same_local_stance, relative_bearing

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
            objective=str(search.get('objective',''))[:240],
            outcome=search.get('outcome'),
            start_observation_id=search.get('start_observation_id'),
            end_observation_id=search.get('end_observation_id'),
            observation_summary=str(search.get('observation_summary') or '')[:400],
            navigation_suggestion=str(search.get('navigation_suggestion') or '')[:280],
            selected_views=actual[-3:],
            waypoints=waypoints))
    blocked=[]
    for block in memory.evidence.obstacle_blocks[-8:]:
        source=by_id.get(block.get('observation_id'))
        if not source:continue
        key=(block['observation_id'],block.get('heading_deg'),
             block.get('motion_epoch'),block.get('viewpoint_generation'))
        repeat=sum(
            (v.get('observation_id'),v.get('heading_deg'),
             v.get('motion_epoch'),v.get('viewpoint_generation'))==key
            for v in memory.evidence.obstacle_blocks)
        blocked.append(dict(
            observation_id=block['observation_id'],
            heading_deg=block.get('heading_deg'),
            pitch_deg=block.get('pitch_deg'),
            motion_epoch=block.get('motion_epoch'),
            source_stance_matches_current=same_local_stance(source.get('pose'),pose),
            repeated_same_frame_heading_attempts=repeat,
            center_support_pixels=block.get('central_support_pixels'),
            nearest_along_m=block.get('nearest_along_m'),
            reason=block.get('reason'),
            obstacle_is_live_only_at_source_stance=True))
    tool_feedback=[]
    for receipt in memory.tool_ledger[-4:]:
        result=receipt.get('result') or {}
        call=receipt.get('call') or {}
        obstacle=result.get('obstacle_observation') or {}
        depth=obstacle.get('depth') or {}
        center=depth.get('center') or {}
        tool_feedback.append(dict(
            tool_call_id=receipt.get('call_id'),tool=call.get('tool'),
            args=copy.deepcopy(call.get('args') or {}),
            status=result.get('interrupted_status') or result.get('status'),
            reason=result.get('reason'),
            executed_actions=result.get('executed_actions'),
            start_observation_id=result.get('start_observation_id'),
            end_observation_id=result.get('end_observation_id'),
            blocked_at=copy.deepcopy(result.get('blocked_at')),
            blocker=dict(source_observation_id=obstacle.get('source_observation_id'),
                central_support_pixels=center.get('central_support_pixels'),
                source_pixel_xy=center.get('source_pixel_xy'),
                focus_region_xyxy_norm=center.get('focus_region_xyxy_norm'),
                nearest_along_m=center.get('nearest_along_m'),
                nearest_height_m=center.get('nearest_height_m'))
                    if center else None))
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
        blocked_at_source_stances=blocked[-5:],
        recent_tool_receipts=tool_feedback,
        successful_motion_segments=(route.get('successful_segments') or [])[-5:],
        bearing_convention=dict(
            absolute_heading_deg='0..360, increases on LEFT turn',
            model_turn_commands='LEFT or RIGHT + nonnegative degrees; NONE + 0',
            stored_relative_bearing='positive LEFT, negative RIGHT; internal evidence only',
            coordinate_scope='local_pose_and_motion_epoch_not_global_localization'),
        historical_clearance_authorizes_move=False)
