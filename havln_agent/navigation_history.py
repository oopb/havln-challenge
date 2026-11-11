"""Complete, causally ordered navigation history from canonical episode records.

Text history covers ALL earlier navigation turns, atomic action receipts,
observation sessions and source-indexed RGB-D records. Only a diverse,
bounded subset of historical RGB-D bytes is attached to each request; the
complete image catalog remains addressable by observation ID. No positions,
physical travel distances, verified revisits or live safety are inferred.
"""
import math

from .heading import same_local_stance, relative_bearing
from .mission_history import selected_history_frames
from .navigation_evidence import salient_forward_segment


def _heading(pose):
    value=pose.get('heading')
    return round(math.degrees(value)%360,2) if isinstance(value,(int,float)) else None


def _source(memory, observation_id):
    return next((o for o in memory.observations
                 if o.get('observation_id')==observation_id),None)


def navigation_episode_history(memory, pose):
    """No sliding time window or discarded early route/observation events."""
    timeline=[]
    sequence=0
    def append(step,priority,kind,**fields):
        nonlocal sequence
        timeline.append(dict(step=step,kind=kind,**fields,
                             _priority=priority,_sequence=sequence))
        sequence+=1

    for turn in getattr(memory,'navigation_turns',[]):
        start=turn.get('step_started',0)
        decision=turn.get('decision') or {}
        append(start,0,'NAVIGATION_DECISION',
            decision_id=turn.get('decision_id'),
            source_observation_id=turn.get('source_observation_id'),
            source_heading_deg=turn.get('source_heading_deg'),
            source_stance_id=turn.get('source_stance_id'),
            source_viewpoint_generation=turn.get('source_viewpoint_generation'),
            tool=decision.get('tool'),args=decision.get('args'),
            progress_narrative=decision.get('progress_narrative'),
            navigation_intent=decision.get('navigation_intent'),
            uncertainties=decision.get('uncertainties'),
            evidence_observation_ids=decision.get('evidence_observation_ids'))
        append(turn.get('step_finished',start),4,'NAVIGATION_TOOL_RESULT',
            decision_id=turn.get('decision_id'),
            tool_call_id=turn.get('tool_call_id'),
            result=turn.get('result'))

    # No separate truncated "recent actions" log. Every actual atomic
    # receipt is included once, including turning, camera looks and STOP.
    for action in memory.actions:
        feedback=action.get('result') or {}
        p=action.get('pose') or {}
        append(action.get('step',0),2,'ATOMIC_ACTION',
            action=action.get('action'),
            observation_before=feedback.get('observation_before'),
            observation_after=feedback.get('observation_after'),
            heading_deg=_heading(p),
            pitch_deg=p.get('pitch_deg'),
            stance_id=p.get('stance_id'),
            viewpoint_generation=p.get('viewpoint_generation'),
            RGBD_changed=feedback.get('visual_change'),
            forward_preflight_status=(
                (feedback.get('forward_depth_preflight') or {}).get('status')
                if action.get('action')=='MOVE_FORWARD' else None),
            metric_displacement_verified=False)

    for i,session in enumerate(memory.observation_searches):
        start=session.get('start_step')
        if start is None:
            source=_source(memory,session.get('start_observation_id'))
            start=(source or {}).get('step',0)
        checkpoints=[]
        for view in session.get('checkpoints') or []:
            visual=view.get('visual') or {}
            checkpoints.append(dict(
                observation_id=view.get('observation_id'),
                heading_deg=view.get('heading_deg'),
                pitch_deg=view.get('pitch_deg'),
                target_type=visual.get('target_type'),
                target_visible=visual.get('target_visible'),
                box=visual.get('box'),
                passage_box=visual.get('passage_box'),
                candidate_approach=visual.get('candidate_approach'),
                exact_bearing_requires_passage_box=True,
                observation_summary=visual.get('observation_summary'),
                navigation_suggestion=visual.get('navigation_suggestion'),
                next_action=visual.get('next_action'),
                turn_deg=visual.get('turn_deg')))
        append(start,1,'OBSERVATION_TASK',
            observation_task_id=session.get('observation_task_id'),
            source_observation_id=session.get('start_observation_id'),
            end_observation_id=session.get('end_observation_id'),
            objective=session.get('objective'),
            outcome=session.get('outcome'),
            observation_summary=session.get('observation_summary'),
            navigation_suggestion=session.get('navigation_suggestion'),
            turn_actions=list(session.get('actions') or []),
            checkpoints=checkpoints,
            candidate_waypoints=[dict(
                source_observation_id=w.get('source_observation_id'),
                target_type=w.get('target_type'),
                target_region=w.get('target_region'),
                semantic_target_region=w.get('semantic_target_region'),
                bearing_region_role=w.get('bearing_region_role'),
                absolute_heading_deg=w.get('absolute_heading_deg'),
                heading_at_capture_deg=w.get('heading_at_capture_deg'),
                motion_authorized=False)
                for w in session.get('waypoints') or []])

    # Compatibility for early/partial episodes whose model decisions have
    # not yet been archived through note_navigation_turn().
    recorded={turn.get('tool_call_id')
              for turn in getattr(memory,'navigation_turns',[])}
    for receipt in memory.tool_ledger:
        if receipt.get('call_id') in recorded:
            continue
        call=receipt.get('call') or {}
        result=receipt.get('result') or {}
        append((receipt.get('request_context') or {}).get(
                   'step', (receipt.get('request_context') or {}).get(
                       'step_started',0)),3,'UNARCHIVED_TOOL_RECEIPT',
            call_id=receipt.get('call_id'),
            source_observation_id=result.get('start_observation_id'),
            tool=call.get('tool'),args=call.get('args'),
            status=result.get('status'),
            end_observation_id=result.get('end_observation_id'),
            executed_motion=result.get('executed_motion'))

    turns=getattr(memory,'navigation_turns',[])
    def owner(event):
        kind=event['kind']
        if kind in ('NAVIGATION_DECISION','NAVIGATION_TOOL_RESULT'):
            return next((i for i,t in enumerate(turns)
                if t.get('decision_id')==event.get('decision_id')
                and ((kind=='NAVIGATION_DECISION'
                      and t.get('step_started')==event['step'])
                     or (kind=='NAVIGATION_TOOL_RESULT'
                      and t.get('step_finished')==event['step']))),
                len(turns))
        step=event['step']
        if kind=='OBSERVATION_TASK':
            # An observation tool can consume zero atomic turns; do not
            # accidentally place its checkpoint after the next decision.
            matches=[i for i,t in enumerate(turns)
                if (t.get('decision') or {}).get('tool')=='observe_for_goal'
                and t.get('step_started',-1)<=step<=t.get('step_finished',-1)]
            if matches:
                return matches[0]
            # Legacy/imported traces may contain an observation session
            # whose parent tool was not labeled observe_for_goal. Keep
            # its original step in the correct causal action span rather
            # than appending an early session after the whole episode.
            return next((i for i,t in enumerate(turns)
                if t.get('step_started',-1)<=step<=t.get('step_finished',-1)),
                len(turns))
        if kind=='ATOMIC_ACTION':
            return next((i for i,t in enumerate(turns)
                if t.get('step_started',-1)<=step<t.get('step_finished',-1)),
                len(turns))
        return len(turns)

    # Action step alone cannot order two zero-action decisions, or a tool
    # result and the next decision at the SAME action count. Use the
    # recorded model-turn index as the primary causal clock, followed by
    # decision -> observation -> atomic actions -> result.
    timeline.sort(key=lambda e:(
        owner(e),e['_priority'],e['step'],e['_sequence']))
    for item in timeline:
        item.pop('_priority')
        item.pop('_sequence')

    catalog=[]
    for o in memory.observations:
        p=o.get('pose') or {}
        catalog.append(dict(
            observation_id=o.get('observation_id'),
            action_step=o.get('step'),
            heading_deg=_heading(p),pitch_deg=p.get('pitch_deg'),
            stance_id=p.get('stance_id'),
            viewpoint_generation=p.get('viewpoint_generation'),
            rgb_hash=o.get('rgb_hash'),depth_hash=o.get('depth_hash'),
            same_pose_revisited=bool(o.get('revisited_static_view')),
            possible_cross_stance_matches=[
                dict(observation_id=v.get('observation_id'),
                     confidence=v.get('confidence'),
                     localization_verified=False)
                for v in (o.get('revisit_matches') or [])],
            physically_localized=False))
    return dict(
        contract='navigation_episode_history_v2',
        chronology='ascending_action_step_with_decisions_before_execution',
        complete_action_history=True,
        complete_navigation_turn_history=(
            len(getattr(memory,'navigation_turns',[]))>=
            len(memory.tool_ledger)),
        missing_archived_turns_recovered_as_tool_receipts=max(
            0,len(memory.tool_ledger)-
            len(getattr(memory,'navigation_turns',[]))),
        complete_observation_task_history=True,
        all_rgbd_frames_indexed=True,
        historical_depth_is_never_live_clearance=True,
        physical_revisit_not_verified_by_visual_similarity=True,
        counts=dict(
            navigation_turns=len(getattr(memory,'navigation_turns',[])),
            atomic_actions=len(memory.actions),
            observation_tasks=len(memory.observation_searches),
            source_frames=len(memory.observations),
            timeline_events=len(timeline)),
        timeline=timeline,
        rgbd_observation_catalog=catalog)


def _same_camera_view(first, second, camera):
    """One view slot per executed-forward stance and calibrated yaw/pitch."""
    a=(first or {}).get('pose') or {}
    b=(second or {}).get('pose') or {}
    yaw_a,yaw_b=_heading(a),_heading(b)
    pa,pb=a.get('pitch_deg'),b.get('pitch_deg')
    return bool(same_local_stance(a,b) and
        yaw_a is not None and yaw_b is not None and
        abs(relative_bearing(yaw_a,yaw_b))<
            max(1.,camera.turn_angle/2) and
        isinstance(pa,(int,float)) and isinstance(pb,(int,float)) and
        abs(pa-pb)<max(1.,camera.tilt_angle/2))


def navigation_visual_anchors(memory, current_id, max_pairs=5):
    """Select real source-backed RGB-D pairs from distinct trajectory phases.

    This selection does NOT remove historical events from the text timeline.
    Prefer possible revisits, earliest scene, recent meaningful evidence,
    recent changed-forward view and high-confidence observation checkpoints.
    No image similarity match is promoted to true physical localization.
    """
    by_id={o['observation_id']:o for o in memory.observations}
    current=by_id.get(current_id) or {}
    options=[]
    def offer(identifier,reason):
        if identifier in by_id and identifier!=current_id:
            options.append((identifier,reason))
    # Keep the original junction/stair-exit RGB accessible when returning
    # from a failed route. Do not infer that the current stance is the same:
    # this is a HISTORICAL view, never a computed turn or localization result.
    references=getattr(memory,'route_references',[]) or []
    if references:
        # Keep the first instruction-relative route choice and the newest
        # local fork visible together. Historical views are never motor turns.
        root=references[0]
        offer(root.get('source_observation_id'),
              'instruction_relative_route_reference')
        if max_pairs>=3:
            offer(root.get('approach_observation_id'),
                  'stair_exit_approach_view_reference')
        latest=references[-1]
        if latest is not root and max_pairs>=3:
            offer(latest.get('source_observation_id'),
                  'recent_route_reference')
        if max_pairs>=5:
            for attempt in (latest.get('route_attempts') or [])[-1:]:
                offer(attempt.get('evidence_observation_id'),
                      'route_attempt_result_reference')
    # Preserve the real scene view the model chose as its active spatial
    # goal. This historical RGB is evidence for matching the room/layout,
    # never a current heading or guaranteed physical localization.
    scene_source=getattr(memory.task,'scene_goal_source_observation_id','')
    if getattr(memory.task,'scene_goal','') and scene_source:
        offer(scene_source,'selected_scene_goal_reference')
    # Reserve actual before/after photos of executed motion, but do not let
    # a long kitchen detour displace the source image of a critical junction.
    # Earlier repeated views at one yaw are not an alternative to the actual
    # before/after pictures of an executed forward trajectory segment.
    yaw=_heading(current.get('pose') or {})
    if yaw is not None:
        all_segments=memory.route_progress_summary(
            max_successful_segments=None).get('successful_segments') or []
        anchor=salient_forward_segment(all_segments,yaw)
        endpoint=by_id.get(anchor.get('end_observation_id')) if anchor else None
        relevant=bool(anchor and (
            abs(relative_bearing(anchor['heading_deg'],yaw))>=105
            or (endpoint and same_local_stance(
                endpoint.get('pose'),current.get('pose')))))
        if relevant:
            offer(anchor['start_observation_id'],
                  'prior_forward_start_visual_evidence')
            offer(anchor['end_observation_id'],
                  'prior_forward_end_visual_evidence')
    # One recent changed-forward image, then meaningful route candidates.
    # If it is already the important endpoint, dedup below removes it.
    for action in reversed(memory.actions):
        if action.get('action')=='MOVE_FORWARD':
            feedback=action.get('result') or {}
            if feedback.get('visual_change'):
                offer(feedback.get('observation_after'),
                      'recent_forward_changed_view')
                break
    for session in reversed(memory.observation_searches):
        candidates=[
            cp for cp in session.get('checkpoints') or []
            if ((cp.get('visual') or {}).get('candidate_approach')
                and (cp.get('visual') or {}).get('passage_box') is not None)]
        if candidates:
            offer(candidates[-1].get('observation_id'),
                  'recent_route_candidate')
            break
    # Prefer a recent representative of a possible visual revisit. Camera
    # pose bins are deduped below, even when regenerated RGB hashes differ.
    order={o['observation_id']:i for i,o in enumerate(memory.observations)}
    matches=sorted(current.get('revisit_matches') or [],
                   key=lambda m:order.get(m.get('observation_id'),-1),
                   reverse=True)
    for match in matches[:3]:
        offer(match.get('observation_id'),
              'possible_cross_stance_visual_revisit')
    if memory.observations:
        offer(memory.observations[0].get('observation_id'),
              'episode_origin_reference')
    # If some slots are still empty, fill them with earlier candidate
    # scenes and diverse checkpoints. All source photos are immutable.
    for session in memory.observation_searches:
        for cp in session.get('checkpoints') or []:
            if ((cp.get('visual') or {}).get('candidate_approach')
                    and (cp.get('visual') or {}).get('passage_box') is not None):
                offer(cp.get('observation_id'),'historical_route_candidate')
                break
    for descriptor in selected_history_frames(
            memory,current_id,max_frames=max(6,max_pairs)):
        offer(descriptor['observation_id'],
              'diverse_observation_checkpoint')
    if len(memory.observations)>2:
        offer(memory.observations[len(memory.observations)//2].get(
            'observation_id'),'middle_episode_reference')

    selected=[]
    for identifier,reason in options:
        record=by_id[identifier]
        pose=record.get('pose') or {}
        # The CURRENT RGB is already attached. Older captures at the same
        # stance/yaw/pitch are not additional viewpoints, even if their
        # hashes differ slightly due to sensor refresh or a moving person.
        if _same_camera_view(record,current,memory.camera):
            continue
        if any(_same_camera_view(record,other,memory.camera)
               for other in selected):
            continue
        selected.append(dict(record,selection_reason=reason))
        if len(selected)>=max_pairs:
            break
    return selected
