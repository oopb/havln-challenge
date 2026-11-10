"""Bounded, source-linked navigation history derived from existing Memory.

No new persistent state, odometry, scene semantics or movement constraints.
The model decides how to interpret competing visual hypotheses.
"""
import math
from .heading import same_local_stance


def _yaw(pose):
    value=(pose or {}).get('heading')
    if isinstance(value,(int,float)) and math.isfinite(value):
        return math.degrees(value)%360
    return None


def _difference(a,b):
    return abs((a-b+180)%360-180)


def _excerpt(text,limit=165):
    return str(text or '').strip()[:limit]


def salient_forward_segment(segments, current_heading_deg, *, min_forward=2):
    """One useful old motion anchor, not an inferred return or completion.

    Prefer a substantially observed prior forward segment in a direction
    opposing the current camera, then a long visual-change segment. Source
    frames and execution counts are facts; spatial coincidence is unknown.
    """
    choices=[s for s in segments if s.get('forward_steps',0)>=min_forward
             and isinstance(s.get('heading_deg'),(int,float))
             and s.get('start_observation_id') and s.get('end_observation_id')]
    if not choices:
        return None
    def rank(s):
        delta=_difference(s['heading_deg'],current_heading_deg)
        return (min(s['forward_steps'],12),int(delta>=135),
                -s.get('first_step',0))
    return max(choices,key=rank)


def _assessment(turn, phase):
    if not turn:
        return None
    decision=turn.get('decision') or {}
    source=turn.get('source_observation_id')
    observations=decision.get('observed_evidence') or []
    if not source or not isinstance(observations,list):
        return None
    observed=next((s for s in observations if isinstance(s,str) and s.strip()),'')
    return dict(phase=phase,source_observation_id=source,
        action_step=turn.get('step_started'),
        source_heading_deg=turn.get('source_heading_deg'),
        model_visual_description=_excerpt(observed,175),
        model_progress_claim=_excerpt(decision.get('progress_narrative'),155),
        interpretation_unverified=True)


def _endpoint_obstacle(memory, segment):
    """Link an actual later front block to the end of a finished MOVE span.

    A pre-decision BLOCKED view is NOT a rejected past MOVE, and its source
    bearing/stance must agree with the completed segment endpoint.
    """
    records={o['observation_id']: (i,o)
             for i,o in enumerate(memory.observations)}
    finish=records.get(segment.get('end_observation_id'))
    if not finish:
        return None
    index,end=finish
    end_pose=end.get('pose') or {}
    yaw=segment.get('heading_deg')
    limit=max(.1,float(memory.camera.turn_angle)/2)
    for kind,items in (
            ('FRONT_BLOCK_OBSERVED_NO_FORWARD_ATTEMPT',
             memory.evidence.predecision_front_obstacles),
            ('NEXT_FORWARD_REJECTED',memory.evidence.obstacle_blocks)):
        for item in items:
            reference=records.get(item.get('observation_id'))
            if not reference or reference[0]<index:
                continue
            source=reference[1]
            bearing=item.get('heading_deg')
            if (not isinstance(bearing,(int,float)) or
                    not isinstance(yaw,(int,float))
                    or _difference(bearing,yaw)>limit
                    or not same_local_stance(source.get('pose'),end_pose)):
                continue
            return dict(
                kind=kind,source_observation_id=item['observation_id'],
                heading_deg=round(bearing,1),
                pitch_deg=(source.get('pose') or {}).get('pitch_deg'),
                same_executed_forward_stance_as_segment_end=True,
                measured_at_segment_endpoint_after_executed_forwards=True,
                attempted_subsequent_forward=kind=='NEXT_FORWARD_REJECTED',
                previously_executed_forwards_not_rejected=True,
                measured_block_does_not_prove_earlier_route_failed=True)
    return None


def route_continuity(memory, pose, segments):
    """Pair executed forward evidence with older, fallible model assessments.

    The earliest important route segment stays retrievable after the latest
    three forward segments have cycled out of the short context. No sign of
    ascent/descent is inferred from commanded motion or camera height.
    """
    heading=_yaw(pose)
    if heading is None:
        return None
    anchor=salient_forward_segment(segments,heading)
    if not anchor:
        return None
    anchor_heading=anchor['heading_deg']
    difference=round(_difference(anchor_heading,heading),1)
    recent=segments[-1] if segments else None
    turns=memory.navigation_turns
    start_turn=next((t for t in turns
        if t.get('source_observation_id')==anchor['start_observation_id']),None)
    end_turn=next((t for t in turns
        if t.get('source_observation_id')==anchor['end_observation_id']),None)
    # Include two early independent assessments looking approximately back
    # along the selected motion heading. They may disagree, and MUST never
    # be auto-reconciled into a verified task stage.
    opposite=[]
    for turn in turns:
        source_heading=turn.get('source_heading_deg')
        if (not isinstance(source_heading,(int,float))
                or turn.get('step_started',0)<=anchor.get('last_step',0)
                or _difference(source_heading,anchor_heading)<150):
            continue
        if turn.get('source_observation_id') in (
                anchor['start_observation_id'],anchor['end_observation_id']):
            continue
        if (turn.get('decision') or {}).get('observed_evidence'):
            opposite.append(turn)
        if len(opposite)>=2:
            break
    snapshots=[]
    for turn,phase in ((start_turn,'before_forward'),
                       (end_turn,'after_forward'),
                       *[(t,'opposing_view_%d'%(i+1))
                         for i,t in enumerate(opposite)]):
        record=_assessment(turn,phase)
        if record:
            snapshots.append(record)
    def segment(s):
        return dict(start_step=s.get('first_step'),
            end_step=s.get('last_step'),heading_deg=s.get('heading_deg'),
            executed_forward_with_RGBD_change=s.get('forward_steps'),
            start_observation_id=s.get('start_observation_id'),
            end_observation_id=s.get('end_observation_id'))
    return dict(
        selected_historical_forward=segment(anchor),
        latest_forward=(segment(recent) if recent is not anchor and recent else None),
        endpoint_front_obstacle=_endpoint_obstacle(memory,anchor),
        current_heading_deg=round(heading,1),
        separation_from_historical_forward_deg=difference,
        opposing_camera_heading=bool(difference>=150),
        previous_forward_and_current_view_may_be_a_retrace_not_proven=bool(difference>=150),
        model_assessment_snapshots=snapshots,
        assessment_claims_are_mutable_not_verified_progress=True,
        executed_steps_do_not_prove_stair_descent_or_metric_displacement=True)


def recent_look_evidence(memory, pose, *, window=8, max_examples=2):
    """Report repeated camera *viewpoints* and the next model visual answer.

    A different RGB hash alone is not semantic information gain. Only new
    yaw/pitch bins and authentic subsequent model descriptions are reported.
    """
    current=pose or {}
    current_epoch=(current.get('motion_epoch'),
                   current.get('viewpoint_generation'))
    obs=memory.observations
    by_id={x['observation_id']:x for x in obs}
    yaw_step=max(1.,float(memory.camera.turn_angle))
    pitch_step=max(1.,float(memory.camera.tilt_angle))
    first_view=set()
    known=set()
    for record in obs:
        p=record.get('pose') or {}
        if (p.get('motion_epoch'),p.get('viewpoint_generation'))!=current_epoch:
            continue
        y=_yaw(p)
        if y is None:
            continue
        key=(round(y/yaw_step)*yaw_step%360,
             round(p.get('pitch_deg',0)/pitch_step)*pitch_step)
        if key not in known:
            first_view.add(record['observation_id'])
        known.add(key)
    turns=memory.navigation_turns
    looks=[]
    for turn in turns:
        d=turn.get('decision') or {}
        if d.get('tool')!='turn_and_observe':
            continue
        args=d.get('args') or {}
        if args.get('look') not in ('up','down'):
            continue
        result=turn.get('result') or {}
        expected='LOOK_UP' if args['look']=='up' else 'LOOK_DOWN'
        if expected not in (result.get('executed_motion') or []):
            continue  # rejected LOOK is not a new camera observation
        after=result.get('end_observation_id')
        source=by_id.get(after)
        if not source:
            continue
        p=source.get('pose') or {}
        if (p.get('motion_epoch'),p.get('viewpoint_generation'))!=current_epoch:
            continue
        # A turn may contain several atomic yaw actions; the question is
        # grounded in the final frame used by the following model decision.
        follow=next((t for t in turns
            if t.get('source_observation_id')==after
            and t.get('turn_index',0)>turn.get('turn_index',0)),None)
        answer=(follow.get('decision') or {}).get('observed_evidence') if follow else []
        description=(next((s for s in answer if isinstance(s,str) and s.strip()),'')
                     if isinstance(answer,list) else '')
        looks.append(dict(
            look=args['look'],question=_excerpt(args.get('purpose'),95),
            source_observation_id=after,
            first_time_camera_yaw_pitch_view=after in first_view,
            following_model_visual_description=_excerpt(description,170)))
    recent=looks[-window:]
    return dict(
        scope='same_executed_forward_stance_navigation_agent_executed_looks',
        sampled_recent_look_decisions=len(recent),
        first_time_yaw_pitch_views=sum(bool(v['first_time_camera_yaw_pitch_view'])
                                      for v in recent),
        revisited_yaw_pitch_views=sum(not v['first_time_camera_yaw_pitch_view']
                                      for v in recent),
        latest_questions_and_model_descriptions=recent[-max_examples:],
        view_novelty_is_not_semantic_information_gain=True,
        model_selects_whether_to_look_or_move=True)
