"""Single source-linked context projection for entire-mission navigation.

No independent second summary of observations, waypoints, failed moves or
natural-language progress is kept here. Both agents read mission_evidence()
from the same immutable Memory records.
"""
import copy
from .mission_history import mission_evidence
from .contracts import CAMERA_PITCH_LIMIT_DEG, camera_look_options

def fields(value,names):
    return {k:copy.deepcopy(value[k]) for k in names
            if isinstance(value,dict) and k in value}

def navigation_context(runtime):
    # The controller pose is the canonical source for pitch and heading.
    # Read it once so camera_controls and mission_evidence cannot disagree,
    # and so lightweight/mock controllers need not expose duplicate fields.
    pose=runtime.motion_controller.pose()
    pitch_deg=pose.get('pitch_deg',0.0)
    evidence=mission_evidence(runtime.memory,pose,
                              max_observation_tasks=3)
    return dict(
        memory=evidence,
        actuator=dict(
            forward_step_m=runtime.camera.forward_step,
            turn_step_deg=runtime.camera.turn_angle,
            max_direct_segment_m=3.0),
        camera_controls=dict(
            current_pitch_deg=pitch_deg,
            pitch_limits_deg=[-CAMERA_PITCH_LIMIT_DEG,CAMERA_PITCH_LIMIT_DEG],
            tilt_step_deg=runtime.camera.tilt_angle,
            allowed_look_now=camera_look_options(
                pitch_deg,runtime.camera.tilt_angle),
            tool_look_enum_stable=['none','up','down'],
            pitch_limit_is_recoverable_tool_feedback=True),
        actions_remaining=max(0,runtime.config['max_steps']-
                              len(runtime.memory.actions)-1),
        navigation_mode=getattr(runtime,'navigation_mode','OBSERVE'),
        liveness_diagnostic=copy.deepcopy(
            getattr(runtime,'navigation_liveness',None)),
        current_rgb_image_attached_separately=True)

def _small(source, names, limits=None):
    """Retain source-grounded evidence, but never duplicate full trace blobs."""
    if not isinstance(source, dict):
        return {}
    limits=limits or {}
    result={}
    for key in names:
        value=source.get(key)
        if value is None:
            continue
        if isinstance(value, str) and key in limits:
            value=value[:limits[key]]
        if value not in ('', [], {}):
            result[key]=copy.deepcopy(value)
    return result


def _same_current_forward_rejection(block, depth):
    """Match an *actual rejected MOVE* to the live RGB-D/pose evidence.

    An observed-only front obstacle, old stance or different camera pitch
    never becomes a failed move at the current view.
    """
    if not isinstance(depth,dict) or depth.get('status')!='BLOCKED':
        return False
    if (block.get('kind')!='BLOCKED_BEFORE_FORWARD' or
            block.get('source_stance_matches_current') is not True):
        return False
    for key in ('rgb_hash','depth_hash'):
        if (not isinstance(depth.get(key),str) or not depth[key]
                or block.get(key)!=depth[key]):
            return False
    for key in ('heading_deg','pitch_deg'):
        old,new=block.get(key),depth.get(key)
        if (isinstance(old,bool) or isinstance(new,bool) or
                not isinstance(old,(int,float)) or
                not isinstance(new,(int,float))):
            return False
        difference=((old-new+180)%360-180 if key=='heading_deg'
                    else old-new)
        if abs(difference)>.01:
            return False
    return True


def _project_navigation_memory(memory, live_depth):
    """Read-only NavigationAgent view of canonical mission_evidence.

    The full shared mission_evidence, complete sensor observations and
    retrievable Memory are UNCHANGED. This projection replaces duplicate
    request fields; it never invents progress, a route or motor constraints.
    """
    memory.pop('recent_forward_failures',None)  # exact block+stall copies
    memory.pop('bearing_convention',None)  # already in tool protocol
    memory.pop('forward_action_segments_are_not_verified_translation',None)
    memory.pop('historical_clearance_authorizes_move',None)
    memory.pop('metric_pose_available',None)

    coverage=memory.get('recent_views_at_current_stance') or {}
    if coverage:
        # The complete all-time heading/pitch bins are preserved. Only
        # redundant per-view exemplars are shortened, not the coverage.
        recent=coverage.get('latest_distinct_views') or []
        memory['recent_views_at_current_stance']={
            'scope':coverage.get('scope'),
            'sampled_observations':coverage.get('sampled_observations',0),
            'turn_actions_since_last_forward':coverage.get(
                'turn_actions_since_last_forward',0),
            'look_actions_since_last_forward':coverage.get(
                'look_actions_since_last_forward',0),
            'repeated_same_RGBD_heading_pitch_views':coverage.get(
                'repeated_same_RGBD_heading_pitch_views',0),
            'distinct_heading_pitch_views':coverage.get(
                'distinct_heading_pitch_views',0),
            'horizontal_360_coverage_observed':coverage.get(
                'horizontal_360_coverage_observed',False),
            'full_coverage_by_pitch':copy.deepcopy(
                coverage.get('full_coverage_by_pitch') or []),
            'latest_distinct_views':[
                _small(v,('source_observation_id','heading_deg','pitch_deg'))
                for v in recent[:2]],
            'coverage_is_information_not_motion':True}

    receipt=memory.get('last_motion_decision')
    if receipt:
        memory['last_motion_decision']=_small(receipt,(
            'source_observation_id','end_observation_id','status','reason',
            'last_heading_deg','executed_forward_actions',
            'forward_actions_with_RGBD_change','atomic_actions_since_motion',
            'last_motion_was_uninterrupted','motion_is_evidence_not_command'),
            {'reason':130})

    # Forward receipts prove an executed atomic command and changed imagery,
    # NEVER metric displacement or having reached a stair landing.
    segments=memory.get('observed_forward_action_segments') or []
    memory['observed_forward_action_segments']=[
        dict(_small(item,(
            'first_step','last_step','heading_deg','forward_steps',
            'start_observation_id','end_observation_id')),
            displacement_verified=False)
        for item in segments[-3:]]

    # Compact a factual long-route anchor and earlier MODEL interpretations.
    # The latest progress narrative may contradict these older assessments;
    # preserve their source IDs rather than silently replacing history.
    continuity=memory.get('route_continuity')
    if isinstance(continuity,dict):
        if continuity.get('endpoint_front_obstacle') is None:
            continuity.pop('endpoint_front_obstacle',None)
        history=continuity.get('model_assessment_snapshots') or []
        continuity['model_assessment_snapshots']=[
            _small(item,('phase','source_observation_id','action_step',
                'source_heading_deg','model_visual_description',
                'model_progress_claim','interpretation_unverified'),
                {'model_visual_description':145,'model_progress_claim':115})
            for item in history[:4]]
    else:
        memory.pop('route_continuity',None)
    looks=memory.get('recent_look_evidence')
    if isinstance(looks,dict) and looks.get('sampled_recent_look_decisions'):
        looks['latest_questions_and_model_descriptions']=[
            _small(item,('look','question','source_observation_id',
                'first_time_camera_yaw_pitch_view',
                'following_model_visual_description'),
                {'question':95,'following_model_visual_description':145})
            for item in (looks.get('latest_questions_and_model_descriptions') or [])[-2:]]
    else:
        memory.pop('recent_look_evidence',None)

    # Match the CURRENT preflight to real previously REJECTED forward
    # requests. The ledger already counts retries at the same sensor view;
    # attach one compact fact to the live preflight rather than duplicating
    # it across Memory/tool history. No new navigation state or veto.
    raw_blocks=memory.get('blocked_at_source_stances') or []
    matching=[item for item in raw_blocks
              if _same_current_forward_rejection(item,live_depth)]
    if matching:
        count=max(int(item.get('repeated_same_frame_heading_attempts') or 0)
                  for item in matching)
        if count:
            latest=matching[0]
            live_depth['previous_identical_view_forward_rejections']=count
            live_depth['previous_rejected_forward_actions_executed']=0
            live_depth['previous_rejection_observation_id']=latest.get(
                'source_observation_id')
            live_depth['rejections_match_current_RGBD_stance_yaw_pitch']=True

    # Historical physical blocks, unverified post-move frame stalls and
    # pre-decision Depth blocks have DIFFERENT semantics. Keep those
    # distinctions, drop duplicate hashes, wide nested geometry and flags.
    fields=('kind','source_observation_id','after_observation_id',
            'heading_deg','pitch_deg','source_stance_matches_current',
            'repeated_same_frame_heading_attempts','central_support_pixels',
            'nearest_along_m','focus_region_xyxy_norm','last_action_step',
            'reason')
    memory['blocked_at_source_stances']=[
        _small(item,fields,{'reason':125})
        for item in raw_blocks
        if not _same_current_forward_rejection(item,live_depth)
    ][:2]
    memory['stalled_forward_at_source_stances']=[
        _small(item,fields,{'reason':125})
        for item in (memory.get('stalled_forward_at_source_stances') or [])[:2]]
    blocks=memory.get('observed_front_blocks') or []
    current_source=(live_depth or {}).get('source_observation_id')
    endpoint_source=((continuity or {}).get('endpoint_front_obstacle') or {}).get(
        'source_observation_id')
    memory['observed_front_blocks']=[
        dict(_small(item,('kind','source_observation_id',
                'heading_deg','source_stance_matches_current',
                'repeat_observation_count')),
            blocker=_small(item.get('blocker') or {},(
                'central_support_pixels','nearest_along_m',
                'focus_region_xyxy_norm')))
        for item in blocks
        if item.get('source_observation_id') not in (current_source,endpoint_source)
    ][:2]

    # Completed turn_and_observe receipts are already represented in the
    # stance coverage. Finished forward receipts are represented above.
    # Only keep exceptional feedback that can change the next decision.
    receipts=memory.get('recent_tool_receipts') or []
    exceptions=[]
    for item in reversed(receipts):
        status=item.get('status')
        if status not in ('CAMERA_PITCH_LIMIT','INVALID_ARGUMENTS',
                          'REQUIRES_ATTENTION','PARTIAL','BLOCKED',
                          'INVALID_TARGET'):
            continue
        if status=='BLOCKED' and item.get('end_observation_id') in {
                b.get('source_observation_id')
                for b in memory['blocked_at_source_stances']}:
            continue
        exceptions.append(_small(item,(
            'tool_call_id','tool','status','reason','end_observation_id',
            'executed_actions'),{'reason':160}))
        if len(exceptions)==2:
            break
    memory['recent_tool_receipts']=list(reversed(exceptions))
    memory['recent_camera_pitch_limits']=[
        _small(item,('current_pitch_deg','rejected_look','allowed_look',
                     'repeated_same_invalid_look'))
        for item in (memory.get('recent_camera_pitch_limits') or [])[:1]]

    # The visual task's full checkpoints, boxes and all original narrative
    # remain in Memory. Only a short latest-result summary is presented
    # in navigation. A source-grounded passage is also in route candidates.
    tasks=memory.get('observation_tasks') or []
    # Long prose truncation may hide the decisive end-of-sentence VLM
    # finding (e.g. "no reader identifiable"). Keep the typed target
    # verdicts from each REAL RGB view, independently of the short summary.
    # False is NOT evidence the person is absent: only not visible here.
    memory['observation_tasks']=[
        dict(_small(item,(
            'observation_task_id','objective','outcome',
            'end_observation_id','observation_summary',
            'navigation_suggestion'),
            {'objective':180,'observation_summary':220,
             'navigation_suggestion':240}),
            target_visibility_checks=[
                _small(view,('observation_id','target_type',
                             'target_visible','confidence'))
                for view in (item.get('selected_views') or [])[-3:]
                if view.get('target_type') is not None
                and isinstance(view.get('target_visible'),bool)],
            target_not_visible_is_not_proven_absent=True)
        for item in tasks[-2:]]
    routes=memory.get('candidate_route_hypotheses') or []
    memory['candidate_route_hypotheses']=[
        _small(item,(
            'candidate_id','source_observation_id','target_type',
            'target_kind','confidence','route_status',
            'bearing_region_role','source_stance_matches_current',
            'relative_bearing_deg_from_current','relative_side_from_current',
            'relative_turn_deg_from_current',
            'historical_bearing_in_source_record_only',
            'evidence','navigation_suggestion','uncertainty'),
            {'evidence':155,'navigation_suggestion':210,'uncertainty':120})
        for item in routes[:3]]
    references=memory.get('instruction_route_references') or []
    memory['instruction_route_references']=[
        dict(_small(item,('reference_id','source_observation_id',
            'source_heading_deg','source_stance_id','approach_heading_deg',
            'approach_observation_id',
            'approach_direction_is_historical_executed_forward_only',
            'place_description','instruction_relation','evidence_status',
            'reference_is_historical_view_not_current_motor_bearing',
            'completion_verified'),
            {'place_description':140,'instruction_relation':220}),
            route_attempts=[
                _small(a,('evidence_observation_id','assessment','evidence_scope'),
                       {'assessment':175})
                for a in (item.get('route_attempts') or [])[-1:]])
        for item in references[:2]]
    if not references:
        memory.pop('instruction_route_references',None)
    history=memory.get('historical_frames') or []
    memory['historical_frames']=[
        _small(item,('observation_id','heading_deg','pitch_deg',
                     'target_type','observation_summary'),
               {'observation_summary':130})
        for item in history[-3:]]

    progress=memory.get('progress') or {}
    if progress:
        progress['uncertainties']=(progress.get('uncertainties') or [])[-3:]
        progress['source_observation_ids']=(
            progress.get('source_observation_ids') or [])[-6:]
        if isinstance(progress.get('narrative'),str):
            progress['narrative']=progress['narrative'][:450]
        prior=progress.get('prior_navigation_intent') or {}
        if isinstance(prior,dict):
            keys=('source','relative_direction_valid_for_current_camera',
                  'historical_only')
            # Once a proper junction reference exists, don't keep repeating
            # a stale short-term camera instruction such as "align with stairs".
            # Keep its source and historical marker for audit.
            if not (references and prior.get('historical_only')):
                keys=('text',)+keys
            progress['prior_navigation_intent']=_small(
                prior,keys,{'text':240})
    return memory


def compact_navigation_context(state):
    """Short NavigationAgent request: canonical memory stays fully available.

    No second persistent summary, no arbitrary overall-string truncation,
    and no new intervention or action validator. Current RGB/mission/Depth
    remain authoritative; all historical records are still retrievable.
    """
    result=copy.deepcopy(state)
    memory=result.get('memory')
    if isinstance(memory,dict):
        _project_navigation_memory(
            memory,result.get('front_depth_preflight'))
    # Actionless liveness is a resource diagnostic, not proof that
    # ordinary turns were progress; omit repeated zero-valued reminders.
    if not (result.get('liveness_diagnostic') or {}).get(
            'consecutive_unchanged_decisions'):
        result.pop('liveness_diagnostic',None)
    return result

def planner_context(runtime):
    return dict(instruction=runtime.memory.task.instruction)
