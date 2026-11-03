"""One-shot panoramic vision-to-navigation decision with authentic memory frames.

The visual model receives ordered historical RGB images; the text NavigationAgent
receives the model's semantic assessment PLUS the exact relative yaw for each
image. Historical pictures never authorize current forward travel.
"""
from .api_client import APIError
from .contracts import navigation_schema, compact_navigation_protocol, navigation_envelope


PANORAMA_NAVIGATION_PROMPT = (
    'You are NavigationAgent making a ONE-TIME PANORAMIC DIRECTION DECISION. '
    'This is a DEDICATED decision, not the ordinary OBSERVE navigation loop. '
    'A separate visual model has inspected the exact saved RGB frames listed '
    'in panorama_frames[]. You receive its uncertain visual evidence and the '
    'matching angle metadata; you have NOT directly seen those images. '
    'The relative_turn_deg_from_current values are signed physical robot yaw: '
    'positive LEFT, negative RIGHT, always from CURRENT heading. '
    'The pitch_deg values are historical CAMERA TILT, not robot turn commands. '
    'Use the current task, visual evidence, navigation history, previous failures '
    'and frame-specific absolute/relative headings to CHOOSE the most plausible '
    'near-term approach direction. Select your own bearing and travel distance. '
    'Submit only one navigate_relative(turn_deg, distance_m) with '
    '-180 <= turn_deg <= 180 and 0.25 <= distance_m <= 3.0. '
    'The executor will perform all needed turns then attempt forward and '
    'check fresh RGB-D depth, terrain and people BEFORE every forward action. '
    'Past EXECUTABLE does not mean currently safe; past BLOCKED might be stale. '
    'A visual opening is a hypothesis, NOT free-space authorization. '
    'Do not invent completion or guarantee a forward will execute. '
    'Do not choose more camera-only scans at this checkpoint. '
    'No hardcoded stair or door bearing: YOU choose the direction based on '
    'the real remembered views. Return ONE schema-valid JSON tool call only. '
)


def select_saved_viewpoints(panorama, max_frames=12):
    """Choose up to twelve diverse observed yaw directions; no fake frames."""
    candidates=list(panorama.get('observed_viewpoints') or ())
    selected=[]
    used=set()
    for offset in range(0,360,max(1,round(360/max_frames))):
        available=[v for v in candidates if v['observation_id'] not in used]
        if not available:
            break
        def score(view):
            delta=(float(view['relative_turn_deg_from_current'])-offset+180)%360-180
            # Favor coverage first, horizontal views in an angular tie.
            return (abs(delta),abs(float(view.get('pitch_deg') or 0)))
        best=min(available,key=score)
        if score(best)[0]>45:
            continue
        used.add(best['observation_id'])
        selected.append(best)
        if len(selected)>=max_frames:
            break
    return selected


def decide_with_panorama(agent, context, signature, validate, offered_tools):
    """VLM looks at saved images; NavigationAgent makes its own final MOVE call."""
    runtime=agent.runtime
    panorama=context['panoramic_observation']
    selected=select_saved_viewpoints(panorama)
    if not selected:
        raise APIError('panorama_has_no_authentic_saved_views',
                       category='structured_field')
    current_heading=float(context['pose']['heading_deg'])
    views=[]
    frames=[]
    for view in selected:
        observation_id=view['observation_id']
        try:
            observation,source=runtime._vision_frame(observation_id)
        except (ValueError, KeyError, OSError) as error:
            raise APIError('panorama_memory_RGB_missing_for_'+observation_id,
                           category='structured_field') from error
        if source['observation_id']!=observation_id:
            raise APIError('panorama_memory_frame_id_mismatch',
                           category='structured_field')
        relative=(float(view['absolute_heading_deg'])-current_heading+180)%360-180
        entry=dict(
            observation_id=observation_id,
            absolute_heading_deg=view['absolute_heading_deg'],
            relative_turn_deg_from_current=round(relative,2),
            pitch_deg=view['pitch_deg'],
            historical_front_state=view['historical_front_state'],
            historical_front_reason=view.get('historical_front_reason'),
            historical_front_clearance_m=view.get('historical_front_clearance_m'))
        frames.append((observation,entry))
        views.append(entry)
    key=(signature,current_heading,tuple(v['observation_id'] for v in views))
    cached=getattr(agent,'_panorama_visual_cache',None)
    if cached is not None and cached['key']==key:
        visual=cached['visual']
    else:
        visual=runtime.visual_perception.panorama_direction_evidence(
            frames,dict(
                task=dict(instruction=runtime.memory.task.instruction,
                          active_subgoal=context.get('goal'),
                          selection=context.get('task_selection')),
                current_heading_deg=current_heading,
                navigation_memory=context.get('navigation_memory'),
                recent_route_failures=context.get('recent_navigation_block')))
        agent._panorama_visual_cache=dict(key=key,visual=visual)
        if hasattr(runtime,'event'):
            runtime.event('panorama_visual_memory_reviewed',
                source='visual_perception',
                historical_observation_ids=[v['observation_id'] for v in views],
                relative_turn_degrees=[v['relative_turn_deg_from_current'] for v in views],
                selected_observation_id=visual['selected_observation_id'],
                visual_only=True)
    # A deliberately compact prompt/context, independent of OBSERVE_PROMPT.
    # Image metadata and the VLM hypothesis share authentic observation IDs.
    nav_context=dict(
        instruction=runtime.memory.task.instruction,
        current_goal=context.get('goal'),
        task_selection=context.get('task_selection'),
        pose=context.get('pose'),
        navigation_memory=context.get('navigation_memory'),
        recent_navigation_block=context.get('recent_navigation_block'),
        recent_motion_intent=context.get('recent_motion_intent'),
        prior_motion_evidence=context.get('historical_motion_evidence'),
        local_geometry_scope=context.get('local_geometry_scope'),
        action_calibration=context.get('action_calibration'),
        panorama_frames=views,
        panorama_visual_evidence=visual,
        panorama_commitment=dict(
            required_decision='SELECT_BEARING_AND_ATTEMPT_FINITE_NAVIGATION',
            allowed_tool='navigate_relative',
            images_reviewed_by='visual_perception',
            frame_yaws_relative_to_current_pose=True,
            historical_RGB_not_motion_permission=True),
        **({'format_feedback':context['format_feedback']}
           if 'format_feedback' in context else {}))
    response=runtime.client.complete(
        'navigation_agent','panorama_navigation_commit',
        PANORAMA_NAVIGATION_PROMPT,
        nav_context,navigation_schema(offered_tools),
        validate=validate,
        format_retry=int('format_feedback' in context),
        retry_safe=True,
        protocol=compact_navigation_protocol(offered_tools),
        validation_schema=navigation_envelope(offered_tools))
    return response,visual,views
