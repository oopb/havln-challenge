"""One visual question per request; semantic reuse is keyed by image and question."""
import copy
import json
import math

from .api_client import TEXT,object_schema,APIError,BudgetExceeded
from .grounding import ground
from .perception import Geometry,image_url,depth_image_url,perceive,perceive_views
from .contexts import fields
from .mission_history import mission_evidence
from .navigation_history import navigation_visual_anchors
from .observation import array_hash
from .model_angles import project_model_angles, project_observation_conversation

# A one-frame, geometry-directed visual description. No model-created pixel box,
# trajectory, clearance decision or task-completion verdict is accepted.
OBSTACLE_VISUAL_SCHEMA = object_schema(dict(
    visible=dict(type='boolean'),
    description=dict(type='string',maxLength=240),
    visual_relation=dict(type='string',maxLength=240),
    uncertainty=dict(type='string',maxLength=240)))


ENTRY_SCHEMA=object_schema(dict(stairs=dict(enum=['visible','possible','unknown','absent']),
    entry_visible=dict(type='boolean'),candidate_bearing=dict(enum=['left','front','right','unknown']),
    suggested_observation=TEXT,uncertainty=TEXT))



# VLM decides whether a goal-specific VISUAL candidate was found, never how
# the robot moves. A visible box is mandatory for an actionable candidate.
SEARCH_SCHEMA = object_schema(dict(
    target_visible=dict(type='boolean'),
    target_type=dict(type='string',minLength=1,maxLength=80),
    confidence=dict(enum=['low','medium','high']),
    box=dict(anyOf=[dict(type='array',minItems=4,maxItems=4,
        items=dict(type='number',minimum=0,maximum=1)),dict(type='null')]),
    evidence=TEXT,
    uncertainty=TEXT,
    suggested_next_view=dict(enum=['continue','look_up','look_down'])))


# The VLM owns the camera-level search policy. Its outputs are visual
# hypotheses plus the next real TURN/LOOK request, never a navigation MOVE.
SCAN_GUIDANCE_SCHEMA = object_schema(dict(
    target_visible=dict(type='boolean'),
    target_type=dict(type='string',minLength=1,maxLength=80),
    confidence=dict(enum=['low','medium','high']),
    box=dict(anyOf=[dict(type='array',minItems=4,maxItems=4,
        items=dict(type='number',minimum=0,maximum=1)),dict(type='null')]),
    evidence=dict(type='string',maxLength=1200),
    uncertainty=dict(type='string',maxLength=300),
    target_kind=dict(enum=['ENTRY','APPROACH','TARGET','CUE','UNKNOWN']),
    candidate_approach=dict(type='boolean'),
    # Object detection box (box) and the optional visually continuous
    # passage REGION must never share a bearing. Old answers may omit this.
    passage_box=dict(anyOf=[dict(type='array',minItems=4,maxItems=4,
        items=dict(type='number',minimum=0,maximum=1)),dict(type='null')]),
    next_action=dict(enum=['turn_left','turn_right','look_up','look_down','finish']),
    turn_deg=dict(enum=[0,15,30,45,60,90]),
    reason=dict(type='string',maxLength=250),
    expected_progress=dict(enum=['ADVANCE','REVERSE','DETOUR','UNKNOWN'])))
# An exploratory movement hypothesis is optional; old model outputs remain
# valid and are interpreted as UNKNOWN rather than fabricating intent.
SCAN_GUIDANCE_SCHEMA['required'].remove('expected_progress')

# The VLM confirms semantic outcomes from real observations, not from a
# Python stair/landing predicate or a claimed movement distance.
_ADVICE_FIELDS=dict(
    observation_summary=dict(type='string',maxLength=750),
    navigation_suggestion=dict(type='string',maxLength=600),
    distance_m_hint=dict(anyOf=[
        dict(type='number',minimum=0.25,maximum=3.0),dict(type='null')]))
SCAN_GUIDANCE_SCHEMA['properties'].update(copy.deepcopy(_ADVICE_FIELDS))
# The VLM also produces its camera choice last. Do not lead a model with
# next_action/turn_deg before it has reported visual facts and its reason.
_VLM_OUTPUT_ORDER=(
    'observation_summary','evidence','target_visible','target_type',
    'target_kind','confidence','box','passage_box','uncertainty',
    'navigation_suggestion','reason','expected_progress','distance_m_hint',
    'candidate_approach','next_action','turn_deg')
SCAN_GUIDANCE_SCHEMA['properties']={
    key:SCAN_GUIDANCE_SCHEMA['properties'][key] for key in _VLM_OUTPUT_ORDER}
SCAN_GUIDANCE_SCHEMA['required']=[
    key for key in _VLM_OUTPUT_ORDER if key in SCAN_GUIDANCE_SCHEMA['required']]

SCAN_GUIDANCE_PROMPT=(
    'You are Observation Agent helping NavigationAgent execute ONE entire '
    'natural-language mission. You receive the ORIGINAL instruction, '
    'revisable narrative progress and the NavigationAgent\'s current '
    'natural-language observation request; NEVER divide them into numbered '
    'subgoals or decide whether a milestone is officially complete. '
    'You inspect the current REAL RGB image and calibrated Depth image '
    '(near black, far white, invalid magenta), plus the FULL dialogue of '
    'all earlier real RGB-D frames in this observation session. Reconcile '
    'views by signed camera offsets relative to the CURRENT view and pitch; do not hallucinate '
    'translation or global localization from a rotation. '
    'The shared mission uses commanded yaw/pitch and conservative '
    'viewpoint_generation stance IDs, NOT a measured robot x/y/z. '
    'TURN/LOOK retain one stance ID; each actually executed forward '
    'starts a new one, so old views are historical hypotheses only. '
    'Do not claim meters of robot motion or completed descent from '
    'the number of forward actions; judge visible changes only. '
    'Report an observation_summary and navigation_suggestion in natural '
    'language. If a useful local direction is visible, optionally mark '
    'candidate_approach=true and give distance_m_hint (0.25..3m) only when '
    'there is a reasoned, locally measured distance; otherwise omit it. '
    'The box field identifies the observed OBJECT (possibly a person, '
    'wall or railing), never automatically a route. Provide passage_box '
    'ONLY when you can actually see a distinct, continuous FLOOR/ENTRY '
    'region the robot might approach; it must cover the possible free '
    'passage, NOT the person/rail/wall being avoided. If no passage '
    'region is visually localized, use passage_box=null or omit it and '
    'provide natural-language navigation_suggestion; a candidate '
    'without passage_box has NO numerical navigation bearing. '
    'A passage_box is still only visual evidence, not proof of space '
    'for the robot, collision clearance or a physically reachable path. '
    'The horizontal center of passage_box supplies only a source-camera '
    'bearing. Do not confuse historical view orientation with live turn goals, off-center '
    'image regions or past positions. Its metric distance is a hypothesis, not walkability '
    'certificates; walls, glass, rails, drop-offs and people may intervene. '
    'Use view_coverage.session_turn_degrees, turn_direction_reversals '
    'and inspected_viewpoints together with mission_memory.'
    'recent_views_at_current_stance to avoid spending the observation budget '
    'revisiting the same measured yaw/pitch at the same stance. '
    'History from another stance or motion epoch is only a visual clue. '
    'When a likely useful direction or localized approach is visible, '
    'return its evidence and navigation_suggestion, then prefer '
    'next_action=finish so NavigationAgent can decide whether to move. '
    'Do not automatically scan the full requested sweep; the sweep is a '
    'maximum allowance, not a required rotation. Conversely, inspect a '
    'new bearing whenever actual route evidence is still missing. '
    'If a turn simply reverses earlier observation with no new visual '
    'question or obstruction evidence, prefer finish over oscillation. '
    'A horizontal yaw sweep at one pitch does not reveal the same yaw at '
    'other pitches. If near-floor clearance, stair treads, or the edge of '
    'a passage is unclear, consider LOOK_DOWN; if the camera points down '
    'and distant corridor, doorway or landmark context is missing, '
    'consider LOOK_UP. Use a pitch change to answer a SPECIFIC unanswered '
    'visual question; when the existing yaw/pitch views already answer it, '
    'report the evidence and finish the observation rather than repeat '
    'LOOK_UP/LOOK_DOWN solely for certainty. Read mission_memory.'
    'route_continuity when present to compare source-linked prior forward '
    'frames with current route hypotheses. Older conflicting model visual '
    'descriptions are NOT verified positions or task stages. '
    'Avoid repeating unchanged views; when helpful choose the next camera '
    'TURN/LOOK, otherwise finish. You may control stationary TURN_LEFT, '
    'TURN_RIGHT, LOOK_UP, LOOK_DOWN or finish; NEVER MOVE or STOP. '
    'Positive yaw is left, negative yaw right; calibrated turns are '
    '15..90 degree multiples of actuator step, LOOK/finish uses 0 degrees. '
    'Identify stairs, rooms, crowds and landmarks from actual pixels '
    'without using hardcoded stair-specific completion rules. '
    'For a doorway, stairs or a moving person, locate the visually '
    'continuous FLOOR/ENTRY corridor in passage_box rather than '
    'the center of a large box containing handrails, legs or walls. '
    'When the object is a person blocking the way, box should locate '
    'the PERSON, passage_box should locate a separate possible GAP; '
    'if gap visibility/width is unclear leave passage_box=null. Compare '
    'visible passages to the left and right of a blocker, note which '
    'side is occluded, and identify evidence needed for another look. '
    'A broad object box never certifies a navigable centerline; keep '
    'route hypotheses and live human/Depth clearance separate. '
    'Use mission_memory as the shared evidence index: previous observation '
    'task IDs, camera-relative viewpoint offsets, and blocked source frames. '
    'For a recent central-depth block, read current_front_depth_preflight '
    'and recent_forward_failures. When obstacle_focus is present, answer '
    'its specific visual question using the CURRENT RGB-D, distinguishing '
    'the measured near-field region from any passage BEYOND it. The local '
    'depth blocker is not proof that every alternative route is blocked. '
    'Do not assert that a measured occupied forward step is safe. '
    'Use the measured blocker '
    'focus_region_xyxy_norm / source_pixel_xy to inspect the corresponding '
    'REAL RGB region. State whether it looks like wall, rail, human or '
    'uncertain; do not invent metric clearance. Distinguish a pre-forward '
    'Depth BLOCKED from STALLED_AFTER_FORWARD (physical move attempted '
    'but RGB-D stayed unchanged, not proof of collision). '
    'Suggest DIFFERENT visually grounded bearings after repeated failures. '
    'Do not relabel a measured wall as passable solely to satisfy the task. '
    'Keep uncertainty explicit, and distinguish sensed obstacles from '
    'historical model guesses. NavigationAgent owns all movement and final '
    'mission completion. '
    'Generate the JSON in this order: describe observed RGB-D evidence '
    'and target facts FIRST (observation_summary, evidence, target fields, '
    'box, passage_box, uncertainty); then give the short visual interpretation and '
    'navigation suggestion (navigation_suggestion, reason, '
    'expected_progress); finally set candidate_approach and choose '
    'next_action plus turn_deg LAST. Camera choice must follow the '
    'observations instead of preceding them. '

    'Return ONE JSON object with EACH field name occurring exactly once; '
    'do not append a correction by repeating target_visible, uncertainty, '
    'or any other field. Do not emit explanatory text outside JSON.'
)

def semantic_cache_context(value):
    if isinstance(value,list):return [semantic_cache_context(v) for v in value]
    if not isinstance(value,dict):return value
    return {key:semantic_cache_context(v) for key,v in value.items()
            if key not in ('observation_id','captured_at','pose','episode_id')}


PANORAMA_VISUAL_SCHEMA = object_schema(dict(
    selected_observation_id=dict(type='string', minLength=1),
    visible_evidence=TEXT,
    uncertainty=TEXT,
    visual_cues=dict(type='array', minItems=1, maxItems=5,
                     items=dict(type='string', minLength=1, maxLength=250)),
    alternative_observation_ids=dict(type='array', maxItems=3,
                                     uniqueItems=True, items=dict(type='string', minLength=1)),
    target_status=dict(enum=['VISIBLE','CUE_ONLY','AMBIGUOUS','NOT_FOUND']),
    target_region=dict(anyOf=[dict(type='array',minItems=4,maxItems=4,
        items=dict(type='number',minimum=0,maximum=1)),dict(type='null')]),
    approach_candidates=dict(type='array',maxItems=3,items=object_schema(dict(
        observation_id=dict(type='string',minLength=1),
        kind=dict(enum=['ENTRY','APPROACH','DETOUR','UNKNOWN']),
        confidence=dict(enum=['low','medium','high']),
        evidence=TEXT))),
    missing_evidence=TEXT))

# Separate from routine scene analysis: compare *recorded* azimuth images
# without mistaking apparent empty floor for verified robot clearance.
PANORAMA_VISUAL_PROMPT = (
    'PANORAMIC VISUAL MEMORY REVIEW. You see saved RGB images captured '
    'at the same estimated robot stance while it turned through a full circle. '
    'The supplied views[] entries align ONE-TO-ONE to the images IN ORDER. '
    'Each view identifies observation_id, image_index, signed view offset '
    'from the robot CURRENT forward=0 (positive LEFT, negative RIGHT), pitch and historical '
    'front depth state (UNMEASURED when geometry was not explicitly requested). '
    'Read those camera-relative offsets; do NOT infer them from image '
    'appearance and do not swap left/right or reinterpret pitch as yaw. '
    'Use the current navigation instruction and failures to identify WHICH '
    'visible corridor, stair entry, door or approach region is the best next '
    'short MOVEMENT HYPOTHESIS. Distinguish the VISIBLE GOAL (for example a '
    'railing or lower floor) from an ACTUAL APPROACH/ENTRY, which may be at '
    'another bearing. Identify target_status and a normalized target_region '
    'only when localized in an image; otherwise use null. Compare remembered '
    'successful forward/downward motion headings with visual approach clues. '
    'Return 0-3 approach_candidates whose observation_ids are real supplied '
    'frames, with ENTRY/APPROACH/DETOUR/UNKNOWN, confidence and concrete evidence. '
    'Do not claim an entry is known merely because a staircase is visible. '
    'Even if no view proves an entry, choose EXACTLY ONE real observation_id '
    'as an information/approach hypothesis and mark AMBIGUOUS or NOT_FOUND; '
    'describe missing_evidence (including needed look_down when appropriate). '
    'Missing depth, UNKNOWN and NEEDS_VIEW are uncertainty, NOT observed '
    'obstacles; only clear current obstacle evidence can veto a real move. '
    'Historical views are semantic hints, not current collision certificates. '
    'You are the eyes, not the motion executor. Return ONE JSON object.'
)


def session_frame_ids(messages):
    """Get source-backed frame IDs from well-formed task-local user turns.

    Do not rely on substring matches: target descriptions and VLM responses
    can incidentally contain other observation IDs.
    """
    current=set()
    historical=set()
    for msg in messages or ():
        if not isinstance(msg,dict) or msg.get('role')!='user':
            continue
        for part in msg.get('content',()):
            if not isinstance(part,dict) or part.get('type')!='text':
                continue
            try:
                request=json.loads(part['text'])
            except (ValueError,TypeError,KeyError):
                continue
            if not isinstance(request,dict):
                continue
            current_id=request.get('source_observation_id',
                                   request.get('observation_id'))
            if isinstance(current_id,str):
                current.add(current_id)
            for frame in (request.get('historical_images')
                          or request.get('historical_keyframes') or ()):
                if isinstance(frame,dict) and isinstance(
                        frame.get('observation_id'),str):
                    historical.add(frame['observation_id'])
    return current,historical


class VisualPerception:
    def __init__(self,runtime):
        self.runtime=runtime

    def complete(self,*args,**kwargs):
        # The actual model sees signed offsets from CURRENT camera yaw.
        # The ORIGINAL ObservationController session stays unchanged; each
        # stateless wire request rebases the complete raw history again.
        original=list(args)
        controller=getattr(self.runtime,'motion_controller',None)
        pose=(controller.pose() if controller is not None
              and callable(getattr(controller,'pose',None)) else None)
        heading=(pose or {}).get('heading')
        if isinstance(heading,(int,float)) and math.isfinite(heading):
            yaw=math.degrees(heading)%360
            original[3]=project_model_angles(args[3],yaw)
            if kwargs.get('conversation_messages'):
                kwargs['conversation_messages']=project_observation_conversation(
                    kwargs['conversation_messages'],yaw)
        if kwargs.get('cache'):
            kwargs['cache_context']=semantic_cache_context(original[3])
        args=tuple(original)  # FIRST request must also use converted angles.
        for attempt in range(2):
            try:
                return self.runtime.client.complete(*args,retry_safe=True,format_retry=attempt,**kwargs)
            except BudgetExceeded:
                raise
            except APIError as error:
                if attempt or error.category not in ('json_syntax','structured_field','json_schema'):
                    raise
                detail=getattr(error,'request_context',{}).get('error_detail',str(error))
                retry=list(original)
                retry[3]=dict(original[3],format_feedback=dict(
                    error=detail[:350],
                    instruction=(
                        'Regenerate ONE complete schema-valid JSON object '
                        'for the SAME visible RGB-D frame. Write each key '
                        'EXACTLY ONCE. If the previous error names a '
                        'duplicate field, include one intended value '
                        'for that field and preserve all other required '
                        'fields; never append corrections as repeated keys. '
                        'Follow the current allowed_camera_actions and '
                        'allowed_turn_deg. No additional simulator action '
                        'is taken on this format retry.')))
                args=tuple(retry)

    def panorama_direction_evidence(self, frames, task_context):
        """Select a visually promising saved view, not a collision certificate.

        Frames are (real Observation, metadata) pairs, ordered exactly as
        sent in the multimodal API request. No synthetic or guessed image is
        ever presented. The IDs/relative yaws are computed by the harness.
        """
        if not frames:
            raise APIError('panorama_historical_frames_unavailable',
                           category='structured_field')
        views=[dict(info, image_index=i) for i, (_,info) in enumerate(frames)]
        identifiers={v['observation_id'] for v in views}
        if len(identifiers)!=len(views):
            raise APIError('panorama_duplicate_observation_id',
                           category='structured_field')
        context=dict(mode='panorama_visual_memory',
                     task=task_context['task'],
                     current_heading_deg=task_context['current_heading_deg'],
                     recent_route_failures=task_context.get('recent_route_failures'),
                     current_navigation_memory=task_context.get('navigation_memory'),
                     views=views,
                     all_images_are_historical=True,
                     live_motion_clearance=False)
        def validate(value):
            if (value['selected_observation_id'] not in identifiers
                    or any(x not in identifiers
                        for x in value['alternative_observation_ids'])
                    or any(x['observation_id'] not in identifiers
                        for x in value.get('approach_candidates',()))):
                raise APIError('panorama_selected_id_must_match_provided_images',
                               category='structured_field')
        return self.complete(
            'visual_perception','panorama_direction_evidence',
            PANORAMA_VISUAL_PROMPT,context,PANORAMA_VISUAL_SCHEMA,
            image_url=[image_url(obs,max_edge=640) for obs,_ in frames],
            cache=False,validate=validate)

    def scene(self,observation,source,question):
        task=self.runtime.memory.task
        context=dict(mode='scene',question=question,
            full_instruction=task.instruction,
            progress_narrative=task.progress_narrative,
            vision_contract=dict(coordinate_space='image_1000_xyxy'))
        return perceive(self,observation,context,
            dict(observation_id=source['observation_id'],
                 scope=source.get('image_scope','current')))

    def search_view(self, observation, source, objective, target_types, force_recheck=False):
        """One RGB checkpoint for an active scan; never grants motion safety."""
        from .vision_protocol import coordinate_schema, coordinate_prompt
        from .vision_protocol import normalize_coordinates
        request=dict(mode='active_observation',objective=objective,
            target_types=list(target_types),
            observation_id=source['observation_id'],
            vision_contract=dict(coordinate_space='image_1000_xyxy'))
        return self.complete('visual_perception','active_observation',
            'Inspect ONLY this current RGB frame for the observation objective and listed '
            'target_types. Return one JSON object. Report one strongest visible candidate '
            'with target_visible=true, matching target_type, medium/high confidence, '
            'specific concise evidence, and its visible-part image box. '
            'If none is visibly localized use target_visible=false and target_type=none; '
            'do not hallucinate an obscured target or infer a safe path. '
            'Use the requested open-vocabulary target name exactly, or none. '
            'If evidence is ambiguous, '
            'suggested_next_view can ask for ONE look_up/look_down refinement; '
            'otherwise return continue. Do not infer metric distance, current human '
            'occupancy, physical reachability, movement safety or task completion. '
            'The controller owns real turns; collision and terrain geometry are ONLY '
            'inspected when NavigationAgent explicitly requests a geometry tool. '
            + coordinate_prompt(request),
            request,coordinate_schema(SEARCH_SCHEMA, 'image_1000_xyxy'),
            image_url=image_url(observation),cache=not force_recheck,
            normalize=lambda value: normalize_coordinates(value, SEARCH_SCHEMA),
            result_schema=SEARCH_SCHEMA)

    def guide_scan_view(self, observation, source, objective, target_types,
                        view_history, remaining_turn_deg, prior_targets=None,
                        observed_yaw_pitch=None, view_correction=None,
                        session=None, mode='APPROACH', route_history=None,
                        geometry_evidence=None, historical_frames=None,
                        view_coverage=None):
        """Vision reasoning over current RGB-D and complete in-session history."""
        from .vision_protocol import coordinate_schema,coordinate_prompt,normalize_coordinates
        task=self.runtime.memory.task
        pose=source['pose']
        turn=float(self.runtime.camera.turn_angle)
        allowed_turns=[0]+[int(i*turn) for i in
            range(1,1+min(6,int(remaining_turn_deg//turn)))
            if int(i*turn)<=90]
        pitch=pose.get('pitch_deg',0)
        tilt=self.runtime.camera.tilt_angle
        # A scan with no remaining yaw budget must not advertise a TURN
        # that can only have turn_deg=0 (an impossible controller command).
        # This removes a dead schema branch, not a model-selected route.
        allowed_actions=(['turn_left','turn_right','finish']
                         if len(allowed_turns)>1 else ['finish'])
        if pitch+tilt<=60:allowed_actions.append('look_up')
        if pitch-tilt>=-60:allowed_actions.append('look_down')
        request=dict(
            mode='natural_language_mission_observation',
            # The shared mission-memory document is injected exactly once
            # at the start of the dialogue (see mission_memory below).
            observation_request=objective,
            requested_targets=list(target_types),
            source_observation_id=source['observation_id'],
            heading_deg=round(math.degrees(pose['heading'])%360,2),
            pitch_deg=pitch,
            remaining_turn_deg=int(remaining_turn_deg),
            allowed_turn_deg=allowed_turns,
            allowed_camera_actions=allowed_actions,
            previously_inspected_views=list(view_history[-12:] if not session else []),
            observed_yaw_pitch=list(observed_yaw_pitch or ())[-12:],
            measured_geometry=geometry_evidence or {},
            mission_memory=(mission_evidence(self.runtime.memory,
                              self.runtime.motion_controller.pose(),
                              max_observation_tasks=4) if not session else
                              'see_initial_observation_session_message'),
            current_front_depth_preflight=(
                self.runtime.forward_preflight(stage='observation_view')
                if callable(getattr(self.runtime,'forward_preflight',None))
                else dict(status='UNKNOWN',reason='depth_preflight_unavailable')),
            current_sensor_images=['RGB','depth_gray_m'],
            depth_visualization=dict(min_m=0,max_m=5,
                near='black',far='white',invalid='magenta',
                alignment='same_pixel_coordinates_as_RGB'),
            waypoint_scope='hypothesis_not_route_permission',
            vision_contract=dict(coordinate_space='image_1000_xyxy'))
        depth_now=request['current_front_depth_preflight']
        if (depth_now.get('status')=='BLOCKED'
                and depth_now.get('source_observation_id')==
                    source['observation_id']):
            blocker=depth_now.get('blocker') or {}
            if blocker.get('focus_region_xyxy_norm') is not None:
                # The existing RGB-D observation request supplies the
                # current ROI; no extra HTTP inference or Python steering.
                request['obstacle_focus']=dict(
                    source_observation_id=source['observation_id'],
                    depth_region_xyxy_norm=copy.deepcopy(
                        blocker.get('focus_region_xyxy_norm')),
                    depth_focus_xy_norm=copy.deepcopy(
                        blocker.get('focus_point_xy_norm')),
                    nearest_along_m=blocker.get('nearest_along_m'),
                    question=('What is visibly present at this measured '
                        'near-field ROI, and is a different passage visible '
                        'beyond or beside it? Separate visible structure '
                        'from measured occupancy; report uncertainty.'),
                    measured_block_of_current_forward_step_only=True,
                    distant_route_walkability_unverified=True)
        if view_correction is not None:request['view_correction']=view_correction
        if view_coverage is not None:request['view_coverage']=view_coverage
        prior_current,prior_historical=session_frame_ids(session)
        already=source['observation_id'] in prior_current
        images=([] if already else [
            image_url(observation,max_edge=640),
            depth_image_url(observation,self.runtime.camera,
                max_edge=640,display_max_m=5.0)])
        if already:
            request['reuse_prior_RGBD_observation_id']=source['observation_id']
        historical=[]
        # Source-aligned, diverse RGB-D memories selected by one shared
        # Memory index, never arbitrary three newest turn-only frames.
        if not session:
            by_id={v['observation_id']:v for v in
                   self.runtime.memory.observations}
            for descriptor in navigation_visual_anchors(
                    self.runtime.memory,source['observation_id'],max_pairs=3):
                item=by_id.get(descriptor['observation_id'])
                if item is None:continue
                try:
                    frame=(self.runtime.load_observation(item)
                        if self.runtime.load_observation else
                        self.runtime.frames[item['observation_id']])
                    # Same immutable image identity rules as NavigationAgent:
                    # do not attach a historical frame under a stale label.
                    if (array_hash(frame.rgb)!=item.get('rgb_hash')
                            or array_hash(frame.depth)!=item.get('depth_hash')):
                        continue
                    saved_pose=item.get('pose') or {}
                    historical.append(dict(
                        observation_id=item['observation_id'],
                        heading_deg=round(math.degrees(
                            saved_pose.get('heading',0))%360,2),
                        pitch_deg=saved_pose.get('pitch_deg'),
                        motion_epoch=saved_pose.get('motion_epoch'),
                        place_id=item.get('place_id'),
                        selection_reason=descriptor.get('selection_reason'),
                        historical_only=True,
                        image_indices=[len(images),len(images)+1]))
                    images.append(image_url(frame,max_edge=640))
                    images.append(depth_image_url(frame,self.runtime.camera,
                        max_edge=640,display_max_m=5.0))
                except (KeyError,OSError,ValueError,TypeError):
                    continue
        if historical:
            request['historical_images']=historical
        schema=copy.deepcopy(SCAN_GUIDANCE_SCHEMA)
        schema['properties']['turn_deg']['enum']=allowed_turns
        schema['properties']['next_action']['enum']=allowed_actions
        def validate(value):
            action=value['next_action']
            turn_action=action in ('turn_left','turn_right')
            if turn_action and value['turn_deg']<=0:
                raise APIError('observation_turn_must_be_positive',
                               category='structured_field')
            if not turn_action and value['turn_deg']!=0:
                raise APIError('observation_look_finish_turn_must_be_zero',
                               category='structured_field')
            if value['turn_deg'] not in allowed_turns or action not in allowed_actions:
                raise APIError('observation_camera_action_not_calibrated',
                               category='structured_field')
        result=self.complete('visual_perception','vlm_guided_observation',
            SCAN_GUIDANCE_PROMPT+coordinate_prompt(request),
            request,coordinate_schema(schema,'image_1000_xyxy'),
            image_url=images if not already else None,cache=False,
            validate=validate,
            normalize=lambda value:normalize_coordinates(value,
                SCAN_GUIDANCE_SCHEMA,default_unit='image_1000_xyxy'),
            result_schema=schema,
            validation_schema=coordinate_schema(schema,'image_1000_xyxy',
                allow_missing_unit=True),
            conversation_messages=session)
        if session is not None:
            msg=[dict(type='text',text=json.dumps(request,ensure_ascii=False))]
            if not already:
                msg.extend(dict(type='image_url',image_url=dict(url=u))
                    for u in images)
            session.append(dict(role='user',content=msg))
            answer=copy.deepcopy(result)
            for region in ('box','passage_box'):
                if answer.get(region) is not None:
                    answer[region]=[int(round(v*1000))
                                    for v in answer[region]]
            answer['coordinate_space']='image_1000_xyxy'
            session.append(dict(role='assistant',content=json.dumps(
                answer,ensure_ascii=False)))
        return result

    def grounding(self,observation,source,target):
        return ground(self,observation,Geometry(observation,self.runtime.camera,self.runtime.config['safety_margin']),
            target,dict(mode='grounding',question='Locate the target and any walkable entry, treads or landing.',
                vision_contract=dict(coordinate_space='image_1000_xyxy')),source)

    def compare(self,frames,question):
        frames=sorted(frames,key=lambda pair:pair[1]['captured_at'])
        aliases={f'view_{i}':source['observation_id'] for i,(_,source) in enumerate(frames)}
        slots=[(obs,dict(observation_id=f'view_{i}',captured_at=i)) for i,(obs,_) in enumerate(frames)]
        value=perceive_views(self,slots,dict(mode='compare_views',question=question,
            vision_contract=dict(coordinate_space='image_1000_xyxy')))
        value=copy.deepcopy(value)
        for item in value['views']:item['observation_id']=aliases[item['observation_id']]
        for item in value['changes']:
            item['from_observation_id']=aliases[item['from_observation_id']]
            item['to_observation_id']=aliases[item['to_observation_id']]
        return value,frames

    def describe_obstacle(self, observation, source, depth_evidence):
        """One current RGB frame, aligned to a measured Depth obstacle region.

        The VLM acts as eyes only: it may describe a visible person, railing,
        barrier or uncertain surface, but never decides collision, path safety
        or the next motion. No turn/look tool is available here.
        """
        request=dict(
            mode='describe_measured_obstacle',
            observation_id=source['observation_id'],
            heading_deg=round(math.degrees(source['pose']['heading']),2),
            pitch_deg=source['pose'].get('pitch_deg',observation.pitch_deg),
            depth_focus_xy_norm=depth_evidence.get('focus_point_xy_norm'),
            depth_region_xyxy_norm=depth_evidence.get('focus_region_xyxy_norm'),
            visual_question=('Describe only what is visibly present at the '
                'Depth-indicated region and whether a surface/person/railing '
                'appears between the camera and any visible distant target.'))
        return self.complete(
            'visual_perception','near_field_obstacle_description',
            'You are a visual observer, not a navigator. Use ONLY the supplied '
            'single current RGB frame. Depth has measured nearby central '
            'occupancy and supplies the normalized focus point/region; those '
            'coordinates are search hints, not proof of an object identity. '
            'Describe the visible object or structure at that region and its '
            'visual relationship to more distant floor or navigation landmarks. '
            'If the region is ambiguous, out of view, partly transparent or '
            'the object cannot be classified, state that explicitly. '
            'Do not fabricate a person, glass, railing or passable opening. '
            'Do not choose camera turns, motion, collision clearance, path '
            'safety, bypass or semantic task completion. Return one short '
            'schema-valid JSON object.',
            request,OBSTACLE_VISUAL_SCHEMA,
            image_url=image_url(observation,max_edge=640),cache=False,
            result_schema=OBSTACLE_VISUAL_SCHEMA)

    def entry(self,question):
        r=self.runtime
        context=dict(mode='entry_assessment',observation_id=r.memory.current['observation_id'],question=question)
        return self.complete('visual_perception','entry_assessment',
            'Answer this one RGB question about an opening, railing or stair entry. A railing alone '
            'does not locate a walkable entrance. State uncertainty and a useful next observation. '
            'Do not infer metric coordinates, path safety or task completion. Return one JSON object.',
            context,ENTRY_SCHEMA,image_url=image_url(r.observation),cache=True,
            cache_context=semantic_cache_context(context))
