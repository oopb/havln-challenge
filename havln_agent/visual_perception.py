"""One visual question per request; semantic reuse is keyed by image and question."""
import copy
import math

from .api_client import TEXT,object_schema,APIError,BudgetExceeded
from .grounding import ground
from .perception import Geometry,image_url,depth_image_url,perceive,perceive_views
from .contexts import fields

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
    evidence=dict(type='string',maxLength=400),
    uncertainty=dict(type='string',maxLength=300),
    target_kind=dict(enum=['ENTRY','APPROACH','TARGET','CUE','UNKNOWN']),
    candidate_approach=dict(type='boolean'),
    next_action=dict(enum=['turn_left','turn_right','look_up','look_down','finish']),
    turn_deg=dict(enum=[0,15,30,45,60,90]),
    reason=dict(type='string',maxLength=250)))

SCAN_GUIDANCE_PROMPT = (
    'You are the VLM OBSERVATION SUBAGENT, responsible for visual search and '
    'all camera turns/look changes within this delegated observation task. '
    'NavigationAgent specified WHAT to find; YOU choose each next viewing '
    'direction using the CURRENT real RGB and metric-labeled Depth images '
    'and the short search history. The first image is RGB; the second is '
    'Depth when supplied (near black, far white, MAGENTA invalid). The '
    'display range in the request is meters, not raw normalized sensor units. '
    'Depth is a perspective measurement, not proof a visible floor is '
    'reachable through a railing, glass or a person. '
    'The robot stays stationary: you may request turn_left, turn_right, '
    'look_up, look_down or finish, but NEVER forward motion, task success, '
    'or collision clearance. turn_deg is an absolute multiple of 15 degrees '
    '(15-90 for turns, 0 for look/finish). Robot yaw: LEFT positive, '
    'RIGHT negative; pitch changes are NOT yaw. Camera pitch POSITIVE looks '
    'UP, NEGATIVE looks DOWN; use the actual pitch_deg in the request, never '
    'say the camera looks down when pitch_deg is positive. The harness will execute '
    'the requested calibrated atomic camera actions before your NEXT image. '
    'Search for a visually actionable region answering the specific objective '
    'and open-vocabulary targets. The objective may describe visual cues '
    'expected at or near the END of the current subgoal: treat them ONLY as '
    'possible visual references. Report whether the current RGB actually '
    'shows such cues, the visible spatial relation (such as treads ending '
    'at a floor versus a distant floor seen from above), and any ambiguity. '
    'If an expected feature is not visible, say so; do not infer presence '
    'from the instruction, hallucinate a floor, or declare arrival or task '
    'completion. Retain other genuinely visible clues even if they do not '
    'match the expectation. Do not turn or scan additional views solely to '
    'verify completion; camera actions still serve the delegated visual '
    'question. A distant staircase/railing is a CUE, '
    'not necessarily a reachable ENTRY; if an entry or approach is unclear, '
    'inspect a DIFFERENT view or look down/up, not immediately finish. '
    'target_type is an open-vocabulary description of what you actually see, '
    'not a required exact match to the requested target_types. Describe '
    'visible evidence freely, including irrelevant or uncertain observations. '
    'Use target_kind ENTRY/APPROACH for a visible approach, TARGET for a '
    'localized non-entry object/person, only if image evidence supports it; '
    'otherwise CUE or UNKNOWN. Return an exact normalized image box when '
    'target_visible and localized; do not hallucinate missing regions. '
    'Separate directly visible textures, surfaces and people from guesses '
    'about navigation affordances. If an attractive goal or floor is seen '
    'BEYOND a visible railing, transparent-looking panel, person or wall, '
    'report that intervening structure in evidence/uncertainty rather than '
    'presenting the distant target as an immediate walkable approach. '
    'If a wall or ambiguous surface occupies '
    'the frame, do not label it a floor, doorway or traversable region solely '
    'because the instruction expects one. Prefer CUE/UNKNOWN when uncertain. '
    'You are also a LOCAL APPROACH SEARCH AGENT: find a plausible direction '
    'from the CURRENT ROBOT position, not just a distant target. If the '
    'camera currently faces a visually promising approach or exploratory '
    'short forward direction, mark candidate_approach=true. This requests '
    'a local one-step Depth obstacle probe at the CURRENT body heading '
    '(not a global path certificate); NO forward is performed by you. '
    'If the candidate is at another bearing, first turn the camera there '
    'and inspect fresh RGB-D, rather than assuming its clearance from '
    'the current image. When candidate_approach=true, preferably choose '
    'next_action=finish; the controller probes first and then hands a '
    'non-blocked hypothesis to NavigationAgent. BLOCKED candidate feedback '
    'is returned to YOU in view_correction and recent observation history '
    'within THIS SAME observation task: choose a different direction/look, '
    'or finish with uncertainty when no useful new direction remains. '
    'Never insist on proof of traversability: UNKNOWN or a side-only Depth '
    'hit is NOT a confirmed blocker and can be handed off as uncertain. '
    'Avoid re-proposing the same blocked current bearing on identical RGB-D. '
    'You never execute MOVE_FORWARD, decide a route distance or complete '
    'semantic goals. Detecting a visual candidate does NOT by itself terminate this task: '
    'if another real camera view would resolve a concrete ambiguity, choose '
    'that look/turn; choose finish when enough evidence is collected. '
    'Avoid already-observed yaw/pitch combinations and useless 15-degree '
    'micro-scans; larger calibrated turns are allowed. If a previous camera '
    'choice was rejected, use view_correction and observed_yaw_pitch to '
    'choose a DIFFERENT real view (often look down/up) within this SAME '
    'subtask. Never repeat a rejected action. Finish when meaningful '
    'visual evidence is sufficient OR the remaining views will not help; '
    'state uncertainty. Return candidate_approach as a boolean on EVERY '
    'output, with the CURRENT camera heading as its proposed forward bearing '
    'when true. You only return evidence and the next camera action. '
    'Do not invent robot motion, measured depth, safety or task completion. '
    'Return ONE JSON object.'
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
    'Each view identifies observation_id, image_index, absolute camera yaw, '
    'signed relative yaw from the robot CURRENT heading, pitch and historical '
    'front depth state (UNMEASURED when geometry was not explicitly requested). '
    'Read those actual angles; do NOT infer them from image '
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


class VisualPerception:
    def __init__(self,runtime):
        self.runtime=runtime

    def complete(self,*args,**kwargs):
        # Format repair uses the same frozen RGB/question, a new full deadline
        # and one bounded correction. Conflicting duplicate keys are rejected.
        original=list(args)
        if kwargs.get('cache'):
            kwargs['cache_context']=semantic_cache_context(args[3])
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
                retry[3]=dict(original[3],format_feedback=dict(error=detail[:350],
                    instruction='Regenerate ONE complete schema-valid JSON object. Every key must occur once. No duplicate or conflicting fields.'))
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
        index=task.current_subgoal_index
        goal=task.subgoals[min(index,len(task.subgoals)-1)] if task.subgoals else {}
        context=dict(mode='scene',question=question,
            current_task=fields(goal,('action','landmark','target','semantic_description')),
            vision_contract=dict(coordinate_space='image_1000_xyxy'))
        if task.subgoals and index>=len(task.subgoals)-1:
            context['stopping_condition']=task.final_target
        return perceive(self,observation,context,
            dict(observation_id=source['observation_id'],scope=source.get('image_scope','current')))

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
                        observed_yaw_pitch=None, view_correction=None):
        """VLM-directed next camera action from one genuine current RGB view.

        Provide only compact, source-labeled history, not a growing conversation
        or full prior images. Every decision uses the current sensor image.
        """
        from .vision_protocol import coordinate_schema, coordinate_prompt
        from .vision_protocol import normalize_coordinates
        task=self.runtime.memory.task
        index=task.current_subgoal_index
        subgoal=(task.subgoals[index] if index<len(task.subgoals) else {})
        pose=source['pose']
        request=dict(mode='vlm_guided_observation',
            instruction=task.instruction,
            subgoal=fields(subgoal,('action','target','landmark','semantic_description')),
            objective=objective,target_types=list(target_types),
            observation_id=source['observation_id'],
            heading_deg=round(__import__('math').degrees(pose['heading']),2),
            pitch_deg=round(pose.get('pitch_deg',0),2),
            remaining_turn_deg=int(remaining_turn_deg),
            previously_inspected_views=list(view_history)[-12:],
            observed_yaw_pitch=list(observed_yaw_pitch or ())[-12:],
            prior_target_hypotheses=list(prior_targets or ())[-2:],
            vision_contract=dict(coordinate_space='image_1000_xyxy'))
        images=[image_url(observation,max_edge=640)]
        camera=getattr(self.runtime,'camera',None)
        if camera is not None and getattr(observation,'depth',None) is not None:
            images.append(depth_image_url(observation,camera,max_edge=640,
                                          display_max_m=5.0))
            request['sensor_images']=['current_rgb','current_depth_gray_m']
            request['depth_visualization']=dict(
                display_min_m=0.0,display_max_m=5.0,
                near_color='black',far_color='white',
                invalid_color='magenta',clip_beyond_max_m=True,
                alignment='same_normalized_pixel_coordinates_as_rgb',
                raw_depth_normalized=bool(camera.normalized),
                note='visual_depth_reference_not_collision_certificate')
        else:
            request['sensor_images']=['current_rgb']
        if view_correction is not None:
            request['view_correction']=view_correction
        def validate(value):
            turns=value['next_action'] in ('turn_left','turn_right')
            if turns and not value['turn_deg']:
                raise APIError('vlm_turn_requires_positive_calibrated_angle',
                               category='structured_field')
            if not turns and value['turn_deg']!=0:
                raise APIError('vlm_look_or_finish_requires_zero_turn',
                               category='structured_field')
        return self.complete(
            'visual_perception','vlm_guided_observation',
            SCAN_GUIDANCE_PROMPT+coordinate_prompt(request),
            request,coordinate_schema(SCAN_GUIDANCE_SCHEMA,'image_1000_xyxy'),
            image_url=images,cache=False,
            validate=validate,
            normalize=lambda value:normalize_coordinates(value,SCAN_GUIDANCE_SCHEMA),
            result_schema=SCAN_GUIDANCE_SCHEMA)

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
