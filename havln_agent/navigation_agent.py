"""NavigationAgent: one whole instruction, local motion and memory updates."""
import math
from jsonschema import Draft7Validator
from .contracts import (navigation_schema, compact_navigation_protocol,
                        navigation_envelope, navigation_generation_schema,
                        navigation_choice_schema)
from .contexts import navigation_context, cost_optimized_navigation_context
from .tool_menu import navigation_tool_menu
from .api_client import APIError
from .perception import image_url
from .observation import array_hash
from .navigation_history import navigation_visual_anchors
from .heading import signed_turn
from .model_angles import project_model_angles
from .contracts import normalize_navigation_route_update

MISSION_BASE_PROMPT=(
    "You are NavigationAgent. Complete the ORIGINAL full navigation instruction. Only you choose movement, observation, replanning and STOP; ObservationAgent supplies RGB-D evidence, never motion authorization. "
    "Read mission_evidence_v1, current RGB, source-linked history and motion receipts. Progress is a model hypothesis. Match instruction landmarks to observed scene geometry (stairs, doors, room and corridor layout); people can move, appear or leave, so their past positions are not fixed waypoints. An unseen person is not evidence the spatial route is wrong. "
    "The FIRST image is CURRENT RGB; history_media.manifest contains signed relative offsets from the CURRENT camera (positive LEFT, negative RIGHT) plus historical source pitch/stance. Past view offsets are NOT movement commands across different stances; always re-ground in current RGB. "
    "Choose turn_direction LEFT, RIGHT, or NONE relative to the CURRENT robot camera plus nonnegative turn_degrees (NONE=0). LEFT means the left side of CURRENT RGB; RIGHT means right. Never calculate an absolute target heading for a motor turn. Pitch uses look=up/down/none separately. "
    "Camera pitch is model-controlled during MOVE (no automatic LOOK_UP); consider LOOK_DOWN for nearby floor/stairs, consider LOOK_UP for distant openings hidden by a down-tilted view. "
    "Choose informative pitch changes, not on a fixed schedule. Use turn_and_observe with look=up/down and no turn when helpful. After checking near-floor steps, consider LOOK_UP for the distant route; LOOK is not translation. "
    "Call observe_for_goal selectively for a SPECIFIC route-critical question: do CURRENT stair treads rise or descend, or which side of a person has a passage? Compare source-linked before/after observation IDs with current RGB-D; a visible staircase does NOT prove descent. Curved-stair steering while still descending is fine. Avoid observation when the route is clear; VLM cannot authorize motion. "
    "front_depth_preflight: BLOCKED is occupied; CLEAR is not full-route proof; absent means UNKNOWN. If previous_identical_view_forward_rejections>0, identical RGB-D/stance/yaw/pitch already caused a denied forward with ZERO actions executed. Changing only distance_m adds no new evidence; consider a new view, direction or focused observe_for_goal. Every atomic forward rechecks fresh Depth; do not chase CLEAR by turning. "
    "Read memory.route_continuity and memory.recent_look_evidence: compare actual forward receipts with model claims; opposing headings or similar scenery do not prove loop closure or descent. Use full_coverage_by_pitch before repeating LOOK/turn; turns are observation, not progress. There is no forced scan or forward rule. "
    "A front BLOCKED after completed forward steps concerns the NEXT step, not failure of the completed steps. Compare before/after RGB before changing goals. If one plausible continuous passage advances the current instruction-consistent scene goal, with no front BLOCKED or obvious person/dropoff risk, favor a short model-chosen forward exploration over repeated TURN/LOOK; each atomic forward still gets fresh Depth preflight. An open exit alone is not reason to leave a matching scene. Multiple passages justify focused observe_for_goal. "
    "Commanded distance (0.25..3m) is nominal atomic steps, NOT measured arrival or stair descent. Cruise farther in stable corridors; shorten steps near crowds, stairs or obstacles. No verified height change or metric robot x/y/z. "
    "Use relative turn_direction plus turn_degrees in BOTH turn_and_observe and navigate_relative. Pure TURN/LOOK uses turn_and_observe; navigate_relative needs distance_m>=0.25. Camera pitch limits are recoverable; read camera_controls.allowed_look_now. "
    "When selecting an instruction-matching scene, set optional scene_goal to its durable SPATIAL target, not a camera turn. Retain across turns/MOVE; change when new positive spatial evidence justifies another scene. Before leaving for a different entrance, compare the ORIGINAL instruction and room/route layout; explain the actual contradiction or better match. A missing moving person or open door alone is insufficient. Turning to observe and evidence-backed backtracking remain allowed. For multiple plausible ROOM entrances compare actual scenes; target NOT_VISIBLE is uncertainty, not room rejection. At a junction optional route_reference_update cites real RGB and instruction-relative branch, not motor turn. Only request_finish success on final-destination evidence. "
)
MISSION_PROMPT=MISSION_BASE_PROMPT + (
    "Output 1-2 SHORT source-grounded observed_evidence items (avoid paraphrasing history), then a one-sentence decision_analysis citing why this move advances the goal. Update progress_narrative, scene_goal and route references only when genuinely changed. Put the executable tool and args LAST. Return one legal registered tool; no hidden chain-of-thought."
)
MISSION_SELECTION_PROMPT=MISSION_BASE_PROMPT + (
    "In this stage choose exactly ONE registered tool, citing 1-2 short "
    "source-grounded observed_evidence facts and one sentence of "
    "decision_analysis. Provide optional progress only if genuinely changed. "
    "Return 'tool' but DO NOT emit 'args' or any physical command parameters. "
    "The selected tool's required arguments are generated in a separate "
    "schema-constrained decision. No hidden chain-of-thought."
)
MISSION_ARGUMENTS_PROMPT=MISSION_BASE_PROMPT + (
    "A tool was already selected by NavigationAgent. You now choose ONLY "
    "its executable arguments from the current RGB, instruction and evidence. "
    "Return exactly the parameter JSON object required by the selected tool, "
    "with every required field, and no 'tool' or 'args' wrapper. "
    "Do not choose a different tool, invent missing observations, or omit "
    "an argument to express uncertainty. Physical movement remains subject "
    "to independent Depth preflight and the registered executor."
)

class NavigationAgent:
    def __init__(self,runtime):
        self.runtime=runtime

    def decide(self,format_feedback=None):
        r=self.runtime
        if r.navigation_mode!='OBSERVE':
            raise RuntimeError('navigation_decision_requires_observe_mode')
        r.registry=r.make_registry()
        tools=navigation_tool_menu(r.registry)
        # Log live Depth every decision; only conclusive results go to LLM.
        # Physical movement keeps its independent fresh calibrated preflight.
        preflight=r.forward_preflight(stage='navigation_decision')
        state=navigation_context(r)
        if preflight.get('status') in ('CLEAR','BLOCKED'):
            state['front_depth_preflight']=preflight
        context=cost_optimized_navigation_context(state)
        # Preserve bounded context, but allow up to five diverse, source-
        # verified historical RGB images for before/after route comparison.
        if format_feedback:
            context['format_feedback']=format_feedback
        evidence=context['memory']
        r.event('mission_context_built',source='memory_projection',
            memory_contract=evidence['contract'],
            observation_task_ids=[v.get('observation_task_id')
                                  for v in evidence['observation_tasks']],
            historical_frame_ids=[v['observation_id']
                                  for v in evidence['historical_frames']],
            blocked_source_ids=[v['source_observation_id']
                                for v in evidence['blocked_at_source_stances']],
            model_progress_revisions=evidence['progress']['model_revision_count'])
        frame=getattr(r,'observation',None)
        view=r.memory.current
        images=None
        if frame is not None and view is not None:
            source=r.source()
            pose=source['pose']
            if (source['observation_id']!=view['observation_id']
                or source['rgb_hash']!=view['rgb_hash']
                or source['depth_hash']!=view['depth_hash']
                or pose!=view['pose']):
                raise APIError('navigation_current_RGBD_pose_identity_mismatch',
                               category='structured_field')
            # v12 input shape + a SMALL number of authentic historical RGB
            # views. Do not attach full RGB-D sessions, full timelines,
            # historical Depth or a duplicated all-frame catalog.
            images=[image_url(frame,max_edge=640)]
            manifest=[dict(
                image_index=0,kind='current_RGB',
                observation_id=source['observation_id'],
                heading_deg=round(math.degrees(pose['heading'])%360,2),
                stance_id=pose.get('stance_id'),
                current_sensor_frame=True)]
            by_id={o['observation_id']:o for o in r.memory.observations}
            # Spend two extra image slots ONLY when the previous route
            # is viewed from a substantially opposing bearing.
            continuity=context['memory'].get('route_continuity') or {}
            history_slots=(5 if (
                continuity.get('separation_from_historical_forward_deg') or 0
                )>=105 else 3)
            for descriptor in navigation_visual_anchors(
                    r.memory,source['observation_id'],max_pairs=history_slots):
                identifier=descriptor['observation_id']
                item=by_id.get(identifier)
                if item is None:continue
                try:
                    historical=(r.load_observation(item)
                        if r.load_observation else r.frames[identifier])
                    if (array_hash(historical.rgb)!=item.get('rgb_hash')
                            or array_hash(historical.depth)!=item.get('depth_hash')):
                        continue
                    history_pose=item.get('pose') or {}
                    h=history_pose.get('heading')
                    if not isinstance(h,(int,float)) or not math.isfinite(h):
                        continue
                    encoded=image_url(historical,max_edge=640)
                    image_index=len(images)
                    images.append(encoded)
                    manifest.append(dict(
                        image_index=image_index,kind='historical_RGB',
                        observation_id=identifier,
                        source_heading_deg=round(math.degrees(h)%360,2),
                        source_pitch_deg=history_pose.get('pitch_deg'),
                        source_stance_id=history_pose.get('stance_id'),
                        selection_reason=descriptor['selection_reason'],
                        historical_only_not_live_clearance=True))
                except (KeyError,OSError,ValueError,TypeError,AttributeError):
                    # Missing historical frame is optional. Current RGB
                    # and v12 text memory are always sufficient to proceed.
                    continue
            # If an authentic historic RGB is attached, its text catalog
            # entry is redundant; keep only unmatched source-grounded
            # descriptions. Never discard the original junction image.
            indexed={v['observation_id'] for v in manifest[1:]}
            historical=context['memory'].get('historical_frames') or []
            if indexed and historical:
                remaining=[v for v in historical
                           if v.get('observation_id') not in indexed]
                if remaining:
                    context['memory']['historical_frames']=remaining[-1:]
                else:
                    context['memory'].pop('historical_frames',None)
            context['history_media']=dict(
                manifest=manifest,
                attached_historical_RGB=len(images)-1,
                source_frame_provenance='actual_RGBD_hashes',
                historical_images_are_not_live_clearance=True)
            context['current_rgb']=dict(
                image_attached=True,observation_id=source['observation_id'],
                heading_deg=round(math.degrees(pose['heading'])%360,2),
                pitch_deg=pose.get('pitch_deg'),rgb_hash=source['rgb_hash'],
                scope='current_public_RGB_not_historical')
            r.event('navigation_current_rgb_attached',source='navigation_agent',
                    source_observation_id=source['observation_id'],
                    rgb_hash=source['rgb_hash'],
                    total_attached_images=len(images),
                    historical_RGB=len(images)-1)
        else:
            context['current_rgb']=dict(image_attached=False,
                                        scope='no_live_image_available')
            context['history_media']=dict(
                manifest=[],attached_historical_RGB=0,
                historical_images_are_not_live_clearance=True)
        schema=navigation_schema(tools)
        def validate(value):
            # Truly malformed tool arguments remain format errors; the error
            # names the specific field, while physical LOOK limits belong to
            # execution-time feedback and do not fail the JSON schema.
            if isinstance(value,dict) and isinstance(value.get('tool'),str):
                spec=tools.get(value['tool'])
                args=value.get('args')
                if spec and isinstance(args,dict):
                    issue=next(Draft7Validator(spec['parameters']).iter_errors(args),
                               None)
                    if issue is not None:
                        field='.'.join(map(str,issue.path)) or 'args'
                        detail=(('allowed='+str(issue.validator_value)+
                                 ', received='+repr(issue.instance)[:80])
                                if issue.validator=='enum' else issue.message)
                        raise APIError(
                            ('navigation_invalid_tool_argument:'+
                             value['tool']+'.'+field+':'+detail)[:300],
                            category='structured_field')
            if not Draft7Validator(schema).is_valid(value):
                raise APIError('navigation_tool_arguments_violate_registered_schema',
                               category='structured_field')
            reference=value.get('route_reference_update')
            if reference is not None:
                allowed={o['observation_id'] for o in r.memory.observations}
                if reference['source_observation_id'] not in allowed:
                    raise APIError('route_reference_unknown_source_observation_id',
                                   category='structured_field')
                outcome=reference.get('outcome_observation_id')
                if outcome is not None and outcome not in allowed:
                    raise APIError('route_reference_unknown_outcome_observation_id',
                                   category='structured_field')
            if value['tool'] in ('navigate_relative','turn_and_observe'):
                args=value['args']
                try:
                    signed_turn(args['turn_direction'],args['turn_degrees'])
                except ValueError as error:
                    raise APIError(str(error),category='structured_field') from None
        # Last-mile model-only projection: keep absolute headings in
        # canonical Memory and events, NEVER in the actual LLM request.
        # Projection is read-only and runs AFTER all source frames are linked.
        context=project_model_angles(
            context,math.degrees(r.motion_controller.pose()['heading']))
        # IDs come from immutable sensor records; optional memory annotations
        # never block a valid tool decision when they are malformed.
        known_ids={o['observation_id'] for o in r.memory.observations}
        protocol=compact_navigation_protocol(tools)
        generation=navigation_generation_schema(tools)
        def staged_decision(reason):
            # A backend without reliable oneOf support cannot constrain
            # tool-specific required args in one response. Split selection
            # and argument generation, both model-owned and schema-validated.
            # No argument defaults or deterministic changes to motor commands.
            r.event('navigation_tool_schema_staging',
                source='navigation_agent',reason=reason,
                observation_id=r.memory.current['observation_id'])
            choice=r.client.complete(
                'navigation_agent','navigation_tool_selection',
                MISSION_SELECTION_PROMPT,context,
                navigation_choice_schema(tools),
                pre_normalize=lambda value: normalize_navigation_route_update(
                    value,known_ids),
                image_url=images,format_retry=int(bool(format_feedback)),
                retry_safe=True,
                protocol=('Return one JSON object with observed_evidence, '
                    'decision_analysis, tool and optional progress. '
                    'DO NOT include args. Tools: '+', '.join(tools)))
            name=choice['tool']
            spec=tools[name]
            arguments_context=dict(context,
                selected_tool=dict(name=name,
                    description=spec['description'],
                    required_arguments=spec['parameters']['required'],
                    prior_decision=choice))
            arguments=r.client.complete(
                'navigation_agent','navigation_tool_arguments',
                MISSION_ARGUMENTS_PROMPT,arguments_context,
                spec['parameters'],image_url=images,
                format_retry=int(bool(format_feedback)),retry_safe=True)
            proposal=dict(choice,args=arguments)
            validate(proposal)
            return proposal

        # Prefer a single native discriminated union whenever possible.
        # If the provider rejects oneOf, capability negotiation is cached
        # and subsequent decisions use the two-stage typed protocol.
        if getattr(r.client,'needs_navigation_staging',
                   lambda role,schema:False)('navigation_agent',generation):
            return staged_decision('native_union_unavailable')
        try:
            return r.client.complete(
                'navigation_agent','navigation_decision',MISSION_PROMPT,context,
                schema,validate=validate,
                pre_normalize=lambda result: normalize_navigation_route_update(
                    result,known_ids),
                image_url=images,
                format_retry=int(bool(format_feedback)),retry_safe=True,
                protocol=protocol,
                validation_schema=navigation_envelope(tools),
                generation_schema=generation)
        except APIError as error:
            # A JSON response can still violate a provider's purported
            # union grammar. Do not execute it or repeatedly regenerate a
            # flat args object; let the model generate exact typed arguments.
            if (error.category not in (
                    'structured_field','json_schema','json_syntax')
                    or not hasattr(r.client,'needs_navigation_staging')):
                raise
            # A model that repeatedly violates a supposedly enforced union
            # should use typed selection+arguments next time, even though
            # the server accepted the native schema syntactically.
            marker=getattr(r.client,'mark_navigation_schema_unreliable',None)
            if marker is not None:
                marker('navigation_agent',generation)
            return staged_decision('native_union_response_invalid')
