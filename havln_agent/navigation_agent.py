"""NavigationAgent: one whole instruction, local motion and memory updates."""
import math
from jsonschema import Draft7Validator
from .contracts import (navigation_schema, compact_navigation_protocol,
                        navigation_envelope)
from .contexts import navigation_context, compact_navigation_context
from .tool_menu import navigation_tool_menu
from .api_client import APIError
from .perception import image_url
from .observation import array_hash
from .navigation_history import navigation_visual_anchors
from .heading import signed_turn
from .model_angles import project_model_angles
from .contracts import normalize_navigation_route_update

MISSION_PROMPT=(
    "You are NavigationAgent. Only you choose movement, observation, replanning and STOP for the ORIGINAL full instruction. "
    "First carefully interpret the COMPLETE instruction as an ordered route: what to do first, BEFORE/AFTER/UNTIL conditions, the reference for each left/right (stair DESCENT direction or named junction, not necessarily the current camera), the onward path and distinguishing FINAL destination evidence. Optional scenery ('might notice') is not a required waypoint. On each decision identify the pending clause and say briefly in decision_analysis WHY this action advances it. Never let the latest visible room silently replace the original goal. "
    "Use current RGB (FIRST image), mission_evidence_v1, source-linked histories and actual motion receipts. Model progress narratives are hypotheses, not verified arrival. Historical RGB manifest gives signed relative offsets from CURRENT camera; offsets are view comparisons, not motor bearings across stances. No global yaw is given. "
    "Choose turn_direction LEFT, RIGHT, or NONE relative to the CURRENT image, with nonnegative turn_degrees. LEFT is image-left, RIGHT image-right; NONE=0. look=up/down/none is independent. Keep pitch model-controlled (no automatic LOOK_UP): consider LOOK_DOWN for nearby treads, consider LOOK_UP for distant openings. Choose views not on a fixed schedule. "
    "On stairs, a visible staircase does NOT prove descent; seeing the distant landing does not prove the last tread passed. Check near-floor continuous flat surface and source-linked before/after observation IDs. Curved-stair steering while still descending is fine. Call observe_for_goal selectively only for a specific unresolved view question, never for mandatory permission. "
    "A command of distance_m 0.25..3m is finite nominal forward steps, not measured translation or completion. Stable one plausible continuous passage should favor a short model-chosen forward exploration instead of repeated scans. In uncertain terrain use a shorter segment. Every atomic forward rechecks fresh Depth. front_depth_preflight BLOCKED concerns the NEXT step (not failure of the completed steps); CLEAR is not full-route proof and absent is UNKNOWN. previous_identical_view_forward_rejections means that unchanged view+pose already denied forward with ZERO actions; changing only distance is not progress. There is no forced scan or forward rule. "
    "Use route_continuity, recent_look_evidence and actual receipts to avoid repeat stationary viewpoints; opposing headings or visual similarity are not proven loop closure. Historical image offsets are not direct motion commands. "
    "Keep a chosen SEMANTIC route/doorway separate from the next CAMERA-relative motor turn. 'Left after stairs' selects a branch relative to the original descent approach: it does not require blindly turning 90 degrees toward a wall. An opposite local turn to align or avoid an obstacle can be sensible without abandoning the selected doorway. If new evidence truly contradicts the chosen route, cite it and revise deliberately rather than drifting to a visually attractive opening. "
    "When multiple plausible ROOM entrances exist, compare each with the original instruction and distinctive person/activity evidence. A chair or doorway is not proof of a seated reader; target NOT_VISIBLE is uncertainty, not proven absence. Prefer informative new views over repeated equivalent looks, but do not demand scans after sufficient evidence. ObservationAgent only reports visual evidence; NavigationAgent moves and decides final success. "
    "Keep progress revisable. At an important junction/stair exit or changed branch, optionally write route_reference_update with source_observation_id (REAL historical RGB), instruction_relation (direction relative to ORIGINAL instruction), optional place_description, evidence_status and cited route_attempt_outcome/outcome_observation_id. On return, compare saved junction image with live RGB; never treat saved orientation as a current turn. Only request_finish success on evidence of the actual FINAL destination. "
    "Output 1..5 concise observed_evidence items, short decision_analysis, then exactly one legal tool with args LAST. Only the registered tool/args execute; route annotations are optional, not extra motion authorization."
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
        context=compact_navigation_context(state)
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
        proposal=r.client.complete(
            'navigation_agent','navigation_decision',MISSION_PROMPT,context,
            schema,validate=validate,
            normalize=lambda result: normalize_navigation_route_update(result,known_ids),
            image_url=images,
            format_retry=int(bool(format_feedback)),retry_safe=True,
            protocol=compact_navigation_protocol(tools),
            validation_schema=navigation_envelope(tools))
        return proposal
