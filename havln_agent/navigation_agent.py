"""NavigationAgent: one whole instruction, local motion and memory updates."""
import math
from jsonschema import Draft7Validator
from .contracts import (navigation_schema, compact_navigation_protocol,
                        navigation_envelope)
from .contexts import navigation_context, compact_navigation_context
from .tool_menu import navigation_tool_menu
from .api_client import APIError, specific_schema_error
from .perception import image_url
from .observation import array_hash
from .navigation_history import navigation_visual_anchors
from .heading import signed_turn

MISSION_PROMPT=(
    'You are NavigationAgent. Complete the ORIGINAL full navigation mission. '
    'Only you choose MOVE, observation, replanning and STOP; ObservationAgent '
    'supplies visual evidence, never movement authorization. '
    'Read current mission evidence, recent receipts and scan coverage. '
    'Omitted old records may still exist. '
    'First image is CURRENT RGB; later images are source-labeled '
    'historical RGB, never live clearance. '
    'Historical LEFT/RIGHT descriptions refer to their original camera '
    'view, not necessarily the CURRENT RGB. After moving or turning, '
    're-evaluate the visible route in CURRENT RGB. Select the physical '
    'direction with turn_direction LEFT, RIGHT, or NONE and a nonnegative '
    'turn_degrees magnitude. For tilt select look=up/down/none when using '
    'turn_and_observe. The executor converts direction words directly '
    'into motor actions; do not calculate absolute target headings. '
    'When an opening or route is visible, select the useful heading '
    'yourself; never turn away from your stated target by accident. '
    'If no route is certain, choose a short informed exploration or '
    'ask ObservationAgent for specific evidence. '
    'front_depth_preflight is present ONLY for measured CLEAR or BLOCKED '
    'on the current view. BLOCKED is confirmed occupancy in the '
    'checked direction; CLEAR is not whole-route clearance. '
    'When this field is absent, no definite forward Depth conclusion '
    'was available: this is neither a veto nor a safety certificate. '
    'Never rotate merely to obtain a CLEAR label. '
    'Choose based on RGB, goals, and source-grounded motion receipts. '
    'Camera pitch is retained during motion, no automatic LOOK_UP. '
    'Fresh Depth uses current pitch before EACH forward. Downward '
    'views may miss distant obstacles; historical Depth cannot '
    'authorize motion. Explicit LOOK is model-owned. '
    'No measured metric pose, loop closure or confirmed translation. '
    'Command distance counts atomic steps, not arrival proof. '
    'Read memory.recent_views_at_current_stance for complete '
    'yaw/pitch coverage, repeated views and turns since last forward. '
    'New camera headings are observations, NOT forward progress. '
    'When turns repeat without progress, distinguish an unanswered '
    'visual question from rechecking the same scene; use the recent '
    'actual forward and blocked receipts to choose the next useful '
    'action. The model chooses; no forced move or scan ban. '
    'A visual semantic object box (e.g. HUMAN) is never a passage '
    'waypoint. Use only a separately grounded passage_box for numerical '
    'route bearings; unboxed route ideas remain prose. Stale bearings '
    'from a prior forward stance require current-frame re-grounding. '
    'Do not follow a past figure location as an entrance. '
    'Use memory.last_motion_decision as a factual action receipt: when '
    'forward actions visibly changed the scene and no new obstruction '
    'contradicts the route, consider continuing the same route rather '
    'than starting an unrelated scan. When a visibly continuous corridor '
    'supports it, you MAY choose a finite 0.5–1m (or longer) segment '
    'instead of only 0.25m per decision: each 0.25m atomic move still '
    'gets fresh Depth and interrupts on a measured block or unchanged RGB-D. '
    'In narrow stairs, close furniture, uncertain human passages, prefer '
    'a short segment or ask for specific new visual evidence; choose based '
    'on the actual scene, not a fixed distance rule. '
    'You alone choose the next action. '
    'You may reobserve changed scenes or dynamic people when useful. '
    'When a person, railing or doorway makes the local passage ambiguous, '
    'consider observe_for_goal with a specific question comparing the '
    'visible LEFT/RIGHT walkable corridors, foreground blockage and '
    'occluded floor; do not assume a pedestrian is static. Avoid repeated '
    'LOOK pitch switches when no new visual question would be answered. '
    'Pure turn or turn+LOOK without forward motion MUST use '
    'turn_and_observe, e.g. RIGHT 30 degrees, look=none. '
    'navigate_relative ALWAYS means execute positive forward distance '
    '(0.25..3m); do not use distance_m=0 to express a pure turn. '
    'Keep progress narratives as revisable hypotheses, distinguishing '
    'what was visually seen, what was physically attempted and what '
    'remains uncertain. Only request_finish success when the final '
    'destination is evidenced at the ROBOT\'s current location; official '
    'success is evaluated externally. '
    'Camera LOOK limits are read from camera_controls.allowed_look_now; '
    'a rejected LOOK is recoverable, not mission failure. '
    'Before committing to an action, output source-grounded '
    'observed_evidence first, then a short decision_analysis, and '
    'finally tool and args as the last fields in the JSON object. '
    'Use the evidence to assess what was actually seen before selecting '
    'a route; describe the rationale concisely rather than exposing '
    'private internal chain-of-thought. The tool and args are the sole '
    'executable decision. Use exactly one registered tool.'
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
        # Compute and log the same live preflight on every decision, but
        # attach it to NavigationAgent ONLY when the measured result is
        # conclusive. UNKNOWN and any other non-confirmed state are
        # deliberately omitted (not misrepresented as CLEAR/BLOCKED).
        # Atomic MOVE_FORWARD still performs its own fresh depth veto.
        preflight=r.forward_preflight(stage='navigation_decision')
        context=compact_navigation_context(navigation_context(r))
        if preflight.get('status') in ('CLEAR','BLOCKED'):
            context['front_depth_preflight']=preflight
        # Keep exactly the old v12 bounded mission context; attach at most
        # 3 source-verified historical RGB images as optional extras.
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
            for descriptor in navigation_visual_anchors(
                    r.memory,source['observation_id'],max_pairs=3):
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
            issue=next(Draft7Validator(schema).iter_errors(value),None)
            if issue is not None:
                issue=specific_schema_error(issue)
                field='.'.join(map(str,issue.absolute_path)) or 'response'
                raise APIError(
                    ('navigation_invalid_output_field:'+field+':'+
                     issue.message)[:350],category='structured_field')
            if value['tool'] in ('navigate_relative','turn_and_observe'):
                args=value['args']
                try:
                    signed_turn(args['turn_direction'],args['turn_degrees'])
                except ValueError as error:
                    raise APIError(str(error),category='structured_field') from None
        return r.client.complete(
            'navigation_agent','navigation_decision',MISSION_PROMPT,context,
            schema,validate=validate,image_url=images,
            format_retry=int(bool(format_feedback)),retry_safe=True,
            protocol=compact_navigation_protocol(tools),
            validation_schema=navigation_envelope(tools))
