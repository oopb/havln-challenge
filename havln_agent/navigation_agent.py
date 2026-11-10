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
from .heading import signed_turn, resolve_target_heading

MISSION_PROMPT=(
    "You are NavigationAgent. Complete the ORIGINAL full navigation instruction. Only you choose movement, observation, replanning and STOP; ObservationAgent supplies RGB-D evidence, never motion authorization. "
    "Read memory (mission_evidence_v1), current RGB, recent action receipts, source-linked visual routes and same-stance yaw/pitch coverage. Omitted old evidence may still exist in Memory. "
    "The FIRST image is CURRENT RGB; history_media.manifest labels later historical images with source heading and stance. A historical LEFT/RIGHT phrase, Depth reading or apparent place match is not valid live steering/clearance. Re-ground the next action in the CURRENT image and pose. "
    "Choose a turn from CURRENT visual evidence. For a visibly RELATIVE turn, choose turn_direction LEFT, RIGHT, or NONE plus nonnegative turn_degrees (NONE=0); look=up/down/none independently controls camera pitch. "
    "When your intended target is a NUMERICAL absolute heading, use target_heading_deg (0..360) as the ONLY turn specification. The executor uses CURRENT yaw to calculate the shortest calibrated LEFT/RIGHT turn. Do NOT mentally convert absolute headings to left/right, and never mix target_heading_deg with turn_direction/turn_degrees. The model still chooses the goal heading and whether to move. "
    "Camera pitch is model-controlled and stays at the chosen angle during forward motion; there is no automatic LOOK_UP or level-camera action. Vertical LOOK is a useful observation, not a movement penalty. The same horizontal heading at a different pitch reveals different information: for nearby floor and stair edges consider LOOK_DOWN; for a distant hallway, doorway or landmark obscured in a down-tilted view consider LOOK_UP. "
    "Choose a new pitch when it answers a concrete missing visual question, not on a fixed schedule. You can look vertically in place with turn_and_observe(turn_direction=NONE, turn_degrees=0, look=up/down), or use a target_heading_deg with look. After looking down to inspect steps, consider looking up again to recover distant route context when needed. A LOOK is not evidence of physical translation or descent. "
    "When a person, railing, stairs, doorway or occlusion makes the route ambiguous, consider observe_for_goal with a focused request to compare near-floor structure, far passage, left/right gaps and useful pitch views from live RGB-D. ObservationAgent may look and turn within its session; it cannot authorize motion. "
    "Current front_depth_preflight appears ONLY for measured CLEAR or BLOCKED. BLOCKED means measured current occupancy in that direction; CLEAR is not a full-path certificate. If absent, there is no conclusive result: neither a movement veto nor guaranteed clearance. Never rotate merely to obtain CLEAR. Every actual forward action gets its own fresh calibrated Depth check at the current pitch; historical Depth cannot authorize motion. "
    "Use memory.recent_views_at_current_stance.full_coverage_by_pitch and turn_actions_since_last_forward to distinguish genuinely new yaw/pitch views from repeated horizontal scanning. A 360-degree yaw sweep at one pitch does NOT mean other pitches were observed. New camera angles are observations, NOT forward progress; do not repeatedly revisit an unchanged view without an unanswered question. There is no forced scan or forced forward rule. "
    "Route hypotheses and passage regions are visual clues, never clearance proof; a human/object box is not an entry. The current source and latest actual motion or obstacle receipts outrank stale bearings. If prior forward visibly changed the scene and the route still appears viable, consider continuing it rather than restarting a scan. "
    "Commanded distance (0.25..3m) is a finite set of atomic forward steps, NOT measured displacement, stair descent or arrival. In stable visible corridors the model may choose a longer short cruise; in crowds, steps or uncertainty choose a shorter move or purposeful observation. There is no metric x/y/z, proven loop closure or confirmed height change. "
    "Use turn_and_observe for a pure TURN and/or LOOK; navigate_relative requires positive distance_m (never 0). Both support an absolute target heading OR relative direction and magnitude, not both. A camera pitch limit is recoverable; consult camera_controls.allowed_look_now and select useful legal LOOK actions. "
    "Keep progress narratives revisable: distinguish visual evidence, actions actually executed and uncertainty. Only request_finish success when CURRENT evidence supports arriving at the FINAL mission destination; scoring is external. "
    "Output source-grounded observed_evidence first, concise decision_analysis next, and tool and args LAST. They alone execute. Return one legal registered tool; describe the short evidence and action rationale, not private internal chain-of-thought."
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
            if not Draft7Validator(schema).is_valid(value):
                raise APIError('navigation_tool_arguments_violate_registered_schema',
                               category='structured_field')
            if value['tool'] in ('navigate_relative','turn_and_observe'):
                args=value['args']
                try:
                    if 'target_heading_deg' in args:
                        resolve_target_heading(args['target_heading_deg'],
                            math.degrees(r.motion_controller.pose()['heading']),
                            r.camera.turn_angle)
                    else:
                        signed_turn(args['turn_direction'],args['turn_degrees'])
                except ValueError as error:
                    raise APIError(str(error),category='structured_field') from None
        proposal=r.client.complete(
            'navigation_agent','navigation_decision',MISSION_PROMPT,context,
            schema,validate=validate,image_url=images,
            format_retry=int(bool(format_feedback)),retry_safe=True,
            protocol=compact_navigation_protocol(tools),
            validation_schema=navigation_envelope(tools))
        # A target yaw is model intent, but left/right conversion must NEVER
        # depend on LLM heading arithmetic. Normalize before existing dispatch,
        # so the physical executor and all legacy relative tools stay intact.
        if (proposal['tool'] in ('navigate_relative','turn_and_observe')
                and 'target_heading_deg' in proposal['args']):
            args=dict(proposal['args'])
            target=args.pop('target_heading_deg')
            source=r.motion_controller.pose()
            current=math.degrees(source['heading'])%360
            direction,degrees=resolve_target_heading(
                target,current,r.camera.turn_angle)
            args['turn_direction']=direction
            args['turn_degrees']=degrees
            r.event('navigation_absolute_heading_resolved',
                source='deterministic_heading',
                source_observation_id=r.memory.current['observation_id'],
                current_heading_deg=round(current,3),
                model_target_heading_deg=target,
                resolved_turn_direction=direction,
                resolved_turn_degrees=degrees,
                expected_heading_deg=round((current+signed_turn(
                    direction,degrees))%360,3))
            proposal=dict(proposal,args=args)
        return proposal
