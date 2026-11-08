"""NavigationAgent: one whole instruction, local motion and memory updates."""
import math
from jsonschema import Draft7Validator
from .contracts import (navigation_schema, compact_navigation_protocol,
                        navigation_envelope)
from .contexts import navigation_context, compact_navigation_context
from .tool_menu import navigation_tool_menu
from .api_client import APIError
from .perception import image_url, depth_image_url
from .observation import array_hash
from .navigation_history import navigation_visual_anchors
from .heading import box_center_bearing
from .heading import signed_turn

MISSION_PROMPT=(
    'You are NavigationAgent. Complete the ORIGINAL full navigation mission. '
    'Only you choose MOVE, observation, replanning and STOP; ObservationAgent '
    'supplies visual evidence, never movement authorization. '
    'Read context.episode_history: the full ordered causal record of ALL '
    'earlier navigation decisions, tool outcomes, atomic action receipts, '
    'observation checkpoints and a source-indexed RGB-D catalog. '
    'Historical RGB-D image pairs attached to this request are labeled in '
    'context.history_media.manifest by exact image index and observation ID. '
    'The FIRST pair is the CURRENT RGB-D; other pairs are HISTORICAL only. '
    'Compare early and recent static scene structure when deciding whether '
    'you may be returning to a previously explored route, but never assert '
    'physical place identity from image resemblance alone. '
    'Historical relative left/right phrases are valid ONLY at their original '
    'source observation and camera heading; progress.prior_navigation_intent '
    'explicitly marks freshness. After turning or moving, reevaluate any '
    'old direction against the CURRENT RGB-D and pose. '
    'For ALIGN_VISIBLE_TARGET you MUST ground a target box in the CURRENT '
    'RGB image, then choose the turn direction that moves its center toward '
    'the CURRENT optical axis. If you intentionally explore elsewhere '
    '(for example to bypass an obstacle), declare EXPLORE_ALTERNATIVE and '
    'explain the evidence rather than saying you are aligning the other way. '
    'Do not invent a target box. With no visible localized target, choose '
    'an explicit exploratory mode or ask ObservationAgent for evidence. '
    'The current front_depth_preflight is calibrated, live, directional '
    'Depth evidence; BLOCKED confirms a measured front obstruction, '
    'UNKNOWN is not a veto, CLEAR does not guarantee physical passage. '
    'The executor will level the camera and repeat Depth preflight for '
    'each actual forward step after any turn. Historical Depth cannot '
    'authorize new motion. Never override measured current block evidence. '
    'No metric robot x/y/z, measured height, true place loop closure or '
    'confirmed displacement is available. Command distance means a finite '
    'count of atomic forward actions, NOT proof of arrival or descent. '
    'Use route/observation history to avoid redundant re-scanning, but you '
    'may reobserve changed scenes or dynamic people when useful. '
    'Keep progress narratives as revisable hypotheses, distinguishing '
    'what was visually seen, what was physically attempted and what '
    'remains uncertain. Only request_finish success when the final '
    'destination is evidenced at the ROBOT\'s current location; official '
    'success is evaluated externally. '
    'Camera LOOK limits are read from camera_controls.allowed_look_now; '
    'a rejected LOOK is recoverable, not mission failure. '
    'Use exactly one registered tool and obey its output protocol.'
)

def validate_direction_intent(value, current_observation_id,
                              camera_hfov, turn_step_deg):
    """Check MODEL-DECLARED grounding consistency, not semantic policy.

    This cannot prove a visual object exists or that a path is walkable.
    If a model says ALIGN a specific CURRENT image box, its turn cannot
    point away from that box. EXPLORE_ALTERNATIVE remains model-autonomous.
    Never silently flip LEFT to RIGHT or infer intent from English words.
    """
    if value.get('tool') not in ('turn_and_observe','navigate_relative'):
        return
    binding=value.get('direction_intent')
    if not isinstance(binding,dict):
        raise APIError('navigation_direction_intent_required_for_motion',
                       category='structured_field')
    mode=binding.get('mode')
    if mode!='ALIGN_VISIBLE_TARGET':
        return
    if binding.get('target_observation_id')!=current_observation_id:
        raise APIError(
            'align_target_requires_current_RGB_observation_id_not_historical',
            category='structured_field')
    box=binding.get('target_box_xyxy_norm')
    if box is None:
        raise APIError(
            'align_visible_target_requires_grounded_current_RGB_box',
            category='structured_field')
    try:
        offset=box_center_bearing(box,camera_hfov)
    except (ValueError,TypeError) as error:
        raise APIError('align_target_invalid_normalized_box:'+str(error),
                       category='structured_field') from None
    args=value['args']
    signed=signed_turn(args['turn_direction'],args['turn_degrees'])
    # Sub-actuator offsets are genuinely ambiguous for a 15-degree robot
    # and should not be treated as a hard steering restriction.
    if abs(offset)>=max(2.,turn_step_deg*.5) and signed*offset<0:
        expected='LEFT' if offset>0 else 'RIGHT'
        raise APIError(
            'direction_intent_turn_away_from_current_RGB_target:'
            'target_observation_id='+str(current_observation_id)+
            ',box_center_offset_deg='+str(offset)+
            ',align_direction='+expected+
            ',commanded='+args['turn_direction']+
            '; either align toward target OR explicitly choose '
            'EXPLORE_ALTERNATIVE with a reason',
            category='structured_field')


class NavigationAgent:
    def __init__(self,runtime):
        self.runtime=runtime

    def decide(self,format_feedback=None):
        r=self.runtime
        if r.navigation_mode!='OBSERVE':
            raise RuntimeError('navigation_decision_requires_observe_mode')
        r.registry=r.make_registry()
        tools=navigation_tool_menu(r.registry)
        # Depth preflight may REFRESH the current camera frame. Build the
        # entire source-indexed episode/history context only AFTER it so
        # historical catalog, current observation ID and RGB-D hashes agree.
        preflight=r.forward_preflight(stage='navigation_decision')
        context=compact_navigation_context(navigation_context(r))
        context['front_depth_preflight']=preflight
        # A text-complete episode history is present even when historical
        # RGB-D files are archived sparsely. Real image bytes are attached
        # below only after source-hash verification.
        if format_feedback:
            context['format_feedback']=format_feedback
        evidence=context['memory']
        r.event('mission_context_built',source='memory_projection',
            memory_contract=evidence['contract'],
            observation_task_ids=[v.get('observation_task_id')
                                  for v in evidence['observation_tasks']],
            historical_frame_ids=[v['observation_id']
                                  for v in evidence['historical_frames']],
            # Shared mission_evidence projections expose source-anchored
            # obstacle facts as source_observation_id (not observation_id).
            # Keep this diagnostic keyed to the public history contract:
            # the event must never crash NavigationAgent before inference.
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
            images=[
                image_url(frame,max_edge=640),
                depth_image_url(frame,r.camera,
                                max_edge=640,display_max_m=5.0)]
            manifest=[
                dict(image_index=0,kind='current_RGB',
                     observation_id=source['observation_id'],
                     heading_deg=round(math.degrees(pose['heading'])%360,2),
                     stance_id=pose.get('stance_id'),
                     current_sensor_frame=True),
                dict(image_index=1,kind='current_Depth',
                     observation_id=source['observation_id'],
                     depth_colormap='near_black_far_white_invalid_magenta',
                     current_sensor_frame=True)]
            by_id={o['observation_id']:o for o in r.memory.observations}
            for descriptor in navigation_visual_anchors(
                    r.memory,source['observation_id'],max_pairs=4):
                identifier=descriptor['observation_id']
                item=by_id.get(identifier)
                if item is None:continue
                try:
                    historical=(r.load_observation(item)
                        if r.load_observation else r.frames[identifier])
                    if (array_hash(historical.rgb)!=item.get('rgb_hash')
                            or array_hash(historical.depth)!=item.get('depth_hash')):
                        continue
                    # Attached text index matches exactly the order of
                    # serialized image_url fields in the wire request.
                    pos=len(images)
                    images.extend([
                        image_url(historical,max_edge=640),
                        depth_image_url(historical,r.camera,max_edge=640,
                                        display_max_m=5.0)])
                    history_pose=item.get('pose') or {}
                    for index,kind in ((pos,'historical_RGB'),
                                       (pos+1,'historical_Depth')):
                        manifest.append(dict(
                            image_index=index,kind=kind,
                            observation_id=identifier,
                            source_action_step=item.get('step'),
                            source_heading_deg=round(math.degrees(
                                history_pose.get('heading',0))%360,2),
                            source_pitch_deg=history_pose.get('pitch_deg'),
                            source_stance_id=history_pose.get('stance_id'),
                            selection_reason=descriptor['selection_reason'],
                            current_sensor_frame=False,
                            historical_only_not_live_clearance=True,
                            physical_revisit_not_verified=True))
                except (KeyError,OSError,ValueError,TypeError):
                    # A sparse diagnostic archive must not crash live nav.
                    continue
            context['history_media']=dict(
                manifest=manifest,
                actual_image_count=len(images),
                complete_text_history_is_independent_of_image_sampling=True,
                omitted_historical_image_bytes_are_indexed_in_episode_history=True,
                image_order='one pair current RGB/Depth then selected historical RGB/Depth',
                historical_depth_never_authorizes_motion=True)
            context['current_rgb']=dict(
                image_attached=True,observation_id=source['observation_id'],
                heading_deg=round(math.degrees(pose['heading'])%360,2),
                pitch_deg=pose.get('pitch_deg'),rgb_hash=source['rgb_hash'],
                scope='current_public_RGB_not_historical')
            r.event('navigation_current_rgb_attached',source='navigation_agent',
                    source_observation_id=source['observation_id'],
                    rgb_hash=source['rgb_hash'],
                    total_attached_images=len(images),
                    historical_pairs=(len(images)-2)//2)
        else:
            context['current_rgb']=dict(image_attached=False,
                                        scope='no_live_image_available')
            context['history_media']=dict(
                manifest=[],actual_image_count=0,
                complete_text_history_is_independent_of_image_sampling=True)
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
                    signed_turn(args['turn_direction'],args['turn_degrees'])
                except ValueError as error:
                    raise APIError(str(error),category='structured_field') from None
                validate_direction_intent(
                    value,(r.memory.current or {}).get('observation_id'),
                    r.camera.hfov,r.camera.turn_angle)
        return r.client.complete(
            'navigation_agent','navigation_decision',MISSION_PROMPT,context,
            schema,validate=validate,image_url=images,
            format_retry=int(bool(format_feedback)),retry_safe=True,
            protocol=compact_navigation_protocol(tools),
            validation_schema=navigation_envelope(tools))
