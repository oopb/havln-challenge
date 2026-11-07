"""NavigationAgent: one whole instruction, local motion and memory updates."""
import math
from jsonschema import Draft7Validator
from .contracts import (navigation_schema, compact_navigation_protocol,
                        navigation_envelope)
from .contexts import navigation_context, compact_navigation_context
from .tool_menu import navigation_tool_menu
from .api_client import APIError
from .perception import image_url
from .heading import signed_turn

MISSION_PROMPT=(
    'You are NavigationAgent controlling a robot in a real RGB-D simulated scene. '
    'Your mission is the COMPLETE original language instruction, not a numbered '
    'list of mandatory milestones. You directly SEE the CURRENT real RGB image. '
    'Read memory.contract=mission_evidence_v1: one shared record of progress, '
    'authentic RGB-D source frames, headings/pitches, measured motion, '
    'obstacles and previous Observation Agent reports. '
    'Do not treat historic poses as the current stance. '
    'You own all route, movement, observation and STOP decisions. '
    'front_depth_preflight is a FRESH calibrated Depth check of the CURRENT '
    'front heading BEFORE this decision. CLEAR means no measured central '
    'obstruction, not guaranteed Habitat passage; BLOCKED includes actual '
    'central pixel ROI and distance; UNKNOWN is uncertain, NOT a veto. '
    'This says nothing about a heading you have not yet turned to. '
    'The motion executor physically levels a tilted camera before an '
    'actual forward and rechecks Depth at the NEW yaw and pitch; a tilted '
    'camera probe must never be treated as permanent route prohibition. '
    'Read observed_front_blocks for previously sensed blocked bearings '
    'where you correctly chose to turn away WITHOUT an attempted move. '
    'They are historical evidence only, not global route prohibitions. '
    'If the current direction is BLOCKED, use that measured evidence to '
    'choose another approach or ask Observation Agent to inspect the '
    'specific ROI; do not blindly repeat the same blocked forward command. '
    'When a direction is clear, make useful finite forward progress; do not '
    'spend the episode validating earlier stages. For unclear visual layout, '
    'call observe_for_goal using a natural-language request that includes the '
    'relevant completed progress and asks what route advances the overall goal. '
    'Observation Agent has full mission context and RGB-D; its descriptions and '
    'waypoints are suggestions, never automatic clearance or MOVE orders. '
    'Carry forward successful route evidence; correct outdated progress when '
    'newer visual or measured evidence conflicts. A historical block is local '
    'to its exact observation, stance and heading. If the current RGB-D at '
    'the SAME stance and heading repeatedly blocks forward motion, do not '
    'request the identical forward step again without new evidence or pose '
    'change. Instead ask Observation Agent to examine the blocked region '
    'and nearby alternatives, or select another direction. A visually '
    'visible stair is not proof that the front center is traversable. '
    'Keep your progress_narrative short and useful: what you believe is done '
    'and what remains; you may revise earlier beliefs. Keep measurements '
    'separate from interpretation and do not claim simulator ground truth. '
    'If no route is certain, choose an informed short exploratory action rather '
    'than indefinite validation. Only request_finish success on evidence that '
    'the ROBOT is beside the FINAL requested destination, not because an '
    'intermediate landmark is visible. Abort only when genuinely unable to '
    'proceed. Human obstacles are dynamic: respect live safety feedback. '
    'camera_controls gives CURRENT allowed_look_now and pitch bounds. '
    'The turn_and_observe tool LOOK enum is stable (none/up/down), '
    'but a LOOK that exceeds physical pitch limits is rejected BEFORE '
    'any TURN, as recoverable CAMERA_PITCH_LIMIT tool feedback. '
    'When this happens, read recent_camera_pitch_limits and choose '
    'another allowed action; do not keep requesting the same invalid LOOK. '
    'This is not task failure, and you keep all navigation authority. '
    'Tool turns use explicit turn_direction LEFT/RIGHT/NONE and NONNEGATIVE '
    'turn_degrees. RIGHT 30 physically turns right 30; LEFT 60 turns left '
    '60; NONE requires zero. NEVER output signed turn angles. Absolute yaw '
    'heading_deg increases on LEFT turns. A target box at image right means '
    'RIGHT of camera centerline. A waypoint with source_stance_matches_current '
    '=false or null current bearing is visual history only, never live '
    'motion clearance. No sg0/sg1 or VERIFY operation exists. '
    'Return exactly one tool + args and optionally progress_narrative, '
    'navigation_intent, uncertainties and evidence_observation_ids.'
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
        context=compact_navigation_context(navigation_context(r))
        # This scan must precede model reasoning, not only the attempted
        # atomic MOVE. Runtime._prepare_forward reuses the SAME calibrated
        # Depth policy on a fresh frame after any model-selected TURN.
        context['front_depth_preflight']=r.forward_preflight(
            stage='navigation_decision')
        if format_feedback:
            context['format_feedback']=format_feedback
        evidence=context['memory']
        r.event('mission_context_built',source='memory_projection',
            memory_contract=evidence['contract'],
            observation_task_ids=[v.get('observation_task_id')
                                  for v in evidence['observation_tasks']],
            historical_frame_ids=[v['observation_id']
                                  for v in evidence['historical_frames']],
            blocked_source_ids=[v['observation_id']
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
            images=image_url(frame,max_edge=640)
            context['current_rgb']=dict(
                image_attached=True,observation_id=source['observation_id'],
                heading_deg=round(math.degrees(pose['heading'])%360,2),
                pitch_deg=pose.get('pitch_deg'),rgb_hash=source['rgb_hash'],
                scope='current_public_RGB_not_historical')
            r.event('navigation_current_rgb_attached',source='navigation_agent',
                    source_observation_id=source['observation_id'],
                    rgb_hash=source['rgb_hash'])
        else:
            context['current_rgb']=dict(image_attached=False,
                                        scope='no_live_image_available')
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
        return r.client.complete(
            'navigation_agent','navigation_decision',MISSION_PROMPT,context,
            schema,validate=validate,image_url=images,
            format_retry=int(bool(format_feedback)),retry_safe=True,
            protocol=compact_navigation_protocol(tools),
            validation_schema=navigation_envelope(tools))
