"""NavigationAgent: one whole instruction, local motion and memory updates."""
import math
from jsonschema import Draft7Validator
from .contracts import (navigation_schema, compact_navigation_protocol,
                        navigation_envelope)
from .contexts import navigation_context, compact_navigation_context
from .tool_menu import navigation_tool_menu
from .api_client import APIError
from .perception import image_url

MISSION_PROMPT=(
    'You are NavigationAgent controlling a robot in a real RGB-D simulated scene. '
    'Your mission is the COMPLETE original language instruction, not a numbered '
    'list of mandatory milestones. You directly SEE the CURRENT real RGB image. '
    'Read the natural-language progress narrative, intent, uncertainties, '
    'observations, measured motion and blocked bearings from Memory. '
    'You own all route, movement, observation and STOP decisions. '
    'When a direction is clear, make useful finite forward progress; do not '
    'spend the episode validating earlier stages. For unclear visual layout, '
    'call observe_for_goal using a natural-language request that includes the '
    'relevant completed progress and asks what route advances the overall goal. '
    'Observation Agent has full mission context and RGB-D; its descriptions and '
    'waypoints are suggestions, never automatic clearance or MOVE orders. '
    'Carry forward successful route evidence; correct outdated progress when '
    'newer visual or measured evidence conflicts. A direction with a historical '
    'obstacle is not universally forbidden, and a historical clear direction is '
    'not guaranteed clear now. Do not keep rescanning an unchanged scene. '
    'Keep your progress_narrative short and useful: what you believe is done '
    'and what remains; you may revise earlier beliefs. Keep measurements '
    'separate from interpretation and do not claim simulator ground truth. '
    'If no route is certain, choose an informed short exploratory action rather '
    'than indefinite validation. Only request_finish success on evidence that '
    'the ROBOT is beside the FINAL requested destination, not because an '
    'intermediate landmark is visible. Abort only when genuinely unable to '
    'proceed. Human obstacles are dynamic: respect live safety feedback. '
    'No sg0/sg1 numbering or VERIFY operation exists. '
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
        if format_feedback:
            context['format_feedback']=format_feedback
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
            if not Draft7Validator(schema).is_valid(value):
                raise APIError('navigation_tool_arguments_violate_registered_schema',
                               category='structured_field')
        return r.client.complete(
            'navigation_agent','navigation_decision',MISSION_PROMPT,context,
            schema,validate=validate,image_url=images,
            format_retry=int(bool(format_feedback)),retry_safe=True,
            protocol=compact_navigation_protocol(tools),
            validation_schema=navigation_envelope(tools))
