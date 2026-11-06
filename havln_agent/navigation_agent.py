"""A short OBSERVE decision followed by a bounded MOVE and another OBSERVE."""
from .contracts import navigation_schema, compact_navigation_protocol, navigation_envelope
from .contexts import navigation_context, compact_navigation_context
from .tool_menu import navigation_tool_menu
from .api_client import APIError
from .panorama_memory import decide_with_panorama
from jsonschema import Draft7Validator

COMMON_PROMPT = (
    'You are NavigationAgent. Own semantic subgoals, route decisions, motion and STOP. '
    'Observation Agent alone owns RGB-D camera exploration, geometry evidence '
    'interpretation and semantic milestone confirmation. '
    'Output exactly one registered JSON tool+args object. '
    'When evidence suggests the current subgoal may be complete, call '
    'observe_for_goal with mode=VERIFY and a concrete question, such as whether '
    'the ROBOT is physically at the bottom of stairs rather than mid-descent. '
    'Only a source-backed CONFIRMED VLM observation advances the subgoal. '
    'A partial descent is progress, not proof of arrival. '
    'On NOT_YET continue motion; on UNCERTAIN choose useful new evidence. '
    'No independent Geometry or fixed Python stair verifier is offered. '
    'Never request_finish success until ALL milestones are VLM confirmed.'
)
OBSERVE_PROMPT = (
    'MODE: OBSERVE. Delegate one meaningful Observation Agent task: SEARCH '
    '(find target), APPROACH (find local approach), OBSTACLE_REVIEW (detour after '
    'blocked forward) or VERIFY (check semantic subgoal completion). '
    'Observation Agent sees real current RGB, calibrated Depth, measured '
    'geometry facts and successful/blocked motion history; it controls TURN/LOOK '
    'and retains all prior RGB-D frames, replies and feedback in the same task. '
    'Do not paraphrase the same unanswered question at unchanged pose/RGB-D. '
    'Read latest_observation_task, route_progress and progress_review. '
    'Successful registered movement remains historical progress evidence; '
    'a later BLOCKED step only blocks the current stance/direction and does '
    'not erase the route already traversed. Compare headings with observed '
    'vertical movement: a view opposite the successful descent could be ascent. '
    'On useful candidate choose navigate_relative with bearing and finite '
    'distance; do not repeatedly re-scan between successful atomic steps. '
    'Do not call independent Geometry. VLM combines depth measurements and '
    'visual evidence, but neither an UNKNOWN depth nor a clear-looking image '
    'certifies a free route. Only a fresh coherent central measured obstacle '
    'halts an atomic forward; then ask OBSTACLE_REVIEW for another approach. '
    'Never claim arrival from commanded actions alone or force movement through '
    'a measured central barrier. Keep subgoal until VLM confirms its endpoint.'
)
# This is a model-visible *decision focus*, not a tool whitelist or a hidden
# physical control action. The agent still owns the bearing and can explain
# missing live safety rather than inventing a safe corridor.
OBSERVATION_INTERVENTION_PROMPT = (
    'OBSERVATION-LOOP DECISION CHECKPOINT (issued ONCE per task/stance): '
    'Existing yaw/pitch views have produced little actionable information. '
    'Historical views may suggest a plausible bearing, but this is '
    'NOT lasting forward permission. Choose a task-relevant short movement '
    'when evidence supports it, OR use a different purposeful action if '
    'the available evidence is insufficient. YOU choose the direction, '
    'route, target and whether new observations are genuinely needed. '
    'Repeating the same absolute view without changed evidence is unlikely '
    'to help. Ask Observation Agent for new RGB-D and measured geometry '
    'only when a specific unanswered obstacle question remains. '
    'Safety depends on fresh measured central obstacles; an old view is '
    'not enduring forward clearance. Do NOT fabricate a free corridor. '
)
BLOCKED_OBSERVATION_INTERVENTION_PROMPT = (
    'OBSERVATION-LOOP / NO CONFIRMED SAMPLED BEARING (issued ONCE per task/stance): '
    'Repeated same-stance observations have not established a currently '
    'executable direction AMONG MEASURED bearings; other headings are UNMEASURED '
    'and not known to be blocked. Do NOT call a turning-only MOVE a successful forward '
    'attempt, and do NOT invent clearance. Choose the next task-relevant '
    'strategy yourself: a genuinely different local goal/detour, a specific '
    'new VLM RGB-D observation that could change the decision, a semantic '
    'task revision, or an explicitly evidence-backed movement direction '
    'for the simulator to execute. Do not merely toggle between already '
    'seen pitch levels or arbitrarily change relative headings. The model '
    'retains full planning authority; geometry is a VLM evidence input only. '
)
MOVE_EXECUTION_RULES = (
    'MODE: MOVE is executed without a new LLM request by navigate_local. '
    'LocalMotionController uses post-action RGB-D odometry for each forward action, '
    'continues until its finite forward count completes or a no-motion, '
    'sensor or Episode interruption; metric arrival is never assumed. '
    'The next NavigationAgent decision is always OBSERVE.'
)

class NavigationAgent:
    def __init__(self, runtime):
        self.runtime = runtime

    def decide(self, format_feedback=None):
        runtime = self.runtime
        if getattr(runtime, 'navigation_mode', 'OBSERVE') != 'OBSERVE':
            raise RuntimeError('navigation_decision_requires_observe_mode')
        runtime.registry = runtime.make_registry()
        def validate(value):
            # Some low-cost providers obey only the envelope/tool names and
            # ignore per-tool argument schemas. Reject malformed angle/args
            # during the decision so the model can repair instead of burning
            # cycles on INVALID_ARGUMENTS from ToolExecutor.
            if not Draft7Validator(navigation_schema(offered_tools)).is_valid(value):
                raise APIError('navigation_tool_arguments_violate_registered_schema',
                               category='structured_field')
            # Optional metadata is outside the navigation command contract.
            # It may be malformed; only tool+args can authorize a motion.
        context = dict(navigation_context(runtime),
                       **({'format_feedback': format_feedback} if format_feedback else {}))
        stagnation=context.get('observation_stagnation') or {}
        advisory=stagnation.get('intervention')
        panorama=context.get('panoramic_observation') or {}
        # Do not force translational movement for an orientation-only
        # instruction whose completion should be independently verified.
        orientation=(context.get('orientation') or {}).get('requirement')
        pose=context.get('pose') or {}
        task=context.get('task_selection') or {}
        xy=pose.get('position_xz_m_est') or ()
        signature=(getattr(runtime.memory,'episode_id',None),
                   getattr(runtime,'plan_revision',None),
                   task.get('current_subgoal_index'),
                   pose.get('motion_epoch'),pose.get('place_id'),
                   tuple(round(float(v),1) for v in xy))
        if (panorama.get('all_azimuths_observed')
                and getattr(self,'_panorama_committed_signature',None)==signature):
            # One model-authored route attempt has already been issued at
            # this stance: don't issue a second, conflicting stall advisory.
            advisory=None
        vlm_observation_just_finished=(
            (context.get('last_tool') or {}).get('observation_owner')
            == 'visual_perception_vlm')
        full_circle_commit=(
            bool(panorama.get('all_azimuths_observed')) and not orientation
            and not vlm_observation_just_finished
            and getattr(self,'_panorama_committed_signature',None)!=signature
            and 'navigate_relative' in runtime.registry)
        # Keep the execution registry intact, but advertise only strategic
        # tools relevant to this decision. VLM owns all camera search steps;
        # specialized historical/pixel/geometry tools reappear on evidence.
        offered_tools=navigation_tool_menu(
            runtime.registry,context,orientation=bool(orientation),
            full_circle_commit=full_circle_commit)

        # A one-shot strategic checkpoint must not be repeated every turn
        # while the physical stance is unchanged. A later new subgoal/stance
        # can legitimately receive a new checkpoint.
        first_checkpoint=(
            not full_circle_commit
            and advisory in ('DIRECTIONAL_ACTION_DECISION_REQUIRED',
                             'BLOCKED_ACTIONABILITY_REASSESSMENT')
            and getattr(self,'_intervention_signature',None)!=signature)
        advisory_prompt=(
            OBSERVATION_INTERVENTION_PROMPT
                if first_checkpoint and advisory=='DIRECTIONAL_ACTION_DECISION_REQUIRED'
            else BLOCKED_OBSERVATION_INTERVENTION_PROMPT
                if first_checkpoint and advisory=='BLOCKED_ACTIONABILITY_REASSESSMENT'
            else '')
        if advisory and not first_checkpoint:
            stagnation['intervention_delivery']='previously_issued_or_pending_route'
        if full_circle_commit:
            result,visual,reviewed_views=decide_with_panorama(
                self,context,signature,validate,offered_tools)
        else:
            result=runtime.client.complete(
                'navigation_agent', 'navigation_decision',
                COMMON_PROMPT + OBSERVE_PROMPT + advisory_prompt,
                compact_navigation_context(dict(
                    context,instruction=getattr(
                        getattr(runtime.memory,'task',None),'instruction',
                        context.get('instruction')))),
                navigation_schema(offered_tools), validate=validate,
                format_retry=int(bool(format_feedback)), retry_safe=True,
                protocol=compact_navigation_protocol(offered_tools),
                validation_schema=navigation_envelope(offered_tools))
        # Always discard any model-authored side-channel memory before tool
        # dispatch. VLM and executor persist grounded facts independently.
        # Invalid tool/args remain fatal through the strict model validator.
        result={'tool':result['tool'],'args':result['args']}
        if full_circle_commit:
            # Compact/no-native-schema providers may ignore enumerated tools.
            # Reject rather than execute an arbitrary TURN or spend another
            # 200 decisions on an identical invalid tool call. scheduler.request
            # can issue one structured-field correction with fresh context.
            if not Draft7Validator(navigation_schema(offered_tools)).is_valid(result):
                raise APIError(
                    'full_circle_requires_schema_valid_navigate_relative_turn_deg_and_distance_m',
                    category='structured_field')
            self._panorama_committed_signature=signature
            if hasattr(runtime,'event'):
                runtime.event('panorama_direction_committed',
                    source='navigation_agent',
                    selected_tool='navigate_relative',
                    selected_turn_deg=result['args']['turn_deg'],
                    selected_distance_m=result['args']['distance_m'],
                    visual_selected_observation_id=visual['selected_observation_id'],
                    reviewed_frame_ids=[v['observation_id'] for v in reviewed_views],
                    observed_yaw_bins=panorama.get('covered_yaw_bins'),
                    physically_forced=False,
                    live_safety_revalidation_required=False,
                    geometry_autorun=False)
        if first_checkpoint:
            self._intervention_signature=signature
            if hasattr(runtime,'event'):
                runtime.event('model_direction_intervention_requested',
                    source='navigation_agent',
                    intervention=advisory,
                    observed_calls=stagnation['same_stance_observation_calls'],
                    low_information_calls=stagnation['low_information_calls'],
                    physically_forced=False)
        return result
