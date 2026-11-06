"""A short OBSERVE decision followed by a bounded MOVE and another OBSERVE."""
from .contracts import navigation_schema, compact_navigation_protocol, navigation_envelope
from .contexts import navigation_context, compact_navigation_context
from .tool_menu import navigation_tool_menu
from .api_client import APIError
from jsonschema import Draft7Validator

COMMON_PROMPT = (
    'You are NavigationAgent. Own semantic subgoals, route decisions, motion and STOP. '
    'Observation Agent alone owns RGB-D camera exploration, geometry evidence '
    'interpretation and semantic milestone confirmation. '
    'Output exactly one registered JSON tool+args object. '
    'When evidence suggests the CURRENT subgoal may be complete, call '
    'observe_for_goal with mode=VERIFY. VERIFY always checks the runtime '
    'active milestone, not a former or future subgoal. For example, if '
    'sg_0 descent is confirmed and sg_1 requires a right turn, VERIFY asks '
    'whether the right turn was accomplished, NEVER whether stairs were '
    'descended again. If previous completion appears doubtful, do not '
    'quietly substitute a past milestone as the current VERIFY objective. '
    'Only a source-backed CONFIRMED VLM observation advances the subgoal. '
    'A partial descent is progress, not proof of arrival. '
    'A VERIFY NOT_YET is a completed assessment of an unchanged scene: '
    'do not immediately re-VERIFY by paraphrasing the same question. '
    'Continue the CURRENT subgoal by useful movement or targeted '
    'OBSTACLE_REVIEW/APPROACH when current movement is blocked. '
    'On UNCERTAIN choose the specific missing evidence. Only a '
    'CONFIRMED milestone may switch to NEXT subgoal; future subgoal '
    'instructions must not drive the still-unfinished movement. '
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
    'The Observation Agent navigation_feedback carries a measured/visual '
    'summary AND a separate advisory. Old subgoal observations are '
    'historical and must not replace the CURRENT milestone. '
    'Recommendations are proposals, 
    'not verified collision clearance or orders. '
    'When a registered forward step shows credible vertical progress '
    'opposite the current task intent, reconsider heading before issuing '
    'another long forward segment. After a new route change or when '
    'direction is uncertain, prefer a short calibrated test step and '
    'review measured RGB-D progress before committing multiple steps. '
    'This is model advice, never a mandatory hard-coded action limit. '
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
PANORAMA_REVIEW_PROMPT = (
    'ONE-TIME COMPLETE-VIEW REVIEW: The robot has inspected directions around '
    'its current stance. Compare earlier visual bearings, registered real '
    'motion and blocked directions. Choose the most useful next action '
    'from the entire available tool menu. A measured obstacle at one '
    'bearing does not block other unmeasured bearings. Moving is an option, '
    'NOT an obligation: you may VERIFY a subgoal, conduct a genuinely '
    'different observation, choose a finite move, or revise the task. '
    'Do not repeat the same unchanged scan or invent a collision-free route. '
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
            # A LOOK at the mechanical pitch limit is an impossible optional
            # pre-scan setting, not an intention to change the route. Normalize
            # it to "VLM chooses its next legal view" before tool-schema
            # validation, rather than paying for repeated INVALID_ARGUMENTS.
            if isinstance(value,dict) and value.get('tool')=='observe_for_goal':
                args=value.get('args')
                if isinstance(args,dict) and args.get('look') in ('up','down'):
                    allowed=set((context.get('legal_look') or []))
                    if args['look'] not in allowed:
                        requested=args.pop('look')
                        if hasattr(runtime,'event'):
                            runtime.event('navigation_invalid_initial_look_removed',
                                source='navigation_agent',
                                requested_look=requested,
                                current_pitch_deg=runtime.pitch_deg,
                                subgoal_id=getattr(runtime,'active_subgoal_id',None),
                                recovery='observation_agent_chooses_legal_view')
            # Other malformed arguments are rejected before dispatch; they
            # cannot consume a full NavigationAgent decision in ToolExecutor.
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
        # A full VLM-owned sweep is precisely when this advice is useful:
        # do not suppress the review just because observation was the last tool.
        panorama_review=(
            bool(panorama.get('all_azimuths_observed')) and not orientation
            and getattr(self,'_panorama_committed_signature',None)!=signature)
        # A completed panorama changes the ADVICE, never the tool whitelist.
        offered_tools=navigation_tool_menu(runtime.registry,context,
            orientation=bool(orientation))

        # A one-shot strategic checkpoint must not be repeated every turn
        # while the physical stance is unchanged. A later new subgoal/stance
        # can legitimately receive a new checkpoint.
        first_checkpoint=(
            not panorama_review
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
        result=runtime.client.complete(
            'navigation_agent', 'navigation_decision',
            COMMON_PROMPT + OBSERVE_PROMPT +
            (PANORAMA_REVIEW_PROMPT if panorama_review else advisory_prompt),
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
        if panorama_review:
            self._panorama_committed_signature=signature
            if hasattr(runtime,'event'):
                runtime.event('panorama_review_prompt_issued',
                    source='navigation_agent',physically_forced=False,
                    offered_tools=list(offered_tools),
                    selected_tool=result['tool'])
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
