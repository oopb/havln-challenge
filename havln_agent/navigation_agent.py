"""A short OBSERVE decision followed by a bounded MOVE and another OBSERVE."""
from .contracts import navigation_schema, compact_navigation_protocol, navigation_envelope
from .contexts import navigation_context, compact_navigation_context
from .api_client import APIError
from .panorama_memory import decide_with_panorama
from jsonschema import Draft7Validator

COMMON_PROMPT = (
    'You are NavigationAgent. TaskPlanner provides ordered semantic subgoals; '
    'you control strategic observation, motion, subgoal selection and STOP. '
    'You see concise JSON, not RGB. VLM owns all goal-directed camera search '
    'inside observe_for_goal; never micro-manage its 15-degree turns. '
    'Return ONE JSON {"tool":"name","args":{...}} with optional evidence-linked '
    'memory_delta. Use only authentic observation IDs and registered tools. '
    'Unknown RGB-D odometry is not evidence of a collision and does not stop '
    'a finite movement plan. A completed atomic action plan is NOT verified '
    'physical arrival. Verify milestones from real visual/motion evidence; '
    'select_subgoal may provisionally advance but cannot certify success. '
    'Do not invent free space, target coordinates or completion; '
    'never request_finish success without independent proof. '
)
OBSERVE_PROMPT = (
    'MODE: OBSERVE. Decide the NEXT useful high-level tool, not a sequence '
    'of tiny camera turns. If visual evidence or a target approach is missing, '
    'delegate ONCE to observe_for_goal(objective,target_types,sweep_degrees?) '
    'with open-vocabulary targets; OMIT direction/checkpoint_degrees and let '
    'the VLM own view selection, pitch and search termination. Read its '
    'best_candidate and best_cue. If a target entry is already localized, '
    'prefer acting on that evidence rather than scanning a full circle again. '
    'If evidence permits movement, you choose navigate_relative(turn_deg,'
    'distance_m) or navigate_to_pixel; the executor performs turns and '
    'forward atomic steps continuously and returns evidence. A 0.25m atomic '
    'forward action is NOT the segment distance limit. '
    'If local_navigation remains useful, resume the SAME waypoint_id via '
    'navigate_local. DO NOT take max_atomic_steps. '
    'Read navigation_memory, candidate_waypoints, current_view, grounding '
    'and recent motion before deciding. Prefer a measured goal-approaching '
    'direction when such evidence exists; grounding is PARTIAL or UNKNOWN '
    'is NOT a movement prohibition. Unknown depth is not measured collision; '
    'UNCONFIRMED is NOT BLOCKED. Never bypass a measured collision. '
    'Geometry is optional and only model-invoked via inspect_geometry; '
    'normal MOVE does not run an automatic geometry gate. '
    'NEEDS_VIEW means a specific missing view; REQUIRES_ATTENTION after '
    'unchanged fresh RGB-D means reassess your strategy, not repeat a '
    'static sweep. You may seek additional evidence or plan a short detour. '
    'ACTIONS_COMPLETED means commanded steps executed, not traversed meters '
    'or semantic goal completion. Do not restart search after every short '
    'move; judge the actual image and target progress. '
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
    'to help. You may explicitly call inspect_geometry for terrain and '
    'human observations. Do NOT fabricate a free corridor. '
)
BLOCKED_OBSERVATION_INTERVENTION_PROMPT = (
    'OBSERVATION-LOOP / NO CONFIRMED SAMPLED BEARING (issued ONCE per task/stance): '
    'Repeated same-stance observations have not established a currently '
    'executable direction AMONG MEASURED bearings; other headings are UNMEASURED '
    'and not known to be blocked. Do NOT call a turning-only MOVE a successful forward '
    'attempt, and do NOT invent clearance. Choose the next task-relevant '
    'strategy yourself: a genuinely different local goal/detour, a specific '
    'live geometry or view test that could change the decision, a semantic '
    'task revision, or an explicitly evidence-backed movement direction '
    'for the simulator to execute. Do not merely toggle between already '
    'seen pitch levels or arbitrarily change relative headings. The model '
    'retains full planning authority; geometry is an explicit tool only. '
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
            known = set(runtime.memory.keyframes)
            if any(not set(item['source_observation_ids']) <= known
                   for item in value.get('memory_delta', [])):
                raise APIError('memory_delta_requires_known_observation_sources',
                               category='structured_field')
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
        full_circle_commit=(
            bool(panorama.get('all_azimuths_observed')) and not orientation
            and getattr(self,'_panorama_committed_signature',None)!=signature
            and 'navigate_relative' in runtime.registry)
        # This is a temporary tool-choice contract, NOT an autonomous move:
        # the LLM picks a bearing/distance; geometry is model-requested only.
        offered_tools=(
            {'navigate_relative':runtime.registry['navigate_relative']}
            if full_circle_commit else dict(runtime.registry))
        if (not full_circle_commit and not orientation
                and 'observe_for_goal' in offered_tools):
            # All camera-search steering is VLM-owned. Keep legacy camera
            # tools in the runtime for archived calls and orientation tasks,
            # but never advertise micro-turn observations to NavigationAgent.
            offered_tools.pop('turn_and_observe',None)
            spec=offered_tools['observe_for_goal']
            parameters=dict(spec['parameters'])
            properties=dict(parameters['properties'])
            properties.pop('direction',None)
            properties.pop('checkpoint_degrees',None)
            parameters['properties']=properties
            offered_tools['observe_for_goal']=dict(spec,parameters=parameters)

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
