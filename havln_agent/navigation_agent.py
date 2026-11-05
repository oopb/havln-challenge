"""A short OBSERVE decision followed by a bounded MOVE and another OBSERVE."""
from .contracts import navigation_schema, compact_navigation_protocol, navigation_envelope
from .contexts import navigation_context, compact_navigation_context
from .tool_menu import navigation_tool_menu
from .api_client import APIError
from .panorama_memory import decide_with_panorama
from jsonschema import Draft7Validator

COMMON_PROMPT = (
    'You are NavigationAgent. TaskPlanner provides ordered semantic subgoals; '
    'you control strategic observation, motion, subgoal selection and STOP. '
    'You see concise JSON, not RGB. VLM owns all goal-directed camera search '
    'inside observe_for_goal; never micro-manage its 15-degree turns. '
    'Return ONLY ONE JSON {"tool":"name","args":{...}}; do NOT provide '
    'memory_delta or commentary. Tools persist provenance-linked visual '
    'evidence automatically. Use only registered tools and real IDs. '
    'Unknown RGB-D odometry is not evidence of a collision and does not stop '
    'a finite movement plan. A completed atomic action plan is NOT verified '
    'physical arrival. Verify milestones from real visual/motion evidence; '
    'select_subgoal may provisionally advance but cannot certify success. '
    'Read progress_review: chronological registered movement, opposite-direction '
    'events, VLM hypotheses, and next ordered subgoal are decision evidence, '
    'NOT task-completion proof. When progress may have reached the next phase, '
    'consider verify_subgoal, targeted grounding or a justified provisional '
    'select_subgoal rather than indefinitely repeating the previous goal. '
    'Neither verification nor subgoal selection is automatic. '
    'Do not invent free space, target coordinates or completion; '
    'never request_finish success without independent proof. '
)
OBSERVE_PROMPT = (
    'MODE: OBSERVE. Decide the NEXT useful high-level tool, not a sequence '
    'of tiny camera turns. If visual evidence or a target approach is missing, '
    'delegate ONCE to observe_for_goal(objective,target_types,sweep_degrees?) '
    'with open-vocabulary targets; OMIT direction/checkpoint_degrees and let '
    'the VLM own view selection, pitch and search termination. '
    'In the EXISTING objective string, state both the immediate visual question '
    'AND the visible scene cues you would expect AT OR NEAR THE END of the '
    'current semantic subgoal, derived from its meaning and the instruction. '
    'For example, when descending stairs, ask whether the descending treads '
    'visibly terminate at a flat landing or floor, not merely whether stairs '
    'are visible; when passing a door, ask what opening/interior is visible '
    'beyond it. These are possible VISUAL landmarks, not preconditions for '
    'moving and NOT proof of arrival or completion. Never assert that the '
    'expected feature actually exists; if an end cue is not visually '
    'identifiable, use an honest visual question without fabricating one. '
    'If earlier observations already answer the question, use that evidence '
    'rather than rescan. Keep target_types concise (up to three); no new '
    'tool arguments or extra observation calls are required. Read its '
    'visual_reports (raw compact VLM observations with source frame, yaw, pitch, '
    'free-form target_type, box, confidence and uncertainty); these are '
    'unfiltered visual hypotheses, even when the label differs from '
    'target_types. YOU judge relevance and whether to move or observe again. '
    'best_candidate/best_cue are conveniences, not mandatory gates. '
    'If a target entry is already localized, '
    'prefer acting on that evidence rather than scanning a full circle again. '
    'If evidence permits movement, choose navigate_relative(turn_deg,'
    'distance_m) or a currently offered target-directed motion tool. '
    'The executor performs turns and '
    'forward atomic steps continuously and returns evidence. A 0.25m atomic '
    'forward action is NOT the segment distance limit. '
    'If local_navigation remains useful, resume the SAME waypoint_id via '
    'navigate_local. DO NOT take max_atomic_steps. '
    'Read navigation_memory, candidate_waypoints, current_view, grounding '
    'and recent motion before deciding. Prefer a measured goal-approaching '
    'direction when such evidence exists; grounding is PARTIAL or UNKNOWN '
    'is NOT a movement prohibition. Unknown depth is not measured collision; '
    'UNCONFIRMED is NOT BLOCKED. Never bypass a measured collision. '
    'You own safety risk assessment; geometry is optional and only '
    'model-invoked via inspect_geometry; '
    'normal MOVE does not run an automatic geometry gate. '
    'NEEDS_VIEW means a specific missing view; REQUIRES_ATTENTION after '
    'unchanged fresh RGB-D means reassess your strategy, not repeat a '
    'static sweep. Read forward_stall_evidence: repeated MOVE_FORWARD actions '
    'at the same absolute heading with IDENTICAL before/after RGB and Depth '
    'did NOT demonstrate movement, even if observe_for_goal keeps finding '
    'the same visual landmark. A floor visible beyond a railing, wall, '
    'person or transparent barrier is NOT a traversable corridor at the robot '
    'body position. Identical images alone do NOT prove a collision or name '
    'the obstacle. Decide yourself whether to call inspect_geometry for '
    'current forward/nearby bearings, choose a different evidence-backed route '
    'or request a genuinely useful view; do not reflexively replay the '
    'previously unsuccessful heading/distance without new evidence. '
    'If you call inspect_geometry, read its returned clearance, '
    'obstacle location, uncertainty, and measured alternative bearings '
    'before your next decision; a historical depth result never guarantees '
    'future clearance around moving people. '
    'You may seek additional evidence or plan a short detour. '
    'Immediately before EVERY executed forward action, a lightweight live '
    'Depth probe checks only coherent central near-field occupancy. When it '
    'finds a measured central obstacle, the current forward action is NOT '
    'executed and VisualPerception describes that exact current RGB region '
    'once without scanning. Read last_tool.obstacle_observation: Depth '
    'contains position/clearance evidence and VLM contains ONLY a visual '
    'description of what might be there. Never turn a VLM label into a '
    'collision certificate, or treat an unknown glass/side geometry as free '
    'space. The model (YOU) alone decides whether to inspect_geometry, '
    'change direction, search for a true opening, or wait for a moving person. '
    'Do not just repeat the same blocked movement at the same stance; '
    'there is no forced steering, automatic goal switch or new verifier. '
    'ACTIONS_COMPLETED means commanded steps executed, not traversed meters '
    'or semantic goal completion. Do not restart search after every short '
    'move; judge the actual image and target progress. If verified RGB-D '
    'motion reverses the requested semantic direction, treat this as a '
    'replanning signal; the estimates may be uncertain, not a forced STOP '
    'or a reason to claim arrival. '

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
