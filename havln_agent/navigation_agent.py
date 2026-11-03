"""A short OBSERVE decision followed by a bounded MOVE and another OBSERVE."""
from .contracts import navigation_schema, compact_navigation_protocol, navigation_envelope
from .contexts import navigation_context
from .api_client import APIError

COMMON_PROMPT = (
    'You are NavigationAgent. TaskPlanner supplies ordered semantic subgoals. '
    'Work on the CURRENT subgoal. Navigation progress and verified completion '
    'are separate: you can select_subgoal to provisionally move to the NEXT '
    'ordered milestone when evidence suggests you reached or passed the current '
    'milestone, even if a rigid metric proof is unavailable. You may revisit '
    'an earlier unverified milestone using select_subgoal. This NEVER marks '
    'a milestone completed: request_finish success still needs all milestones '
    'independently verified. Do not advance solely to evade a collision, '
    'without relevant visual/motion evidence. '
    'You receive structured state, not raw RGB or depth; use visual tools as needed. '
    'Return exactly ONE JSON object containing one registered tool and args, '
    'plus optional evidence-linked memory_delta. Wait for the tool result. '
    'Do not invent observations, movement, target coordinates or completion. '
    'Each MOVE requires the local controller to check fresh RGB-D geometry. '
    'A local TARGET_REACHED is not semantic task completion. Read '
    'historical_motion_evidence even after motion_epoch changes; an epoch '
    'invalidates old *coordinates*, not historical verified RGB-D movement. '
    'Treat PROVISIONAL_UNVERIFIED transitions as hypotheses. Use verify_subgoal '
    'with authentic observations and registered movements; if evidence is still '
    'incomplete but sensor history suggests the next milestone, use '
    'select_subgoal rather than repeating the same visual query. Do not treat '
    'a model-selected milestone as VERIFIED, and never request_finish '
    'just because exploration is slow. '
)
OBSERVE_PROMPT = (
    'MODE: OBSERVE. Choose the NEXT finite local movement target, not a '
    'complete global route. A navigate_local/navigate_relative/navigate_to_pixel '
    'call executes the model-selected TURNs and forward segment continuously '
    'until arrival, live safety/pose interruption, or global Episode budget; '
    'then returns control to OBSERVE with new RGB-D feedback. '
    'FIRST inspect navigation_memory, last_tool, candidate_waypoints, '
    'local_navigation and current target evidence. Preserve useful progress. '
    'If an IN_PROGRESS waypoint remains useful, you may navigate_local using the SAME '
    'waypoint_id, or pause it to observe, choose another direction, or cancel. '
    'Compare goal_alignment and estimated target distance before/after. '
    'Prefer a measured goal-approaching direction over a merely easy turn. '
    'An executable exploration waypoint does not itself imply task progress. '
    'A candidate waypoint is OPTIONAL. You may call navigate_relative with '
    'a relative bearing and desired distance directly (the executor quantizes '
    'to the real action increments), or use '
    'navigate_to_pixel with a normalized RGB-D image target and movement distance. '
    'Do not invent free space: the executor checks current depth for every forward. '

    'You decide whether a direction serves the task using the instruction, '
    'visual hints, current depth, waypoint candidates, recent motion, and memory. '
    'You may request get_waypoint_candidates even when grounding is PARTIAL or UNKNOWN: '
    'partial localization is evidence uncertainty, NOT a movement prohibition. '
    'local_geometry describes ONLY the currently sampled relative bearings, '
    'usually angle_degrees=0 (straight ahead), NOT a 360-degree clearance scan. '
    'Read local_geometry_scope: unmeasured yaw bins are UNKNOWN, not blocked; '
    'a blank currently_plausible_local_bearings list does NOT mean there is '
    'no viable direction anywhere. Distinguish a measured BLOCKED direction '
    'from a measured NEEDS_VIEW/unknown direction. Turning changes the live '
    'sampled heading; each next forward still requires new depth checks. '
    'Treat goal_alignment, estimated distance and direction compatibility '
    'as descriptive evidence. A detour that temporarily moves away from '
    'a landmark can be appropriate. Do not require complete grounding, '
    'a particular scene category, entry point or monotonic distance decrease '
    'before choosing a short MOVE. The local controller alone checks live '
    'depth, human occupancy and physical action validity. Never bypass '
    'a measured collision or invent geometric clearance. '
    'Registered RGB-D translation is ESTIMATED, not simulator ground truth. '
    'If fast depth detects a suspected obstacle, pause MOVE and call '
    'inspect_geometry when detailed terrain or obstacle confirmation is '
    'needed. Full geometry diagnosis is a tool you invoke ON DEMAND, '
    'not an automatic permission gate before every forward step. '
    'BLOCKED means inspect another candidate direction. NEEDS_VIEW means obtain '
    'the specific missing camera view. REQUIRES_ATTENTION means reanchor using '
    'current RGB-D and refresh geometry/grounding. Camera-only view directions '
    'are not executable movement waypoints. '
    'Treat NO_NEW_INFORMATION and REPEATED_SEARCH_NO_NEW_VIEW as evidence of '
    'low information value, not absolute prohibitions. Use purpose=intentional_recheck '
    'or force_recheck=true when an older view must actually be rechecked. Prefer '
    'evaluating a short available waypoint or an actually different view. '
    'Observe whenever it materially informs the decision, even during a paused '
    'local route; the route will be revalidated before it resumes. '
    'Only use turn_and_observe or observe_for_goal when new information is needed '
    'to choose a movement. Avoid repeated static scans and rediscovering an '
    'already identified target. A completed short waypoint is a movement '
    'receipt, not a reason to restart the entire semantic search. '
    'After each MOVE, choose the next short segment using the new observation '
    'and retained memory; continue the same subgoal. '
    'Navigation tools DO NOT take max_atomic_steps or a per-call action '
    'budget. Provide only a target direction, distance, or waypoint ID; '
    'the executor completes all needed turns before attempting forward. '
    'Every forward receives fresh depth, terrain and people checks. '
    'Safety/pose interruption returns control for YOUR next decision. '
)

# This is a model-visible *decision focus*, not a tool whitelist or a hidden
# physical control action. The agent still owns the bearing and can explain
# missing live safety rather than inventing a safe corridor.
OBSERVATION_INTERVENTION_PROMPT = (
    'OBSERVATION-LOOP DECISION CHECKPOINT (issued ONCE per task/stance): '
    'Existing yaw/pitch views have produced little actionable information. '
    'The current geometry contains a plausible local bearing, but this is '
    'NOT lasting forward permission. Choose a task-relevant short movement '
    'when evidence supports it, OR use a different purposeful action if '
    'the available evidence is insufficient. YOU choose the direction, '
    'route, target and whether new observations are genuinely needed. '
    'Repeating the same absolute view without changed evidence is unlikely '
    'to help. Any MOVE_FORWARD still requires current RGB-D, terrain and '
    'human safety checks. Do NOT fabricate a free corridor. '
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
    'for the safety controller to test. Do not merely toggle between already '
    'seen pitch levels or arbitrarily change relative headings. The model '
    'retains full planning authority and the executor retains physical veto. '
)
MOVE_EXECUTION_RULES = (
    'MODE: MOVE is executed without a new LLM request by navigate_local. '
    'LocalMotionController checks fresh RGB-D for each forward action, '
    'continues to target arrival or an actual safety/pose/episode interruption. '
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
            known = set(runtime.memory.keyframes)
            if any(not set(item['source_observation_ids']) <= known
                   for item in value.get('memory_delta', [])):
                raise APIError('memory_delta_requires_known_observation_sources',
                               category='structured_field')
        context = dict(navigation_context(runtime),
                       **({'format_feedback': format_feedback} if format_feedback else {}))
        stagnation=context.get('observation_stagnation') or {}
        advisory=stagnation.get('intervention')
        # A one-shot strategic checkpoint must not be repeated every turn
        # while the physical stance is unchanged. A later new subgoal/stance
        # can legitimately receive a new checkpoint.
        pose=context.get('pose') or {}
        task=context.get('task_selection') or {}
        xy=pose.get('position_xz_m_est') or ()
        signature=(getattr(runtime.memory,'episode_id',None),
                   getattr(runtime,'plan_revision',None),
                   task.get('current_subgoal_index'),
                   pose.get('motion_epoch'),pose.get('place_id'),
                   tuple(round(float(v),1) for v in xy))
        first_checkpoint=(
            advisory in ('DIRECTIONAL_ACTION_DECISION_REQUIRED',
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
            COMMON_PROMPT + OBSERVE_PROMPT + advisory_prompt,
            context, navigation_schema(runtime.registry), validate=validate,
            format_retry=int(bool(format_feedback)), retry_safe=True,
            protocol=compact_navigation_protocol(runtime.registry),
            validation_schema=navigation_envelope(runtime.registry))
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
