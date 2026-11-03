"""A short OBSERVE decision followed by a bounded MOVE and another OBSERVE."""
from .contracts import navigation_schema, compact_navigation_protocol, navigation_envelope
from .contexts import navigation_context
from .api_client import APIError
from .panorama_memory import decide_with_panorama
from jsonschema import Draft7Validator

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
    'Each MOVE uses fresh RGB-D for odometry AFTER movement. Unknown '
    'depth, unmeasured FOV and unlocalized visual hypotheses do NOT prove '
    'collision; only explicit model calls to inspect_geometry or get_waypoint_candidates invoke obstacle/terrain detection. '
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
    'until arrival, pose/sensor interruption, or global Episode budget; '
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
    'Do not invent free space: the executor never invokes geometry automatically. '

    'You decide whether a direction serves the task using the instruction, '
    'visual hints, current depth, waypoint candidates, recent motion, and memory. '
    'You may request get_waypoint_candidates even when grounding is PARTIAL or UNKNOWN: '
    'partial localization is evidence uncertainty, NOT a movement prohibition. '
    'local_geometry describes ONLY the currently sampled relative bearings, '
    'usually angle_degrees=0 (straight ahead), NOT a 360-degree clearance scan. '
    'Read local_geometry_scope: unmeasured yaw bins are UNKNOWN, not blocked; '
    'a blank currently_plausible_local_bearings list does NOT mean there is '
    'no viable direction anywhere. Distinguish a measured BLOCKED direction '
    'from a measured UNCONFIRMED/unknown direction. Turning changes the live '
    'sampled heading; the next forward does NOT run a geometry probe. You are responsible for safety risk assessment and may call inspect_geometry before moving. '
    'Treat goal_alignment, estimated distance and direction compatibility '
    'as descriptive evidence. A detour that temporarily moves away from '
    'a landmark can be appropriate. Do not require complete grounding, '
    'a particular scene category, entry point or monotonic distance decrease '
    'before choosing a short MOVE. A model-requested inspect_geometry tool checks '
    'depth and human occupancy. Never bypass '
    'a measured collision or invent geometric clearance. '
    'Registered RGB-D translation is ESTIMATED, not simulator ground truth. '
    'When visually uncertain about obstacles or terrain, proactively call '
    'inspect_geometry when detailed terrain or obstacle confirmation is '
    'needed. Full geometry diagnosis is a tool you invoke ON DEMAND, '
    'never an automatic permission gate before forward steps. '
    'BLOCKED means an observed obstacle requires a different direction. '
    'UNCONFIRMED is NOT BLOCKED; you may move toward it or seek more evidence. '
    'NEEDS_VIEW from a semantic tool means obtaining a specific missing view. '
    'REQUIRES_ATTENTION means reanchor using '
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
    'After a successful MOVE, explicitly compare the just-completed heading, '
    'registered vertical/forward progress, target memory and next-step depth. '
    'If the previous direction still serves the goal and no obstacle was '
    'observed, consider continuing that same heading instead of restarting '
    'a full static scan. A 0.25m atomic forward action is NOT the maximum '
    'distance_m for navigate_relative: choose the finite 0.25-3.0m segment '
    'appropriate to current evidence (multiple atomic steps run continuously). '
    'You still decide whether to continue, turn, inspect or revise the goal. '
    'turn_and_observe is a tiny calibrated camera tool: its current maximum '
    'absolute angle is action_calibration.turn_and_observe_max_deg; DO NOT '
    'request a 90/150/180 degree single camera-only turn unless its registered '
    'schema explicitly permits it. Use observe_for_goal for planned sweeps. '
    'Navigation tools DO NOT take max_atomic_steps or a per-call action '
    'budget. Provide only a target direction, distance, or waypoint ID; '
    'the executor completes all needed turns before attempting forward. '
    'Every forward uses public RGB-D for motion estimation, NOT collision/terrain/people approval. '
    'Pose/no-motion/budget interruption returns control for YOUR next decision. '
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
    'continues to target arrival or a pose/no-motion/episode interruption. '
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
            if not Draft7Validator(navigation_schema(runtime.registry)).is_valid(value):
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
            if full_circle_commit else runtime.registry)

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
                context, navigation_schema(offered_tools), validate=validate,
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
