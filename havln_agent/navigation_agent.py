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
    'with authentic observations and registered movements; review stair_replay '
    'for historical scene citations and actual registered descents, and NEVER '
    'mistake a recorded descent for independently proven bottom arrival. '
    'If historical components exist but the landing is not verified, navigate '
    'to collect genuinely missing independent evidence or select_subgoal '
    'provisionally when further immediate verification is not useful. '
    'The verifier is a separate model-led evidence reviewer with read-only '
    'historical inspection. It can request recorded views and detailed motion '
    'receipts before deciding; its VERIFIED outcome is authenticated by Runtime. '
    'Use verify_subgoal after meaningful task progress or a decisive new view, '
    'not after every rotation. When verification returns missing evidence, '
    'read verification_guidance.next_evidence_action and obtain that evidence '
    'instead of restarting a whole-room scan. An IN_PROGRESS milestone can '
    'remain provisionally selected while navigating. '
    'If evidence is still '
    'incomplete but sensor history suggests the next milestone, use '
    'select_subgoal rather than repeating the same visual query. Do not treat '
    'a model-selected milestone as VERIFIED, and never request_finish '
    'just because exploration is slow. '
)
OBSERVE_PROMPT = (
    'MODE: OBSERVE. Decide the NEXT short movement toward the current subgoal, '
    'not a complete global route. A navigate_local tool call executes a bounded '
    'MOVE batch and then returns control to OBSERVE with new RGB-D feedback. '
    'FIRST inspect navigation_memory, last_tool, candidate_waypoints, '
    'local_navigation and current target evidence. Preserve useful progress. '
    'If an IN_PROGRESS waypoint remains useful, you may navigate_local using the SAME '
    'waypoint_id, or pause it to observe, choose another direction, or cancel. '
    'Compare goal_alignment and estimated target distance before/after. '
    'Prefer a measured goal-approaching direction over a merely easy turn. '
    'A tentative_goal_bearings hint is ONLY a same-view partial visual '
    'direction, never verified target identity, reliable distance, a proven '
    'doorway or movement permission. It can help compare geometric candidate '
    'directions when every waypoint is still EXPLORATION_WAYPOINT. '
    'If visual evidence later rejects a doorway-like object as a window or '
    'solid panel, do not persist toward it as the intended doorway. '
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
    'current RGB-D and refresh geometry/grounding. If a MOVE reports '
    'no_resolvable_forward_translation or no_observed_frame_change_after_forward, '
    'the old endpoint is invalid: do not repeat the same forward corridor '
    'blindly; compare recent failures and choose another short path or obtain '
    'a relevant fresh depth view. Such evidence never proves collision. '
    'Camera-only view directions '
    'are not executable movement waypoints. '
    'Treat NO_NEW_INFORMATION and REPEATED_SEARCH_NO_NEW_VIEW as evidence of '
    'low information value, not absolute prohibitions. Use purpose=intentional_recheck '
    'or force_recheck=true when an older view must actually be rechecked. '
    'For a genuinely required in-place turn toward a previously viewed heading, '
    'set turn_and_observe purpose=intentional_recheck so it remains executable; '
    'do not mark such a physical orientation action as an exploration scan. '
    'A default purpose=explore sweep repeating known camera sectors is '
    'deduplicated even during an orientation milestone. Prefer '
    'evaluating a short available waypoint or an actually different view. '
    'Observe whenever it materially informs the decision, even during a paused '
    'local route; the route will be revalidated before it resumes. '
    'Use exploration_feedback and persistent_semantic_targets to distinguish '
    'previously observed target identity from present pose estimates. Persist '
    'a candidate target after camera turns; when a potentially useful door, '
    'stair or opening has appeared, use ground_landmark or a directed visual '
    'query if new geometric target evidence is needed. Existing saved scene '
    'semantics never authorize movement; call navigate_relative or a fresh '
    'waypoint with current Depth safety when the local route is defensible. '
    'Do not spend the remaining budget on equivalent camera sweeps after '
    'repeated NO_NEW_INFORMATION; prefer an alternate executable short '
    'route, purposeful new viewpoint, provisional subgoal change, or an '
    'explicit task review. Never fake a route or ignore BLOCKED safety. '
    'Only use turn_and_observe or observe_for_goal when new information is needed '
    'to choose a movement. Avoid repeated static scans and rediscovering an '
    'already identified target. A completed short waypoint is a movement '
    'receipt, not a reason to restart the entire semantic search. '
    'After each MOVE, choose the next short segment using the new observation '
    'and retained memory; continue the same subgoal. '
)
# Model-owned decisional emphasis after documented low-information scanning.
# This is not a turn quota, forced forward action, automatic STOP or bypass of
# live depth/terrain/human checks.
ROUTE_REASSESSMENT_PROMPT = (
    'EXECUTION-FOCUS REVIEW: This episode has spent many actions observing '
    'without making registered forward progress. Do not restart the same '
    'room scan or repeat an unchanged verification claim. First consider '
    'whether the saved current geometry, local_navigation or candidate '
    'waypoints support a short, genuinely task-useful movement. '
    'You can choose navigate_relative directly without a waypoint ID and '
    'the executor will validate fresh public RGB-D and human safety before '
    'each step. Do NOT wait for perfect full-room semantic localization when '
    'a short safe viewpoint change or task-directed detour is reasonable. '
    'If every relevant movement is actually blocked or the goal identity is '
    'contradictory, choose ONE specific information-producing tool instead: '
    'compare_views for the target_identity_review observation IDs, '
    'ground_landmark for a genuinely new viewpoint, or inspect_geometry '
    'for a new measured obstruction. Explain the change of viewpoint in '
    'memory_delta when appropriate. Do not claim a forbidden move is safe. '
    'When verification_followup indicates no new evidence since a prior '
    'INSUFFICIENT_EVIDENCE decision, move or observe for that missing evidence '
    'instead of immediately calling verify_subgoal again. '
)
TARGET_IDENTITY_PROMPT = (
    'TARGET CONTINUITY: target_identity_review contains competing historical '
    'VLM landmark hypotheses. Grounding a generic bright opening is not proof '
    'that it is the SAME doorway/landmark named by the instruction. If newer '
    'visual evidence may be a different object, use compare_views on saved '
    'reference and new frame rather than silently replacing the target. '
    'Maintain the original goal identity across camera rotation; bearings '
    'recorded at different headings cannot be compared directly without '
    'the historical pose. Navigation remains model-directed and safe geometry '
    'is still independently required.'
)

MOVE_EXECUTION_RULES = (
    'MODE: MOVE is executed without a new LLM request by navigate_local. '
    'LocalMotionController checks fresh RGB-D for each forward action, '
    'executes up to the requested atomic-action budget and stops on obstruction '
    'or attention. The next NavigationAgent decision is always OBSERVE.'
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
        focus = context.get('exploration_feedback') or {}
        system_prompt = COMMON_PROMPT + OBSERVE_PROMPT
        if focus.get('scan_dominated_without_translation'):
            system_prompt += ROUTE_REASSESSMENT_PROMPT
        if (context.get('target_identity_review') or {}).get('candidates'):
            system_prompt += TARGET_IDENTITY_PROMPT
        return runtime.client.complete(
            'navigation_agent', 'navigation_decision',
            system_prompt,
            context, navigation_schema(runtime.registry), validate=validate,
            format_retry=int(bool(format_feedback)), retry_safe=True,
            protocol=compact_navigation_protocol(runtime.registry),
            validation_schema=navigation_envelope(runtime.registry))
