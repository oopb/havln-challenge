"""A short OBSERVE decision followed by a bounded MOVE and another OBSERVE."""
from .contracts import navigation_schema, compact_navigation_protocol, navigation_envelope
from .contexts import navigation_context
from .api_client import APIError

COMMON_PROMPT = (
    'You are NavigationAgent. TaskPlanner supplies ordered semantic subgoals. '
    'Work on the CURRENT subgoal until independently verified. '
    'You receive structured state, not raw RGB or depth; use visual tools as needed. '
    'Return exactly ONE JSON object containing one registered tool and args, '
    'plus optional evidence-linked memory_delta. Wait for the tool result. '
    'Do not invent observations, movement, target coordinates or completion. '
    'Each MOVE requires the local controller to check fresh RGB-D geometry. '
    'A local TARGET_REACHED is not semantic task completion. Use verify_subgoal '
    'with authentic observations and registered movements, and never request_finish '
    'just because exploration is slow. '
)
OBSERVE_PROMPT = (
    'MODE: OBSERVE. Decide the NEXT short movement toward the current subgoal, '
    'not a complete global route. A navigate_local tool call executes a bounded '
    'MOVE batch and then returns control to OBSERVE with new RGB-D feedback. '
    'FIRST inspect navigation_memory, last_tool, candidate_waypoints, '
    'local_navigation and current target evidence. Preserve useful progress. '
    'If an IN_PROGRESS waypoint remains relevant, navigate_local with the SAME '
    'waypoint_id. Compare goal_alignment and estimated target distance before/after. '
    'Prefer a measured goal-approaching direction over a merely easy turn. '
    'An executable exploration waypoint does not itself imply task progress. '
    'If a relevant EXECUTABLE candidate already exists, navigate_local '
    'before asking for another scan. '
    'If a visible goal has reliable grounding, request get_waypoint_candidates '
    'with purpose="approach_landmark". If no grounded candidate is possible, '
    'use EXPLICIT purpose="explore" for a short geometry-guided approach, not '
    'an unrelated direction that undoes successful progress. '
    'For descent, remember recent DOWN translation and check heading before '
    'choosing a reverse/uphill candidate. Registered RGB-D translation is '
    'ESTIMATED, not simulator ground truth: consider uncertainty and new evidence. '
    'If fast depth detects a suspected obstacle, pause MOVE and call '
    'inspect_geometry when detailed terrain or obstacle confirmation is '
    'needed. Full geometry diagnosis is a tool you invoke ON DEMAND, '
    'not an automatic permission gate before every forward step. '
    'BLOCKED means inspect another candidate direction. NEEDS_VIEW means obtain '
    'the specific missing camera view. REQUIRES_ATTENTION means reanchor using '
    'current RGB-D and refresh geometry/grounding. Camera-only view directions '
    'are not executable movement waypoints. '
    'Treat NO_NEW_INFORMATION and REPEATED_SEARCH_NO_NEW_VIEW as unsuccessful '
    'observation attempts: do not repeat them from the same stance. Prefer '
    'evaluating a short available waypoint or an actually different view. '
    'Only use turn_and_observe or observe_for_goal when new information is needed '
    'to choose a movement. Avoid repeated static scans and rediscovering an '
    'already identified staircase. A completed short waypoint is a movement '
    'receipt, not a reason to restart the entire semantic search. '
    'After each MOVE, choose the next short segment using the new observation '
    'and retained memory; continue the same subgoal. '
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
        return runtime.client.complete(
            'navigation_agent', 'navigation_decision',
            COMMON_PROMPT + OBSERVE_PROMPT,
            context, navigation_schema(runtime.registry), validate=validate,
            format_retry=int(bool(format_feedback)), retry_safe=True,
            protocol=compact_navigation_protocol(runtime.registry),
            validation_schema=navigation_envelope(runtime.registry))
