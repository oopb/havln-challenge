"""Small role-based NavigationAgent tool menu.

Execution registry stays complete for archived actions, visual subagents and
tool dispatch. Only the current decision's advertised model-facing tools are
selected here; this does not modify the runtime or bypass its validation.
"""

# Exactly nine stable strategic tools. Search/scene analysis belongs to VLM.
CORE_NAVIGATION_TOOLS = (
    'observe_for_goal',
    'navigate_relative',
    'navigate_local',
    'ground_landmark',
    'inspect_geometry',
    'verify_subgoal',
    'select_subgoal',
    'request_task_review',
    'request_finish',
)

# Auxiliary capabilities never disappear from ToolExecutor; add them only
# when live state or a trusted tool receipt makes their purpose relevant.
CONTEXTUAL_TOOLS = (
    'turn_and_observe',
    'navigate_to_pixel',
    'cancel_local_navigation',
    'get_waypoint_candidates',
    'recall_observation',
    'query_spatial_memory',
    'query_task_memory',
    'read_memory_record',
    'compare_views',
)
# Scene/entry understanding is delegated to VLM-led observe_for_goal, not
# re-exposed as competing main-agent observation entrypoints.
VLM_INTERNAL_TOOLS = ('analyze_scene', 'assess_entry')


def _tool_result(state):
    return state.get('last_tool') or {}


def contextual_tool_names(state, *, orientation=False):
    """Derive conditional capabilities from visible, recent task evidence.

    No geometric/semantic threshold here authorizes movement or forces a
    heading; this chooses the tool vocabulary only. All decision ownership
    remains with NavigationAgent.
    """
    result = set()
    last = _tool_result(state)
    observation = last.get('observation_task') or {}
    latest = state.get('latest_observation_task') or {}
    current_id = (state.get('current_view') or {}).get('observation_id')

    if orientation:
        # Turns without translation remain possible for orientation tasks.
        result.add('turn_and_observe')

    local = state.get('local_navigation') or {}
    if local.get('waypoint_id') and local.get('status') == 'IN_PROGRESS':
        result.add('cancel_local_navigation')

    # Camera-centered movement is useful only with a fresh localized box.
    # A historical image region is NOT a current-frame motion command.
    candidates = (observation.get('best_candidate'),
                  latest.get('candidate'))
    grounded = (state.get('grounding_evidence') or [])[:2]
    has_current_box = any(
        isinstance(item, dict)
        and item.get('box') is not None
        and item.get('observation_id') == current_id
        for item in candidates)
    has_current_grounding = any(
        item.get('is_current_view') is True
        and item.get('status') in ('LOCALIZED', 'PARTIAL')
        for item in grounded if isinstance(item, dict))
    if has_current_box or has_current_grounding:
        result.add('navigate_to_pixel')

    if (last.get('tool') == 'inspect_geometry'
            or state.get('recent_navigation_block')
            or last.get('measured_collision')):
        result.add('get_waypoint_candidates')

    memory = state.get('navigation_memory') or {}
    targets = memory.get('visual_target_memory') or []
    ambiguous_visual = any(
        t.get('status') in ('CUE_ONLY', 'AMBIGUOUS', 'NOT_FOUND')
        for t in targets[-2:] if isinstance(t, dict))
    if (state.get('historical_hint') or ambiguous_visual
            or (latest.get('outcome') in ('CUE_ONLY',
                'SEARCH_REGION_COVERED_NO_MATCH'))):
        result.add('recall_observation')
        result.add('query_spatial_memory')
    if (last.get('memory') or state.get('memory_hints')
            or state.get('candidate_waypoints')):
        # Not every memory hint is a reason to fetch complete records; raw
        # records are for follow-up inspection, not ordinary movement calls.
        if last.get('tool') in ('recall_observation','query_spatial_memory',
                                'query_task_memory','read_memory_record'):
            result.add('read_memory_record')

    selection = state.get('task_selection') or {}
    if (selection.get('provisional_transitions')
            or last.get('tool') == 'verify_subgoal'):
        result.add('query_task_memory')

    # Respect the precise missing-evidence options returned by a verifier
    # or explicit geometry probe, rather than guessing scene-specific rules.
    requested = set(last.get('information_options') or ())
    result.update(requested.intersection(CONTEXTUAL_TOOLS))
    # A tool suggestion must never reintroduce ordinary single-step camera
    # control. Only genuine orientation instructions need that legacy tool.
    if not orientation:
        result.discard('turn_and_observe')
    if last.get('tool') == 'verify_subgoal' and last.get('status') in (
            'REJECTED', 'INSUFFICIENT_EVIDENCE'):
        result.add('compare_views')
    return result


def navigation_tool_menu(registry, state, *, orientation=False,
                         full_circle_commit=False):
    """Return the exact schema offered to this one main-agent decision.

    This is NOT the full execution registry. The post-model executor still
    validates against the unchanged registry and owns every atomic action.
    """
    if full_circle_commit:
        return ({'navigate_relative': registry['navigate_relative']}
                if 'navigate_relative' in registry else {})

    extras = contextual_tool_names(state, orientation=orientation)
    names = CORE_NAVIGATION_TOOLS + tuple(
        name for name in CONTEXTUAL_TOOLS if name in extras)
    offered = {name: registry[name] for name in names if name in registry}

    if 'observe_for_goal' in offered and not orientation:
        # Normal navigation only sees VLM-owned observation. Legacy fixed
        # direction subtask arguments remain supported in the executor.
        spec = dict(offered['observe_for_goal'])
        params = dict(spec['parameters'])
        props = dict(params['properties'])
        props.pop('direction', None)
        props.pop('checkpoint_degrees', None)
        params['properties'] = props
        spec['parameters'] = params
        offered['observe_for_goal'] = spec
    return offered
