"""Small, stable mission-level tool menu.

The language model controls navigation decisions, when to look and when to
finish. No stage-specific whitelist or hidden completion-state transition.
"""
CORE_NAVIGATION_TOOLS=(
    'observe_for_goal','navigate_relative','turn_and_observe',
    'navigate_local','cancel_local_navigation','query_task_memory',
    'read_memory_record','request_finish')

def navigation_tool_menu(registry, state=None, *, orientation=False,
                         full_circle_commit=False):
    # Tool availability may not silently impose semantic/task policies.
    # The runtime still validates every requested action and physical limits.
    return {name:registry[name] for name in CORE_NAVIGATION_TOOLS
            if name in registry}
