"""Retired numbered-milestone planner.

The current episode pipeline never instantiates TaskPlanner. There is one
mission narrated in Memory, not a sequence of verified semantic subgoals.
"""
class TaskPlanner:
    def __init__(self,*args,**kwargs):
        raise RuntimeError('TaskPlanner retired: use whole-mission NavigationAgent')

    @staticmethod
    def parse(*args,**kwargs):
        raise RuntimeError('numbered_task_decomposition_removed')
