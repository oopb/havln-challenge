"""Regression tests for smoke-500-9 initial planning and evidence ownership."""
import copy
import unittest

from havln_agent.progress import stair_motion_direction, requires_landmark_crossing
from havln_agent.task_planner import TaskPlanner, required_conditions


class FakeClient:
    def __init__(self, response):
        self.response = response

    def complete(self, *args, **kwargs):
        value = copy.deepcopy(self.response)
        kwargs['validate'](value)
        return value


class TaskPlannerContractTests(unittest.TestCase):
    def test_descent_is_stair_motion_not_level_only(self):
        for name in ('Descent to the bottom of the stairs.', 'Descending the stairs carefully.'):
            goal = dict(action=name, landmark='stairs', target='bottom landing')
            self.assertEqual(stair_motion_direction(goal), 'DOWN')
            self.assertEqual(required_conditions(goal), {
                'entry_confirmed', 'actual_descent_observed', 'target_nearby'
            })

    def test_through_space_until_hallway_is_not_passing_hallway(self):
        goal = dict(action='Continue through the space until the hallway is reached.',
                    landmark='hallway', target='hallway')
        self.assertFalse(requires_landmark_crossing(goal))
        self.assertNotIn('landmark_passed', required_conditions(goal))
        self.assertTrue(requires_landmark_crossing(dict(action='Enter the foyer.')))

    def test_missing_model_predicates_do_not_abort_startup(self):
        text = ('Begin by descending the stairs. Make a right turn. '
                'Continue through the space until the hallway is reached.')
        raw = dict(subgoals=[
            dict(action='Descent to the bottom of the stairs.', landmark='stairs',
                 target='landing', source_sentence_ids=[1]),
            dict(action='In-place right turn.', landmark='landing', target='right',
                 source_sentence_ids=[2]),
            dict(action='Continue through the space until the hallway is reached.',
                 landmark='hallway', target='hallway',
                 success_criteria=['landmark_passed'], source_sentence_ids=[3]),
        ], final_target='hallway', constraints=[])
        plan = TaskPlanner.parse(FakeClient(raw), text)
        self.assertEqual(plan['subgoals'][0]['success_criteria'], [
            'actual_descent_observed', 'entry_confirmed', 'target_nearby'
        ])
        self.assertEqual(plan['subgoals'][1]['success_criteria'], ['orientation_completed'])
        self.assertEqual(plan['subgoals'][2]['success_criteria'], [
            'final_arrival_visible', 'landmark_visible', 'registered_translation', 'target_nearby'
        ])

    def test_missing_source_sentence_is_still_invalid(self):
        from havln_agent.api_client import APIError
        raw = dict(subgoals=[dict(action='Descent.',landmark='stairs',
            target='landing',success_criteria=['target_nearby'],source_sentence_ids=[1])],
            final_target='landing',constraints=[])
        with self.assertRaises(APIError):
            TaskPlanner.parse(FakeClient(raw), 'Descend the stairs. Turn right.')


if __name__ == '__main__':
    unittest.main()
