"""Semantic planning contract: no model-fabricated completion predicates."""
import copy
import unittest

from havln_agent.api_client import APIError
from havln_agent.progress import stair_motion_direction,turn_requirement
from havln_agent.task_planner import TaskPlanner,GOAL_SCHEMA


class FakeClient:
    def __init__(self,response):
        self.response=response
    def complete(self,*args,**kwargs):
        result=copy.deepcopy(self.response)
        kwargs['validate'](result)
        return result


class TaskPlannerContractTests(unittest.TestCase):
    def test_stair_motion_classification_is_only_a_terrain_policy_hint(self):
        self.assertEqual(stair_motion_direction(dict(action='Descend the stairs')),'DOWN')
        self.assertEqual(stair_motion_direction(dict(action='Ascend the staircase')),'UP')
        self.assertIsNone(stair_motion_direction(dict(action='Turn right at the landing')))
        self.assertEqual(turn_requirement(dict(action='In-place right turn'))['direction'],'right')
        self.assertIsNone(turn_requirement(dict(action='Enter the foyer')))

    def test_goal_schema_does_not_contain_success_predicates(self):
        self.assertNotIn('success_criteria',GOAL_SCHEMA['properties'])

    def test_turn_with_descriptive_chandelier_is_one_required_action(self):
        instruction=('Turn to your left, where you will see a chandelier '
                     'hanging above a green table.')
        raw=dict(subgoals=[
            dict(action='Turn left to face the described area',
                 landmark='chandelier above green table',
                 target='leftward orientation',source_sentence_ids=[1])],
            final_target='leftward orientation',constraints=[])
        class InspectingClient(FakeClient):
            def complete(self,*args,**kwargs):
                self.system=args[2]
                return super().complete(*args,**kwargs)
        client=InspectingClient(raw)
        result=TaskPlanner.parse(client,instruction)
        self.assertEqual(len(result['subgoals']),1)
        self.assertIn('NOT automatically require a second physical action',
                      client.system)
        self.assertIn('leftward orientation',client.system)
        self.assertEqual(result['subgoals'][0]['source_text'],instruction)

    def test_three_milestones_from_one_sentence_preserve_semantic_source(self):
        instruction='Descend the stairs, turn right at the landing and enter the foyer.'
        raw=dict(subgoals=[
            dict(action='Descend the stairs',landmark='stairs',
                target='landing',source_sentence_ids=[1]),
            dict(action='Turn right at the landing',landmark='landing',
                target='foyer direction',source_sentence_ids=[1]),
            dict(action='Enter the foyer',landmark='foyer',
                target='inside foyer',source_sentence_ids=[1])],
            final_target='foyer',constraints=[])
        plan=TaskPlanner.parse(FakeClient(raw),instruction)
        self.assertEqual([g['action'] for g in plan['subgoals']],
                         [g['action'] for g in raw['subgoals']])
        self.assertTrue(all(g['source_text']==instruction for g in plan['subgoals']))
        self.assertTrue(all('success_criteria' not in g for g in plan['subgoals']))

    def test_missing_source_sentence_is_rejected(self):
        raw=dict(subgoals=[dict(action='Descend',landmark='stairs',
            target='landing',source_sentence_ids=[1])],
            final_target='landing',constraints=[])
        with self.assertRaises(APIError):
            TaskPlanner.parse(FakeClient(raw),'Descend the stairs. Turn right.')

    def test_second_attempt_salvages_missing_context_without_dropping_instruction(self):
        instruction=('Descend the staircase. A person is standing nearby. '
                     'Then turn right at the bottom.')
        raw=dict(subgoals=[
            dict(action='Descend the staircase',landmark='stairs',target='bottom',
                 source_sentence_ids=[1]),
            dict(action='Turn right at the bottom',landmark='landing',
                 target='right',source_sentence_ids=[3])],
            final_target='landing',constraints=[])
        plan=TaskPlanner.parse(FakeClient(raw),instruction,
            format_feedback=dict(error='first_planning_attempt_missing_source_2'))
        self.assertEqual(plan['source_coverage_repaired_ids'],[2])
        self.assertEqual(set(i for g in plan['subgoals']
            for i in g['source_sentence_ids']),{1,2,3})
        self.assertIn('A person is standing nearby.',
                      ' '.join(g['source_text'] for g in plan['subgoals']))
        self.assertEqual(plan['subgoals'][0]['action'],'Descend the staircase')
        self.assertEqual(plan['subgoals'][1]['action'],'Turn right at the bottom')

    def test_first_attempt_missing_mapping_reports_exact_missing_clause(self):
        instruction='Descend stairs. Turn right. Enter the foyer.'
        raw=dict(subgoals=[
            dict(action='Descend stairs',landmark='stairs',target='landing',
                source_sentence_ids=[1]),
            dict(action='Enter the foyer',landmark='foyer',target='foyer',
                source_sentence_ids=[3])], final_target='foyer',constraints=[])
        with self.assertRaises(APIError) as err:
            TaskPlanner.parse(FakeClient(raw),instruction)
        self.assertIn('missing_source_ids=[2]',str(err.exception))
        self.assertIn('Turn right',str(err.exception))

    def test_duplicate_or_invalid_source_order_is_rejected(self):
        raw=dict(subgoals=[dict(action='Move',landmark='hallway',
            target='foyer',source_sentence_ids=[2,1])],
            final_target='foyer',constraints=[])
        with self.assertRaises(APIError):
            TaskPlanner.parse(FakeClient(raw),'Turn left. Enter foyer.')

    def test_source_id_coverage_stays_strict_when_goals_are_semantic(self):
        raw=dict(subgoals=[dict(action='Move',landmark='hallway',
            target='foyer',source_sentence_ids=[2])],
            final_target='foyer',constraints=[])
        with self.assertRaises(APIError):
            TaskPlanner.parse(FakeClient(raw),'Turn left. Enter foyer.')


if __name__=='__main__':
    unittest.main()
