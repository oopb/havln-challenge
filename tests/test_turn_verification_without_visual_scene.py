"""Actual in-place turn receipts, without an unrelated VLM requirement."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from havln_agent.memory_records import TaskMemory
from havln_agent.progress import verify_progress


def make_memory(final=False):
    task=TaskMemory()
    task.subgoals=[dict(action='Turn right at the bottom of the stairs',
        source_text='Turn right at the bottom.',target='foyer direction',
        landmark='stair bottom',success_criteria=['orientation_completed'])]
    if not final:
        task.subgoals.append(dict(action='Enter foyer',
            target='foyer',landmark='foyer',success_criteria=[
                'landmark_visible','registered_translation','target_nearby',
                'final_arrival_visible']))
    task.subgoal_start_steps={'0':0}
    return SimpleNamespace(task=task,scene_memory=None,
        observations=[dict(observation_id='obs_before'),dict(observation_id='obs_after')],
        groundings=[],actions=[
            dict(action='TURN_RIGHT',step=i,subgoal=0,result=dict(visual_change=True))
            for i in range(6)],
        camera=SimpleNamespace(turn_angle=15),current=dict(pose=dict(motion_epoch=2)),
        progress_stage='NOT_STARTED',progress_feedback=None,event=Mock())


class PureTurnVerificationTests(unittest.TestCase):
    def test_intermediate_right_turn_verified_without_visual_scene(self):
        m=make_memory()
        result=verify_progress(m,'subgoal_completed','Executed a 90 degree right turn',
            ['obs_after'],15.,1.25)
        self.assertEqual(result['status'],'VERIFIED')
        self.assertTrue(result['completion_confirmed'])
        self.assertEqual(m.task.completed_subgoals,[0])
        self.assertEqual(m.task.current_subgoal_index,1)
        self.assertEqual(m.task.subgoal_started,6)

    def test_last_goal_still_needs_final_visual_evidence(self):
        m=make_memory(final=True)
        result=verify_progress(m,'subgoal_completed','Turned right',
            ['obs_after'],15.,1.25)
        self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE')
        self.assertFalse(result['completion_confirmed'])
        self.assertIn('final_arrival_visible',result['missing'])
        self.assertEqual(m.task.current_subgoal_index,0)

    def test_command_without_turn_receipt_never_verifies(self):
        m=make_memory()
        m.actions=[]
        result=verify_progress(m,'subgoal_completed','Just a claim',
            ['obs_after'],15.,1.25)
        self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE')
        self.assertFalse(result['completion_confirmed'])


if __name__=='__main__':
    unittest.main()
