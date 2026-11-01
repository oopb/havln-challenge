"""Replay the reported failure pattern without episode-specific hardcoded policy.

The model can move on provisionally, but history must retain measured descent;
no unverified subgoal can counterfeit final success.
"""
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from havln_agent.progress import (_stair_evidence, registered_motion_history,
                                  verify_progress, turn_requirement)
from havln_agent.task_planner import required_conditions, normalize_conditions
from havln_agent.runtime import AgentRuntime
from havln_agent.memory_records import TaskMemory


def observation(i, epoch):
    return dict(observation_id=f'obs_{i:04d}',step=i,
        rgb_hash=f'rgb{i}',depth_hash=f'depth{i}',
        pose=dict(x=0.,z=0.,heading=0.,motion_epoch=epoch))


def movement(step, epoch, vertical=-.11, valid=True):
    pair=dict(before_rgb_hash=f'rgb{step}',before_depth_hash=f'depth{step}',
        after_rgb_hash=f'rgb{step+1}',after_depth_hash=f'depth{step+1}')
    measurement=(dict(status='ESTIMATED',vertical_motion='DOWN' if vertical<-.04 else 'LEVEL',
        uncertainty_m=.022,observation_pair=pair,
        translation_m=dict(x_right=0.,y_up=vertical,z_forward=.25))
        if valid else dict(status='UNKNOWN',reason='inconsistent_or_dynamic_correspondences'))
    return dict(action='MOVE_FORWARD',step=step,subgoal=0,
        pose=dict(heading=-2.094395102393195,motion_epoch=epoch),
        result=dict(rgbd_motion=measurement,observation_before=f'obs_{step:04d}',
            observation_after=f'obs_{step+1:04d}',visual_change=True,
            landing_support=dict(status='NOT_EVALUATED')))


def grounding(index,epoch,entry='UNKNOWN'):
    source=dict(observation_id=f'obs_{index:04d}',subgoal=0,
        pose=dict(heading=-1.8325957145940455,motion_epoch=epoch))
    return dict(status='PARTIAL' if entry=='UNKNOWN' else 'LOCALIZED',
        source=source,target='Descending staircase and bottom of stairs',
        semantic=dict(stair_direction='down',confidence='high',target_visible=True,
            landing_visible=False,lower_landing_visible=False,landing_box=None),
        semantic_presence='VISIBLE',
        position=dict(status='PARTIAL',bearing_degrees=-63.83,
            x_right=-.3,z_forward=1.5),
        entry=dict(status=entry,bearing_degrees=-23.6,
            x_right=-.2,z_forward=1.2,depth_spread_metres=.2),
        landing_geometry=dict(status='UNKNOWN'),
        stair_evidence=dict(direction_rejection_reasons=[]))


def memory():
    obs=[observation(i,1 if i<22 else 4 if i<=25 else 5 if i<=55 else 11)
         for i in range(65)]
    actions=[movement(i,4) for i in range(22,26)]
    actions.append(movement(28,5,valid=False))
    task=TaskMemory()
    task.subgoals=[dict(action='Careful descent to the bottom of the stairs',
        source_text='Begin by descending the stairs, ensuring each step is taken with care.',
        landmark='Stairs',target='Bottom of the stairs',
        success_criteria=['entry_confirmed','actual_descent_observed','target_nearby'])]
    task.subgoal_start_steps={'0':0}
    return SimpleNamespace(
        actions=actions,observations=obs,task=task,
        groundings=[grounding(18,1),grounding(51,5,'LOCALIZED')],
        current=dict(pose=dict(motion_epoch=11),observation_id='obs_0064'),
        camera=SimpleNamespace(forward_step=.25,turn_angle=15),
        scene_memory=None,progress_stage='NOT_STARTED',progress_feedback=None,
        event=Mock())


class CrossEpochHistoryTests(unittest.TestCase):
    def test_registered_descent_survives_many_reanchors(self):
        m=memory()
        history=registered_motion_history(m)
        self.assertEqual(history['registered_down_steps'],[22,23,24,25])
        self.assertEqual(history['unknown_forward_steps'],[28])
        self.assertAlmostEqual(history['observed_downward_m_est'],.44)
        self.assertEqual(history['registered_forward_steps'],4)
        self.assertIn(4,history['motion_epochs'])
        stage,supported,missing,contradictions,detail=_stair_evidence(
            m,m.task.subgoals[0],{'obs_0018','obs_0051'},0.,10.,1.25)
        self.assertEqual(detail['source_observation_ids'][0],'obs_0018')
        self.assertEqual(detail['vertical_motion_steps'],[22,23,24,25])
        self.assertIn('multi_frame_registered_down_motion',supported)
        self.assertEqual(detail['historical_motion_evidence']['registered_down_steps'],[22,23,24,25])
        self.assertNotEqual(stage,'COMPLETED')
        self.assertIn('multi_frame_level_landing_transition',missing)

    def test_no_landing_receipts_means_not_verified_even_with_early_descent(self):
        m=memory()
        with patch('havln_agent.progress.current_visual_evidence',return_value=True):
            result=verify_progress(m,'subgoal_completed','I think I reached bottom',
                ['obs_0018','obs_0051'],10.,1.25)
        self.assertFalse(result['completion_confirmed'])
        self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE')
        self.assertIn('multi_frame_registered_down_motion',result['supported'])
        self.assertEqual(m.task.current_subgoal_index,0)
        self.assertFalse(m.task.completed_subgoals)

    def test_unverified_commands_do_not_create_descent_evidence(self):
        m=memory()
        for a in m.actions:
            a['result']['rgbd_motion']['observation_pair']={}
        h=registered_motion_history(m)
        self.assertFalse(h['registered_down_steps'])
        stage,supported,_,_,d=_stair_evidence(m,m.task.subgoals[0],
            {'obs_0018'},0.,10.,1.25)
        self.assertNotIn('multi_frame_registered_down_motion',supported)
        self.assertFalse(d['vertical_motion_steps'])

    def test_pure_turn_and_foyer_translation_require_correct_evidence(self):
        turn=dict(action='In-place rightward reorientation at the bottom of the stairs',
            source_text='Upon reaching the bottom, make a right turn.',
            landmark='Bottom of the stairs',target='Right-facing orientation before proceeding')
        entry=dict(action='Entry into the foyer following the right turn',
            source_text='As you proceed, you enter an entryway or foyer area where an individual is practicing dance moves.',
            landmark='Foyer',target='Inside the foyer')
        self.assertEqual(turn_requirement(turn)['direction'],'right')
        self.assertEqual(required_conditions(turn),{'orientation_completed'})
        self.assertIsNone(turn_requirement(entry))
        self.assertIn('registered_translation',required_conditions(entry))
        self.assertNotIn('orientation_completed',required_conditions(entry))
        self.assertNotIn('landmark_passed',required_conditions(entry))
        self.assertEqual(normalize_conditions(turn),['orientation_completed'])


class ProvisionalTaskControlTests(unittest.TestCase):
    def runtime(self):
        r=AgentRuntime.__new__(AgentRuntime)
        t=TaskMemory()
        t.subgoals=[dict(action=f'goal{i}') for i in range(3)]
        t.subgoal_start_steps={'0':0}
        r.memory=SimpleNamespace(task=t,actions=[dict(step=0)],
            observations=[dict(observation_id='o0')])
        r._cancel_local_navigation=Mock()
        r.set_semantic_goal=Mock()
        r.event=Mock()
        return r

    def test_model_moves_forward_without_fabricating_verification(self):
        r=self.runtime()
        result=r._select_subgoal(1,'Visually at the bottom with measured descent',['o0'])
        self.assertEqual(result['status'],'OK')
        self.assertEqual(result['selection'],'PROVISIONAL_UNVERIFIED')
        self.assertEqual(r.memory.task.current_subgoal_index,1)
        self.assertEqual(r.memory.task.subgoal_start_steps['1'],1)
        self.assertFalse(r.memory.task.completed_subgoals)
        self.assertEqual(len(r.memory.task.provisional_transitions),1)
        r.set_semantic_goal.assert_called_once_with()

    def test_model_can_revisit_unverified_goal_without_deleting_history(self):
        r=self.runtime()
        r._select_subgoal(1,'new evidence',['o0'])
        r.memory.actions.append(dict(step=1))
        result=r._select_subgoal(0,'recheck earlier evidence',['o0'])
        self.assertEqual(result['status'],'OK')
        self.assertEqual(r.memory.task.current_subgoal_index,0)
        self.assertEqual(r.memory.task.subgoal_started,0)
        self.assertEqual(len(r.memory.task.provisional_transitions),2)
        self.assertFalse(r.memory.task.completed_subgoals)

    def test_model_cannot_skip_unknown_tasks_or_use_fake_citations(self):
        r=self.runtime()
        self.assertEqual(r._select_subgoal(2,'skip',['o0'])['status'],'INVALID_ARGUMENTS')
        self.assertEqual(r._select_subgoal(1,'fake',['missing'])['status'],'INVALID_ARGUMENTS')
        self.assertEqual(r.memory.task.current_subgoal_index,0)
        self.assertFalse(r.memory.task.provisional_transitions)

    def test_verified_subgoal_cannot_be_provisionally_reactivated(self):
        r=self.runtime()
        r._select_subgoal(1,'continue',['o0'])
        r.memory.task.completed_subgoals=[0]
        self.assertEqual(r._select_subgoal(0,'revisit',['o0'])['status'],'INVALID_ARGUMENTS')


if __name__=='__main__':
    unittest.main()
