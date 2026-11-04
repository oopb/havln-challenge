"""Batch 4-12: visual refinement and evidence-led (not forced) task progression."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

from havln_agent.active_observation import ObservationController
from havln_agent.contexts import navigation_progress_review,compact_navigation_context
from havln_agent.navigation_agent import NavigationAgent
from havln_agent.contracts import registry
from test_active_observation import FakeRuntime
from test_vlm_led_observation import VLMGuide,guidance


class Batch412ObservationTests(unittest.TestCase):
    def test_vlm_can_continue_after_found_entry_in_same_observation_task(self):
        r=FakeRuntime([])
        r.visual_perception=VLMGuide([
            guidance('look_down',kind='ENTRY',visible=True),
            guidance('finish')])
        r.memory.note_active_observation_target=Mock()
        result=ObservationController(r).scan_vlm(
            'Find a walkable approach',['entry'],sweep_degrees=180)
        self.assertEqual(result['outcome'],'TARGET_CANDIDATE_FOUND')
        self.assertEqual(result['status'],'OK')
        self.assertEqual(result['candidate_refinement_views'],1)
        self.assertEqual(r.step_calls,['LOOK_DOWN'])
        self.assertEqual(len(r.visual_perception.calls),2)
        self.assertEqual(result['best_candidate']['observation_id'],'obs_0000')
        r.memory.note_active_observation_target.assert_called_once()

    def test_candidate_refinement_is_bounded_without_overriding_vlm_views(self):
        r=FakeRuntime([])
        r.visual_perception=VLMGuide([
            guidance('look_down',kind='ENTRY',visible=True),
            guidance('turn_left',15,kind='ENTRY',visible=True),
            guidance('look_up',kind='CUE',visible=True),
            guidance('turn_left',15,kind='CUE',visible=True)])
        result=ObservationController(r).scan_vlm(
            'Find entry',['entry'],sweep_degrees=360)
        self.assertEqual(result['outcome'],'TARGET_CANDIDATE_FOUND')
        self.assertEqual(result['reason'],'bounded_vlm_candidate_refinement_completed')
        self.assertEqual(result['candidate_refinement_views'],3)
        self.assertEqual(r.step_calls,['LOOK_DOWN','TURN_LEFT','LOOK_UP'])
        self.assertEqual(len(r.visual_perception.calls),4)

    def test_long_180_degree_undo_at_new_pitch_is_returned_to_vlm_not_executed(self):
        r=FakeRuntime([])
        r.visual_perception=VLMGuide([
            guidance('turn_right',90),guidance('turn_right',90),
            guidance('look_down'),guidance('turn_left',90),
            guidance('finish')])
        result=ObservationController(r).scan_vlm(
            'Look for a passage',['entry'],sweep_degrees=360)
        self.assertEqual(result['outcome'],'VLM_SEARCH_FINISHED')
        self.assertEqual(result['view_replan_attempts'],1)
        self.assertEqual(result['covered_turn_deg'],180)
        self.assertEqual(r.step_calls,['TURN_RIGHT']*12+['LOOK_DOWN'])
        self.assertEqual(r.visual_perception.calls[-1]['view_correction']['reason'],
                         'large_reverse_sweep_to_previously_visited_yaw')
        # No physical return to the original angle, no claim of seeing
        # images from intermediate turns.

    def test_vlm_target_hypothesis_does_not_turn_into_safety_or_completion(self):
        r=FakeRuntime([])
        r.visual_perception=VLMGuide([
            guidance('finish',kind='ENTRY',visible=True)])
        result=ObservationController(r).scan_vlm('Find passage',['entry'])
        self.assertFalse(result['usable_for_live_safety'])
        self.assertFalse(result['best_candidate']['geometry']['motion_authorized'])


class Batch412ProgressTests(unittest.TestCase):
    def fixture(self):
        task=SimpleNamespace(current_subgoal_index=0,
            subgoals=[
                dict(action='Go down the stairs',target='stairs',landmark='stairs'),
                dict(action='Turn right to hallway',target='hallway',landmark='doorway')],
            completed_subgoals=[],subgoal_start_steps={'0':0})
        memory=SimpleNamespace(task=task,tool_ledger=[],visual_targets=[
            dict(subgoal_index=0,selected_observation_id='obs_0035',
                target_type='landing',target_kind='CUE',status='CUE_ONLY',
                absolute_heading_deg=90,pitch_deg=-30,
                visual_evidence='Wood floor with railing at edge')])
        history=dict(recorded_forward_steps=5,registered_forward_steps=4,
            unknown_forward_steps=[8],observed_downward_m_est=.6,
            observed_upward_m_est=.3,
            rows=[
                dict(step=2,status='ESTIMATED',delta_y_m=-.30,
                    uncertainty_m=.05,observation_after='obs_0003',motion_epoch=0),
                dict(step=3,status='ESTIMATED',delta_y_m=-.30,
                    uncertainty_m=.05,observation_after='obs_0004',motion_epoch=0),
                dict(step=8,status='UNKNOWN',delta_y_m=None,
                    uncertainty_m=None,observation_after='obs_0009',motion_epoch=1),
                dict(step=10,status='ESTIMATED',delta_y_m=.15,
                    uncertainty_m=.05,observation_after='obs_0011',motion_epoch=1),
                dict(step=11,status='ESTIMATED',delta_y_m=.15,
                    uncertainty_m=.05,observation_after='obs_0012',motion_epoch=1)])
        return memory,history

    def test_registered_up_after_down_surfaces_chronological_reversal(self):
        memory,history=self.fixture()
        with patch('havln_agent.contexts.registered_motion_history',return_value=history):
            result=navigation_progress_review(memory)
        self.assertEqual(result['directional_history'][0]['direction'],'DOWN')
        self.assertEqual(result['directional_history'][1]['direction'],'UP')
        self.assertTrue(result['observed_direction_reversal'])
        self.assertEqual(result['next_subgoal']['action'],'Turn right to hallway')
        self.assertEqual(result['unknown_forward_steps'],1)
        self.assertEqual(result['latest_visual_hypotheses'][0]['observation_id'],
                         'obs_0035')
        self.assertIn('unverified',result['interpretation'])
        self.assertNotIn('completion_confirmed',result)

    def test_unknown_only_motion_cannot_create_milestone_proof(self):
        memory,history=self.fixture()
        history.update(rows=[dict(step=8,status='UNKNOWN',delta_y_m=None,
            uncertainty_m=None)],registered_forward_steps=0,
            observed_downward_m_est=0,observed_upward_m_est=0)
        with patch('havln_agent.contexts.registered_motion_history',return_value=history):
            result=navigation_progress_review(memory)
        self.assertEqual(result['directional_history'],[])
        self.assertFalse(result['observed_direction_reversal'])

    def test_compact_prompt_carries_review_and_model_owns_verification(self):
        memory,history=self.fixture()
        with patch('havln_agent.contexts.registered_motion_history',return_value=history):
            review=navigation_progress_review(memory)
        state=dict(progress_review=review,
            task_selection=dict(current_subgoal_index=0,total_subgoals=2),
            goal=dict(action='Go down the stairs',target='stairs'),
            pose=dict(heading_deg=90))
        compact=compact_navigation_context(state)
        self.assertTrue(compact['progress_review']['observed_direction_reversal'])
        self.assertEqual(compact['progress_review']['directional_history'][-1]['direction'],'UP')
        self.assertEqual(compact['progress_review']['next_subgoal']['target'],'hallway')
        specs=registry(SimpleNamespace(turn_angle=15,tilt_angle=30),3)
        client=Mock()
        client.complete.return_value=dict(tool='select_subgoal',
            args=dict(subgoal_index=1,reason='Model chose a new route',
                      observation_ids=['obs_0035']))
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value=specs),
            memory=SimpleNamespace(keyframes={'obs_0035':{}},
                task=SimpleNamespace(instruction='Descend and turn right')))
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value=state):
            chosen=NavigationAgent(runtime).decide()
        self.assertEqual(chosen['tool'],'select_subgoal')
        self.assertTrue(client.complete.call_args.args[3]['progress_review']
                        ['observed_direction_reversal'])
        self.assertIn('progress_review',client.complete.call_args.args[2])


if __name__=='__main__':
    unittest.main()
