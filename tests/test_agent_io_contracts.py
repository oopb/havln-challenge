"""Agent I/O contract regressions: real image evidence, route continuity, authority."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jsonschema import Draft7Validator

from havln_agent.active_observation import ObservationController, recent_authentic_motion_frames
from havln_agent.api_client import APIError
from havln_agent.memory_records import MemoryRecords
from havln_agent.visual_perception import VisualPerception
from havln_agent.tool_menu import navigation_tool_menu
from havln_agent.contracts import registry
from havln_agent.task_planner import TaskPlanner
from test_active_observation import FakeRuntime
from test_vlm_led_observation import VLMGuide, guidance
from test_rgbd_vlm_approach_search import make_runtime, candidate


def verification(action='finish', status='CONFIRMED', turn=0):
    return dict(guidance(action, turn),
        verification_status=status,
        verification_evidence='Current real RGB-D shows the robot at the destination',
        uncertainty='no further observed step')


class AgentIOContractsTests(unittest.TestCase):
    def test_different_questions_about_same_target_are_not_same_cached_answer(self):
        r=FakeRuntime([])
        r.visual_perception=VLMGuide([guidance(), guidance()])
        first=ObservationController(r).scan_vlm(
            'Where is the staircase entrance?',['stairs'],90)
        second=ObservationController(r).scan_vlm(
            'Are these stairs ascending or descending?',['stairs'],90)
        self.assertEqual(first['status'],'OK')
        self.assertEqual(second['status'],'OK')
        self.assertEqual(len(r.visual_perception.calls),2)

    def test_identical_question_and_stance_still_reuse_result_without_api(self):
        r=FakeRuntime([])
        r.visual_perception=VLMGuide([guidance()])
        scan=ObservationController(r)
        scan.scan_vlm('Locate stairs',['stairs'],90)
        cached=scan.scan_vlm('  LOCATE  STAIRS  ',['stairs'],90)
        self.assertEqual(cached['status'],'NO_NEW_INFORMATION')
        self.assertEqual(len(r.visual_perception.calls),1)

    def test_unblocked_approach_can_turn_and_compare_before_finalizing(self):
        first=dict(candidate(),next_action='turn_left',turn_deg=30,
            expected_progress='ADVANCE')
        second=dict(candidate(),next_action='finish',turn_deg=0,
            expected_progress='DETOUR')
        r,g=make_runtime([first,second],[False,False])
        result=ObservationController(r).scan_vlm(
            'Find best descent direction',['stairs'],90)
        self.assertEqual(r.step_calls,['TURN_LEFT','TURN_LEFT'])
        self.assertEqual(len(r.visual_perception.calls),2)
        self.assertEqual(result['outcome'],'APPROACH_CANDIDATE_FOUND')
        self.assertEqual(len(result['approach_candidates']),2)
        self.assertEqual(result['best_approach']['expected_progress'],'DETOUR')
        self.assertEqual(result['approach_candidates'][0]['expected_progress'],
                         'ADVANCE')
        self.assertFalse(result['usable_for_live_safety'])
        g.assess_step.assert_not_called()

    def test_interrupted_verify_never_commits_provisional_confirmed_answer(self):
        r=FakeRuntime([],fail_on=1)
        r.visual_perception=VLMGuide([
            verification(action='turn_left',status='CONFIRMED',turn=15)])
        result=ObservationController(r).scan_vlm(
            'Is robot at the lower landing?',['landing'],90,mode='VERIFY')
        self.assertEqual(result['status'],'PARTIAL')
        self.assertEqual(result['outcome'],'INTERRUPTED')
        self.assertEqual(result['verification']['status'],'UNCERTAIN')
        self.assertFalse(result['verification']['final_verdict'])
        self.assertEqual(result['verification']['evidence'],'')

    def test_full_panorama_does_not_remove_verify_or_observation_tools(self):
        tools=registry(SimpleNamespace(turn_angle=15,tilt_angle=30),3)
        offered=navigation_tool_menu(tools,{},full_circle_commit=True)
        self.assertIn('observe_for_goal',offered)
        self.assertIn('request_finish',offered)
        self.assertIn('navigate_relative',offered)
        self.assertNotIn('inspect_geometry',offered)
        self.assertNotIn('verify_subgoal',offered)

    @staticmethod
    def visual_runtime():
        task=SimpleNamespace(instruction='Descend stairs',
            current_subgoal_index=0,subgoals=[dict(action='descend',
                target='landing',landmark='stairs',
                completion_description='Robot is on lower flat landing')])
        rt=SimpleNamespace(
            camera=SimpleNamespace(normalized=False,turn_angle=15),
            memory=SimpleNamespace(task=task))
        return VisualPerception(rt)

    def test_dynamic_turn_schema_matches_remaining_budget(self):
        vp=self.visual_runtime()
        vp.complete=Mock(return_value=guidance())
        with patch('havln_agent.visual_perception.image_url',return_value='rgb'):
            vp.guide_scan_view(SimpleNamespace(depth=None),
                dict(observation_id='obs_1',pose=dict(heading=0,pitch_deg=0)),
                'Find route',['stairs'],[],30)
        args,kw=vp.complete.call_args
        self.assertEqual(args[3]['allowed_turn_deg'],[0,15,30])
        turn_enum=kw['result_schema']['properties']['turn_deg']['enum']
        self.assertEqual(turn_enum,[0,15,30])
        self.assertTrue(Draft7Validator(kw['result_schema']).is_valid(guidance()))
        self.assertFalse(Draft7Validator(kw['result_schema']).is_valid(
            guidance('turn_left',90)))

    def test_repeated_same_frame_does_not_duplicate_images_in_session(self):
        vp=self.visual_runtime()
        vp.complete=Mock(return_value=guidance())
        session=[]
        source=dict(observation_id='obs_0010',
            pose=dict(heading=math.radians(45),pitch_deg=0))
        observation=SimpleNamespace(depth='true_depth')
        with patch('havln_agent.visual_perception.image_url',
                   return_value='rgb') as encode_rgb,patch(
                   'havln_agent.visual_perception.depth_image_url',
                   return_value='depth') as encode_depth:
            vp.guide_scan_view(observation,source,'Find descent',['stairs'],[],
                90,session=session)
            first_call=vp.complete.call_args
            vp.guide_scan_view(observation,source,'Find descent',['stairs'],[],
                90,session=session)
            second_call=vp.complete.call_args
        self.assertEqual(encode_rgb.call_count,1)
        self.assertEqual(encode_depth.call_count,1)
        self.assertEqual(first_call.kwargs['image_url'],['rgb','depth'])
        self.assertIsNone(second_call.kwargs['image_url'])
        self.assertIn('route_history',first_call.args[3])
        self.assertNotIn('route_history',second_call.args[3])
        self.assertEqual(second_call.args[3]['route_history_reference'],
                         'initial_session_message')
        self.assertEqual(len(session),4)
        self.assertEqual(len(session[0]['content']),3)
        self.assertEqual(len(session[2]['content']),1)
        self.assertEqual(second_call.kwargs['conversation_messages'],session)

    def test_real_historical_rgbd_are_labeled_and_sent_once(self):
        vp=self.visual_runtime()
        vp.complete=Mock(return_value=guidance())
        source=dict(observation_id='obs_0028',pose=dict(heading=0,pitch_deg=0))
        current=SimpleNamespace(depth='curr')
        old=SimpleNamespace(depth='past')
        session=[]
        with patch('havln_agent.visual_perception.image_url',
                   side_effect=['current_rgb','old_rgb']),patch(
                   'havln_agent.visual_perception.depth_image_url',
                   side_effect=['current_depth','old_depth']):
            vp.guide_scan_view(current,source,'Continue descending',['stairs'],[],
                90,session=session,historical_frames=[(
                    old,dict(observation_id='obs_0024',heading_deg=225,pitch_deg=-60))])
        args,kwargs=vp.complete.call_args
        self.assertEqual(kwargs['image_url'],
            ['current_rgb','current_depth','old_rgb','old_depth'])
        self.assertEqual(args[3]['historical_images'][0]['observation_id'],'obs_0024')
        self.assertTrue(args[3]['historical_images'][0]['historical_only'])
        self.assertEqual(args[3]['historical_images'][0]['image_indices'],[2,3])

    def test_verify_rejects_citations_to_images_not_in_session(self):
        vp=self.visual_runtime()
        vp.complete=Mock(return_value=verification())
        with patch('havln_agent.visual_perception.image_url',return_value='rgb'):
            vp.guide_scan_view(SimpleNamespace(depth=None),
                dict(observation_id='obs_1',pose=dict(heading=0,pitch_deg=0)),
                'Verify arrival',['landing'],[],90,mode='VERIFY')
        kwargs=vp.complete.call_args.kwargs
        with self.assertRaises(APIError):
            kwargs['validate'](dict(verification(),
                evidence_observation_ids=['obs_not_real']))

    def test_prior_motion_keyframes_use_real_saved_rgbd_and_current_subgoal(self):
        r=FakeRuntime([])
        r.memory.route_progress_summary=Mock(return_value=dict(
            successful_segments=[dict(subgoal=0,
                start_observation_id='obs_0024',end_observation_id='obs_0028')],
            blocked_at_prior_stances=[dict(subgoal='sg_0',
                source_observation_id='obs_0044'),
                dict(subgoal='sg_5',source_observation_id='obs_0500')]))
        def read_frame(ident):
            return (SimpleNamespace(rgb='saved_rgb',depth='saved_depth'),
                dict(observation_id=ident,pose=dict(
                    heading=math.radians(225),pitch_deg=-30,motion_epoch=0)))
        r._vision_frame=Mock(side_effect=read_frame)
        frames=recent_authentic_motion_frames(r,'obs_0071')
        self.assertEqual([info['observation_id'] for _,info in frames],
            ['obs_0024','obs_0028','obs_0044'])
        self.assertTrue(all(info['historical_only'] for _,info in frames))
        self.assertEqual(r._vision_frame.call_count,3)

    def test_old_blockage_stores_stance_and_uncertainty_not_global_veto(self):
        mem=MemoryRecords(SimpleNamespace(turn_angle=15))
        mem.actions=[dict(step=4,action='MOVE_FORWARD',subgoal=0,
            pose=dict(heading=math.radians(225),motion_epoch=2),
            result=dict(observation_before='obs_04',
                observation_after='obs_05',rgbd_motion=dict(
                    status='ESTIMATED',uncertainty_m=.05,
                    translation_m=dict(y_up=-.15))))]
        # An old successful descent must survive many later camera turns.
        mem.actions.extend(dict(step=i,action='TURN_LEFT',subgoal=0,
            pose=dict(heading=math.radians(270),motion_epoch=2),
            result={}) for i in range(5,135))
        mem.observations=[dict(observation_id='obs_08',
            pose=dict(x=.4,z=1.2,heading=math.radians(270),
                motion_epoch=2,viewpoint_generation=1),place_id='P1')]
        mem.tool_ledger=[dict(subgoal_id='sg_0',
            call=dict(tool='navigate_relative'),
            request_context=dict(pose=dict(heading=math.radians(225))),
            result=dict(status='BLOCKED',end_observation_id='obs_08',
                reason='measured_centerline_obstacle'))]
        summary=mem.route_progress_summary()
        self.assertEqual(summary['successful_segments'][0]['uncertainty_m_est'],.05)
        blocked=summary['blocked_at_prior_stances'][0]
        self.assertEqual(blocked['heading_deg'],270)
        self.assertEqual(blocked['motion_epoch'],2)
        self.assertEqual(blocked['local_pose_xz_m_est'],[.4,1.2])
        self.assertEqual(blocked['place_id'],'P1')

    def test_task_plan_completion_description_is_optional_for_old_models(self):
        instruction='Descend the stairs.'
        response=dict(subgoals=[dict(action='Descend stairs',
            target='bottom landing',landmark='stairs',
            source_sentence_ids=[1])],final_target='landing',constraints=[])
        class Client:
            def complete(self,*args,**kwargs):
                kwargs['validate'](response)
                return response
        plan=TaskPlanner.parse(Client(),instruction)
        self.assertEqual(plan['subgoals'][0]['completion_description'],
                         'Descend stairs bottom landing')


if __name__=='__main__':
    unittest.main()
