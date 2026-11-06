"""Evidence-grounded multimodal navigation regression and interface contracts.

Tests require no Habitat runtime, no external API keys and no privileged pose.
Their inputs model real RGB-D receipts and deliberately uncertain hypotheses.
"""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from jsonschema import Draft7Validator

from havln_agent.api_client import APIError
from havln_agent.contracts import registry
from havln_agent.evidence_ledger import EvidenceLedger
from havln_agent.memory_records import MemoryRecords
from havln_agent.memory_service import MemoryService
from havln_agent.active_observation import ObservationController
from test_active_observation import FakeRuntime
from test_vlm_led_observation import VLMGuide
from havln_agent.navigation_agent import NavigationAgent
from havln_agent.surface_measurement import measure_surface_region
from havln_agent.tool_menu import navigation_tool_menu
from havln_agent.visual_perception import (
    VisualPerception, SCAN_GUIDANCE_SCHEMA, COMPLETION_REVIEW_SCHEMA)


def test_candidate(progress='ADVANCE'):
    return dict(id='visual_target_0001',selected_observation_id='obs_0066',
        absolute_heading_deg=90,expected_progress=progress)


def test_action(dy=.073,uncertainty=.034):
    return dict(step=65,action='MOVE_FORWARD',subgoal=0,
        pose=dict(heading=math.pi/2,motion_epoch=0),
        result=dict(observation_before='obs_0064',
            observation_after='obs_0065',
            rgbd_motion=dict(status='ESTIMATED',uncertainty_m=uncertainty,
                translation_m=dict(y_up=dy))))


class EvidenceContractTests(unittest.TestCase):
    def test_predicted_descent_measured_ascent_records_explicit_contradiction(self):
        ledger=EvidenceLedger()
        actual=ledger.motion_outcome(test_candidate(),test_action(),'DOWN')
        self.assertEqual(actual['prediction_consistency'],'CONTRADICTED_BY_MOTION')
        self.assertEqual(actual['candidate_id'],'visual_target_0001')
        self.assertEqual(actual['actual_heading_deg'],90)
        self.assertEqual(actual['measured_vertical_motion'],'UP')
        self.assertEqual(actual['vertical_change_m_est'],.073)
        self.assertEqual(actual['candidate_source_observation_id'],'obs_0066')
        self.assertFalse(actual['movement_permission'])

    def test_different_candidate_types_do_not_turn_motion_into_false_veto(self):
        ledger=EvidenceLedger()
        self.assertEqual(ledger.motion_outcome(
            test_candidate('DETOUR'),test_action(),'DOWN')
            ['prediction_consistency'],'INCONCLUSIVE')
        self.assertEqual(ledger.motion_outcome(
            test_candidate(),test_action(dy=None,uncertainty=None),'DOWN')
            ['prediction_consistency'],'INCONCLUSIVE')
        self.assertEqual(ledger.motion_outcome(
            test_candidate(),test_action(dy=-.11),'DOWN')
            ['prediction_consistency'],'SUPPORTED_BY_MOTION')

    def test_obstacle_bearing_is_measured_after_turn_at_real_camera_pose(self):
        ledger=EvidenceLedger()
        blocked=ledger.blocked_forward(dict(observation_id='obs_0070',
            rgb_hash='rgb',depth_hash='depth',subgoal=0),
            dict(x=1.3,z=-.4,heading=0.,pitch_deg=-60,
                motion_epoch=2,viewpoint_generation=3),
            reason='measured_centerline_obstacle',
            center=dict(central_support_pixels=27,nearest_along_m=.2))
        self.assertEqual(blocked['heading_deg'],0)
        self.assertEqual(blocked['observation_id'],'obs_0070')
        self.assertEqual(blocked['local_position_xz_m_est'],[1.3,-.4])
        self.assertEqual(blocked['motion_epoch'],2)
        self.assertFalse(blocked['forward_atomic_action_executed'])

    def test_surface_depth_is_height_measurement_not_completion_verdict(self):
        yy,xx=np.mgrid[0:80,0:100]
        geo=SimpleNamespace(depth=np.ones((80,100)),
            valid=np.ones((80,100),dtype=bool),
            height=np.full((80,100),-.524)+xx*.00001,
            x=(xx-50)/100.,z=1+yy/100.)
        outcome=measure_surface_region(geo,[.2,.2,.8,.8],'obs_0025')
        self.assertEqual(outcome['status'],'MEASURED')
        self.assertAlmostEqual(outcome['relative_height_m_est'],-.524,places=2)
        self.assertFalse(outcome['measured_completion'])
        self.assertEqual(outcome['reachability'],'UNKNOWN')
        self.assertEqual(outcome['surface_identity'],'VLM_PROPOSED_UNVERIFIED')

    def test_surface_measurement_rejects_invalid_roi_and_sparse_depth(self):
        yy,xx=np.mgrid[0:80,0:100]
        geo=SimpleNamespace(depth=np.ones((80,100)),
            valid=np.zeros((80,100),dtype=bool),
            height=xx*0.,x=xx*0.,z=yy*0.)
        with self.assertRaises(ValueError):
            measure_surface_region(geo,[.9,.2,.1,.7],'obs_0')
        measurement=measure_surface_region(geo,[0,0,1,1],'obs_0')
        self.assertEqual(measurement['status'],'INSUFFICIENT_DEPTH')
        self.assertFalse(measurement['measured_completion'])

    def test_navigation_agent_receives_matching_current_rgb_and_pose(self):
        source=dict(observation_id='obs_0065',rgb_hash='hashRGB',
            depth_hash='hashDepth',pose=dict(
                heading=math.pi/2,pitch_deg=-60,x=1,z=2,motion_epoch=0))
        camera=SimpleNamespace(turn_angle=15,tilt_angle=30)
        client=SimpleNamespace(complete=Mock(return_value=dict(
            tool='observe_for_goal',args=dict(mode='APPROACH',
                objective='Find accessible descent',target_types=['stairs']))))
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=lambda:registry(camera,3,pitch_deg=-60),
            memory=SimpleNamespace(episode_id='ep',keyframes={},task=
                SimpleNamespace(instruction='Descend',current_subgoal_index=0),
                current=dict(source,observation_id='obs_0065')),
            observation=SimpleNamespace(rgb=np.zeros((4,4,3),dtype=np.uint8)),
            source=lambda:source,event=Mock())
        runtime.memory.current['pose']=source['pose']
        state=dict(goal=dict(description='Descend'),
            pose=dict(heading_deg=90,motion_epoch=0,
                place_id='P0',position_xz_m_est=[1,2]),
            task_selection=dict(current_subgoal_index=0))
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value=state),patch(
                   'havln_agent.navigation_agent.image_url',
                   return_value='data:image/png;base64,AAAA'):
            NavigationAgent(runtime).decide()
        _args,kwargs=client.complete.call_args
        self.assertEqual(kwargs['image_url'],'data:image/png;base64,AAAA')
        self.assertEqual(_args[3]['current_rgb']['observation_id'],'obs_0065')
        self.assertEqual(_args[3]['current_rgb']['heading_deg'],90)
        self.assertTrue(_args[3]['current_rgb']['image_attached'])
        runtime.event.assert_called_once()

    def test_navigation_agent_cannot_attach_rgb_to_mismatched_pose(self):
        camera=SimpleNamespace(turn_angle=15,tilt_angle=30)
        runtime=SimpleNamespace(client=Mock(),registry=None,
            make_registry=lambda:registry(camera,3),
            memory=SimpleNamespace(episode_id='ep',keyframes={},task=
                SimpleNamespace(instruction='Descend'),
                current=dict(observation_id='obs_A',rgb_hash='A',
                    depth_hash='D',pose=dict(heading=0))),
            observation=SimpleNamespace(rgb=np.zeros((4,4,3),dtype=np.uint8)),
            source=lambda:dict(observation_id='obs_B',rgb_hash='B',
                depth_hash='D',pose=dict(heading=0)))
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value={}):
            with self.assertRaises(APIError):
                NavigationAgent(runtime).decide()
        runtime.client.complete.assert_not_called()

    def test_review_reopens_previous_and_supersedes_dependents_without_deletion(self):
        mem=MemoryRecords(SimpleNamespace(turn_angle=15))
        mem.task.subgoals=[dict(action='descend'),dict(action='turn'),
                           dict(action='foyer')]
        mem.task.completed_subgoals=[0,1]
        mem.task.current_subgoal_index=2
        mem.task.subgoal_start_steps={'0':0,'1':5,'2':8}
        mem.task.completion_evidence=[
            dict(subgoal=0,status='ACTIVE',observation_ids=['obs_0028'],
                 verification=dict(source_observation_id='obs_0028')),
            dict(subgoal=1,status='ACTIVE',observation_ids=['obs_0035'],
                 verification=dict(source_observation_id='obs_0035'))]
        mem.observations=[
            dict(observation_id='obs_0028',step=4),
            dict(observation_id='obs_0035',step=7),
            dict(observation_id='obs_0043',step=10)]
        mem.current=mem.observations[-1]
        mem.actions=[dict(step=0,action='MOVE_FORWARD',subgoal=0)]
        actual=mem.apply_completion_review(0,'REOPEN',
            'New RGB-D contradicts prior physical arrival',
            ['obs_0028','obs_0043'],'obs_0043')
        self.assertEqual(actual['review_outcome'],'REOPENED')
        self.assertEqual(actual['superseded_subgoals'],[0,1])
        self.assertEqual(mem.task.current_subgoal_index,0)
        self.assertEqual(mem.task.completed_subgoals,[])
        self.assertEqual([e['status'] for e in mem.task.completion_evidence],
                         ['SUPERSEDED','SUPERSEDED'])
        self.assertEqual(mem.task.subgoal_generations['0'],1)
        self.assertEqual(len(mem.actions),1)
        self.assertEqual(mem.progress_stage,'IN_PROGRESS')

    def test_review_uncertain_does_not_change_task_state(self):
        mem=MemoryRecords(SimpleNamespace(turn_angle=15))
        mem.task.subgoals=[dict(action='descend'),dict(action='turn')]
        mem.task.completed_subgoals=[0]
        mem.task.current_subgoal_index=1
        mem.task.completion_evidence=[dict(subgoal=0,observation_ids=['old'],
            verification=dict(source_observation_id='old'))]
        mem.observations=[dict(observation_id='old',step=2),
                          dict(observation_id='new',step=5)]
        mem.current=mem.observations[-1]
        answer=mem.apply_completion_review(0,'UNCERTAIN','Not enough evidence',
                                           ['old','new'],'new')
        self.assertEqual(answer['review_outcome'],'UNCERTAIN')
        self.assertEqual(mem.task.current_subgoal_index,1)
        self.assertEqual(mem.task.completed_subgoals,[0])

    def test_review_requires_newer_real_sensor_frame(self):
        mem=MemoryRecords(SimpleNamespace(turn_angle=15))
        mem.task.subgoals=[dict(action='descend'),dict(action='turn')]
        mem.task.completed_subgoals=[0]
        mem.task.current_subgoal_index=1
        mem.task.completion_evidence=[dict(subgoal=0,observation_ids=['old'],
            verification=dict(source_observation_id='old'))]
        mem.observations=[dict(observation_id='old',step=4)]
        mem.current=mem.observations[0]
        answer=mem.apply_completion_review(0,'REOPEN','An opinion alone',
                                           ['old'],'old')
        self.assertNotEqual(answer.get('review_outcome'),'REOPENED')
        self.assertEqual(mem.task.completed_subgoals,[0])

    def test_model_review_receives_old_and_new_rgbd_and_validates_citations(self):
        camera=SimpleNamespace(normalized=False,turn_angle=15,tilt_angle=30)
        vp=VisualPerception(SimpleNamespace(camera=camera))
        vp.complete=Mock(return_value=dict(verdict='REOPEN',
            rationale='New frame contradicts arrival',uncertainty='',
            evidence_observation_ids=['old','new'],missing_evidence=''))
        sources=[dict(observation_id=ident,pose=dict(heading=0,pitch_deg=0))
                 for ident in ('old','new')]
        frames=[(SimpleNamespace(rgb='source',depth='source'),source)
                for source in sources]
        with patch('havln_agent.visual_perception.image_url',
                   return_value='RGB'),patch(
                   'havln_agent.visual_perception.depth_image_url',
                   return_value='DEPTH'):
            vp.review_completion(dict(action='descend'),
                dict(evidence='Old claimed floor arrival'),frames,dict())
        args,kw=vp.complete.call_args
        self.assertEqual(kw['image_url'],['RGB','DEPTH','RGB','DEPTH'])
        self.assertEqual([v['observation_id'] for v in args[3]['image_order']],
                         ['old','new'])
        with self.assertRaises(APIError):
            kw['validate'](dict(verdict='REOPEN',rationale='Wrong frames',
                evidence_observation_ids=['old','fake']))

    def test_tool_menu_exposes_review_only_after_a_confirmed_milestone(self):
        camera=SimpleNamespace(turn_angle=15,tilt_angle=30)
        specs=registry(camera,3)
        self.assertNotIn('review_completed_subgoal',
                         navigation_tool_menu(specs,{}))
        offered=navigation_tool_menu(specs,dict(task_selection=dict(
            current_subgoal_index=1,verified_subgoals=[0])))
        self.assertIn('review_completed_subgoal',offered)
        self.assertEqual(offered['review_completed_subgoal']['permission'],
                         'NAVIGATION')
        schema=specs['navigate_relative']['parameters']
        self.assertTrue(Draft7Validator(schema).is_valid(dict(
            turn_deg=0,distance_m=.25,candidate_id='visual_target_0001')))
        self.assertTrue(Draft7Validator(schema).is_valid(dict(
            turn_deg=0,distance_m=.25)))
        self.assertFalse(Draft7Validator(schema).is_valid(dict(
            turn_deg=0,distance_m=.25,candidate_id='')))

    def test_memory_service_links_only_explicitly_selected_candidate_to_motion(self):
        runtime=SimpleNamespace(camera=SimpleNamespace(turn_angle=15),
            semantic_motion_policy='DESCEND_STAIRS')
        mem=MemoryService(runtime)
        mem.visual_targets=[dict(test_candidate(),subgoal_index=0,
            subgoal_generation=0)]
        actual=mem.record_candidate_motion('visual_target_0001',test_action())
        self.assertEqual(actual['prediction_consistency'],'CONTRADICTED_BY_MOTION')
        self.assertEqual(len(mem.evidence.route_outcomes),1)
        self.assertIsNone(mem.record_candidate_motion('nonexistent',test_action()))
        self.assertEqual(len(mem.evidence.route_outcomes),1)

    def test_blocked_route_summary_uses_exact_post_turn_pose_without_old_keyframe(self):
        mem=MemoryRecords(SimpleNamespace(turn_angle=15))
        blocked=mem.evidence.blocked_forward(
            dict(observation_id='obs_0070',subgoal=0),
            dict(x=1,z=2,heading=0,pitch_deg=-60,motion_epoch=1),
            reason='measured_centerline_obstacle',center={})
        mem.tool_ledger=[dict(call=dict(tool='navigate_relative'),
            result=dict(status='BLOCKED',blocked_at=blocked),
            subgoal_id='sg_0')]
        route=mem.route_progress_summary()
        self.assertEqual(route['blocked_at_prior_stances'][0]['heading_deg'],0)
        self.assertEqual(route['blocked_at_prior_stances'][0]
                         ['source_observation_id'],'obs_0070')

    def test_verify_uses_same_frame_depth_measurement_then_reassesses(self):
        from havln_agent.evidence_ledger import EvidenceLedger
        r=FakeRuntime([])
        r.memory.actions=[]
        r.memory.evidence=EvidenceLedger()
        first=dict(verification_status='NOT_YET',
            verification_evidence='Floor appears lower than robot',
            uncertainty='surface identity unclear',
            next_action='finish',turn_deg=0,reason='request surface measure',
            candidate_surface_box=[.3,.4,.7,.9])
        second=dict(verification_status='NOT_YET',
            verification_evidence='Depth is insufficient; not at landing',
            uncertainty='low depth',next_action='finish',turn_deg=0,
            reason='final grounded assessment')
        r.visual_perception=VLMGuide([first,second])
        output=ObservationController(r).scan_vlm('Verify landing',
            ['landing'],90,mode='VERIFY')
        self.assertEqual(output['outcome'],'SUBGOAL_NOT_YET')
        self.assertEqual(len(r.visual_perception.calls),2)
        self.assertEqual(r.step_calls,[])
        self.assertEqual(r.visual_perception.calls[1]['view_correction']
                         ['reason'],'requested_surface_depth_measurement')
        self.assertEqual(len(r.memory.evidence.surface_measurements),1)

    def test_reopened_subgoal_does_not_reuse_old_confirmed_observation(self):
        r=FakeRuntime([])
        r.memory.actions=[]
        r.memory.evidence=EvidenceLedger()
        r.memory.task.subgoal_generations={'0':1}
        old=dict(mode='vlm_led',intent='VERIFY',subgoal=0,
            subgoal_generation=0,objective='Verify landing',
            target_types=['landing'],outcome='SUBGOAL_CONFIRMED',
            start_pose=r.pose(),start_rgb_hash=None,start_depth_hash=None)
        r.memory.observation_searches.append(old)
        reply=dict(verification_status='NOT_YET',
            verification_evidence='Robot not yet at target',
            uncertainty='',next_action='finish',turn_deg=0,reason='new review')
        r.visual_perception=VLMGuide([reply])
        outcome=ObservationController(r).scan_vlm('Verify landing',
            ['landing'],90,mode='VERIFY')
        self.assertEqual(outcome['outcome'],'SUBGOAL_NOT_YET')
        self.assertEqual(len(r.visual_perception.calls),1)
        self.assertEqual(r.memory.observation_searches[-1]['subgoal_generation'],1)

    def test_normal_observation_evidence_accepts_496_characters(self):
        value=dict(target_visible=True,target_type='stairs',
            confidence='low',box=[.1,.2,.7,.9],evidence='x'*496,
            uncertainty='',target_kind='CUE',candidate_approach=False,
            next_action='finish',turn_deg=0,reason='done')
        self.assertTrue(Draft7Validator(SCAN_GUIDANCE_SCHEMA).is_valid(value))


if __name__=='__main__':
    unittest.main()
