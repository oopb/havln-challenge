"""Memory-owned evidence, current RGB, and runnable tool-registry regressions.

This suite deliberately tests no candidate-id coupling or historical
subgoal-reopening state machine. Movement remains model-controlled.
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
from havln_agent.navigation_agent import NavigationAgent
from havln_agent.surface_measurement import measure_surface_region
from havln_agent.tool_menu import navigation_tool_menu
from havln_agent.visual_perception import SCAN_GUIDANCE_SCHEMA
from havln_agent.active_observation import ObservationController
from test_active_observation import FakeRuntime
from test_vlm_led_observation import VLMGuide


class EvidenceContractTests(unittest.TestCase):
    def test_registry_instantiates_with_real_camera_parameters(self):
        # Batch 4-21 crashed before any episode action because PR #69 used
        # object_schema(..., required=...), unsupported by the helper.
        # Schema construction MUST complete for ordinary navigation workers.
        camera=SimpleNamespace(turn_angle=15,tilt_angle=30)
        spec=registry(camera,3,pitch_deg=-60)
        self.assertEqual(len(spec),20)
        schema=spec['navigate_relative']['parameters']
        self.assertTrue(Draft7Validator(schema).is_valid(
            dict(turn_deg=0,distance_m=.5)))
        self.assertFalse(Draft7Validator(schema).is_valid(
            dict(turn_deg=0,distance_m=.5,candidate_id='removed')))
        self.assertNotIn('review_completed_subgoal',spec)
        offered=navigation_tool_menu(spec,dict(task_selection=dict(
            verified_subgoals=[0],current_subgoal_index=1)))
        self.assertNotIn('review_completed_subgoal',offered)

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

    def test_normal_observation_evidence_accepts_496_characters(self):
        value=dict(target_visible=True,target_type='stairs',
            confidence='low',box=[.1,.2,.7,.9],evidence='x'*496,
            uncertainty='',target_kind='CUE',candidate_approach=False,
            next_action='finish',turn_deg=0,reason='done')
        self.assertTrue(Draft7Validator(SCAN_GUIDANCE_SCHEMA).is_valid(value))

if __name__ == '__main__':
    unittest.main()
