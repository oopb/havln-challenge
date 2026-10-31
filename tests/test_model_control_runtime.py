"""Direct-motion API, visual bearing, paused route, and safe evidence contracts."""
import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from jsonschema import Draft7Validator

from havln_agent.contracts import registry
from havln_agent.runtime import AgentRuntime
from havln_agent.perception import Camera, Geometry, Observation
from havln_agent.spatial import relative_goal


def camera():
    return Camera(width=64,height=64,hfov=90,min_depth=.1,max_depth=10.,
                  normalized=False,camera_height=1.25,robot_height=1.7,
                  robot_radius=.15,forward_step=.25,turn_angle=15,
                  tilt_angle=30,max_step_height=.2)


def direct_runtime():
    r=AgentRuntime.__new__(AgentRuntime)
    r.camera=camera()
    pose=dict(x=0.,y=0.,z=0.,heading=0.,pitch_deg=0.,
              motion_epoch=0,position_uncertainty=.05,motion_error_total=0)
    r.motion_controller=SimpleNamespace(pose=lambda:dict(pose),target=None)
    r.memory=SimpleNamespace(tool_ledger=[],task=SimpleNamespace(current_subgoal_index=0))
    r.refresh=Mock()
    r.source=Mock(return_value=dict(observation_id='obs_0000',
        rgb_hash='r',depth_hash='d',pose=pose,subgoal=0,
        captured_at=time.monotonic()))
    r._navigate_local=Mock(return_value=dict(status='PARTIAL',
        reason='tool_action_limit'))
    r.paths={}
    return r


class ModelControlRuntimeTests(unittest.TestCase):
    def test_model_direct_motion_creates_resumable_route_without_candidates(self):
        r=direct_runtime()
        result=r._navigate_relative(turn_deg=45,distance_m=2.,max_atomic_steps=3)
        self.assertEqual(result['status'],'PARTIAL')
        self.assertEqual(result['waypoint_id'],'direct_0000')
        self.assertEqual(r.paths['direct_0000']['type'],'EXPLORATION_WAYPOINT')
        self.assertEqual(r.paths['direct_0000']['distance'],2.)
        self.assertTrue(r.paths['direct_0000']['requires_post_turn_recheck'])
        self.assertEqual(r.paths['direct_0000']['objective']['purpose'],
                         'model_directed_motion')
        r._navigate_local.assert_called_once_with('direct_0000',3)
        self.assertEqual(result['requested_motion']['authorization'],
                         'live_per_atomic_action_only')

    def test_model_continuous_intent_is_quantized_not_rejected(self):
        r=direct_runtime()
        got=r._navigate_relative(19.,.61,2)
        self.assertEqual(got['status'],'PARTIAL')
        self.assertEqual(got['requested_motion']['quantized_turn_deg'],15.)
        self.assertEqual(got['requested_motion']['quantized_distance_m'],.5)
        self.assertEqual(r.paths['direct_0000']['distance'],.5)

    def test_out_of_bounds_motion_does_not_execute(self):
        r=direct_runtime()
        self.assertEqual(r._navigate_relative(15,3.25,1)['status'],'INVALID_ARGUMENTS')
        self.assertEqual(r._navigate_relative(float('nan'),.5,1)['status'],'INVALID_ARGUMENTS')
        r._navigate_local.assert_not_called()
        self.assertEqual(r.paths,{})

    def test_pixel_bearing_uses_current_depth_as_hypothesis_not_permission(self):
        r=direct_runtime()
        obs=Observation(np.zeros((64,64,3),dtype=np.uint8),
                        np.full((64,64),2.,dtype=np.float32),
                        'find exit',time.monotonic())
        r.geometry=Mock(return_value=Geometry(obs,r.camera,0.05))
        r._navigate_relative=Mock(return_value=dict(status='PARTIAL',
                                                  waypoint_id='direct_0000'))
        outcome=r._navigate_to_pixel(.75,.5,.5,2)
        self.assertEqual(outcome['status'],'PARTIAL')
        self.assertLess(outcome['pixel_target']['selected_turn_deg'],0)
        self.assertEqual(outcome['pixel_target']['measured_depth_m'],2.)
        self.assertIn('not_walkability_certificate',
                      outcome['pixel_target']['interpretation'])
        self.assertEqual(r._navigate_relative.call_args.args[1:],(.5,2))

    def test_unknown_pixel_depth_requires_view_without_motion(self):
        r=direct_runtime()
        obs=Observation(np.zeros((64,64,3),dtype=np.uint8),
                        np.full((64,64),np.nan,dtype=np.float32),
                        'find exit',time.monotonic())
        r.geometry=Mock(return_value=Geometry(obs,r.camera,0.05))
        r._navigate_relative=Mock()
        self.assertEqual(r._navigate_to_pixel(.5,.5,.5,1)['status'],'NEEDS_VIEW')
        r._navigate_relative.assert_not_called()

    def test_registry_allows_open_targets_and_direct_motion(self):
        tools=registry(camera(),3)
        self.assertTrue(Draft7Validator(tools['navigate_relative']['parameters'])
                        .is_valid(dict(turn_deg=30,distance_m=1.5,max_atomic_steps=3)))
        self.assertTrue(Draft7Validator(tools['navigate_to_pixel']['parameters'])
                        .is_valid(dict(u=.65,v=.33,distance_m=.5,max_atomic_steps=1)))
        self.assertEqual(tools['navigate_relative']['permission'],'NAVIGATION')
        self.assertTrue(Draft7Validator(tools['observe_for_goal']['parameters'])
                        .is_valid(dict(objective='locate the dancer',
                            target_types=['dancing person'],direction='left',
                            sweep_degrees=30,force_recheck=True)))

    def test_grounding_target_is_not_replaced_by_scene_specific_entry(self):
        pose=dict(x=0.,z=0.,heading=0.,position_uncertainty=.05,motion_epoch=0,
                  motion_error_total=0.)
        sample=dict(status='LOCALIZED',x_right=0.,z_forward=1.5,
                    depth_spread_metres=.1)
        entry=dict(status='LOCALIZED',x_right=1.,z_forward=.2,
                   depth_spread_metres=.01)
        grounding=dict(position=sample,entry=entry,
            semantic=dict(stair_direction='down',confidence='high'),
            source=dict(pose=pose))
        result=relative_goal(grounding,pose)
        self.assertEqual(result['region'],'target_point')
        self.assertAlmostEqual(result['bearing_deg'],0.,places=4)


if __name__=='__main__':
    unittest.main()
