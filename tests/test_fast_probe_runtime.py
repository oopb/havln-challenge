"""Real Geometry and ToolExecutor forward-path regression for on-demand terrain.

The low-cost body sweep is allowed in normal navigation. Full terrain support
is reserved for on-demand inspect_geometry and post-move LEVEL landing evidence.
"""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from havln_agent.perception import Camera, Geometry, Observation
from havln_agent.safety import forward_probe
from havln_agent.tools import ToolExecutor
from havln_agent.waypoint import generate_waypoints


def camera():
    return Camera(width=64, height=64, hfov=90, min_depth=0.1,
                  max_depth=10, normalized=False, camera_height=1.25,
                  robot_height=1.7, robot_radius=0.15, forward_step=0.25,
                  turn_angle=15, tilt_angle=15, max_step_height=0.2)


def observation(depth=1.4):
    return Observation(np.zeros((64,64,3),dtype=np.uint8),
                       np.full((64,64),depth,dtype=np.float32),
                       'descend the stairs',time.monotonic())


class FastProbeRuntimeTests(unittest.TestCase):
    def test_real_geometry_fast_candidates_never_fit_terrain_planes(self):
        cam=camera()
        geometry=Geometry(observation(),cam,0)
        with patch('havln_agent.terrain.review',
                   side_effect=AssertionError('terrain review must be on demand')):
            check=forward_probe(geometry)
            self.assertEqual(check['state'],'EXECUTABLE')
            candidates=generate_waypoints(geometry,0.5,probe=True)
            self.assertTrue(candidates)
            self.assertEqual(check['check_mode'],'fast_live_depth_body_sweep')

    def run_forward(self, motion_direction, landing_expected):
        cam=camera()
        before=observation(1.4)
        after=observation(1.5)
        runner=ToolExecutor.__new__(ToolExecutor)
        runner.camera=cam
        runner.config=dict(max_steps=128,max_depth_age_seconds=5.0,safety_margin=0.0)
        runner.done=False
        runner.stop_reason=None
        runner.tool_id='tool_0000'
        runner.active_subgoal_id='sg_0'
        runner.context_version='test'
        runner.motion_policy='LOCAL_TERRAIN'
        runner.semantic_motion_policy='DESCEND_STAIRS'
        runner.observation=before
        runner.refresh=Mock()
        runner._prepare_forward=Mock(return_value=None)
        geometry=SimpleNamespace(
            human_mask=np.zeros((64,64),dtype=bool),
            human_status=dict(unlocalized=0),
            movement_probe=Mock(return_value=dict(
                state='EXECUTABLE',reason=None,depth=dict(
                    accepted=True,path_state='FREE',clearance=0.25),
                support=dict(status='DEFERRED'))),
            metadata=lambda:dict(camera='fake'))
        runner.geometry=Mock(return_value=geometry)
        runner.step_callback=Mock(return_value=(after,False))
        runner.motion_controller=SimpleNamespace(
            pose=Mock(return_value=dict(x=0,z=0,heading=0)),
            accept_feedback=Mock())
        runner.memory=SimpleNamespace(
            actions=[], current=dict(observation_id='obs_0000'),
            active_attempt=None, scene_memory=None, event=Mock())
        def record_action(action,pose,feedback):
            runner.memory.actions.append(dict(action=action,result=feedback))
        runner.memory.record_action=Mock(side_effect=record_action)
        def accept(observation):
            runner.memory.current=dict(observation_id='obs_0001')
        runner.accept_observation=Mock(side_effect=accept)
        runner.record_step=Mock()
        measurement=dict(status='ESTIMATED',vertical_motion=motion_direction,
                         translation_m=dict(x_right=0,y_up=-0.1,z_forward=0.25),
                         uncertainty_m=0.02)
        with patch('havln_agent.tools.estimate_translation',return_value=measurement), \
             patch('havln_agent.tools.landing_support',
                   return_value=dict(status='SUPPORTED')) as land:
            result=runner._step('MOVE_FORWARD')
        self.assertEqual(result['status'],'EXECUTED')
        self.assertEqual(result['feedback']['landing_support']['status'],
                         'SUPPORTED' if landing_expected else 'NOT_EVALUATED')
        self.assertEqual(land.call_count,1 if landing_expected else 0)
        if motion_direction == 'UP':
            self.assertTrue(result['feedback']['semantic_direction_mismatch'])
            self.assertNotIn('motion_alert',result['feedback'])
        geometry.movement_probe.assert_called_once_with(0.0)
        self.assertEqual(runner.record_step.call_args.args[0]['safety']['scope'],
                         'current_depth_next_step_body_only')

    def test_downward_motion_does_not_trigger_auto_terrain_fit(self):
        self.run_forward('DOWN',False)

    def test_stair_landing_level_motion_keeps_required_completion_evidence(self):
        self.run_forward('LEVEL',True)

    def test_motion_opposite_to_task_is_not_auto_motion_alert(self):
        self.run_forward('UP',False)


if __name__=='__main__':
    unittest.main()
