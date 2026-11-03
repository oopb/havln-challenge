"""Exercise real local-controller + memory with a direct model motion intent."""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from havln_agent.local_motion_controller import LocalMotionController
from havln_agent.memory_service import MemoryService
from havln_agent.perception import Camera, Observation
from havln_agent.runtime import AgentRuntime


def frame():
    return Observation(np.zeros((64,64,3),dtype=np.uint8),
                       np.full((64,64),1.4,dtype=np.float32),
                       'go down the hall',time.monotonic())


def runtime():
    r=AgentRuntime.__new__(AgentRuntime)
    r.camera=Camera(width=64,height=64,hfov=90,min_depth=.1,max_depth=10,
        normalized=False,camera_height=1.25,robot_height=1.7,
        robot_radius=.15,forward_step=.25,turn_angle=15,
        tilt_angle=15,max_step_height=.2)
    r.config=dict(max_steps=128,tool_max_actions=3,waypoint_distance=1.0,
                  max_depth_age_seconds=10,safety_margin=0.0)
    r.motion_controller=LocalMotionController(r.camera)
    r.memory=MemoryService(r)
    r.memory.reset('unit-test','go down the hall')
    r.memory.task.subgoals=[dict(action='proceed',landmark='hall',target='hall')]
    r.observation=None
    r.paths={}
    r.owner=threading.get_ident()
    r.started=time.monotonic()
    r.event_callback=None
    r.event_seq=0
    r.done=False
    r.stop_reason=None
    r.active_waypoint_id=None
    r.active_subgoal_id='sg_0'
    r.plan_revision=1
    r.phase='NAVIGATING'
    r.motion_policy='LEVEL_ONLY'
    r.semantic_motion_policy='LEVEL_ONLY'
    r.context_version='test'
    r.tool_id='tool_0000'
    r.executable_segment=None
    r._authorized_action=True
    r.record_step=None
    r.record_observation=None
    r.load_observation=None
    r.record_tool=None
    r.frames={}
    r.refresh_callback=Mock(side_effect=lambda:frame())
    r.step_callback=Mock(side_effect=lambda action:(frame(),False))
    r.accept_observation(frame())
    return r


class DirectMovementExecutionTests(unittest.TestCase):
    def test_one_model_call_completes_all_turns_and_forward_steps(self):
        r=runtime()
        motion=dict(status='ESTIMATED',vertical_motion='LEVEL',
            translation_m=dict(x_right=0.,y_up=0.,z_forward=.25),
            uncertainty_m=.01)
        # Previously a 30-degree turn used up the per-call limit; now a
        # single finite 0.5m route must complete without asking for a resume.
        with patch('havln_agent.tools.estimate_translation',return_value=motion):
            result=r._navigate_relative(30,.5)
        self.assertEqual(result['status'],'TARGET_REACHED')
        self.assertEqual([x['action'] for x in r.memory.actions],
                         ['TURN_LEFT','TURN_LEFT','MOVE_FORWARD','MOVE_FORWARD'])
        self.assertIsNone(r.motion_controller.target)
        self.assertIsNone(r.active_waypoint_id)
        self.assertEqual(result['requested_motion']['quantized_distance_m'],.5)
        self.assertFalse(result['waypoint_arrival_completes_task'])

    def test_registry_navigation_has_no_atomic_action_limit(self):
        r=runtime()
        r.client=SimpleNamespace(trace_context={})
        r.current_decision_id='nav_unit_test'
        r.navigation_mode='OBSERVE'
        motion=dict(status='ESTIMATED',vertical_motion='LEVEL',
            translation_m=dict(x_right=0.,y_up=0.,z_forward=.25),
            uncertainty_m=.01)
        binding=dict(plan_revision=r.plan_revision,
            subgoal_id=r.active_subgoal_id,
            observation_id=r.memory.current['observation_id'],
            actor='navigation_agent')
        with patch('havln_agent.tools.estimate_translation',return_value=motion):
            outcome=r.execute(dict(tool='navigate_relative',
                args=dict(turn_deg=30,distance_m=.5)),binding)
        self.assertEqual(outcome['status'],'TARGET_REACHED')
        self.assertEqual(outcome['executed_actions'],4)
        self.assertEqual(outcome['requested_actions'],4)
        self.assertEqual(outcome['movement_intent_feedback'][
            'executed_forward_actions'],2)
        self.assertFalse(outcome['movement_intent_feedback'][
            'same_waypoint_pending'])
        self.assertEqual(outcome['executed_motion'],
            ['TURN_LEFT','TURN_LEFT','MOVE_FORWARD','MOVE_FORWARD'])

    def test_episode_budget_is_the_only_action_count_limit(self):
        r=runtime()
        r.config['max_steps']=3  # Two navigation actions + one STOP.
        motion=dict(status='ESTIMATED',vertical_motion='LEVEL',
            translation_m=dict(x_right=0.,y_up=0.,z_forward=.25),
            uncertainty_m=.01)
        with patch('havln_agent.tools.estimate_translation',return_value=motion):
            partial=r._navigate_relative(0,.75)
        self.assertEqual(partial['status'],'PARTIAL')
        self.assertEqual(partial['reason'],'episode_action_budget_exhausted')
        self.assertEqual([x['action'] for x in r.memory.actions],
                         ['MOVE_FORWARD','MOVE_FORWARD'])
        self.assertEqual(len(r.memory.actions),r.config['max_steps']-1)
        # No more physical forward actions can be issued, even to continue
        # the model's otherwise valid pending route.
        self.assertEqual(r._step('MOVE_FORWARD')['status'],'BUDGET_LIMIT')

    def test_arrival_on_final_allowed_action_is_success(self):
        r=runtime()
        r.config['max_steps']=3  # Exactly enough for 0.5m + reserved STOP.
        motion=dict(status='ESTIMATED',vertical_motion='LEVEL',
            translation_m=dict(x_right=0.,y_up=0.,z_forward=.25),
            uncertainty_m=.01)
        with patch('havln_agent.tools.estimate_translation',return_value=motion):
            arrived=r._navigate_relative(0,.5)
        self.assertEqual(arrived['status'],'TARGET_REACHED')
        self.assertEqual(arrived['reason'],'original_endpoint_reached')
        self.assertEqual(len(r.memory.actions),2)
        self.assertIsNone(r.motion_controller.target)

    def test_identical_rgbd_after_forward_interrupts_instead_of_repeating(self):
        r=runtime()
        unknown=dict(status='UNKNOWN',
            reason='inconsistent_with_forward_camera_motion')
        # The test fixture returns byte-identical RGB and depth frames, as
        # observed in 13 real forward actions in evaluation episode 4.
        with patch('havln_agent.tools.estimate_translation',return_value=unknown):
            result=r._navigate_relative(0,.75)
        self.assertEqual(len(r.memory.actions),1)
        self.assertEqual(r.memory.actions[0]['action'],'MOVE_FORWARD')
        feedback=r.memory.actions[0]['result']
        self.assertFalse(feedback['rgb_changed'])
        self.assertFalse(feedback['depth_changed'])
        self.assertEqual(feedback['motion_alert']['reason'],
                         'no_observed_frame_change_after_forward')
        self.assertFalse(feedback['motion_alert']['collision_proven'])
        # Internal _navigate_relative yields the raw controller status; the
        # outer ToolExecutor.execute wraps executed actions as PARTIAL.
        self.assertEqual(result['status'],'REQUIRES_ATTENTION')
        self.assertEqual(result['reason'],
                         'no_observed_frame_change_after_forward')
        self.assertFalse(result['target_reached'])
        self.assertFalse(result['endpoint_estimate_valid'])
        self.assertIsNone(result['remaining_distance'])
        self.assertFalse(result['remaining_path']['endpoint_estimate_valid'])
        self.assertEqual(r.motion_controller.motion_epoch,1)
        self.assertTrue(any(e['event']=='no_observed_forward_motion'
                            for e in r.memory.events))
        self.assertEqual(r.motion_controller.z,0.)

    def test_direct_route_starts_unverified_before_first_live_probe(self):
        r=runtime()
        r._navigate_local=Mock(return_value=dict(status='PARTIAL'))
        result=r._navigate_relative(-45,.75)
        route=r.memory.routes[result['waypoint_id']]
        self.assertEqual(route['evidence_state'],'model_motion_proposal_unverified')
        self.assertEqual(route['traversability'],'UNKNOWN')


if __name__=='__main__':
    unittest.main()
