"""Exercise real local-controller + memory with a direct model motion intent."""
import threading
import time
import unittest
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
    r.owner=threading.get_ident()
    r.started=time.monotonic()
    r.event_callback=None
    r.event_seq=0
    r.done=False
    r.stop_reason=None
    r.active_waypoint_id=None
    r.active_subgoal_id='sg_0'
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
    def test_turn_move_pause_observe_resume_and_validate_route(self):
        r=runtime()
        motion=dict(status='ESTIMATED',vertical_motion='LEVEL',
            translation_m=dict(x_right=0.,y_up=0.,z_forward=.25),
            uncertainty_m=.01)
        with patch('havln_agent.tools.estimate_translation',return_value=motion):
            result=r._navigate_relative(30,.5,3)
            self.assertEqual(result['status'],'PARTIAL')
            self.assertEqual([x['action'] for x in r.memory.actions],
                             ['TURN_LEFT','TURN_LEFT','MOVE_FORWARD'])
            pid=result['waypoint_id']
            self.assertEqual(r.active_waypoint_id,pid)
            self.assertEqual(r.memory.routes[pid]['evidence_state'],
                             'geometry_supported')
            self.assertEqual(r.memory.routes[pid]['state'],'EXECUTABLE')
            self.assertEqual(r.motion_controller.target is not None,True)
            # The model may observe without cancelling its bounded local route.
            r._step('TURN_LEFT')
            self.assertEqual(r.active_waypoint_id,pid)
            completed=r._navigate_local(pid,3)
        self.assertEqual(completed['status'],'TARGET_REACHED')
        self.assertIsNone(r.motion_controller.target)
        self.assertIsNone(r.active_waypoint_id)
        self.assertEqual([a['action'] for a in r.memory.actions].count('MOVE_FORWARD'),2)
        self.assertEqual([a['action'] for a in r.memory.actions].count('TURN_RIGHT'),1)

    def test_direct_route_starts_unverified_before_first_live_probe(self):
        r=runtime()
        r._navigate_local=Mock(return_value=dict(status='PARTIAL'))
        result=r._navigate_relative(-45,.75,1)
        route=r.memory.routes[result['waypoint_id']]
        self.assertEqual(route['evidence_state'],'model_motion_proposal_unverified')
        self.assertEqual(route['traversability'],'UNKNOWN')


if __name__=='__main__':
    unittest.main()
