"""Regression: an UNKNOWN RGB-D translation must not immediately drop a short route.

No simulator ground truth is used. Every continuing forward action requires new
live geometry. Even at the commanded horizon, arrival is NEVER claimed without
measured pose evidence.
"""
import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from havln_agent.local_motion_controller import LocalMotionController
from havln_agent.tools import ToolExecutor


class CurrentGeometry:
    def __init__(self, blocked=False):
        self.blocked = blocked
        self.diagnose_calls = 0

    def diagnose(self, bearing, distance):
        self.diagnose_calls += 1
        return dict(path_state='OCCUPIED' if self.blocked else 'FREE',
            clearance=0 if self.blocked else distance,
            required_forward_step=distance, rejection_reasons=(
                ['clearance_below_forward_step'] if self.blocked else []))

    def assess_step(self, angle=0):
        return dict(state='BLOCKED' if self.blocked else 'EXECUTABLE',
            reason='occupied_next_step' if self.blocked else None,
            support=dict(status='SUPPORTED'),
            depth=dict(clearance=0 if self.blocked else .25))


class RelativeMotionTests(unittest.TestCase):
    def agent(self, max_actions=2, outcomes=None, blocked_after=None):
        camera=SimpleNamespace(forward_step=.25, turn_angle=15)
        executor=ToolExecutor.__new__(ToolExecutor)
        executor.camera=camera
        executor.config=dict(max_depth_age_seconds=3)
        executor.motion_controller=LocalMotionController(camera)
        controller=executor.motion_controller
        origin=controller.pose()
        path=dict(id='p0', angle=0.,distance=.5,first_step='EXECUTABLE',
            source=dict(subgoal=0,pose=origin),execution_status='PROPOSED')
        executor.paths={'p0':path}
        executor.active_waypoint_id=None
        executor.executable_segment=None
        executor.done=False
        executor.observation=SimpleNamespace(captured_at=time.monotonic())
        executor.refresh=Mock(side_effect=lambda:self.update_capture(executor))
        executor.source=lambda:dict(observation_id=executor.memory.current['observation_id'],
                                    pose=controller.pose())
        geom=CurrentGeometry()
        executor.geometry=lambda:geom
        memory=SimpleNamespace(current=dict(observation_id='obs_0000'),
            active_attempt=None,task=SimpleNamespace(current_subgoal_index=0),
            note_validation=Mock())
        def start_attempt(chosen, pose):
            memory.active_attempt=dict(target=chosen,interruptions=[])
        def finish_attempt(status,reason=None):
            memory.active_attempt=None
        memory.start_attempt=start_attempt
        memory.finish_attempt=Mock(side_effect=finish_attempt)
        executor.memory=memory
        executor._revalidate_path=Mock(return_value=dict(
            status='OK',original_target=[0.,.5],
            executable_segment=dict(path_id='p0')))
        results=list(outcomes or ['UNKNOWN','ESTIMATED'])
        actions=[]
        def execute_action(action):
            actions.append(action)
            index=len(actions)
            if action!='MOVE_FORWARD':
                raise AssertionError('unexpected turn / look during short straight recovery')
            status=results.pop(0)
            if status=='UNKNOWN':
                measurement=dict(status='UNKNOWN')
            else:
                measurement=dict(status='ESTIMATED',
                    translation_m=dict(x_right=0.,y_up=0.,z_forward=.25),
                    uncertainty_m=.02)
            controller.accept_feedback(action,measurement)
            executor.memory.current=dict(observation_id=f'obs_{index:04d}')
            if blocked_after is not None and index>=blocked_after:
                geom.blocked=True
            return dict(status='EXECUTED',feedback=dict(
                translation_supported=status=='ESTIMATED',
                motion_alert=dict(reason='motion_estimate_unknown') if status=='UNKNOWN' else None))
        executor._step=execute_action
        return executor,path,actions,geom

    @staticmethod
    def update_capture(agent):
        agent.observation.captured_at=time.monotonic()

    def test_unknown_first_move_allows_one_more_checked_forward_then_reanchors(self):
        r,path,actions,geometry=self.agent()
        result=r._execute_waypoint('p0',2)
        self.assertEqual(actions,['MOVE_FORWARD','MOVE_FORWARD'])
        self.assertEqual(geometry.diagnose_calls,1)
        self.assertEqual(result['status'],'REQUIRES_ATTENTION')
        self.assertEqual(result['reason'],
            'relative_forward_horizon_exhausted_without_verified_pose')
        self.assertEqual(result['recovery']['mode'],'REANCHOR_CURRENT_RGBD')
        self.assertIsNone(r.motion_controller.target)
        self.assertEqual(r.motion_controller.motion_epoch,1)
        self.assertEqual(r.motion_controller.z,.25) # only measured second motion
        self.assertFalse(result['target_reached'])

    def test_unknown_motion_one_step_continues_same_waypoint_next_call(self):
        r,path,actions,geometry=self.agent()
        first=r._execute_waypoint('p0',1)
        self.assertEqual(first['status'],'PARTIAL')
        self.assertEqual(first['recovery']['mode'],'CONTINUE_BOUNDED_RELATIVE_MOTION')
        self.assertIsNotNone(r.motion_controller.target)
        self.assertTrue(r.memory.active_attempt['unknown_motion_seen'])
        second=r._execute_waypoint('current',1)
        self.assertEqual(actions,['MOVE_FORWARD','MOVE_FORWARD'])
        self.assertEqual(second['status'],'REQUIRES_ATTENTION')
        self.assertEqual(second['recovery']['executed_forward_actions'],2)

    def test_fresh_blocked_geometry_still_denies_second_move(self):
        r,path,actions,geometry=self.agent(blocked_after=1)
        result=r._execute_waypoint('p0',2)
        self.assertEqual(actions,['MOVE_FORWARD'])
        self.assertEqual(result['status'],'BLOCKED')
        self.assertEqual(result['reason'],'occupied_next_step')
        self.assertEqual(path['execution_status'],'BLOCKED')

    def test_relative_motion_cannot_steer_to_stale_global_endpoint(self):
        r,path,actions,geometry=self.agent()
        first=r._execute_waypoint('p0',1)
        r.motion_controller.accept_feedback('TURN_LEFT',None)
        result=r._execute_waypoint('current',1)
        self.assertEqual(actions,['MOVE_FORWARD'])
        self.assertEqual(result['status'],'REQUIRES_ATTENTION')
        self.assertEqual(result['reason'],'relative_motion_heading_changed')

    def test_normal_measured_steps_still_reach_target(self):
        r,path,actions,geometry=self.agent(outcomes=['ESTIMATED','ESTIMATED'])
        result=r._execute_waypoint('p0',2)
        self.assertEqual(result['status'],'TARGET_REACHED')
        self.assertEqual(actions,['MOVE_FORWARD','MOVE_FORWARD'])
        self.assertEqual(r.motion_controller.motion_epoch,0)

    def test_one_step_horizon_cannot_retry_unknown_motion(self):
        r,path,actions,geometry=self.agent()
        path['distance']=.25
        r._revalidate_path.return_value=dict(
            status='OK',original_target=[0.,.25],
            executable_segment=dict(path_id='p0'))
        result=r._execute_waypoint('p0',2)
        self.assertEqual(actions,['MOVE_FORWARD'])
        self.assertEqual(result['status'],'REQUIRES_ATTENTION')
        self.assertFalse(result['target_reached'])


if __name__ == '__main__':
    unittest.main()
