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
        self.assertIsNone(result['remaining_distance'])
        self.assertFalse(result['endpoint_estimate_valid'])
        self.assertEqual(result['recovery']['local_anchor_observation_id'],'obs_0002')
        self.assertTrue(result['recovery']['missing_translation_still_unknown'])
        self.assertEqual(r.motion_controller.pose()['coordinate_scope'],
                         'reanchored_local_relative')

    def test_unknown_motion_one_step_continues_same_waypoint_next_call(self):
        r,path,actions,geometry=self.agent()
        first=r._execute_waypoint('p0',1)
        self.assertEqual(first['status'],'PARTIAL')
        self.assertEqual(first['recovery']['mode'],'CONTINUE_BOUNDED_RELATIVE_MOTION')
        self.assertIsNotNone(r.motion_controller.target)
        self.assertTrue(r.memory.active_attempt['unknown_motion_seen'])
        self.assertIsNone(first['remaining_distance'])
        self.assertFalse(first['endpoint_estimate_valid'])
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

    def test_memory_place_epoch_changes_without_fabricated_route(self):
        from havln_agent.memory_service import MemoryService
        runtime=SimpleNamespace(camera=SimpleNamespace(),pitch_deg=0)
        memory=MemoryService(runtime)
        memory.reset('ep','Descend the stairs')
        memory.places={'P0':dict(id='P0',position=[0.,0.],height_y_m_est=0.,
            confidence='estimated',position_uncertainty_m=.05,motion_epoch=0,
            observation_ids=['obs_0000'],headings_deg=[0.])}
        memory.current_place_id='P0'
        record=dict(observation_id='obs_0001',pose=dict(
            x=0.,y=0.,z=0.,heading=0.,pitch_deg=0.,
            motion_epoch=1,position_uncertainty=.3),
            rgb_hash='a',depth_hash='b',captured_at=1.,
            representative_keyframe=True)
        memory.note_observation(record)
        self.assertEqual(memory.current_place_id,'P1')
        self.assertEqual(memory.places['P1']['motion_epoch'],1)
        self.assertEqual(memory.routes,{})
        self.assertEqual(record['place_id'],'P1')

    def test_old_global_target_is_stale_but_new_local_target_valid(self):
        r,old,actions,geometry=self.agent()
        source=old['source']['pose']
        r.motion_controller.accept_feedback('MOVE_FORWARD',dict(status='UNKNOWN'))
        self.assertFalse(r.motion_controller.pose()['global_chain_continuous'])
        self.assertFalse(r.motion_controller.pose()['local_anchor_observation_id'])
        self.assertTrue(r.motion_controller.anchor_relative_frame('obs_new'))
        r.observation.captured_at=time.monotonic()
        # An old global target must stay invalid even after a fresh local
        # anchor is established; the unknown motion is not reconstructed.
        stale=r.motion_controller.validate(r,old)
        self.assertEqual(stale['status'],'NEEDS_VIEW')
        self.assertEqual(stale['reason'],'pose_uncertainty_requires_reconfirmation')
        new_pose=r.motion_controller.pose()
        self.assertNotEqual(source['motion_epoch'],new_pose['motion_epoch'])
        new_path=dict(id='new',angle=0.,distance=.25,
            source=dict(subgoal=0,pose=new_pose))
        current=r.motion_controller.validate(r,new_path)
        self.assertEqual(current['status'],'OK')
        self.assertEqual(current['validation']['next_atomic_action'],'MOVE_FORWARD')
        self.assertEqual(current['validation']['relative_motion_only'],False)
        self.assertFalse(current['validation']['requires_post_turn_recheck'])

    def test_navigation_llm_receives_unknown_distance_and_local_anchor(self):
        from havln_agent.contexts import tool_summary
        result=dict(tool='navigate_local',status='PARTIAL',
            interrupted_status='REQUIRES_ATTENTION',
            reason='relative_forward_horizon_exhausted_without_verified_pose',
            endpoint_estimate_valid=False,remaining_distance=None,
            recovery=dict(mode='REANCHOR_CURRENT_RGBD',
                local_anchor_observation_id='obs_0099',
                coordinate_scope='new_local_relative_only',
                missing_translation_still_unknown=True))
        summary=tool_summary(result)
        self.assertIsNone(summary['remaining_distance'])
        self.assertFalse(summary['endpoint_estimate_valid'])
        self.assertEqual(summary['recovery']['local_anchor_observation_id'],'obs_0099')
        self.assertEqual(summary['interrupted_status'],'REQUIRES_ATTENTION')

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
