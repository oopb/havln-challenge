"""v12 action-receipt stance contract: no metric-odometry navigation dependency.

Run: python -m unittest discover -s tests -p 'test_v12_action_stance.py' -v
Only local public RGB-D fixtures, no Habitat datasets or paid API calls.
"""
import math
import unittest
from unittest.mock import Mock, patch

from havln_agent.evidence_ledger import physical_view_key
from havln_agent.heading import same_local_stance
from havln_agent.local_exploration import same_stance, multi_pitch_coverage
from havln_agent.local_motion_controller import LocalMotionController
from havln_agent.mission_history import mission_evidence
from havln_agent.motion import registered_action
from havln_agent.observation import ObservationAudit
from havln_agent.spatial import relative_goal
from test_forward_preflight_v9 import test_runtime, geometry, frame, camera


class ActionStanceV12(unittest.TestCase):
    def test_no_fabricated_metric_pose_at_initialization_or_after_forward(self):
        controller=LocalMotionController(camera())
        initial=controller.pose()
        self.assertEqual(initial['viewpoint_generation'],0)
        self.assertIsNone(initial['x'])
        self.assertIsNone(initial['y'])
        self.assertIsNone(initial['z'])
        self.assertIsNone(initial['position_uncertainty'])
        self.assertFalse(initial['position_coordinates_valid'])
        self.assertFalse(initial['metric_position_valid'])
        controller.accept_feedback('MOVE_FORWARD',dict(
            status='ESTIMATED',observed_frame_change=True,
            translation_m=dict(x_right=42,y_up=-7,z_forward=3),
            uncertainty_m=.001))
        after=controller.pose()
        # Even a numeric legacy measurement cannot silently reintroduce VO.
        self.assertEqual(after['viewpoint_generation'],1)
        self.assertIsNone(after['x'])
        self.assertIsNone(after['y'])
        self.assertIsNone(after['z'])
        self.assertEqual(controller.z,0.)
        self.assertEqual(after['forward_actions_attempted'],1)
        self.assertEqual(after['forward_actions_with_frame_change'],1)
        self.assertEqual(after['unknown_forward_actions'],1)

    def test_turn_and_look_keep_current_stance_and_reuse_prior_views(self):
        r,actions=test_runtime()
        initial=r.motion_controller.pose()
        for action in ('TURN_LEFT','TURN_LEFT','LOOK_DOWN'):
            result=r._step(action)
            self.assertEqual(result['status'],'EXECUTED')
        now=r.motion_controller.pose()
        self.assertEqual(now['viewpoint_generation'],0)
        self.assertTrue(same_local_stance(initial,now))
        self.assertTrue(same_stance(initial,now))
        history=mission_evidence(r.memory,now)
        summary=history['recent_views_at_current_stance']
        self.assertGreaterEqual(summary['distinct_heading_pitch_views'],3)
        self.assertGreaterEqual(len(summary['latest_distinct_views']),3)
        self.assertIsNone(history['current']['position_xz_m_est'])
        self.assertEqual(history['current']['stance_id'],'0:0')
        self.assertEqual(len(r.memory.places),1)

    def test_forward_separates_old_views_without_requiring_rgbd_registration(self):
        r,actions=test_runtime(changed_forwards=True)
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        old_pose=r.motion_controller.pose()
        r._step('TURN_LEFT')
        old_history=mission_evidence(r.memory,r.motion_controller.pose())
        self.assertGreaterEqual(old_history['recent_views_at_current_stance'][
            'distinct_heading_pitch_views'],2)
        with patch('havln_agent.motion.estimate_translation',
                   side_effect=AssertionError('online VO must not be called')):
            execution=r._step('MOVE_FORWARD')
        self.assertEqual(execution['status'],'EXECUTED')
        self.assertEqual(execution['feedback']['rgbd_motion']['status'],
                         'NOT_ESTIMATED')
        self.assertIsNone(execution['feedback']['rgbd_motion']['translation_m'])
        self.assertFalse(execution['feedback']['translation_supported'])
        self.assertTrue(execution['feedback']['visual_change'])
        after=r.motion_controller.pose()
        self.assertEqual(after['viewpoint_generation'],1)
        self.assertFalse(same_local_stance(old_pose,after))
        self.assertFalse(same_stance(old_pose,after))
        self.assertIsNone(after['x'])
        coverage=mission_evidence(r.memory,after)[
            'recent_views_at_current_stance']
        self.assertEqual(coverage['distinct_heading_pitch_views'],1)
        self.assertEqual(len(r.memory.places),2)
        self.assertEqual(r.memory.places['P1']['position'],None)
        self.assertIsNone(r.memory.places['P1']['height_y_m_est'])
        self.assertEqual(r.memory.current['place_id'],'P1')

    def test_full_finite_route_executes_with_live_depth_and_no_online_vo(self):
        r,actions=test_runtime(changed_forwards=True)
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        with patch('havln_agent.motion.estimate_translation',
                   side_effect=AssertionError('no online VO')):
            result=r._navigate_relative('RIGHT',30,.75)
        self.assertEqual(result['status'],'ACTIONS_COMPLETED')
        self.assertEqual(actions,
                         ['TURN_RIGHT','TURN_RIGHT']+['MOVE_FORWARD']*3)
        self.assertEqual(r.motion_controller.viewpoint_generation,3)
        self.assertIsNone(r.motion_controller.target)
        self.assertIsNone(result['original_target'])
        self.assertIsNone(result['pending_target'])
        self.assertFalse(result['endpoint_estimate_valid'])
        self.assertEqual(result['commanded_forward_actions'],3)
        self.assertFalse(result['target_reached'])
        self.assertEqual(r.motion_controller.forward_actions_with_frame_change,3)
        records=[a for a in r.memory.actions
                 if a['action']=='MOVE_FORWARD']
        self.assertTrue(all(a['result']['displacement_verified'] is False
                            for a in records))
        self.assertTrue(all(a['result']['rgbd_motion']['status']==
                            'NOT_ESTIMATED' for a in records))
        route=r.memory.route_progress_summary()
        self.assertIsNone(route['successful_segments'][-1][
            'observed_height_change_m_est'])
        self.assertFalse(route['successful_segments'][-1][
            'displacement_verified'])

    def test_depth_veto_does_not_change_stance_or_spend_forward_action(self):
        r,actions=test_runtime()
        r.geometry=Mock(return_value=geometry('BLOCKED'))
        result=r._navigate_relative('NONE',0,.5)
        self.assertEqual(result['status'],'BLOCKED')
        self.assertEqual(actions,[])
        self.assertEqual(r.motion_controller.viewpoint_generation,0)
        self.assertEqual(r.motion_controller.forward_actions_attempted,0)
        self.assertEqual(r.memory.current['place_id'],'P0')
        self.assertEqual(len(r.memory.evidence.obstacle_blocks),1)

    def test_identical_rgbd_still_interrupts_route_without_claiming_collision(self):
        r,actions=test_runtime()
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        result=r._navigate_relative('NONE',0,.75)
        self.assertEqual(result['status'],'REQUIRES_ATTENTION')
        self.assertEqual(actions,['MOVE_FORWARD'])
        self.assertEqual(r.motion_controller.viewpoint_generation,1)
        self.assertEqual(r.motion_controller.forward_stall_actions,1)
        self.assertEqual(len(r.memory.evidence.motion_stalls),1)
        self.assertFalse(r.memory.evidence.motion_stalls[0]['collision_proven'])
        self.assertEqual(r.memory.actions[0]['evidence_state'],
                         'executed_forward_no_confirmed_displacement')
        self.assertIsNone(r.motion_controller.pose()['z'])

    def test_actual_navigation_tool_has_null_metric_feedback(self):
        r,actions=test_runtime(changed_forwards=True)
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        call=dict(tool='navigate_relative',args=dict(
            turn_direction='NONE',turn_degrees=0,distance_m=.5))
        result=r.execute(call,dict(
            observation_id=r.memory.current['observation_id'],
            actor='navigation_agent'))
        self.assertEqual(result['status'],'ACTIONS_COMPLETED')
        self.assertEqual(result['executed_motion'],
                         ['MOVE_FORWARD','MOVE_FORWARD'])
        self.assertIsNone(result['translation_m'])
        self.assertIsNone(result['vertical_translation_m_est'])
        self.assertEqual(result['translation_confidence'],'not_estimated')
        self.assertEqual(result['vertical_motion'],'UNKNOWN')
        self.assertFalse(result['physical_displacement_verified'])
        self.assertFalse(result['metric_odometry_enabled'])
        self.assertEqual(result['forward_actions_executed'],2)
        self.assertEqual(result['forward_actions_with_RGBD_change'],2)
        self.assertFalse(result['target_reached'])
        self.assertEqual(r.motion_controller.viewpoint_generation,2)

    def test_active_route_can_resume_across_forward_stance_generation(self):
        # A finite model-authored route can be partially executed before
        # budget exhaustion. Synthetic budget extension verifies the resume
        # branch without falsely projecting a metric endpoint.
        r,actions=test_runtime(changed_forwards=True)
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        r.config['max_steps']=2
        partial=r._navigate_relative('NONE',0,.5)
        self.assertEqual(partial['status'],'PARTIAL')
        self.assertEqual(actions,['MOVE_FORWARD'])
        waypoint_id=partial['waypoint_id']
        self.assertEqual(r.active_waypoint_id,waypoint_id)
        self.assertEqual(r.motion_controller.target,waypoint_id)
        self.assertEqual(r.motion_controller.viewpoint_generation,1)
        r.config['max_steps']=88
        done=r._navigate_local(waypoint_id)
        self.assertEqual(done['status'],'ACTIONS_COMPLETED')
        self.assertEqual(actions,['MOVE_FORWARD','MOVE_FORWARD'])
        self.assertEqual(r.motion_controller.viewpoint_generation,2)
        self.assertIsNone(r.motion_controller.target)
        self.assertIsNone(done['remaining_path']['original_endpoint_xz_m'])
        self.assertIsNone(done['original_target'])

    def test_saved_candidate_stales_after_forward_receipt_not_after_turn(self):
        r,_=test_runtime(changed_forwards=True)
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        source=r.motion_controller.pose()
        path=dict(id='sample',angle=0.,distance=.25,
                  source=dict(pose=source))
        r._step('TURN_RIGHT')
        before=r.motion_controller.validate(r,path)
        self.assertEqual(before['status'],'OK')
        r._step('MOVE_FORWARD')
        stale=r.motion_controller.validate(r,path)
        self.assertEqual(stale['status'],'STALE_TARGET')
        self.assertEqual(stale['reason'],
                         'executed_forward_since_waypoint_proposal')
        self.assertEqual(r.motion_controller.viewpoint_generation,1)

    def test_source_linked_action_receipts_not_metric_motion_verification(self):
        r,_=test_runtime(changed_forwards=True)
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        r._step('MOVE_FORWARD')
        entry=r.memory.actions[-1]
        observations={o['observation_id']:o
                      for o in r.memory.observations}
        self.assertTrue(registered_action(entry,observations))
        self.assertEqual(entry['evidence_state'],
                         'executed_forward_with_RGBD_change_not_displacement_proof')
        self.assertFalse(entry['result']['displacement_verified'])
        self.assertIsNone(entry['result']['rgbd_motion']['uncertainty_m'])

    def test_same_stance_grounded_bearing_survives_turn_but_not_forward(self):
        c=LocalMotionController(camera())
        source=c.pose()
        g=dict(
            position=dict(status='LOCALIZED',x_right=.2,
                          z_forward=1.5,depth_spread_metres=.1),
            source=dict(pose=source),
            semantic=dict(confidence='high'))
        c.accept_feedback('TURN_LEFT',None)
        current=relative_goal(g,c.pose())
        self.assertEqual(current['status'],'ESTIMATED')
        self.assertFalse(current['requires_reconfirmation'])
        self.assertFalse(current['metric_robot_position_known'])
        c.accept_feedback('MOVE_FORWARD',dict(observed_frame_change=True))
        invalid=relative_goal(g,c.pose())
        self.assertEqual(invalid['status'],'UNKNOWN')
        self.assertTrue(invalid['requires_reconfirmation'])
        self.assertIn('executed_forward_since_target_source',
                      invalid['reasons'])

    def test_observation_audit_and_obstacle_keys_use_stance_generation(self):
        audit=ObservationAudit(camera())
        p=LocalMotionController(camera())
        first=p.pose()
        audit.update(frame(),first,None)
        p.accept_feedback('MOVE_FORWARD',
                          dict(observed_frame_change=True))
        second=p.pose()
        new=audit.update(frame(),second,'MOVE_FORWARD')
        self.assertFalse(new['repeated_direction'])
        self.assertFalse(new['repeated_view'])
        payload=dict(rgb_hash='same',depth_hash='same',
                     heading_deg=0,pitch_deg=0,motion_epoch=0,
                     local_position_xz_m_est=None)
        self.assertNotEqual(
            physical_view_key(dict(payload,viewpoint_generation=0)),
            physical_view_key(dict(payload,viewpoint_generation=1)))


if __name__=='__main__':
    unittest.main()
