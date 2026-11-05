"""Live forward Depth -> one-frame VLM -> navigation tool result, without stair rules."""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from havln_agent.api_client import APIError
from havln_agent.perception import Camera, Observation, Geometry
from havln_agent.contexts import compact_navigation_context, tool_summary
from havln_agent.visual_perception import OBSTACLE_VISUAL_SCHEMA, VisualPerception
from test_direct_motion_execution import runtime


def depth_probe(blocked=True):
    return dict(state='BLOCKED' if blocked else 'UNCONFIRMED',
        reason='occupied_next_step' if blocked else 'unknown_depth_or_fov',
        depth=dict(path_state='OCCUPIED' if blocked else 'UNKNOWN',
            clearance=0.04 if blocked else 0.0,
            required_forward_step=.25,
            nearest_obstacle_distance=.14,
            nearest_collision_travel=.04,
            nearest_collision_point=dict(pixel_xy=[30,32],
                depth_m=.16,identity='unverified_depth_point'),
            rejection_reasons=['clearance_below_forward_step'] if blocked else
                ['unknown_depth_in_collision_strip'],
            collision_points=75 if blocked else 0,
            unknown_sweep_pixels=0,
            human_occupancy=dict(corridor_points=0,unlocalized=0)))


def center(confirmed=True):
    return dict(confirmed=confirmed,central_support_pixels=120 if confirmed else 0,
        required_support_pixels=4,
        reason='measured_high_centerline_occupancy' if confirmed
               else 'no_coherent_centerline_obstacle',
        nearest_along_m=.14,nearest_depth_m=.16,nearest_height_m=.9,
        source_pixel_xy=[30,32],
        focus_point_xy_norm=[.48,.51],
        focus_region_xyxy_norm=[.4,.3,.6,.7],
        identity='unverified_depth_obstacle')


def visual_result():
    return dict(visible=True,description='A railing-like transparent barrier',
        visual_relation='A wooden floor is visible beyond it',
        uncertainty='Material and physical clearance are unverified')


class ForwardDepthVisualHandoffTests(unittest.TestCase):
    def blocked_runtime(self, center_value=None):
        r=runtime()
        geom=SimpleNamespace(
            movement_probe=Mock(return_value=depth_probe()),
            central_forward_obstruction=Mock(
                return_value=center_value if center_value is not None else center()))
        r.geometry=Mock(return_value=geom)
        r.visual_perception=SimpleNamespace(
            describe_obstacle=Mock(return_value=visual_result()))
        return r,geom

    def test_confirmed_centerline_blocks_before_any_simulator_forward(self):
        r,g=self.blocked_runtime()
        result=r._navigate_relative(0,.5)
        self.assertEqual(result['status'],'BLOCKED')
        self.assertEqual(result['reason'],'measured_centerline_obstacle')
        self.assertEqual(r.step_callback.call_count,0)
        self.assertEqual(len(r.memory.actions),0)
        self.assertEqual(g.movement_probe.call_count,1)
        self.assertEqual(g.central_forward_obstruction.call_count,1)
        r.visual_perception.describe_obstacle.assert_called_once()
        self.assertEqual(result['obstacle_observation']['visual']['description'],
                         visual_result()['description'])
        self.assertEqual(result['obstacle_observation']['depth']['nearest_along_m'],.14)
        self.assertEqual(result['obstacle_observation']['source_observation_id'],'obs_0000')
        self.assertFalse(result['target_reached'])

    def test_identical_rgbd_visual_obstacle_description_is_reused(self):
        r,g=self.blocked_runtime()
        first=r._navigate_relative(0,.5)
        second=r._navigate_relative(0,.5)
        self.assertEqual(first['status'],'BLOCKED')
        self.assertEqual(second['status'],'BLOCKED')
        self.assertTrue(second['obstacle_observation']['visual_reused_for_identical_frame'])
        self.assertEqual(r.visual_perception.describe_obstacle.call_count,1)
        self.assertEqual(r.step_callback.call_count,0)

    def test_uncertain_wide_depth_not_an_automatic_gate_or_vlm_call(self):
        r,g=self.blocked_runtime()
        g.movement_probe.return_value=depth_probe(blocked=False)
        with patch('havln_agent.tools.estimate_translation',
                   return_value=dict(status='UNKNOWN',
                       reason='inconsistent_feature_matches')):
            result=r._navigate_relative(0,.5)
        self.assertEqual(result['status'],'ACTIONS_COMPLETED')
        self.assertEqual([a['action'] for a in r.memory.actions],
                         ['MOVE_FORWARD','MOVE_FORWARD'])
        g.central_forward_obstruction.assert_not_called()
        r.visual_perception.describe_obstacle.assert_not_called()
        self.assertEqual(r.memory.actions[0]['result']
                         ['forward_depth_preflight']['status'],'UNCONFIRMED')

    def test_side_only_depth_block_does_not_prevent_forward(self):
        r,g=self.blocked_runtime(center(False))
        with patch('havln_agent.tools.estimate_translation',
                   return_value=dict(status='UNKNOWN',
                       reason='inconsistent_feature_matches')):
            result=r._navigate_relative(0,.5)
        self.assertEqual(result['status'],'ACTIONS_COMPLETED')
        self.assertEqual(r.step_callback.call_count,2)
        r.visual_perception.describe_obstacle.assert_not_called()
        self.assertFalse(r.memory.actions[0]['result']
                         ['forward_depth_preflight']['center_confirmed'])

    def test_vlm_failure_preserves_measured_block_and_navigation_authority(self):
        r,g=self.blocked_runtime()
        r.visual_perception.describe_obstacle.side_effect=APIError(
            'vision schema failure',category='structured_field')
        result=r._navigate_relative(0,.5)
        self.assertEqual(result['status'],'BLOCKED')
        self.assertEqual(result['obstacle_observation']['visual']['status'],
                         'UNAVAILABLE')
        self.assertEqual(result['obstacle_observation']['depth']['confirmed'],True)
        self.assertEqual(r.step_callback.call_count,0)
        self.assertEqual(r._navigate_relative(0,.5)['status'],'BLOCKED')
        r.visual_perception.describe_obstacle.assert_called_once()

    def test_joint_depth_and_visual_survive_compact_navigation_context(self):
        r,g=self.blocked_runtime()
        result=r._navigate_relative(0,.5)
        summary=tool_summary(dict(result,tool='navigate_relative'))
        context=compact_navigation_context(dict(last_tool=summary))
        handoff=context['last_tool']['obstacle_observation']
        self.assertEqual(handoff['source_observation_id'],'obs_0000')
        self.assertEqual(handoff['depth']['focus_point_xy_norm'],[.48,.51])
        self.assertEqual(handoff['visual']['description'],
                         visual_result()['description'])
        self.assertEqual(handoff['next_action_selected_by'],'navigation_agent')
        self.assertFalse(handoff['visual_reused_for_identical_frame'])

    def test_actual_synthetic_depth_center_vs_side_not_a_stair_keyword(self):
        cam=Camera(width=64,height=64,hfov=90,min_depth=0,max_depth=10,
            normalized=False,camera_height=1.25,robot_height=1.5,
            robot_radius=.1,forward_step=.25,turn_angle=15)
        def g(cols):
            depth=np.full((64,64),1.4,dtype=np.float32)
            depth[:,cols]=.2
            obs=Observation(np.zeros((64,64,3),np.uint8),depth,
                'generic navigation task',time.monotonic(),pitch_deg=0)
            return Geometry(obs,cam,0.0)
        central=g(slice(25,39))
        side=g(slice(45,53))
        self.assertEqual(central.movement_probe()['state'],'BLOCKED')
        self.assertTrue(central.central_forward_obstruction()['confirmed'])
        self.assertEqual(side.movement_probe()['state'],'BLOCKED')
        self.assertFalse(side.central_forward_obstruction()['confirmed'])

    def test_single_rgb_visual_request_is_geometry_focused_and_has_no_actions(self):
        vp=VisualPerception(SimpleNamespace(client=None))
        frame=Observation(np.zeros((64,64,3),np.uint8),
            np.full((64,64),.3,np.float32),
            'descend the stairs',time.monotonic())
        source=dict(observation_id='obs_0065',
            pose=dict(heading=0,pitch_deg=-60))
        with patch('havln_agent.visual_perception.image_url',
                   return_value='data:image/png;base64,AAAA'):
            with patch.object(vp,'complete',return_value=visual_result()) as call:
                returned=vp.describe_obstacle(frame,source,center())
        args,kwargs=call.call_args
        self.assertEqual(args[:2],
            ('visual_perception','near_field_obstacle_description'))
        self.assertEqual(args[3]['depth_focus_xy_norm'],[.48,.51])
        self.assertEqual(args[3]['depth_region_xyxy_norm'],[.4,.3,.6,.7])
        self.assertEqual(args[3]['observation_id'],'obs_0065')
        self.assertEqual(args[3]['pitch_deg'],-60)
        self.assertEqual(args[4],OBSTACLE_VISUAL_SCHEMA)
        self.assertEqual(kwargs['result_schema'],OBSTACLE_VISUAL_SCHEMA)
        self.assertFalse(kwargs['cache'])
        self.assertIn('never',args[2].lower())
        self.assertEqual(returned,visual_result())

    def test_visual_observer_has_no_waypoint_or_completion_schema(self):
        from havln_agent.navigation_agent import OBSERVE_PROMPT
        self.assertIn('last_tool.obstacle_observation',OBSERVE_PROMPT)
        self.assertEqual(set(OBSTACLE_VISUAL_SCHEMA['properties']),
                         {'visible','description','visual_relation','uncertainty'})
        self.assertNotIn('can_move',OBSTACLE_VISUAL_SCHEMA['properties'])
        self.assertNotIn('subgoal_completed',OBSTACLE_VISUAL_SCHEMA['properties'])


if __name__=='__main__':
    unittest.main()
