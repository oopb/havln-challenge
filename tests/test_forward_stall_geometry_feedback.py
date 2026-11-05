"""Failed forward attempts and explicit depth must reach the navigation model.

No automatic collision gate or deterministic detour/STOP is introduced.
"""
import math
import unittest
from types import SimpleNamespace

from havln_agent.local_exploration import recent_forward_stall_evidence
from havln_agent.contexts import compact_navigation_context, tool_summary


def pose(yaw=0, x=0, generation=0):
    return dict(x=x,y=0,z=0,heading=math.radians(yaw),
                viewpoint_generation=generation,motion_epoch=0,pitch_deg=-60)


def stalled_forward(step, yaw=135, x=0, generation=0):
    return dict(step=step,subgoal=0,action='MOVE_FORWARD',
                pose=pose(yaw,x,generation),result=dict(
                    visual_change=False,rgb_changed=False,depth_changed=False,
                    observation_before='obs_%04d'%step,
                    observation_after='obs_%04d'%(step+1),
                    motion_alert=dict(reason='no_observed_frame_change_after_forward',
                                      collision_proven=False)))


def changed_forward(step, yaw=135):
    return dict(step=step,subgoal=0,action='MOVE_FORWARD',
                pose=pose(yaw),result=dict(
                    visual_change=True,rgb_changed=True,depth_changed=True,
                    observation_after='obs_%04d'%(step+1)))


def memory(actions):
    return SimpleNamespace(actions=actions,
        task=SimpleNamespace(current_subgoal_index=0))


class ForwardStallGeometryFeedbackTests(unittest.TestCase):
    def test_batch415_repeated_identical_forward_across_observation_and_turn(self):
        actions=[changed_forward(52),
                 stalled_forward(65),
                 dict(step=66,subgoal=0,action='LOOK_DOWN',pose=pose(135)),
                 stalled_forward(69),
                 dict(step=70,subgoal=0,action='TURN_LEFT',pose=pose(150)),
                 dict(step=71,subgoal=0,action='TURN_RIGHT',pose=pose(135)),
                 stalled_forward(80),stalled_forward(84)]
        result=recent_forward_stall_evidence(memory(actions),pose(135))
        self.assertEqual(result['total_failed_forward_attempts'],4)
        self.assertEqual(result['current_heading_failed_forward_attempts'],4)
        self.assertEqual(result['failed_bearings'][0]['absolute_heading_deg'],135)
        self.assertEqual(result['last_failed_observation_id'],'obs_0085')
        self.assertFalse(result['collision_proven'])
        self.assertFalse(result['depth_obstacle_inspected'])

    def test_distinct_yaw_is_not_conflated_with_failed_forward_direction(self):
        actions=[stalled_forward(64,135),stalled_forward(65,135),
                 stalled_forward(68,180)]
        result=recent_forward_stall_evidence(memory(actions),pose(180))
        self.assertEqual(result['total_failed_forward_attempts'],3)
        self.assertEqual(result['current_heading_failed_forward_attempts'],1)
        self.assertEqual(len(result['failed_bearings']),2)
        self.assertEqual(result['failed_bearings'][0]['absolute_heading_deg'],180)
        self.assertEqual(result['failed_bearings'][1]['failed_forward_attempts'],2)

    def test_actual_changed_view_or_different_stance_resets_stall_history(self):
        actions=[stalled_forward(12),changed_forward(13),stalled_forward(14)]
        result=recent_forward_stall_evidence(memory(actions),pose(135))
        self.assertEqual(result['total_failed_forward_attempts'],1)
        # A new observed stance must not inherit the old near-obstacle report.
        self.assertIsNone(recent_forward_stall_evidence(
            memory([stalled_forward(5,generation=0)]),pose(135,generation=1)))
        self.assertIsNone(recent_forward_stall_evidence(
            memory([changed_forward(8)]),pose(135)))

    def test_explicit_inspect_geometry_survives_compaction_with_collision_pixel(self):
        receipt=dict(tool='inspect_geometry',status='BLOCKED',
            reason='occupied_next_step',
            geometry=dict(front=dict(
                motion_state='BLOCKED',depth_state='OCCUPIED',
                clearance=0.09,nearest_obstacle_distance=0.19,
                nearest_collision_point=dict(
                    pixel_xy=[120,160],along_m=.19,lateral_m=0.02,
                    depth_m=.21,in_human_semantic_mask=False,
                    identity='unverified_depth_point'),
                rejection_reasons=['clearance_below_forward_step'],
                human_occupancy=dict(corridor_points=0,confirmed=0),
                human_evidence=dict(semantic_identity='UNKNOWN'),
                terrain_support=dict(status='UNKNOWN'),
                unknown_fraction=0.1)),
            directions=[dict(bearing_deg=0,status='BLOCKED',
                reason='occupied_next_step',clearance_m=.09),
                dict(bearing_deg=30,status='EXECUTABLE',
                reason=None,clearance_m=.7)],
            alternatives=[dict(bearing_deg=30,status='EXECUTABLE',
                clearance_m=.7)])
        summary=tool_summary(receipt)
        state=dict(last_tool=summary,
            forward_stall_evidence=recent_forward_stall_evidence(
                memory([stalled_forward(65),stalled_forward(69)]),pose(135)))
        result=compact_navigation_context(state)
        front=result['last_tool']['geometry']
        self.assertEqual(front['nearest_collision_point']['pixel_xy'],[120,160])
        self.assertEqual(front['clearance'],.09)
        self.assertEqual(front['human_evidence']['semantic_identity'],'UNKNOWN')
        self.assertEqual(result['last_tool']['directions'][1]['bearing_deg'],30)
        self.assertEqual(result['last_tool']['alternatives'][0]['clearance_m'],.7)
        self.assertEqual(result['forward_stall_evidence'][
            'current_heading_failed_forward_attempts'],2)
        self.assertFalse(result['forward_stall_evidence']['collision_proven'])
        # The scene label is not fabricated even when the geometry indicates
        # near-field occupancy: a glass barrier vs a human remains unknown.
        self.assertEqual(front['nearest_collision_point']['identity'],
                         'unverified_depth_point')

    def test_vlm_observation_does_not_make_stalls_count_as_progress(self):
        actions=[stalled_forward(40),
                 dict(step=41,subgoal=0,action='TURN_LEFT',pose=pose(150)),
                 dict(step=42,subgoal=0,action='LOOK_UP',pose=pose(150)),
                 dict(step=43,subgoal=0,action='TURN_RIGHT',pose=pose(135)),
                 stalled_forward(44)]
        result=recent_forward_stall_evidence(memory(actions),pose(135))
        self.assertEqual(result['total_failed_forward_attempts'],2)
        self.assertIn('choose_a_new_bearing_or_call_inspect_geometry',
                      result['model_options'])


if __name__=='__main__':
    unittest.main()
