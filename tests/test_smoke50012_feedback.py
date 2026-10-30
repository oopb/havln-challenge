"""smoke-500-12: measured block feedback and looped view novelty regressions."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

from havln_agent.contexts import collision_summary, tool_summary
from havln_agent.memory_records import MemoryRecords
from havln_agent.perception import Camera, Observation, Geometry


def make_camera():
    return Camera(width=16,height=16,hfov=90,min_depth=.05,max_depth=5,
                  normalized=False,camera_height=1,robot_height=1.5,
                  robot_radius=.1,forward_step=.25,turn_angle=15,tilt_angle=30)


def pose(pitch, x=0):
    return dict(x=x,y=0,z=0,heading=0,pitch_deg=pitch,
                motion_epoch=0,position_uncertainty=.05)


def frame(value, pitch, captured):
    return Observation(np.full((16,16,3), value, np.uint8),
                       np.ones((16,16),np.float32),
                       "Descend the stairs.", captured, pitch)


class ObservationLoopTests(unittest.TestCase):
    def test_revisiting_old_pitch_is_not_new_keyframe(self):
        memory=MemoryRecords(make_camera())
        a=memory.observe(frame(40,0,1),pose(0),[])
        b=memory.observe(frame(180,-30,2),pose(-30),[])
        again=memory.observe(frame(40,0,3),pose(0),[])
        self.assertTrue(a["new_visual_information"])
        self.assertTrue(b["new_visual_information"])
        self.assertFalse(again["new_visual_information"])
        self.assertTrue(again["revisited_static_view"])
        self.assertFalse(again["representative_keyframe"])
        self.assertIn(a["observation_id"],again["revisited_view_ids"])

    def test_new_position_is_not_static_revisit_permission(self):
        memory=MemoryRecords(make_camera())
        memory.observe(frame(40,0,1),pose(0),[])
        moved=memory.observe(frame(40,0,2),pose(0,x=.25),[])
        self.assertFalse(moved["revisited_static_view"])
        self.assertTrue(moved["new_visual_information"])

    def test_collision_details_survive_to_navigator(self):
        hit=dict(path_state="OCCUPIED",clearance=.141,required_forward_step=.25,
                 nearest_obstacle_distance=.213,nearest_collision_travel=.141,
                 collision_points=1721,unknown_sweep_pixels=0,
                 rejection_reasons=["clearance_below_forward_step"],
                 human_occupancy=dict(confirmed=0,uncertain=0,unlocalized=0,corridor_points=0),
                 nearest_collision_point=dict(pixel_xy=[109,140],height_above_local_origin_m=.32))
        result=dict(tool="navigate_local",status="BLOCKED",reason="occupied_next_step",
                    safety=hit,validation=dict(geometry=hit))
        summary=tool_summary(result)
        self.assertEqual(summary["measured_collision"]["clearance"],.141)
        self.assertEqual(summary["measured_collision"]["required_forward_step"],.25)
        self.assertEqual(summary["measured_collision"]["nearest_collision_point"]["pixel_xy"],[109,140])
        self.assertIn("semantic identity unverified",summary["measured_collision"]["identity_note"])

    def test_safety_diagnostic_contains_nearest_measured_depth_pixel(self):
        cam=SimpleNamespace(width=4,height=4,hfov=90,max_depth=5.,
                            forward_step=.25,robot_height=1.5,robot_radius=.1)
        g=Geometry.__new__(Geometry)
        g.camera=cam
        g.margin=.05
        g.fx=2.
        g.valid=np.ones((4,4),bool)
        g.depth=np.ones((4,4),float)
        g.x=np.zeros((4,4),float)
        g.z=np.ones((4,4),float)
        g.height=np.zeros((4,4),float)
        g.z[2,2]=.3
        g.height[2,2]=.4
        g.obstacles=np.zeros((4,4),bool)
        g.obstacles[2,2]=True
        g.human_mask=np.zeros((4,4),bool)
        g.human_status=dict(confirmed=0,uncertain=0,unlocalized=0)
        g._unknown_sweep=Mock(return_value=(np.zeros(0,bool),[]))
        report=g.diagnose(0,.25)
        self.assertEqual(report["path_state"],"OCCUPIED")
        self.assertEqual(report["nearest_collision_point"]["pixel_xy"],[2,2])
        self.assertAlmostEqual(report["nearest_collision_point"]["height_above_local_origin_m"],.4)
        self.assertFalse(report["nearest_collision_point"]["in_human_semantic_mask"])


if __name__=="__main__":
    unittest.main()
