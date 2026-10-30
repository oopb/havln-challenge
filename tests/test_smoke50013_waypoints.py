"""Regression for smoke-500-13: unknown camera bearings must not become routes."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from havln_agent.waypoint import generate_waypoints
from havln_agent.tools import ToolExecutor
from havln_agent.runtime import AgentRuntime
from havln_agent.contexts import tool_summary
from havln_agent.local_motion_controller import LocalMotionController


class GeometryStub:
    def __init__(self, state="NEEDS_VIEW"):
        self.camera = SimpleNamespace(turn_angle=15., hfov=90., forward_step=.25)
        self.state = state

    def assess_step(self, angle=0):
        status = "BLOCKED" if abs(angle) < math.radians(20) else self.state
        return dict(state=status, reason="occupied_next_step" if status=="BLOCKED"
                    else "unknown_depth_or_fov",
                    depth=dict(clearance=0, path_state="OCCUPIED" if status=="BLOCKED" else "UNKNOWN",
                               rejection_reasons=["outside_observed_fov"] if status=="NEEDS_VIEW"
                               else ["clearance_below_forward_step"]),
                    support=dict(status="UNKNOWN"))


class WaypointLoopTests(unittest.TestCase):
    def test_needs_view_is_a_direction_not_navigable_path(self):
        geometry=GeometryStub()
        diagnostics=[]
        paths=generate_waypoints(geometry,.5,diagnostics)
        self.assertEqual(paths,[])
        self.assertTrue(diagnostics)
        self.assertTrue(any(d["motion_state"]=="NEEDS_VIEW" for d in diagnostics))
        self.assertTrue(all(d["motion_state"]!="EXECUTABLE" for d in diagnostics))

    def test_executable_direction_still_produces_waypoint(self):
        geometry=GeometryStub("EXECUTABLE")
        paths=generate_waypoints(geometry,.5)
        self.assertTrue(paths)
        self.assertTrue(all(p["first_step"]=="EXECUTABLE" for p in paths))
        self.assertTrue(all(p["requires_post_turn_recheck"] for p in paths))

    def test_legacy_unverified_waypoint_cannot_start_motion(self):
        executor=ToolExecutor.__new__(ToolExecutor)
        executor.paths={"p0":dict(id="p0",first_step="NEEDS_VIEW",
                                  execution_status="PROPOSED")}
        executor._revalidate_path=Mock(side_effect=AssertionError("must not execute"))
        rejected=executor._execute_waypoint("p0",max_actions=3)
        self.assertEqual(rejected["status"],"NEEDS_VIEW")
        self.assertIn("waypoint_requires_view",rejected["reason"])
        executor._revalidate_path.assert_not_called()

    def test_runtime_returns_camera_views_separately_and_no_routes(self):
        r=AgentRuntime.__new__(AgentRuntime)
        r.motion_policy="LEVEL_ONLY"
        r.camera=SimpleNamespace(forward_step=.25)
        r.refresh=Mock()
        r._generate_waypoints=Mock(return_value=dict(
            paths=[],inspected_directions=[
                dict(angle_degrees=-30.,motion_state="NEEDS_VIEW",path_state="UNKNOWN",
                     reason="unknown_depth_or_fov",rejection_reasons=["outside_observed_fov"]),
                dict(angle_degrees=0.,motion_state="BLOCKED",path_state="OCCUPIED",
                     reason="occupied_next_step",rejection_reasons=["clearance_below_forward_step"])],
            source={"observation_id":"obs_1"}))
        r.paths={}
        r.geometry=Mock(return_value=GeometryStub())
        r.motion_controller=SimpleNamespace(
            needs_view=LocalMotionController.needs_view,
            next_step_state=lambda geom:dict(reason="occupied_next_step"))
        r.memory=SimpleNamespace(note_candidates=Mock())
        value=r._get_waypoint_candidates(purpose="explore")
        self.assertEqual(value["paths"],[])
        self.assertEqual(value["status"],"NEEDS_VIEW")
        self.assertEqual(value["reason"],"alternate_directions_need_new_camera_view")
        self.assertEqual(value["view_directions"][0]["bearing_deg"],-30.)
        self.assertFalse(value["view_directions"][0]["route_authorized"])
        self.assertIn("view_directions",tool_summary(value))
        self.assertEqual(r.motion_policy,"LEVEL_ONLY")


if __name__=="__main__":
    unittest.main()
