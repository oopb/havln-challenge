"""Regression: unknown bearings remain candidates, confirmed blocks do not."""
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
    def test_unconfirmed_direction_is_a_route_hypothesis(self):
        geometry=GeometryStub()
        diagnostics=[]
        paths=generate_waypoints(geometry,.5,diagnostics)
        self.assertTrue(paths)
        self.assertTrue(diagnostics)
        self.assertTrue(any(d["motion_state"]=="NEEDS_VIEW" for d in diagnostics))
        self.assertTrue(all(p['first_step']!='BLOCKED' for p in paths))

    def test_executable_direction_still_produces_waypoint(self):
        geometry=GeometryStub("EXECUTABLE")
        paths=generate_waypoints(geometry,.5)
        self.assertTrue(paths)
        self.assertTrue(all(p["first_step"]=="EXECUTABLE" for p in paths))
        self.assertTrue(all(p["requires_post_turn_recheck"] for p in paths))

    def test_historical_blocked_preview_does_not_veto_new_model_route(self):
        executor=ToolExecutor.__new__(ToolExecutor)
        executor.paths={"p0":dict(id="p0",first_step="BLOCKED",
                                  execution_status="PROPOSED")}
        executor._revalidate_path=Mock(return_value=dict(
            status='STALE_TARGET',reason='synthetic_pose_error'))
        result=executor._execute_waypoint("p0")
        self.assertEqual(result["status"],'STALE_TARGET')
        executor._revalidate_path.assert_called_once()

    def test_runtime_returns_camera_views_separately_and_no_routes(self):
        r=AgentRuntime.__new__(AgentRuntime)
        r.motion_policy="LEVEL_ONLY"
        r.camera=SimpleNamespace(forward_step=.25,turn_angle=15.)
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
            next_step_state=lambda geom:dict(reason="occupied_next_step"),
            pose=lambda:dict(x=0.,y=0.,z=0.,heading=0.,pitch_deg=0.,motion_epoch=0))
        r.memory=SimpleNamespace(note_candidates=Mock(),observations=[],
            task=SimpleNamespace(current_subgoal_index=0))
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
