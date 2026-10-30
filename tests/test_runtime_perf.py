"""Avoid expensive RGB-D geometry reviews during no-motion refresh/turns."""
import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch

import numpy as np

from havln_agent.tools import ToolExecutor
from havln_agent.api_client import APIClient


class RuntimePerformanceTests(unittest.TestCase):
    def test_unchanged_sensor_refresh_reuses_direction_summary(self):
        frame1 = SimpleNamespace(rgb=np.zeros((4, 4, 3), np.uint8),
                                 depth=np.ones((4, 4), np.float32),
                                 pitch_deg=0.0, captured_at=1.)
        frame2 = SimpleNamespace(rgb=frame1.rgb.copy(), depth=frame1.depth.copy(),
                                 pitch_deg=0.0, captured_at=2.)
        pose = dict(x=0.0, z=0.0, heading=0.0)
        current = dict(observation_id="obs_0000", pose=pose,
                       directions=[dict(angle_degrees=0, motion_state="NEEDS_VIEW")])
        executor = ToolExecutor.__new__(ToolExecutor)
        executor.observation = frame1
        executor.memory = SimpleNamespace(current=current, observe=Mock(return_value=current))
        executor.motion_controller = SimpleNamespace(pose=lambda: pose)
        executor.geometry = Mock(side_effect=AssertionError("redundant geometry computation"))
        executor.load_observation = None
        executor.record_observation = None
        executor.frames = {}
        with patch("havln_agent.tools.generate_waypoints", side_effect=AssertionError("redundant sweeps")):
            result = executor.accept_observation(frame2)
        self.assertIs(result, current)
        executor.memory.observe.assert_called_once_with(
            frame2, pose, current["directions"]
        )

    def test_turn_and_stop_never_evaluate_collision_geometry(self):
        for action in ("TURN_LEFT", "TURN_RIGHT", "LOOK_DOWN", "LOOK_UP", "STOP"):
            with self.subTest(action=action):
                frame = SimpleNamespace(
                    rgb=np.zeros((4, 4, 3), np.uint8),
                    depth=np.ones((4, 4), np.float32),
                    captured_at=1.0)
                executor = ToolExecutor.__new__(ToolExecutor)
                executor.done = False
                executor.stop_reason = None
                executor.tool_id = "tool_test"
                executor.context_version = "test"
                executor.active_subgoal_id = "sg_0"
                executor.config = dict(max_steps=500, waypoint_distance=1.25,
                                       safety_margin=0.05)
                executor.observation = frame
                executor.camera = object()
                executor.memory = SimpleNamespace(
                    actions=[], current=dict(observation_id="obs_0000"),
                    scene_memory=None, active_attempt=None)
                executor.memory.record_action = Mock(
                    side_effect=lambda name, pose, feedback: executor.memory.actions.append(
                        dict(action=name, result=feedback)))
                executor.step_callback = Mock(return_value=(frame, action == "STOP"))
                executor.motion_controller = SimpleNamespace(
                    pose=Mock(return_value=dict(heading=0.)),
                    accept_feedback=Mock())
                executor.accept_observation = Mock()
                executor.geometry = Mock(side_effect=AssertionError("no forward geometry needed"))
                executor.record_step = Mock()
                with patch("havln_agent.tools.frame_info", return_value={}), \
                     patch("havln_agent.tools.generate_waypoints",
                           side_effect=AssertionError("no waypoint generation in atomic action")):
                    result = executor._step(action)
                self.assertEqual(result["status"], "EXECUTED")
                self.assertEqual(len(executor.memory.actions), 1)
                self.assertIsNone(executor.record_step.call_args.args[0]["geometry"])
                self.assertEqual(executor.record_step.call_args.args[0]["safety"]["scope"],
                                 "no_forward_displacement")

    def test_inflight_api_marker_reports_phase_without_credentials(self):
        client = APIClient.__new__(APIClient)
        with tempfile.TemporaryDirectory() as temp:
            client.log_path = Path(temp) / "api.jsonl"
            record = dict(role="navigation_agent", purpose="navigation_decision",
                          request_number=4, attempt_number=1, model="test-model",
                          request_phase="connect_and_headers",
                          effective_timeout_seconds=120,
                          error_detail="DO_NOT_LOG_SECRET_TOKEN",
                          base_url="https://example.test/private-path")
            client._inflight(record, "in_progress")
            marker_file = Path(temp) / "api-inflight.json"
            marker = json.loads(marker_file.read_text())
            self.assertEqual(marker["role"], "navigation_agent")
            self.assertEqual(marker["request_phase"], "connect_and_headers")
            self.assertEqual(marker["status"], "in_progress")
            self.assertNotIn("DO_NOT_LOG_SECRET_TOKEN", marker_file.read_text())
            self.assertNotIn("base_url", marker)
            client._inflight(record, "completed")
            self.assertEqual(json.loads(marker_file.read_text())["status"], "completed")


if __name__ == "__main__":
    unittest.main()
