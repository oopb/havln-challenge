"""Offline contract tests for the independent Codex MCP navigation path.

No Habitat installation, cloud API, Codex login or Docker daemon required.
"""
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import unittest
from zipfile import ZipFile
from unittest.mock import patch, MagicMock, Mock

import numpy as np

from havln_agent.codex_episode import CodexEpisode
from havln_agent.codex_mcp import MCPServer
from havln_agent.replay_server import ReplaySource, EpisodeReplay
from scripts.pack_agent_diagnostics import pack
from scripts.run_codex_episode import (preflight, require_mcp_runtime,
                                       summarize_codex_events, run_codex_cli,
                                       require_fresh_container_output,
                                       collect_codex_audit, archive_codex_run)
from havln_agent.perception import Camera, adapt_observation
from scripts.run_codex_episode import (build_command, load_episode_instruction, PROMPT, START_PROMPT)


def camera():
    return Camera(width=16, height=12, hfov=90, min_depth=0,
                  max_depth=10, normalized=False, camera_height=1.5,
                  robot_height=1.6, robot_radius=0.1,
                  forward_step=0.25, turn_angle=15)


class FakeHabitat:
    def __init__(self):
        self.current_episode = SimpleNamespace(
            episode_id="1",
            instruction=SimpleNamespace(instruction_text="Go down the stairs."))
        self.actions = []
        self.habitat_env = SimpleNamespace(sim=self)

    def get_observations_at(self, keep_agent_at_new_pose):
        return self.image()

    def image(self):
        return {
            "rgb": np.full((12, 16, 3), len(self.actions) % 255, dtype=np.uint8),
            "depth": np.ones((12, 16), dtype=np.float32),
            "instruction": {"text": "Go down the stairs."},
        }

    def reset(self):
        return self.image()

    def step(self, action):
        self.actions.append(action)
        return self.image(), 0.0, action == "STOP", {}


class CodexEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.env = FakeHabitat()
        self.controller = CodexEpisode(
            self.env, camera(), "1", "test", 4, self.tempdir.name)

    def test_observe_returns_real_png_without_repeated_instruction(self):
        with patch("havln_agent.codex_episode.read_current_observation",
                   side_effect=lambda env, cam, instruction:
                   adapt_observation(env.image(), cam)):
            info, encoded, depth = self.controller.observe()
        self.assertNotIn("instruction", info)
        self.assertEqual(self.controller.instruction, "Go down the stairs.")
        self.assertEqual(json.loads((self.controller.directory / "episode.json").read_text())["instruction"],
                         "Go down the stairs.")
        self.assertEqual(base64.b64decode(encoded)[:8],
                         bytes([137, 80, 78, 71, 13, 10, 26, 10]))
        self.assertEqual(self.env.actions, [])
        self.assertIsNone(depth)
        self.assertFalse(info["depth_sent"])
        self.assertNotIn("look_down_for_clearance", info)
        self.assertEqual(info["pitch_deg"], 0.0)
        self.assertTrue((self.controller.directory /
                         ("observations/" + info["observation_id"] + ".npz")).exists())

    def test_front_obstacle_alone_triggers_optional_depth_image(self):
        base = self.env.image()
        base["depth"] = np.full((12, 16), 0.2, dtype=np.float32)
        with patch("havln_agent.codex_episode.read_current_observation",
                   return_value=adapt_observation(base, camera())):
            info, rgb, depth = self.controller.observe()
        self.assertTrue(info["depth_sent"])
        self.assertEqual(info["depth_reason"], "front_obstacle")
        self.assertTrue(info["look_down_for_clearance"])
        self.assertLess(info["front_obstacle_m"], camera().forward_step + camera().robot_radius)
        self.assertEqual(base64.b64decode(rgb)[:8], bytes([137, 80, 78, 71, 13, 10, 26, 10]))
        self.assertEqual(base64.b64decode(depth)[:8], bytes([137, 80, 78, 71, 13, 10, 26, 10]))
        self.assertEqual(self.env.actions, [])

        # MCP image order is RGB first, Depth second, only for this trigger.
        with patch("havln_agent.codex_episode.read_current_observation",
                   return_value=adapt_observation(base, camera())):
            result = MCPServer(self.controller)._dispatch("tools/call", {
                "name": "observe", "arguments": {}})
        self.assertEqual([part["type"] for part in result["content"]],
                         ["text", "image", "image"])
        metadata = json.loads(result["content"][0]["text"])
        self.assertTrue(metadata["depth_sent"])
        self.assertNotIn("look_down_for_clearance", metadata)
        self.assertNotEqual(result["content"][1]["data"], result["content"][2]["data"])

    def test_first_observe_at_changed_heading_includes_depth(self):
        server = MCPServer(self.controller)
        self.controller.step(["TURN_RIGHT"])
        result = server._dispatch("tools/call", {
            "name": "observe", "arguments": {"include_depth": False}})
        self.assertEqual([p["type"] for p in result["content"]], ["text", "image", "image"])
        self.assertEqual(json.loads(result["content"][0]["text"])["depth_reason"],
                         "view_changed")
        receipt = json.loads((self.controller.directory / "tools.jsonl")
                             .read_text().splitlines()[-1])
        self.assertEqual(receipt["call"]["args"], {"include_depth": False})

        repeated = server._dispatch("tools/call", {"name": "observe"})
        self.assertEqual([p["type"] for p in repeated["content"]], ["text", "image"])
        self.assertEqual(self.env.actions, ["TURN_RIGHT"])

    def test_observe_rejects_invalid_depth_argument(self):
        server = MCPServer(self.controller)
        for arguments in ({"include_depth": None}, {"include_depth": "true"},
                          {"include_depth": 1}, {"depth": True}, []):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                server._dispatch("tools/call", {"name": "observe", "arguments": arguments})
        self.assertEqual(self.controller.tool_count, 0)

    def test_imminent_side_obstacle_sends_depth_and_blocks_forward(self):
        # The legacy center-only corridor misses a close RIGHT side obstacle.
        # A coherent public-Depth patch at right x=0.053..0.088 m is within
        # the robot body width, but outside the center +/-0.025 m corridor.
        from havln_agent.perception import Geometry
        raw = self.env.image()
        raw["depth"] = np.full((12, 16), 4.0, dtype=np.float32)
        raw["depth"][3:9, 9:11] = 0.28
        observation = adapt_observation(raw, camera())
        self.assertFalse(Geometry(observation, camera(), margin=0.0)
                         .central_forward_obstruction()["confirmed"])
        with patch("havln_agent.codex_episode.read_current_observation",
                   return_value=observation):
            result = MCPServer(self.controller)._dispatch(
                "tools/call", {"name": "observe", "arguments": {}})
        self.assertEqual([p["type"] for p in result["content"]],
                         ["text", "image", "image"])
        info = json.loads(result["content"][0]["text"])
        self.assertTrue(info["depth_sent"])
        self.assertEqual(info["depth_reason"], "near_body_risk")
        self.assertTrue(info["forward_blocked_by_depth"])
        self.assertNotIn("front_obstacle_m", info)
        self.assertEqual(info["near_body_risk"]["identity"],
                         "unverified_obstacle")
        regions = [r["region"] for r in
                   info["near_body_risk"]["affected_regions"]]
        self.assertIn("right_edge", regions)
        with patch("havln_agent.codex_episode.read_current_observation",
                   return_value=observation):
            blocked = self.controller.step(["MOVE_FORWARD", "MOVE_FORWARD"])
        self.assertEqual(blocked["status"], "FORWARD_BLOCKED")
        self.assertEqual(blocked["executed"], [])
        self.assertEqual(blocked["skipped_count"], 2)
        self.assertEqual(blocked["remaining_actions"], 4)
        self.assertEqual(blocked["near_body_risk"]["affected_regions"][0]["region"],
                         "right_edge")
        self.assertEqual(self.env.actions, [])
        self.assertFalse((self.controller.directory / "steps.jsonl").exists())
        # The normal sensor no longer shows the side obstacle.
        moved = self.controller.step(["MOVE_FORWARD", "MOVE_FORWARD"])
        self.assertEqual(moved["status"], "RUNNING")
        self.assertEqual(moved["executed"],
                         ["MOVE_FORWARD", "MOVE_FORWARD"])
        self.assertEqual(self.env.actions, moved["executed"])
        self.assertNotIn("near_body_risk", moved)

    def test_partial_batch_rechecks_depth_from_previous_forward(self):
        clear_image = self.env.image

        def image():
            raw = clear_image()
            if self.env.actions:
                raw["depth"][3:9, 9:11] = 0.28
            return raw

        with patch.object(self.env, "image", side_effect=image):
            reply = self.controller.step(["MOVE_FORWARD"] * 3)
        self.assertEqual(reply["status"], "FORWARD_BLOCKED")
        self.assertEqual(reply["executed"], ["MOVE_FORWARD"])
        self.assertEqual(reply["skipped_count"], 2)
        self.assertEqual(reply["remaining_actions"], 3)
        self.assertEqual(self.env.actions, ["MOVE_FORWARD"])
        row = json.loads((self.controller.directory / "steps.jsonl").read_text())
        self.assertEqual(Path(row["post_file"]).stem, reply["last_observation_id"])

    def test_further_side_risk_allows_first_step_then_asks_to_observe(self):
        clear_image = self.env.image

        def image():
            raw = clear_image()
            raw["depth"][3:9, 9:11] = 0.37
            return raw

        with patch.object(self.env, "image", side_effect=image):
            reply = self.controller.step(["MOVE_FORWARD"] * 3)
        self.assertEqual(reply["status"], "REOBSERVE_FORWARD")
        self.assertEqual(reply["executed"], ["MOVE_FORWARD"])
        self.assertEqual(reply["skipped_count"], 2)
        self.assertIn("near_body_risk", reply)

    def test_turn_then_observe_then_forward_uses_new_heading_depth(self):
        clear_image = self.env.image

        def image():
            raw = clear_image()
            if "TURN_RIGHT" in self.env.actions:
                raw["depth"][3:9, 9:11] = 0.28
            return raw

        with patch.object(self.env, "image", side_effect=image):
            turn = self.controller.step(["TURN_RIGHT"])
            self.assertEqual(turn["executed"], ["TURN_RIGHT"])
            early = self.controller.step(["MOVE_FORWARD"])
            self.assertEqual(early["status"], "REOBSERVE_FORWARD")
            self.assertEqual(early["executed"], [])
            observed, _, depth = self.controller.observe()
            self.assertIsNotNone(depth)
            self.assertTrue(observed["forward_blocked_by_depth"])
            reply = self.controller.step(["MOVE_FORWARD"])
        self.assertEqual(reply["status"], "FORWARD_BLOCKED")
        self.assertEqual(reply["executed"], [])
        self.assertEqual(self.env.actions, ["TURN_RIGHT"])

    def test_mcp_rejects_mixed_view_and_forward_batches(self):
        server = MCPServer(self.controller)
        for actions in (["TURN_RIGHT", "MOVE_FORWARD"], ["MOVE_FORWARD", "LOOK_DOWN"]):
            with self.subTest(actions=actions):
                result = server._dispatch("tools/call", {
                    "name": "step", "arguments": {"actions": actions}})
                self.assertTrue(result["isError"])
                receipt = json.loads(result["content"][0]["text"])
                self.assertEqual(receipt["status"], "TOOL_ERROR")
                self.assertEqual(receipt["executed"], [])
                self.assertEqual(receipt["skipped_count"], len(actions))
                self.assertEqual(self.env.actions, [])

    def test_changed_rgb_during_side_slide_stops_batch_and_requires_observation(self):
        position = np.zeros(3)
        # A +90-degree yaw faces world -X. The longer Z displacement is lateral.
        rotation = SimpleNamespace(w=np.sqrt(0.5), x=0.0, y=np.sqrt(0.5), z=0.0)
        self.env.get_agent_state = lambda: SimpleNamespace(position=position, rotation=rotation)
        original = self.env.step

        def slide(action):
            position[:] += [-0.02, 0.0, 0.22]
            return original(action)

        with patch.object(self.env, "step", side_effect=slide):
            result = self.controller.step(["MOVE_FORWARD"] * 3)
            self.assertEqual(result["status"], "LOW_FORWARD_PROGRESS")
            self.assertEqual(result["executed"], ["MOVE_FORWARD"])
            self.assertEqual(result["skipped_count"], 2)
            self.assertEqual(result["forward_progress_m"], 0.02)
            retry = self.controller.step(["MOVE_FORWARD"])
            self.assertEqual(retry["status"], "REOBSERVE_FORWARD")
            self.assertEqual(retry["executed"], [])
        info, _, depth = self.controller.observe()
        self.assertIsNotNone(depth)
        self.assertEqual(info["motion_hint"]["recent_stall_directions_deg"], [0])
        row = json.loads((self.controller.directory / "steps.jsonl").read_text())
        self.assertEqual(row["motion_feedback"], "low_forward_progress")
        self.assertEqual(row["forward_progress_m"], 0.02)
        stationary_view = self.env.image()

        def advance(action):
            position[0] -= 0.25
            self.env.actions.append(action)
            return stationary_view, 0.0, False, {}

        with patch.object(self.env, "step", side_effect=advance):
            recovered = self.controller.step(["MOVE_FORWARD"])
        self.assertEqual(recovered["status"], "RUNNING")
        self.assertEqual(recovered["executed"], ["MOVE_FORWARD"])
        self.assertNotIn("motion_hint", recovered)

    def test_preflight_sensor_error_consumes_no_action_and_can_recover(self):
        with patch.object(self.env, "get_observations_at", return_value=None):
            reply = self.controller.step(["MOVE_FORWARD"])
        self.assertEqual(reply["status"], "TOOL_ERROR")
        self.assertEqual(reply["executed"], [])
        self.assertFalse(reply["done"])
        self.assertEqual(self.controller.step(["MOVE_FORWARD"])["executed"], ["MOVE_FORWARD"])

    def test_exact_unchanged_rgb_interrupts_only_remaining_batch_actions(self):
        before = self.env.image()
        def stationary_step(action):
            self.env.actions.append(action)
            return before, 0., action == "STOP", {}
        with patch.object(self.env, "step", side_effect=stationary_step):
            result = self.controller.step(["MOVE_FORWARD"] * 3)
        self.assertEqual(result["status"], "NO_VISUAL_PROGRESS")
        self.assertEqual(result["executed"], ["MOVE_FORWARD"])
        self.assertEqual(result["skipped_count"], 2)
        self.assertFalse(result["done"])
        self.assertEqual(self.env.actions, ["MOVE_FORWARD"])
        log = [json.loads(line) for line in
               (self.controller.directory / "steps.jsonl").read_text().splitlines()]
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["motion_feedback"], "rgb_unchanged")
        receipt = json.loads((self.controller.directory / "tools.jsonl")
                             .read_text().splitlines()[-1])
        self.assertEqual(receipt["result"]["skipped_count"], 2)
        self.assertEqual(receipt["result"]["status"], "NO_VISUAL_PROGRESS")

        # Do not create a movement authorization lock after a stall.
        # Turning is allowed even when the RGB is unchanged.
        with patch.object(self.env, "step", side_effect=stationary_step):
            turning = self.controller.step(["TURN_LEFT"])
        self.assertEqual(turning["status"], "RUNNING")
        self.assertEqual(turning["executed"], ["TURN_LEFT"])

    def test_look_down_then_up_restores_pitch_without_auto_forward(self):
        agent = CodexEpisode(FakeHabitat(), camera(), "1", "test", 12,
                             Path(self.tempdir.name) / "pitchcycle")
        down = agent.step(["LOOK_DOWN", "LOOK_DOWN"])
        self.assertEqual(down["status"], "RUNNING")
        self.assertEqual(agent.pitch_deg, -2 * camera().tilt_angle)
        self.assertNotIn("motion_hint", down)
        with patch("havln_agent.codex_episode.read_current_observation",
                   side_effect=lambda env, cam, instruction:
                   adapt_observation(env.image(), cam)):
            info, rgb, depth = agent.observe()
        self.assertTrue(info["depth_sent"])
        self.assertEqual(info["depth_reason"], "view_changed")
        self.assertIsNotNone(depth)
        self.assertEqual(agent.current.pitch_deg, -2 * camera().tilt_angle)
        self.assertEqual(info["pitch_deg"], -2 * camera().tilt_angle)
        up = agent.step(["LOOK_UP", "LOOK_UP"])
        self.assertEqual(up["status"], "RUNNING")
        self.assertEqual(agent.pitch_deg, 0)
        self.assertEqual(agent.actions,
                         ["LOOK_DOWN", "LOOK_DOWN", "LOOK_UP", "LOOK_UP"])
        self.assertNotIn("motion_hint", up)
        with patch("havln_agent.codex_episode.read_current_observation",
                   side_effect=lambda env, cam, instruction:
                   adapt_observation(env.image(), cam)):
            recovered, _, _ = agent.observe()
        self.assertEqual(recovered["pitch_deg"], 0.0)
        self.assertEqual(recovered["depth_reason"], "view_changed")

    def test_post_stall_inspection_depth_is_one_shot_even_without_obstacle(self):
        agent = CodexEpisode(FakeHabitat(), camera(), "1", "test", 32,
                             Path(self.tempdir.name) / "stall_depth")
        initial = agent.env.image()
        def blocked(action):
            agent.env.actions.append(action)
            return initial, 0., action == "STOP", {}
        with patch.object(agent.env, "step", side_effect=blocked), \
             patch.object(agent.env, "get_observations_at", return_value=initial):
            first = agent.step(["MOVE_FORWARD"] * 3)
            self.assertEqual(first["status"], "NO_VISUAL_PROGRESS")
            self.assertEqual(first["executed"], ["MOVE_FORWARD"])
            # A second stall before inspection must not queue extra images.
            second = agent.step(["MOVE_FORWARD"])
            self.assertEqual(second["status"], "NO_VISUAL_PROGRESS")
        with patch("havln_agent.codex_episode.read_current_observation",
                   side_effect=lambda env, cam, instruction:
                   adapt_observation(initial, cam)):
            first_observation = MCPServer(agent)._dispatch("tools/call", {
                "name": "observe", "arguments": {}})
            second_observation = MCPServer(agent)._dispatch("tools/call", {
                "name": "observe", "arguments": {}})
        self.assertEqual([p["type"] for p in first_observation["content"]],
                         ["text", "image", "image"])
        info = json.loads(first_observation["content"][0]["text"])
        self.assertTrue(info["depth_sent"])
        self.assertEqual(info["depth_reason"], "post_stall_inspection")
        self.assertNotIn("front_obstacle_m", info)
        next_info = json.loads(second_observation["content"][0]["text"])
        self.assertFalse(next_info["depth_sent"])
        self.assertEqual([p["type"] for p in second_observation["content"]],
                         ["text", "image"])
        # Another stall within the same locally blocked period is still
        # allowed but cannot trigger an endless series of extra Depth frames.
        with patch.object(agent.env, "step", side_effect=blocked), \
             patch.object(agent.env, "get_observations_at", return_value=initial):
            self.assertEqual(agent.step(["MOVE_FORWARD"])["status"],
                             "NO_VISUAL_PROGRESS")
        with patch("havln_agent.codex_episode.read_current_observation",
                   side_effect=lambda env, cam, instruction:
                   adapt_observation(initial, cam)):
            later_info, _, later_depth = agent.observe()
        self.assertFalse(later_info["depth_sent"])
        self.assertIsNone(later_depth)
        # New visually changing forward enables a new one-shot inspection.
        self.assertEqual(agent.step(["MOVE_FORWARD"])["status"], "RUNNING")
        with patch.object(agent.env, "step", side_effect=lambda action: (
                agent.env.image(), 0., action == "STOP", {})):
            # This unmodified image is now unchanged versus the current frame.
            self.assertEqual(agent.step(["MOVE_FORWARD"])["status"],
                             "NO_VISUAL_PROGRESS")
        with patch("havln_agent.codex_episode.read_current_observation",
                   side_effect=lambda env, cam, instruction:
                   adapt_observation(env.image(), cam)):
            refreshed, _, refreshed_depth = agent.observe()
        self.assertTrue(refreshed["depth_sent"])
        self.assertEqual(refreshed["depth_reason"], "post_stall_inspection")
        self.assertIsNotNone(refreshed_depth)

    def test_post_stall_depth_not_lost_when_observing_after_a_tilt(self):
        agent = CodexEpisode(FakeHabitat(), camera(), "1", "test", 32,
                             Path(self.tempdir.name) / "tilt_depth")
        raw = agent.env.image()
        def blocked(action):
            agent.env.actions.append(action)
            return raw, 0., action == "STOP", {}
        with patch.object(agent.env, "step", side_effect=blocked):
            agent.step(["MOVE_FORWARD"])
        agent.step(["LOOK_DOWN"])
        with patch("havln_agent.codex_episode.read_current_observation",
                   side_effect=lambda env, cam, instruction:
                   adapt_observation(env.image(), cam)):
            info, _, depth = agent.observe()
        self.assertEqual(info["pitch_deg"], -camera().tilt_angle)
        self.assertTrue(info["depth_sent"])
        self.assertEqual(info["depth_reason"], "post_stall_inspection")
        self.assertIsNotNone(depth)

    def test_stall_with_reserved_stop_does_not_miscount_skipped_actions(self):
        before = self.env.image()
        def stationary_step(action):
            self.env.actions.append(action)
            return before, 0., action == "STOP", {}
        with patch.object(self.env, "step", side_effect=stationary_step), \
             patch.object(self.env, "get_observations_at", return_value=before):
            self.controller.step(["TURN_LEFT", "TURN_RIGHT"])
            self.controller.observe()
            # One MOVE_FORWARD uses the last non-STOP action; reserved STOP
            # is still appended by the pre-existing budget guard.
            result = self.controller.step(["MOVE_FORWARD"])
        self.assertEqual(result["status"], "DONE")
        self.assertEqual(result["executed"], ["MOVE_FORWARD", "STOP"])
        self.assertEqual(result["skipped_count"], 0)
        self.assertEqual(self.env.actions[-2:], ["MOVE_FORWARD", "STOP"])

    def test_turn_history_across_batches_exposes_full_circle_without_veto(self):
        agent = CodexEpisode(FakeHabitat(), camera(), "1", "test", 128,
                             Path(self.tempdir.name) / "circle")
        for count in (3, 9, 6):
            reply = agent.step(["TURN_LEFT"] * count)
            self.assertEqual(reply["status"], "RUNNING")
        self.assertEqual(reply["motion_hint"]["turn_total_deg"], 270)
        self.assertEqual(reply["motion_hint"]["turn_net_deg"], -270)
        reply = agent.step(["TURN_LEFT"] * 6)
        self.assertEqual(reply["motion_hint"]["turn_total_deg"], 360)
        self.assertEqual(reply["motion_hint"]["turn_net_deg"], -360)
        self.assertEqual(agent.actions, ["TURN_LEFT"] * 24)
        self.assertFalse(agent.done)

    def test_opposite_turns_count_even_when_net_zero(self):
        agent = CodexEpisode(FakeHabitat(), camera(), "1", "test", 128,
                             Path(self.tempdir.name) / "cancel")
        first = agent.step(["TURN_LEFT"] * 9)
        self.assertNotIn("motion_hint", first)
        second = agent.step(["TURN_RIGHT"] * 9)
        self.assertEqual(second["motion_hint"]["turn_total_deg"], 270)
        self.assertEqual(second["motion_hint"]["turn_net_deg"], 0)
        self.assertEqual(second["executed"], ["TURN_RIGHT"] * 9)

    def test_long_turn_batch_reports_turns_before_a_separate_forward(self):
        agent = CodexEpisode(FakeHabitat(), camera(), "1", "test", 128,
                             Path(self.tempdir.name) / "batchturn")
        reply = agent.step(["TURN_LEFT"] * 15)
        self.assertEqual(reply["status"], "RUNNING")
        self.assertEqual(reply["motion_hint"]["batch_turn_total_deg"], 225)
        self.assertEqual(reply["motion_hint"]["batch_turn_net_deg"], -225)
        agent.observe()
        self.assertEqual(agent.step(["MOVE_FORWARD"])["status"], "RUNNING")
        # The forward has a different RGB, so current-place turn counts reset.
        again = agent.step(["TURN_LEFT"])
        self.assertNotIn("motion_hint", again)

    def test_possible_reversal_requires_fresh_views_and_two_single_forwards(self):
        agent = CodexEpisode(FakeHabitat(), camera(), "1", "test", 64,
                             Path(self.tempdir.name) / "route_recheck")
        heading = np.array([1.0, 0.0])
        poses = [(np.zeros(2), heading), (heading * 0.25, heading)]
        with patch("havln_agent.codex_episode.read_planar_pose", side_effect=poses * 2):
            self.assertEqual(agent.step(["MOVE_FORWARD"] * 2)["status"], "RUNNING")

        self.assertEqual(agent.step(["TURN_LEFT"] * 9)["status"], "RUNNING")
        reverse = agent.step(["TURN_LEFT"] * 3)
        self.assertEqual(reverse["status"], "ROUTE_RECHECK")
        self.assertEqual(reverse["executed"], ["TURN_LEFT"] * 3)
        self.assertEqual(reverse["skipped_count"], 0)
        self.assertEqual(reverse["route_recheck_steps"], 2)

        with patch("havln_agent.codex_episode.read_planar_pose", side_effect=poses * 4):
            for remaining in (1, 0):
                wait = agent.step(["MOVE_FORWARD"] * 3)
                self.assertEqual(wait["status"], "ROUTE_RECHECK")
                self.assertEqual(wait["executed"], [])
                agent.observe()
                checked = agent.step(["MOVE_FORWARD"] * 3)
                self.assertEqual(checked["executed"], ["MOVE_FORWARD"])
                self.assertEqual(checked["skipped_count"], 2)
                self.assertEqual(checked.get("route_recheck_steps", 0), remaining)
            resumed = agent.step(["MOVE_FORWARD"] * 2)
        self.assertEqual(resumed["status"], "RUNNING")
        self.assertEqual(resumed["executed"], ["MOVE_FORWARD"] * 2)

    def test_low_depth_view_avoids_repeated_cue_for_one_straight_step(self):
        env = FakeHabitat()
        agent = CodexEpisode(env, camera(), "1", "test", 32,
                             Path(self.tempdir.name) / "low_view")
        close = env.image()
        close["depth"].fill(0.2)
        with patch.object(env, "get_observations_at", return_value=close):
            info, _, _ = agent.observe()
        self.assertTrue(info["look_down_for_clearance"])

        agent.step(["LOOK_DOWN"])
        low, _, depth = agent.observe()
        self.assertLess(low["pitch_deg"], 0)
        self.assertIsNotNone(depth)
        agent.step(["LOOK_UP"])
        early = agent.step(["MOVE_FORWARD"])
        self.assertEqual(early["status"], "REOBSERVE_FORWARD")
        self.assertEqual(early["executed"], [])
        level, _, depth = agent.observe()
        self.assertEqual(level["pitch_deg"], 0)
        self.assertIsNotNone(depth)
        self.assertEqual(agent.step(["MOVE_FORWARD"])["executed"], ["MOVE_FORWARD"])
        with patch.object(env, "get_observations_at", return_value=close):
            info, _, _ = agent.observe()
        self.assertNotIn("look_down_for_clearance", info)
        agent.step(["MOVE_FORWARD"])
        with patch.object(env, "get_observations_at", return_value=close):
            info, _, _ = agent.observe()
        self.assertTrue(info["look_down_for_clearance"])

    def test_stall_directions_are_relative_and_reset_after_visual_change(self):
        agent = CodexEpisode(FakeHabitat(), camera(), "1", "test", 128,
                             Path(self.tempdir.name) / "stallhistory")
        original = agent.env.image()
        def blocked(action):
            agent.env.actions.append(action)
            return original, 0., action == "STOP", {}
        with patch.object(agent.env, "step", side_effect=blocked), \
             patch.object(agent.env, "get_observations_at", return_value=original):
            first = agent.step(["MOVE_FORWARD"] * 3)
            self.assertEqual(first["status"], "NO_VISUAL_PROGRESS")
            self.assertEqual(first["motion_hint"]["recent_stall_directions_deg"], [0])
            turned = agent.step(["TURN_RIGHT", "TURN_RIGHT"])
            self.assertEqual(turned["motion_hint"]["recent_stall_directions_deg"], [-30])
            agent.observe()
            second = agent.step(["MOVE_FORWARD"] * 2)
            self.assertEqual(second["status"], "NO_VISUAL_PROGRESS")
            self.assertEqual(second["motion_hint"]["recent_stall_directions_deg"],
                             [-30, 0])
        # A changing RGB clears short-term local obstacle attempts.
        cleared = agent.step(["MOVE_FORWARD"])
        self.assertEqual(cleared["status"], "RUNNING")
        self.assertNotIn("motion_hint", cleared)
        self.assertEqual(len(agent.actions), 5)

    def test_tiny_normal_actions_do_not_add_motion_hint_or_change_export(self):
        first = self.controller.step(["TURN_LEFT"])
        self.assertNotIn("motion_hint", first)
        self.controller.observe()
        last = self.controller.step(["MOVE_FORWARD"] * 2)
        self.assertEqual(last["executed"], ["MOVE_FORWARD", "MOVE_FORWARD", "STOP"])
        self.assertNotIn("motion_hint", last)
        self.assertEqual(self.env.actions,
                         ["TURN_LEFT", "MOVE_FORWARD", "MOVE_FORWARD", "STOP"])

    def test_ordered_batch_and_reserved_stop_produce_legal_export(self):
        self.controller.step(["TURN_LEFT"])
        self.controller.observe()
        first = self.controller.step(["MOVE_FORWARD"])
        self.assertEqual(first["status"], "RUNNING")
        self.assertEqual(first["executed"], ["MOVE_FORWARD"])
        second = self.controller.step(["MOVE_FORWARD"])
        self.assertEqual(second["executed"], ["MOVE_FORWARD", "STOP"])
        self.assertTrue(second["done"])
        self.assertEqual(self.env.actions,
                         ["TURN_LEFT", "MOVE_FORWARD", "MOVE_FORWARD", "STOP"])
        result = json.loads((self.controller.directory / "result.json").read_text())
        self.assertEqual(result["row"]["actions"], self.env.actions)
        self.assertEqual(len((self.controller.directory / "steps.jsonl")
                             .read_text().splitlines()), 4)
        self.assertTrue(json.loads((self.controller.directory /
                                    "summary.json").read_text())["completed"])
        diagnostic = json.loads((Path(self.tempdir.name) /
                                 "test.diagnostic.json").read_text())
        self.assertEqual(diagnostic["format_version"], 1)
        self.assertEqual(diagnostic["split"], "test")
        self.assertEqual(diagnostic["episodes"][0]["actions"], self.env.actions)

    def test_explicit_stop_and_reject_invalid_batch(self):
        failed = self.controller.step(["STOP", "TURN_LEFT"])
        self.assertEqual(failed["status"], "TOOL_ERROR")
        self.assertFalse(self.env.actions)
        stopped = self.controller.step(["STOP"])
        self.assertEqual(stopped["status"], "DONE")
        self.assertEqual(self.env.actions, ["STOP"])
        self.assertEqual(self.controller.step(["MOVE_FORWARD"])["status"],
                         "ALREADY_DONE")

    def test_closed_without_stop_is_not_a_valid_submission(self):
        self.controller.step(["MOVE_FORWARD"])
        self.controller.finish()
        self.assertFalse((self.controller.directory / "result.json").exists())
        summary = json.loads((self.controller.directory / "summary.json").read_text())
        self.assertFalse(summary["completed"])
        self.assertEqual(summary["stop_reason"], "session_ended_without_stop")

    def test_mcp_stdio_protocol_includes_image_and_tool_receipt(self):
        payloads = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "observe", "arguments": {"include_depth": True}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
             "params": {"name": "step", "arguments": {"actions": ["STOP"]}}},
        ]
        wire_in = io.StringIO("\n".join(json.dumps(row) for row in payloads) + "\n")
        wire_out = io.StringIO()
        with patch("havln_agent.codex_episode.read_current_observation",
                   side_effect=lambda env, cam, instruction:
                   adapt_observation(env.image(), cam)):
            MCPServer(self.controller, reader=wire_in, writer=wire_out).serve()
        replies = [json.loads(line) for line in wire_out.getvalue().splitlines()]
        self.assertEqual([row["id"] for row in replies], [1, 2, 3, 4])
        observe = replies[1]["result"]["tools"][0]
        self.assertEqual(observe["name"], "observe")
        self.assertEqual(observe["inputSchema"]["properties"]["include_depth"]["type"], "boolean")
        content = replies[2]["result"]["content"]
        self.assertEqual([p["type"] for p in content], ["text", "image", "image"])
        self.assertEqual([p["mimeType"] for p in content[1:]], ["image/png", "image/png"])
        self.assertEqual(json.loads(content[0]["text"])["depth_reason"], "requested")
        self.assertEqual(replies[3]["result"]["isError"], False)
        self.assertEqual(self.env.actions, ["STOP"])

    def test_codex_cli_events_are_viewable_and_archived_without_faking_order(self):
        self.controller.step(["STOP"])
        codex_dir = self.controller.directory / "codex"
        codex_dir.mkdir()
        events = [{"type": "thread.started", "thread_id": "fresh-session"},
                  {"type": "item.started", "item": {"type": "mcp_tool_call",
                   "name": "havln.observe", "id": "call-1"}},
                  {"type": "item.completed", "item": {"type": "agent_message",
                   "text": "I see the stairs.", "id": "out-1"}}]
        with (codex_dir / "codex_events.jsonl").open("w") as handle:
            for row in events:
                handle.write(json.dumps(row) + "\n")
        (codex_dir / "prompt.txt").write_text("One episode, new session.")
        (codex_dir / "run_status.json").write_text(json.dumps({
            "codex_exit_code": 0, "export_valid": True}))
        source = ReplaySource(self.tempdir.name)
        replay = EpisodeReplay(source, "test-1")
        overview = replay.overview()
        self.assertTrue(overview["codex_independent_timeline"])
        self.assertEqual(overview["counts"]["codex"], 3)
        self.assertEqual(overview["codex_prompt"], "One episode, new session.")
        self.assertEqual(overview["codex_run_status"]["codex_exit_code"], 0)
        self.assertEqual(overview["events"][-1]["kind"], "codex")
        self.assertEqual(replay.detail(len(overview["events"])-1)["record"], events[-1])

        archive = Path(self.tempdir.name) / "replay.zip"
        pack(self.tempdir.name, archive, frames="all")
        zipped = ReplaySource(archive)
        zipped_view = EpisodeReplay(zipped, "test-1")
        self.assertEqual(zipped_view.overview()["counts"]["codex"], 3)
        self.assertEqual(zipped_view.frame("obs_000000")[:8],
                         bytes([137, 80, 78, 71, 13, 10, 26, 10]))
        zipped.zip.close()

    def test_verified_codex_observe_has_true_image_and_step_has_exact_before_after(self):
        with patch("havln_agent.codex_episode.read_current_observation",
                   side_effect=lambda env, cam, instruction:
                   adapt_observation(env.image(), cam)):
            received, _, _ = self.controller.observe(include_depth=True)
        self.controller.step(["TURN_LEFT", "STOP"])
        folder = self.controller.directory / "codex"
        folder.mkdir()
        events = [
            {"type": "thread.started"},
            {"type": "item.started", "item": {
                "type": "mcp_tool_call", "id": "item_observe",
                "server": "havln", "tool": "observe", "arguments": {"include_depth": True}}},
            {"type": "item.completed", "item": {
                "type": "mcp_tool_call", "id": "item_observe",
                "server": "havln", "tool": "observe", "arguments": {"include_depth": True},
                "result": {"content": [
                    {"type": "text", "text": "observation details"},
                    {"type": "image", "mimeType": "image/png",
                     "data": "raw-encoded-image-data"},
                    {"type": "image", "mimeType": "image/png",
                     "data": "raw-encoded-depth-data"}]}}},
            {"type": "item.started", "item": {
                "type": "mcp_tool_call", "id": "item_step",
                "server": "havln", "tool": "step",
                "arguments": {"actions": ["TURN_LEFT", "STOP"]}}},
            {"type": "item.completed", "item": {
                "type": "mcp_tool_call", "id": "item_step",
                "server": "havln", "tool": "step",
                "arguments": {"actions": ["TURN_LEFT", "STOP"]}}},
        ]
        raw_path = folder / "codex_events.jsonl"
        raw_path.write_text("".join(json.dumps(row) + "\n" for row in events))
        view = EpisodeReplay(ReplaySource(self.tempdir.name), "test-1")
        items = [row for row in view.overview()["events"]
                 if row["kind"] == "codex"]
        self.assertNotIn("observation_id", items[1])
        self.assertEqual(items[2]["observation_id"],
                         received["observation_id"])
        self.assertEqual(items[2]["linked_tool"], "observe")
        self.assertTrue(items[2]["depth_sent"])
        self.assertEqual(items[3]["observation_id"],
                         received["observation_id"])
        self.assertEqual(items[4]["after_id"], "obs_000003")
        self.assertEqual(view.frame(received["observation_id"])[:8],
                         bytes([137, 80, 78, 71, 13, 10, 26, 10]))
        result = view.detail(view.habitat_event_count + 2)["record"]
        image = result["item"]["result"]["content"][1]
        self.assertTrue(image["data_omitted_from_viewer"])
        self.assertNotIn("data", image)
        self.assertIn("raw-encoded-image-data", raw_path.read_text())

    def test_codex_image_link_fails_closed_on_mismatched_action_arguments(self):
        self.controller.step(["STOP"])
        folder = self.controller.directory / "codex"
        folder.mkdir()
        (folder / "codex_events.jsonl").write_text(json.dumps({
            "type": "item.started", "item": {
                "type": "mcp_tool_call", "id": "wrong",
                "server": "havln", "tool": "step",
                "arguments": {"actions": ["MOVE_FORWARD"]}}}) + "\n")
        view = EpisodeReplay(ReplaySource(self.tempdir.name), "test-1")
        event = view.overview()["events"][-1]
        self.assertNotIn("observation_id", event)
        self.assertNotIn("after_id", event)
        self.assertNotIn("linked_tool", event)

    def test_codex_event_summary_and_read_only_preflight(self):
        events = Path(self.tempdir.name) / "events.jsonl"
        events.write_text('{"type":"thread.started"}\n'
                          '{"type":"turn.completed"}\n'
                          '{"type":"turn.completed"}\n'
                          'not-json\n')
        summary = summarize_codex_events(events)
        self.assertEqual(summary["events_by_type"]["turn.completed"], 2)
        self.assertEqual(summary["malformed_events"], 1)
        args = SimpleNamespace(codex_bin="codex", docker_bin="docker",
                               container="havln-cma", episode_id=1)
        with patch("scripts.run_codex_episode.subprocess.run",
                   return_value=SimpleNamespace(returncode=0,
                      stdout="ready", stderr="")) as run:
            self.assertEqual(preflight(args, "test"), 0)
            self.assertEqual(run.call_count, 6)
            self.assertTrue(any("codex_episode" in str(c.args)
                                for c in run.call_args_list))

    def test_mcp_import_probe_uses_container_python_and_reports_traceback(self):
        args = SimpleNamespace(docker_bin="docker", container="havln-cma")
        with patch("scripts.run_codex_episode.subprocess.run",
                   return_value=SimpleNamespace(
                       returncode=0,
                       stdout="MCP controller import OK, Python 3.8.18\\n",
                       stderr="")) as run:
            self.assertIn("3.8.18", require_mcp_runtime(args))
        command = run.call_args.args[0]
        self.assertEqual(command[:6],
                         ["docker", "exec", "-w", "/toolkit",
                          "havln-cma", "python"])
        self.assertEqual(command[6], "-c")
        self.assertIn("from havln_agent.codex_episode import CodexEpisode",
                      command[-1])
        with patch("scripts.run_codex_episode.subprocess.run",
                   return_value=SimpleNamespace(
                       returncode=1, stdout="",
                       stderr="Traceback: incompatible dependency\\n")):
            with self.assertRaisesRegex(RuntimeError,
                                        "incompatible dependency"):
                require_mcp_runtime(args)

    def test_post_action_capture_fault_preserves_real_action_and_fails_closed(self):
        env = FakeHabitat()
        agent = CodexEpisode(env, camera(), "1", "test", 32,
                             Path(self.tempdir.name) / "capture_error")
        save = agent._save

        def fail_after_action(observation):
            if env.actions:
                raise OSError("simulated disk error")
            return save(observation)

        with patch.object(agent, "_save", side_effect=fail_after_action):
            reply = agent.step(["MOVE_FORWARD", "MOVE_FORWARD"])
        self.assertEqual(reply["status"], "TOOL_ERROR")
        self.assertEqual(reply["executed"], ["MOVE_FORWARD"])
        self.assertEqual(reply["skipped_count"], 1)
        self.assertEqual(reply["executed_actions"], 1)
        self.assertTrue(reply["done"])
        self.assertEqual(reply["reason"], "post_action_processing_exception")
        self.assertEqual(agent.actions, env.actions)
        row = json.loads((agent.directory / "steps.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["action"], "MOVE_FORWARD")
        self.assertIsNone(row["post_file"])
        self.assertIn("simulated disk error", row["post_action_error"])
        self.assertFalse((agent.directory / "result.json").exists())
        self.assertTrue((agent.directory / "failure.json").exists())

    def test_post_stop_capture_failure_must_not_export_false_success(self):
        env = FakeHabitat()
        agent = CodexEpisode(env, camera(), "1", "test", 32,
                             Path(self.tempdir.name) / "stop_capture_error")
        with patch.object(agent, "_save",
                          side_effect=OSError("simulated STOP capture error")):
            reply = agent.step(["STOP"])
        self.assertEqual(reply["status"], "TOOL_ERROR")
        self.assertEqual(reply["executed"], ["STOP"])
        self.assertEqual(agent.actions, env.actions)
        self.assertTrue(reply["done"])
        self.assertFalse((agent.directory / "result.json").exists())
        summary = json.loads((agent.directory / "summary.json").read_text())
        self.assertFalse(summary["completed"])
        self.assertEqual(summary["stop_reason"],
                         "post_action_processing_exception")

    def test_codex_wall_time_limit_terminates_stalled_process(self):
        output = Path(self.tempdir.name) / "stdout.log"
        errors = Path(self.tempdir.name) / "stderr.log"
        with output.open("w") as out, errors.open("w") as err:
            status = run_codex_cli(
                [sys.executable, "-c", "import time;time.sleep(20)"],
                self.tempdir.name, out, err, 0.2)
        self.assertEqual(status, 124)
        self.assertIn("exceeded wall-time limit", errors.read_text())

    def test_ctrl_c_also_terminates_codex_and_preserves_cleanup_path(self):
        process = SimpleNamespace(pid=12345,
                                  wait=Mock(side_effect=[KeyboardInterrupt(), 0]))
        manager = MagicMock()
        manager.__enter__.return_value = process
        stderr = io.StringIO()
        with patch("scripts.run_codex_episode.subprocess.Popen",
                   return_value=manager), \
             patch("scripts.run_codex_episode.os.killpg") as kill:
            code = run_codex_cli(["codex", "exec"], self.tempdir.name,
                                 io.StringIO(), stderr, 120)
        self.assertEqual(code, 130)
        self.assertIn("interrupted by user", stderr.getvalue())
        kill.assert_called_once()

    def test_habitat_execution_exception_is_terminal_and_not_exported(self):
        with patch.object(self.env, "step", side_effect=RuntimeError("sim failure")):
            result = self.controller.step(["MOVE_FORWARD"])
        self.assertEqual(result["status"], "TOOL_ERROR")
        self.assertTrue(result["done"])
        self.assertEqual(result["reason"], "habitat_action_exception")
        self.assertFalse((self.controller.directory / "result.json").exists())
        self.assertTrue((self.controller.directory / "failure.json").exists())

    def test_single_output_auto_archive_commands(self):
        args = SimpleNamespace(docker_bin="docker", container="havln-cma",
                               output_dir="/workspace/agent-runs/codex-one",
                               archive_frames="critical")
        staged = Path(self.tempdir.name) / "codex"
        staged.mkdir()
        (staged / "run_status.json").write_text('{"export_valid":false}')
        trace = args.output_dir + "/traces/test-1"
        with patch("scripts.run_codex_episode.subprocess.run") as run:
            collect_codex_audit(args, staged, True, trace)
            calls = [c.args[0] for c in run.call_args_list]
            self.assertEqual(calls[0][-2:], ["-p", trace])
            self.assertEqual(calls[1][-1], "havln-cma:" + trace)
            self.assertEqual(calls[2][-1],
                             "havln-cma:" + args.output_dir + "/run_status.json")
            run.reset_mock()
            collect_codex_audit(args, staged, False, trace)
            calls = [c.args[0] for c in run.call_args_list]
            self.assertEqual(calls[1][-1], "havln-cma:" + args.output_dir)
        with patch("scripts.run_codex_episode.subprocess.run",
                   return_value=SimpleNamespace(
                       stdout="Archived\nCODEX_ARCHIVE_PATH=/workspace/agent-runs/diagnostic-archives/codex-one-20261118T123456Z-diagnostic.zip\n")
                   ) as run:
            archive = archive_codex_run(args)
            self.assertEqual(
                archive,
                "/workspace/agent-runs/diagnostic-archives/codex-one-20261118T123456Z-diagnostic.zip")
            cmd = run.call_args.args[0]
            self.assertIn("from havln_agent.export import auto_archive",
                          cmd[cmd.index("-c") + 1])
            self.assertEqual(cmd[-2:], [args.output_dir, "critical"])
            args.archive_frames = "auto"
            archive_codex_run(args)
            self.assertEqual(run.call_args.args[0][-1], "all")

    def test_refuse_existing_container_output_before_inference(self):
        args = SimpleNamespace(docker_bin="docker", container="havln-cma",
                               output_dir="/workspace/agent-runs/already-there")
        with patch("scripts.run_codex_episode.subprocess.run",
                   return_value=SimpleNamespace(returncode=0, stderr="")):
            with self.assertRaisesRegex(ValueError, "already exists"):
                require_fresh_container_output(args)
        with patch("scripts.run_codex_episode.subprocess.run",
                   return_value=SimpleNamespace(returncode=1, stderr="")):
            require_fresh_container_output(args)

    def test_archive_startup_failure_with_no_habitat_trace(self):
        root = Path(self.tempdir.name) / "failed-run"
        codex = root / "codex"
        codex.mkdir(parents=True)
        (codex / "codex_events.jsonl").write_text(
            '{"type":"turn.failed","message":"MCP could not start"}\n')
        (codex / "codex_stderr.log").write_text("Connection failed")
        (root / "run_status.json").write_text('{"success":false}\n')
        archive = root / "diagnostic.zip"
        pack(root, archive, frames="critical", allow_empty=True)
        with ZipFile(archive) as zipped:
            names = set(zipped.namelist())
            self.assertIn("failed-run/codex/codex_events.jsonl", names)
            self.assertIn("failed-run/codex/codex_stderr.log", names)
            self.assertIn("failed-run/run_status.json", names)
            self.assertIn("failed-run/diagnostic_manifest.json", names)

    def test_critical_archive_includes_every_model_observed_frame(self):
        root = Path(self.tempdir.name) / "observed-run"
        agent = CodexEpisode(self.env, camera(), "1", "test", 100, root)
        agent.step(["TURN_LEFT"] * 10)
        with patch("havln_agent.codex_episode.read_current_observation",
                   side_effect=lambda env, cam, instruction:
                   adapt_observation(env.image(), cam)):
            info, _, _ = agent.observe()
        agent.step(["TURN_LEFT"] * 20)
        agent.finish()
        archive = root / "diagnostic.zip"
        pack(root, archive, frames="critical")
        with ZipFile(archive) as zipped:
            self.assertIn("observed-run/traces/test-1/observations/" +
                          info["observation_id"] + ".npz", zipped.namelist())

    def test_episode_instruction_is_loaded_from_dataset_once_before_codex(self):
        args = SimpleNamespace(docker_bin="docker", container="havln-cma",
                               data_root="/data/havln2", episode_id=103)
        with patch("scripts.run_codex_episode.subprocess.run",
                   return_value=SimpleNamespace(stdout='"Take the stairs left."')) as run:
            instruction = load_episode_instruction(args, "test")
        self.assertEqual(instruction, "Take the stairs left.")
        command = run.call_args.args[0]
        self.assertEqual(command[-3:], ["/data/havln2", "test", "103"])
        self.assertIn("gzip", command[7])
        self.assertIn("instruction_text", command[7])
        with patch("scripts.run_codex_episode.subprocess.run",
                   return_value=SimpleNamespace(stdout='""')):
            with self.assertRaisesRegex(ValueError, "empty"):
                load_episode_instruction(args, "test")

    def test_fresh_codex_command_uses_scoped_mcp_and_no_resume(self):
        args = SimpleNamespace(split=None, phase="phase2",
                               docker_bin="docker", container="havln-cma",
                               episode_id=1, max_steps=128, gpu_id=0, seed=0,
                               output_dir="/workspace/agent-runs/test-codex-1",
                               codex_bin="codex", reasoning_effort="medium",
                               model="gpt-6-astra")
        system_file = Path("/tmp/havln-codex-system-prompt.txt")
        cmd = build_command(args, Path("/tmp/last.txt"), system_file)
        self.assertIn('model_instructions_file="' + str(system_file) + '"', cmd)
        self.assertEqual(cmd[-1], START_PROMPT)
        self.assertNotIn("Go down the stairs.", cmd[-1])
        self.assertIn("sole navigation agent", PROMPT)
        self.assertIn("People can move between views", PROMPT)
        self.assertIn("Keep a safe gap", PROMPT)
        self.assertIn("keep moving when clear", PROMPT)
        self.assertIn("Follow landmarks in order", PROMPT)
        self.assertIn("exact stopping place", PROMPT)
        self.assertIn("magenta=invalid", PROMPT)
        self.assertIn("next unverified landmark", PROMPT)
        self.assertIn("motion_hint", PROMPT)
        self.assertIn("favor longer MOVE_FORWARD batches", PROMPT)
        self.assertIn("take one MOVE_FORWARD, then observe again", PROMPT)
        self.assertIn("parallel to the furniture edge", PROMPT)
        self.assertIn("Do not cut diagonally across a doorway", PROMPT)
        self.assertIn("until the whole body has cleared the corner or door frame", PROMPT)
        self.assertIn('request {"include_depth": true} if the needed Depth is missing', PROMPT)
        self.assertIn("Turn in place, observe at the new heading", PROMPT)
        self.assertIn("Check left and right clearance, the nearest corner, and the floor ahead", PROMPT)
        self.assertIn("LOOK_DOWN once and observe the low RGB-D view", PROMPT)
        self.assertIn("Look down again only if it can reveal missing evidence", PROMPT)
        self.assertIn("Missing or invalid Depth leaves the affected clearance unknown", PROMPT)
        self.assertIn("matching LOOK_UP actions", PROMPT)
        self.assertIn("call observe and wait for its result", PROMPT)
        self.assertIn("short turn batches", PROMPT)
        self.assertIn("If ROUTE_RECHECK appears", PROMPT)
        self.assertIn("the last required route segment is complete", PROMPT)
        self.assertIn("If forward_blocked_by_depth appears", PROMPT)
        self.assertIn("Do not repeatedly retry a stalled direction", PROMPT)
        self.assertIn("left/right at the place specified", PROMPT)
        self.assertIn("exec", cmd)
        self.assertNotIn("resume", cmd)
        self.assertIn("--json", cmd)
        self.assertIn("--ephemeral", cmd)
        self.assertIn("--ignore-user-config", cmd)
        self.assertIn('mcp_servers.havln.enabled_tools=["observe","step"]',
                      cmd)
        self.assertIn("mcp_servers.havln.supports_parallel_tool_calls=false", cmd)
        args_override = next(cmd[i + 1] for i, word in enumerate(cmd)
                             if word == "-c" and cmd[i + 1].startswith(
                                 "mcp_servers.havln.args="))
        self.assertIn("havln_mcp_server.py", args_override)
        self.assertIn("--episode-id", args_override)


if __name__ == "__main__":
    unittest.main()
