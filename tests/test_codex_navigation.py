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
from unittest.mock import patch

import numpy as np

from havln_agent.codex_episode import CodexEpisode
from havln_agent.codex_mcp import MCPServer
from havln_agent.replay_server import ReplaySource, EpisodeReplay
from scripts.pack_agent_diagnostics import pack
from scripts.run_codex_episode import preflight, summarize_codex_events, run_codex_cli
from havln_agent.perception import Camera, adapt_observation
from scripts.run_codex_episode import build_command


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

    def test_observe_returns_real_png_and_instruction(self):
        with patch("havln_agent.codex_episode.read_current_observation",
                   side_effect=lambda env, cam, instruction:
                   adapt_observation(env.image(), cam)):
            info, encoded = self.controller.observe()
        self.assertEqual(info["instruction"], "Go down the stairs.")
        self.assertEqual(base64.b64decode(encoded)[:8],
                         bytes([137, 80, 78, 71, 13, 10, 26, 10]))
        self.assertEqual(self.env.actions, [])
        self.assertTrue((self.controller.directory /
                         ("observations/" + info["observation_id"] + ".npz")).exists())

    def test_ordered_batch_and_reserved_stop_produce_legal_export(self):
        first = self.controller.step(["TURN_LEFT", "MOVE_FORWARD"])
        self.assertEqual(first["status"], "RUNNING")
        self.assertEqual(first["executed"], ["TURN_LEFT", "MOVE_FORWARD"])
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
             "params": {"name": "observe", "arguments": {}}},
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
        self.assertEqual(replies[1]["result"]["tools"][0]["name"], "observe")
        self.assertEqual(replies[2]["result"]["content"][1]["mimeType"], "image/png")
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
            self.assertEqual(run.call_count, 5)

    def test_codex_wall_time_limit_terminates_stalled_process(self):
        output = Path(self.tempdir.name) / "stdout.log"
        errors = Path(self.tempdir.name) / "stderr.log"
        with output.open("w") as out, errors.open("w") as err:
            status = run_codex_cli(
                [sys.executable, "-c", "import time;time.sleep(20)"],
                self.tempdir.name, out, err, 0.2)
        self.assertEqual(status, 124)
        self.assertIn("exceeded wall-time limit", errors.read_text())

    def test_habitat_execution_exception_is_terminal_and_not_exported(self):
        with patch.object(self.env, "step", side_effect=RuntimeError("sim failure")):
            result = self.controller.step(["MOVE_FORWARD"])
        self.assertEqual(result["status"], "TOOL_ERROR")
        self.assertTrue(result["done"])
        self.assertEqual(result["reason"], "habitat_action_exception")
        self.assertFalse((self.controller.directory / "result.json").exists())
        self.assertTrue((self.controller.directory / "failure.json").exists())

    def test_fresh_codex_command_uses_scoped_mcp_and_no_resume(self):
        args = SimpleNamespace(split=None, phase="phase2",
                               docker_bin="docker", container="havln-cma",
                               episode_id=1, max_steps=128, gpu_id=0, seed=0,
                               output_dir="/workspace/agent-runs/test-codex-1",
                               codex_bin="codex", reasoning_effort="medium",
                               model="gpt-6-astra")
        cmd = build_command(args, Path("/tmp/last.txt"))
        self.assertIn("exec", cmd)
        self.assertNotIn("resume", cmd)
        self.assertIn("--json", cmd)
        self.assertIn("--ephemeral", cmd)
        self.assertIn("--ignore-user-config", cmd)
        self.assertIn('mcp_servers.havln.enabled_tools=["observe","step"]',
                      cmd)
        args_override = next(cmd[i + 1] for i, word in enumerate(cmd)
                             if word == "-c" and cmd[i + 1].startswith(
                                 "mcp_servers.havln.args="))
        self.assertIn("havln_mcp_server.py", args_override)
        self.assertIn("--episode-id", args_override)


if __name__ == "__main__":
    unittest.main()
