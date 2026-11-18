"""Offline batch scheduling, packet integrity and replay compatibility tests."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zipfile import ZipFile

import numpy as np

from havln_agent.codex_batch import select_ids, aggregate
from havln_agent.replay_server import ReplaySource, EpisodeReplay
from scripts.pack_agent_diagnostics import pack
from scripts.run_codex_batch import (build_child_command, main, run_batch)


class CodexBatchTests(unittest.TestCase):
    def test_original_ids_are_unique_deterministic_and_bounds_checked(self):
        ids = list(range(1, 3409))
        self.assertEqual(select_ids(ids, 100, 3), [100, 101, 102])
        a = select_ids(ids, 100, 25, True, 73)
        b = select_ids(ids, 100, 25, True, 73)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 25)
        self.assertEqual(len(set(a)), 25)
        self.assertGreaterEqual(min(a), 100)
        self.assertLessEqual(max(a), 3408)
        self.assertNotEqual(a, select_ids(ids, 100, 25, True, 74))
        with self.assertRaisesRegex(ValueError, "exceed"):
            select_ids(ids, 3407, 3)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_ids([1, 1, 2], 1, 2)

    def _write(self, file, data):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(data) + "\n", encoding="utf-8")

    def test_aggregate_keeps_episodes_isolated_and_partial_export_safe(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            self._write(root / "batch_plan.json", {
                "architecture": "codex_batch_v1", "phase": "phase2", "split": "test",
                "selected_episode_ids": [2, 9], "auto_archive": False})
            self._write(root / "batch_worker_results.json", {
                "episodes": {"2": {"returncode": 0, "log_file": "worker-test-2.log"},
                             "9": {"returncode": 2, "log_file": "worker-test-9.log"}}})
            first = root / "traces/test-2"
            self._write(first / "summary.json", {"completed": True, "steps": 2})
            self._write(first / "result.json", {
                "row": {"episode_id": "2", "actions": ["MOVE_FORWARD", "STOP"]}})
            self._write(first / "codex/run_status.json", {"codex_exit_code": 0})
            self._write(first / "codex/codex_usage.json", {"tokens": {
                "status": "complete", "input_tokens": 100,
                "cached_input_tokens": 80, "output_tokens": 20,
                "reasoning_output_tokens": 10, "total_tokens": 120}})
            second = root / "traces/test-9"
            self._write(second / "summary.json", {"completed": False})
            self._write(second / "codex/run_status.json", {"codex_exit_code": 124})
            self._write(second / "codex/codex_usage.json", {"tokens": {
                "status": "unavailable", "input_tokens": None,
                "cached_input_tokens": None, "output_tokens": None,
                "reasoning_output_tokens": None, "total_tokens": None}})
            result = aggregate(root)
            self.assertFalse(result["success"])
            self.assertEqual(result["completed_episodes"], 1)
            self.assertEqual(result["failed_episodes"], 1)
            self.assertEqual(result["tokens"]["status"], "partial")
            self.assertEqual(result["tokens"]["total_tokens"], 120)
            diagnostic = json.loads((root / "test.diagnostic.json").read_text())
            self.assertTrue(diagnostic["not_an_official_submission"])
            self.assertTrue(diagnostic["partial"])
            self.assertEqual([r["episode_id"] for r in diagnostic["episodes"]], ["2"])
            self.assertFalse(json.loads((root / "run_status.json").read_text())["success"])

    def test_independent_child_command_and_per_gpu_concurrency(self):
        args = SimpleNamespace(phase="phase2", max_steps=128,
                               timeout_seconds=100, seed=42,
                               model="gpt-6.2-sol", reasoning_effort="low",
                               docker_bin="docker", codex_bin="codex",
                               container="havln-cma", data_root="/data/havln2",
                               output_dir="/workspace/agent-runs/batch",
                               batch=2, gpu_ids=[0, 1])
        cmd = build_child_command(args, "test", 7, 1)
        self.assertIn("--batch-child", cmd)
        self.assertIn("--no-auto-archive", cmd)
        self.assertEqual(cmd[cmd.index("--episode-id") + 1], "7")
        self.assertEqual(cmd[cmd.index("--gpu-id") + 1], "1")
        self.assertEqual(cmd[cmd.index("--output-dir") + 1], args.output_dir)
        self.assertEqual(cmd[cmd.index("--model") + 1], "gpt-6.2-sol")
        lock = threading.Lock()
        active = {0: 0, 1: 0}
        peaks = {0: 0, 1: 0}
        class FakePopen:
            def __init__(self, command, **kwargs):
                self.gpu = int(command[command.index("--gpu-id") + 1])
                self.pid = os.getpid()
            def __enter__(self):
                return self
            def __exit__(self, *unused):
                return False
            def wait(self):
                with lock:
                    active[self.gpu] += 1
                    peaks[self.gpu] = max(peaks[self.gpu], active[self.gpu])
                time.sleep(0.035)
                with lock:
                    active[self.gpu] -= 1
                return 0
        with tempfile.TemporaryDirectory() as work, \
             patch("scripts.run_codex_batch.subprocess.Popen", FakePopen), \
             patch("scripts.run_codex_batch.subprocess.run",
                   return_value=SimpleNamespace(returncode=0)):
            results = run_batch(args, "test", list(range(1, 9)), work)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(x["returncode"] == 0 for x in results.values()))
        self.assertEqual(peaks, {0: 2, 1: 2})

    def test_preflight_and_no_archive_flag_do_not_start_paid_model_unnecessarily(self):
        with patch("scripts.run_codex_batch.preflight"), \
             patch("scripts.run_codex_batch.discover_ids", return_value=list(range(1, 51))), \
             patch("scripts.run_codex_batch.reserve_output_directory") as reserve, \
             patch("scripts.run_codex_batch.archive_codex_run") as archive, \
             patch("scripts.run_codex_batch.read_rate_limits",
                   return_value={"status": "unavailable"}), \
             patch("scripts.run_codex_batch.copy_json"), \
             patch("scripts.run_codex_batch.run_batch",
                   return_value={"4": {"returncode": 0}}) as batch, \
             patch("scripts.run_codex_batch._docker",
                   return_value=SimpleNamespace(stdout=json.dumps({
                       "success": True, "episode_count": 1,
                       "completed_episodes": 1, "tokens": {
                           "total_tokens": 123, "status": "complete"}}))):
            base = ["--output-dir", "/workspace/agent-runs/test-batch",
                    "--episode-start", "4", "--episode-limit", "1",
                    "--no-auto-archive"]
            self.assertEqual(main(base + ["--preflight"]), 0)
            reserve.assert_not_called()
            batch.assert_not_called()
            self.assertEqual(main(base), 0)
            reserve.assert_called_once()
            batch.assert_called_once()
            archive.assert_not_called()

    def test_zip_dedup_and_replay_resolve_all_original_observation_ids(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work) / "run"
            images = root / "traces/test-8/observations"
            images.mkdir(parents=True)
            rgb = np.full((12, 12, 3), 100, dtype=np.uint8)
            depth = np.full((12, 12), .6, dtype=np.float32)
            np.savez_compressed(images / "obs_000000.npz",
                                rgb=rgb, depth=depth, pitch_deg=0.0)
            shutil.copyfile(images / "obs_000000.npz", images / "obs_000001.npz")
            (images / "../episode.json").write_text('{"episode_id":"8"}')
            archive = Path(work) / "diagnostic.zip"
            manifest = pack(root, archive, frames="all")
            self.assertEqual(manifest["episodes"]["test-8"]["included_frames"], 2)
            self.assertEqual(manifest["episodes"]["test-8"]["stored_frames"], 1)
            self.assertEqual(manifest["frame_aliases"]["test-8"]["obs_000001.npz"],
                             "obs_000000.npz")
            with ZipFile(archive) as zipfile:
                names = zipfile.namelist()
                self.assertIn("run/traces/test-8/observations/obs_000000.npz", names)
                self.assertNotIn("run/traces/test-8/observations/obs_000001.npz", names)
            source = ReplaySource(archive)
            self.addCleanup(source.zip.close)
            replay = EpisodeReplay(source, "test-8")
            self.assertEqual(replay.frame("obs_000001", "rgb"),
                             replay.frame("obs_000000", "rgb"))
            self.assertEqual(replay.frame("obs_000001", "depth"),
                             replay.frame("obs_000000", "depth"))

    def test_startup_only_diagnostics_open_in_replay(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work) / "run"
            (root / "codex").mkdir(parents=True)
            (root / "codex/codex_events.jsonl").write_text(
                '{"type":"thread.started"}\n{"type":"turn.failed"}\n')
            (root / "run_status.json").write_text('{"success":false}')
            path = Path(work) / "startup.zip"
            pack(root, path, frames="none", allow_empty=True)
            source = ReplaySource(path)
            self.addCleanup(source.zip.close)
            self.assertEqual(source.episodes(), ["codex-startup"])
            replay = EpisodeReplay(source, "codex-startup")
            self.assertEqual(replay.overview()["counts"]["codex"], 2)
            self.assertFalse(replay.failure["success"])
            self.assertEqual(replay.overview()["events"][-1]["event_type"],
                             "turn.failed")


if __name__ == "__main__":
    unittest.main()
