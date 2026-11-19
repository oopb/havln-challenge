"""Default true-Habitat metric capture; no API or Habitat dependencies."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from havln_agent.simulator_evaluation import (
    SimulatorMetricCapture, publish_run_metrics, _metric_sample)
from havln_agent.competition_scoring import summarize_run
from havln_agent.replay_server import ReplaySource, _load_scoring_report


class DummyEnv:
    def __init__(self):
        self.habitat_env = SimpleNamespace(
            sim=SimpleNamespace(get_agent_state=lambda: SimpleNamespace(
                position=(1.0, 0.0, 2.0))),
            get_metrics=lambda: {"distance_to_goal": 4, "success": 0,
                                 "collisions": {"count": 0}})


class MetricCaptureTests(unittest.TestCase):
    def test_only_actual_simulator_fields_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data" / "HA-R2R-tools"
            data.mkdir(parents=True)
            (data / "collision_num_val_seen.json").write_text(
                json.dumps({"1": 1}), encoding="utf-8")
            trace = root / "run" / "traces" / "val_seen-1"
            trace.mkdir(parents=True)
            env = DummyEnv()
            capture = SimulatorMetricCapture(trace, "1", "val_seen", root / "data")
            capture.reset(env)
            capture.action("MOVE_FORWARD", {
                "distance_to_goal": 2.5, "success": 0,
                "collisions": {"count": 1, "is_collision": True},
                "collisions_detail": {"count": 1, "steps": [True]},
                "path_length": 0.25, "steps_taken": 1,
            }, env)
            capture.action("STOP", {
                "distance_to_goal": 2.5, "success": 1,
                "collisions": {"count": 1, "is_collision": False},
                "path_length": 0.25, "steps_taken": 2
            }, env)
            result = capture.finish(True, ["MOVE_FORWARD", "STOP"])
            self.assertTrue(result["complete"])
            self.assertEqual(result["navigation_error_m"], 2.5)
            self.assertEqual(result["reference_collision_count"], 1)
            self.assertTrue(result["success"])
            self.assertEqual(result["score_episode"]["collisions_count"], 1)
            self.assertNotIn("steps", result["final_metrics"].get("collisions_detail", {}))
            self.assertEqual(len((trace / "simulator_metrics.jsonl")
                                 .read_text().splitlines()), 3)
            self.assertEqual(result["final_simulator_position_m"], [1.0, 0.0, 2.0])
            report = publish_run_metrics(root / "run", {"val_seen": ["1"]})
            entry = report["splits"]["val_seen"]
            self.assertEqual(entry["fully_scored_episodes"], 1)
            self.assertEqual(entry["calculated_metrics"]["SR"], 1)
            self.assertGreater(entry["calculated_score"], 0)
            self.assertTrue((root / "run" / "score_metrics.json").is_file())

    def test_hidden_test_truth_is_not_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "traces" / "test-1"
            trace.mkdir(parents=True)
            capture = SimulatorMetricCapture(trace, "1", "test", root)
            capture.reset(DummyEnv())
            capture.action("STOP", {"collisions": {"count": 4},
                                    "path_length": 0.0}, DummyEnv())
            result = capture.finish(True, ["STOP"])
            self.assertEqual(result["collisions_count"], 4)
            self.assertNotIn("score_episode", result)
            report = publish_run_metrics(root, {"test": ["1"]})
            entry = report["splits"]["test"]
            self.assertEqual(entry["fully_scored_episodes"], 0)
            self.assertNotIn("calculated_score", entry)
            self.assertEqual(entry["components"]["raw_collisions_mean_observed"]["value"], 4)

    def test_missing_metrics_and_partial_actions_never_become_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "traces" / "val_unseen-9"
            trace.mkdir(parents=True)
            capture = SimulatorMetricCapture(trace, "9", "val_unseen", root)
            capture.reset(DummyEnv())
            capture.action("MOVE_FORWARD", {
                "distance_to_goal": 2.0, "success": 1,
                "collisions": {"count": 0}}, DummyEnv())
            result = capture.finish(False, ["MOVE_FORWARD"])
            self.assertFalse(result["complete"])
            self.assertNotIn("score_episode", result)
            report = publish_run_metrics(root, {"val_unseen": ["9", "10"]})
            self.assertEqual(report["splits"]["val_unseen"]["fully_scored_episodes"], 0)
            self.assertEqual(len(report["splits"]["val_unseen"]["missing_or_partial"]), 2)

    def test_real_collector_output_is_readable_by_replay_score(self):
        # Regression: collector used calculated_metrics but Replay required metrics,
        # causing an incorrect 'no independent evaluation' warning.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = root / "data" / "HA-R2R-tools"
            refs.mkdir(parents=True)
            (refs / "collision_num_val_seen.json").write_text(
                '{"1": 0}', encoding="utf-8")
            trace = root / "run" / "traces" / "val_seen-1"
            trace.mkdir(parents=True)
            cap = SimulatorMetricCapture(trace, "1", "val_seen", root / "data")
            cap.reset(DummyEnv())
            cap.action("STOP", {
                "distance_to_goal": 1.1, "success": 1,
                "collisions": {"count": 0},
                "path_length": 2.0, "steps_taken": 1,
            }, DummyEnv())
            cap.finish(True, ["STOP"])
            original = publish_run_metrics(root / "run", {"val_seen": ["1"]})
            self.assertIn("metrics", original["splits"]["val_seen"])
            source = ReplaySource(root / "run")
            imported, filename = _load_scoring_report(source)
            viewed = summarize_run({"val_seen": ["1"]}, imported, filename)
            split = viewed["by_split"]["val_seen"]
            self.assertEqual(split["status"], "scored")
            self.assertEqual(split["metrics"]["SR"], 1)
            self.assertGreater(split["score"], 0)
            self.assertEqual(split["observed_episodes"]["1"]["collisions_count"], 0)
            self.assertEqual(split["captured_complete"], 1)

    def test_test_split_shows_real_metrics_without_faking_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "traces" / "test-1"
            trace.mkdir(parents=True)
            cap = SimulatorMetricCapture(trace, "1", "test", root)
            cap.reset(DummyEnv())
            cap.action("STOP", {
                "collisions": {"count": 3},
                "path_length": 0.75, "steps_taken": 1
            }, DummyEnv())
            cap.finish(True, ["STOP"])
            report = publish_run_metrics(root, {"test": ["1"]})
            split = summarize_run({"test": ["1"]}, report, "score_metrics.json")["by_split"]["test"]
            self.assertEqual(split["status"], "partial_metrics")
            self.assertIsNone(split["score"])
            self.assertEqual(split["components"]["raw_collisions_mean_observed"]["value"], 3)
            self.assertEqual(split["observed_episodes"]["1"]["collisions_count"], 3)
            self.assertEqual(split["fully_scored_episodes"], 0)

    def test_earlier_collector_format_is_still_accepted(self):
        legacy = {"splits": {"val_seen": {
            "calculated_metrics": {"SR": 0.5, "NE": 3, "CR": 0.2, "TCR": 2},
            "fully_scored_episodes": 1, "requested_episodes": 1,
            "scope": "complete_run"}}}
        split = summarize_run({"val_seen": ["1"]}, legacy, "score_metrics.json")["by_split"]["val_seen"]
        self.assertEqual(split["status"], "scored")
        self.assertEqual(split["metrics"]["NE"], 3)

    def test_metric_parser_ignores_text_and_nonfinite_values(self):
        parsed = _metric_sample(dict(distance_to_goal="2.5", success=float("nan"),
            collisions={"count": "five"}, path_length=1.25))
        self.assertEqual(parsed, {"path_length": 1.25})


if __name__ == "__main__":
    unittest.main()
