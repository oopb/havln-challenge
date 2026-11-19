"""Default true-Habitat metric capture; no API or Habitat dependencies."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from havln_agent.simulator_evaluation import (
    SimulatorMetricCapture, publish_run_metrics, _metric_sample)


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

    def test_metric_parser_ignores_text_and_nonfinite_values(self):
        parsed = _metric_sample(dict(distance_to_goal="2.5", success=float("nan"),
            collisions={"count": "five"}, path_length=1.25))
        self.assertEqual(parsed, {"path_length": 1.25})


if __name__ == "__main__":
    unittest.main()
