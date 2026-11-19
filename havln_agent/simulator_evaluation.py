"""Capture actual Habitat metrics outside the agent's observation/tool protocol.

This module is for diagnostics and offline score reconstruction only. Nothing
recorded here is returned to NavigationAgent, Codex, MCP tools, or model prompts.
Hidden test goals/reference collisions are never inferred from an action trace.
"""
from functools import lru_cache
import json
import math
from pathlib import Path
import time

from .competition_scoring import (
    aggregate_episodes, published_reference_collision_count)
from .submission_contract import atomic_json


def finite_number(value):
    """A scalar real simulator measurement, not a model guess or bool."""
    if type(value) is bool:
        return float(value)
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            return None
    if type(value) not in (int, float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


@lru_cache(maxsize=12)
def _collision_reference(path):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def reference_collision_count(data_root, split, episode_id):
    """Only published validation allowances; absent test allowances stay unknown."""
    if split not in ("val_seen", "val_unseen"):
        return None
    number = None
    if data_root is not None:
        path = Path(data_root) / "HA-R2R-tools" / ("collision_num_" + split + ".json")
        number = finite_number(_collision_reference(str(path)).get(str(episode_id)))
    if number is not None and number >= 0:
        return number
    # Released validation annotations are public, and can be bundled with
    # this toolkit when the operator mounted only HA-R2R episodes/assets.
    # Never substitute these for withheld Phase 2 test annotations.
    return published_reference_collision_count(split, episode_id)


def _metric_sample(info):
    if not isinstance(info, dict):
        return {}
    out = {}
    for key in ("distance_to_goal", "success", "spl", "path_length",
                "steps_taken", "oracle_success"):
        value = finite_number(info.get(key))
        if value is not None:
            out[key] = value
    for key in ("collisions", "collisions_detail"):
        record = info.get(key)
        if isinstance(record, dict):
            compact = {}
            count = finite_number(record.get("count"))
            if count is not None and count >= 0:
                compact["count"] = count
            if type(record.get("is_collision")) is bool:
                compact["is_collision"] = record["is_collision"]
            if compact:
                out[key] = compact
    return out


def _position(env):
    try:
        position = env.habitat_env.sim.get_agent_state().position
        values = [finite_number(x) for x in position]
        if len(values) == 3 and all(value is not None for value in values):
            return values
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return None


class SimulatorMetricCapture:
    """Separate, append-only physical measurements from real environment steps."""

    def __init__(self, directory, episode_id, split, data_root=None):
        self.directory = Path(directory)
        self.episode_id, self.split = str(episode_id), split
        self.reference = reference_collision_count(data_root, split, episode_id)
        self.records = []
        self.error = None
        self.path = self.directory / "simulator_metrics.jsonl"
        # Create an explicit log even if the first reset fails.
        self.path.touch()

    def _record(self, phase, env, info, action=None):
        row = dict(phase=phase, step_index=len(self.records) - 1 if phase == "reset"
                   else sum(r["phase"] == "action" for r in self.records),
                   recorded_monotonic_s=time.monotonic(),
                   action=action, metrics=_metric_sample(info),
                   simulator_position_m=_position(env))
        self.records.append(row)
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
        except OSError as error:
            self.error = str(error)[:200]

    def reset(self, env):
        try:
            info = env.habitat_env.get_metrics()
        except (AttributeError, RuntimeError, ValueError):
            info = {}
        self._record("reset", env, info)

    def action(self, action, info, env):
        self._record("action", env, info, action)

    def finish(self, complete, actions):
        """Write only observed simulator metrics; partial sessions remain partial."""
        observations = [r for r in self.records if r["phase"] == "action"]
        last = observations[-1]["metrics"] if observations else {}
        total = len(observations)
        actions_match = [r["action"] for r in observations] == list(actions)
        result = dict(episode_id=self.episode_id, split=self.split,
            capture="live_habitat_measurements", official=False,
            complete=bool(complete and actions_match and total and
                          (actions[-1] == "STOP" or total == 500)),
            recorded_actions=total, action_log_matches=actions_match,
            available=list(sorted(last)), final_metrics=last,
            final_simulator_position_m=(observations[-1]["simulator_position_m"]
                if observations else None),
            reference_collision_count=self.reference,
            diagnostics_error=self.error)
        nav = finite_number(last.get("distance_to_goal"))
        arrival = finite_number(last.get("success"))
        collisions = last.get("collisions") or {}
        detail = last.get("collisions_detail") or {}
        count = finite_number(collisions.get("count"))
        if count is None:
            count = finite_number(detail.get("count"))
        if nav is not None and nav >= 0:
            result["navigation_error_m"] = nav
        if arrival in (0.0, 1.0):
            result["success"] = bool(arrival)
        if count is not None and count >= 0:
            result["collisions_count"] = count
        if result["complete"] and all(k in result for k in
                ("navigation_error_m", "success", "collisions_count")) \
                and self.reference is not None:
            result["score_episode"] = dict(
                episode_id=self.episode_id,
                navigation_error_m=result["navigation_error_m"],
                success=result["success"],
                collisions_count=result["collisions_count"],
                reference_collision_count=self.reference)
        atomic_json(self.directory / "simulator_metrics.json", result)
        return result


def publish_run_metrics(root, expected):
    """Aggregate existing evaluator artifacts, no Habitat/model rerun.

    Partial metrics are published independently; composite Score requires
    complete genuine navigation, success and adjusted collision measurements.
    """
    root = Path(root)
    report = dict(source="actual_live_habitat_measurements",
        official=False, submitted_replay=False,
        note=("Live closed-loop simulator measurements are not an official "
              "CodaBench replay; dynamic human timing can differ."),
        splits={})
    for split, episode_ids in expected.items():
        evaluated, partial, samples, observed = [], [], [], []
        for episode_id in episode_ids:
            path = root / "traces" / (split + "-" + str(episode_id)) / "simulator_metrics.json"
            if not path.is_file():
                partial.append(dict(episode_id=str(episode_id), reason="missing_capture"))
                continue
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                partial.append(dict(episode_id=str(episode_id), reason="invalid_capture"))
                continue
            if row.get("episode_id") != str(episode_id) or row.get("split") != split:
                partial.append(dict(episode_id=str(episode_id), reason="capture_identity_mismatch"))
                continue
            # Publish the measured components separately from fully scored
            # episodes. Missing goal truth or collision allowance is not a zero.
            observed.append({key: row[key] for key in (
                "episode_id", "complete", "recorded_actions",
                "navigation_error_m", "success", "collisions_count",
                "reference_collision_count", "final_simulator_position_m",
                "final_metrics", "available") if key in row})
            if row.get("complete"):
                samples.append(row)
            score_episode = row.get("score_episode") if row.get("complete") else None
            if score_episode is not None:
                evaluated.append(score_episode)
            else:
                partial.append(dict(episode_id=str(episode_id),
                    reason=("incomplete_actions" if not row.get("complete")
                            else "required_ground_truth_unavailable"),
                    available=row.get("available", []),
                    collisions_count=row.get("collisions_count"),
                    navigation_error_m=row.get("navigation_error_m"),
                    success=row.get("success")))
        def mean(key):
            values = [finite_number(s.get(key)) for s in samples]
            values = [v for v in values if v is not None]
            return (sum(values)/len(values) if values else None), len(values)
        components = {}
        for key, label in (("navigation_error_m", "NE_m_observed"),
                           ("success", "arrival_success_rate_observed"),
                           ("collisions_count", "raw_collisions_mean_observed")):
            value, count = mean(key)
            components[label] = dict(value=value, episodes=count)
        for key, label in (("path_length", "path_length_m_observed"),
                           ("steps_taken", "simulator_steps_mean_observed")):
            vals = [finite_number(s.get("final_metrics", {}).get(key)) for s in samples]
            vals = [v for v in vals if v is not None]
            components[label] = dict(value=sum(vals)/len(vals) if vals else None,
                                      episodes=len(vals))
        entry = dict(requested_episodes=len(episode_ids), captured_complete=len(samples),
                     fully_scored_episodes=len(evaluated),
                     missing_or_partial=partial, components=components,
                     episodes=evaluated, observed_episodes=observed,
                     scope="complete_run" if len(evaluated)==len(episode_ids)
                           else "diagnostic_subset_or_missing_truth")
        if evaluated:
            values, _, normalization = aggregate_episodes(evaluated, split)
            entry["calculated_metrics"] = {key: values[key] for key in ("SR","NE","CR","TCR")}
            # Keep the original field for old consumers; the established
            # Replay importer reads 'metrics'.
            entry["metrics"] = dict(entry["calculated_metrics"])
            entry["calculated_score"] = values["score"]
            entry["collision_normalization"] = normalization
        report["splits"][split] = entry
    atomic_json(root / "score_metrics.json", report)
    return report
