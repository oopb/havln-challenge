"""One isolated, tool-driven HA-VLN episode. No scheduler or cloud API client.

The controller only owns public Habitat observations, atomic action receipts,
the action budget and durable diagnostics. A Codex/MCP adapter owns the agent
conversation; neither side imports the existing NavigationAgent.
"""
from dataclasses import asdict
import base64
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .habitat_adapter import read_current_observation
from .perception import Geometry, adapt_observation, depth_image_url
from .replay_capture import ReplayRecorder
from .submission_contract import ACTIONS, atomic_json, check_rows
from .simulator_evaluation import SimulatorMetricCapture, publish_run_metrics


class CodexEpisode:
    def __init__(self, env, camera, episode_id, split, max_steps, output_dir,
                 batch_mode=False, data_root=None):
        if type(max_steps) is not int or not 1 <= max_steps <= 500:
            raise ValueError("max_steps must be an integer in [1, 500]")
        if split not in ("test", "val_seen", "val_unseen"):
            raise ValueError("unknown evaluation split")
        if not str(episode_id).isdigit():
            raise ValueError("episode_id must be numeric")

        self.env, self.camera = env, camera
        self.episode_id, self.split = str(episode_id), split
        self.max_steps = max_steps
        self.batch_mode = batch_mode
        self.directory = Path(output_dir) / "traces" / (split + "-" + self.episode_id)
        if self.directory.exists():
            raise FileExistsError("Refusing to overwrite an existing Codex episode: " +
                                  str(self.directory))
        self.directory.mkdir(parents=True)
        (self.directory / "observations").mkdir()
        self.evaluator = SimulatorMetricCapture(
            self.directory, self.episode_id, split, data_root)
        self.replay = ReplayRecorder(self.directory, camera)
        self.actions = []
        self.started = time.monotonic()
        self.observation_count = 0
        self.tool_count = 0
        self.pitch_deg = 0.0
        # Relative actions only: never expose simulator pose or infer odometry.
        self._heading_deg = 0.0
        self._turn_total_deg = 0.0
        self._turn_net_deg = 0.0
        self._recent_stalls = []
        # Grant one extra public Depth frame after a visual stall; do not
        # repeatedly send it for the same blocked approach.
        self._stall_depth_pending = False
        self._stall_depth_used = False
        self.done = False
        self.end_reason = None
        self.finalized = False

        raw = self.env.reset()
        self.evaluator.reset(self.env)
        if str(self.env.current_episode.episode_id) != self.episode_id:
            raise ValueError("Habitat reset selected the wrong episode")
        self.current = adapt_observation(raw, camera)
        expected = self.env.current_episode.instruction.instruction_text
        if self.current.instruction != expected:
            raise ValueError("Instruction sensor disagrees with episode metadata")
        self.instruction = expected
        self.current_id = self._save(self.current)
        atomic_json(self.directory / "episode.json", {
            "episode_id": self.episode_id,
            "split": split,
            "instruction": self.instruction,
            "architecture": "codex_mcp_v1",
            "camera": asdict(camera),
            "max_steps": max_steps,
        })

    def _append(self, filename, row):
        with (self.directory / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def _save(self, observation):
        identifier = "obs_%06d" % self.observation_count
        self.observation_count += 1
        np.savez_compressed(self.directory / "observations" / (identifier + ".npz"),
                            rgb=observation.rgb, depth=observation.depth,
                            pitch_deg=observation.pitch_deg)
        self.replay.observation(identifier)
        return identifier

    def _start_tool(self, name):
        self.tool_count += 1
        ident = "codex_tool_%04d" % self.tool_count
        self.replay.event("tool", tool_index=self.tool_count - 1,
                          tool_call_id=ident, tool=name)
        return ident

    def _end_tool(self, ident, name, arguments, result):
        self._append("tools.jsonl", {
            "call_id": ident, "actor": "codex",
            "call": {"tool": name, "args": arguments},
            "result": result,
        })

    @property
    def remaining(self):
        return self.max_steps - len(self.actions)

    def observe(self):
        ident = self._start_tool("observe")
        # Re-render through the public sensor suite; no privileged pose/goal data.
        if not self.done:
            observation = read_current_observation(self.env, self.camera,
                                                   self.instruction)
            observation.pitch_deg = self.pitch_deg
            self.current = observation
            self.current_id = self._save(observation)
        frame = Image.fromarray(self.current.rgb)
        buffer = io.BytesIO()
        frame.save(buffer, format="PNG")
        # Public Depth is visual evidence, never a geometry movement gate.
        # A recent exact-RGB stall can request one extra inspection even when
        # the conservative center-footprint heuristic finds no obstacle.
        geometry = Geometry(self.current, self.camera, margin=0.0)
        obstacle = geometry.central_forward_obstruction()
        body_risk = geometry.forward_body_risk()
        stall_inspection = self._stall_depth_pending and not self.done
        # Full robot width is used for optional Depth evidence, including
        # side-clearance risks the old narrow centerline check cannot see.
        # This selects observation content only; it never vetoes an action.
        send_depth = obstacle["confirmed"] or body_risk is not None or stall_inspection
        depth_data = (depth_image_url(self.current, self.camera).split(",", 1)[1]
                      if send_depth else None)
        if stall_inspection:
            self._stall_depth_pending = False
        info = {
            "episode_id": self.episode_id,
            "observation_id": self.current_id,
            "executed_actions": len(self.actions),
            "remaining_actions": self.remaining,
            "done": self.done,
            "reason": self.end_reason,
            "pitch_deg": self.pitch_deg,  # commanded tilt relative to initial view
            "depth_sent": send_depth,
        }
        if send_depth:
            info["depth_reason"] = (
                "front_obstacle" if obstacle["confirmed"] else
                "near_body_risk" if body_risk is not None else
                "post_stall_inspection")
        if obstacle["confirmed"]:
            info["front_obstacle_m"] = obstacle["nearest_along_m"]
        if body_risk is not None:
            # A measured left/right/center risk is not an object identity
            # or proof of a collision. Codex still chooses the next action.
            info["near_body_risk"] = body_risk
        self._end_tool(ident, "observe", {}, dict(info, status="OK"))
        return info, base64.b64encode(buffer.getvalue()).decode("ascii"), depth_data

    def _motion_hint(self, executed):
        """Tiny, conditional factual receipt; never blocks a Codex decision."""
        turns = [a for a in executed if a in ("TURN_LEFT", "TURN_RIGHT")]
        batch_total = len(turns) * self.camera.turn_angle
        batch_net = (turns.count("TURN_RIGHT") - turns.count("TURN_LEFT")) * self.camera.turn_angle
        hint = {}
        # Stats since the last forward with changing RGB are a visual proxy,
        # NOT confirmation that the robot physically moved.
        if self._turn_total_deg >= 180:
            hint.update(turn_total_deg=self._turn_total_deg,
                        turn_net_deg=self._turn_net_deg)
        if batch_total >= 180:
            hint.update(batch_turn_total_deg=batch_total,
                        batch_turn_net_deg=batch_net)
        recent = [(idx, heading) for idx, heading in self._recent_stalls
                  if len(self.actions) - 1 - idx <= 16]
        if recent:
            # R positive, L negative, relative to the *current* camera heading.
            hint["recent_stall_directions_deg"] = [
                (heading - self._heading_deg + 180) % 360 - 180
                for _, heading in recent
            ]
        return hint or None

    def _atomic(self, action, tool_id, origin):
        before = self.current_id
        before_rgb = self.current.rgb if action == "MOVE_FORWARD" else None
        try:
            raw, _, environment_done, info = self.env.step(action)
        except Exception as error:
            # An uncertain side effect cannot safely be retried or replayed.
            self.done, self.end_reason = True, "habitat_action_exception"
            raise RuntimeError("Habitat action failed; execution stopped") from error
        if isinstance(info, dict) and info.get("error"):
            self.done, self.end_reason = True, "habitat_action_rejected"
            raise RuntimeError("Habitat rejected action: " + str(action))
        # env.step succeeded: record the atomic action before any fallible
        # evaluator, observation conversion or disk capture work. A later
        # diagnostic failure must never make a real simulator action disappear.
        index = len(self.actions)
        self.actions.append(action)
        try:
            self.evaluator.action(action, info, self.env)
            if action == "LOOK_UP":
                self.pitch_deg += self.camera.tilt_angle
            elif action == "LOOK_DOWN":
                self.pitch_deg -= self.camera.tilt_angle
            observation = adapt_observation(raw, self.camera)
            observation.pitch_deg = self.pitch_deg
            self.current = observation
            self.current_id = self._save(observation)
        except Exception as error:
            self.done, self.end_reason = True, "post_action_processing_exception"
            self._append("steps.jsonl", {
                "step": index, "action": action, "tool_call_id": tool_id,
                "origin": origin,
                "input_file": "observations/" + before + ".npz",
                "post_file": None,
                "post_action_error": type(error).__name__ + ": " + str(error)[:240],
            })
            raise RuntimeError(
                "Habitat executed " + action +
                " but post-action capture failed; episode stopped") from error
        # Preserve the best-performing baseline's exact-RGB stall policy.
        stalled = before_rgb is not None and np.array_equal(before_rgb, observation.rgb)
        if action in ("TURN_LEFT", "TURN_RIGHT"):
            angle = self.camera.turn_angle * (1 if action == "TURN_RIGHT" else -1)
            self._heading_deg += angle
            self._turn_net_deg += angle
            self._turn_total_deg += abs(angle)
        elif action == "MOVE_FORWARD":
            if stalled:
                self._recent_stalls.append((index, self._heading_deg))
                self._recent_stalls = self._recent_stalls[-2:]
                if not self._stall_depth_used:
                    self._stall_depth_pending = True
                    self._stall_depth_used = True
            else:
                # Only "RGB changed", not measured displacement. Reset the
                # local turn window and stale obstacle attempts conservatively.
                self._turn_total_deg = 0.0
                self._turn_net_deg = 0.0
                self._recent_stalls.clear()
                self._stall_depth_pending = False
                self._stall_depth_used = False
        row = {
            "step": index, "action": action, "tool_call_id": tool_id,
            "origin": origin,
            "input_file": "observations/" + before + ".npz",
            "post_file": "observations/" + self.current_id + ".npz",
        }
        if stalled:
            row["motion_feedback"] = "rgb_unchanged"
        self._append("steps.jsonl", row)
        self.replay.action(row)
        if action == "STOP":
            self.done, self.end_reason = True, (
                "budget_guard_stop" if origin == "budget_guard" else "codex_stop")
        elif environment_done:
            self.done, self.end_reason = True, "habitat_terminated_without_stop"
        elif self.remaining == 0:
            self.done, self.end_reason = True, "action_budget_exhausted"
        return stalled

    def step(self, requested):
        ident = self._start_tool("step")
        args = {"actions": requested}
        action_start = len(self.actions)
        try:
            if self.done:
                result = self.status()
                result["status"] = "ALREADY_DONE"
            else:
                if (not isinstance(requested, list) or not requested
                        or any(not isinstance(a, str) or a not in ACTIONS
                               for a in requested)):
                    raise ValueError("actions must be a non-empty list of six legal action names")
                if "STOP" in requested[:-1]:
                    raise ValueError("STOP may only be the final action in a batch")
                # A final STOP is reserved for diagnostic budgets under 500,
                # so their action rows remain valid for the official contract.
                available = self.remaining if self.max_steps == 500 else max(0, self.remaining - 1)
                if len(requested) > available and not (
                        len(requested) == self.remaining and requested[-1] == "STOP"):
                    raise ValueError("batch exceeds remaining action budget; reserve final STOP")
                executed = []
                stalled = False
                for action in requested:
                    if self.done:
                        break
                    unchanged = self._atomic(action, ident, "codex")
                    executed.append(action)
                    if unchanged:
                        stalled = True
                        break  # The remaining actions, including STOP, are not executed.
                requested_executed = len(executed)
                if (not self.done and self.max_steps < 500
                        and self.remaining == 1):
                    self._atomic("STOP", ident, "budget_guard")
                    executed.append("STOP")
                result = self.status()
                result.update(status=("DONE" if self.done else
                                      "NO_VISUAL_PROGRESS" if stalled else "RUNNING"),
                              executed=executed, requested=requested)
                if stalled:
                    result["skipped_count"] = len(requested) - requested_executed
                    result["motion_feedback"] = "MOVE_FORWARD left RGB unchanged; choose whether to turn or observe."
                if not self.done:
                    hint = self._motion_hint(executed)
                    if hint:
                        result["motion_hint"] = hint
        except (ValueError, RuntimeError) as error:
            result = self.status()
            result.update(status="TOOL_ERROR", error=str(error))
            if isinstance(requested, list):
                result["requested"] = requested
                result["executed"] = self.actions[action_start:]
                result["skipped_count"] = max(
                    0, len(requested) - len(result["executed"]))
        self._end_tool(ident, "step", args, result)
        if self.done:
            self.finish()
        return result

    def status(self):
        return {
            "episode_id": self.episode_id,
            "executed_actions": len(self.actions),
            "remaining_actions": self.remaining,
            "last_observation_id": self.current_id,
            "done": self.done,
            "reason": self.end_reason,
        }

    def finish(self):
        if self.finalized:
            return
        self.finalized = True
        integrity_failure = self.end_reason in (
            "post_action_processing_exception",
            "habitat_action_exception", "habitat_action_rejected")
        completed = bool(not integrity_failure and self.actions and (
            self.actions[-1] == "STOP" or len(self.actions) == 500))
        if completed:
            check_rows([{"episode_id": self.episode_id,
                         "actions": self.actions}], [self.episode_id])
        summary = {
            "episode_id": self.episode_id,
            "architecture": "codex_mcp_v1",
            "completed": completed,
            "stop_reason": self.end_reason or "session_ended_without_stop",
            "steps": len(self.actions),
            "move_forward_steps": self.actions.count("MOVE_FORWARD"),
            "turn_steps": self.actions.count("TURN_LEFT") + self.actions.count("TURN_RIGHT"),
            "look_steps": self.actions.count("LOOK_UP") + self.actions.count("LOOK_DOWN"),
            "tool_calls": self.tool_count,
            "seconds": time.monotonic() - self.started,
            "official_metrics_available": False,
        }
        measured = self.evaluator.finish(completed, self.actions)
        summary["simulator_metric_capture"] = dict(
            file="simulator_metrics.json",
            available=measured["available"], complete=measured["complete"],
            official=False)
        atomic_json(self.directory / "summary.json", summary)
        if completed:
            row = {"episode_id": self.episode_id, "actions": self.actions}
            atomic_json(self.directory / "result.json", {
                "row": row,
                "validated_by": "check_rows",
            })
            # Match the original run_api_agent.py root-level diagnostic format.
            # One-episode runs are never full official split submissions.
            if not self.batch_mode:
                atomic_json(self.directory.parents[1] /
                            (self.split + ".diagnostic.json"), {
                            "format_version": 1, "split": self.split,
                            "episodes": [row],
                        })
        else:
            atomic_json(self.directory / "failure.json", {
                "reason": summary["stop_reason"],
                "actions_executed": len(self.actions),
                "not_an_official_submission": True,
            })
        if not self.batch_mode:
            publish_run_metrics(self.directory.parents[1],
                                {self.split: [self.episode_id]})
