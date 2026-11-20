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

from .habitat_adapter import read_current_observation, read_planar_pose
from .perception import Geometry, adapt_observation, depth_image_url
from .replay_capture import ReplayRecorder
from .submission_contract import ACTIONS, atomic_json, check_rows
from .simulator_evaluation import SimulatorMetricCapture, publish_run_metrics


FORWARD_INTERRUPTION_FEEDBACK = {
    "FORWARD_BLOCKED": "Depth shows an obstacle within the next forward body's path. Observe and realign or choose a route around it.",
    "REOBSERVE_FORWARD": "Inspect the current view before continuing forward. Leave clearance and use one step near obstacles.",
    "LOW_FORWARD_PROGRESS": "The last forward action made little progress along its heading. Observe, then realign or reroute before trying again.",
    "NO_VISUAL_PROGRESS": "MOVE_FORWARD left RGB unchanged; choose whether to turn or observe.",
}


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
        # Tool hints use relative turns; the controller keeps robot pose local.
        self._heading_deg = 0.0
        self._last_observed_view = (self._heading_deg, self.pitch_deg)
        self._turn_total_deg = 0.0
        self._turn_net_deg = 0.0
        self._recent_stalls = []
        self._unreviewed_stall_heading = None
        # Grant one extra public Depth frame after a stall; do not
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

    def _refresh_observation(self):
        observation = read_current_observation(self.env, self.camera, self.instruction)
        observation.pitch_deg = self.pitch_deg
        self.current_id = self._save(observation)
        self.current = observation

    def observe(self, *, include_depth=None):
        ident = self._start_tool("observe")
        # Re-render through the public sensor suite; no privileged pose/goal data.
        if not self.done:
            self._refresh_observation()
        frame = Image.fromarray(self.current.rgb)
        buffer = io.BytesIO()
        frame.save(buffer, format="PNG")
        geometry = Geometry(self.current, self.camera, margin=0.0)
        obstacle = geometry.central_forward_obstruction()
        body_risk = geometry.forward_body_risk()
        stall_inspection = self._stall_depth_pending and not self.done
        view = (self._heading_deg % 360, self.pitch_deg)
        depth_reason = None
        if obstacle["confirmed"]:
            depth_reason = "front_obstacle"
        elif body_risk is not None:
            depth_reason = "near_body_risk"
        elif stall_inspection:
            depth_reason = "post_stall_inspection"
        elif include_depth:
            depth_reason = "requested"
        elif not self.done and view != self._last_observed_view:
            depth_reason = "view_changed"
        send_depth = depth_reason is not None
        depth_data = (depth_image_url(self.current, self.camera).split(",", 1)[1]
                      if send_depth else None)
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
            info["depth_reason"] = depth_reason
        if obstacle["confirmed"]:
            info["front_obstacle_m"] = obstacle["nearest_along_m"]
        if body_risk is not None:
            info["near_body_risk"] = body_risk
        if self._recent_stalls and not self.done:
            hint = self._motion_hint(())
            if hint:
                info["motion_hint"] = hint
        arguments = {} if include_depth is None else {"include_depth": include_depth}
        self._end_tool(ident, "observe", arguments, dict(info, status="OK"))
        if stall_inspection:
            self._stall_depth_pending = False
        self._last_observed_view = view
        self._unreviewed_stall_heading = None
        return info, base64.b64encode(buffer.getvalue()).decode("ascii"), depth_data

    def _motion_hint(self, executed):
        """Report local turn history and recent stalled directions."""
        turns = [a for a in executed if a in ("TURN_LEFT", "TURN_RIGHT")]
        batch_total = len(turns) * self.camera.turn_angle
        batch_net = (turns.count("TURN_RIGHT") - turns.count("TURN_LEFT")) * self.camera.turn_angle
        hint = {}
        # The window resets after measured progress or the RGB fallback.
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
        forward = action == "MOVE_FORWARD"
        before_rgb = self.current.rgb if forward else None
        pose_before = read_planar_pose(self.env) if forward else None
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
        progress = None
        interruption = None
        if forward:
            pose_after = read_planar_pose(self.env) if pose_before is not None else None
            if pose_after is not None:
                position, heading = pose_before
                progress = float((pose_after[0] - position) @ heading)
                if progress < self.camera.forward_step * 0.20:
                    interruption = "LOW_FORWARD_PROGRESS"
            elif np.array_equal(before_rgb, observation.rgb):
                interruption = "NO_VISUAL_PROGRESS"
        if action in ("TURN_LEFT", "TURN_RIGHT"):
            angle = self.camera.turn_angle * (1 if action == "TURN_RIGHT" else -1)
            self._heading_deg += angle
            self._turn_net_deg += angle
            self._turn_total_deg += abs(angle)
        elif forward:
            if interruption:
                self._recent_stalls.append((index, self._heading_deg))
                self._recent_stalls = self._recent_stalls[-2:]
                if interruption == "LOW_FORWARD_PROGRESS":
                    self._unreviewed_stall_heading = self._heading_deg % 360
                if not self._stall_depth_used:
                    self._stall_depth_pending = True
                    self._stall_depth_used = True
            else:
                self._turn_total_deg = 0.0
                self._turn_net_deg = 0.0
                self._recent_stalls.clear()
                self._unreviewed_stall_heading = None
                self._stall_depth_pending = False
                self._stall_depth_used = False
        row = {
            "step": index, "action": action, "tool_call_id": tool_id,
            "origin": origin,
            "input_file": "observations/" + before + ".npz",
            "post_file": "observations/" + self.current_id + ".npz",
        }
        if progress is not None:
            row["forward_progress_m"] = round(progress, 4)
        if interruption == "NO_VISUAL_PROGRESS":
            row["motion_feedback"] = "rgb_unchanged"
        elif interruption == "LOW_FORWARD_PROGRESS":
            row["motion_feedback"] = "low_forward_progress"
        self._append("steps.jsonl", row)
        self.replay.action(row)
        if action == "STOP":
            self.done, self.end_reason = True, (
                "budget_guard_stop" if origin == "budget_guard" else "codex_stop")
        elif environment_done:
            self.done, self.end_reason = True, "habitat_terminated_without_stop"
        elif self.remaining == 0:
            self.done, self.end_reason = True, "action_budget_exhausted"
        return interruption, progress

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
                interruption = None
                risk = None
                progress = None
                for action in requested:
                    if self.done:
                        break
                    if action == "MOVE_FORWARD":
                        if self._unreviewed_stall_heading == self._heading_deg % 360:
                            interruption = "REOBSERVE_FORWARD"
                            break
                        if not executed:
                            # Refresh after time spent in the model call. Later
                            # actions use the immediately preceding step's frame.
                            try:
                                self._refresh_observation()
                            except (ValueError, RuntimeError, OSError) as error:
                                raise RuntimeError("Could not refresh RGB-D before MOVE_FORWARD") from error
                        geometry = Geometry(self.current, self.camera, margin=0.0)
                        risk = geometry.forward_body_risk()
                        if risk is not None:
                            obstruction = geometry.forward_body_risk(next_step_only=True)
                            if obstruction is not None:
                                risk = obstruction
                                interruption = "FORWARD_BLOCKED"
                                break
                            if "MOVE_FORWARD" in executed:
                                interruption = "REOBSERVE_FORWARD"
                                break
                    interruption, progress = self._atomic(action, ident, "codex")
                    executed.append(action)
                    if interruption:
                        break  # The remaining actions, including STOP, are not executed.
                requested_executed = len(executed)
                if (not self.done and self.max_steps < 500
                        and self.remaining == 1):
                    self._atomic("STOP", ident, "budget_guard")
                    executed.append("STOP")
                result = self.status()
                result.update(status=("DONE" if self.done else interruption or "RUNNING"),
                              executed=executed, requested=requested)
                if interruption:
                    result["skipped_count"] = len(requested) - requested_executed
                    result["motion_feedback"] = FORWARD_INTERRUPTION_FEEDBACK[interruption]
                    if risk is not None:
                        result["near_body_risk"] = risk
                    if interruption == "LOW_FORWARD_PROGRESS":
                        result["forward_progress_m"] = round(progress, 4)
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
