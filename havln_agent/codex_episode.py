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


class CodexEpisode:
    def __init__(self, env, camera, episode_id, split, max_steps, output_dir,
                 batch_mode=False):
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
        self.replay = ReplayRecorder(self.directory, camera)
        self.actions = []
        self.started = time.monotonic()
        self.observation_count = 0
        self.tool_count = 0
        self.pitch_deg = 0.0
        self.done = False
        self.end_reason = None
        self.finalized = False

        raw = self.env.reset()
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
        # A conservative near-center Depth obstacle is only a reason to
        # show Codex the Depth image, never a permission check for movement.
        obstacle = Geometry(self.current, self.camera, margin=0.0).central_forward_obstruction()
        depth_data = None
        if obstacle["confirmed"]:
            depth_data = depth_image_url(self.current, self.camera).split(",", 1)[1]
        info = {
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "observation_id": self.current_id,
            "executed_actions": len(self.actions),
            "remaining_actions": self.remaining,
            "done": self.done,
            "reason": self.end_reason,
            "depth_sent": depth_data is not None,
        }
        if depth_data is not None:
            info["front_obstacle_m"] = obstacle["nearest_along_m"]
        self._end_tool(ident, "observe", {}, dict(info, status="OK"))
        return info, base64.b64encode(buffer.getvalue()).decode("ascii"), depth_data

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
        if action == "LOOK_UP":
            self.pitch_deg += self.camera.tilt_angle
        elif action == "LOOK_DOWN":
            self.pitch_deg -= self.camera.tilt_angle
        observation = adapt_observation(raw, self.camera)
        observation.pitch_deg = self.pitch_deg
        self.current = observation
        self.current_id = self._save(observation)
        # Only exact pixel identity is a stall: no motion estimation, geometry
        # veto or image-difference threshold. Dynamic scene changes may evade
        # detection; never infer actual displacement from a changed RGB image.
        stalled = before_rgb is not None and np.array_equal(before_rgb, observation.rgb)
        index = len(self.actions)
        self.actions.append(action)
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
        except (ValueError, RuntimeError) as error:
            result = self.status()
            result.update(status="TOOL_ERROR", error=str(error))
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
        completed = bool(self.actions and (
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
