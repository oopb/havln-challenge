"""Parent-side progress for quiet multi-GPU Habitat workers.

Read only append-only trace deltas and the secret-free API status marker.
Never print raw model prompts, responses, URLs, tokens or API keys.
"""
from collections import Counter
import json
from pathlib import Path
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def duration(seconds):
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s" if hours else f"{minutes}m{seconds:02d}s"


class JSONLCursor:
    def __init__(self):
        self.offset, self.count, self.last = 0, 0, None

    def consume(self, path):
        new = []
        try:
            with path.open("rb") as source:
                source.seek(0, 2)
                if source.tell() < self.offset:
                    self.offset, self.count, self.last = 0, 0, None
                source.seek(self.offset)
                while True:
                    line = source.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    self.offset = source.tell()
                    try:
                        value = json.loads(line)
                    except ValueError:
                        continue
                    self.count += 1
                    self.last = value
                    new.append(value)
        except OSError:
            pass
        return new


class WorkerProgress:
    def __init__(self):
        self.episode = None
        self.steps, self.tools = JSONLCursor(), JSONLCursor()
        self.actions = Counter()
        self.last_signature, self.last_print = None, 0

    def select(self, episode):
        if episode != self.episode:
            self.__init__()
            self.episode = episode


class ProgressMonitor:
    def __init__(self, output, gpu_ids, *, emit=print, clock=time.time):
        self.output, self.emit, self.clock = Path(output), emit, clock
        self.workers = {gpu: WorkerProgress() for gpu in gpu_ids}

    def report(self, heartbeat_seconds=15):
        now = self.clock()
        for gpu, state in self.workers.items():
            marker = read_json(self.output / f"worker-{gpu}.progress.json")
            if marker is None:
                label, signature = "worker starting / initializing", ("startup",)
            else:
                phase = marker.get("phase", "unknown")
                split, eid = marker.get("split", "?"), marker.get("episode_id", "?")
                episode = f"{split}-{eid}"
                if phase not in ("running", "initializing"):
                    label, signature = f"{episode} [{phase}]", (episode, phase)
                else:
                    state.select(episode)
                    directory = self.output / "traces" / episode
                    for row in state.steps.consume(directory / "steps.jsonl"):
                        if isinstance(row, dict) and isinstance(row.get("action"), str):
                            state.actions[row["action"]] += 1
                    state.tools.consume(directory / "tools.jsonl")
                    result = (state.tools.last or {}).get("result", {})
                    call = (state.tools.last or {}).get("call", {})
                    api = read_json(directory / "api-inflight.json") or {}
                    api_status = api.get("status", "not_started")
                    if api_status == "in_progress":
                        age = max(0, int(now - api.get("updated_at_epoch", now)))
                        api_label = (f"{api.get('role','?')}/{api.get('purpose','?')}:"
                                     f"{api.get('request_phase','?')} {age}s")
                    else:
                        api_label = "idle" if api_status == "completed" else "not_started"
                    label = (f"{episode} [{phase}] {state.steps.count}/{marker.get('max_steps','?')} steps "
                             f"FWD={state.actions['MOVE_FORWARD']} "
                             f"TURN={state.actions['TURN_LEFT'] + state.actions['TURN_RIGHT']} "
                             f"LOOK={state.actions['LOOK_UP'] + state.actions['LOOK_DOWN']} "
                             f"tools={state.tools.count} "
                             f"last={call.get('tool','-')}/{result.get('status','-')} "
                             f"API={api_label} "
                             f"elapsed={duration(now-marker.get('started_at_epoch',now))}")
                    signature = (episode, phase, state.steps.count, state.tools.count,
                                 api_status, api.get("role"), api.get("purpose"), api.get("request_phase"))
            if signature != state.last_signature or now-state.last_print >= heartbeat_seconds:
                self.emit(f"[HA-VLN {time.strftime('%H:%M:%S',time.localtime(now))} GPU {gpu}] {label}", flush=True)
                state.last_signature, state.last_print = signature, now
