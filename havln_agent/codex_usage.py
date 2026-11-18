"""Codex session token accounting and read-only ChatGPT quota snapshots.

No inference is performed by the quota probe. Token totals come only from the
Codex CLI's structured turn.completed events, not model pricing estimates.
"""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import time


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _count(value):
    return value if type(value) is int and value >= 0 else None


def summarize_token_usage(events_path):
    """Aggregate per-turn CLI usage; never double-count cached input tokens."""
    totals = {"input_tokens": 0, "cached_input_tokens": 0,
              "output_tokens": 0, "reasoning_output_tokens": 0}
    completed = failed = reported = malformed = mcp_calls = 0
    missing_reasoning = missing_cached = 0
    with Path(events_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                malformed += 1
                continue
            if not isinstance(event, dict):
                malformed += 1
                continue
            kind = event.get("type")
            if kind == "turn.failed":
                failed += 1
            if kind == "item.completed" and isinstance(event.get("item"), dict):
                if event["item"].get("type") == "mcp_tool_call":
                    mcp_calls += 1
            if kind != "turn.completed":
                continue
            completed += 1
            usage = event.get("usage")
            if not isinstance(usage, dict):
                continue
            inputs = _count(usage.get("input_tokens"))
            outputs = _count(usage.get("output_tokens"))
            cached = _count(usage.get("cached_input_tokens"))
            reasoning = _count(usage.get("reasoning_output_tokens"))
            if inputs is None or outputs is None:
                continue
            if cached is not None and cached > inputs:
                continue
            totals["input_tokens"] += inputs
            totals["output_tokens"] += outputs
            if cached is None:
                missing_cached += 1
            else:
                totals["cached_input_tokens"] += cached
            if reasoning is None:
                missing_reasoning += 1
            else:
                totals["reasoning_output_tokens"] += reasoning
            reported += 1

    status = ("unavailable" if not reported else
              "complete" if reported == completed and failed == 0 and
              malformed == 0 else "partial")
    return {
        "status": status,
        "source": "codex_exec_json_turn_completed",
        "turns_completed": completed,
        "turns_failed": failed,
        "turns_with_usage": reported,
        "mcp_tool_calls_completed": mcp_calls,
        "malformed_events": malformed,
        "input_tokens": totals["input_tokens"] if reported else None,
        "cached_input_tokens": (totals["cached_input_tokens"]
                                if reported and not missing_cached else None),
        "non_cached_input_tokens": (totals["input_tokens"] - totals["cached_input_tokens"]
                                    if reported and not missing_cached else None),
        "output_tokens": totals["output_tokens"] if reported else None,
        "reasoning_output_tokens": (totals["reasoning_output_tokens"]
                                    if reported and not missing_reasoning else None),
        "total_tokens": (totals["input_tokens"] + totals["output_tokens"]
                         if reported else None),
        "note": "Cached input is already included in input_tokens; reasoning output is included in output_tokens. No per-MCP-call token attribution."
    }


def _window(data):
    if not isinstance(data, dict):
        return None
    value = data.get("usedPercent")
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
        return None
    duration = _count(data.get("windowDurationMins"))
    reset = _count(data.get("resetsAt"))
    return {
        "used_percent": value,
        "remaining_percent": round(100 - value, 4),
        "window_duration_minutes": duration,
        "resets_at": reset,
    }


def _normalize_rate_limits(result):
    if not isinstance(result, dict):
        raise ValueError("Unexpected rate limits response")
    buckets = result.get("rateLimitsByLimitId")
    if not isinstance(buckets, dict) or not buckets:
        single = result.get("rateLimits")
        buckets = {str(single.get("limitId") or "default"): single} if isinstance(single, dict) else {}
    limits = {}
    for limit_id, item in buckets.items():
        if not isinstance(item, dict):
            continue
        windows = {role: parsed for role in ("primary", "secondary")
                   if (parsed := _window(item.get(role))) is not None}
        if windows:
            limits[str(limit_id)] = windows
    if not limits:
        raise ValueError("No usable quota windows in response")
    return limits


def read_rate_limits(codex_bin="codex", timeout_seconds=12):
    """Read account/rateLimits/read over Codex app-server stdio, never infer.

    Only quota percentages, duration and reset timestamps are saved. Account
    identifiers, login credentials and unfiltered RPC responses are discarded.
    Failure is non-fatal to the navigation run.
    """
    snapshot = {"status": "unavailable", "captured_utc": _utc_now(), "limits": {}}
    proc = None
    try:
        proc = subprocess.Popen([codex_bin, "app-server"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                bufsize=0)
        def send(payload):
            proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            proc.stdin.flush()
        send({"method": "initialize", "id": 0, "params": {
            "clientInfo": {"name": "havln_codex_usage", "title": "HA-VLN Codex Usage",
                           "version": "1.0.0"}}})
        deadline = time.monotonic() + timeout_seconds
        selector = selectors.DefaultSelector()
        pending = b""
        initialized = False
        try:
            selector.register(proc.stdout, selectors.EVENT_READ)
            while time.monotonic() < deadline:
                events = selector.select(max(0, deadline - time.monotonic()))
                if not events:
                    break
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                pending += chunk
                if len(pending) > 1024 * 1024:
                    raise ValueError("Excessive app-server response")
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    try:
                        message = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if not isinstance(message, dict):
                        continue
                    if message.get("id") == 0:
                        if "error" in message:
                            raise ValueError("App-server initialization rejected")
                        if not initialized and "result" in message:
                            initialized = True
                            send({"method": "initialized", "params": {}})
                            send({"method": "account/rateLimits/read", "id": 1})
                    elif message.get("id") == 1:
                        if "error" in message:
                            raise ValueError("Rate limits are unavailable for this account")
                        snapshot["limits"] = _normalize_rate_limits(message.get("result"))
                        snapshot["status"] = "ok"
                        snapshot["captured_utc"] = _utc_now()
                        return snapshot
        finally:
            selector.close()
        raise TimeoutError("No usable quota response before deadline")
    except (OSError, ValueError, TimeoutError, BrokenPipeError) as exc:
        snapshot["reason"] = type(exc).__name__ + ": " + str(exc)[:120]
        return snapshot
    finally:
        if proc is not None:
            if proc.stdin:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def compare_rate_limits(before, after):
    """Report percentage-point deltas only for identical, unreset windows."""
    result = {"status": "unavailable", "by_limit": {},
              "note": "Quota deltas are account-wide observations, not per-task billing; concurrent account usage can affect them."}
    if before.get("status") != "ok" or after.get("status") != "ok":
        return result
    for limit_id, old in before["limits"].items():
        new = after["limits"].get(limit_id, {})
        for role, window in old.items():
            next_window = new.get(role)
            if not next_window:
                continue
            same = (window["window_duration_minutes"] is not None
                    and window["resets_at"] is not None
                    and window["window_duration_minutes"] == next_window["window_duration_minutes"]
                    and window["resets_at"] == next_window["resets_at"])
            delta = (next_window["used_percent"] - window["used_percent"]
                     if same else None)
            comparable = delta is not None and delta >= 0
            result["by_limit"].setdefault(limit_id, {})[role] = {
                "status": "comparable" if comparable else "not_comparable",
                "reason": (None if comparable else
                           "window reset/changed or usage decreased"),
                "used_before_percent": window["used_percent"],
                "used_after_percent": next_window["used_percent"],
                "remaining_before_percent": window["remaining_percent"],
                "remaining_after_percent": next_window["remaining_percent"],
                "consumed_percentage_points": round(delta, 4) if comparable else None,
                "window_duration_minutes": window["window_duration_minutes"],
                "resets_at": next_window["resets_at"],
            }
    if any(row["status"] == "comparable" for limits in result["by_limit"].values()
           for row in limits.values()):
        result["status"] = "available"
    else:
        result["status"] = "not_comparable"
    return result
