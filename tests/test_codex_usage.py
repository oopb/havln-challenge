"""Offline contracts for Codex usage and rate-limit accounting."""
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from havln_agent.codex_usage import (
    compare_rate_limits, format_quota_summary, read_rate_limits, summarize_token_usage)
from scripts.pack_agent_diagnostics import pack


class CodexUsageTests(unittest.TestCase):
    def test_token_totals_and_cached_input_not_double_counted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            events = [
                {"type": "turn.completed", "usage": {
                    "input_tokens": 120, "cached_input_tokens": 70,
                    "output_tokens": 15, "reasoning_output_tokens": 5}},
                {"type": "item.completed", "item": {"type": "mcp_tool_call"}},
                {"type": "turn.completed", "usage": {
                    "input_tokens": 80, "cached_input_tokens": 20,
                    "output_tokens": 25, "reasoning_output_tokens": 10}},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in events))
            usage = summarize_token_usage(path)
            self.assertEqual(usage["status"], "complete")
            self.assertEqual(usage["turns_completed"], 2)
            self.assertEqual(usage["mcp_tool_calls_completed"], 1)
            self.assertEqual(usage["input_tokens"], 200)
            self.assertEqual(usage["cached_input_tokens"], 90)
            self.assertEqual(usage["non_cached_input_tokens"], 110)
            self.assertEqual(usage["output_tokens"], 40)
            self.assertEqual(usage["total_tokens"], 240)
            self.assertEqual(usage["reasoning_output_tokens"], 15)

    def test_failure_missing_usage_and_truncated_json_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            path.write_text('{"type":"turn.failed"}\n{"type":"turn.completed"}\nnot-json\n')
            usage = summarize_token_usage(path)
            self.assertEqual(usage["status"], "unavailable")
            self.assertIsNone(usage["total_tokens"])
            self.assertEqual(usage["turns_failed"], 1)
            self.assertEqual(usage["malformed_events"], 1)
            path.write_text('{"type":"turn.completed","usage":' +
                            '{"input_tokens":10,"output_tokens":4}}\n')
            usage = summarize_token_usage(path)
            self.assertIsNone(usage["cached_input_tokens"])
            self.assertEqual(usage["total_tokens"], 14)
            self.assertEqual(usage["status"], "complete")

    def test_final_quota_summary_shows_real_remaining_and_unknown(self):
        before = {"status": "ok", "limits": {"codex": {
            "secondary": {"used_percent": 21, "remaining_percent": 79,
                          "window_duration_minutes": 10080,
                          "resets_at": 1900000000}}}}
        after = {"status": "ok", "limits": {"codex": {
            "secondary": {"used_percent": 25, "remaining_percent": 75,
                          "window_duration_minutes": 10080,
                          "resets_at": 1900000000}}}}
        quota = {"before": before, "after": after,
                 "delta": compare_rate_limits(before, after)}
        lines = format_quota_summary(quota)
        self.assertEqual(len(lines), 1)
        self.assertIn("weekly", lines[0])
        self.assertIn("remaining: 75%", lines[0])
        self.assertIn("4 pp", lines[0])
        self.assertIn("unavailable", format_quota_summary({
            "after": {"status": "unavailable", "reason": "offline"}})[0])
        # Reset windows must not be attributed as zero quota consumption.
        after["limits"]["codex"]["secondary"]["resets_at"] += 86400
        quota["delta"] = compare_rate_limits(before, after)
        self.assertIn("not comparable", format_quota_summary(quota)[0])

    def test_quota_deltas_are_percentage_points_and_fail_closed_on_reset(self):
        before = {"status": "ok", "limits": {"codex": {
            "primary": {"used_percent": 25, "remaining_percent": 75,
                        "window_duration_minutes": 300, "resets_at": 1900000000}}}}
        after = {"status": "ok", "limits": {"codex": {
            "primary": {"used_percent": 29, "remaining_percent": 71,
                        "window_duration_minutes": 300, "resets_at": 1900000000}}}}
        row = compare_rate_limits(before, after)["by_limit"]["codex"]["primary"]
        self.assertEqual(row["consumed_percentage_points"], 4)
        self.assertEqual(row["status"], "comparable")
        after["limits"]["codex"]["primary"]["resets_at"] += 300
        row = compare_rate_limits(before, after)["by_limit"]["codex"]["primary"]
        self.assertEqual(row["status"], "not_comparable")
        self.assertIsNone(row["consumed_percentage_points"])
        self.assertEqual(compare_rate_limits({"status": "unavailable"}, after)["status"],
                         "unavailable")

    def test_read_only_app_server_handshake_and_no_account_identifiers(self):
        with tempfile.TemporaryDirectory() as temp:
            script = Path(temp) / "codex"
            script.write_text("""#!/usr/bin/env python3
import json,sys
init=json.loads(sys.stdin.readline())
assert init["method"]=="initialize"
print(json.dumps({"id":0,"result":{"userAgent":"test"}}),flush=True)
ready=json.loads(sys.stdin.readline())
assert ready["method"]=="initialized"
query=json.loads(sys.stdin.readline())
assert query["method"]=="account/rateLimits/read"
print(json.dumps({"id":1,"result":{"rateLimits":{
  "limitId":"codex","primary":{"usedPercent":25,
  "windowDurationMins":300,"resetsAt":1900000000},
  "secondary":None},"email":"should-not-be-stored"}}),flush=True)
""")
            script.chmod(0o755)
            snapshot = read_rate_limits(str(script), timeout_seconds=3)
            self.assertEqual(snapshot["status"], "ok")
            self.assertEqual(snapshot["limits"]["codex"]["primary"]["remaining_percent"], 75)
            self.assertNotIn("email", json.dumps(snapshot))
        unavailable = read_rate_limits("/not/a/real/codex", timeout_seconds=1)
        self.assertEqual(unavailable["status"], "unavailable")

    def test_diagnostic_zip_retains_usage_even_when_habitat_never_starts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            (root / "codex").mkdir(parents=True)
            (root / "codex" / "codex_usage.json").write_text('{"tokens":{}}\n')
            (root / "run_status.json").write_text('{"success":false}\n')
            archive = Path(temp) / "run.zip"
            pack(root, archive, frames="none", allow_empty=True)
            with ZipFile(archive) as file:
                self.assertIn("run/codex/codex_usage.json", file.namelist())


if __name__ == "__main__":
    unittest.main()
