"""Live progress works despite quiet workers and exposes no private API data."""
import json
from pathlib import Path
import tempfile
import unittest

from havln_agent.progress_monitor import ProgressMonitor, JSONLCursor


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.now = 1000.0
        self.lines = []
        self.monitor = ProgressMonitor(self.root, [0,1],
            emit=lambda line, **kwargs: self.lines.append(line), clock=lambda: self.now)

    def write(self, name, payload):
        (self.root/name).write_text(json.dumps(payload))

    def append(self, name, payload):
        with (self.root/name).open("a") as target:
            target.write(json.dumps(payload)+"\n")

    def test_initializing_then_running_steps_tools_and_inflight_api(self):
        self.monitor.report()
        self.assertTrue(any("worker starting" in line for line in self.lines))
        self.write("worker-0.progress.json",
            dict(phase="initializing", split="val_seen", episode_id="12",
                 started_at_epoch=990, max_steps=100))
        self.now = 1001
        self.monitor.report()
        self.assertTrue(any("[initializing]" in line for line in self.lines))

        folder = self.root/"traces"/"val_seen-12"
        folder.mkdir(parents=True)
        self.write("worker-0.progress.json",
            dict(phase="running", split="val_seen", episode_id="12",
                 started_at_epoch=990, max_steps=100))
        self.append("traces/val_seen-12/steps.jsonl", dict(action="TURN_LEFT"))
        self.append("traces/val_seen-12/steps.jsonl", dict(action="MOVE_FORWARD"))
        self.append("traces/val_seen-12/tools.jsonl",
                    dict(call=dict(tool="navigate_local"),result=dict(status="BLOCKED")))
        self.append("traces/val_seen-12/api.jsonl",dict(role="navigation_agent",success=True))
        self.write("traces/val_seen-12/api-inflight.json",
                   dict(status="in_progress",role="visual_perception",purpose="scene",
                        request_phase="connect_and_headers",updated_at_epoch=1000,
                        api_key="NEVER_SHOW_SECRET",raw_response="HIDDEN"))
        self.now = 1004
        self.monitor.report()
        latest = self.lines[-1]
        self.assertIn("val_seen-12 [running]",latest)
        self.assertIn("2/100 steps",latest)
        self.assertIn("FWD=1",latest)
        self.assertIn("TURN=1",latest)
        self.assertIn("navigate_local/BLOCKED",latest)
        self.assertIn("API_calls=1",latest)
        self.assertIn("visual_perception/scene:connect_and_headers 4s",latest)
        self.assertNotIn("NEVER_SHOW_SECRET",latest)
        self.assertNotIn("HIDDEN",latest)

        before = len(self.lines)
        self.now=1005
        self.monitor.report()
        self.assertEqual(len(self.lines),before)
        self.now=1020
        self.monitor.report()
        self.assertEqual(len(self.lines),before+2)  # heartbeat for both GPU processes
        self.assertIn("16s",self.lines[-2])

        self.append("traces/val_seen-12/steps.jsonl",dict(action="LOOK_DOWN"))
        self.now=1021
        self.monitor.report()
        self.assertIn("3/100 steps",self.lines[-1])
        self.assertIn("LOOK=1",self.lines[-1])

    def test_partial_jsonl_and_offset_are_safe(self):
        folder = self.root/"steps.jsonl"
        cursor=JSONLCursor()
        folder.write_bytes(b'{"action":"MOVE_FORWARD"')
        self.assertEqual(cursor.consume(folder),[])
        with folder.open("ab") as handle:
            handle.write(b'}\n{"action":"TURN_RIGHT"}\n')
        self.assertEqual(len(cursor.consume(folder)),2)
        self.assertEqual(cursor.count,2)
        self.assertEqual(cursor.consume(folder),[])

    def test_completed_worker_status_is_printed(self):
        self.write("worker-0.progress.json",
            dict(phase="completed",split="val_unseen",episode_id="5",actions=99))
        self.monitor.report()
        self.assertTrue(any("val_unseen-5 [completed]" in line for line in self.lines))


if __name__ == "__main__":
    unittest.main()
