"""Batch diagnostics must not silently mix old results or stale success markers."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from havln_agent.export import (
    verify_existing_run_provenance, worker_failure_details, main)
from scripts.export_cma_submission import atomic_json


class BatchFailureDiagnosticsTests(unittest.TestCase):
    def test_saved_shard_from_prior_version_blocks_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            atomic_json(path/'shards/val_seen-scan-part0000.json',
                        dict(fingerprint='old-fingerprint',rows=[]))
            with self.assertRaisesRegex(ValueError,'Mixed evaluation provenance'):
                verify_existing_run_provenance(path,'new-fingerprint')

    def test_saved_episode_with_old_fingerprint_blocks_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            atomic_json(path/'traces/val_seen-3/result.json',
                        dict(fingerprint='old-fingerprint',row=dict(
                            episode_id='3',actions=['STOP'])))
            with self.assertRaisesRegex(ValueError,'NEW --output-dir'):
                verify_existing_run_provenance(path,'new-fingerprint')

    def test_matching_saved_shards_and_episode_are_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            atomic_json(path/'shards/val_seen-scan-part0000.json',
                        dict(fingerprint='same',rows=[]))
            atomic_json(path/'traces/val_seen-1/result.json',
                        dict(fingerprint='same',row=dict(
                            episode_id='1',actions=['STOP'])))
            self.assertIsNone(verify_existing_run_provenance(path,'same'))

    def test_exception_overwrites_stale_finished_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            task=output/'worker-0-b3.json'
            atomic_json(task,dict(worker_id='0-b3',output=str(output),gpu=0))
            atomic_json(output/'worker-0-b3.progress.json',
                        dict(phase='finished',episode_id='3',split='val_seen'))
            with patch.object(sys,'argv',['run_api_agent.py','--worker-task',str(task)]):
                with patch('havln_agent.export.run_worker',
                           side_effect=ValueError('Incomplete episode 3 has API work')):
                    with self.assertRaisesRegex(ValueError,'Incomplete episode 3'):
                        main()
            state=json.loads((output/'worker-0-b3.progress.json').read_text())
            self.assertEqual(state['phase'],'failed')
            self.assertEqual(state['episode_id'],'3')
            self.assertEqual(state['error_type'],'ValueError')
            self.assertIn('Incomplete episode 3',state['error'])

    def test_failed_worker_id_and_log_path_appear_in_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            atomic_json(output/'worker-0-b3.progress.json',
                dict(phase='failed',episode_id='3',
                     error='Incomplete episode 3 has API work'))
            slots=[dict(worker_id='0-b0',gpu=0),dict(worker_id='0-b3',gpu=0)]
            message=worker_failure_details(output,slots,[0,1])
            self.assertIn('worker=0-b3',message)
            self.assertIn('episode=3',message)
            self.assertIn('phase=failed',message)
            self.assertIn('worker-0-b3.log',message)
            self.assertNotIn('worker=0-b0',message)


if __name__=='__main__':
    unittest.main()
