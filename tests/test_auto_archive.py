"""Successful and failed runs preserve diagnostic files automatically."""
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from havln_agent.export import auto_archive


class AutoArchiveTests(unittest.TestCase):
    def test_auto_archive_keeps_all_smoke_frames_and_archives_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'smoke-test'
            observations=output/'traces'/'val_seen-1'/'observations'
            observations.mkdir(parents=True)
            for i in range(3):
                (observations/('obs_%04d.npz' % i)).write_bytes(b'RGBD'+bytes([i]))
            (observations.parent/'steps.jsonl').write_text('')
            (observations.parent/'failure.json').write_text('{"error":"test failure"}')
            first=auto_archive(output,frames='all')
            second=auto_archive(output,frames='all')
            self.assertIsNotNone(first)
            self.assertNotEqual(first,second)
            self.assertTrue(first.is_file())
            self.assertEqual(first.parent,output.parent/'diagnostic-archives')
            self.assertTrue((observations/'obs_0000.npz').exists())
            with ZipFile(first) as zipfile:
                files=zipfile.namelist()
            self.assertEqual(sum(x.endswith('.npz') for x in files),3)
            self.assertIn('smoke-test/traces/val_seen-1/failure.json',files)


    def test_cli_triggers_auto_archive_on_success_and_worker_exception(self):
        import sys
        from unittest.mock import patch
        from havln_agent.export import main
        for failing in (False,True):
            with self.subTest(failing=failing),tempfile.TemporaryDirectory() as temp:
                output=Path(temp)/'smoke'
                def fake_export(args,config):
                    trace=output/'traces'/'val_seen-1'
                    (trace/'observations').mkdir(parents=True)
                    (trace/'observations'/'obs_0000.npz').write_bytes(b'rgbd')
                    (trace/'steps.jsonl').write_text('')
                    (trace/'memory.json').write_text('{"observations":[]}')
                    if failing:
                        (trace/'failure.json').write_text('{"error":"worker crashed"}')
                        raise RuntimeError('worker crashed')
                args=['run_api_agent.py','--output-dir',str(output),
                      '--episode-limit','1']
                with patch.object(sys,'argv',args),patch(
                    'havln_agent.export.load_config',return_value=dict(
                    runtime=dict(max_steps=128))),patch(
                    'havln_agent.export.export',side_effect=fake_export):
                    if failing:
                        with self.assertRaisesRegex(RuntimeError,'worker crashed'):
                            main()
                    else:
                        main()
                archives=list((output.parent/'diagnostic-archives').glob('*.zip'))
                self.assertEqual(len(archives),1)
                with ZipFile(archives[0]) as z:
                    entries=z.namelist()
                self.assertTrue(any(name.endswith('obs_0000.npz') for name in entries))
                self.assertEqual(failing,any(name.endswith('failure.json') for name in entries))

    def test_startup_failure_archives_worker_log_without_episode(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'startup-fail'
            output.mkdir()
            (output/'worker-0.log').write_text('simulator initialization failed')
            archive=auto_archive(output,frames='critical')
            self.assertTrue(archive.is_file())
            with ZipFile(archive) as z:
                self.assertIn('startup-fail/worker-0.log',z.namelist())

    def test_no_trace_does_not_create_spurious_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'empty-run'
            self.assertIsNone(auto_archive(output))
            self.assertFalse((Path(temp)/'diagnostic-archives').exists())


if __name__=='__main__':
    unittest.main()
