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

    def test_no_trace_does_not_create_spurious_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'empty-run'
            self.assertIsNone(auto_archive(output))
            self.assertFalse((Path(temp)/'diagnostic-archives').exists())


if __name__=='__main__':
    unittest.main()
