"""Offline Phase 2 selected-run merge and submission ZIP validation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from scripts.merge_phase2_submissions import (
    TOTAL, collect_runs, main, package, read_run,
)


class Phase2MergeTests(unittest.TestCase):
    def make_run(self, root, name, start, end, *, dataset='same',
                 fingerprint=None, action='STOP', finished=True):
        directory = root / name
        directory.mkdir()
        fp = fingerprint or name
        rows = [dict(episode_id=str(i), actions=[action])
                for i in range(start, end + 1)]
        metadata = dict(
            complete=finished,
            diagnostic=True,
            fingerprint=fp,
            episodes=dict(test=len(rows)),
            identity=dict(phase='phase2', splits=['test'],
                          episode_start=start, episode_limit=end-start+1,
                          datasets=dict(test=dataset)),
        )
        (directory / 'export_metadata.json').write_text(json.dumps(metadata))
        (directory / 'test.diagnostic.json').write_text(json.dumps(
            dict(format_version=1, split='test', episodes=rows)))
        return directory

    def test_select_independent_directories_and_verify_final_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selected = [
                self.make_run(root, f'batch-{i:03d}', start,
                              min(TOTAL, start + 99))
                for i, start in enumerate(range(1, TOTAL + 1, 100), 1)
            ]
            output = root / 'submission'
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(['--run-dirs', *map(str, reversed(selected)),
                               '--output-dir', str(output)])
            self.assertEqual(status, 0)
            with ZipFile(output / 'submission.zip') as archive:
                self.assertEqual(archive.namelist(), ['test.json'])
                self.assertIsNone(archive.testzip())
                payload = json.loads(archive.read('test.json'))
            self.assertEqual(payload['format_version'], 1)
            self.assertEqual(payload['split'], 'test')
            self.assertEqual(len(payload['episodes']), TOTAL)
            self.assertEqual([row['episode_id'] for row in payload['episodes']],
                             [str(i) for i in range(1, TOTAL + 1)])
            self.assertEqual(json.loads((output / 'test.json').read_text()),
                             payload)

    def test_partial_selection_reports_missing_without_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.make_run(root, 'batch-1', 1, 100)
            out = root / 'submission'
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                self.assertEqual(main(['--run-dirs', str(run),
                                       '--check-only']), 0)
            self.assertIn('101-3408', log.getvalue())
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--run-dirs', str(run),
                                       '--output-dir', str(out)]), 1)
            self.assertFalse(out.exists())

    def test_duplicate_episode_rejected_even_for_identical_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.make_run(root, 'first', 1, 100)
            second = self.make_run(root, 'second', 100, 200)
            with self.assertRaisesRegex(ValueError, 'Duplicate Episode 100'):
                collect_runs([first, second])

    def test_duplicate_input_directory_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.make_run(root, 'first', 1, 100)
            with self.assertRaisesRegex(ValueError, 'selected more than once'):
                collect_runs([first, first])

    def test_unfinished_batch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.make_run(Path(temp), 'unfinished', 1, 100, finished=False)
            with self.assertRaisesRegex(ValueError, 'not marked complete'):
                read_run(run)

    def test_mismatched_data_hash_rejected_but_different_model_run_fingerprints_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.make_run(root, 'first', 1, 100, dataset='fileA')
            second = self.make_run(root, 'second', 101, 200, dataset='fileB')
            with self.assertRaisesRegex(ValueError, 'different Phase 2 Test datasets'):
                collect_runs([first, second])
            meta_path = second / 'export_metadata.json'
            meta = json.loads(meta_path.read_text())
            meta['identity']['datasets']['test'] = 'fileA'
            meta_path.write_text(json.dumps(meta))
            with contextlib.redirect_stdout(io.StringIO()):
                rows, missing, runs = collect_runs([first, second])
            self.assertEqual(len(rows), 200)
            self.assertEqual(len(missing), TOTAL - 200)
            self.assertNotEqual(runs[0]['fingerprint'], runs[1]['fingerprint'])

    def test_invalid_actions_and_unexpected_range_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = self.make_run(root, 'bad', 1, 100)
            result = directory / 'test.diagnostic.json'
            original = json.loads(result.read_text())
            original['episodes'][0]['actions'] = ['WAIT']
            result.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'Invalid actions'):
                read_run(directory)
            original['episodes'][0]['actions'] = ['STOP']
            original['episodes'][0]['episode_id'] = '3408'
            result.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'ID coverage'):
                read_run(directory)

    def test_retained_trace_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.make_run(root, 'with-trace', 1, 1)
            trace = run / 'traces/test-1'
            trace.mkdir(parents=True)
            (trace / 'result.json').write_text(json.dumps({
                'fingerprint': 'with-trace',
                'row': dict(episode_id='1', actions=['STOP'])}))
            (trace / 'steps.jsonl').write_text(json.dumps({
                'action': 'TURN_LEFT'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'executed-action log'):
                read_run(run)

    def test_check_existing_output_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            rows = [dict(episode_id=str(i), actions=['STOP'])
                    for i in range(1, TOTAL + 1)]
            with contextlib.redirect_stdout(io.StringIO()):
                package(rows, out)
            with self.assertRaisesRegex(ValueError, '--overwrite'):
                package(rows, out)
            with contextlib.redirect_stdout(io.StringIO()):
                package(rows, out, overwrite=True)
            self.assertTrue((out / 'submission.zip').is_file())

    def test_never_write_output_inside_a_source_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.make_run(root, 'batch-1', 1, TOTAL)
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--run-dirs', str(run),
                                       '--output-dir', str(run / 'submission')]), 1)
            self.assertFalse((run / 'submission').exists())


if __name__ == '__main__':
    unittest.main()
