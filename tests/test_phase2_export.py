"""Phase 2 dataset, CLI and final single-file submission regression tests."""
import contextlib
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from havln_agent.export import PHASE2_EPISODES, assemble, build_jobs, main
from scripts.export_cma_submission import atomic_json


class Phase2ExportTests(unittest.TestCase):
    def _dataset(self, root, *, count=PHASE2_EPISODES, include_humans=True):
        data = root / 'data'
        source = data / 'HA-R2R/test'
        source.mkdir(parents=True)
        episodes = [
            dict(episode_id=i, scene_id='mp3d/sceneA/sceneA.glb',
                 instruction=dict(instruction_text='Move to the goal'))
            for i in range(1, count + 1)
        ]
        with gzip.open(source / 'test.json.gz', 'wt') as handle:
            json.dump(dict(episodes=episodes), handle)
        scene = data / 'scene_datasets/mp3d/sceneA'
        scene.mkdir(parents=True)
        (scene / 'sceneA.glb').write_bytes(b'fixture')
        (data / 'HAPS2_0').mkdir()
        motions = data / 'Multi-Human-Annotations'
        motions.mkdir()
        (motions / 'human_motion.json').write_text(
            json.dumps({'sceneA': {}} if include_humans else {}))
        return data

    def test_full_test_dataset_without_phase1_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            data = self._dataset(Path(folder))
            jobs, expected, hashes = build_jobs(
                data, ['test'], Path(folder) / 'no-phase1-manifest.json',
                4, phase='phase2')
            self.assertEqual(len(expected['test']), 4)
            self.assertEqual(expected['test'], ['1', '2', '3', '4'])
            self.assertEqual([job['key'] for job in jobs], ['test-sceneA'])
            self.assertEqual(len(hashes['test']), 64)

    def test_test_dataset_rejects_missing_ids_even_in_small_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            data = self._dataset(Path(folder), count=PHASE2_EPISODES - 1)
            with self.assertRaisesRegex(ValueError, 'official phase2 split'):
                build_jobs(data, ['test'], None, 4, phase='phase2')

    def test_disjoint_phase2_ranges_keep_original_ids_and_trim_final_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            data = self._dataset(Path(folder))
            def select(start, limit):
                jobs, expected, _ = build_jobs(
                    data, ['test'], None, limit,
                    phase='phase2', episode_start=start)
                self.assertEqual(
                    sorted(i for job in jobs for i in job['ids']),
                    sorted(expected['test']))
                return expected['test']
            first = select(1, 100)
            second = select(101, 100)
            last = select(3401, 100)
            self.assertEqual(first, [str(i) for i in range(1, 101)])
            self.assertEqual(second, [str(i) for i in range(101, 201)])
            self.assertEqual(last, [str(i) for i in range(3401, 3409)])
            self.assertEqual(len(set(first + second + last)), 208)
            self.assertEqual(select(3405, None), ['3405', '3406', '3407', '3408'])

    def test_episode_start_rejects_out_of_range_and_phase1_usage(self):
        for start in (0, -1, PHASE2_EPISODES + 1):
            with self.subTest(start=start), self.assertRaisesRegex(
                    ValueError, 'Phase 2 --episode-start'):
                build_jobs(None, ['test'], None, 100, 'phase2', start)
        with self.assertRaisesRegex(ValueError, 'only supported for Phase 2'):
            build_jobs(None, ['val_seen'], None, 100, 'phase1', 2)

    def test_episode_start_cli_validation(self):
        for phase, start in (('phase2', '0'), ('phase2', '3409'),
                             ('phase1', '101')):
            with self.subTest(phase=phase, start=start):
                argv = ['run_api_agent.py', '--phase', phase,
                        '--episode-start', start]
                with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(
                        io.StringIO()), self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)

    def test_missing_test_human_motion_fails_before_api_or_simulation(self):
        with tempfile.TemporaryDirectory() as folder:
            data = self._dataset(Path(folder), include_humans=False)
            with self.assertRaisesRegex(ValueError, 'lacks test scenes: sceneA'):
                build_jobs(data, ['test'], None, 4, phase='phase2')

    def test_cli_defaults_to_test_split_only_for_phase2(self):
        with tempfile.TemporaryDirectory() as folder:
            seen = []
            with patch.object(sys, 'argv', ['run_api_agent.py',
                       '--phase', 'phase2', '--output-dir', folder,
                       '--episode-start', '101', '--episode-limit', '4',
                       '--no-auto-archive']), patch(
                       'havln_agent.export.load_config', return_value={}), patch(
                       'havln_agent.export.export',
                       side_effect=lambda args, config: seen.append(args)):
                main()
            self.assertEqual(seen[0].phase, 'phase2')
            self.assertEqual(seen[0].splits, ['test'])
            self.assertEqual(seen[0].episode_limit, 4)
            self.assertEqual(seen[0].episode_start, 101)

    def test_phase2_formal_archive_has_exact_test_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            rows = [dict(episode_id=str(i), actions=['MOVE_FORWARD', 'STOP'])
                    for i in range(1, PHASE2_EPISODES + 1)]
            job = dict(key='test-sceneA', split='test',
                       ids=[str(i) for i in range(1, PHASE2_EPISODES + 1)])
            atomic_json(output / 'shards/test-sceneA.json',
                        dict(fingerprint='v2', rows=rows))
            with patch('havln_agent.export.subprocess.run') as run:
                assemble(output, [job], {'test': job['ids']},
                         'v2', False, phase='phase2')
                run.assert_not_called()
            with ZipFile(output / 'submission.zip') as archive:
                self.assertEqual(archive.namelist(), ['test.json'])
                payload = json.loads(archive.read('test.json'))
            self.assertEqual(payload['split'], 'test')
            self.assertEqual(payload['format_version'], 1)
            self.assertEqual(len(payload['episodes']), PHASE2_EPISODES)

    def test_phase2_partial_run_only_writes_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            rows = [dict(episode_id=str(i), actions=['STOP'])
                    for i in range(1, 5)]
            job = dict(key='test-sceneA', split='test',
                       ids=[str(i) for i in range(1, 5)])
            atomic_json(output / 'shards/test-sceneA.json',
                        dict(fingerprint='partial', rows=rows))
            assemble(output, [job], {'test': job['ids']},
                     'partial', True, phase='phase2')
            self.assertTrue((output / 'test.diagnostic.json').is_file())
            self.assertFalse((output / 'submission.zip').exists())


if __name__ == '__main__':
    unittest.main()
