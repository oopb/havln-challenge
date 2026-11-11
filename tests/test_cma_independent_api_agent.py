"""Regression: API-Agent startup and submission format do not require CMA."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from havln_agent.export import export, main
from havln_agent.submission_contract import (
    ACTIONS, SPLITS, atomic_json, check_rows, sha256,
)


class CmaIndependentAgentTests(unittest.TestCase):
    def test_shared_submission_actions_and_validation_match_existing_contract(self):
        self.assertEqual(ACTIONS, (
            'STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT',
            'LOOK_UP', 'LOOK_DOWN'))
        self.assertEqual(SPLITS, ('val_seen', 'val_unseen'))
        valid = [dict(episode_id='101', actions=['MOVE_FORWARD', 'STOP'])]
        check_rows(valid, ['101'])
        with self.assertRaisesRegex(ValueError, 'action sequence'):
            check_rows([dict(episode_id='101', actions=['TURN_LEFT'])], ['101'])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            check_rows(valid, ['102'])
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'out.json'
            atomic_json(file, dict(episodes=valid))
            self.assertEqual(json.loads(file.read_text())['episodes'], valid)
            self.assertEqual(len(sha256(file)), 64)

    def test_cli_does_not_default_to_cma_source_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            args = []
            with patch.object(sys, 'argv', ['run_api_agent.py', '--phase',
                       'phase2', '--output-dir', directory,
                       '--episode-limit', '1', '--no-auto-archive']), patch(
                       'havln_agent.export.load_config', return_value={}), patch(
                       'havln_agent.export.export',
                       side_effect=lambda opts, conf: args.append(opts)):
                main()
            self.assertIsNone(args[0].source_root)

    def test_export_passes_missing_cma_release_marker_before_api_preflight(self):
        """Regression for the real Phase 2 .release.json FileNotFoundError."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / 'runtime.yaml'
            runtime.write_text('runtime: fake\n')
            annotations = root / 'data/Multi-Human-Annotations/human_motion.json'
            annotations.parent.mkdir(parents=True)
            annotations.write_text('{}')
            # Mock only heavyweight Habitat module imports. The Agent's real
            # code hashes, provenance builder and release-free export run.
            habitat_dir = root / 'habitat'
            simulator = habitat_dir / 'sims/habitat_simulator/habitat_simulator.py'
            simulator.parent.mkdir(parents=True)
            simulator.write_text('# simulator fixture\n')
            habitat = types.ModuleType('habitat')
            habitat.__file__ = str(habitat_dir / '__init__.py')
            habitat.__version__ = '0.1.7'
            modules = dict(habitat=habitat)
            for name in ('habitat_sim', 'numpy', 'PIL', 'jsonschema'):
                module = types.ModuleType(name)
                module.__version__ = 'mock'
                modules[name] = module
            fake_env_class = type('FakeEnv', (), {})
            fake_get_config = lambda path: 'mock-runtime-configuration'
            args = types.SimpleNamespace(
                data_root=root / 'data',
                output_dir=root / 'phase2-smoke',
                source_root=root / 'no-cma-source-checkout',
                runtime_config=runtime,
                manifest=root / 'no-phase1-manifest',
                splits=['test'],
                episode_start=1,
                episode_limit=1,
                phase='phase2',
                seed=0,
                gpu_ids=[0],
                batch=1,
            )
            class PreflightReached(Exception):
                pass
            with patch.dict(sys.modules, modules), patch(
                    'havln_agent.export.runtime_imports',
                    return_value=(fake_get_config, fake_env_class)), patch(
                    'havln_agent.export.build_jobs',
                    return_value=([], {'test': ['1']}, {'test': 'dataset-hash'})), patch(
                    'havln_agent.export.APIClient',
                    side_effect=PreflightReached), patch(
                    'havln_agent.export.subprocess.run',
                    return_value=types.SimpleNamespace(returncode=0, stdout='mock\n')):
                with self.assertRaises(PreflightReached):
                    export(args, {'runtime': {'max_steps': 128}})
            self.assertTrue((args.output_dir / 'cache').exists()
                            or args.output_dir.is_dir())


if __name__ == '__main__':
    unittest.main()
