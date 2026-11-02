"""Regression: PR #13 introduced runtime liveness fields absent from export's schema.

Startup must load the shipped configs/agent.yaml *before* any paid API request,
but remain strict about mistakes and invalid watchdog settings.
"""
import copy
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import yaml

from havln_agent.export import load_config
from havln_agent.scheduler import EvidenceLivenessGuard


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs' / 'agent.yaml'


class ConfigLivenessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        raw = CONFIG.read_text()
        names = set(re.findall(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', raw))
        self.env = {name: ('https://example.invalid' if name.endswith('BASE_URL')
                    else 'test-model' if name.endswith('MODEL') else 'fake-value')
                    for name in names}

    def read(self, config=None):
        if config is None:
            path = CONFIG
        else:
            path = Path(self.tmp.name) / 'agent.yaml'
            path.write_text(yaml.safe_dump(config,sort_keys=False))
        with patch.dict(os.environ,self.env):
            return load_config(path)

    def test_shipped_agent_yaml_accepts_new_liveness_field(self):
        config=self.read()
        runtime=config['runtime']
        self.assertEqual(runtime['max_unchanged_evidence_decisions'],8)
        self.assertEqual(runtime['max_steps'],500)

    def test_legacy_runtime_config_has_backward_compatible_defaults(self):
        config=self.read()
        del config['runtime']['max_unchanged_evidence_decisions']
        validated=self.read(config)
        self.assertNotIn('max_unchanged_evidence_decisions',validated['runtime'])
        memory=type('Memory',(),{})()
        memory.actions=[]
        memory.current=dict(observation_id='frame_0')
        memory.task=type('Task',(),dict(
            current_subgoal_index=0,completed_subgoals=[]))()
        runtime=type('Runtime',(),dict(config=validated['runtime'],memory=memory))()
        guard=EvidenceLivenessGuard(runtime)
        self.assertEqual(guard.limit,8)

    def test_optional_max_model_decisions_is_validated(self):
        config=self.read()
        config['runtime']['max_model_decisions']=210
        validated=self.read(config)
        self.assertEqual(validated['runtime']['max_model_decisions'],210)
        for value in (0,-1,True,'100',1.5):
            bad=copy.deepcopy(config)
            bad['runtime']['max_model_decisions']=value
            with self.subTest(value=value),self.assertRaisesRegex(
                    ValueError,'runtime.max_model_decisions'):
                self.read(bad)

    def test_invalid_liveness_value_rejected(self):
        config=self.read()
        for value in (0,1,-1,True,'eight',1.5):
            bad=copy.deepcopy(config)
            bad['runtime']['max_unchanged_evidence_decisions']=value
            with self.subTest(value=value),self.assertRaisesRegex(
                    ValueError,'runtime.max_unchanged_evidence_decisions'):
                self.read(bad)

    def test_unknown_and_missing_runtime_keys_still_rejected(self):
        config=self.read()
        bad=copy.deepcopy(config)
        bad['runtime']['misspelled_liveness_setting']=8
        with self.assertRaisesRegex(ValueError,'unknown=.*misspelled_liveness_setting'):
            self.read(bad)
        bad=copy.deepcopy(config)
        del bad['runtime']['max_depth_age_seconds']
        with self.assertRaisesRegex(ValueError,'missing=.*max_depth_age_seconds'):
            self.read(bad)


if __name__=='__main__':
    unittest.main()
