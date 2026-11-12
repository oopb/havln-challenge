"""Offline regressions for phase2-test-8 DeepSeek format compatibility.

All samples represent observed failure shapes, not simulated navigation success.
No paid API, image, or Habitat instance is required.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml
from jsonschema import Draft7Validator

from havln_agent.api_client import (
    APIClient, APIError, decode_response, next_reasoning_mode,
)
from havln_agent.contracts import navigation_envelope, navigation_schema, registry
from havln_agent.visual_perception import SCAN_GUIDANCE_SCHEMA
from test_api_parameter_compat import accepted, rejected, settings


TOOLS = registry(None, 100)
NAV_SCHEMA = navigation_schema(TOOLS)
NAV_ENVELOPE = navigation_envelope(TOOLS)


def nav_decision(**updates):
    result = {
        'observed_evidence': ['Current RGB shows a descending staircase.'],
        'decision_analysis': 'Continue toward the lower landing.',
        'tool': 'navigate_relative',
        'args': {'turn_direction': 'NONE', 'turn_degrees': 0,
                 'distance_m': 0.5},
    }
    result.update(updates)
    return result


def vision_decision():
    return {
        'observation_summary': 'A doorway and floor continue ahead.',
        'evidence': 'Current RGB shows a doorway; Depth is uncertain.',
        'target_visible': False,
        'target_type': 'tiled stair-foot/foyer opening leading toward the bedroom doorway (bed, nightstands, dresser, seated reader not yet in view)',
        'target_kind': 'ENTRY',
        'confidence': 'medium',
        'box': None,
        'passage_box': None,
        'uncertainty': 'Whether the earlier bedroom doorway lies beyond this tiled area is not confirmed. The threshold appears level, but clearance beyond the opening remains uncertain. The moving reader need not be visible in this particular frame. Absence of a person is not evidence of a different room; compare the fixed doorway and floor connections from the actual image before changing the route.',
        'navigation_suggestion': 'Observe the bedroom doorway past the foyer.',
        'reason': 'The doorway and floor are visible.',
        'expected_progress': 'ADVANCE',
        'candidate_approach': False,
        'next_action': 'finish',
        'turn_deg': 0,
    }


class DeepSeekFormatCompatibility(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        env = patch.dict('os.environ', {'COMPAT_TEST_KEY': 'offline-only'})
        env.start()
        self.addCleanup(env.stop)

    def client(self, raw=None, response=None):
        opts = settings('https://api.deepseek.com/v1')
        opts.update(reasoning='low', temperature=None, max_retries=0)
        c = APIClient({'navigation_agent': opts,
                       'visual_perception': opts}, self.temp.name)
        c.begin_episode('phase2-test-8', 12,
                        Path(self.temp.name) / 'api.jsonl')
        if response is None:
            response = accepted(raw)
        c._request = Mock(return_value=response)
        return c

    @staticmethod
    def ask_navigation(c):
        return c.complete(
            'navigation_agent', 'navigation_decision',
            'Return one legal decision', {}, NAV_SCHEMA,
            validation_schema=NAV_ENVELOPE,
            validate=lambda item: Draft7Validator(NAV_SCHEMA).validate(item),
        )

    @staticmethod
    def ask_visual(c):
        return c.complete(
            'visual_perception', 'vlm_guided_observation',
            'Describe one image', {}, SCAN_GUIDANCE_SCHEMA)

    def test_echoes_of_input_state_are_ignored_without_mutating_action(self):
        original = nav_decision(
            scene_goal='Descend staircase and enter the bedroom.',
            scene_goal_source_observation_id='obs_0004',
            scene_goal_is_model_selected_not_verified=True)
        c = self.client(json.dumps(original))
        output = self.ask_navigation(c)
        self.assertNotIn('scene_goal_source_observation_id', output)
        self.assertNotIn('scene_goal_is_model_selected_not_verified', output)
        self.assertEqual(output['scene_goal'], original['scene_goal'])
        self.assertEqual(output['tool'], original['tool'])
        self.assertEqual(output['args'], original['args'])
        self.assertEqual(c.calls, 1)
        audit = json.loads((Path(self.temp.name) / 'api.jsonl').read_text().splitlines()[-1])
        self.assertEqual(audit['ignored_non_executable_echo_fields'], [
            'scene_goal_is_model_selected_not_verified',
            'scene_goal_source_observation_id'])
        self.assertTrue(audit['success'])

    def test_two_object_explanation_then_one_action_merges_safely(self):
        sample = nav_decision(progress_narrative='Continue to the stair foot.')
        metadata = {k: v for k, v in sample.items() if k not in ('tool', 'args')}
        command = {k: v for k, v in sample.items() if k in ('tool', 'args')}
        raw = json.dumps(metadata) + '\n' + json.dumps(command)
        c = self.client(raw)
        result = self.ask_navigation(c)
        self.assertEqual(result, sample)
        self.assertEqual(c.calls, 1)
        audit = json.loads((Path(self.temp.name) / 'api.jsonl').read_text().splitlines()[-1])
        self.assertEqual(audit['response_normalization'],
                         'navigation_split_metadata_and_action')

    def test_two_actions_or_arbitrary_json_are_never_merged(self):
        original = nav_decision()
        action = {'tool': original['tool'], 'args': original['args']}
        for raw in (
            json.dumps(original) + '\n' + json.dumps(action),
            json.dumps(action) + '\n' + json.dumps(action),
            json.dumps({'observed_evidence': original['observed_evidence'],
                        'decision_analysis': original['decision_analysis'],
                        'hidden_action': 'STOP'}) + '\n' + json.dumps(action),
            json.dumps({'observed_evidence': original['observed_evidence'],
                        'decision_analysis': original['decision_analysis']}) +
            '\n' + json.dumps(action) + '\n' + json.dumps(action),
        ):
            with self.subTest(raw=raw[:100]):
                with self.assertRaises(APIError):
                    self.ask_navigation(self.client(raw))

    def test_nonwhitelisted_extra_field_does_not_disappear(self):
        original = nav_decision(unknown_policy_override='turn left anyway')
        c = self.client(json.dumps(original))
        with self.assertRaises(APIError) as cm:
            self.ask_navigation(c)
        context = cm.exception.request_context
        self.assertIn('unknown_policy_override', context['error_detail'])
        self.assertIn('unknown_policy_override', context['validation_field'])

    def test_bad_executable_arguments_are_never_repaired(self):
        for args in (
            {'turn_direction': 'NONE', 'turn_degrees': 0, 'distance_m': 0},
            {'turn_direction': 'WRONG', 'turn_degrees': 45, 'distance_m': 0.5},
            {'turn_direction': 'RIGHT', 'turn_degrees': 190, 'distance_m': 0.5},
        ):
            with self.subTest(args=args):
                decision = nav_decision(
                    scene_goal_source_observation_id='obs_0004',
                    args=args)
                with self.assertRaises(APIError):
                    self.ask_navigation(self.client(json.dumps(decision)))

    def test_realistic_verbose_explanations_preserved_not_clipped(self):
        sample = nav_decision(
            observed_evidence=['Evidence %d: source-backed RGB observation.' % i
                               for i in range(6)],
            decision_analysis='Observed image and historical route evidence. ' * 16)
        self.assertGreater(len(sample['decision_analysis']), 700)
        self.assertLess(len(sample['decision_analysis']), 1000)
        c = self.client(json.dumps(sample))
        self.assertEqual(self.ask_navigation(c), sample)
        self.assertEqual(c.calls, 1)

    def test_visual_verbose_descriptions_preserved(self):
        sample = vision_decision()
        self.assertGreater(len(sample['target_type']), 80)
        self.assertGreater(len(sample['uncertainty']), 300)
        c = self.client(json.dumps(sample))
        self.assertEqual(self.ask_visual(c), sample)
        self.assertEqual(c.calls, 1)

    def test_duplicate_keys_remain_rejected(self):
        action = '{"tool":"navigate_relative","tool":"request_finish","args":{}}'
        with self.assertRaises(ValueError):
            decode_response(action)

    def test_both_role_defaults_are_low_and_not_upgraded_by_boolean_fallback(self):
        with open('configs/agent.yaml', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        for role in ('navigation_agent', 'visual_perception'):
            self.assertEqual(config[role]['reasoning'], 'low')
        self.assertIsNone(next_reasoning_mode(
            {'reasoning': 'low', 'model': 'deepseek-flash',
             'base_url': 'https://api.deepseek.com/v1'},
            'reasoning_effort'))
        c = self.client()
        bodies = []
        def deny(req, _, rec):
            body = json.loads(req.data)
            bodies.append(body)
            rejected(req, rec, 'reasoning_effort')
        c._request = Mock(side_effect=deny)
        with self.assertRaises(APIError):
            self.ask_navigation(c)
        self.assertEqual(c.calls, 1)
        self.assertEqual(bodies[0]['reasoning_effort'], 'low')
        self.assertNotIn('thinking', bodies[0])


if __name__ == '__main__':
    unittest.main()
