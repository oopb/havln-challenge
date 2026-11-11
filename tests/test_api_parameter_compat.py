"""Offline tests for explicit-error Chat Completions parameter adaptation.

No model calls, Habitat assets or external network access.
"""
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from havln_agent.api_client import APIClient, APIError, BudgetExceeded


SCHEMA = {
    'type': 'object',
    'properties': {'ok': {'type': 'boolean'}},
    'required': ['ok'],
    'additionalProperties': False,
}


def settings(base_url='https://api.openai.com/v1', response_format='json_object'):
    return dict(provider='openai_compatible', base_url=base_url,
                api_key_env='COMPAT_TEST_KEY', model='gpt-6-astra',
                response_format=response_format, reasoning='medium',
                temperature=0, max_tokens=900, timeout_seconds=5,
                max_retries=0)


def accepted(content='{"ok":true}'):
    return dict(choices=[dict(message=dict(content=content),
                              finish_reason='stop')], usage={})


def rejected(request, record, field, code='unsupported_parameter', message=None,
             http_status=400):
    reason = message or ("Unknown parameter: '%s'." % field)
    record['error_detail'] = json.dumps(dict(error=dict(
        param=field, code=code, message=reason)))
    raise urllib.error.HTTPError(request.full_url, http_status,
                                 reason, None, None)


class ChatParameterCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict('os.environ', {'COMPAT_TEST_KEY': 'offline-only'})
        env.start()
        self.addCleanup(env.stop)

    def client(self, opts=None, budget=20):
        opts = opts or settings()
        client = APIClient({'navigation_agent': opts, 'visual_perception': opts},
                           self.tmp.name)
        client.begin_episode('offline', budget, Path(self.tmp.name) / 'api.jsonl')
        return client

    def ask(self, client):
        return client.complete('navigation_agent', 'offline_test',
                               'Return only JSON', {}, SCHEMA)

    def test_official_endpoint_uses_modern_token_limit(self):
        c = self.client()
        bodies = []
        c._request = Mock(side_effect=lambda req, t, record:
                          (bodies.append(json.loads(req.data)), accepted())[1])
        self.assertEqual(self.ask(c), {'ok': True})
        self.assertEqual(bodies[0]['max_completion_tokens'], 900)
        self.assertNotIn('max_tokens', bodies[0])
        self.assertEqual(bodies[0]['reasoning_effort'], 'medium')

    def test_legacy_gateway_uses_max_tokens_and_switches_on_400(self):
        c = self.client(settings('https://gateway.example/v1'))
        bodies = []
        def send(req, timeout, record):
            body = json.loads(req.data)
            bodies.append(body)
            if len(bodies) == 1:
                rejected(req, record, 'max_tokens', message=(
                    "Unsupported parameter: 'max_tokens' is not supported with "
                    "this model. Use 'max_completion_tokens' instead."))
            return accepted()
        c._request = Mock(side_effect=send)
        self.assertEqual(self.ask(c), {'ok': True})
        self.assertEqual(c.calls, 2)
        self.assertEqual(bodies[0]['max_tokens'], 900)
        self.assertEqual(bodies[1]['max_completion_tokens'], 900)
        self.assertNotIn('max_tokens', bodies[1])
        self.assertEqual(self.ask(c), {'ok': True})
        self.assertIn('max_completion_tokens', bodies[2])
        self.assertEqual(c.request_modes[
            ('https://gateway.example/v1','gpt-6-astra')]['token_field'],
            'max_completion_tokens')

    def test_modern_field_rejected_by_legacy_model_switches_once(self):
        c = self.client()
        bodies = []
        def send(req, timeout, record):
            body = json.loads(req.data)
            bodies.append(body)
            if len(bodies) == 1:
                rejected(req, record, 'max_completion_tokens')
            return accepted()
        c._request = Mock(side_effect=send)
        self.assertEqual(self.ask(c), {'ok': True})
        self.assertNotIn('max_completion_tokens', bodies[1])
        self.assertEqual(bodies[1]['max_tokens'], 900)

    def test_both_token_fields_rejected_fails_without_unlimited_output(self):
        c = self.client(settings('https://gateway.example/v1'))
        bodies = []
        def send(req, timeout, record):
            body = json.loads(req.data)
            bodies.append(body)
            name = ('max_tokens' if 'max_tokens' in body
                    else 'max_completion_tokens')
            rejected(req, record, name)
        c._request = Mock(side_effect=send)
        with self.assertRaises(APIError):
            self.ask(c)
        self.assertEqual(c.calls, 2)
        self.assertTrue(all('max_tokens' in b or
                            'max_completion_tokens' in b for b in bodies))

    def test_temperature_only_default_is_supported_then_omitted(self):
        c = self.client()
        bodies = []
        def send(req, timeout, record):
            body = json.loads(req.data)
            bodies.append(body)
            if len(bodies) == 1:
                rejected(req, record, 'temperature', code='unsupported_value',
                         message=("Unsupported value: 'temperature' does not "
                                  "support 0.0 with this model. "
                                  "Only the default (1) value is supported."))
            return accepted()
        c._request = Mock(side_effect=send)
        self.assertEqual(self.ask(c), {'ok': True})
        self.assertEqual(bodies[0]['temperature'], 0)
        self.assertNotIn('temperature', bodies[1])
        self.assertEqual(self.ask(c), {'ok': True})
        self.assertNotIn('temperature', bodies[2])
        self.assertEqual(c.request_modes[
            ('https://api.openai.com/v1', 'gpt-6-astra')]['omit_temperature'], True)

    def test_invalid_temperature_error_is_not_masked(self):
        c = self.client()
        def send(req, timeout, record):
            rejected(req, record, 'temperature', code='invalid_value',
                     message='Temperature must be between 0 and 2.')
        c._request = Mock(side_effect=send)
        with self.assertRaises(APIError):
            self.ask(c)
        self.assertEqual(c.calls, 1)
        self.assertEqual(c.request_modes, {})

    def test_json_object_format_rejected_retains_local_json_validation(self):
        c = self.client()
        bodies = []
        def send(req, timeout, record):
            body = json.loads(req.data)
            bodies.append(body)
            if len(bodies) == 1:
                rejected(req, record, 'response_format', code='unsupported_value',
                         message='response_format json_object is not supported')
            return accepted()
        c._request = Mock(side_effect=send)
        self.assertEqual(self.ask(c), {'ok': True})
        self.assertIn('response_format', bodies[0])
        self.assertNotIn('response_format', bodies[1])
        self.assertIn('Return only JSON satisfying this schema',
                      bodies[1]['messages'][0]['content'])

    def test_json_schema_format_rejected_keeps_system_schema(self):
        c = self.client(settings(response_format='json_schema'))
        bodies = []
        def send(req, timeout, record):
            body = json.loads(req.data)
            bodies.append(body)
            if len(bodies) == 1:
                rejected(req, record, 'response_format', code='unsupported_parameter',
                         message='Unsupported parameter: response_format')
            return accepted()
        c._request = Mock(side_effect=send)
        self.assertEqual(self.ask(c), {'ok': True})
        self.assertNotIn('response_format', bodies[1])
        self.assertIn('"ok"', bodies[1]['messages'][0]['content'])
        self.assertIn('Return only JSON satisfying this schema',
                      bodies[1]['messages'][0]['content'])
        self.assertEqual(c.request_modes[
            ('https://api.openai.com/v1','gpt-6-astra')]['omit_response_format'],
            True)

    def test_format_omission_still_rejects_invalid_response(self):
        c = self.client()
        seq = 0
        def send(req, timeout, record):
            nonlocal seq
            seq += 1
            if seq == 1:
                rejected(req, record, 'response_format',
                         message='Unknown parameter: response_format')
            return accepted('{"unexpected":true}')
        c._request = Mock(side_effect=send)
        with self.assertRaises(APIError):
            self.ask(c)
        self.assertEqual(c.calls, 2)

    def test_multi_field_negotiation_is_bounded_and_preserves_payload(self):
        c = self.client(settings('https://gateway.example/v1'))
        bodies = []
        def send(req, timeout, record):
            body = json.loads(req.data)
            bodies.append(body)
            if len(bodies) == 1:
                rejected(req, record, 'max_tokens')
            if len(bodies) == 2:
                rejected(req, record, 'temperature', code='unsupported_value',
                         message='Only the default (1) temperature is supported')
            if len(bodies) == 3:
                rejected(req, record, 'response_format',
                         message='Unknown parameter: response_format')
            if len(bodies) == 4:
                rejected(req, record, 'reasoning_effort')
            return accepted()
        c._request = Mock(side_effect=send)
        self.assertEqual(self.ask(c), {'ok': True})
        self.assertEqual(c.calls, 5)
        self.assertTrue(all(x['messages'] == bodies[0]['messages'] for x in bodies))
        self.assertTrue(all(x['model']=='gpt-6-astra' for x in bodies))
        self.assertEqual(bodies[-1]['max_completion_tokens'], 900)
        self.assertNotIn('temperature', bodies[-1])
        self.assertNotIn('response_format', bodies[-1])
        self.assertNotIn('reasoning_effort', bodies[-1])

    def test_403_and_auth_failure_never_trigger_negotiation(self):
        c = self.client()
        c._request = Mock(side_effect=lambda req, timeout, record:
            rejected(req, record, 'max_completion_tokens', http_status=403))
        with self.assertRaises(APIError):
            self.ask(c)
        self.assertEqual(c.calls, 1)
        self.assertEqual(c.request_modes, {})

    def test_negotiation_is_counted_against_episode_budget(self):
        c = self.client(settings('https://gateway.example/v1'), budget=1)
        c._request = Mock(side_effect=lambda req, timeout, record:
            rejected(req, record, 'max_tokens'))
        with self.assertRaises(BudgetExceeded):
            self.ask(c)
        self.assertEqual(c.calls, 1)

    def test_rejected_changes_are_logged_and_replay_body_hash_is_distinct(self):
        c = self.client(settings('https://gateway.example/v1'))
        def send(req, timeout, record):
            if 'max_tokens' in json.loads(req.data):
                rejected(req, record, 'max_tokens')
            return accepted()
        c._request = Mock(side_effect=send)
        self.ask(c)
        records = [json.loads(row) for row in
                   (Path(self.tmp.name) / 'api.jsonl').read_text().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['request_parameter_adaptation'],
                         'token_field:max_completion_tokens')
        self.assertEqual(records[1]['request_token_field'],
                         'max_completion_tokens')
        self.assertNotEqual(records[0]['request_body_sha256'],
                            records[1]['request_body_sha256'])


if __name__ == '__main__':
    unittest.main()
