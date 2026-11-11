"""Offline contract tests for adaptive Chat Completions reasoning negotiation."""
import json
import tempfile
import unittest
import urllib.error
from unittest.mock import Mock, patch

from havln_agent.api_client import (
    APIClient, APIError, BudgetExceeded, model_endpoints,
    unsupported_reasoning_parameter,
)


SCHEMA = {
    'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
    'required': ['ok'], 'additionalProperties': False,
}


def endpoint(url='https://api.openai.com/v1', model='gpt-6-astra', reasoning='medium'):
    value = dict(provider='openai_compatible', base_url=url,
                 api_key_env='TEST_REASONING_KEY', model=model,
                 response_format='json_object', temperature=0,
                 timeout_seconds=10, max_retries=0, max_tokens=200)
    if reasoning is not None:
        value['reasoning'] = reasoning
    return value


def success():
    return dict(choices=[dict(message=dict(content='{"ok":true}'),
                              finish_reason='stop')], model='test', usage={})


def reject_unknown(request, record, name, code=400, message=None):
    text = message or ("Unknown parameter: '%s'." % name)
    record['error_detail'] = json.dumps(dict(error=dict(
        message=text, type='invalid_request_error',
        param=name, code='unknown_parameter')))
    raise urllib.error.HTTPError(request.full_url, code, text, None, None)


class AdaptiveReasoningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict('os.environ', {'TEST_REASONING_KEY': 'test-secret'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.temp.cleanup)

    def client(self, config=None):
        config = config or endpoint()
        client = APIClient({'navigation_agent': config, 'visual_perception': config},
                           self.temp.name)
        client.begin_episode('unit', 20, None)
        return client

    def call(self, client):
        return client.complete('navigation_agent', 'unit_reasoning',
                               'Reply with a JSON object', {}, SCHEMA)

    def test_openai_sends_reasoning_effort_not_thinking(self):
        client = self.client()
        bodies = []
        def run(request, timeout, record):
            bodies.append(json.loads(request.data))
            return success()
        client._request = Mock(side_effect=run)
        self.assertEqual(self.call(client), {'ok': True})
        self.assertEqual(bodies[0]['reasoning_effort'], 'medium')
        self.assertNotIn('thinking', bodies[0])
        self.assertEqual(client.calls, 1)

    def test_unsupported_reasoning_parameter_retries_without_transport_budget(self):
        client = self.client()
        bodies = []
        def run(request, timeout, record):
            body = json.loads(request.data)
            bodies.append(body)
            if len(bodies) == 1:
                reject_unknown(request, record, 'reasoning_effort')
            return success()
        client._request = Mock(side_effect=run)
        self.assertEqual(self.call(client), {'ok': True})
        self.assertEqual(client.calls, 2)
        self.assertIn('reasoning_effort', bodies[0])
        self.assertNotIn('reasoning_effort', bodies[1])
        self.assertNotIn('thinking', bodies[1])
        self.assertEqual(client.reasoning_modes[
            ('https://api.openai.com/v1', 'gpt-6-astra')], 'omit')
        self.assertEqual(self.call(client), {'ok': True})
        self.assertNotIn('reasoning_effort', bodies[2])
        self.assertEqual(client.calls, 3)

    def test_deepseek_boolean_extension_after_explicit_rejection(self):
        client = self.client(endpoint('https://proxy.example/v1', 'deepseek-v4'))
        bodies = []
        def run(request, timeout, record):
            body = json.loads(request.data)
            bodies.append(body)
            if len(bodies) == 1:
                reject_unknown(request, record, 'reasoning_effort')
            return success()
        client._request = Mock(side_effect=run)
        self.assertEqual(self.call(client), {'ok': True})
        self.assertEqual(bodies[1]['thinking'], {'type': 'enabled'})
        self.assertNotIn('reasoning_effort', bodies[1])
        self.assertEqual(client.reasoning_modes[
            ('https://proxy.example/v1', 'deepseek-v4')], 'thinking')
        self.assertEqual(self.call(client), {'ok': True})
        self.assertEqual(bodies[2]['thinking'], {'type': 'enabled'})

    def test_unsupported_thinking_extension_falls_back_to_omit(self):
        client = self.client(endpoint('https://proxy.example/v1', 'deepseek-v4'))
        bodies = []
        def run(request, timeout, record):
            body = json.loads(request.data)
            bodies.append(body)
            if len(bodies) == 1:
                reject_unknown(request, record, 'reasoning_effort')
            if len(bodies) == 2:
                reject_unknown(request, record, 'thinking')
            return success()
        client._request = Mock(side_effect=run)
        self.assertEqual(self.call(client), {'ok': True})
        self.assertEqual(client.calls, 3)
        self.assertEqual(client.reasoning_modes[
            ('https://proxy.example/v1', 'deepseek-v4')], 'omit')
        self.assertNotIn('thinking', bodies[2])
        self.assertNotIn('reasoning_effort', bodies[2])

    def test_bad_reasoning_value_does_not_trigger_adaptation(self):
        client = self.client()
        def run(request, timeout, record):
            record['error_detail'] = json.dumps(dict(error=dict(
                code='invalid_value', param='reasoning_effort',
                message="Unsupported value: 'medium' for reasoning_effort")))
            raise urllib.error.HTTPError(request.full_url, 400,
                                         'invalid_value', None, None)
        client._request = Mock(side_effect=run)
        with self.assertRaises(APIError):
            self.call(client)
        self.assertEqual(client.calls, 1)
        self.assertEqual(client.reasoning_modes, {})

    def test_403_never_triggers_parameter_adaptation(self):
        client = self.client()
        def run(request, timeout, record):
            reject_unknown(request, record, 'reasoning_effort', code=403)
        client._request = Mock(side_effect=run)
        with self.assertRaises(APIError) as caught:
            self.call(client)
        self.assertEqual(caught.exception.request_context['http_status'], 403)
        self.assertEqual(client.calls, 1)
        self.assertEqual(client.reasoning_modes, {})

    def test_api_budget_includes_parameter_negotiation(self):
        client = self.client()
        client.begin_episode('unit', 1, None)
        client._request = Mock(side_effect=lambda req,timeout,record:
            reject_unknown(req, record, 'reasoning_effort'))
        with self.assertRaises(BudgetExceeded):
            self.call(client)
        self.assertEqual(client.calls, 1)

    def test_invalid_configuration_and_legacy_thinking_rejected(self):
        for value in ('disabled', 'maximum', '', 0, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    model_endpoints(endpoint(reasoning=value))
        legacy = endpoint()
        legacy['thinking'] = 'disabled'
        with self.assertRaises(ValueError):
            model_endpoints(legacy)

    def test_no_reasoning_config_preserves_prior_request_shape(self):
        client = self.client(endpoint(reasoning=None))
        bodies = []
        client._request = Mock(side_effect=lambda request, timeout, record:
            (bodies.append(json.loads(request.data)), success())[1])
        self.assertEqual(self.call(client), {'ok': True})
        self.assertNotIn('reasoning_effort', bodies[0])
        self.assertNotIn('thinking', bodies[0])

    def test_plain_text_error_matching_is_strict(self):
        self.assertTrue(unsupported_reasoning_parameter(
            "Unknown parameter: 'reasoning_effort'.", 'reasoning_effort'))
        self.assertFalse(unsupported_reasoning_parameter(
            "Unsupported value: medium for reasoning_effort", 'reasoning_effort'))
        self.assertFalse(unsupported_reasoning_parameter(
            "Unknown parameter: 'temperature'.", 'reasoning_effort'))
        self.assertTrue(unsupported_reasoning_parameter(
            json.dumps({'error': {'param': None,
                                   'message': "Unknown parameter: 'reasoning_effort'."}}),
            'reasoning_effort'))


if __name__ == '__main__':
    unittest.main()
