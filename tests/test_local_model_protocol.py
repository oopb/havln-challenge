"""Offline protocol regressions for local vLLM and mixed compatible gateways.

No external API requests, simulator or GPU required.
"""
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from jsonschema import Draft7Validator

from havln_agent.api_client import APIClient, APIError
from havln_agent.contracts import (
    navigation_envelope, navigation_generation_schema,
    navigation_schema, registry,
)


def opts(url='http://127.0.0.1:8000/v1'):
    return dict(provider='openai_compatible', base_url=url,
        api_key_env='LOCAL_TEST_KEY', model='havln-local',
        response_format='auto', reasoning=None, prompt_cache='off',
        temperature=0, timeout_seconds=5, max_retries=0, max_tokens=512)


def answer(value):
    return dict(choices=[dict(message=dict(content=json.dumps(value)),
                              finish_reason='stop')], usage={})


def error(request, record, message, param='response_format', status=400):
    record['error_detail'] = json.dumps(dict(error=dict(
        message=message, param=param, code='invalid_request_error')))
    raise urllib.error.HTTPError(request.full_url, status, message, None, None)


SIMPLE = dict(type='object', properties=dict(ok=dict(type='boolean')),
              required=['ok'], additionalProperties=False)


class LocalModelProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        context = patch.dict('os.environ', {'LOCAL_TEST_KEY': 'offline-key'})
        context.start()
        self.addCleanup(context.stop)
        self.client = APIClient(
            {'navigation_agent': opts(), 'visual_perception': opts()},
            self.tmp.name)
        self.client.begin_episode('offline', 40,
                                  Path(self.tmp.name) / 'api.jsonl')

    def ask(self):
        return self.client.complete(
            'navigation_agent', 'protocol_test', 'Output JSON', {}, SIMPLE)

    def test_native_schema_is_first_and_reused_when_supported(self):
        bodies = []
        def handler(req, timeout, rec):
            bodies.append(json.loads(req.data))
            return answer({'ok': True})
        self.client._request = Mock(side_effect=handler)
        self.assertEqual(self.ask(), {'ok': True})
        self.assertEqual(self.ask(), {'ok': True})
        self.assertEqual(len(bodies), 2)
        self.assertTrue(all(b['response_format']['type']=='json_schema'
                            for b in bodies))
        self.assertTrue(all(b['response_format']['json_schema']['schema']==SIMPLE
                            for b in bodies))

    def test_schema_rejection_downgrades_to_object_and_remembers(self):
        bodies = []
        def handler(req, timeout, rec):
            body = json.loads(req.data)
            bodies.append(body)
            if len(bodies)==1:
                error(req, rec, 'response_format json_schema is not supported')
            return answer({'ok': True})
        self.client._request = Mock(side_effect=handler)
        self.assertEqual(self.ask(), {'ok': True})
        self.assertEqual(self.ask(), {'ok': True})
        self.assertEqual([b['response_format']['type'] for b in bodies],
                         ['json_schema','json_object','json_object'])
        self.assertEqual(self.client.request_modes[
            ('http://127.0.0.1:8000/v1','havln-local')]['response_format_mode'],
                         'json_object')
        self.assertIn('Return only JSON satisfying this schema',
                      bodies[1]['messages'][0]['content'])

    def test_vllm_grammar_compilation_error_downgrades_422(self):
        bodies = []
        def handler(req, timeout, rec):
            bodies.append(json.loads(req.data))
            if len(bodies)==1:
                error(req,rec,'Failed to compile JSON schema using xgrammar',
                      param=None,status=422)
            return answer({'ok': True})
        self.client._request = Mock(side_effect=handler)
        self.assertEqual(self.ask(), {'ok': True})
        self.assertEqual(bodies[1]['response_format']['type'],'json_object')

    def test_both_formats_rejected_uses_plain_schema_prompt(self):
        bodies = []
        def handler(req, timeout, rec):
            body = json.loads(req.data)
            bodies.append(body)
            if len(bodies)==1:
                error(req,rec,'response_format json_schema not supported')
            if len(bodies)==2:
                error(req,rec,'response_format json_object not supported')
            return answer({'ok':True})
        self.client._request = Mock(side_effect=handler)
        self.assertEqual(self.ask(), {'ok':True})
        self.assertEqual(self.ask(), {'ok':True})
        self.assertEqual([b.get('response_format',{}).get('type')
                          for b in bodies],
                         ['json_schema','json_object',None,None])
        self.assertIn('Return only JSON satisfying this schema',
                      bodies[3]['messages'][0]['content'])

    def test_unrelated_http_400_and_auth_do_not_downgrade(self):
        for status, field, message in (
            (400,'model','The model does not exist'),
            (400,'max_tokens','max_tokens is invalid'),
            (401,'response_format','json_schema not supported')):
            with self.subTest(status=status,field=field):
                with tempfile.TemporaryDirectory() as tmp:
                    c=APIClient({'navigation_agent':opts(),
                                 'visual_perception':opts()},tmp)
                    c.begin_episode('offline',10,None)
                    c._request=Mock(side_effect=lambda req,t,rec:
                        error(req,rec,message,param=field,status=status))
                    with self.assertRaises(APIError):
                        c.complete('navigation_agent','test','JSON',{},SIMPLE)
                    self.assertEqual(c.calls,1)
                    self.assertEqual(c.request_modes,{})

    def test_native_schema_does_not_replace_local_validation(self):
        self.client._request=Mock(return_value=answer({'bad':True}))
        with self.assertRaises(APIError):
            self.ask()
        self.assertEqual(self.client.calls,1)

    def test_metadata_string_is_normalized_but_missing_analysis_is_not(self):
        tools=registry(None,100)
        schema=navigation_schema(tools)
        envelope=navigation_envelope(tools)
        generation=navigation_generation_schema(tools)
        item=dict(
            observed_evidence=['RGB shows a hallway.'],
            decision_analysis='Advance in clear space.',
            uncertainties='The doorway is not yet confirmed.',
            tool='navigate_relative',
            args=dict(turn_direction='NONE',turn_degrees=0,distance_m=0.25))
        def ask_nav(value):
            self.client._request=Mock(return_value=answer(value))
            return self.client.complete(
                'navigation_agent','navigation_decision',
                'JSON',{},schema,
                validation_schema=envelope, generation_schema=generation,
                validate=lambda item: Draft7Validator(schema).validate(item))
        result=ask_nav(item)
        self.assertEqual(result['uncertainties'],
                         ['The doorway is not yet confirmed.'])
        self.assertEqual(result['args'],item['args'])
        del item['decision_analysis']
        with self.assertRaises(APIError):
            ask_nav(item)

    def test_executable_args_are_never_silently_modified(self):
        tools=registry(None,100)
        schema=navigation_schema(tools)
        envelope=navigation_envelope(tools)
        item=dict(
            observed_evidence=['RGB only.'],
            decision_analysis='Turn toward an opening.',
            tool='navigate_relative',
            args=dict(turn_direction='RIGHT',turn_degrees=190,
                      distance_m=0.25))
        self.client._request=Mock(return_value=answer(item))
        with self.assertRaises(APIError):
            self.client.complete('navigation_agent','navigation_decision',
                'JSON',{},schema,validation_schema=envelope,
                generation_schema=navigation_generation_schema(tools),
                validate=lambda x: Draft7Validator(schema).validate(x))
        self.assertEqual(item['args']['turn_degrees'],190)

    def test_generation_schema_preserves_registry_argument_constraints(self):
        tools=registry(None,100)
        schema=navigation_generation_schema(tools)
        args=schema['properties']['args']
        self.assertEqual(args['additionalProperties'],False)
        self.assertEqual(args['properties']['turn_direction']['enum'],
                         ['LEFT','RIGHT','NONE'])
        self.assertEqual(args['properties']['distance_m']['minimum'],0.25)
        self.assertEqual(args['properties']['outcome']['enum'],
                         ['success','aborted'])
        self.assertEqual(set(schema['required']),
                         {'observed_evidence','decision_analysis','tool','args'})


if __name__ == '__main__':
    unittest.main()
