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

from havln_agent.api_client import (
    APIClient, APIError, native_generation_schema, schema_fingerprint)
from havln_agent.contracts import (
    navigation_envelope, navigation_generation_schema,
    navigation_choice_schema, navigation_schema, registry,
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

    def test_deepseek_unavailable_format_adapts_with_null_param_and_caches(self):
        """Replays the exact DeepSeek HTTP 400 from api(4).jsonl."""
        bodies = []
        def handler(req, timeout, rec):
            body = json.loads(req.data)
            bodies.append(body)
            if len(bodies) == 1:
                error(req, rec,
                    'This response_format type is unavailable now '
                    '(request_id: d36ffe82-4f95-4a2f-95c6-ed1987041cf9)',
                    param=None, status=400)
            return answer({'ok': True})
        self.client._request = Mock(side_effect=handler)
        self.assertEqual(self.ask(), {'ok': True})
        self.assertEqual(self.ask(), {'ok': True})
        self.assertEqual([b['response_format']['type'] for b in bodies],
                         ['json_schema', 'json_object', 'json_object'])
        self.assertEqual(self.client.request_modes[
            ('http://127.0.0.1:8000/v1', 'havln-local')]['response_format_mode'],
            'json_object')
        self.assertIn('Return only JSON satisfying this schema',
                      bodies[1]['messages'][0]['content'])
        rows = [json.loads(line) for line in
                (Path(self.tmp.name) / 'api.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['validation_error_type'], 'protocol_unsupported')
        self.assertEqual(rows[0]['retry_scheduled'], True)
        self.assertEqual(rows[0]['request_response_format_mode'], 'json_schema')

    def test_deepseek_unavailable_both_formats_falls_back_to_plain_json(self):
        bodies = []
        def handler(req, timeout, rec):
            body = json.loads(req.data)
            bodies.append(body)
            if 'response_format' in body:
                error(req, rec, 'This response_format type is unavailable now',
                      param=None, status=400)
            return answer({'ok': True})
        self.client._request = Mock(side_effect=handler)
        self.assertEqual(self.ask(), {'ok': True})
        self.assertEqual(self.ask(), {'ok': True})
        self.assertEqual([body.get('response_format', {}).get('type')
                          for body in bodies],
                         ['json_schema', 'json_object', None, None])
        self.assertIn('Return only JSON satisfying this schema',
                      bodies[2]['messages'][0]['content'])
        self.assertEqual(self.client.request_modes[
            ('http://127.0.0.1:8000/v1', 'havln-local')]['response_format_mode'],
            'omit')

    def test_deepseek_plain_fallback_still_rejects_invalid_json_schema(self):
        bodies = []
        def handler(req, timeout, rec):
            body = json.loads(req.data)
            bodies.append(body)
            if 'response_format' in body:
                error(req, rec, 'This response_format type is unavailable now',
                      param=None, status=400)
            return answer({'unexpected_tool': 'request_finish'})
        self.client._request = Mock(side_effect=handler)
        with self.assertRaises(APIError):
            self.ask()
        self.assertEqual(len(bodies), 3)
        self.assertNotIn('response_format', bodies[-1])

    def test_unrelated_model_unavailable_does_not_change_format(self):
        bodies = []
        def handler(req, timeout, rec):
            bodies.append(json.loads(req.data))
            error(req, rec, 'This model is unavailable now',
                  param=None, status=400)
        self.client._request = Mock(side_effect=handler)
        with self.assertRaises(APIError):
            self.ask()
        self.assertEqual(len(bodies), 1)
        self.assertEqual(self.client.request_modes, {})

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

    def test_local2_native_schema_excludes_uniqueitems_but_local_enforces_it(self):
        """Actual local-test-2 schema: uniqueItems is a Python-only assertion."""
        tools=registry(None,100)
        full=navigation_envelope(tools)
        generation=navigation_generation_schema(tools)
        native=native_generation_schema(generation)
        source=full['properties']['evidence_observation_ids']
        self.assertTrue(source['uniqueItems'])
        branches={b['properties']['tool']['const']:b
                  for b in generation['oneOf']}
        native_branches={b['properties']['tool']['const']:b
                         for b in native['oneOf']}
        self.assertEqual(set(branches),set(tools))
        for name in tools:
            self.assertTrue(branches[name]['properties']
                            ['evidence_observation_ids']['uniqueItems'])
            self.assertNotIn('uniqueItems',
                native_branches[name]['properties']['evidence_observation_ids'])
            self.assertEqual(branches[name]['required'],full['required'])
        self.assertEqual(native_branches['navigate_relative']['properties']
                         ['args']['properties']['distance_m']['minimum'],0.25)
        sent=[]
        item=dict(observed_evidence=['Current RGB shows a corridor.'],
            decision_analysis='Continue through the corridor.',
            evidence_observation_ids=['obs_0001'],
            tool='navigate_relative',
            args=dict(turn_direction='NONE',turn_degrees=0,distance_m=0.25))
        def respond(req,timeout,record):
            sent.append(json.loads(req.data))
            return answer(item)
        self.client._request=Mock(side_effect=respond)
        def call():
            return self.client.complete(
                'navigation_agent','navigation_decision','JSON',{},
                navigation_schema(tools),validation_schema=full,
                generation_schema=generation,
                validate=lambda value: Draft7Validator(
                    navigation_schema(tools)).validate(value))
        self.assertEqual(call(),item)
        wire=sent[0]['response_format']['json_schema']['schema']
        self.assertEqual(set(wire),{'oneOf'})
        self.assertTrue(all('uniqueItems' not in
            branch['properties']['evidence_observation_ids']
            for branch in wire['oneOf']))
        self.assertEqual(sent[0]['response_format']['type'],'json_schema')
        item['evidence_observation_ids']=['obs_0001','obs_0001']
        with self.assertRaises(APIError):
            call()
        self.assertEqual(len(sent),2)
        self.assertTrue(full['properties']['evidence_observation_ids']['uniqueItems'])

    def test_schema_projection_does_not_delete_property_names_or_enum_data(self):
        original=dict(type='object', properties={
            'uniqueItems':dict(type='string'),
            'ids':dict(type='array',uniqueItems=True,
                       items=dict(type='string')),
            'flags':dict(enum=[{'uniqueItems': True}])})
        projected=native_generation_schema(original)
        self.assertIn('uniqueItems',projected['properties'])
        self.assertNotIn('uniqueItems',projected['properties']['ids'])
        self.assertEqual(projected['properties']['flags']['enum'],
                         [{'uniqueItems': True}])
        self.assertTrue(original['properties']['ids']['uniqueItems'])

    def test_exact_local2_grammar_400_downgrades_only_format(self):
        """Replay the actual vLLM 0.33/XGrammar error, including null param."""
        sent=[]
        original_error=dict(error=dict(
            message='Grammar error: Unimplemented keys: ["uniqueItems"]',
            type='BadRequestError',param=None,code=400))
        def respond(req,timeout,record):
            body=json.loads(req.data)
            sent.append(body)
            if len(sent)==1:
                record['error_detail']=json.dumps(original_error)
                raise urllib.error.HTTPError(req.full_url,400,
                    'Bad Request',None,None)
            return answer({'ok':True})
        self.client._request=Mock(side_effect=respond)
        self.assertEqual(self.ask(),{'ok':True})
        self.assertEqual([x['response_format']['type'] for x in sent],
                         ['json_schema','json_object'])
        self.assertEqual(sent[0]['model'],sent[1]['model'])
        self.assertEqual(sent[0]['messages'][1],sent[1]['messages'][1])
        self.assertEqual(self.client.request_modes[
            ('http://127.0.0.1:8000/v1','havln-local')]['schema_format_modes']
            [schema_fingerprint(SIMPLE)],'json_object')
        self.assertEqual(self.ask(),{'ok':True})
        self.assertEqual(sent[2]['response_format']['type'],'json_object')

    def test_union_grammar_failure_keeps_native_schema_for_typed_arguments(self):
        """Union-only XGrammar limitation must not disable all JSON Schema."""
        tools=registry(None,100)
        union=navigation_generation_schema(tools)
        full=navigation_schema(tools)
        envelope=navigation_envelope(tools)
        result=dict(observed_evidence=['Visible corridor.'],
                    decision_analysis='Go to the corridor.',
                    tool='navigate_relative',
                    args=dict(turn_direction='NONE',turn_degrees=0,
                              distance_m=0.25))
        sent=[]
        def respond(req,timeout,record):
            payload=json.loads(req.data)
            sent.append(payload)
            if len(sent)==1:
                record['error_detail']=json.dumps(dict(error=dict(
                    message='Grammar error: Unimplemented keys: ["oneOf"]',
                    param=None,code=400)))
                raise urllib.error.HTTPError(req.full_url,400,
                    'Bad Request',None,None)
            if len(sent)==2:
                return answer(result)
            return answer({'ok':True})
        self.client._request=Mock(side_effect=respond)
        received=self.client.complete(
            'navigation_agent','navigation_decision','JSON',{},full,
            validation_schema=envelope, generation_schema=union,
            validate=lambda x:Draft7Validator(full).validate(x))
        self.assertEqual(received,result)
        self.assertEqual([r['response_format']['type'] for r in sent[:2]],
                         ['json_schema','json_object'])
        self.assertTrue(self.client.needs_navigation_staging(
            'navigation_agent',union))
        self.assertFalse(self.client.needs_navigation_staging(
            'navigation_agent',SIMPLE))
        self.assertEqual(self.ask(),{'ok':True})
        self.assertEqual(sent[2]['response_format']['type'],'json_schema')
        # No global response_format downgrade was recorded.
        modes=self.client.request_modes[
            ('http://127.0.0.1:8000/v1','havln-local')]
        self.assertNotIn('response_format_mode',modes)

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

    def test_generation_schema_binds_tool_to_its_own_required_args(self):
        tools=registry(None,100)
        schema=navigation_generation_schema(tools)
        self.assertEqual(set(schema),{'oneOf'})
        self.assertEqual(len(schema['oneOf']),len(tools))
        branches={branch['properties']['tool']['const']:branch
                  for branch in schema['oneOf']}
        self.assertEqual(set(branches),set(tools))
        for name,spec in tools.items():
            branch=branches[name]
            self.assertIn('tool',branch['required'])
            self.assertIn('args',branch['required'])
            self.assertEqual(branch['properties']['args'],spec['parameters'])
            self.assertFalse(branch['additionalProperties'])
        args=branches['navigate_relative']['properties']['args']
        self.assertEqual(set(args['required']),
            {'turn_direction','turn_degrees','distance_m'})
        self.assertEqual(set(args['properties']),
            {'turn_direction','turn_degrees','distance_m'})
        self.assertEqual(args['properties']['turn_direction']['enum'],
            ['LEFT','RIGHT','NONE'])
        self.assertEqual(args['properties']['distance_m']['minimum'],0.25)
        self.assertEqual(branches['request_finish']['properties']['args']
                         ['properties']['outcome']['enum'],
                         ['success','aborted'])
        self.assertEqual(set(navigation_choice_schema(tools)['required']),
            {'observed_evidence','decision_analysis','tool'})
        self.assertNotIn('args',navigation_choice_schema(tools)['properties'])

    def test_19_like_tool_argument_failures_are_rejected_by_native_union(self):
        tools=registry(None,100)
        native=native_generation_schema(navigation_generation_schema(tools))
        base=dict(observed_evidence=['Current RGB: landing.'],
                  decision_analysis='Proceed toward stairwell.',
                  tool='navigate_relative')
        good=dict(base,args=dict(
            turn_direction='LEFT',turn_degrees=90,distance_m=0.25))
        self.assertTrue(Draft7Validator(native).is_valid(good))
        invalid=[
            dict(distance_m=0.25,look='down',record_id='obs_0000'),
            dict(turn_degrees=90,distance_m=0.25),
            dict(turn_direction='LEFT',distance_m=0.25),
            dict(turn_direction='LEFT',turn_degrees=90),
            dict(turn_direction='LEFT',turn_degrees=90,
                 distance_m=0.25,look='none'),
            dict(turn_direction='LEFT',turn_degrees=90,distance_m=0),
        ]
        for args in invalid:
            with self.subTest(args=args):
                self.assertFalse(Draft7Validator(native).is_valid(
                    dict(base,args=args)))
        # All branch choices are still valid when the correct args are used.
        for name,spec in tools.items():
            args={}
            for key,field in spec['parameters'].get('properties',{}).items():
                if key not in spec['parameters']['required']:
                    continue
                if 'enum' in field: value=field['enum'][0]
                elif field.get('type')=='number': value=max(field.get('minimum',0),0.25)
                elif field.get('type')=='integer': value=30
                else: value='Observed scene evidence.'
                args[key]=value
            example=dict(base,tool=name,args=args)
            self.assertTrue(Draft7Validator(native).is_valid(example),
                            msg=name)


if __name__ == '__main__':
    unittest.main()
