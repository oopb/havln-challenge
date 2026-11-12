"""Offline integration tests for navigation's native and staged tool contracts.

These tests use real schema validators and a minimal Runtime facade. They do
not require Habitat, GPUs, provider keys, or remote model responses.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from jsonschema import Draft7Validator, ValidationError

from havln_agent.api_client import APIError
from havln_agent.contracts import registry
from havln_agent.navigation_agent import NavigationAgent


def decision(tool='navigate_relative', args=None):
    item=dict(observed_evidence=['Current view shows a corridor.'],
              decision_analysis='Continue toward the next corridor.',
              tool=tool)
    if args is not None:
        item['args']=args
    return item


class FakeClient:
    def __init__(self, staged, responses):
        self.staged=staged
        self.responses=list(responses)
        self.calls=[]

    def needs_navigation_staging(self, role, generation_schema):
        self.seen_union=generation_schema
        return self.staged

    def mark_navigation_schema_unreliable(self, role, generation_schema):
        self.marked_schema=generation_schema
        self.staged=True

    def complete(self, role, purpose, prompt, context, schema, **kwargs):
        self.calls.append(dict(role=role,purpose=purpose,prompt=prompt,
                               context=context,schema=schema,kwargs=kwargs))
        item=self.responses.pop(0)
        if isinstance(item,Exception):
            raise item
        try:
            Draft7Validator(schema).validate(item)
        except ValidationError as error:
            raise APIError(str(error),category='structured_field') from None
        if kwargs.get('validate'):
            kwargs['validate'](item)
        return item


def fake_runtime(client):
    memory=SimpleNamespace(
        current=dict(observation_id='obs_0000'),
        observations=[dict(observation_id='obs_0000')])
    events=[]
    runtime=SimpleNamespace(
        client=client,memory=memory,navigation_mode='OBSERVE',
        observation=None,events=events,
        make_registry=lambda:registry(None,100),
        forward_preflight=lambda stage:{'status':'UNKNOWN'},
        motion_controller=SimpleNamespace(pose=lambda:dict(heading=0.0)),
        event=lambda event,**kwargs:events.append(dict(event=event,**kwargs)))
    return runtime


def stub_context():
    return {'memory':dict(
        contract='mission_evidence_v1',observation_tasks=[],
        historical_frames=[],blocked_at_source_stances=[],
        progress=dict(model_revision_count=0))}


class NavigationStructuredProtocolTests(unittest.TestCase):
    def decide(self, client):
        runtime=fake_runtime(client)
        with (patch('havln_agent.navigation_agent.navigation_context',
                    return_value={}),
              patch('havln_agent.navigation_agent.cost_optimized_navigation_context',
                    side_effect=lambda _:stub_context()),
              patch('havln_agent.navigation_agent.project_model_angles',
                    side_effect=lambda context,angle:context)):
            result=NavigationAgent(runtime).decide()
        return result,runtime

    def test_native_union_returns_complete_command_in_one_request(self):
        args=dict(turn_direction='LEFT',turn_degrees=90,distance_m=0.5)
        client=FakeClient(False,[decision(args=args)])
        result,runtime=self.decide(client)
        self.assertEqual(result['args'],args)
        self.assertEqual(len(client.calls),1)
        call=client.calls[0]
        self.assertEqual(call['purpose'],'navigation_decision')
        self.assertEqual(set(call['kwargs']['generation_schema']),
                         {'oneOf'})
        self.assertFalse(any(e['event']=='navigation_tool_schema_staging'
                             for e in runtime.events))

    def test_backend_without_union_uses_two_typed_model_decisions(self):
        selection=decision(tool='navigate_relative')
        args=dict(turn_direction='RIGHT',turn_degrees=45,distance_m=0.25)
        client=FakeClient(True,[selection,args])
        result,runtime=self.decide(client)
        self.assertEqual(result,dict(selection,args=args))
        self.assertEqual([c['purpose'] for c in client.calls],[
            'navigation_tool_selection','navigation_tool_arguments'])
        self.assertNotIn('args',client.calls[0]['schema']['properties'])
        spec=registry(None,100)['navigate_relative']['parameters']
        self.assertEqual(client.calls[1]['schema'],spec)
        self.assertEqual(client.calls[1]['context']['selected_tool']['name'],
                         'navigate_relative')
        self.assertEqual(client.calls[1]['context']['selected_tool']
                         ['prior_decision'],selection)
        self.assertEqual(runtime.events[-1]['reason'],
                         'native_union_unavailable')

    def test_invalid_union_result_recovers_with_typed_generation(self):
        selection=decision(tool='turn_and_observe')
        args=dict(turn_direction='NONE',turn_degrees=0,look='down')
        client=FakeClient(False,[
            APIError('missing turn_direction',category='structured_field'),
            selection,args])
        result,runtime=self.decide(client)
        self.assertEqual(result['tool'],'turn_and_observe')
        self.assertEqual(result['args'],args)
        self.assertEqual([c['purpose'] for c in client.calls],[
            'navigation_decision','navigation_tool_selection',
            'navigation_tool_arguments'])
        self.assertEqual(runtime.events[-1]['reason'],
                         'native_union_response_invalid')
        self.assertTrue(client.staged)
        self.assertEqual(client.marked_schema,client.seen_union)

    def test_invalid_staged_action_is_rejected_not_autofilled(self):
        selection=decision(tool='navigate_relative')
        bad_args=dict(turn_degrees=90,distance_m=0.25)
        client=FakeClient(True,[selection,bad_args])
        with self.assertRaises(APIError):
            self.decide(client)
        self.assertEqual(len(client.calls),2)
        self.assertNotIn('turn_direction',bad_args)

    def test_second_stage_never_accepts_fields_from_other_tools(self):
        selection=decision(tool='navigate_relative')
        bad_args=dict(turn_direction='LEFT',turn_degrees=45,
                      distance_m=.25,look='down',record_id='obs_0000')
        client=FakeClient(True,[selection,bad_args])
        with self.assertRaises(APIError):
            self.decide(client)
        self.assertEqual(len(client.calls),2)


if __name__ == '__main__':
    unittest.main()
