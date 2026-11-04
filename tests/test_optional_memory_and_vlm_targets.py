"""Bad optional model notes must never veto valid physical navigation."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

from jsonschema import Draft7Validator

from havln_agent.api_client import APIError,decode_response
from havln_agent.contracts import (
    navigation_schema,navigation_envelope,compact_navigation_protocol,registry)
from havln_agent.navigation_agent import NavigationAgent
from havln_agent.memory_service import MemoryService


def tools():
    camera=SimpleNamespace(turn_angle=15,tilt_angle=30)
    return registry(camera,3)


class OptionalMemoryAndVLMTargetTests(unittest.TestCase):
    def test_batch_411_legacy_memory_objects_do_not_reject_valid_move(self):
        allowed={'navigate_relative':tools()['navigate_relative']}
        schema=navigation_schema(allowed)
        envelope=navigation_envelope(allowed)
        examples=[
            '{"tool":"navigate_relative","args":{"turn_deg":0,'
            '"distance_m":0.25},"memory_delta":{"observation_ids":'
            '["obs_0010"],"note":"Stairs confirmed ahead"}}',
            '{"tool":"navigate_relative","args":{"turn_deg":0,'
            '"distance_m":0.25},"memory_delta":[{"observation_id":'
            '"obs_0010","note":"Stairs visible ahead"}]}',
            '{"tool":"navigate_relative","args":{"turn_deg":0,'
            '"distance_m":0.25},"memory_delta":"bad memory side-channel"',
        ]
        for raw in examples:
            with self.subTest(raw=raw):
                result,_=decode_response(raw)
                Draft7Validator(envelope).validate(result)
                Draft7Validator(schema).validate(result)
                self.assertEqual(result['args']['distance_m'],.25)
        self.assertNotIn('Optional memory_delta',compact_navigation_protocol(allowed))
        self.assertIn('Do NOT include memory_delta',compact_navigation_protocol(allowed))

    def test_bad_tool_and_arguments_still_rejected_by_schema(self):
        exposed={'navigate_relative':tools()['navigate_relative']}
        validator=Draft7Validator(navigation_schema(exposed))
        self.assertFalse(validator.is_valid(dict(tool='turn_and_observe',
            args=dict(relative_turn_deg=15,look='none'),
            memory_delta=dict(note='bad'))))
        self.assertFalse(validator.is_valid(dict(tool='navigate_relative',
            args=dict(turn_deg=400,distance_m=.25),
            memory_delta=dict(note='bad'))))
        self.assertFalse(validator.is_valid(dict(tool='navigate_relative',
            args=dict(turn_deg=0,distance_m=99),
            memory_delta=dict(note='bad'))))

    def test_navigation_discards_both_bad_memory_formats_before_execution(self):
        specs=tools()
        for malformed in (
            dict(observation_ids=['obs_0010'],note='Stairs confirmed'),
            [dict(observation_id='obs_0010',note='Stairs visible')]):
            with self.subTest(malformed=malformed):
                model_response=dict(tool='navigate_relative',
                    args=dict(turn_deg=0,distance_m=.25),memory_delta=malformed)
                client=SimpleNamespace(complete=Mock(return_value=model_response))
                rt=SimpleNamespace(client=client,registry=None,
                    make_registry=Mock(return_value=specs),
                    memory=SimpleNamespace(keyframes={},
                        task=SimpleNamespace(instruction='Descend stairs')))
                with patch('havln_agent.navigation_agent.navigation_context',
                           return_value=dict(
                               pose=dict(heading_deg=0),
                               task_selection=dict(current_subgoal_index=0))):
                    actual=NavigationAgent(rt).decide()
                self.assertEqual(actual,dict(tool='navigate_relative',
                    args=dict(turn_deg=0,distance_m=.25)))
                kwargs=client.complete.call_args.kwargs
                kwargs['validate'](model_response)

    def test_active_vlm_target_persists_real_id_pitch_heading_and_no_clearance(self):
        memory=MemoryService.__new__(MemoryService)
        memory.task=SimpleNamespace(current_subgoal_index=1)
        memory.visual_targets=[]
        memory.keyframes={'obs_0010':dict(place_id='P0',
            pitch_deg=-60,heading_deg=90)}
        memory.runtime=SimpleNamespace(motion_controller=SimpleNamespace(
            pose=lambda:dict(heading=math.pi/2,motion_epoch=0,
                viewpoint_generation=0)))
        memory.event=Mock()
        candidate=dict(target_type='stairs',target_kind='ENTRY',
            confidence='high',box=[.23,.56,.72,1.],
            evidence='Visible descending treads',
            uncertainty='walkability not independently verified',
            observation_id='obs_0010',heading_deg=90,pitch_deg=-60)
        result=memory.note_active_observation_target(candidate,
            'Locate accessible stair entry',['stairs'],goal=dict(
                action='descend',target='stairs'))
        self.assertEqual(result['selected_observation_id'],'obs_0010')
        self.assertEqual(result['target_region'],[.23,.56,.72,1.])
        self.assertEqual(result['pitch_deg'],-60)
        self.assertEqual(result['absolute_heading_deg'],90)
        self.assertEqual(result['relative_heading_deg_at_review'],0)
        self.assertEqual(result['target_kind'],'ENTRY')
        self.assertEqual(result['status'],'VISIBLE')
        self.assertEqual(result['evidence_state'],
                         'historical_visual_hypothesis_not_clearance')
        self.assertEqual(len(memory.visual_targets),1)
        same=memory.note_active_observation_target(candidate,
            'Locate accessible stair entry',['stairs'])
        self.assertEqual(same['id'],result['id'])
        self.assertEqual(len(memory.visual_targets),1)

    def test_cue_remains_uncertain_and_fake_id_is_rejected(self):
        memory=MemoryService.__new__(MemoryService)
        memory.task=SimpleNamespace(current_subgoal_index=0)
        memory.visual_targets=[]
        memory.keyframes={'real':dict(place_id='P0',
            pitch_deg=0,heading_deg=0)}
        memory.runtime=SimpleNamespace(motion_controller=SimpleNamespace(
            pose=lambda:dict(heading=0,motion_epoch=0)))
        memory.event=Mock()
        cue=dict(observation_id='real',target_kind='CUE',target_type='stairs',
            confidence='medium',box=[.1,.3,.6,.9],
            heading_deg=0,evidence='Railing only')
        saved=memory.note_active_observation_target(
            cue,'Look for accessible stairs',['stairs'])
        self.assertEqual(saved['status'],'CUE_ONLY')
        self.assertEqual(saved['missing_evidence'],'confirm_actual_walkable_entry')
        with self.assertRaises(ValueError):
            memory.note_active_observation_target(
                dict(cue,observation_id='fake'),'find',['stairs'])


if __name__=='__main__':
    unittest.main()
