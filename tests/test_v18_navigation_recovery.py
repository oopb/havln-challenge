"""v18: pure turn, bounded schema regeneration, and camera-grounded route guidance.

Run: python -m unittest discover -s tests -p 'test_v18_navigation_recovery.py' -v
No Habitat, paid API or scene assets required.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from jsonschema import Draft7Validator

from havln_agent.api_client import APIError
from havln_agent.contracts import registry,navigation_schema
from havln_agent.navigation_agent import MISSION_PROMPT
from havln_agent.scheduler import (
    request,run,format_repair_feedback,
    FORMAT_REGENERATIONS_PER_DECISION,MAX_RECOVERABLE_FORMAT_CYCLES)
from havln_agent.visual_perception import SCAN_GUIDANCE_PROMPT
from test_forward_preflight_v9 import camera,test_runtime,geometry


def bad_distance_error():
    error=APIError('invalid_model_response',category='structured_field')
    error.request_context=dict(
        error_detail='navigation_invalid_tool_argument:navigate_relative.distance_m:'
                     '0 is less than the minimum of 0.25',
        validation_field=['args','distance_m'])
    return error


class NavigationOutputV18(unittest.TestCase):
    def test_long_but_useful_visual_evidence_no_longer_retries(self):
        schema=navigation_schema(registry(camera(),3))
        base=dict(
            observed_evidence=['A'*420],
            decision_analysis='Observed a corridor; take a limited move.',
            tool='navigate_relative',
            args=dict(turn_direction='NONE',turn_degrees=0,distance_m=.5))
        self.assertTrue(Draft7Validator(schema).is_valid(base))
        base['observed_evidence']=['A'*601]
        self.assertFalse(Draft7Validator(schema).is_valid(base))

    def test_nav_validator_returns_specific_invalid_distance_field(self):
        rt,_=test_runtime()
        rt.geometry=Mock(return_value=geometry('EXECUTABLE'))
        rt.navigation_agent.decide()
        validate=rt.client.complete.call_args.kwargs['validate']
        invalid=dict(observed_evidence=['Inspect the passage at right.'],
            decision_analysis='Rotate right without moving.',
            tool='navigate_relative',args=dict(
                turn_direction='RIGHT',turn_degrees=30,distance_m=0))
        with self.assertRaisesRegex(APIError,'distance_m'):
            validate(invalid)
        self.assertEqual(invalid['args']['distance_m'],0)

    def test_zero_distance_still_invalid_and_pure_turn_is_valid(self):
        schema=navigation_schema(registry(camera(),3))
        message=dict(
            observed_evidence=['Person ahead, inspect right-hand side.'],
            decision_analysis='Turn right to inspect without moving.',
            tool='navigate_relative',
            args=dict(turn_direction='RIGHT',turn_degrees=30,distance_m=0))
        self.assertFalse(Draft7Validator(schema).is_valid(message))
        message.update(tool='turn_and_observe',
                       args=dict(turn_direction='RIGHT',
                                 turn_degrees=30,look='none'))
        self.assertTrue(Draft7Validator(schema).is_valid(message))
        self.assertIn('distance_m=0',MISSION_PROMPT)

    def test_model_controls_cruise_length_no_forced_minimum(self):
        schema=navigation_schema(registry(camera(),3))
        for distance in (.25,.5,1.0,3.0):
            self.assertTrue(Draft7Validator(schema).is_valid(dict(
                observed_evidence=['Current RGB shows a traversable-looking floor.'],
                decision_analysis='Select a finite, revisable step sequence.',
                tool='navigate_relative',
                args=dict(turn_direction='NONE',turn_degrees=0,
                          distance_m=distance))))
        self.assertIn('each 0.25m atomic move still',MISSION_PROMPT)
        self.assertIn('The model chooses.',registry(camera(),3)[
            'navigate_relative']['description'])

    def test_observation_model_talks_about_passages_not_objects_alone(self):
        self.assertIn('continuous FLOOR/ENTRY corridor',SCAN_GUIDANCE_PROMPT)
        self.assertIn('moving person',SCAN_GUIDANCE_PROMPT)
        self.assertIn('broad object box',SCAN_GUIDANCE_PROMPT)


class FormatRecoveryV18(unittest.TestCase):
    def test_field_specific_regeneration_keeps_original_pure_turn_intent(self):
        error=bad_distance_error()
        feedback=format_repair_feedback(error,error.request_context)
        self.assertEqual(feedback['failed_field'],'args.distance_m')
        self.assertIn('turn_and_observe',feedback['repair_instruction'])
        self.assertIn('No physical action executed',feedback['repair_instruction'])

    def test_request_regenerates_a_broken_pure_turn_without_auto_forward(self):
        rt=SimpleNamespace(
            memory=SimpleNamespace(current=None),
            refresh=Mock(),decision_id=Mock(),event=Mock())
        output=dict(observed_evidence=['Person ahead'],
            decision_analysis='Look to the right',tool='turn_and_observe',
            args=dict(turn_direction='RIGHT',turn_degrees=30,look='none'))
        attempts=[]
        def operation(feedback):
            attempts.append(feedback)
            if len(attempts)<FORMAT_REGENERATIONS_PER_DECISION:
                raise bad_distance_error()
            return output
        recorded=[]
        decision=request(rt,'navigation_agent',operation,
                         lambda role,value:recorded.append(value))
        self.assertEqual(decision,output)
        self.assertEqual(len(attempts),FORMAT_REGENERATIONS_PER_DECISION)
        self.assertIsNone(attempts[0])
        self.assertIn('turn_and_observe',attempts[1]['repair_instruction'])
        self.assertEqual(rt.refresh.call_count,0)
        self.assertEqual(len(recorded),2)

    def test_scheduler_does_not_abort_after_one_exhausted_repair_cycle(self):
        rt,actions=test_runtime()
        rt.client.availability=lambda:'available'
        failed=FORMAT_REGENERATIONS_PER_DECISION
        output=dict(observed_evidence=['Unit test ends after recovery.'],
            decision_analysis='End test without claiming success.',
            tool='request_finish',args=dict(
                outcome='aborted',evidence='Test completed.'))
        rt.navigation_agent=SimpleNamespace(
            decide=Mock(side_effect=[
                *(bad_distance_error() for _ in range(failed)),
                output]))
        logs=[]
        run(rt,lambda role,item:logs.append(item))
        self.assertEqual(rt.stop_reason,'model_aborted')
        self.assertEqual(actions,['STOP'])
        self.assertEqual(rt.navigation_agent.decide.call_count,failed+1)
        self.assertTrue(any(v.get('event')=='navigation_format_cycle_recovered'
                            for v in logs if isinstance(v,dict)))

    def test_infinite_invalid_decisions_have_a_finite_abort_boundary(self):
        rt,actions=test_runtime()
        rt.client.availability=lambda:'available'
        bad_count=FORMAT_REGENERATIONS_PER_DECISION*(
            MAX_RECOVERABLE_FORMAT_CYCLES+1)
        rt.navigation_agent=SimpleNamespace(
            decide=Mock(side_effect=[bad_distance_error()
                                     for _ in range(bad_count)]))
        logs=[]
        run(rt,lambda role,item:logs.append(item))
        self.assertEqual(rt.navigation_agent.decide.call_count,bad_count)
        self.assertEqual(rt.stop_reason,'api_error')
        self.assertEqual(actions,['STOP'])
        self.assertEqual(sum(
            v.get('event')=='navigation_format_cycle_recovered'
            for v in logs if isinstance(v,dict)),
            MAX_RECOVERABLE_FORMAT_CYCLES)


if __name__=='__main__':
    unittest.main()
