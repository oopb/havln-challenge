"""v10 camera tool schema and physically recoverable pitch-boundary behavior.

python -m unittest discover -s tests -p 'test_recoverable_camera_pitch.py' -v
No Habitat assets or API keys required.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from jsonschema import Draft7Validator

from havln_agent.contracts import (
    registry, navigation_schema, camera_look_options)
from havln_agent.contexts import navigation_context
from havln_agent.mission_history import mission_evidence
from havln_agent.scheduler import run
from test_forward_preflight_v9 import test_runtime


class CameraToolBoundaryRegression(unittest.TestCase):
    def test_schema_is_stable_even_when_physical_camera_is_at_limit(self):
        r,_=test_runtime()
        for pitch in (-60,0,60):
            with self.subTest(pitch=pitch):
                schema=registry(r.camera,12,pitch)['turn_and_observe'][
                    'parameters']
                self.assertEqual(schema['properties']['look']['enum'],
                                 ['none','up','down'])
                Draft7Validator.check_schema(schema)
                self.assertTrue(Draft7Validator(schema).is_valid(dict(
                    turn_direction='RIGHT',turn_degrees=30,look='down')))
                self.assertTrue(Draft7Validator(schema).is_valid(dict(
                    turn_direction='LEFT',turn_degrees=30,look='up')))
        self.assertEqual(camera_look_options(-60,15),['none','up'])
        self.assertEqual(camera_look_options(60,15),['none','down'])

    def test_lower_boundary_error_is_normal_tool_feedback_and_no_turn(self):
        r,actions=test_runtime()
        r.motion_controller.pitch_deg=-60
        r.refresh()
        result=r.execute(dict(tool='turn_and_observe',
            args=dict(turn_direction='RIGHT',turn_degrees=30,look='down')),
            decision=dict(observation_id=r.memory.current['observation_id'],
                          actor='navigation_agent'))
        self.assertEqual(result['status'],'CAMERA_PITCH_LIMIT')
        self.assertTrue(result['recoverable'])
        self.assertEqual(result['requested_look'],'down')
        self.assertEqual(result['allowed_look'],['none','up'])
        self.assertEqual(result['current_pitch_deg'],-60)
        self.assertEqual(result['requested_pitch_deg'],-75)
        self.assertEqual(result['executed_actions'],0)
        self.assertEqual(result['executed_camera_actions'],[])
        self.assertEqual(actions,[])
        self.assertEqual(r.motion_controller.pitch_deg,-60)
        r.assess(result)
        self.assertFalse(r.done)
        self.assertIsNone(r.stop_reason)
        self.assertEqual(len(r.memory.tool_ledger),1)
        view=mission_evidence(r.memory,r.motion_controller.pose())
        self.assertEqual(view['recent_tool_receipts'][-1]['status'],
                         'CAMERA_PITCH_LIMIT')
        self.assertEqual(view['recent_tool_receipts'][-1][
            'camera_pitch_limit']['allowed_look'],['none','up'])
        self.assertEqual(view['recent_camera_pitch_limits'][0][
            'rejected_look'],'down')

    def test_upper_boundary_rejected_before_any_turn(self):
        r,actions=test_runtime()
        r.motion_controller.pitch_deg=60
        r.refresh()
        result=r.execute(dict(tool='turn_and_observe',
            args=dict(turn_direction='LEFT',turn_degrees=45,look='up')),
            decision=dict(observation_id=r.memory.current['observation_id'],
                          actor='navigation_agent'))
        self.assertEqual(result['status'],'CAMERA_PITCH_LIMIT')
        self.assertEqual(result['allowed_look'],['none','down'])
        self.assertEqual(result['executed_actions'],0)
        self.assertEqual(actions,[])
        self.assertEqual(r.pitch_deg,60)

    def test_valid_up_at_lower_boundary_is_executed(self):
        r,actions=test_runtime()
        r.motion_controller.pitch_deg=-60
        r.refresh()
        value=r.execute(dict(tool='turn_and_observe',
            args=dict(turn_direction='NONE',turn_degrees=0,look='up')),
            decision=dict(observation_id=r.memory.current['observation_id'],
                          actor='navigation_agent'))
        self.assertEqual(value['status'],'ACTIONS_COMPLETED')
        self.assertEqual(actions,['LOOK_UP'])
        self.assertEqual(r.pitch_deg,-45)

    def test_navigator_receives_current_look_limits_in_context(self):
        r,_=test_runtime()
        r.motion_controller.pitch_deg=-60
        r.refresh()
        context=navigation_context(r)
        camera=context['camera_controls']
        self.assertEqual(camera['current_pitch_deg'],-60)
        self.assertEqual(camera['pitch_limits_deg'],[-60,60])
        self.assertEqual(camera['tilt_step_deg'],15)
        self.assertEqual(camera['allowed_look_now'],['none','up'])

    def test_same_invalid_tool_feedback_is_summarized_not_terminal(self):
        r,actions=test_runtime()
        r.motion_controller.pitch_deg=-60
        r.refresh()
        for _ in range(2):
            feedback=r.execute(dict(tool='turn_and_observe',
                args=dict(turn_direction='RIGHT',turn_degrees=30,look='down')),
                decision=dict(observation_id=r.memory.current['observation_id'],
                              actor='navigation_agent'))
            self.assertEqual(feedback['status'],'CAMERA_PITCH_LIMIT')
            r.assess(feedback)
        evidence=mission_evidence(r.memory,r.motion_controller.pose())
        self.assertEqual(evidence['recent_camera_pitch_limits'][0][
            'repeated_same_invalid_look'],2)
        self.assertEqual(actions,[])
        self.assertFalse(r.done)

    def test_scheduler_can_make_valid_decisions_after_pitch_rejection(self):
        r,actions=test_runtime()
        r.motion_controller.pitch_deg=-60
        r.refresh()
        r.client.availability=lambda:'available'
        choices=[
            dict(tool='turn_and_observe',
                 args=dict(turn_direction='RIGHT',turn_degrees=30,
                           look='down')),
            dict(tool='turn_and_observe',
                 args=dict(turn_direction='NONE',turn_degrees=0,
                           look='up')),
            dict(tool='request_finish',
                 args=dict(outcome='aborted',
                           evidence='End the isolated unit test.'))]
        r.navigation_agent=SimpleNamespace(decide=Mock(side_effect=choices))
        logs=[]
        run(r,lambda role,value:logs.append((role,value)))
        self.assertEqual(r.navigation_agent.decide.call_count,3)
        self.assertEqual(actions,['LOOK_UP','STOP'])
        self.assertNotEqual(r.stop_reason,'api_error')
        self.assertEqual(r.stop_reason,'model_aborted')
        self.assertTrue(r.done)
        self.assertTrue(any(x.get('feedback',{}).get('status')==
            'CAMERA_PITCH_LIMIT' for _,x in logs if isinstance(x,dict)))


if __name__=='__main__':
    unittest.main()
