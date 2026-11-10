"""v22/v24: explicit relative robot-frame turn actions and voluntary pitch.
Offline; no Habitat, remote APIs or privileged world pose.
"""
import unittest
from unittest.mock import Mock
from jsonschema import Draft7Validator

from havln_agent.contracts import registry, navigation_schema, compact_navigation_protocol
from havln_agent.heading import signed_turn
from havln_agent.navigation_agent import MISSION_PROMPT
from havln_agent.visual_perception import SCAN_GUIDANCE_PROMPT
from test_forward_preflight_v9 import camera, geometry, test_runtime


class RelativeDirectionOnly(unittest.TestCase):
    def setUp(self):
        self.r, self.actions = test_runtime()
        self.tools = registry(camera(), 88)

    def test_relative_turn_is_the_only_accepted_model_tool_contract(self):
        for tool,args in (
            ('turn_and_observe',dict(turn_direction='LEFT',turn_degrees=30,look='none')),
            ('turn_and_observe',dict(turn_direction='NONE',turn_degrees=0,look='down')),
            ('navigate_relative',dict(turn_direction='RIGHT',turn_degrees=45,distance_m=.5)),
            ('navigate_relative',dict(turn_direction='NONE',turn_degrees=0,distance_m=.25))):
            with self.subTest(tool=tool,args=args):
                self.assertTrue(Draft7Validator(
                    self.tools[tool]['parameters']).is_valid(args))
        for tool,args in (
            ('turn_and_observe',dict(target_heading_deg=90,look='up')),
            ('navigate_relative',dict(target_heading_deg=135,distance_m=.5)),
            ('turn_and_observe',dict(turn_direction='LEFT',turn_degrees=30,
                                      target_heading_deg=45,look='none')),
            ('turn_and_observe',dict(look='up')),
            ('navigate_relative',dict(turn_direction='LEFT',distance_m=.5)),
            ('navigate_relative',dict(turn_direction='NONE',turn_degrees=0,distance_m=0))):
            with self.subTest(tool=tool,args=args):
                self.assertFalse(Draft7Validator(
                    self.tools[tool]['parameters']).is_valid(args))
        protocol=compact_navigation_protocol(self.tools)
        self.assertNotIn('target_heading_deg:',protocol)
        self.assertIn('turn_direction',protocol)
        self.assertIn('look=up/down/none',protocol)

    def test_robot_turns_model_chosen_left_right_without_heading_math(self):
        self.r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        self.r.client.complete.return_value=dict(
            observed_evidence=['Current RGB shows a passage to robot LEFT.'],
            decision_analysis='Turn left to inspect the gap.',
            tool='turn_and_observe',
            args=dict(turn_direction='LEFT',turn_degrees=30,look='none'))
        decision=self.r.navigation_agent.decide()
        self.assertEqual(decision['args'],dict(turn_direction='LEFT',
                                              turn_degrees=30,look='none'))
        result=self.r.execute(dict(tool=decision['tool'],args=decision['args']),
            dict(observation_id=self.r.memory.current['observation_id'],
                 actor='navigation_agent'))
        self.assertEqual(result['status'],'ACTIONS_COMPLETED')
        self.assertEqual(self.actions,['TURN_LEFT','TURN_LEFT'])
        self.assertEqual(signed_turn('RIGHT',60),-60)

    def test_pure_camera_look_still_executes_without_rotation(self):
        self.r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        self.r.client.complete.return_value=dict(
            observed_evidence=['Need to see the low floor.'],
            decision_analysis='Look down without yaw.',
            tool='turn_and_observe',
            args=dict(turn_direction='NONE',turn_degrees=0,look='down'))
        decision=self.r.navigation_agent.decide()
        result=self.r.execute(dict(tool=decision['tool'],args=decision['args']),
            dict(observation_id=self.r.memory.current['observation_id'],
                 actor='navigation_agent'))
        self.assertEqual(result['status'],'ACTIONS_COMPLETED')
        self.assertEqual(self.actions,['LOOK_DOWN'])
        self.assertEqual(self.r.pitch_deg,-15)

    def test_no_model_person_region_contract_after_rollback(self):
        schema=navigation_schema(self.tools)
        Draft7Validator.check_schema(schema)
        decision=dict(observed_evidence=['A person is visible.'],
            decision_analysis='Choose a relative turn after observing.',
            tool='turn_and_observe',
            args=dict(turn_direction='LEFT',turn_degrees=15,look='none'))
        self.assertTrue(Draft7Validator(schema).is_valid(decision))
        decision['visible_people_boxes']=[[.6,.2,.9,.95]]
        self.assertFalse(Draft7Validator(schema).is_valid(decision))

    def test_prompt_carries_pitch_and_relative_steering_without_look_quota(self):
        for phrase in ('turn_direction LEFT, RIGHT, or NONE',
                       'look=up/down/none','consider LOOK_UP','consider LOOK_DOWN',
                       'no automatic LOOK_UP','not on a fixed schedule',
                       'no forced scan','observe_for_goal'):
            self.assertIn(phrase,MISSION_PROMPT)
        self.assertNotIn('target_heading_deg',MISSION_PROMPT)
        self.assertNotIn('near_field_proximity',MISSION_PROMPT)
        self.assertNotIn('visible_people_boxes',MISSION_PROMPT)
        self.assertIn('LOOK_UP',SCAN_GUIDANCE_PROMPT)
        self.assertIn('LOOK_DOWN',SCAN_GUIDANCE_PROMPT)
        self.assertLess(len(MISSION_PROMPT),5000)


if __name__=='__main__':
    unittest.main()
