"""v22 regression: information-valued pitch choices and exact absolute yaw.

Offline: no DeepSeek request, Habitat assets, or added motion policy.
"""
import math
import unittest
from unittest.mock import Mock

from jsonschema import Draft7Validator

from havln_agent.contracts import registry, navigation_schema, compact_navigation_protocol
from havln_agent.heading import resolve_target_heading, signed_turn
from havln_agent.navigation_agent import MISSION_PROMPT
from havln_agent.visual_perception import SCAN_GUIDANCE_PROMPT
from test_forward_preflight_v9 import camera, geometry, test_runtime


class AbsoluteHeadingMath(unittest.TestCase):
    def test_270_to_180_does_not_turn_left_to_zero(self):
        self.assertEqual(resolve_target_heading(180, 270, 15), ('RIGHT', 90))
        self.assertEqual(resolve_target_heading(30, 300, 15), ('LEFT', 90))
        self.assertEqual(resolve_target_heading(90, 150, 15), ('RIGHT', 60))
        self.assertEqual(resolve_target_heading(300, 0, 15), ('RIGHT', 60))
        self.assertEqual(resolve_target_heading(30, 0, 15), ('LEFT', 30))
        self.assertEqual(resolve_target_heading(360, 0, 15), ('NONE', 0))
        self.assertEqual(resolve_target_heading(180, 0, 15), ('RIGHT', 180))

    def test_calibrated_turn_rounding_and_validation(self):
        self.assertEqual(resolve_target_heading(32, 0, 15), ('LEFT', 30))
        self.assertEqual(resolve_target_heading(4, 359, 15), ('NONE', 0))
        self.assertEqual(resolve_target_heading(15, 0, 15), ('LEFT', 15))
        for bad in (-1, 361, float('nan'), float('inf'), True, '30'):
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                resolve_target_heading(bad, 0, 15)
        with self.assertRaises(ValueError):
            resolve_target_heading(30, 0, 0)

    def test_converted_move_intent_matches_robot_yaw(self):
        for current, target in ((270, 180), (0, 30), (150, 90),
                                (300, 30), (0, 360)):
            with self.subTest(current=current, target=target):
                side, amount = resolve_target_heading(target, current, 15)
                final = (current + signed_turn(side, amount)) % 360
                delta = (target % 360 - final + 180) % 360 - 180
                self.assertLessEqual(abs(delta), 7.5)


class HeadingDecisionProtocol(unittest.TestCase):
    def setUp(self):
        self.r, self.actions = test_runtime()
        self.spec = registry(camera(), 88)

    def test_one_exclusive_turn_form_and_legal_look(self):
        for tool, args in (
                ('turn_and_observe', dict(target_heading_deg=90, look='up')),
                ('turn_and_observe', dict(target_heading_deg=360, look='none')),
                ('turn_and_observe', dict(turn_direction='NONE', turn_degrees=0,
                                          look='down')),
                ('navigate_relative', dict(target_heading_deg=180,
                                           distance_m=.5)),
                ('navigate_relative', dict(turn_direction='RIGHT',
                                           turn_degrees=30, distance_m=.5))):
            with self.subTest(tool=tool, args=args):
                self.assertTrue(Draft7Validator(
                    self.spec[tool]['parameters']).is_valid(args))
        for tool, args in (
                ('turn_and_observe', dict(target_heading_deg=90,
                                          turn_direction='RIGHT',
                                          turn_degrees=30, look='none')),
                ('turn_and_observe', dict(target_heading_deg=90,
                                          turn_degrees=30, look='none')),
                ('turn_and_observe', dict(look='up')),
                ('navigate_relative', dict(target_heading_deg=-1,
                                           distance_m=.5)),
                ('navigate_relative', dict(target_heading_deg=60,
                                           distance_m=0)),
                ('navigate_relative', dict(turn_direction='RIGHT',
                                           distance_m=.25))):
            with self.subTest(tool=tool, args=args):
                self.assertFalse(Draft7Validator(
                    self.spec[tool]['parameters']).is_valid(args))
        self.assertIn('target_heading_deg', compact_navigation_protocol(self.spec))
        self.assertIn('look=up/down/none', MISSION_PROMPT)

    def test_response_normalization_before_existing_runtime_dispatch(self):
        self.r.geometry = Mock(return_value=geometry('EXECUTABLE'))
        proposal = dict(
            observed_evidence=['Current RGB shows an opening near yaw 330.'],
            decision_analysis='Aim at heading 330 by executor math.',
            tool='turn_and_observe',
            args=dict(target_heading_deg=330, look='none',
                      purpose='inspect the opening'))
        self.r.client.complete.return_value = proposal
        decided = self.r.navigation_agent.decide()
        self.assertEqual(decided['args'], dict(
            look='none',purpose='inspect the opening',
            turn_direction='RIGHT', turn_degrees=30))
        self.assertNotIn('target_heading_deg', decided['args'])
        ctx = self.r.client.complete.call_args.args[3]
        self.assertEqual(ctx['front_depth_preflight']['status'], 'CLEAR')
        self.assertIn('recent_views_at_current_stance', ctx['memory'])
        self.assertIn('history_media', ctx)
        executed = self.r.execute(
            dict(tool=decided['tool'], args=decided['args']),
            dict(observation_id=self.r.memory.current['observation_id'],
                 actor='navigation_agent'))
        self.assertEqual(executed['status'], 'ACTIONS_COMPLETED')
        self.assertEqual(self.actions, ['TURN_RIGHT', 'TURN_RIGHT'])

    def test_absolute_heading_plus_forward_normalizes_without_changing_safety(self):
        self.r.geometry = Mock(return_value=geometry('UNCONFIRMED'))
        self.r.client.complete.return_value = dict(
            observed_evidence=['A local route at bearing 30 is a hypothesis.'],
            decision_analysis='Aim at absolute heading 30, then move one step.',
            tool='navigate_relative',
            args=dict(target_heading_deg=30, distance_m=.25))
        decided = self.r.navigation_agent.decide()
        self.assertEqual(decided['args'], dict(
            distance_m=.25, turn_direction='LEFT', turn_degrees=30))
        ctx = self.r.client.complete.call_args.args[3]
        self.assertNotIn('front_depth_preflight', ctx)
        self.assertEqual(self.actions, [])

    def test_pure_vertical_look_survives_absolute_heading_conversion(self):
        self.r.geometry = Mock(return_value=geometry('EXECUTABLE'))
        self.r.client.complete.return_value = dict(
            observed_evidence=['Need to inspect low stair edges.'],
            decision_analysis='Change pitch without changing yaw.',
            tool='turn_and_observe',
            args=dict(target_heading_deg=0, look='down'))
        decided = self.r.navigation_agent.decide()
        self.assertEqual(decided['args']['turn_direction'], 'NONE')
        self.assertEqual(decided['args']['turn_degrees'], 0)
        executed = self.r.execute(
            dict(tool=decided['tool'], args=decided['args']),
            dict(observation_id=self.r.memory.current['observation_id'],
                 actor='navigation_agent'))
        self.assertEqual(executed['status'], 'ACTIONS_COMPLETED')
        self.assertEqual(self.actions, ['LOOK_DOWN'])
        self.assertEqual(self.r.pitch_deg, -15)

    def test_legacy_relative_actions_are_not_changed(self):
        self.r.geometry = Mock(return_value=geometry('EXECUTABLE'))
        self.r.client.complete.return_value = dict(
            observed_evidence=['Current RGB left appears useful.'],
            decision_analysis='Make a relative left turn.',
            tool='turn_and_observe',
            args=dict(turn_direction='LEFT', turn_degrees=15,
                      look='none'))
        decided = self.r.navigation_agent.decide()
        self.assertEqual(decided['args'], dict(turn_direction='LEFT',
                                               turn_degrees=15,look='none'))

    def test_visual_information_value_without_automatic_look_or_scan(self):
        self.assertIn('same horizontal heading at a different pitch', MISSION_PROMPT)
        self.assertIn('consider LOOK_UP', MISSION_PROMPT)
        self.assertIn('consider LOOK_DOWN', MISSION_PROMPT)
        self.assertIn('observe_for_goal', MISSION_PROMPT)
        self.assertIn('no automatic LOOK_UP', MISSION_PROMPT)
        self.assertIn('not on a fixed schedule', MISSION_PROMPT)
        self.assertIn('LOOK_UP', SCAN_GUIDANCE_PROMPT)
        self.assertIn('LOOK_DOWN', SCAN_GUIDANCE_PROMPT)
        self.assertIn('no forced scan', MISSION_PROMPT)
        self.assertLess(len(MISSION_PROMPT), 5000)
        Draft7Validator.check_schema(navigation_schema(self.spec))


if __name__ == '__main__':
    unittest.main()
