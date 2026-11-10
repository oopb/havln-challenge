"""v20: navigation decides directional words, not signed yaw; Depth input only for
conclusive live CLEAR/BLOCKED states.

No simulator scenes, external API or geometry changes are required.
"""
import unittest
from unittest.mock import Mock

from havln_agent.contracts import compact_navigation_protocol,registry
from havln_agent.navigation_agent import MISSION_PROMPT
from havln_agent.heading import signed_turn
from test_forward_preflight_v9 import test_runtime,geometry,camera


def navigation_request_for(raw, confirmed=True):
    runtime,actions=test_runtime()
    runtime.geometry=Mock(return_value=geometry(raw,confirmed))
    runtime.navigation_agent.decide()
    context=runtime.client.complete.call_args.args[3]
    return runtime,actions,context


class V20DepthProjection(unittest.TestCase):
    def test_clear_is_attached_with_live_source(self):
        r,_,context=navigation_request_for('EXECUTABLE')
        self.assertEqual(context['front_depth_preflight']['status'],'CLEAR')
        self.assertEqual(context['front_depth_preflight'][
            'source_observation_id'],context['current_rgb']['observation_id'])
        self.assertNotIn('turn_heading_reference',context)

    def test_confirmed_block_is_attached_including_real_roi(self):
        r,_,context=navigation_request_for('BLOCKED',confirmed=True)
        block=context['front_depth_preflight']
        self.assertEqual(block['status'],'BLOCKED')
        self.assertEqual(block['blocker']['central_support_pixels'],257)
        self.assertEqual(block['source_observation_id'],
                         context['current_rgb']['observation_id'])
        self.assertEqual(len(r.memory.evidence.obstacle_blocks),0)
        self.assertEqual(len(r.memory.evidence.predecision_front_obstacles),1)

    def test_all_inconclusive_states_omit_entire_front_preflight(self):
        for raw,confirmed in (
                ('UNCONFIRMED',True),
                ('BLOCKED',False)):
            with self.subTest(raw=raw,confirmed=confirmed):
                runtime,actions,context=navigation_request_for(
                    raw,confirmed)
                self.assertNotIn('front_depth_preflight',context)
                self.assertNotIn('turn_heading_reference',context)
                self.assertIn('current_rgb',context)
                self.assertIn('memory',context)
                self.assertEqual(actions,[])
                # It was still measured/logged; it was only withheld from
                # the navigator model's request payload.
                self.assertTrue(runtime.geometry.called)

    def test_inconclusive_live_depth_does_not_turn_into_movement_veto(self):
        r,actions=test_runtime(changed_forwards=True)
        r.geometry=Mock(return_value=geometry('UNCONFIRMED'))
        result=r._navigate_relative('NONE',0,.25)
        self.assertEqual(result['status'],'ACTIONS_COMPLETED')
        self.assertEqual(actions,['MOVE_FORWARD'])
        self.assertEqual(r._last_forward_depth_probe['status'],'UNKNOWN')

    def test_confirmed_block_still_vetoes_atomic_forward(self):
        r,actions=test_runtime(changed_forwards=True)
        r.geometry=Mock(return_value=geometry('BLOCKED',True))
        result=r._navigate_relative('NONE',0,.25)
        self.assertNotIn('MOVE_FORWARD',actions)
        self.assertEqual(r._last_forward_depth_probe['status'],'BLOCKED')
        self.assertNotEqual(result['status'],'ACTIONS_COMPLETED')


class V20DirectionWords(unittest.TestCase):
    def test_model_sees_direction_words_not_heading_arithmetic_table(self):
        r,_,context=navigation_request_for('EXECUTABLE')
        self.assertNotIn('turn_heading_reference',context)
        self.assertIn('turn_direction LEFT, RIGHT, or NONE',MISSION_PROMPT)
        self.assertIn('look=up/down/none',MISSION_PROMPT)
        self.assertNotIn('turn_heading_reference',MISSION_PROMPT)
        self.assertNotIn('positive/negative',MISSION_PROMPT)
        protocol=compact_navigation_protocol(registry(camera(),20))
        self.assertIn('turn_direction',protocol)
        self.assertIn('look=up/down/none',protocol)
        self.assertNotIn('LEFT positive yaw',protocol)

    def test_word_to_motor_sign_is_internal_and_unchanged(self):
        self.assertEqual(signed_turn('LEFT',30),30)
        self.assertEqual(signed_turn('RIGHT',30),-30)
        self.assertEqual(signed_turn('NONE',0),0)
        r,actions=test_runtime()
        result=r._turn_and_observe('RIGHT',30,'none')
        self.assertEqual(result['status'],'ACTIONS_COMPLETED')
        self.assertEqual(actions,['TURN_RIGHT','TURN_RIGHT'])
        r2,actions2=test_runtime()
        result2=r2._turn_and_observe('LEFT',15,'up')
        self.assertEqual(result2['status'],'ACTIONS_COMPLETED')
        self.assertEqual(actions2,['TURN_LEFT','LOOK_UP'])


if __name__=='__main__':
    unittest.main()
