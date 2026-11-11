"""Phase 2 Episode 2: compare room entrants by actual target evidence.

The last VLM report identified chairs but no seated magazine reader.
Compacting its long floor-boundary prose previously hid this distinction.
Tests check the actual model-facing context and keep movement model-owned.
"""
import copy
import json
import unittest
from unittest.mock import Mock

from havln_agent.contexts import compact_navigation_context
from havln_agent.contracts import registry
from havln_agent.mission_history import mission_evidence
from havln_agent.model_angles import project_model_angles
from havln_agent.navigation_agent import MISSION_PROMPT
from havln_agent.tool_menu import navigation_tool_menu
from havln_agent.visual_perception import SCAN_GUIDANCE_PROMPT
from test_forward_preflight_v9 import camera, geometry, test_runtime
from test_v23_route_evidence import fixture


def inspected_room(observations):
    ids=[o['observation_id'] for o in observations]
    return dict(
        observation_task_id='observation_001',
        objective='Identify a seated person reading a magazine at this entrance.',
        outcome='OBSERVATION_COMPLETED',
        start_observation_id=ids[0],
        end_observation_id=ids[-1],
        observation_summary=(
            'A diagonal seam marks the nearby floor threshold between jambs. '
            'The robot appears beside a doorway with chairs in a long narrow '
            'side room, but the entrance position itself is not verified. '
            'No seated person reading a magazine is identifiable in any '
            'inspected camera view.'),
        navigation_suggestion=(
            'The near doorway is visible but the occupant is unconfirmed. '
            'Consider the other plausible entrance before declaring success.'),
        checkpoints=[
            dict(observation_id=ids[0],visual=dict(
                target_type='seated magazine reader',target_visible=False,
                confidence='medium',observation_summary='Cream chairs, no human.')),
            dict(observation_id=ids[1],visual=dict(
                target_type='seated magazine reader',target_visible=False,
                confidence='medium',observation_summary='Furniture, no reader visible.')),
            dict(observation_id=ids[2],visual=dict(
                target_type='room doorway threshold',target_visible=True,
                confidence='medium',observation_summary='Floor boundary visible.')),
        ])


class Phase2RoomDisambiguation(unittest.TestCase):
    def test_target_not_visible_survives_compaction_that_truncates_prose(self):
        m=fixture()
        m.observation_searches=[inspected_room(m.observations[-3:])]
        raw=mission_evidence(m,m.current['pose'])
        before=copy.deepcopy(raw)
        self.assertEqual([x['target_visible'] for x in
                          raw['observation_tasks'][0]['selected_views']],
                         [False,False,True])
        compact=compact_navigation_context(dict(memory=raw))['memory']
        self.assertEqual(raw,before)
        item=compact['observation_tasks'][0]
        # The original short prose projection loses the end-of-sentence
        # 'No seated person reading ...' despite the actual VLM verdicts.
        self.assertNotIn('No seated person reading',item['observation_summary'])
        self.assertEqual([c['target_visible'] for c in
                          item['target_visibility_checks']],[False,False,True])
        self.assertEqual([c['target_type'] for c in
                          item['target_visibility_checks']],
                         ['seated magazine reader','seated magazine reader',
                          'room doorway threshold'])
        self.assertTrue(item['target_not_visible_is_not_proven_absent'])
        self.assertEqual([c['observation_id'] for c in
                          item['target_visibility_checks']],
                         [o['observation_id'] for o in m.observations[-3:]])

    def test_actual_navigation_wire_includes_negative_reader_checks(self):
        r,_=test_runtime()
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        ids=[o['observation_id'] for o in r.memory.observations]
        current=ids[-1]
        # A single real checkpoint with an explicit negative visibility
        # verdict is enough to test the model-facing API context.
        r.memory.observation_searches=[dict(
            observation_task_id='observation_001',
            objective='Find the person reading in this room.',
            outcome='OBSERVATION_COMPLETED',
            end_observation_id=current,
            observation_summary='Doorway shows upholstered chairs, no reader.',
            checkpoints=[dict(
                observation_id=current,visual=dict(
                    target_type='person reading magazine',
                    target_visible=False,confidence='medium'))])]
        r.navigation_agent.decide()
        sent=r.client.complete.call_args.args[3]
        result=sent['memory']['observation_tasks'][0]
        self.assertFalse(result['target_visibility_checks'][0]['target_visible'])
        self.assertTrue(result['target_not_visible_is_not_proven_absent'])
        self.assertEqual(result['target_visibility_checks'][0]['observation_id'],
                         current)
        self.assertNotIn('heading_deg',json.dumps(sent))
        self.assertEqual(result['target_visibility_checks'][0]['target_type'],
                         'person reading magazine')

    def test_relative_angles_and_original_move_tools_unchanged(self):
        full=fixture()
        full.observation_searches=[inspected_room(full.observations[-3:])]
        snapshot=mission_evidence(full,full.current['pose'])
        wire=project_model_angles(
            compact_navigation_context(dict(memory=snapshot)),45)
        checks=wire['memory']['observation_tasks'][0]['target_visibility_checks']
        self.assertIs(checks[0]['target_visible'],False)
        self.assertIs(checks[-1]['target_visible'],True)
        tools=navigation_tool_menu(registry(camera(),3))
        self.assertIn('navigate_relative',tools)
        self.assertIn('observe_for_goal',tools)
        self.assertIn('request_finish',tools)
        self.assertIn('turn_direction',
                      tools['navigate_relative']['parameters']['properties'])

    def test_model_has_comparative_room_evidence_without_hardcoded_straight_rule(self):
        tools=navigation_tool_menu(registry(camera(),3))
        self.assertIn('multiple plausible ROOM entrances',MISSION_PROMPT)
        self.assertIn('target NOT_VISIBLE is uncertainty',MISSION_PROMPT)
        self.assertIn('route_reference_update',MISSION_PROMPT)
        self.assertIn('competing room entrances',
                      tools['observe_for_goal']['description'])
        self.assertIn('another plausible room',
                      tools['request_finish']['description'])
        self.assertIn('target not visible from this',SCAN_GUIDANCE_PROMPT)
        self.assertIn('cannot move between entrances',SCAN_GUIDANCE_PROMPT)
        self.assertNotIn('always choose straight',MISSION_PROMPT.lower())
        self.assertLess(len(MISSION_PROMPT),5000)


if __name__=='__main__':
    unittest.main()
