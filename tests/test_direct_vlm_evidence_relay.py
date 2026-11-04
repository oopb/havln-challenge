"""VLM statements flow to NavigationAgent without semantic target-name filtering."""
import unittest
from unittest.mock import Mock,patch

from havln_agent.active_observation import ObservationController
from havln_agent.contexts import compact_navigation_context,tool_summary,compact_vlm_reports
from test_active_observation import FakeRuntime
from test_vlm_led_observation import VLMGuide,guidance


class DirectVLMRelayTests(unittest.TestCase):
    def test_batch413_free_text_target_reaches_navigation_without_enum_match(self):
        r=FakeRuntime([])
        visual=dict(guidance('finish',kind='APPROACH',visible=True),
            target_type='wooden floor landing',
            confidence='medium',
            evidence='Wooden floor and the edge of a railing')
        r.visual_perception=VLMGuide([visual])
        r.memory.note_active_observation_target=Mock()
        result=ObservationController(r).scan_vlm(
            'Locate the lower landing',['stairs','landing','floor'])
        self.assertEqual(result['outcome'],'TARGET_CANDIDATE_FOUND')
        self.assertEqual(result['best_candidate']['target_type'],
                         'wooden floor landing')
        self.assertEqual(result['visual_reports'][0]['target_type'],
                         'wooden floor landing')
        self.assertEqual(result['visual_reports'][0]['observation_id'],'obs_0000')
        self.assertEqual(result['visual_reports'][0]['pitch_deg'],0)
        self.assertEqual(result['visual_reports'][0]['confidence'],'medium')
        self.assertFalse(result['visual_reports'][0]['usable_for_live_safety'])
        r.memory.note_active_observation_target.assert_called_once()

    def test_low_confidence_missing_box_is_reported_not_validated_away(self):
        r=FakeRuntime([])
        visual=dict(guidance('finish',kind='APPROACH',visible=True),
            target_type='wood-looking surface',confidence='low',box=None,
            evidence='Possible landing but no distinct boundary',
            uncertainty='Cannot identify traversable floor')
        r.visual_perception=VLMGuide([visual])
        result=ObservationController(r).scan_vlm(
            'Inspect stairs',['landing'],sweep_degrees=60)
        self.assertEqual(result['outcome'],'TARGET_CANDIDATE_FOUND')
        self.assertIsNone(result['best_candidate']['box'])
        self.assertEqual(result['best_candidate']['confidence'],'low')
        self.assertEqual(result['visual_reports'][0]['uncertainty'],
                         'Cannot identify traversable floor')

    def test_negative_or_unrelated_visual_result_is_still_reported(self):
        r=FakeRuntime([])
        visual=dict(guidance('finish'),target_type='stone wall',
            evidence='Only a nearby wall is in view',
            uncertainty='Target not visible',target_kind='UNKNOWN',
            target_visible=False)
        r.visual_perception=VLMGuide([visual])
        result=ObservationController(r).scan_vlm(
            'Find stairs',['stairs'],sweep_degrees=60)
        self.assertEqual(result['outcome'],'VLM_SEARCH_FINISHED')
        self.assertIsNone(result['best_candidate'])
        self.assertEqual(result['visual_reports'][0]['target_type'],'stone wall')
        self.assertFalse(result['visual_reports'][0]['target_visible'])
        summary=tool_summary(dict(result,tool='observe_for_goal'))
        self.assertEqual(summary['observation_task']['visual_reports'][0]
                         ['target_type'],'stone wall')

    def test_multiple_actual_frames_retained_not_just_best_candidate(self):
        r=FakeRuntime([])
        r.visual_perception=VLMGuide([
            dict(guidance('look_down'),target_type='wall',
                 evidence='A nearby wall',target_kind='UNKNOWN'),
            dict(guidance('finish',kind='APPROACH',visible=True),
                 target_type='stair bottom wooden floor')])
        result=ObservationController(r).scan_vlm(
            'Find an approach',['stairs'],sweep_degrees=120)
        reports=result['visual_reports']
        self.assertEqual(len(reports),2)
        self.assertEqual([p['observation_id'] for p in reports],
                         ['obs_0000','obs_0001'])
        self.assertEqual([p['pitch_deg'] for p in reports],[0,-30])
        self.assertEqual(reports[0]['target_type'],'wall')
        self.assertEqual(reports[1]['target_type'],'stair bottom wooden floor')

    def test_direct_evidence_survives_context_compaction_and_not_double_counted(self):
        raw=[dict(observation_id='obs_0001',heading_deg=180,pitch_deg=-30,
            target_visible=True,target_type='wooden floor landing',
            target_kind='APPROACH',confidence='low',box=None,
            evidence='Close wooden texture, may not be a landing',
            uncertainty='Possibly a wall',next_action='finish',
            usable_for_live_safety=False)]
        state=dict(
            latest_observation_task=dict(mode='vlm_led',objective='Find stairs',
                outcome='TARGET_CANDIDATE_FOUND',visual_reports=raw),
            last_tool=dict(tool='observe_for_goal',status='OK',
                observation_task=dict(outcome='TARGET_CANDIDATE_FOUND',
                    visual_reports=raw)),
            pose=dict(heading_deg=90))
        result=compact_navigation_context(state)
        self.assertEqual(result['latest_observation_task']['visual_reports'][0]
                         ['target_type'],'wooden floor landing')
        self.assertIsNone(result['latest_observation_task']['visual_reports'][0]['box'])
        self.assertNotIn('visual_reports',result['last_tool']['observation_task'])
        self.assertFalse(result['latest_observation_task']['visual_reports'][0]
                         ['usable_for_live_safety'])

    def test_visual_reports_are_provenance_labeled_and_text_bounded(self):
        reports=compact_vlm_reports([dict(observation_id='obs_0002',
            heading_deg=-45,pitch_deg=60,target_visible=False,
            target_type='unknown',target_kind='UNKNOWN',
            evidence='a'*700,uncertainty='b'*300,reason='c'*200)])
        self.assertEqual(reports[0]['observation_id'],'obs_0002')
        self.assertEqual(len(reports[0]['evidence']),180)
        self.assertEqual(len(reports[0]['uncertainty']),120)
        self.assertEqual(len(reports[0]['reason']),90)


if __name__=='__main__':
    unittest.main()
