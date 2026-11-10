"""v19: observation obstacle boxes must never become route waypoints.

Offline Python-only regressions. No Habitat scenes or API keys required.
"""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from jsonschema import Draft7Validator

from havln_agent.active_observation import ObservationController
from havln_agent.contexts import navigation_context
from havln_agent.memory_records import MemoryRecords
from havln_agent.mission_history import (
    mission_evidence,recent_views_at_stance,selected_history_frames)
from havln_agent.navigation_history import navigation_visual_anchors
from havln_agent.vision_protocol import (
    coordinate_schema,normalize_coordinates)
from havln_agent.visual_perception import SCAN_GUIDANCE_SCHEMA
from test_mission_navigation_architecture import camera,pose
from test_forward_preflight_v9 import test_runtime
from test_v16_observation_continuity import observed,pose as stance_pose


def make_runtime(guidance):
    m=MemoryRecords(camera())
    m.reset('test','Descend the stairs then pass the person to reach a piano')
    state=pose()
    m.observations=[dict(observation_id='obs_0000',pose=state,
        rgb_hash='rgb',depth_hash='depth')]
    m.current=m.observations[0]
    m.keyframes={}
    r=SimpleNamespace(memory=m,camera=camera(),done=False,
        config={'max_steps':100},
        motion_controller=SimpleNamespace(pose=lambda:state),
        pitch_deg=0,observation=SimpleNamespace(),
        visual_perception=SimpleNamespace(
            guide_scan_view=Mock(return_value=guidance)),
        source=lambda:dict(observation_id='obs_0000',
            rgb_hash='rgb',depth_hash='depth',pose=state),
        event=Mock())
    return r


def person_reply(passage_box=None):
    return dict(next_action='finish',turn_deg=0,
        observation_summary='A person blocks the center of the stairs.',
        navigation_suggestion='The possible gap is to the left, but its width is uncertain.',
        candidate_approach=True,target_visible=True,confidence='medium',
        target_kind='TARGET',target_type='dark person on stairs',
        box=[.52,.04,.82,.9],passage_box=passage_box,
        evidence='Visible person with uncertain side gap.',
        uncertainty='Cannot certify the passage width.',
        reason='Person blocks center; do not target the person itself.')


class PassageGrounding(unittest.TestCase):
    def test_person_box_no_passage_box_produces_no_waypoint_or_bearing(self):
        r=make_runtime(person_reply())
        result=ObservationController(r).scan_mission(
            'Find a route around the person',sweep_degrees=90)
        self.assertEqual(result['waypoints'],[])
        self.assertEqual(len(r.memory.route_candidates),1)
        candidate=r.memory.route_candidates[0]
        self.assertIsNone(candidate['absolute_heading_deg'])
        self.assertFalse(candidate['angular_bearing_grounded'])
        self.assertEqual(candidate['semantic_target_box'],[.52,.04,.82,.9])
        self.assertIsNone(candidate['passage_box'])
        self.assertEqual(candidate['route_status'],
                         'SEMANTIC_VISUAL_CUE_WITHOUT_LOCALIZED_PASSAGE')
        self.assertIn('gap is to the left',candidate['navigation_suggestion'])

    def test_separate_left_passage_box_guides_left_not_into_person(self):
        r=make_runtime(person_reply([.08,.50,.38,.94]))
        result=ObservationController(r).scan_mission(
            'Find a route around the person',sweep_degrees=90)
        self.assertEqual(len(result['waypoints']),1)
        wp=result['waypoints'][0]
        self.assertEqual(wp['bearing_source'],
            'source_RGB_passage_box_center_and_calibrated_HFOV')
        self.assertGreater(wp['relative_bearing_deg_from_current'],0)
        self.assertEqual(wp['recommended_turn_direction'],'LEFT')
        self.assertEqual(wp['semantic_target_region'],[.52,.04,.82,.9])
        self.assertEqual(wp['passage_region'],[.08,.50,.38,.94])
        candidate=r.memory.route_candidates[0]
        self.assertGreater(candidate['absolute_heading_deg'],90)
        self.assertEqual(candidate['target_box'],[.08,.50,.38,.94])
        self.assertEqual(candidate['semantic_target_box'],[.52,.04,.82,.9])
        self.assertEqual(candidate['bearing_region_role'],
            'visually_proposed_passage_not_obstacle')
        self.assertEqual(candidate['route_status'],
                         'VISUAL_PASSAGE_REGION_NOT_VERIFIED_TRAVERSABILITY')

    def test_legacy_no_passage_field_keeps_prose_not_numeric_route(self):
        old=person_reply()
        del old['passage_box']
        r=make_runtime(old)
        result=ObservationController(r).scan_mission('Inspect the gap',90)
        self.assertEqual(result['waypoints'],[])
        self.assertIsNone(r.memory.route_candidates[0]['absolute_heading_deg'])

    def test_coordinates_decode_both_boxes_independently(self):
        self.assertIn('passage_box',SCAN_GUIDANCE_SCHEMA['properties'])
        self.assertNotIn('passage_box',SCAN_GUIDANCE_SCHEMA['required'])
        raw=dict(box=[520,40,820,900],passage_box=[80,500,380,940],
                 coordinate_space='image_1000_xyxy')
        normalized=normalize_coordinates(raw)
        self.assertEqual(normalized['box'],[.52,.04,.82,.9])
        self.assertEqual(normalized['passage_box'],[.08,.5,.38,.94])
        thousand=coordinate_schema(SCAN_GUIDANCE_SCHEMA,'image_1000_xyxy')
        self.assertEqual(thousand['properties']['passage_box']['anyOf'][0]
                         ['items']['maximum'],1000)


class HeadingAndHistory(unittest.TestCase):
    def test_turn_sign_conversion_remains_internal(self):
        from havln_agent.heading import signed_turn
        self.assertEqual(signed_turn('RIGHT',60),-60)
        self.assertEqual(signed_turn('LEFT',45),45)
        self.assertEqual(signed_turn('NONE',0),0)
        r,_=test_runtime()
        self.assertNotIn('turn_heading_reference',navigation_context(r))

    def test_stale_passage_heading_hidden_but_source_record_kept(self):
        m=MemoryRecords(camera())
        m.reset('e','Find stairs')
        m.observations=[observed('obs_0001',90)]
        m.current=m.observations[0]
        m.note_visual_route_candidate(dict(
            source_observation_id='obs_0001',target_type='stairs',
            target_kind='APPROACH',evidence='Visible passage',
            objective='Find stairs',passage_box=[.3,.2,.6,.9],
            absolute_heading_deg=120))
        same=mission_evidence(m,stance_pose(90))
        self.assertEqual(same['candidate_route_hypotheses'][0]
                         ['relative_bearing_deg_from_current'],30)
        later=mission_evidence(m,stance_pose(90,1))
        stale=later['candidate_route_hypotheses'][0]
        self.assertFalse(stale['source_stance_matches_current'])
        self.assertIsNone(stale['absolute_heading_deg'])
        self.assertIsNone(stale['relative_bearing_deg_from_current'])
        self.assertTrue(stale['historical_bearing_in_source_record_only'])
        self.assertEqual(m.route_candidates[0]['absolute_heading_deg'],120)

    def test_object_only_frame_not_prioritized_as_route_anchor(self):
        m=MemoryRecords(camera())
        m.reset('e','Navigate past the person toward the hall')
        m.observations=[
            observed('origin',0),
            observed('person',30),
            observed('hall',90),
            observed('current',120)]
        m.current=m.observations[-1]
        m.observation_searches=[
            dict(observation_task_id='old',checkpoints=[
                dict(observation_id='person',visual=dict(
                    candidate_approach=True,
                    box=[.5,.1,.8,.9],passage_box=None))]),
            dict(observation_task_id='recent',checkpoints=[
                dict(observation_id='hall',visual=dict(
                    candidate_approach=True,
                    box=[.1,.3,.9,.8],
                    passage_box=[.2,.5,.4,.9]))])]
        anchors=navigation_visual_anchors(m,'current',max_pairs=3)
        by_id={v['observation_id']:v for v in anchors}
        self.assertIn('hall',by_id)
        self.assertEqual(by_id['hall']['selection_reason'],
                         'recent_route_candidate')
        self.assertNotEqual(by_id.get('person',{}).get('selection_reason'),
                            'recent_route_candidate')

    def test_repeated_unboxed_person_is_one_route_cue_not_three_paths(self):
        m=MemoryRecords(camera())
        m.reset('e','Descend past the person')
        m.observations=[observed('person_%d'%i,i*15)
                        for i in range(3)]
        m.current=m.observations[-1]
        for i in range(3):
            m.note_visual_route_candidate(dict(
                source_observation_id='person_%d'%i,
                target_type='dark figure',
                target_kind='TARGET',objective='How can we pass?',
                evidence='Possible gap to the left of figure',
                absolute_heading_deg=None))
        self.assertEqual(len(m.route_candidates),3)
        compressed=mission_evidence(m,stance_pose(0))[
            'candidate_route_hypotheses']
        self.assertEqual(len(compressed),1)
        self.assertEqual(len(compressed[0]['other_source_observation_ids']),2)
        self.assertTrue(compressed[0][
            'compact_unlocalized_cue_not_independent_route'])
        self.assertIsNone(compressed[0]['absolute_heading_deg'])

    def test_full_turn_coverage_and_turn_cost_are_factual_not_forced(self):
        m=MemoryRecords(camera())
        m.reset('e','Find passage')
        m.observations=[observed('obs_%04d'%i,i*15,0)
                        for i in range(24)]
        m.current=m.observations[-1]
        m.actions=[dict(action='TURN_LEFT') for _ in range(24)]
        coverage=recent_views_at_stance(m,stance_pose(0))
        self.assertEqual(coverage['turn_actions_since_last_forward'],24)
        self.assertTrue(coverage['horizontal_360_coverage_observed'])
        self.assertEqual(coverage['distinct_horizontal_heading_bins'],24)
        self.assertTrue(coverage['action_guidance_only_no_hard_view_ban'])


if __name__=='__main__':
    unittest.main()
