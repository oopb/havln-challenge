"""v11 targeted contracts: read-only VLM JSON, coverage, RGB-D session replay.

Run: python -m unittest discover -s tests -p 'test_v11_observation_review.py' -v
No Habitat scenes or paid model calls required.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jsonschema import Draft7Validator

from havln_agent.api_client import APIClient, APIError, decode_response
from havln_agent.active_observation import ObservationController
from havln_agent.memory_records import MemoryRecords
from havln_agent.mission_history import recent_views_at_stance, mission_evidence
from havln_agent.visual_perception import (
    SCAN_GUIDANCE_SCHEMA, VisualPerception)
from test_forward_preflight_v9 import test_runtime, geometry


def visual_answer(next_action='finish', turn_deg=0):
    return dict(
        target_visible=False, target_type='unknown', confidence='low',
        box=None, evidence='Hallway direction is uncertain.',
        uncertainty='No confirmed floor entry.', target_kind='UNKNOWN',
        candidate_approach=False, next_action=next_action,
        turn_deg=turn_deg, reason='Observe if a different view is useful.',
        observation_summary='Stone archway in view.',
        navigation_suggestion='Consider short, checked exploration.',
        expected_progress='UNKNOWN', distance_m_hint=None)


class V11ReadOnlyVisualJson(unittest.TestCase):
    def test_identical_duplicate_boolean_recovers_only_when_opted_in(self):
        raw=json.dumps(visual_answer()).replace(
            '"target_visible": false',
            '"target_visible": false, "target_visible": false',1)
        with self.assertRaises(json.JSONDecodeError):
            decode_response(raw)
        recovered,source=decode_response(
            raw,allow_identical_duplicates=True)
        self.assertEqual(recovered,visual_answer())
        self.assertIn('identical_duplicate_keys[target_visible]',source)
        Draft7Validator(SCAN_GUIDANCE_SCHEMA).validate(recovered)

    def test_conflicting_duplicate_uncertainty_remains_an_error(self):
        raw=json.dumps(visual_answer()).replace(
            '"uncertainty": "No confirmed floor entry."',
            '"uncertainty": "Possible hall", "uncertainty": "No confirmed floor entry."',1)
        with self.assertRaisesRegex(json.JSONDecodeError,
                                    'Conflicting duplicate JSON key: uncertainty'):
            decode_response(raw,allow_identical_duplicates=True)

    def test_boolean_and_integer_duplicates_are_not_equal_json_types(self):
        raw='{"target_visible": true, "target_visible": 1}'
        with self.assertRaises(json.JSONDecodeError):
            decode_response(raw,allow_identical_duplicates=True)

    def test_duplicate_recovery_never_bypasses_required_schema(self):
        raw='{"target_visible": false, "target_visible": false}'
        parsed,_=decode_response(raw,allow_identical_duplicates=True)
        self.assertFalse(Draft7Validator(SCAN_GUIDANCE_SCHEMA).is_valid(parsed))

    def test_actual_client_accepts_identical_duplicate_only_for_vlm_observation(self):
        settings=dict(
            provider='openai_compatible',base_url='https://test.invalid/v1',
            api_key_env='FAKE_HAVLN_TEST_KEY',model='test-model',
            response_format='json_object',temperature=0,
            timeout_seconds=10,max_retries=0,max_tokens=500)
        response=json.dumps(visual_answer()).replace(
            '"target_visible": false',
            '"target_visible": false, "target_visible": false',1)
        with tempfile.TemporaryDirectory() as root, patch.dict(
                'os.environ',{'FAKE_HAVLN_TEST_KEY':'fake-key'}):
            client=APIClient({
                'navigation_agent':settings,'visual_perception':settings},root)
            client.begin_episode('episode',12,None)
            client._request=Mock(return_value=dict(
                choices=[dict(message=dict(content=response),
                              finish_reason='stop')],
                model='test-model',usage={}))
            parsed=client.complete(
                'visual_perception','vlm_guided_observation','Observe',
                dict(source_observation_id='obs_0000'),SCAN_GUIDANCE_SCHEMA)
            self.assertEqual(parsed,visual_answer())
            self.assertEqual(client.normalizations,1)
            self.assertEqual(client.calls,1)
            # Same duplicated JSON is NOT allowed by NavigationAgent.
            client._request=Mock(return_value=dict(
                choices=[dict(message=dict(content='{"tool":"x","tool":"x"}'),
                              finish_reason='stop')],
                model='test-model',usage={}))
            nav_schema=dict(type='object',properties=dict(
                tool=dict(type='string')),required=['tool'])
            with self.assertRaises(APIError) as caught:
                client.complete('navigation_agent','navigation_decision',
                                'Navigate',{},nav_schema)
            self.assertEqual(caught.exception.category,'json_syntax')

    def test_model_vision_format_retry_keeps_exact_same_source(self):
        client=Mock()
        error=APIError('invalid_model_response',category='json_syntax')
        error.request_context=dict(
            error_detail='Conflicting duplicate JSON key: uncertainty')
        reply=visual_answer()
        client.complete.side_effect=[error,reply]
        perception=VisualPerception(SimpleNamespace(client=client))
        original=dict(source_observation_id='obs_0005',
                      heading_deg=90,allowed_camera_actions=['finish'])
        returned=perception.complete(
            'visual_perception','vlm_guided_observation','system',
            original,SCAN_GUIDANCE_SCHEMA,image_url=['rgb','depth'])
        self.assertIs(returned,reply)
        self.assertEqual(client.complete.call_count,2)
        first=client.complete.call_args_list[0]
        second=client.complete.call_args_list[1]
        self.assertEqual(first.args[3],original)
        self.assertEqual(second.args[3]['source_observation_id'],'obs_0005')
        self.assertIn('uncertainty',
                      second.args[3]['format_feedback']['error'])
        self.assertIn('EXACTLY ONCE',
                      second.args[3]['format_feedback']['instruction'])
        self.assertEqual(first.kwargs['image_url'],
                         second.kwargs['image_url'])
        self.assertEqual([first.kwargs['format_retry'],
                          second.kwargs['format_retry']],[0,1])

    def test_audit_counts_replayed_multimodal_history_without_storing_images(self):
        # APIClient._log receives only safe message metadata, not historical
        # base64 image payloads. The effective input length must count both.
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'api.jsonl'
            stub=SimpleNamespace(
                log_path=path,episode_id='test',trace_context={},keys={})
            record=dict(
                messages=[dict(role='system',content='SYS'),
                          dict(role='user',content='VIEW',
                               image_sha256=['rgb-hash','depth-hash'],
                               prior_conversation_turns=4,
                               prior_conversation_text_chars=200,
                               prior_conversation_image_count=6)],
                native_schema_chars=10)
            APIClient._log(stub,record)
            data=json.loads(path.read_text().splitlines()[0])
            self.assertEqual(data['prior_conversation_turns'],4)
            self.assertEqual(data['prior_conversation_text_chars'],200)
            self.assertEqual(data['input_text_chars'],207)
            self.assertEqual(data['request_text_plus_schema_chars'],217)
            self.assertEqual(data['image_count'],8)
            self.assertNotIn('data:image',path.read_text())


class V11ObservedCoverage(unittest.TestCase):
    def test_prior_observations_at_same_stance_survive_new_scan_session(self):
        from test_mission_history_and_headings import cam,pose
        m=MemoryRecords(cam())
        m.reset('episode','Navigate to the piano')
        same=pose(heading=90)
        left=pose(heading=120)
        different_place=pose(heading=60,x=.5)
        new_epoch=pose(heading=180,epoch=1)
        m.observations=[
            dict(observation_id='obs_0000',pose=same),
            dict(observation_id='obs_0001',pose=left),
            dict(observation_id='obs_0002',pose=different_place),
            dict(observation_id='obs_0003',pose=new_epoch),
            dict(observation_id='obs_0004',pose=left)]
        m.current=m.observations[-1]
        data=mission_evidence(m,left)
        coverage=data['recent_views_at_current_stance']
        self.assertEqual(coverage['sampled_observations'],3)
        self.assertEqual(coverage['distinct_heading_pitch_views'],2)
        self.assertEqual({v['source_observation_id']
                          for v in coverage['latest_distinct_views']},
                         {'obs_0000','obs_0004'})
        self.assertTrue(coverage['action_guidance_only_no_hard_view_ban'])
        self.assertTrue(coverage['historical_depth_does_not_authorize_move'])

    def test_actual_observation_session_sends_coverage_and_turn_cost(self):
        r,actions=test_runtime()
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        r.visual_perception=SimpleNamespace(
            guide_scan_view=Mock(side_effect=[
                visual_answer('turn_left',30),visual_answer('finish',0)]))
        result=ObservationController(r).scan_mission(
            'Locate a walkable route to the hall.',sweep_degrees=90)
        self.assertEqual(result['status'],'OK')
        self.assertEqual(actions,['TURN_LEFT','TURN_LEFT'])
        calls=r.visual_perception.guide_scan_view.call_args_list
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0].kwargs['view_coverage'][
            'session_observed_views'],0)
        self.assertEqual(calls[1].kwargs['view_coverage'][
            'session_observed_views'],1)
        self.assertEqual(calls[1].kwargs['view_coverage'][
            'session_turn_degrees'],30)
        self.assertEqual(calls[1].kwargs['view_coverage'][
            'turn_direction_reversals'],0)
        self.assertEqual(calls[1].kwargs['view_coverage'][
            'remaining_sweep_degrees'],60)

    def test_current_and_prior_rgbd_are_replayed_in_one_observation_session(self):
        r,_=test_runtime()
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        perception=VisualPerception(r)
        perception.complete=Mock(return_value=visual_answer())
        source=r.source()
        session=[]
        coverage=dict(session_observed_views=0,session_turn_degrees=0)
        with patch('havln_agent.visual_perception.image_url',
                   return_value='data:image/jpeg;base64,RGB'),patch(
                   'havln_agent.visual_perception.depth_image_url',
                   return_value='data:image/png;base64,DEPTH'):
            perception.guide_scan_view(
                r.observation,source,'Find hall',['hall'],[],90,
                session=session,view_coverage=coverage)
            self.assertEqual(len(session),2)
            first=perception.complete.call_args
            self.assertEqual(len(first.kwargs['image_url']),2)
            self.assertEqual(first.args[3]['view_coverage'],coverage)
            self.assertIn('recent_views_at_current_stance',
                          first.args[3]['mission_memory'])
            perception.guide_scan_view(
                r.observation,source,'Find hall',['hall'],[],90,
                session=session,view_coverage=dict(
                    session_observed_views=1,session_turn_degrees=0))
            second=perception.complete.call_args
            self.assertEqual(len(second.kwargs['conversation_messages']),2)
            self.assertIsNone(second.kwargs['image_url'])
            self.assertEqual(second.args[3]['mission_memory'],
                             'see_initial_observation_session_message')
            self.assertEqual(second.args[3]['reuse_prior_RGBD_observation_id'],
                             source['observation_id'])
            self.assertEqual(len(session),4)

    def test_exhausted_scan_yaw_budget_does_not_offer_zero_degree_turn(self):
        r,_=test_runtime()
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        perception=VisualPerception(r)
        perception.complete=Mock(return_value=visual_answer())
        with patch('havln_agent.visual_perception.image_url',
                   return_value='data:image/jpeg;base64,RGB'),patch(
                   'havln_agent.visual_perception.depth_image_url',
                   return_value='data:image/png;base64,DEPTH'):
            perception.guide_scan_view(
                r.observation,r.source(),'Find route',['hall'],[],0)
        schema=perception.complete.call_args.args[4]
        actions=schema['properties']['next_action']['enum']
        self.assertNotIn('turn_left',actions)
        self.assertNotIn('turn_right',actions)
        self.assertIn('finish',actions)


if __name__=='__main__':
    unittest.main()
