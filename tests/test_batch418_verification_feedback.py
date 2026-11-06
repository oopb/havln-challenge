"""Regression cases grounded in Batch 4-18: VERIFY loops, schema and advisory scope."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jsonschema import Draft7Validator

from havln_agent.active_observation import ObservationController
from havln_agent.api_client import APIClient
from havln_agent.memory_service import MemoryService
from havln_agent.visual_perception import VisualPerception, VERIFY_GUIDANCE_SCHEMA
from test_active_observation import FakeRuntime
from test_vlm_led_observation import VLMGuide, guidance


def verify(status='NOT_YET',action='finish',turn=0,**extra):
    result=dict(verification_status=status,
        verification_evidence='The robot is still above multiple visible treads.',
        uncertainty='Bottom landing not confirmed',
        next_action=action,turn_deg=turn,reason='visual assessment')
    result.update(extra)
    return result


class Batch418VerificationFeedbackTests(unittest.TestCase):
    def test_600_char_batch418_evidence_no_longer_rejected(self):
        value=verify(verification_evidence='The current RGB-D shows steps. '+('X'*565))
        self.assertGreater(len(value['verification_evidence']),592)
        self.assertTrue(Draft7Validator(VERIFY_GUIDANCE_SCHEMA).is_valid(value))

    def test_verify_returns_sourced_natural_language_not_motion_authorization(self):
        r=FakeRuntime([])
        r.visual_perception=VLMGuide([verify(
            observation_summary='Several descending steps remain in obs_0000',
            navigation_suggestion='Continue the current descent only after checking current obstruction',
            recommended_next='CONTINUE_CURRENT_TASK',
            evidence_observation_ids=['obs_0000'])])
        response=ObservationController(r).scan_vlm(
            'Is robot already on the bottom landing?',['stairs'],90,mode='VERIFY')
        self.assertEqual(response['outcome'],'SUBGOAL_NOT_YET')
        info=response['navigation_feedback']
        self.assertEqual(info['recommended_next'],'CONTINUE_CURRENT_TASK')
        self.assertIn('steps remain',info['observation_summary'])
        self.assertEqual(info['evidence_observation_ids'],['obs_0000'])
        self.assertTrue(info['advice_only_not_motion_authorization'])
        self.assertEqual(r.step_calls,[])

    def test_conflicting_proceed_next_task_advice_is_not_passed_as_valid_enum(self):
        r=FakeRuntime([])
        r.visual_perception=VLMGuide([verify(
            recommended_next='PROCEED_NEXT_TASK',
            navigation_suggestion='Switch to the future task')])
        response=ObservationController(r).scan_vlm(
            'Confirm arrival',['landing'],90,mode='VERIFY')
        self.assertEqual(response['verification']['status'],'NOT_YET')
        self.assertEqual(response['navigation_feedback']['recommended_next'],'UNKNOWN')
        self.assertNotIn('Switch to the future task',
                         response['navigation_feedback']['navigation_suggestion'])

    def test_rewording_verify_same_unchanged_rgbd_reuses_observation(self):
        r=FakeRuntime([])
        r.visual_perception=VLMGuide([verify()])
        observation=ObservationController(r)
        first=observation.scan_vlm('Check bottom of stairs',['stairs'],90,
                                   mode='VERIFY')
        second=observation.scan_vlm('Is the robot now at the flat landing?',
                                    ['stairs'],90,mode='VERIFY')
        self.assertEqual(first['outcome'],'SUBGOAL_NOT_YET')
        self.assertEqual(second['status'],'NO_NEW_INFORMATION')
        self.assertEqual(second['verification']['status'],'NOT_YET')
        self.assertEqual(second['navigation_feedback'],
                         first['navigation_feedback'])
        self.assertEqual(len(r.visual_perception.calls),1)

    def test_verify_checkpoint_budget_returns_provisional_uncertainty(self):
        r=FakeRuntime([])
        r.visual_perception=VLMGuide([verify('UNCERTAIN','turn_left',15)
                                      for _ in range(6)])
        result=ObservationController(r).scan_vlm(
            'Are we at the landing?',['stairs'],180,mode='VERIFY')
        self.assertEqual(len(r.visual_perception.calls),5)
        self.assertEqual(result['outcome'],'SUBGOAL_UNCERTAIN')
        self.assertEqual(result['verification']['status'],'UNCERTAIN')
        self.assertFalse(result['verification']['final_verdict'])
        self.assertNotIn('MOVE_FORWARD',r.step_calls)

    def test_verify_vlm_guide_passes_optional_advice_to_controller(self):
        rt=SimpleNamespace(camera=SimpleNamespace(normalized=False,turn_angle=15),
            memory=SimpleNamespace(task=SimpleNamespace(
                instruction='Descend stairs',current_subgoal_index=0,
                subgoals=[dict(action='descend',target='landing',
                               completion_description='Physically stand on lower landing')])))
        vp=VisualPerception(rt)
        vp.complete=Mock(return_value=verify(
            navigation_suggestion='Inspect another heading',
            observation_summary='Wooden treads continue below the robot',
            recommended_next='GATHER_MORE_EVIDENCE'))
        with patch('havln_agent.visual_perception.image_url',return_value='rgb'):
            response=vp.guide_scan_view(SimpleNamespace(depth=None),
                dict(observation_id='obs_12',
                     pose=dict(heading=0,pitch_deg=0)),
                'Verify descent',['stairs'],[],90,mode='VERIFY',session=[])
        self.assertIn('Wooden treads',response['observation_summary'])
        self.assertEqual(response['recommended_next'],'GATHER_MORE_EVIDENCE')
        self.assertEqual(vp.complete.call_args.kwargs['result_schema']
                         ['properties']['verification_evidence']['maxLength'],1800)

    def test_expected_progress_survives_persistent_visual_memory(self):
        rt=SimpleNamespace(camera=SimpleNamespace(turn_angle=15),
            motion_controller=SimpleNamespace(pose=lambda:dict(heading=0,
                motion_epoch=0,viewpoint_generation=0)))
        mem=MemoryService(rt)
        mem.task.subgoals=[dict(action='descend',target='landing')]
        mem.keyframes['obs_0044']=dict(place_id='P1',heading_deg=225,
                                       pitch_deg=-30)
        entry=mem.note_active_observation_target(dict(
            observation_id='obs_0044',heading_deg=225,
            pitch_deg=-30,target_kind='APPROACH',
            target_type='descend passage',expected_progress='DETOUR',
            evidence='Opening beside stair rail',uncertainty='unconfirmed'),
            'Find alternate descent',['stairs'])
        self.assertEqual(entry['expected_progress'],'DETOUR')
        self.assertEqual(mem.visual_targets[-1]['expected_progress'],'DETOUR')

    def test_visual_validation_failures_included_in_role_stats(self):
        client=object.__new__(APIClient)
        client.by_role={'visual_perception':APIClient._empty_stats()}
        client.by_endpoint={}
        client._account('visual_perception',dict(endpoint_id='primary',
            model='test',base_url='test',seconds=1.0,attempt=0,
            endpoint_index=0,format_retry=0,success=False,
            error_type='ValidationError',error='schema_maxLength',
            usage={}))
        self.assertEqual(client.by_role['visual_perception']['errors'],1)
        self.assertEqual(client.by_role['visual_perception']['validation_errors'],1)


if __name__=='__main__':
    unittest.main()
