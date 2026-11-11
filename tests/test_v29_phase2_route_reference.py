"""P0 Phase 2 regression: historical stair-exit reference and safe return.

No simulator, API key, metric VO, or hardcoded movement policy is required.
"""
import copy
import json
import math
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

from jsonschema import Draft7Validator

from havln_agent.api_client import APIError
from havln_agent.contexts import compact_navigation_context
from havln_agent.contracts import (
    compact_navigation_protocol, navigation_envelope, navigation_schema,
    registry)
from havln_agent.memory_service import MemoryService
from havln_agent.mission_history import mission_evidence
from havln_agent.model_angles import project_model_angles
from havln_agent.navigation_agent import MISSION_PROMPT
from havln_agent.navigation_history import navigation_visual_anchors
from havln_agent.visual_perception import SCAN_GUIDANCE_PROMPT
from test_forward_preflight_v9 import camera, geometry, test_runtime
from test_v23_route_evidence import fixture


def update(*, outcome=None, outcome_id=None):
    result=dict(
        source_observation_id='obs_0019',
        place_description='foyer at bottom of curved staircase',
        instruction_relation='LEFT relative to the DESCENT APPROACH, '
                             'not LEFT from the current robot camera',
        evidence_status='candidate')
    if outcome is not None:
        result['route_attempt_outcome']=outcome
        if outcome_id is not None:
            result['outcome_observation_id']=outcome_id
    return result


def proposal(reference):
    return dict(
        observed_evidence=['The current RGB shows the stair exit.'],
        decision_analysis='Store the historical junction and continue safely.',
        route_reference_update=reference,
        tool='turn_and_observe',
        args=dict(turn_direction='NONE',turn_degrees=0,look='up'))


class Phase2RouteReference(unittest.TestCase):
    def test_schema_is_optional_and_does_not_add_motor_actions(self):
        tools=registry(camera(),3)
        self.assertTrue(Draft7Validator(navigation_schema(tools)).is_valid(
            proposal(update())))
        self.assertTrue(Draft7Validator(navigation_envelope(tools)).is_valid(
            proposal(update())))
        without=proposal(update())
        del without['route_reference_update']
        self.assertTrue(Draft7Validator(navigation_schema(tools)).is_valid(without))
        missing=proposal(update())
        missing['route_reference_update'].pop('instruction_relation')
        self.assertFalse(Draft7Validator(navigation_schema(tools)).is_valid(
            missing))
        extra=proposal(update())
        extra['route_reference_update']['turn_direction']='LEFT'
        self.assertFalse(Draft7Validator(navigation_schema(tools)).is_valid(
            extra))
        self.assertIn('route_reference_update',compact_navigation_protocol(tools))

    def test_historical_source_preserved_and_attempt_outcome_bound_to_evidence(self):
        m=fixture()
        initial=m.note_navigation_assessment(
            dict(route_reference_update=update()))
        self.assertTrue(initial['changed'])
        self.assertEqual(len(m.route_references),1)
        original=copy.deepcopy(m.route_references[0])
        self.assertEqual(original['source_observation_id'],'obs_0019')
        self.assertEqual(original['source_heading_deg'],225.)
        self.assertTrue(
            original['instruction_relation_is_historical_not_current_turn'])
        self.assertEqual(original['route_attempts'],[])
        m.note_navigation_assessment(dict(route_reference_update=update(
            outcome='Kitchen corridor led to garage; not the bedroom.',
            outcome_id='obs_0052')))
        ref=m.route_references[0]
        self.assertEqual(ref['reference_id'],'route_ref_0000')
        self.assertEqual(ref['source_heading_deg'],225.)
        self.assertEqual(ref['route_attempts'][0]['evidence_observation_id'],
                         'obs_0052')
        self.assertIn('garage',ref['route_attempts'][0]['assessment'])
        count=len(m.task.progress_history)
        m.note_navigation_assessment(dict(route_reference_update=update(
            outcome='Kitchen corridor led to garage; not the bedroom.',
            outcome_id='obs_0052')))
        self.assertEqual(len(m.route_references),1)
        self.assertEqual(len(ref['route_attempts']),1)
        self.assertEqual(len(m.task.progress_history),count)
        self.assertEqual(ref['source_observation_id'],original['source_observation_id'])

    def test_memory_projection_retains_anchor_but_not_current_turn(self):
        m=fixture()
        m.note_navigation_assessment(dict(route_reference_update=update(
            outcome='Wrong branch reached garage.',outcome_id='obs_0052')))
        raw=mission_evidence(m,m.current['pose'])
        before=copy.deepcopy(raw)
        refs=raw['instruction_route_references']
        self.assertEqual(refs[0]['source_observation_id'],'obs_0019')
        self.assertIn('DESCENT APPROACH',refs[0]['instruction_relation'])
        self.assertFalse(refs[0]['completion_verified'])
        compact=compact_navigation_context(dict(memory=raw))['memory']
        self.assertEqual(raw,before)
        wire=project_model_angles(dict(memory=compact),45)
        projected=wire['memory']['instruction_route_references'][0]
        self.assertNotIn('source_heading_deg',projected)
        self.assertEqual(projected['source_view_offset_deg_from_current'],-180)
        self.assertTrue(projected[
            'reference_is_historical_view_not_current_motor_bearing'])
        self.assertIn('garage',projected['route_attempts'][0]['assessment'])
        self.assertNotIn('target_heading_deg',json.dumps(wire))

    def test_old_camera_navigation_intent_does_not_replace_junction_reference(self):
        m=fixture()
        m.task.navigation_intent='Align with stair opening and begin descent.'
        m.task.navigation_intent_anchor=dict(
            source_observation_id='obs_0012',source_heading_deg=225)
        m.note_route_reference(update())
        full=mission_evidence(m,m.current['pose'])
        self.assertTrue(full['progress']['prior_navigation_intent']['historical_only'])
        compact=compact_navigation_context(dict(memory=full))['memory']
        self.assertNotIn('text',compact['progress']['prior_navigation_intent'])
        self.assertEqual(compact['instruction_route_references'][0][
            'source_observation_id'],'obs_0019')

    def test_historical_junction_frame_precedes_kitchen_detour_frame(self):
        m=fixture()
        baseline=navigation_visual_anchors(m,'obs_0052',max_pairs=3)
        self.assertEqual(baseline[0]['observation_id'],'obs_0012')
        m.note_route_reference(update(
            outcome='Kitchen route was wrong.',outcome_id='obs_0051'))
        photos=navigation_visual_anchors(m,'obs_0052',max_pairs=3)
        self.assertEqual(photos[0]['observation_id'],'obs_0019')
        self.assertEqual(photos[0]['selection_reason'],
                         'instruction_relative_route_reference')
        self.assertEqual(len({f['observation_id'] for f in photos}),len(photos))
        self.assertTrue(all(f['observation_id']!='obs_0052' for f in photos))
        five=navigation_visual_anchors(m,'obs_0052',max_pairs=5)
        self.assertIn('obs_0051',[f['observation_id'] for f in five])

    def test_approach_yaw_is_from_last_executed_forward_not_post_turn_camera(self):
        m=fixture()
        source=next(o for o in m.observations
                    if o['observation_id']=='obs_0019')
        # Stair exit source view may already face a new direction. Preserve
        # the earlier actual forward orientation as a separate historical
        # view; it is not a current motor bearing or measured descent.
        source['step']=19
        source['pose']['heading']=math.radians(270)
        m.note_route_reference(update())
        ref=m.route_references[0]
        self.assertEqual(ref['source_heading_deg'],270)
        self.assertEqual(ref['approach_heading_deg'],225)
        self.assertEqual(ref['approach_observation_id'],'obs_0018')
        frames=navigation_visual_anchors(m,'obs_0052',max_pairs=3)
        self.assertEqual([v['observation_id'] for v in frames[:2]],
                         ['obs_0019','obs_0018'])
        wire=project_model_angles(dict(memory=compact_navigation_context(
            dict(memory=mission_evidence(m,m.current['pose'])))['memory']),45)
        sent=wire['memory']['instruction_route_references'][0]
        self.assertEqual(sent['source_view_offset_deg_from_current'],-135)
        self.assertEqual(sent['historical_approach_view_offset_deg_from_current'],
                         -180)
        self.assertNotIn('approach_heading_deg',sent)

    def test_invalid_or_unseen_reference_is_not_written(self):
        m=fixture()
        missing=update()
        missing['source_observation_id']='obs_fake'
        self.assertIsNone(m.note_route_reference(missing))
        self.assertFalse(m.route_references)
        invalid=update(outcome='Wrong garage',outcome_id='obs_fake')
        self.assertIsNone(m.note_route_reference(invalid))
        self.assertFalse(m.route_references)
        self.assertEqual(m.events[-1]['event'],'route_reference_rejected')

    def test_actual_navigation_request_keeps_route_reference_relative_only(self):
        r,_=test_runtime()
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        current=r.memory.current['observation_id']
        reference=update()
        reference['source_observation_id']=current
        r.memory.note_navigation_assessment(
            dict(route_reference_update=reference))
        r.navigation_agent.decide()
        payload=r.client.complete.call_args.args[3]
        refs=payload['memory']['instruction_route_references']
        self.assertEqual(len(refs),1)
        self.assertEqual(refs[0]['source_observation_id'],current)
        self.assertNotIn('source_heading_deg',json.dumps(payload))
        self.assertEqual(refs[0]['source_view_offset_deg_from_current'],0)
        self.assertTrue(refs[0][
            'reference_is_historical_view_not_current_motor_bearing'])

    def test_invalid_source_id_is_recoverable_api_format_error(self):
        r,_=test_runtime()
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        invalid=proposal(update())
        invalid['route_reference_update']['source_observation_id']='obs_fake'
        def request(*args,**kwargs):
            kwargs['validate'](invalid)
        r.client.complete.side_effect=request
        with self.assertRaisesRegex(APIError,'route_reference_unknown_source'):
            r.navigation_agent.decide()
        self.assertEqual(r.memory.route_references,[])

    def test_route_reference_record_is_individually_retrievable(self):
        env=SimpleNamespace(camera=camera())
        m=MemoryService(env)
        m.observations=fixture().observations
        m.current=m.observations[-1]
        m.note_route_reference(update())
        result=m.read_memory_record('route_ref_0000')
        self.assertEqual(result['original_visual_frame'],'obs_0019')
        self.assertTrue(result['instruction_relation_is_historical_not_motor_command'])
        self.assertFalse(result['completion_verified'])

    def test_prompt_does_not_block_all_stair_turns_or_create_stage_machine(self):
        self.assertIn('Curved-stair steering while still descending is fine',
                      MISSION_PROMPT)
        self.assertIn('route_reference_update',MISSION_PROMPT)
        self.assertIn('last step',SCAN_GUIDANCE_PROMPT)
        self.assertIn('Only you choose movement',MISSION_PROMPT)
        self.assertLess(len(MISSION_PROMPT),5000)


if __name__=='__main__':
    unittest.main()
