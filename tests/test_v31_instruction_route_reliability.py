"""Phase 2 instruction-route regressions based on real failed provider responses."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from jsonschema import Draft7Validator
from havln_agent.api_client import APIClient, model_endpoints
from havln_agent.contracts import (
    registry, navigation_schema, navigation_envelope, compact_navigation_protocol,
    normalize_navigation_route_update)
from havln_agent.navigation_agent import MISSION_PROMPT
from havln_agent.mission_history import mission_evidence
from havln_agent.navigation_history import navigation_visual_anchors
from test_api_parameter_compat import settings, accepted
from test_forward_preflight_v9 import camera
from test_v23_route_evidence import fixture


def decision(reference):
    return dict(observed_evidence=['The visible stair exit is a candidate.'],
                decision_analysis='Continue instruction-relative branch.',
                route_reference_update=reference, tool='turn_and_observe',
                args=dict(turn_direction='RIGHT',turn_degrees=30,look='none'))


class Phase2InstructionRouteReliability(unittest.TestCase):
    def test_provider_default_temperature_does_not_send_parameter(self):
        config=settings()
        config['temperature']=None
        self.assertIsNone(model_endpoints(config)[0]['temperature'])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                'os.environ',{'COMPAT_TEST_KEY':'offline-only'}):
            client=APIClient({'navigation_agent':config,
                              'visual_perception':config},tmp)
            client.begin_episode('phase2',8,Path(tmp)/'api.jsonl')
            bodies=[]
            client._request=Mock(side_effect=lambda req,timeout,record: (
                bodies.append(json.loads(req.data)),accepted())[1])
            result=client.complete('navigation_agent','test','JSON',{},
                {'type':'object','properties':{'ok':{'type':'boolean'}},
                 'required':['ok'],'additionalProperties':False})
            self.assertEqual(result,{'ok':True})
            self.assertEqual(client.calls,1)
            self.assertNotIn('temperature',bodies[0])

    def test_real_phase2_aliases_are_canonical_without_changing_motor_decision(self):
        ids={'obs_0028','obs_0035'}
        old=decision(dict(source_observation_id='obs_0028',
            instruction_relative_relation='LEFT relative to staircase descent',
            junction_description='Flat landing near an interior opening',
            tried_route=dict(outcome_observation_id='obs_0035',
                             outcome='Turned toward close wall, not yet traversed')))
        before=copy.deepcopy(old)
        result=normalize_navigation_route_update(old,ids)
        self.assertEqual(old,before)
        self.assertEqual(result['args'],old['args'])
        self.assertEqual(result['tool'],old['tool'])
        ref=result['route_reference_update']
        self.assertEqual(ref['instruction_relation'],
                         'LEFT relative to staircase descent')
        self.assertEqual(ref['place_description'],
                         'Flat landing near an interior opening')
        self.assertEqual(ref['route_attempt_outcome'],
                         'Turned toward close wall, not yet traversed')
        self.assertEqual(ref['outcome_observation_id'],'obs_0035')
        self.assertTrue(Draft7Validator(navigation_schema(
            registry(camera(),3))).is_valid(result))

    def test_unusable_reference_is_dropped_but_not_movement(self):
        for ref in (
            {'source_observation_id':'obs_unknown',
             'instruction_relative_relation':'LEFT'},
            {'source_observation_id':'obs_0028'},
            {'source_observation_id':'obs_0028',
             'instruction_relation':'LEFT','instruction_relative_relation':'RIGHT'},
            {'source_observation_id':'obs_0028',
             'instruction_relation':'LEFT','turn_direction':'RIGHT',
             'place_description':'Route','evidence_status':'candidate'},
            'broken',
        ):
            raw=decision(ref)
            normalized=normalize_navigation_route_update(raw,{'obs_0028'})
            self.assertEqual(normalized['args'],raw['args'])
            self.assertEqual(normalized['tool'],raw['tool'])
            self.assertTrue(Draft7Validator(navigation_schema(
                registry(camera(),3))).is_valid(normalized))
            if not isinstance(ref,dict) or 'instruction_relation' not in ref and (
                    'instruction_relative_relation' not in ref):
                self.assertNotIn('route_reference_update',normalized)

    def test_valid_action_and_optional_alias_survive_one_api_call_and_save(self):
        opts=settings()
        opts['temperature']=None
        raw=decision(dict(
            source_observation_id='obs_0019',
            instruction_relative_relation='LEFT OF DESCENT direction'))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                'os.environ',{'COMPAT_TEST_KEY':'offline-only'}):
            client=APIClient({'navigation_agent':opts,
                              'visual_perception':opts},tmp)
            client.begin_episode('phase2',8,Path(tmp)/'api.jsonl')
            client._request=Mock(return_value=accepted(json.dumps(raw)))
            tools=registry(camera(),3)
            result=client.complete('navigation_agent','nav','JSON',{},
                navigation_schema(tools),
                validation_schema=navigation_envelope(tools),
                protocol=compact_navigation_protocol(tools),
                normalize=lambda x: normalize_navigation_route_update(
                    x,{'obs_0019'}))
            self.assertEqual(client.calls,1)
            self.assertEqual(result['args'],raw['args'])
            self.assertEqual(result['route_reference_update']['instruction_relation'],
                             'LEFT OF DESCENT direction')
            memory=fixture()
            memory.note_navigation_assessment(result)
            self.assertEqual(len(memory.route_references),1)
            self.assertEqual(memory.route_references[0]['source_observation_id'],
                             'obs_0019')

    def test_reference_root_survives_later_forks_and_photo_selection(self):
        memory=fixture()
        for source in ['obs_0019','obs_0041','obs_0050','obs_0051']:
            memory.note_route_reference(dict(
                source_observation_id=source,
                instruction_relation='Original instruction-relative route',
                evidence_status='candidate'))
        refs=mission_evidence(memory,memory.current['pose'])[
            'instruction_route_references']
        self.assertEqual(refs[0]['source_observation_id'],'obs_0019')
        self.assertIn('obs_0051',[r['source_observation_id'] for r in refs])
        frames=navigation_visual_anchors(memory,'obs_0052',max_pairs=5)
        self.assertEqual(frames[0]['observation_id'],'obs_0019')

    def test_instruction_analysis_preserves_model_autonomy(self):
        self.assertIn('COMPLETE instruction as an ordered route',MISSION_PROMPT)
        self.assertIn('BEFORE/AFTER/UNTIL',MISSION_PROMPT)
        self.assertIn('not necessarily the current camera',MISSION_PROMPT)
        self.assertIn('SEMANTIC route/doorway',MISSION_PROMPT)
        self.assertIn('An opposite local turn',MISSION_PROMPT)
        self.assertNotIn('must immediately turn LEFT',MISSION_PROMPT)
        self.assertLess(len(MISSION_PROMPT),5000)


if __name__=='__main__':
    unittest.main()
