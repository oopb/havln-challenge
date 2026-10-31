"""Active observation contract: one bounded delegation, valid visual evidence, no motion."""
import copy
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from jsonschema import Draft7Validator

from havln_agent.active_observation import ObservationController, meaningful_candidate, comparable_scan_pose
from havln_agent.contracts import registry
from havln_agent.api_client import APIError
from havln_agent.visual_perception import SEARCH_SCHEMA, VisualPerception
from havln_agent.vision_protocol import coordinate_schema, normalize_coordinates


def visual(found=False, suggestion='continue'):
    return dict(target_visible=found,
        target_type='stair_entry' if found else 'none',
        confidence='high' if found else 'low',
        box=[.2,.25,.6,.8] if found else None,
        evidence='Wooden downward stair entry visible' if found else '',
        uncertainty='',suggested_next_view=suggestion)


class FakeVisual:
    def __init__(self, replies):
        self.replies=list(replies)
        self.calls=[]

    def search_view(self, image, source, objective, targets):
        self.calls.append((source['observation_id'],objective,list(targets)))
        reply=self.replies.pop(0)
        if isinstance(reply,Exception):
            raise reply
        return copy.deepcopy(reply)


class FakeRuntime:
    def __init__(self, replies, fail_on=None):
        self.camera=SimpleNamespace(turn_angle=15.,tilt_angle=30.)
        self.heading=0.
        self.pitch_deg=0.
        self.step_calls=[]
        self.fail_on=fail_on
        self.motion_controller=SimpleNamespace(target=None,pose=self.pose)
        self.memory=SimpleNamespace(
            current=dict(observation_id='obs_0000',new_visual_information=True,
                         revisited_view_ids=[]),
            observation_searches=[],task=SimpleNamespace(current_subgoal_index=0),
            note_semantics=Mock())
        self.observation=SimpleNamespace(rgb=None,depth=None)
        self.visual_perception=FakeVisual(replies)
        self.event=Mock()
        self.done=False

    def pose(self):
        return dict(x=0.,y=0.,z=0.,heading=self.heading,pitch_deg=self.pitch_deg,
                    motion_epoch=0)

    def source(self):
        return dict(observation_id=self.memory.current['observation_id'],
            pose=self.pose(),subgoal=0,captured_at=1)

    def geometry(self):
        return SimpleNamespace(assess_step=lambda:dict(
            state='BLOCKED',reason='occupied_next_step',
            depth=dict(clearance=.12),support=dict(status='UNKNOWN')))

    def _step(self, action):
        self.step_calls.append(action)
        if self.fail_on is not None and len(self.step_calls)>=self.fail_on:
            return dict(status='BUDGET_LIMIT',reason='action_budget_reserved_for_stop')
        if action=='TURN_LEFT': self.heading+=math.radians(15)
        elif action=='TURN_RIGHT': self.heading-=math.radians(15)
        elif action=='LOOK_DOWN': self.pitch_deg-=30
        elif action=='LOOK_UP': self.pitch_deg+=30
        else: raise AssertionError('Observation subtask attempted movement')
        self.memory.current=dict(
            observation_id='obs_%04d'%len(self.step_calls),
            new_visual_information=True,revisited_view_ids=[])
        return dict(status='EXECUTED')


class ActiveObservationTests(unittest.TestCase):
    def test_stops_early_on_localized_target_after_two_turns(self):
        r=FakeRuntime([visual(found=True)])
        answer=ObservationController(r).scan(
            'Find the downward stair entry',['stair_entry'],'left',120)
        self.assertEqual(answer['status'],'OK')
        self.assertEqual(answer['outcome'],'TARGET_CANDIDATE_FOUND')
        self.assertEqual(answer['covered_turn_deg'],30)
        self.assertEqual(r.step_calls,['TURN_LEFT','TURN_LEFT'])
        self.assertEqual(len(r.visual_perception.calls),1)
        candidate=answer['best_candidate']
        self.assertEqual(candidate['observation_id'],'obs_0002')
        self.assertEqual(candidate['heading_deg'],30)
        self.assertFalse(candidate['usable_for_live_safety'])
        self.assertFalse(candidate['geometry']['motion_authorized'])
        r.memory.note_semantics.assert_called_once()

    def test_full_sweep_uses_checkpoint_vlm_not_per_15degree_step(self):
        r=FakeRuntime([visual(),visual(),visual()])
        result=ObservationController(r).scan(
            'Find a corridor',['corridor'],'right',90,checkpoint_degrees=30)
        self.assertEqual(result['outcome'],'SEARCH_REGION_COVERED_NO_MATCH')
        self.assertEqual(result['covered_turn_deg'],90)
        self.assertEqual(r.step_calls,['TURN_RIGHT']*6)
        self.assertEqual(len(r.visual_perception.calls),3)
        self.assertEqual([p['heading_deg'] for p in result['checkpoints']],[-30.,-60.,-90.])
        self.assertEqual(result['best_candidate'],None)
        self.assertEqual(len(r.memory.observation_searches),1)

    def test_vlm_requests_at_most_one_pitch_refinement(self):
        r=FakeRuntime([visual(suggestion='look_down'),
                       visual(suggestion='look_down'),visual()])
        result=ObservationController(r).scan(
            'Find stairs',['stairs_down'],'left',60,checkpoint_degrees=30)
        self.assertEqual(result['outcome'],'SEARCH_REGION_COVERED_NO_MATCH')
        self.assertEqual(r.step_calls.count('LOOK_DOWN'),1)
        self.assertEqual(r.step_calls.count('TURN_LEFT'),4)
        self.assertEqual(len(r.visual_perception.calls),3)
        self.assertEqual(r.pitch_deg,-30.)

    def test_active_navigation_can_pause_to_scan_without_cancelling_path(self):
        r=FakeRuntime([visual(found=True)])
        r.motion_controller.target=(.2,.3)
        result=ObservationController(r).scan(
            'Find stairs',['stair_entry'],'left',30)
        self.assertEqual(result['status'],'OK')
        self.assertTrue(result['paused_waypoint_preserved'])
        self.assertEqual(r.motion_controller.target,(.2,.3))
        self.assertEqual(r.step_calls,['TURN_LEFT','TURN_LEFT'])

    def test_budget_limit_preserves_partial_scan(self):
        r=FakeRuntime([],fail_on=2)
        result=ObservationController(r).scan(
            'Find stairs',['stair_entry'],'left',90)
        self.assertEqual(result['status'],'PARTIAL')
        self.assertEqual(result['outcome'],'INTERRUPTED')
        self.assertEqual(result['covered_turn_deg'],15)
        self.assertEqual(len(r.visual_perception.calls),0)

    def test_failed_vlm_returns_vision_unavailable_with_observation_receipt(self):
        r=FakeRuntime([APIError('temporary failure')])
        result=ObservationController(r).scan(
            'Find stairs',['stair_entry'],'left',90)
        self.assertEqual(result['outcome'],'VISION_UNAVAILABLE')
        self.assertEqual(result['status'],'PARTIAL')
        self.assertEqual(result['covered_turn_deg'],30)
        self.assertEqual(len(result['checkpoints']),0)

    def test_reuse_requires_same_task_and_same_explicit_objective(self):
        r=FakeRuntime([visual(found=True)])
        first=ObservationController(r).scan(
            'Find staircase entry',['stair_entry'],'left',30)
        origin=first['checkpoints'][0]['observation_id']
        r.visual_perception.replies=[]
        old_step=r._step
        def repeating_step(action):
            outcome=old_step(action)
            r.memory.current['revisited_view_ids']=[origin]
            return outcome
        r._step=repeating_step
        second=ObservationController(r).scan(
            'Find staircase entry',['stair_entry'],'left',30)
        self.assertEqual(second['outcome'],'TARGET_CANDIDATE_FOUND')
        self.assertEqual(len(r.visual_perception.calls),1)
        self.assertEqual(second['best_candidate']['semantic_source_observation_id'],origin)
        self.assertEqual(second['best_candidate']['evidence_origin'],
                         'reused_static_visual_hypothesis')
        # Identical target type is NOT enough if the task objective changed.
        r.visual_perception.replies=[visual()]
        third=ObservationController(r).scan(
            'Locate the lower landing',['stair_entry'],'left',30)
        self.assertEqual(third['outcome'],'SEARCH_REGION_COVERED_NO_MATCH')
        self.assertEqual(len(r.visual_perception.calls),2)

    def test_no_candidate_without_visible_box_and_confidence(self):
        self.assertFalse(meaningful_candidate(dict(visual(found=True),box=None),['stair_entry']))
        self.assertFalse(meaningful_candidate(dict(visual(found=True),confidence='low'),['stair_entry']))
        self.assertFalse(meaningful_candidate(visual(found=True),['corridor']))

    def test_vlm_box_unit_contract_normalizes_and_requires_evidence(self):
        raw=dict(coordinate_space='image_1000_xyxy',
            target_visible=True,target_type='stair_entry',confidence='high',
            box=[200,250,600,800],evidence='Visible stair entry',
            uncertainty='',suggested_next_view='continue')
        provider=coordinate_schema(SEARCH_SCHEMA,'image_1000_xyxy')
        Draft7Validator(provider).validate(raw)
        normalized=normalize_coordinates(raw,SEARCH_SCHEMA)
        Draft7Validator(SEARCH_SCHEMA).validate(normalized)
        self.assertEqual(normalized['box'],[.2,.25,.6,.8])
        self.assertTrue(meaningful_candidate(normalized,['stair_entry']))
        self.assertFalse(Draft7Validator(provider).is_valid(
            dict(raw,box=[.2,.25,.6,.8])))

    def test_vlm_checkpoint_uses_current_image_and_single_target_question(self):
        from unittest.mock import patch
        runtime=SimpleNamespace(client=None)
        visual_agent=VisualPerception(runtime)
        response=visual(found=True)
        with patch('havln_agent.visual_perception.image_url', return_value='data:image/png;base64,AA'):
            with patch.object(visual_agent,'complete',return_value=response) as call:
                result=visual_agent.search_view(
                    SimpleNamespace(rgb=None),dict(observation_id='obs_0042'),
                    'Find the down stair entry',['stair_entry'])
        self.assertEqual(result,response)
        args,kwargs=call.call_args
        self.assertEqual(args[0:2],('visual_perception','active_observation'))
        self.assertEqual(args[3]['observation_id'],'obs_0042')
        self.assertEqual(args[3]['target_types'],['stair_entry'])
        self.assertTrue(kwargs['cache'])
        self.assertEqual(kwargs['result_schema'],SEARCH_SCHEMA)
        self.assertIn('Do not infer metric distance',args[2])

    def test_exact_recent_scan_is_idempotent_without_extra_turns(self):
        r=FakeRuntime([visual()])
        first=ObservationController(r).scan(
            'Find a corridor',['corridor'],'left',30)
        self.assertEqual(first['outcome'],'SEARCH_REGION_COVERED_NO_MATCH')
        self.assertEqual(len(r.step_calls),2)
        r.heading=2*math.pi  # heading wrap is same pose
        second=ObservationController(r).scan(
            'Find a corridor',['corridor'],'left',30)
        self.assertEqual(second['outcome'],'REPEATED_SEARCH_NO_NEW_VIEW')
        self.assertEqual(second['status'],'NO_NEW_INFORMATION')
        self.assertEqual(second['covered_turn_deg'],0)
        self.assertFalse(second['usable_for_live_safety'])
        self.assertEqual(len(r.step_calls),2)
        self.assertEqual(len(r.visual_perception.calls),1)
        self.assertEqual(len(r.memory.observation_searches),1)
        # A meaningful translation invalidates the former view coverage.
        self.assertFalse(comparable_scan_pose(
            dict(r.pose(),x=.25),r.pose()))

    def test_tool_contract_and_atomic_permission(self):
        camera=SimpleNamespace(turn_angle=15.,tilt_angle=30.)
        specs=registry(camera,3)
        spec=specs['observe_for_goal']
        self.assertEqual(spec['permission'],'NAVIGATION')
        proper=dict(objective='Find stair entry',target_types=['stair_entry'],
            direction='left',sweep_degrees=90,checkpoint_degrees=30)
        Draft7Validator(spec['parameters']).validate(proper)
        self.assertFalse(Draft7Validator(spec['parameters']).is_valid(dict(
            proper,sweep_degrees=360)))
        self.assertTrue(Draft7Validator(spec['parameters']).is_valid(dict(
            proper,target_types=['person'])))
        self.assertFalse(Draft7Validator(spec['parameters']).is_valid(dict(
            proper,target_types=[''])))


if __name__=='__main__':
    unittest.main()
