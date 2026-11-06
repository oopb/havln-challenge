"""VLM owns camera search; NavigationAgent receives only compact evidence."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

from jsonschema import Draft7Validator

from havln_agent.active_observation import ObservationController
from havln_agent.visual_perception import (
    SCAN_GUIDANCE_SCHEMA,VisualPerception)
from havln_agent.contexts import compact_navigation_context
from havln_agent.navigation_agent import NavigationAgent
from havln_agent.contracts import registry
from test_active_observation import FakeRuntime


def guidance(action='finish',turn=0,kind='UNKNOWN',visible=False):
    return dict(target_visible=visible,
        target_type='entry' if visible else 'none',
        confidence='high' if visible else 'low',
        box=[.1,.2,.7,.9] if visible else None,
        evidence='Walkable opening visible' if visible else '',
        uncertainty='',target_kind=kind,candidate_approach=False,
        next_action=action,turn_deg=turn,reason='search policy')


class VLMGuide:
    def __init__(self,answers):
        self.answers=list(answers)
        self.calls=[]
    def guide_scan_view(self,frame,source,objective,targets,history,remaining_turn_deg,
                        prior_targets=None,observed_yaw_pitch=None,view_correction=None,**kwargs):
        self.calls.append(dict(observation_id=source['observation_id'],
            objective=objective,targets=targets,history=history,
            remaining_turn_deg=remaining_turn_deg,
            observed_yaw_pitch=observed_yaw_pitch,
            view_correction=view_correction))
        return self.answers.pop(0)


class VLMObservationTests(unittest.TestCase):
    def test_vlm_drives_multi_turn_then_pitch_and_returns_actionable_evidence(self):
        r=FakeRuntime([])
        r.visual_perception=VLMGuide([
            guidance('turn_left',90),
            guidance('look_down',0,kind='CUE',visible=True),
            guidance('finish',0,kind='ENTRY',visible=True)])
        r.motion_controller.target=(.2,.3)
        result=ObservationController(r).scan_vlm(
            'Find a walkable opening',['entry'],360)
        self.assertEqual(result['status'],'OK')
        self.assertEqual(result['outcome'],'TARGET_CANDIDATE_FOUND')
        self.assertEqual(result['observation_owner'],'visual_perception_vlm')
        self.assertEqual(r.step_calls,['TURN_LEFT']*6+['LOOK_DOWN'])
        self.assertEqual(len(r.visual_perception.calls),3)
        self.assertEqual(result['covered_turn_deg'],90)
        self.assertEqual(result['best_candidate']['heading_deg'],90)
        self.assertEqual(result['best_candidate']['pitch_deg'],-30)
        self.assertTrue(result['paused_waypoint_preserved'])
        self.assertEqual(r.motion_controller.target,(.2,.3))
        self.assertNotIn('MOVE_FORWARD',r.step_calls)
        self.assertEqual(r.visual_perception.calls[1]['history'][0]['observation_id'],'obs_0000')

    def test_cue_only_is_not_falsely_an_actionable_entry(self):
        r=FakeRuntime([])
        r.visual_perception=VLMGuide([
            guidance('look_down',kind='CUE',visible=True),
            guidance('finish',kind='CUE',visible=True)])
        result=ObservationController(r).scan_vlm('Find entry',['entry'])
        self.assertEqual(result['status'],'OK')
        self.assertEqual(result['outcome'],'CUE_ONLY')
        self.assertIsNone(result['best_candidate'])
        self.assertEqual(result['best_cue']['target_kind'],'CUE')
        self.assertEqual(r.step_calls,['LOOK_DOWN'])
        r.memory.note_semantics.assert_not_called()

    def test_revisiting_same_heading_pitch_yields_to_navigation(self):
        r=FakeRuntime([])
        r.visual_perception=VLMGuide([
            guidance('turn_left',30), guidance('turn_right',30),
            guidance('look_down'),guidance('finish')])
        result=ObservationController(r).scan_vlm(
            'Search for entry',['entry'],sweep_degrees=120)
        self.assertEqual(result['status'],'OK')
        self.assertEqual(result['outcome'],'VLM_SEARCH_FINISHED')
        self.assertEqual(result['view_replan_attempts'],1)
        self.assertEqual(r.step_calls,['TURN_LEFT','TURN_LEFT','LOOK_DOWN'])
        self.assertEqual(len(r.visual_perception.calls),4)
        self.assertEqual(r.visual_perception.calls[2]['view_correction']['rejected_action'],
                         'turn_right')
        self.assertEqual(r.visual_perception.calls[2]['view_correction']['reason'],
                         'already_inspected_heading_pitch')

    def test_repeated_invalid_view_is_bounded_without_turn_loop(self):
        r=FakeRuntime([])
        r.visual_perception=VLMGuide([
            guidance('turn_left',30),
            guidance('turn_right',30),
            guidance('turn_right',30),
            guidance('turn_right',30)])
        result=ObservationController(r).scan_vlm(
            'Search entry',['entry'],sweep_degrees=120)
        self.assertEqual(result['outcome'],'REPEATED_VIEW_REQUEST')
        self.assertEqual(result['status'],'NO_NEW_INFORMATION')
        self.assertEqual(result['view_replan_attempts'],3)
        self.assertEqual(r.step_calls,['TURN_LEFT','TURN_LEFT'])
        self.assertEqual(len(r.visual_perception.calls),4)

    def test_visual_candidate_is_handed_to_typed_persistent_memory(self):
        r=FakeRuntime([])
        r.memory.note_active_observation_target=Mock()
        r.visual_perception=VLMGuide([
            guidance('finish',kind='ENTRY',visible=True)])
        result=ObservationController(r).scan_vlm(
            'Locate stair entry',['entry'],sweep_degrees=90)
        self.assertEqual(result['outcome'],'TARGET_CANDIDATE_FOUND')
        args,kwargs=r.memory.note_active_observation_target.call_args
        self.assertEqual(args[0]['observation_id'],'obs_0000')
        self.assertEqual(args[0]['target_kind'],'ENTRY')
        self.assertEqual(args[0]['relative_turn_deg_from_current'],0)
        self.assertEqual(args[1],'Locate stair entry')
        self.assertEqual(args[2],['entry'])

    def test_non_entry_object_can_be_a_visual_target(self):
        r=FakeRuntime([])
        r.visual_perception=VLMGuide([
            dict(guidance('finish',kind='TARGET',visible=True),target_type='person')])
        result=ObservationController(r).scan_vlm(
            'Locate a person',['person'],sweep_degrees=90)
        self.assertEqual(result['status'],'OK')
        self.assertEqual(result['outcome'],'TARGET_CANDIDATE_FOUND')
        self.assertEqual(result['best_candidate']['target_kind'],'TARGET')
        self.assertEqual(r.step_calls,[])

    def test_unregistered_changed_view_must_not_reuse_prior_scan(self):
        from havln_agent.active_observation import comparable_scan_pose
        pose=dict(x=0.,z=0.,heading=0.,pitch_deg=0,
                  motion_epoch=0,viewpoint_generation=1)
        self.assertFalse(comparable_scan_pose(
            dict(pose,viewpoint_generation=0),pose))

    def test_budget_interrupt_preserves_evidence_without_claiming_arrival(self):
        r=FakeRuntime([],fail_on=2)
        r.visual_perception=VLMGuide([guidance('turn_right',90)])
        result=ObservationController(r).scan_vlm(
            'Locate stairs',['stairs'],sweep_degrees=360)
        self.assertEqual(result['status'],'PARTIAL')
        self.assertEqual(result['outcome'],'INTERRUPTED')
        self.assertEqual(result['covered_turn_deg'],15)
        self.assertFalse(result['usable_for_live_safety'])
        self.assertEqual(r.step_calls,['TURN_RIGHT','TURN_RIGHT'])

    def test_vlm_request_carries_task_angle_and_history_not_navigation_turns(self):
        task=SimpleNamespace(instruction='Go down stairs',
            current_subgoal_index=0,subgoals=[dict(action='descend',target='stairs')])
        vp=VisualPerception(SimpleNamespace(memory=SimpleNamespace(task=task),client=None))
        src=dict(observation_id='obs_0042',
            pose=dict(heading=math.radians(30),pitch_deg=-30))
        with patch('havln_agent.visual_perception.image_url',return_value='data:image/jpeg;base64,AA'):
            with patch.object(vp,'complete',return_value=guidance('turn_left',45)) as call:
                v=vp.guide_scan_view(SimpleNamespace(),src,'Find entry',
                    ['entry'],[dict(observation_id='obs_0040',heading_deg=0)],225)
        args,kwargs=call.call_args
        self.assertEqual(args[0:2],('visual_perception','vlm_guided_observation'))
        self.assertEqual(args[3]['instruction'],'Go down stairs')
        self.assertEqual(args[3]['heading_deg'],30)
        self.assertEqual(args[3]['previously_inspected_views'][0]['observation_id'],'obs_0040')
        self.assertEqual(kwargs['result_schema'],SCAN_GUIDANCE_SCHEMA)
        self.assertFalse(kwargs['cache'])
        self.assertEqual(v['next_action'],'turn_left')
        self.assertTrue(Draft7Validator(SCAN_GUIDANCE_SCHEMA).is_valid(v))

    def test_navigation_offers_vlm_scan_not_micro_turns(self):
        camera=SimpleNamespace(turn_angle=15,tilt_angle=30)
        specs=registry(camera,3)
        client=Mock()
        client.complete.return_value=dict(tool='observe_for_goal',
            args=dict(objective='Find walkable stairs entry',
                target_types=['stairs']))
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value=specs),
            memory=SimpleNamespace(keyframes={},
                task=SimpleNamespace(instruction='Descend stairs')))
        state=dict(goal=dict(action='descend',target='stairs'),
            navigation_memory=dict(visual_target_memory=[dict(
                id='v',status='CUE_ONLY',
                visual_evidence='A'*3000,missing_evidence='look_down')]),
            observation_stagnation=dict(repeated_low_information=False),
            pose=dict(heading_deg=0),current_view=dict(observation_id='obs1'))
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value=state):
            proposal=NavigationAgent(runtime).decide()
        self.assertEqual(proposal['tool'],'observe_for_goal')
        args=client.complete.call_args.args
        exposed={x['properties']['tool']['const'] for x in args[4]['oneOf']}
        self.assertNotIn('turn_and_observe',exposed)
        self.assertIn('observe_for_goal',exposed)
        offered=next(x for x in args[4]['oneOf']
                     if x['properties']['tool']['const']=='observe_for_goal')
        parameters=offered['properties']['args']
        self.assertNotIn('direction',parameters['properties'])
        self.assertNotIn('look',parameters['properties'])
        self.assertLess(len(args[2]),7000)
        self.assertLess(len(str(args[3])),5000)
        self.assertEqual(args[3]['navigation_memory']['visual_targets'][0]['visual_evidence'],'A'*150)
        self.assertNotIn('local_exploration',args[3])


if __name__=='__main__':
    unittest.main()
