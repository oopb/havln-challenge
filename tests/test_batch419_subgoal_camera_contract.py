"""Batch 4-19: sg0 -> sg1 task scope and repeated impossible LOOK regressions."""
import math
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jsonschema import Draft7Validator

from havln_agent.active_observation import ObservationController
from havln_agent.api_client import APIError
from havln_agent.contexts import (
    latest_observation_for_subgoal, last_tool_result_for_subgoal)
from havln_agent.contracts import registry, navigation_schema
from havln_agent.memory_records import MemoryRecords
from havln_agent.navigation_agent import NavigationAgent
from havln_agent.runtime import AgentRuntime
from havln_agent.tool_menu import navigation_tool_menu
from havln_agent.visual_perception import VisualPerception


def goals():
    return [
        dict(action='Descend stairs',target='Bottom of stairs',
             landmark='Stairs',
             completion_description='Robot physically at the lower landing'),
        dict(action='Make a right turn',target='Facing after right turn',
             landmark='Bottom of stairs',
             completion_description='Robot has completed a right turn'),
        dict(action='Walk into foyer',target='Foyer',
             landmark='Foyer',completion_description='Robot inside foyer'),
    ]


class TestBatch419SubgoalCameraContract(unittest.TestCase):
    def test_dynamic_observation_schema_rejects_impossible_pitch(self):
        camera=SimpleNamespace(turn_angle=15,tilt_angle=30)
        low=registry(camera,3,pitch_deg=-60)
        param=low['observe_for_goal']['parameters']
        self.assertEqual(param['properties']['look']['enum'],['none','up'])
        self.assertTrue(Draft7Validator(param).is_valid(dict(
            objective='Verify current sg1',target_types=['turn'],
            mode='VERIFY',look='up')))
        self.assertFalse(Draft7Validator(param).is_valid(dict(
            objective='Verify current sg1',target_types=['turn'],
            mode='VERIFY',look='down')))
        # sg1 is an orientation task, so the full model-facing tool menu
        # intentionally retains LOOK; it MUST still be calibrated.
        exposed=navigation_tool_menu(low,{},orientation=True)
        self.assertEqual(exposed['observe_for_goal']['parameters']
                         ['properties']['look']['enum'],['none','up'])
        self.assertFalse(Draft7Validator(navigation_schema(exposed)).is_valid(
            dict(tool='observe_for_goal',args=dict(mode='VERIFY',
                objective='Check current milestone',target_types=['turn'],
                look='down'))))
        high=registry(camera,3,pitch_deg=60)
        self.assertEqual(high['observe_for_goal']['parameters']
                         ['properties']['look']['enum'],['none','down'])
        middle=registry(camera,3,pitch_deg=-30)
        self.assertEqual(middle['observe_for_goal']['parameters']
                         ['properties']['look']['enum'],['none','up','down'])

    def test_runtime_verify_overrides_stale_sg0_objective_and_labels(self):
        scan=Mock(return_value=dict(status='OK',
            outcome='SUBGOAL_NOT_YET',verification=dict(status='NOT_YET')))
        task=SimpleNamespace(current_subgoal_index=1,subgoals=goals(),
                             completed_subgoals=[0])
        rt=SimpleNamespace(memory=SimpleNamespace(task=task),
            active_subgoal_id='sg_1',camera=SimpleNamespace(tilt_angle=30),
            pitch_deg=-60,observation_controller=SimpleNamespace(scan_vlm=scan),
            event=Mock(),_cancel_local_navigation=Mock(),set_semantic_goal=Mock())
        result=AgentRuntime._observe_for_goal(rt,
            objective='Has sg0 fully descended the stairs?',
            target_types=['bottom of stairs','stair treads'],
            mode='VERIFY',look='down')
        self.assertEqual(result['status'],'OK')
        self.assertTrue(result['verification_goal']['objective_bound_to_runtime'])
        self.assertTrue(result['verification_goal']['model_objective_overridden'])
        self.assertEqual(result['verification_goal']['subgoal_id'],'sg_1')
        sent_objective=scan.call_args.args[0]
        sent_labels=scan.call_args.args[1]
        self.assertIn('sg_1',sent_objective)
        self.assertIn('completed a right turn',sent_objective)
        self.assertNotIn('Has sg0 fully descended',sent_objective)
        self.assertNotIn('bottom of stairs',sent_labels)
        self.assertEqual(scan.call_args.kwargs['look'],'none')
        rt.event.assert_called_once()
        self.assertEqual(rt.event.call_args.args[0],
                         'observation_initial_pitch_clamped')
        rt.set_semantic_goal.assert_not_called()
        self.assertEqual(task.current_subgoal_index,1)

    def test_current_goal_verified_before_advancing_to_sg2(self):
        task=SimpleNamespace(current_subgoal_index=1,subgoals=goals(),
            completed_subgoals=[0])
        mem=SimpleNamespace(task=task,confirm_subgoal_observation=Mock(
            return_value=dict(status='CONFIRMED',subgoal=1,next_subgoal=2)))
        rt=SimpleNamespace(memory=mem,active_subgoal_id='sg_1',
            camera=SimpleNamespace(tilt_angle=30),pitch_deg=-30,
            observation_controller=SimpleNamespace(scan_vlm=Mock(
                return_value=dict(status='OK',outcome='SUBGOAL_CONFIRMED',
                                  verification=dict(status='CONFIRMED')))),
            event=Mock(),_cancel_local_navigation=Mock(),set_semantic_goal=Mock())
        response=AgentRuntime._observe_for_goal(rt,
            objective='Re-verify sg0',target_types=['stairs'],
            mode='VERIFY')
        self.assertEqual(response['subgoal_transition']['next_subgoal'],2)
        rt.set_semantic_goal.assert_called_once()
        self.assertIn('sg_1',rt.memory.confirm_subgoal_observation.call_args.args[1])

    def test_mismatched_active_subgoal_is_not_verified(self):
        task=SimpleNamespace(current_subgoal_index=1,subgoals=goals())
        scan=Mock()
        rt=SimpleNamespace(memory=SimpleNamespace(task=task),
            active_subgoal_id='sg_0',camera=SimpleNamespace(tilt_angle=30),
            pitch_deg=0,observation_controller=SimpleNamespace(scan_vlm=scan))
        result=AgentRuntime._observe_for_goal(rt,'Check sg0',['stairs'],mode='VERIFY')
        self.assertEqual(result['status'],'STALE_TARGET')
        scan.assert_not_called()

    def test_current_subgoal_context_excludes_stale_sg0_hypotheses(self):
        memory=SimpleNamespace(
            observation_searches=[
                dict(subgoal=0,objective='Check bottom of stairs',outcome='SUBGOAL_CONFIRMED'),
                dict(subgoal=1,objective='Check right turn',outcome='SUBGOAL_NOT_YET')],
            tool_ledger=[dict(subgoal_id='sg_0',result=dict(
                status='OK',outcome='SUBGOAL_CONFIRMED'))])
        self.assertEqual(latest_observation_for_subgoal(memory,1)['objective'],
                         'Check right turn')
        self.assertIsNone(last_tool_result_for_subgoal(memory,1))
        memory.tool_ledger.append(dict(subgoal_id='sg_1',result=dict(
            status='OK',outcome='SUBGOAL_NOT_YET')))
        self.assertEqual(last_tool_result_for_subgoal(memory,1)['outcome'],
                         'SUBGOAL_NOT_YET')
        self.assertIsNone(latest_observation_for_subgoal(memory,2))

    def test_navigation_repairs_impossible_optional_look_before_dispatch(self):
        class Client:
            def complete(self,role,purpose,system,context,schema,**kwargs):
                payload=dict(tool='observe_for_goal',
                    args=dict(mode='VERIFY',
                        objective='Re-verify sg0 bottom of stairs',
                        target_types=['stairs'],look='down'))
                kwargs['validate'](payload)
                self.result=payload
                return payload
        camera=SimpleNamespace(turn_angle=15,tilt_angle=30)
        client=Client()
        rt=SimpleNamespace(client=client,
            navigation_mode='OBSERVE',make_registry=lambda:registry(
                camera,3,pitch_deg=-60),
            memory=SimpleNamespace(episode_id='ep',keyframes={},
                task=SimpleNamespace(instruction='Descend stairs then right')),
            plan_revision=2,active_subgoal_id='sg_1',pitch_deg=-60,
            event=Mock())
        state=dict(task_selection=dict(current_subgoal_index=1),
            goal=dict(id='sg_1',description='Make a right turn'),
            legal_look=['up'],orientation=dict(requirement='right'),
            pose=dict(pitch_deg=-60,motion_epoch=0,
                      place_id='P0',position_xz_m_est=[0,0]))
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value=state):
            response=NavigationAgent(rt).decide()
        self.assertNotIn('look',response['args'])
        self.assertNotIn('look',client.result['args'])
        rt.event.assert_called_once()
        self.assertEqual(rt.event.call_args.args[0],
                         'navigation_invalid_initial_look_removed')

    def test_vlm_verify_receives_actual_sg1_turns_not_earlier_stairs(self):
        task=SimpleNamespace(instruction='Descend then turn right then enter foyer',
            current_subgoal_index=1,subgoals=goals(),completed_subgoals=[0])
        actions=[dict(step=i,subgoal=1,action='TURN_RIGHT',
            result=dict(visual_change=True,
                observation_after='obs_%04d'%(30+i)))
            for i in range(6)]
        camera=SimpleNamespace(normalized=False,turn_angle=15,tilt_angle=30)
        vp=VisualPerception(SimpleNamespace(camera=camera,
            memory=SimpleNamespace(task=task,actions=actions)))
        response=dict(verification_status='CONFIRMED',
            verification_evidence='Robot turned right according to action receipts',
            uncertainty='',next_action='finish',turn_deg=0,
            reason='observed right rotation')
        vp.complete=Mock(return_value=response)
        with patch('havln_agent.visual_perception.image_url',return_value='rgb'):
            result=vp.guide_scan_view(SimpleNamespace(depth=None),
                dict(observation_id='obs_0035',pose=dict(
                    heading=math.radians(135),pitch_deg=-60)),
                'Verify sg1 turn',['right turn'],[],90,
                mode='VERIFY',session=[])
        args,kwargs=vp.complete.call_args
        request=args[3]
        self.assertEqual(request['current_subgoal_id'],'sg_1')
        self.assertEqual(request['subgoal']['action'],'Make a right turn')
        self.assertEqual(request['current_subgoal_action_evidence']
                         ['executed_right_turn_deg'],90)
        self.assertEqual(request['current_subgoal_action_evidence']
                         ['net_signed_turn_deg'],-90)
        self.assertEqual(request['previously_confirmed_subgoal_ids'],['sg_0'])
        self.assertNotIn('look_down',request['allowed_camera_actions'])
        self.assertNotIn('look_down',kwargs['result_schema']
                         ['properties']['next_action']['enum'])
        self.assertEqual(result['verification_status'],'CONFIRMED')

    def test_new_subgoal_progress_is_not_left_as_completed(self):
        mem=MemoryRecords(SimpleNamespace(turn_angle=15))
        mem.task.subgoals=goals()
        mem.observations=[dict(observation_id='obs_0028',subgoal=0)]
        mem.current=mem.observations[0]
        report=dict(status='CONFIRMED',subgoal_index=0,
            source='vlm_real_rgbd_history',
            observation_ids=['obs_0028'],source_observation_id='obs_0028',
            evidence='Robot reached lower landing',
            uncertainty='')
        output=mem.confirm_subgoal_observation(report,'Check sg0')
        self.assertEqual(output['next_subgoal'],1)
        self.assertEqual(mem.task.current_subgoal_index,1)
        self.assertEqual(mem.progress_stage,'NOT_STARTED')
        self.assertIsNone(mem.progress_feedback)


if __name__=='__main__':
    unittest.main()
