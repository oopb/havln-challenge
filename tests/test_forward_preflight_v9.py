"""v9 pre-decision Depth, physical camera recenter, ROI and stall regressions.

Run: python -m unittest discover -s tests -p test_forward_preflight_v9.py -v
No Habitat install, simulation assets, network or API keys required.
"""
import copy
import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

from havln_agent.runtime import AgentRuntime
from havln_agent.perception import Camera, Observation
from havln_agent.mission_history import mission_evidence
from havln_agent.forward_preflight import sense_forward


def camera():
    return Camera(width=64,height=64,hfov=90,min_depth=.1,max_depth=10,
        normalized=False,camera_height=1.25,robot_height=1.7,
        robot_radius=.15,forward_step=.25,turn_angle=15,tilt_angle=15)


def frame(depth=1.4):
    return Observation(np.zeros((64,64,3),dtype=np.uint8),
        np.full((64,64),depth,dtype=np.float32),
        'Go down the stairs and stop near the piano',time.monotonic())


def probe(state='EXECUTABLE'):
    return dict(state=state,reason=(
        'occupied_next_step' if state=='BLOCKED' else
        'unknown_depth_or_fov' if state=='UNCONFIRMED' else None),
        depth=dict(path_state='OCCUPIED' if state=='BLOCKED' else
            'UNKNOWN' if state=='UNCONFIRMED' else 'FREE',
            clearance=0.03 if state=='BLOCKED' else 1.0,
            required_forward_step=.25,
            rejection_reasons=['occupied_next_step']
                if state=='BLOCKED' else [],
            unknown_sweep_pixels=0))


def center(confirmed=True):
    return dict(
        confirmed=confirmed,
        central_support_pixels=257 if confirmed else 0,
        required_support_pixels=4,
        nearest_along_m=.316,
        nearest_depth_m=.32,
        nearest_height_m=.6,
        source_pixel_xy=[31,39],
        focus_point_xy_norm=[.4922,.6172],
        focus_region_xyxy_norm=[.42,.34,.61,.78],
        reason='measured_high_centerline_occupancy',
        identity='unverified_depth_obstacle')


def test_runtime(*,changed_forwards=False):
    state={'depth':1.4}
    actions=[]
    def step(action):
        actions.append(action)
        if action=='MOVE_FORWARD' and changed_forwards:
            state['depth']+=.02
        return frame(state['depth']),False
    def refresh():
        return frame(state['depth'])
    r=AgentRuntime(
        client=SimpleNamespace(trace_context={},
            complete=Mock(return_value=dict(tool='turn_and_observe',
                args=dict(turn_direction='LEFT',turn_degrees=15,
                          look='none')))),
        camera=camera(),
        config=dict(max_steps=88,tool_max_actions=12,
            safety_margin=0,max_depth_age_seconds=30,
            max_scene_age_seconds=30),
        step=step,refresh=refresh)
    r.memory.reset('v9-unit','Go down the stairs and stop near the piano')
    r.accept_observation(refresh())
    r._authorized_action=True
    r.phase='NAVIGATING'
    r.tool_id='test-tool'
    r.context_version='test'
    r.current_decision_id='nav-v9'
    return r,actions


def geometry(state='EXECUTABLE',confirmed=True):
    return SimpleNamespace(
        movement_probe=Mock(return_value=probe(state)),
        central_forward_obstruction=Mock(return_value=center(confirmed)))


class ForwardPreflightV9(unittest.TestCase):
    def test_statuses_are_shared_and_unknown_not_a_veto(self):
        r,_=test_runtime()
        for raw,confirmed,expected in [
            ('EXECUTABLE',True,'CLEAR'),
            ('UNCONFIRMED',True,'UNKNOWN'),
            ('BLOCKED',False,'UNKNOWN'),
            ('BLOCKED',True,'BLOCKED')]:
            with self.subTest(raw=raw,confirmed=confirmed):
                g=geometry(raw,confirmed)
                r.geometry=Mock(return_value=g)
                result=r.forward_preflight(stage='navigation_decision')
                self.assertEqual(result['status'],expected)
                self.assertEqual(result['source_observation_id'],
                                 r.memory.current['observation_id'])
                if expected=='BLOCKED':
                    self.assertEqual(result['blocker'][
                        'focus_region_xyxy_norm'],[.42,.34,.61,.78])
                else:
                    self.assertIsNone(result['blocker'])
                if raw!='BLOCKED':
                    g.central_forward_obstruction.assert_not_called()

    def test_model_receives_depth_roi_before_inference(self):
        r,_=test_runtime()
        r.geometry=Mock(return_value=geometry('BLOCKED'))
        r.navigation_agent.decide()
        context=r.client.complete.call_args.args[3]
        result=context['front_depth_preflight']
        self.assertEqual(result['status'],'BLOCKED')
        self.assertEqual(result['blocker']['central_support_pixels'],257)
        self.assertEqual(result['stage'],'navigation_decision')
        self.assertEqual(result['source_observation_id'],
                         context['current_rgb']['observation_id'])
        self.assertEqual(len(r.memory.evidence.obstacle_blocks),0)

    def test_camera_recenter_is_real_and_precedes_forward_after_turn(self):
        r,actions=test_runtime()
        r.motion_controller.pitch_deg=-30
        r.refresh()
        r.geometry=Mock(return_value=geometry('BLOCKED'))
        result=r._navigate_relative('RIGHT',30,.5)
        self.assertEqual(result['status'],'BLOCKED')
        self.assertEqual(actions,
            ['TURN_RIGHT','TURN_RIGHT','LOOK_UP','LOOK_UP'])
        self.assertEqual(r.pitch_deg,0)
        self.assertEqual(result['obstacle_observation']['depth'][
            'focus_region_xyxy_norm'],[.42,.34,.61,.78])
        self.assertEqual(result['forward_depth_preflight']['pitch_deg'],0)
        self.assertEqual(result['forward_depth_preflight']['heading_deg'],330)
        self.assertEqual(len(r.memory.evidence.obstacle_blocks),1)
        self.assertEqual(r.runtime_event_counts[
            'atomic_forward_confirmed_depth_block'],1)

    def test_no_unnecessary_look_steps_during_continuous_forward(self):
        r,actions=test_runtime(changed_forwards=True)
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        motion=dict(status='ESTIMATED',vertical_motion='LEVEL',
            translation_m=dict(x_right=0.,y_up=0.,z_forward=.25),
            uncertainty_m=.01)
        with patch('havln_agent.tools.estimate_translation',
                   return_value=motion):
            result=r._navigate_relative('NONE',0,.5)
        self.assertEqual(result['status'],'ACTIONS_COMPLETED')
        self.assertEqual(actions,['MOVE_FORWARD','MOVE_FORWARD'])

    def test_uncertain_side_only_probe_does_not_become_veto(self):
        r,actions=test_runtime(changed_forwards=True)
        g=geometry('BLOCKED',False)
        r.geometry=Mock(return_value=g)
        motion=dict(status='ESTIMATED',vertical_motion='LEVEL',
            translation_m=dict(x_right=0.,y_up=0.,z_forward=.25),
            uncertainty_m=.01)
        with patch('havln_agent.tools.estimate_translation',
                   return_value=motion):
            result=r._navigate_relative('NONE',0,.5)
        self.assertEqual(result['status'],'ACTIONS_COMPLETED')
        self.assertEqual(actions,['MOVE_FORWARD','MOVE_FORWARD'])
        self.assertEqual(len(r.memory.evidence.obstacle_blocks),0)
        self.assertGreaterEqual(g.central_forward_obstruction.call_count,2)

    def test_camera_alignment_cannot_spend_reserved_stop_action(self):
        r,actions=test_runtime()
        r.config['max_steps']=2  # Only one non-STOP action remains.
        r.motion_controller.pitch_deg=-30
        r.refresh()
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        result=r._navigate_relative('NONE',0,.25)
        self.assertEqual(result['status'],'BUDGET_LIMIT')
        self.assertEqual(actions,['LOOK_UP'])
        self.assertEqual(r.pitch_deg,-15)
        self.assertFalse(any(a=='MOVE_FORWARD' for a in actions))

    def test_stall_is_not_silently_treated_as_collision(self):
        r,actions=test_runtime()
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        motion=dict(status='UNKNOWN',
            reason='no_static_matches',vertical_motion='UNKNOWN')
        with patch('havln_agent.tools.estimate_translation',
                   return_value=motion):
            out=r._navigate_relative('NONE',0,.5)
        self.assertEqual(out['status'],'REQUIRES_ATTENTION')
        self.assertEqual(actions,['MOVE_FORWARD'])
        self.assertEqual(len(r.memory.evidence.motion_stalls),1)
        record=r.memory.evidence.motion_stalls[0]
        self.assertFalse(record['collision_proven'])
        self.assertTrue(record['forward_atomic_action_executed'])
        shared=mission_evidence(r.memory,r.motion_controller.pose())
        self.assertEqual(shared['stalled_forward_at_source_stances'][0][
            'kind'],'STALLED_AFTER_FORWARD')
        self.assertEqual(r.runtime_event_counts['forward_motion_stalled'],1)

    def test_two_observation_ids_same_RGBD_heading_are_one_failed_view(self):
        r,_=test_runtime()
        src=r.source()
        pose=r.motion_controller.pose()
        blocker=center()
        first=r.memory.evidence.blocked_forward(
            src,pose,reason='measured_centerline_obstacle',center=blocker)
        second_source=dict(src,observation_id='obs_9999')
        second=r.memory.evidence.blocked_forward(
            second_source,pose,reason='measured_centerline_obstacle',
            center=blocker)
        self.assertEqual((first['repeat_count_at_recording'],
                          second['repeat_count_at_recording']),(1,2))
        r.memory.evidence.stalled_forward(src,pose,action_step=4,
                                         after_observation_id='obs_0000')
        r.memory.evidence.stalled_forward(
            second_source,pose,action_step=8,
            after_observation_id='obs_0000')
        index=mission_evidence(r.memory,pose)
        self.assertEqual(len(index['blocked_at_source_stances']),1)
        self.assertEqual(index['blocked_at_source_stances'][0][
            'repeated_same_frame_heading_attempts'],2)
        self.assertEqual(len(index['stalled_forward_at_source_stances']),1)
        self.assertEqual(index['stalled_forward_at_source_stances'][0][
            'repeated_same_frame_heading_attempts'],2)
        self.assertEqual(index['recent_forward_failures'][0][
            'kind'],'BLOCKED_BEFORE_FORWARD')
        self.assertEqual(index['recent_forward_failures'][1][
            'kind'],'STALLED_AFTER_FORWARD')

    def test_historical_tool_receipt_roi_reads_actual_flat_depth_field(self):
        r,_=test_runtime()
        src=r.source()
        r.memory.tool_ledger.append(dict(
            call_id='tool_0001',
            call=dict(tool='navigate_relative',args=dict(
                turn_direction='NONE',turn_degrees=0,distance_m=.25)),
            result=dict(status='BLOCKED',reason='measured_centerline_obstacle',
                obstacle_observation=dict(source_observation_id=src[
                    'observation_id'],depth=center())),
            request_context=dict(actor='navigation_agent')))
        index=mission_evidence(r.memory,r.motion_controller.pose())
        roi=index['recent_tool_receipts'][0]['blocker']
        self.assertEqual(roi['central_support_pixels'],257)
        self.assertEqual(roi['focus_region_xyxy_norm'],[.42,.34,.61,.78])


if __name__=='__main__':
    unittest.main()
