"""RGB-D observation agent chooses current-bearing approaches and retries after a block.

All probes use public current Depth; all motion is still NavigationAgent-owned.
"""
import base64
import io
import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np
from PIL import Image
from jsonschema import Draft7Validator

from havln_agent.active_observation import ObservationController
from havln_agent.contexts import compact_navigation_context,tool_summary
from havln_agent.perception import Camera,Observation,depth_image_url
from havln_agent.visual_perception import VisualPerception,SCAN_GUIDANCE_SCHEMA
from test_active_observation import FakeRuntime
from test_vlm_led_observation import VLMGuide,guidance


def candidate(visible=True):
    return dict(guidance('finish',kind='APPROACH',visible=visible),
                candidate_approach=True,
                target_type='descending approach',
                evidence='Stair treads appear ahead',
                uncertainty='next step unconfirmed')


def obstacle(confirmed):
    return dict(confirmed=confirmed,
        reason='measured_high_centerline_occupancy' if confirmed
               else 'no_coherent_centerline_obstacle',
        central_support_pixels=45 if confirmed else 0,
        nearest_along_m=.15 if confirmed else None,
        source_pixel_xy=[14,18] if confirmed else None)


def make_runtime(replies,checks):
    r=FakeRuntime([])
    r.visual_perception=VLMGuide(replies)
    measured=Mock(side_effect=[obstacle(check) for check in checks])
    geometry=SimpleNamespace(central_forward_obstruction=measured,
        assess_step=Mock(side_effect=AssertionError('full terrain not allowed')))
    r.geometry=Mock(return_value=geometry)
    return r,geometry


class RGBDApproachAgentTests(unittest.TestCase):
    def test_clear_hypothesis_returns_to_navigation_without_any_forward(self):
        r,g=make_runtime([candidate()],[False])
        result=ObservationController(r).scan_vlm(
            'Find a descending approach',['stairs'],sweep_degrees=90)
        self.assertEqual(result['outcome'],'APPROACH_CANDIDATE_FOUND')
        self.assertEqual(result['status'],'OK')
        self.assertEqual(result['best_approach']['heading_deg'],0)
        self.assertEqual(result['best_approach']['probe']['status'],'UNCONFIRMED')
        self.assertTrue(result['best_approach'][
            'direction_is_a_hypothesis_not_motion_authorization'])
        self.assertEqual(len(result['approach_candidates']),1)
        self.assertEqual(result['blocked_approaches'],[])
        self.assertEqual(r.step_calls,[])
        g.central_forward_obstruction.assert_called_once_with()
        g.assess_step.assert_not_called()

    def test_blocked_bearing_is_returned_to_same_vlm_then_alternative_is_found(self):
        r,g=make_runtime([
            candidate(),
            dict(guidance('turn_left',45,kind='CUE',visible=True),
                 evidence='A railing blocks the front; look toward an open side'),
            candidate()],[True,False])
        result=ObservationController(r).scan_vlm(
            'Find an approach toward the stairs',['stairs'],sweep_degrees=120)
        self.assertEqual(result['outcome'],'APPROACH_CANDIDATE_FOUND')
        self.assertEqual(result['best_approach']['heading_deg'],45)
        self.assertEqual(result['best_approach']['relative_turn_deg_from_current'],0)
        self.assertEqual(len(result['blocked_approaches']),1)
        self.assertEqual(result['blocked_approaches'][0]['heading_deg'],0)
        self.assertEqual(result['blocked_approaches'][0]['probe']['status'],'BLOCKED')
        self.assertEqual(len(r.visual_perception.calls),3)
        second=r.visual_perception.calls[1]
        self.assertEqual(second['view_correction']['reason'],
            'candidate_forward_measured_center_obstacle')
        self.assertEqual(second['view_correction']['rejected_heading_deg'],0)
        self.assertEqual(second['history'][-1]['approach_probe_status'],'BLOCKED')
        self.assertEqual(r.step_calls,['TURN_LEFT']*3)
        self.assertNotIn('MOVE_FORWARD',r.step_calls)
        self.assertEqual(g.central_forward_obstruction.call_count,2)
        g.assess_step.assert_not_called()
        self.assertEqual([a['probe']['status'] for a in
                          result['approach_candidates']],['UNCONFIRMED'])

    def test_repeated_same_bearing_does_not_enter_endless_observation(self):
        r,g=make_runtime([candidate(),candidate()],[True,True])
        result=ObservationController(r).scan_vlm(
            'Find an entrance',['stairs'],sweep_degrees=60)
        self.assertEqual(result['outcome'],'NO_RELIABLE_CANDIDATE')
        self.assertEqual(result['status'],'OK')
        self.assertEqual(len(r.visual_perception.calls),2)
        self.assertEqual(len(result['blocked_approaches']),1)
        self.assertIsNone(result['best_approach'])
        self.assertEqual(r.step_calls,[])
        self.assertEqual(g.central_forward_obstruction.call_count,2)

    def test_vlm_can_end_with_uncertainty_after_block_without_turn_loop(self):
        r,g=make_runtime([
            candidate(),
            dict(guidance('finish',kind='CUE',visible=True),
                 evidence='Railing remains; cannot find another visual approach')
        ],[True])
        result=ObservationController(r).scan_vlm(
            'Find entrance',['stairs'],sweep_degrees=60)
        self.assertEqual(result['outcome'],'NO_RELIABLE_CANDIDATE')
        self.assertEqual(result['status'],'OK')
        self.assertEqual(len(r.visual_perception.calls),2)
        self.assertEqual(r.step_calls,[])

    def test_cached_same_observation_reuses_candidate_and_blocked_evidence(self):
        r,g=make_runtime([candidate()],[False])
        first=ObservationController(r).scan_vlm(
            'Find entry',['stairs'],sweep_degrees=90)
        again=ObservationController(r).scan_vlm(
            'Find entry',['stairs'],sweep_degrees=90)
        self.assertEqual(first['outcome'],'APPROACH_CANDIDATE_FOUND')
        self.assertEqual(again['status'],'NO_NEW_INFORMATION')
        self.assertEqual(again['best_approach']['heading_deg'],0)
        self.assertEqual(len(r.visual_perception.calls),1)

    def test_approach_and_rejection_survive_last_tool_and_latest_observation(self):
        r,g=make_runtime([
            candidate(),
            dict(guidance('turn_left',30),candidate_approach=False),
            candidate()],[True,False])
        result=ObservationController(r).scan_vlm(
            'Find entry',['stairs'],sweep_degrees=90)
        receipt=tool_summary(dict(result,tool='observe_for_goal'))
        state=dict(last_tool=receipt,
            latest_observation_task=dict(
                mode='vlm_led',objective='Find entry',
                outcome=result['outcome'],visual_reports=result['visual_reports'],
                best_approach=result['best_approach'],
                approach_candidates=result['approach_candidates'],
                blocked_approaches=result['blocked_approaches']))
        compact=compact_navigation_context(state)
        self.assertEqual(compact['last_tool']['observation_task']
            ['best_approach']['heading_deg'],30)
        self.assertEqual(compact['latest_observation_task']
            ['blocked_approaches'][0]['heading_deg'],0)
        self.assertEqual(compact['latest_observation_task']
            ['approach_candidates'][0]['observation_id'],'obs_0002')

    def test_public_depth_image_encodes_metric_distance_and_invalid_pixels(self):
        cam=Camera(width=2,height=2,hfov=90,min_depth=0.1,max_depth=10.,
            normalized=True,camera_height=1.25,robot_height=1.5,
            robot_radius=.1,forward_step=.25,turn_angle=15)
        obs=Observation(np.zeros((2,2,3),np.uint8),
            np.array([[0,.10],[.50,np.nan]],np.float32),
            'find floor',time.monotonic())
        uri=depth_image_url(obs,cam,max_edge=640,display_max_m=5.0)
        image=Image.open(io.BytesIO(base64.b64decode(uri.split(',',1)[1])))
        rgb=np.asarray(image)
        self.assertEqual(tuple(rgb[0,0]),(255,0,255))
        self.assertEqual(tuple(rgb[1,1]),(255,0,255))
        self.assertTrue(0<int(rgb[0,1,0])<255)
        self.assertEqual(tuple(rgb[1,0]),(255,255,255))

    def test_vlm_request_has_rgb_then_calibrated_depth_and_compact_history(self):
        cam=Camera(width=4,height=4,hfov=90,min_depth=0,max_depth=10.,
            normalized=False,camera_height=1.25,robot_height=1.5,
            robot_radius=.1,forward_step=.25,turn_angle=15)
        task=SimpleNamespace(instruction='Descend stairs',
            current_subgoal_index=0,subgoals=[dict(action='Descend',target='stairs')])
        vp=VisualPerception(SimpleNamespace(camera=cam,
            memory=SimpleNamespace(task=task),client=None))
        frame=Observation(np.zeros((4,4,3),np.uint8),
            np.full((4,4),.6,np.float32),'Descend stairs',time.monotonic())
        src=dict(observation_id='obs_0031',
            pose=dict(heading=math.radians(90),pitch_deg=-30))
        with patch.object(vp,'complete',return_value=candidate()) as call:
            result=vp.guide_scan_view(frame,src,'find entry',['stairs'],
                [dict(observation_id='obs_0030',heading_deg=75,
                      evidence='Railing blocks earlier view',
                      approach_probe_status='BLOCKED')],90,
                view_correction=dict(reason='candidate_forward_measured_center_obstacle'))
        args,kwargs=call.call_args
        self.assertEqual(args[3]['sensor_images'],
                         ['current_rgb','current_depth_gray_m'])
        self.assertEqual(len(kwargs['image_url']),2)
        self.assertEqual(args[3]['depth_visualization']['display_max_m'],5.)
        self.assertEqual(args[3]['previously_inspected_views'][0]
                         ['approach_probe_status'],'BLOCKED')
        self.assertEqual(args[3]['view_correction']['reason'],
            'candidate_forward_measured_center_obstacle')
        self.assertTrue(Draft7Validator(SCAN_GUIDANCE_SCHEMA).is_valid(result))
        self.assertFalse(kwargs['cache'])

    def test_geometry_unavailable_is_uncertain_not_a_hard_block(self):
        r,g=make_runtime([candidate()],[False])
        r.geometry=Mock(side_effect=RuntimeError('unavailable current geometry'))
        result=ObservationController(r).scan_vlm(
            'Find entry',['stairs'],sweep_degrees=90)
        self.assertEqual(result['outcome'],'APPROACH_CANDIDATE_FOUND')
        self.assertEqual(result['best_approach']['probe']['status'],'UNCONFIRMED')
        self.assertEqual(r.step_calls,[])


if __name__=='__main__':
    unittest.main()
