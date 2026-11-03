"""Authentic panoramic RGB memory is paired with exact current-relative yaw."""
import base64
import io
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image

from havln_agent.api_client import APIError
from havln_agent.perception import Observation
from havln_agent.visual_perception import VisualPerception
from havln_agent.memory_service import MemoryService
from havln_agent.panorama_memory import (
    PANORAMA_NAVIGATION_PROMPT, select_saved_viewpoints, decide_with_panorama)


def picture(color, width=800, height=400):
    rgb=np.zeros((height,width,3),dtype=np.uint8)
    rgb[:]=color
    return Observation(rgb,np.ones((height,width),dtype=np.float32),
                       'Find the stairs',1.,0.)


def snapshot(ident, heading, relative, pitch=0):
    return dict(observation_id=ident,absolute_heading_deg=heading,
        relative_turn_deg_from_current=relative,pitch_deg=pitch,
        historical_front_state='UNMEASURED',historical_front_reason=None,
        historical_front_clearance_m=None)


class CapturedVisionClient:
    def __init__(self):
        self.requests=[]
    def complete(self,*args,**kwargs):
        self.requests.append((args,kwargs))
        return dict(selected_observation_id='obs0',
            visible_evidence='An opening is visible on the left',
            visual_cues=['door opening'],uncertainty='medium',
            alternative_observation_ids=['obs1'])


class PanoramaMemoryTests(unittest.TestCase):
    def test_vlm_receives_real_images_in_exact_angle_order_and_resized(self):
        client=CapturedVisionClient()
        perception=VisualPerception(SimpleNamespace(client=client))
        frames=[
            (picture((255,0,0)),snapshot('obs0',350,-20)),
            (picture((0,255,0)),snapshot('obs1',10,0)),
            (picture((0,0,255)),snapshot('obs2',70,60,-30))]
        result=perception.panorama_direction_evidence(frames,dict(
            task=dict(instruction='Find stairs'),current_heading_deg=10,
            navigation_memory=dict(failures=['blocked']),
            recent_route_failures=dict(reason='occupied_next_step')))
        self.assertEqual(result['selected_observation_id'],'obs0')
        self.assertIn('target_status',args_schema:=client.requests[0][0][4]['properties'])
        self.assertIn('approach_candidates',args_schema)
        args,kwargs=client.requests[0]
        self.assertEqual(args[0:2],('visual_perception','panorama_direction_evidence'))
        self.assertEqual([x['observation_id'] for x in args[3]['views']],
                         ['obs0','obs1','obs2'])
        self.assertEqual([x['relative_turn_deg_from_current']
                          for x in args[3]['views']],[-20,0,60])
        self.assertEqual([x['image_index'] for x in args[3]['views']],[0,1,2])
        self.assertIn('IN ORDER',args[2])
        self.assertEqual(len(kwargs['image_url']),3)
        for index,expected in enumerate([(255,0,0),(0,255,0),(0,0,255)]):
            data=base64.b64decode(kwargs['image_url'][index].split(',',1)[1])
            rgb=np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))
            self.assertEqual(rgb.shape,(320,640,3))
            self.assertEqual(tuple(rgb[0,0]),expected)
        with self.assertRaises(APIError):
            kwargs['validate'](dict(result,selected_observation_id='invented'))
        with self.assertRaises(APIError):
            kwargs['validate'](dict(result,
                alternative_observation_ids=['never_observed']))
        with self.assertRaises(APIError):
            kwargs['validate'](dict(result,approach_candidates=[dict(
                observation_id='invented',kind='ENTRY',confidence='high',evidence='fake')]))

    def test_saved_yaw_diversity_and_no_synthetic_frames(self):
        panorama=dict(observed_viewpoints=[
            snapshot('obs%d' % i,(i*15)%360,(i*15)%360)
            for i in range(24)])
        selected=select_saved_viewpoints(panorama)
        self.assertEqual(len(selected),12)
        self.assertEqual(len({x['observation_id'] for x in selected}),12)
        self.assertEqual([x['absolute_heading_deg'] for x in selected],
                         list(range(0,360,30)))
        self.assertEqual(select_saved_viewpoints(dict(observed_viewpoints=[])),[])

    def test_navigation_receives_vlm_evidence_and_recomputed_signed_yaw(self):
        chosen=[
            snapshot('obs0',10,999),snapshot('obs1',100,-999),
            snapshot('obs2',190,-999),snapshot('obs3',280,-999),
            snapshot('obs4',330,999)]
        visuals=SimpleNamespace(panorama_direction_evidence=Mock(
            return_value=dict(selected_observation_id='obs0',
                visible_evidence='Stairs possible',visual_cues=['treads'],
                uncertainty='medium',alternative_observation_ids=[])))
        client=SimpleNamespace(complete=Mock(return_value=dict(
            tool='navigate_relative',args=dict(turn_deg=20,distance_m=.25))))
        runtime=SimpleNamespace(client=client,visual_perception=visuals,
            _vision_frame=Mock(side_effect=lambda obs_id:
                (picture((1,2,3)),dict(observation_id=obs_id))),
            memory=SimpleNamespace(task=SimpleNamespace(instruction='Descend stairs')),
            event=Mock())
        agent=SimpleNamespace(runtime=runtime)
        context=dict(panoramic_observation=dict(observed_viewpoints=chosen),
            pose=dict(heading_deg=350),
            goal=dict(action='descend',target='stairs'),
            task_selection=dict(current_subgoal_index=0),
            navigation_memory=dict(recent_failed_headings=[]),
            recent_navigation_block=dict(reason='BLOCKED'))
        tool=dict(description='model movement',permission='NAVIGATION',
            parameters=dict(type='object',properties=dict(
                turn_deg=dict(type='number'),distance_m=dict(type='number')),
                required=['turn_deg','distance_m'],additionalProperties=False))
        result,visual,views=decide_with_panorama(agent,context,('ep',1),
                                                Mock(),dict(navigate_relative=tool))
        self.assertEqual(result['args']['turn_deg'],20)
        by_id={x['observation_id']:x for x in views}
        self.assertEqual(by_id['obs0']['relative_turn_deg_from_current'],20)
        self.assertEqual(by_id['obs4']['relative_turn_deg_from_current'],-20)
        self.assertEqual(by_id['obs0']['absolute_heading_deg'],10)
        nav_args,nav_kwargs=client.complete.call_args
        self.assertEqual(nav_args[0:2],('navigation_agent','panorama_navigation_commit'))
        self.assertEqual(nav_args[2],PANORAMA_NAVIGATION_PROMPT)
        self.assertNotIn('MODE: OBSERVE',nav_args[2])
        self.assertEqual(nav_args[3]['panorama_visual_evidence']['visible_evidence'],
                         'Stairs possible')
        self.assertEqual(nav_args[3]['instruction'],'Descend stairs')
        self.assertEqual(nav_args[3]['recent_navigation_block']['reason'],'BLOCKED')
        self.assertEqual(visuals.panorama_direction_evidence.call_count,1)
        self.assertEqual(runtime.event.call_args.args[0],
                         'panorama_visual_memory_reviewed')
        # A NavigationAgent formatting retry repeats the proposal request but
        # must not upload/inspect the same historical images a second time.
        decide_with_panorama(agent,context,('ep',1),Mock(),
                             dict(navigate_relative=tool))
        self.assertEqual(visuals.panorama_direction_evidence.call_count,1)
        self.assertEqual(client.complete.call_count,2)

    def test_panorama_target_memory_remains_hypothesis_not_collision_permission(self):
        mem=MemoryService.__new__(MemoryService)
        mem.visual_targets=[]
        mem.keyframes={'obs0':dict(place_id='P0'),
                       'obs1':dict(place_id='P0')}
        mem.task=SimpleNamespace(current_subgoal_index=0)
        mem.runtime=SimpleNamespace(motion_controller=SimpleNamespace(
            pose=lambda:dict(motion_epoch=2)))
        mem.note_semantics=Mock()
        mem.event=Mock()
        views=[snapshot('obs0',225,-15),snapshot('obs1',240,0)]
        visual=dict(selected_observation_id='obs0',
            visible_evidence='railing and lower floor, entry uncertain',
            uncertainty='not enough view of treads',target_status='CUE_ONLY',
            target_region=None,missing_evidence='look down near platform',
            approach_candidates=[dict(observation_id='obs1',kind='APPROACH',
                                      confidence='medium',evidence='open floor')])
        record=mem.note_panorama_target(visual,views,goal=dict(target='stairs'))
        self.assertEqual(record['status'],'CUE_ONLY')
        self.assertEqual(record['source_observation_ids'],['obs0','obs1'])
        self.assertEqual(record['approach_candidates'][0]['absolute_heading_deg'],240)
        self.assertEqual(record['evidence_state'],
                         'historical_visual_hypothesis_not_clearance')
        mem.note_semantics.assert_not_called()
        self.assertEqual(len(mem.visual_targets),1)

    def test_missing_historical_image_never_creates_a_fabricated_view(self):
        runtime=SimpleNamespace(
            _vision_frame=Mock(side_effect=ValueError('file not present')),
            memory=SimpleNamespace(task=SimpleNamespace(instruction='Find exit')),
            client=SimpleNamespace(complete=Mock()),event=Mock())
        agent=SimpleNamespace(runtime=runtime)
        context=dict(panoramic_observation=dict(
            observed_viewpoints=[snapshot('obs0',0,0)]),
            pose=dict(heading_deg=0))
        with self.assertRaises(APIError):
            decide_with_panorama(agent,context,('ep',1),
                                 Mock(),dict(navigate_relative={}))
        runtime.client.complete.assert_not_called()


if __name__=='__main__':
    unittest.main()
