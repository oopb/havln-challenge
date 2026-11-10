"""v27: the actual NavigationAgent and VLM wire contexts never expose
world-frame absolute yaw. Every recorded viewpoint is relative to CURRENT RGB.
"""
import copy
import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from havln_agent.model_angles import (
    project_model_angles,project_observation_conversation)
from havln_agent.navigation_agent import MISSION_PROMPT
from havln_agent.visual_perception import VisualPerception,SCAN_GUIDANCE_PROMPT
from test_forward_preflight_v9 import camera,test_runtime,geometry


ABS_KEYS={
    'heading_deg','source_heading_deg','current_heading_deg',
    'last_heading_deg','absolute_heading_deg','heading_at_capture_deg',
    'yaw_deg','source_yaw_deg','current_yaw_deg','camera_yaw_deg',
    'observed_heading_bins_deg','heading_bins_deg',
}

def forbidden(value):
    if isinstance(value,dict):
        return ({k for k in value if k in ABS_KEYS} |
                set().union(*(forbidden(v) for v in value.values())))
    if isinstance(value,list):
        return set().union(*(forbidden(v) for v in value))
    return set()


class RelativeWireAngles(unittest.TestCase):
    def test_all_structured_angles_are_relative_to_current(self):
        raw=dict(
            current=dict(heading_deg=90,pitch_deg=-30),
            history_media=dict(manifest=[
                dict(observation_id='now',heading_deg=90,pitch_deg=-30),
                dict(observation_id='past',source_heading_deg=210,
                     source_pitch_deg=-60,source_stance_id='0:2')]),
            memory=dict(
                recent_views_at_current_stance=dict(
                    full_coverage_by_pitch=[
                        dict(pitch_deg=0,observed_heading_bins_deg=[
                            0,90,180,270])]),
                route_continuity=dict(
                    current_heading_deg=90,
                    selected_historical_forward=dict(
                        heading_deg=255,executed_forward_with_RGBD_change=6),
                    endpoint_front_obstacle=dict(
                        heading_deg=255,kind='FRONT_BLOCK_OBSERVED_NO_FORWARD_ATTEMPT'),
                    model_assessment_snapshots=[dict(
                        source_heading_deg=210,source_observation_id='obs_0019')]),
                observed_forward_action_segments=[dict(
                    heading_deg=255,forward_steps=6)],
                blocked_at_source_stances=[dict(
                    heading_deg=45,source_observation_id='obs_0065')] ),
            pose=dict(heading=math.radians(210),pitch_deg=-60),
            action=dict(turn_degrees=30,turn_direction='LEFT',
                        sweep_degrees=180),
            current_rgb=dict(heading_deg=90))
        before=copy.deepcopy(raw)
        projected=project_model_angles(raw,90)
        self.assertEqual(raw,before)
        self.assertFalse(forbidden(projected))
        self.assertEqual(projected['current'][
            'view_offset_deg_from_current'],0)
        self.assertEqual(projected['history_media']['manifest'][1][
            'source_view_offset_deg_from_current'],120)
        self.assertEqual(projected['memory']['route_continuity'][
            'selected_historical_forward']['view_offset_deg_from_current'],165)
        self.assertEqual(projected['memory']['blocked_at_source_stances'][0][
            'view_offset_deg_from_current'],-45)
        self.assertEqual(projected['pose']['view_offset_deg_from_current'],120)
        self.assertEqual(projected['memory']['recent_views_at_current_stance'][
            'full_coverage_by_pitch'][0]['observed_view_offsets_deg_from_current'],
            [-180,-90,0,90])
        self.assertEqual(projected['action']['turn_degrees'],30)
        self.assertEqual(projected['action']['sweep_degrees'],180)
        self.assertEqual(projected['current']['pitch_deg'],-30)

    def test_historical_narrative_explicit_heading_is_rebased(self):
        raw=dict(
            progress=dict(narrative='Robot was at heading 255, pitch -30; turn RIGHT 30°.',
                uncertainties=['At heading 210, passage might be visible.']),
            route_continuity=dict(model_assessment_snapshots=[
                dict(model_visual_description=
                    'CURRENT RGB (obs_0024, heading 210, pitch 0) has an arch.')]),
            instruction='Turn LEFT 30° after the stair landing.')
        a=project_model_angles(raw,90)
        self.assertIn('view offset LEFT 165 degrees from current camera',
                      a['progress']['narrative'])
        self.assertIn('turn RIGHT 30°',a['progress']['narrative'])
        self.assertIn('view offset LEFT 120 degrees from current camera',
                      a['progress']['uncertainties'][0])
        self.assertNotIn('heading 210',json.dumps(a))
        self.assertEqual(a['instruction'],raw['instruction'])
        self.assertEqual(raw['progress']['narrative'],
                         'Robot was at heading 255, pitch -30; turn RIGHT 30°.')

    def test_wraparound_and_unsafe_cross_stance_target_are_not_motion(self):
        raw=dict(target_type='STAIRS',absolute_heading_deg=330,
                 relative_bearing_deg_from_current=None,
                 source_stance_matches_current=False,
                 pitch_deg=15)
        p=project_model_angles(raw,30)
        self.assertNotIn('absolute_heading_deg',p)
        self.assertIsNone(p['relative_bearing_deg_from_current'])
        self.assertNotIn('historical_target_offset_deg_from_current',p)
        self.assertEqual(p['pitch_deg'],15)
        orientation=project_model_angles(dict(
            heading_deg=330,heading_at_capture_deg=330),30)
        self.assertEqual(orientation['view_offset_deg_from_current'],-60)
        self.assertEqual(orientation['capture_view_offset_deg_from_current'],-60)

    def test_raw_vlm_conversation_rebased_after_real_turn(self):
        request=dict(observation_id='obs_0025',heading_deg=90,
            mission_memory=dict(candidate_route_hypotheses=[dict(
                absolute_heading_deg=210,relative_bearing_deg_from_current=120,
                relative_side_from_current='LEFT',
                relative_turn_deg_from_current=120,
                source_stance_matches_current=True)]))
        session=[dict(role='user',content=[
            dict(type='text',text=json.dumps(request)),
            dict(type='image_url',image_url=dict(url='data:image/png;base64,X'))]),
            dict(role='assistant',content='{"next_action":"finish"}')]
        raw=copy.deepcopy(session)
        current=project_observation_conversation(session,120)
        self.assertEqual(session,raw)
        prior=json.loads(current[0]['content'][0]['text'])
        self.assertNotIn('heading_deg',prior)
        self.assertEqual(prior['view_offset_deg_from_current'],-30)
        route=prior['mission_memory']['candidate_route_hypotheses'][0]
        self.assertEqual(route['relative_bearing_deg_from_current'],90)
        self.assertEqual(route['relative_side_from_current'],'LEFT')
        self.assertEqual(route['relative_turn_deg_from_current'],90)
        self.assertNotIn('absolute_heading_deg',route)
        self.assertEqual(current[0]['content'][1],raw[0]['content'][1])
        self.assertEqual(current[1],raw[1])

    def test_assistant_history_narrative_rebased_without_changing_action(self):
        response=dict(next_action='turn_left',turn_deg=30,
            observation_summary='At heading 210 there is a stair edge.')
        session=[dict(role='assistant',content=json.dumps(response))]
        original=copy.deepcopy(session)
        new=project_observation_conversation(session,90)
        body=json.loads(new[0]['content'])
        self.assertEqual(body['next_action'],'turn_left')
        self.assertEqual(body['turn_deg'],30)
        self.assertIn('view offset LEFT 120 degrees from current camera',
                      body['observation_summary'])
        self.assertEqual(session,original)

    def test_visual_agent_actual_complete_path_rebases_all_turns(self):
        class Controller:
            heading=math.radians(120)
            def pose(self):
                return dict(heading=self.heading,pitch_deg=-30)
        client=SimpleNamespace(complete=Mock(return_value={'ok':True}))
        runtime=SimpleNamespace(client=client,
                                motion_controller=Controller())
        visual=VisualPerception(runtime)
        old=dict(source_observation_id='obs_0001',heading_deg=90)
        session=[dict(role='user',content=[
            dict(type='text',text=json.dumps(old))])]
        original=copy.deepcopy(session)
        visual.complete('visual_perception','test','system',
            dict(heading_deg=120,history=[dict(source_heading_deg=210)]),
            dict(type='object'),conversation_messages=session)
        args=client.complete.call_args
        self.assertFalse(forbidden(args.args[3]))
        self.assertEqual(args.args[3]['view_offset_deg_from_current'],0)
        self.assertEqual(args.args[3]['history'][0][
            'source_view_offset_deg_from_current'],90)
        earlier=json.loads(args.kwargs['conversation_messages'][0]['content'][0]['text'])
        self.assertEqual(earlier['view_offset_deg_from_current'],-30)
        self.assertEqual(session,original)
        runtime.motion_controller.heading=math.radians(180)
        visual.complete('visual_perception','test','system',
            dict(heading_deg=180),dict(type='object'),
            conversation_messages=session)
        args=client.complete.call_args
        earlier=json.loads(args.kwargs['conversation_messages'][0]['content'][0]['text'])
        self.assertEqual(earlier['view_offset_deg_from_current'],-90)
        self.assertEqual(session,original)

    def test_navigation_agent_actual_model_input_no_absolute(self):
        r,_=test_runtime(changed_forwards=True)
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        # A turn provides a nonzero real reference orientation.
        r._step('TURN_LEFT')
        r.navigation_mode='OBSERVE'
        r.navigation_agent.decide()
        actual=r.client.complete.call_args.args[3]
        self.assertFalse(forbidden(actual))
        self.assertEqual(actual['current_rgb']['view_offset_deg_from_current'],0)
        self.assertEqual(actual['history_media']['manifest'][0][
            'view_offset_deg_from_current'],0)
        self.assertIn('view_angle_convention',actual)
        self.assertTrue('positive=LEFT' in actual['view_angle_convention'])
        self.assertIn('signed relative offsets',MISSION_PROMPT)
        self.assertNotIn('absolute camera yaw',SCAN_GUIDANCE_PROMPT)
        self.assertLess(len(MISSION_PROMPT),5000)

if __name__=='__main__':
    unittest.main()
