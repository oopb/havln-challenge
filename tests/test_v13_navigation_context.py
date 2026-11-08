"""v13 navigation history and grounded-direction regression.

No API calls, privileged simulator poses or external datasets required.
Run: python -m unittest discover -s tests -p 'test_v13_navigation_context.py' -v
"""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jsonschema import Draft7Validator

from havln_agent.api_client import APIError
from havln_agent.contexts import compact_navigation_context, navigation_context
from havln_agent.contracts import navigation_schema, registry
from havln_agent.memory_records import MemoryRecords
from havln_agent.mission_history import mission_evidence
from havln_agent.navigation_agent import (
    NavigationAgent, MISSION_PROMPT, validate_direction_intent)
from havln_agent.navigation_history import (
    navigation_episode_history, navigation_visual_anchors)
from havln_agent.perception import Camera
from test_forward_preflight_v9 import test_runtime, geometry


def camera():
    return Camera(
        width=64,height=64,hfov=90,min_depth=.1,max_depth=10,
        normalized=False,camera_height=1.25,robot_height=1.7,
        robot_radius=.15,forward_step=.25,turn_angle=15,
        tilt_angle=15)


def pose(deg=0,generation=0):
    return dict(x=None,y=None,z=None,heading=math.radians(deg),
                pitch_deg=0,motion_epoch=0,
                viewpoint_generation=generation,
                stance_id='0:'+str(generation),
                position_coordinates_valid=False)


def observation(identifier,step,heading=0,generation=0):
    return dict(
        observation_id=identifier,step=step,
        pose=pose(heading,generation),
        rgb_hash='rgb_'+identifier,depth_hash='depth_'+identifier,
        revisit_matches=[],revisited_static_view=False)


def motion_decision(direction,mode,source,box=None):
    return dict(
        tool='turn_and_observe',
        args=dict(turn_direction=direction,turn_degrees=30,
                  look='none',purpose='Inspect an opening beside the person.'),
        direction_intent=dict(
            mode=mode,target_observation_id=source,
            target_box_xyxy_norm=box,
            reason='Compare the declared target direction with the camera.',
            expected_observation='Learn if a passage is accessible.'))


class CompleteHistoryTests(unittest.TestCase):
    def test_every_decision_action_checkpoint_and_frame_is_in_causal_index(self):
        m=MemoryRecords(camera())
        m.reset('ep','Find the hall past the arch')
        m.observations=[
            observation('obs_0000',0,0,0),
            observation('obs_0001',1,15,0),
            observation('obs_0002',2,15,1)]
        m.current=m.observations[-1]
        m.actions=[
            dict(step=0,action='TURN_LEFT',pose=pose(15,0),
                 result=dict(observation_before='obs_0000',
                             observation_after='obs_0001',
                             visual_change=True)),
            dict(step=1,action='MOVE_FORWARD',pose=pose(15,1),
                 result=dict(observation_before='obs_0001',
                             observation_after='obs_0002',
                             visual_change=True,
                             displacement_verified=False))]
        m.observation_searches=[
            dict(observation_task_id='observation_000',
                 start_step=0,objective='Look for the arch',
                 start_observation_id='obs_0000',
                 end_observation_id='obs_0001',
                 outcome='OBSERVATION_COMPLETED',
                 observation_summary='Stone arch visible.',
                 actions=['TURN_LEFT'],waypoints=[],
                 checkpoints=[dict(observation_id='obs_0001',
                     heading_deg=15,pitch_deg=0,
                     visual=dict(candidate_approach=True,
                                 observation_summary='Arch on left',
                                 next_action='finish'))])]
        chosen=dict(tool='navigate_relative',
                    args=dict(turn_direction='NONE',turn_degrees=0,
                              distance_m=.25),
                    direction_intent=dict(
                        mode='CONTINUE_CURRENT_ROUTE',
                        reason='Continue toward current opening.'),
                    navigation_intent='Follow the arch slightly right.')
        m.note_navigation_turn(chosen,dict(
            tool_call_id='tool_00',status='ACTIONS_COMPLETED',
            executed_actions=2,executed_motion=[
                'TURN_LEFT','MOVE_FORWARD'],
            end_observation_id='obs_0002'),
            'obs_0000',0,'navigation_00')
        timeline=navigation_episode_history(m,m.current['pose'])
        self.assertEqual(timeline['counts']['navigation_turns'],1)
        self.assertEqual(timeline['counts']['atomic_actions'],2)
        self.assertEqual(timeline['counts']['observation_tasks'],1)
        self.assertEqual(timeline['counts']['source_frames'],3)
        kinds=[x['kind'] for x in timeline['timeline']]
        self.assertEqual(kinds.count('NAVIGATION_DECISION'),1)
        self.assertEqual(kinds.count('NAVIGATION_TOOL_RESULT'),1)
        self.assertEqual(kinds.count('ATOMIC_ACTION'),2)
        self.assertEqual(kinds.count('OBSERVATION_TASK'),1)
        self.assertEqual([v['step'] for v in timeline['timeline']],
                         sorted(v['step'] for v in timeline['timeline']))
        self.assertEqual(
            timeline['timeline'][0]['source_observation_id'],'obs_0000')
        self.assertEqual(
            timeline['timeline'][0]['direction_intent']['mode'],
            'CONTINUE_CURRENT_ROUTE')
        self.assertEqual(
            timeline['rgbd_observation_catalog'][0]['observation_id'],
            'obs_0000')
        self.assertEqual(
            timeline['rgbd_observation_catalog'][-1]['observation_id'],
            'obs_0002')
        self.assertTrue(timeline['complete_action_history'])
        self.assertTrue(timeline['all_rgbd_frames_indexed'])

    def test_old_relative_intent_is_not_current_after_turn_or_forward(self):
        m=MemoryRecords(camera())
        m.reset('episode','Follow the arch')
        m.observations=[observation('obs_0091',91,195,24)]
        m.current=m.observations[0]
        m.note_navigation_assessment(dict(
            navigation_intent='Follow the opening slightly right.',
            progress_narrative='We saw the arch.'))
        live=mission_evidence(m,pose(195,24))['progress']
        self.assertEqual(live['navigation_intent'],
                         'Follow the opening slightly right.')
        self.assertTrue(live['prior_navigation_intent'][
            'relative_direction_valid_for_current_camera'])
        after_turn=mission_evidence(m,pose(180,24))['progress']
        self.assertIsNone(after_turn['navigation_intent'])
        self.assertEqual(after_turn['prior_navigation_intent']['text'],
                         'Follow the opening slightly right.')
        self.assertTrue(after_turn['prior_navigation_intent']['historical_only'])
        after_forward=mission_evidence(m,pose(195,25))['progress']
        self.assertIsNone(after_forward['navigation_intent'])
        self.assertFalse(after_forward['prior_navigation_intent'][
            'relative_direction_valid_for_current_camera'])
        self.assertEqual(after_forward['prior_navigation_intent']['source'][
            'source_observation_id'],'obs_0091')
        # A model that merely repeats an old "slightly right" sentence
        # at obs_0095 must NOT silently reanchor it to the new camera.
        m.observations.append(observation('obs_0095',95,180,25))
        m.current=m.observations[-1]
        m.note_navigation_assessment(dict(
            navigation_intent='Follow the opening slightly right.'))
        repeated=mission_evidence(m,pose(180,25))['progress']
        self.assertIsNone(repeated['navigation_intent'])
        self.assertEqual(
            repeated['prior_navigation_intent']['source'][
                'source_observation_id'],'obs_0091')

    def test_history_not_cut_at_last_three_observation_tasks(self):
        m=MemoryRecords(camera())
        m.reset('ep','Find the rug')
        m.observations=[observation('obs_%04d'%i,i,i*15%360,i)
                        for i in range(13)]
        m.current=m.observations[-1]
        m.observation_searches=[
            dict(observation_task_id='observation_%03d'%i,
                 start_step=i,objective='Look around '+str(i),
                 start_observation_id='obs_%04d'%i,
                 end_observation_id='obs_%04d'%i,
                 outcome='OBSERVATION_COMPLETED',
                 checkpoints=[],waypoints=[],actions=[])
            for i in range(10)]
        history=navigation_episode_history(m,m.current['pose'])
        self.assertEqual(history['counts']['observation_tasks'],10)
        self.assertEqual(sum(x['kind']=='OBSERVATION_TASK'
                             for x in history['timeline']),10)
        self.assertEqual(len(history['rgbd_observation_catalog']),13)
        anchors=navigation_visual_anchors(m,'obs_0012',max_pairs=4)
        self.assertLessEqual(len(anchors),4)
        self.assertTrue(all(x['observation_id']!='obs_0012'
                            for x in anchors))
        self.assertIn('obs_0000',[x['observation_id'] for x in anchors])


class GroundedTurnContractTests(unittest.TestCase):
    def test_obs95_left_arch_right_turn_rejected_but_alternative_allowed(self):
        bad=motion_decision(
            'RIGHT','ALIGN_VISIBLE_TARGET','obs_0095',
            [.1,.25,.3,.75])
        with self.assertRaisesRegex(
                APIError,'direction_intent_turn_away_from_current_RGB_target'):
            validate_direction_intent(bad,'obs_0095',90,15)
        corrected=motion_decision(
            'LEFT','ALIGN_VISIBLE_TARGET','obs_0095',
            [.1,.25,.3,.75])
        validate_direction_intent(corrected,'obs_0095',90,15)
        alternative=motion_decision(
            'RIGHT','EXPLORE_ALTERNATIVE','obs_0095',
            [.1,.25,.3,.75])
        validate_direction_intent(alternative,'obs_0095',90,15)

    def test_historical_box_not_accepted_for_current_alignment(self):
        bad=motion_decision(
            'LEFT','ALIGN_VISIBLE_TARGET','obs_0091',
            [.1,.2,.3,.7])
        with self.assertRaisesRegex(
                APIError,'align_target_requires_current_RGB_observation_id'):
            validate_direction_intent(bad,'obs_0095',90,15)

    def test_align_without_box_is_rejected_not_autofilled(self):
        bad=motion_decision('LEFT','ALIGN_VISIBLE_TARGET','obs_0095',None)
        with self.assertRaisesRegex(
                APIError,'align_visible_target_requires_grounded_current_RGB_box'):
            validate_direction_intent(bad,'obs_0095',90,15)

    def test_near_center_box_not_an_arbitrary_hard_turn_veto(self):
        turn=motion_decision(
            'RIGHT','ALIGN_VISIBLE_TARGET','obs_0095',
            [.48,.2,.50,.8])
        validate_direction_intent(turn,'obs_0095',90,15)

    def test_registered_schema_requires_intent_only_for_motion(self):
        specs=registry(camera(),10)
        schema=navigation_schema(specs)
        Draft7Validator.check_schema(schema)
        base=dict(tool='navigate_relative',args=dict(
            turn_direction='LEFT',turn_degrees=15,distance_m=.25))
        self.assertFalse(Draft7Validator(schema).is_valid(base))
        base['direction_intent']=dict(
            mode='EXPLORE_ALTERNATIVE',reason='View a new corridor')
        self.assertTrue(Draft7Validator(schema).is_valid(base))
        self.assertTrue(Draft7Validator(schema).is_valid(dict(
            tool='observe_for_goal',args=dict(
                objective='Find a visible hall entrance'))))

    def test_real_nav_request_contains_rgbd_media_and_full_history(self):
        r,actions=test_runtime(changed_forwards=True)
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        # Add a distinct historical view: first sensor frame remains valid
        # and is indexed separately from the new camera pose.
        r._step('TURN_LEFT')
        r._step('TURN_RIGHT')
        r.navigation_mode='OBSERVE'
        with patch('havln_agent.navigation_agent.image_url',
                   return_value='data:image/jpeg;base64,R'),patch(
                   'havln_agent.navigation_agent.depth_image_url',
                   return_value='data:image/png;base64,D'):
            r.navigation_agent.decide()
        call=r.client.complete.call_args
        ctx=call.args[3]
        images=call.kwargs['image_url']
        self.assertEqual(ctx['context_contract'],
                         'navigation_context_v2_complete_causal_history')
        self.assertEqual(ctx['episode_history']['counts']['atomic_actions'],2)
        self.assertEqual(ctx['episode_history']['counts']['source_frames'],
                         len(r.memory.observations))
        self.assertEqual(ctx['current_rgb']['observation_id'],
                         r.memory.current['observation_id'])
        self.assertEqual(len(images)%2,0)
        self.assertGreaterEqual(len(images),4)
        manifest=ctx['history_media']['manifest']
        self.assertEqual([v['image_index'] for v in manifest],
                         list(range(len(images))))
        self.assertEqual(manifest[0]['kind'],'current_RGB')
        self.assertEqual(manifest[1]['kind'],'current_Depth')
        self.assertTrue(all(not i['current_sensor_frame']
                            for i in manifest[2:]))
        self.assertEqual(manifest[2]['observation_id'],'obs_0000')
        self.assertNotIn('observation_tasks',ctx['memory'])
        self.assertIn('recent_views_at_current_stance',ctx['memory'])
        self.assertLess(len(MISSION_PROMPT),5000)
        validate=call.kwargs['validate']
        old_id=ctx['current_rgb']['observation_id']
        with self.assertRaisesRegex(
                APIError,'direction_intent_turn_away_from_current_RGB_target'):
            validate(motion_decision(
                'RIGHT','ALIGN_VISIBLE_TARGET',old_id,
                [.1,.3,.3,.7]))
        validate(motion_decision(
            'RIGHT','EXPLORE_ALTERNATIVE',old_id,None))


if __name__=='__main__':
    unittest.main()
