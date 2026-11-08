"""v13 internal provenance and v14 compact NavigationAgent-input regression.

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
from havln_agent.contracts import (navigation_schema, navigation_envelope,
                                   compact_navigation_protocol, registry)
from havln_agent.memory_records import MemoryRecords
from havln_agent.mission_history import mission_evidence
from havln_agent.navigation_agent import NavigationAgent, MISSION_PROMPT
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


def motion_decision(direction):
    return dict(
        tool='turn_and_observe',
        args=dict(turn_direction=direction,turn_degrees=30,
                  look='none',purpose='Inspect an opening beside the person.'))


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
        self.assertNotIn('direction_intent',timeline['timeline'][0])
        self.assertEqual(
            timeline['rgbd_observation_catalog'][0]['observation_id'],
            'obs_0000')
        self.assertEqual(
            timeline['rgbd_observation_catalog'][-1]['observation_id'],
            'obs_0002')
        self.assertTrue(timeline['complete_action_history'])
        self.assertTrue(timeline['all_rgbd_frames_indexed'])

    def test_zero_action_turns_keep_decision_result_order(self):
        m=MemoryRecords(camera())
        m.reset('ep','Continue to the rug')
        m.observations=[observation('obs_0000',0)]
        m.current=m.observations[0]
        m.note_navigation_turn(dict(tool='query_task_memory',args={}),
            dict(tool_call_id='tool_1',status='OK',executed_actions=0),
            'obs_0000',0,'navigation_1')
        m.note_navigation_turn(dict(tool='observe_for_goal',
                                     args=dict(objective='Inspect foyer')),
            dict(tool_call_id='tool_2',status='OK',executed_actions=0),
            'obs_0000',0,'navigation_2')
        timeline=navigation_episode_history(m,pose())['timeline']
        self.assertEqual([
            (v['kind'],v.get('decision_id')) for v in timeline],[
            ('NAVIGATION_DECISION','navigation_1'),
            ('NAVIGATION_TOOL_RESULT','navigation_1'),
            ('NAVIGATION_DECISION','navigation_2'),
            ('NAVIGATION_TOOL_RESULT','navigation_2')])

    def test_current_yaw_provenance_when_navigation_intent_is_anchored(self):
        # Regression for worker startup crash: mission_evidence used to
        # reference undefined 'current_y' whenever an anchored intent was
        # evaluated at a valid source stance.
        m=MemoryRecords(camera())
        m.reset('ep','Find the passage on the left')
        m.observations=[observation('obs_0000',0,90,0)]
        m.current=m.observations[0]
        m.note_navigation_assessment(dict(
            navigation_intent='Inspect the opening on the left.'))
        same=mission_evidence(m,pose(90,0))['progress']
        self.assertEqual(same['navigation_intent'],
                         'Inspect the opening on the left.')
        self.assertTrue(same['prior_navigation_intent'][
            'relative_direction_valid_for_current_camera'])
        turned=mission_evidence(m,pose(105,0))['progress']
        self.assertIsNone(turned['navigation_intent'])
        self.assertTrue(turned['prior_navigation_intent']['historical_only'])
        forward=mission_evidence(m,pose(90,1))['progress']
        self.assertIsNone(forward['navigation_intent'])

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


class SimpleNavigationContractTests(unittest.TestCase):
    def test_tool_and_args_are_sufficient_for_turns_and_moves(self):
        specs=registry(camera(),10)
        schema=navigation_schema(specs)
        envelope=navigation_envelope(specs)
        Draft7Validator.check_schema(schema)
        for direction in ('LEFT','RIGHT'):
            decision=motion_decision(direction)
            self.assertTrue(Draft7Validator(schema).is_valid(decision))
            self.assertTrue(Draft7Validator(envelope).is_valid(decision))
        move=dict(tool='navigate_relative',args=dict(
            turn_direction='NONE',turn_degrees=0,distance_m=.25))
        self.assertTrue(Draft7Validator(schema).is_valid(move))
        self.assertTrue(Draft7Validator(schema).is_valid(dict(
            tool='observe_for_goal',
            args=dict(objective='Find a visible hall entrance'))))
        # The removed field must not be part of the output protocol at
        # either level. Old responses must regenerate as tool + args.
        extra=motion_decision('RIGHT')
        extra['direction_intent']=dict(mode='EXPLORE_ALTERNATIVE',
                                      reason='Redundant wrapper')
        self.assertFalse(Draft7Validator(schema).is_valid(extra))
        self.assertFalse(Draft7Validator(envelope).is_valid(extra))
        nested=motion_decision('RIGHT')
        nested['args']['direction_intent']=extra['direction_intent']
        self.assertFalse(Draft7Validator(schema).is_valid(nested))

    def test_prompt_protocol_is_simple(self):
        protocol=compact_navigation_protocol(registry(camera(),10))
        self.assertIn('top-level tool and args',protocol)
        self.assertNotIn('direction_intent',protocol)
        self.assertNotIn('ALIGN_VISIBLE_TARGET',MISSION_PROMPT)
        self.assertNotIn('EXPLORE_ALTERNATIVE',MISSION_PROMPT)

    def test_missing_archived_frames_keep_bounded_v12_memory(self):
        r,_=test_runtime()
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        r._step('TURN_LEFT')
        r.navigation_mode='OBSERVE'
        r.frames.clear()
        with patch('havln_agent.navigation_agent.image_url',
                   return_value='data:image/jpeg;base64,R'):
            r.navigation_agent.decide()
        ctx=r.client.complete.call_args.args[3]
        images=r.client.complete.call_args.kwargs['image_url']
        self.assertEqual(len(images),1)
        self.assertEqual(len(ctx['history_media']['manifest']),1)
        self.assertNotIn('episode_history',ctx)
        self.assertEqual(ctx['memory']['contract'],'mission_evidence_v1')
        self.assertIn('observation_tasks',ctx['memory'])
        self.assertIn('historical_frames',ctx['memory'])
        self.assertIn('recent_tool_receipts',ctx['memory'])
        self.assertIn('recent_views_at_current_stance',ctx['memory'])

    def test_navigation_request_reverts_to_v12_input_plus_small_RGB_history(self):
        r,actions=test_runtime(changed_forwards=True)
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        r._step('TURN_LEFT')
        r._step('TURN_RIGHT')
        r.navigation_mode='OBSERVE'
        with patch('havln_agent.navigation_agent.image_url',
                   return_value='data:image/jpeg;base64,R'):
            r.navigation_agent.decide()
        call=r.client.complete.call_args
        ctx=call.args[3]
        images=call.kwargs['image_url']
        # This is the exact bounded mission-evidence input surface from v12,
        # not a reserialized 128-step, 10-image v13 episode transcript.
        self.assertNotIn('context_contract',ctx)
        self.assertNotIn('episode_history',ctx)
        self.assertEqual(ctx['memory']['contract'],'mission_evidence_v1')
        self.assertIn('observation_tasks',ctx['memory'])
        self.assertIn('recent_tool_receipts',ctx['memory'])
        self.assertIn('observed_forward_action_segments',ctx['memory'])
        self.assertIn('recent_views_at_current_stance',ctx['memory'])
        self.assertEqual(ctx['current_rgb']['observation_id'],
                         r.memory.current['observation_id'])
        self.assertGreaterEqual(len(images),2)
        self.assertLessEqual(len(images),4)
        manifest=ctx['history_media']['manifest']
        self.assertEqual([v['image_index'] for v in manifest],
                         list(range(len(images))))
        self.assertEqual(manifest[0]['kind'],'current_RGB')
        self.assertTrue(all(v['kind']=='historical_RGB'
                            for v in manifest[1:]))
        self.assertEqual(manifest[1]['observation_id'],'obs_0000')
        self.assertTrue(all(v.get('historical_only_not_live_clearance')
                            for v in manifest[1:]))
        self.assertEqual(ctx['history_media']['attached_historical_RGB'],
                         len(images)-1)
        self.assertLess(len(MISSION_PROMPT),5000)
        validate=call.kwargs['validate']
        # A robot may turn either direction according to model judgment.
        # No visual-box or semantic-direction gate rejects valid commands.
        validate(motion_decision('RIGHT'))
        validate(motion_decision('LEFT'))
        bad=motion_decision('LEFT')
        bad['args']['turn_degrees']=-30
        with self.assertRaises(APIError):
            validate(bad)

if __name__=='__main__':
    unittest.main()
