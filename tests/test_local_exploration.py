"""Regress smoke-500-16's post-obstacle TURN/LOOK oscillation.

The simulated robot has moved 0.25m inside navigate_local, then received
PARTIAL + interrupted_status=BLOCKED on the next step. This failure must
persist in navigation context and suppress repeated absolute camera views
without preventing genuinely new headings or fresh collision revalidation.
"""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from havln_agent.local_exploration import (
    last_block, local_coverage, predicted_scan, same_stance, covered_view,
    multi_pitch_coverage, observation_stagnation)
from havln_agent.runtime import AgentRuntime
from havln_agent.contexts import navigation_context, tool_summary
from havln_agent.active_observation import ObservationController


def pose(heading=0, pitch=-60, z=.25, epoch=0):
    return dict(x=0.,z=z,y=0.,heading=math.radians(heading),
                pitch_deg=pitch,motion_epoch=epoch,position_uncertainty=.05)


def record(ident, heading, pitch=-60, z=.25, *, step=3,
           motion='BLOCKED',epoch=0):
    return dict(observation_id=ident,subgoal=0,step=step,
        pose=pose(heading,pitch,z,epoch),
        rgb_hash='rgb'+ident,depth_hash='depth'+ident,
        directions=[dict(angle_degrees=0,motion_state=motion,clearance=.2)])


class FakeMemory:
    def __init__(self):
        self.task=SimpleNamespace(current_subgoal_index=0,
            subgoals=[dict(target='stairs',landmark='stairs')])
        self.actions=[dict(action='MOVE_FORWARD')]+[
            dict(action='TURN_LEFT') for _ in range(5)]
        self.observations=[
            record('obs_at_end',90,step=1),
            record('obs_0090',90,step=2),
            record('obs_0105',105,step=3),
            record('obs_0120',120,step=4)]
        self.current=self.observations[-1]
        self.scene_memory=None
        self.tool_ledger=[dict(call_id='tool_block',
            call=dict(tool='navigate_local'),
            request_context=dict(pose=pose(90,z=0)),
            result=dict(tool='navigate_local',status='PARTIAL',
                interrupted_status='BLOCKED',reason='occupied_next_step',
                end_observation_id='obs_at_end',
                safety=dict(path_state='OCCUPIED',clearance=.245,
                    required_forward_step=.25,rejection_reasons=[
                        'clearance_below_forward_step'],
                    nearest_collision_point=dict(pixel_xy=[137,139]))))]
        self.events=[]
        self.observation_searches=[]
        self.keyframes={}
        self.dynamic_summaries=lambda place:[]
        self.recall_observation=lambda place,landmark,top_k:dict(observations=[])
        self.current_place_id='P0'
        self.scene_history=[]
        self.groundings=[]
        self.note_candidates=Mock()


class ExplorationTests(unittest.TestCase):
    def setUp(self):
        self.mem=FakeMemory()

    def test_partial_block_is_preserved_at_post_move_pose(self):
        block=last_block(self.mem,pose(120))
        self.assertIsNotNone(block)
        self.assertEqual(block['outer_status'],'PARTIAL')
        self.assertEqual(block['interrupted_status'],'BLOCKED')
        self.assertEqual(block['observation_id'],'obs_at_end')
        self.assertEqual(block['absolute_heading_deg'],90)
        self.assertFalse(block['requires_live_revalidation'] is False)
        self.assertIsNone(last_block(self.mem,pose(120,z=0)))
        self.assertIsNone(last_block(self.mem,pose(120,epoch=1)))

    def test_history_does_not_expire_when_a_nonexecuted_route_is_rejected(self):
        self.mem.tool_ledger.append(dict(call_id='tool_no_path',
            call=dict(tool='navigate_local'),
            result=dict(status='NEEDS_VIEW',executed_motion=[])))
        self.assertIsNotNone(last_block(self.mem,pose(120)))
        self.mem.tool_ledger.append(dict(call_id='tool_turn_only',
            call=dict(tool='navigate_local'),
            result=dict(status='PARTIAL',reason='tool_action_limit',
                        executed_motion=['TURN_LEFT','TURN_LEFT'])))
        self.assertIsNotNone(last_block(self.mem,pose(120)))
        self.mem.tool_ledger.append(dict(call_id='tool_moved',
            call=dict(tool='navigate_local'),
            result=dict(status='TARGET_REACHED',executed_motion=['MOVE_FORWARD'])))
        self.assertIsNone(last_block(self.mem,pose(120)))

    def test_direct_motion_block_is_retained_at_post_action_pose(self):
        # Unlike old navigate_local, model-directed navigation can move
        # without generating a waypoint first. Both need identical physical
        # feedback in the navigation context.
        for tool in ('navigate_relative','navigate_to_pixel'):
            with self.subTest(tool=tool):
                self.mem.tool_ledger=[dict(call_id='direct_block',
                    call=dict(tool=tool),
                    result=dict(status='PARTIAL',interrupted_status='BLOCKED',
                        reason='occupied_next_step',executed_motion=['TURN_RIGHT'],
                        end_observation_id='obs_at_end',
                        safety=dict(path_state='OCCUPIED',clearance=.2)))]
                block=last_block(self.mem,pose(120))
                self.assertIsNotNone(block)
                self.assertEqual(block['tool_call_id'],'direct_block')
                self.assertEqual(block['absolute_heading_deg'],90)
                self.assertEqual(block['status'],'BLOCKED')
                self.assertIsNone(last_block(self.mem,pose(120,epoch=3)))

    def test_new_direct_success_supersedes_previous_collision(self):
        self.mem.tool_ledger.append(dict(call_id='direct_success',
            call=dict(tool='navigate_relative'),
            result=dict(status='TARGET_REACHED',executed_motion=['MOVE_FORWARD'])))
        self.assertIsNone(last_block(self.mem,pose(120)))

    def test_absolute_coverage_detects_repeated_relative_sweep(self):
        state=local_coverage(self.mem,pose(120),15,24)
        self.assertIn(90,state['checked_absolute_headings_deg'])
        self.assertIn(105,state['historical_blocked_headings_deg'])
        self.assertTrue(state['unexplored_directions'])
        self.assertFalse(state['motion_authorized'])
        repeated=predicted_scan(self.mem,pose(120),-30,'none',15,30)
        self.assertFalse(repeated['new_view_expected'])
        fresh=predicted_scan(self.mem,pose(120),30,'none',15,30)
        self.assertTrue(fresh['new_view_expected'])
        self.assertEqual(fresh['unseen_views'][0]['heading_deg'],135)
        self.assertTrue(predicted_scan(self.mem,pose(120),-30,'up',15,30)['new_view_expected'])

    def test_360_wrap_new_epoch_and_translation_invalidate_old_view(self):
        self.mem.observations.append(record('wrapped',-15,step=6))
        self.assertEqual(covered_view(self.mem.observations,pose(345),0,345,-60,15)
                         ['observation_id'],'wrapped')
        self.assertFalse(same_stance(pose(),pose(z=.5)))
        self.assertIsNone(covered_view(self.mem.observations,
            pose(105,epoch=1),0,105,-60,15))

    def test_multi_pitch_coverage_preserves_absolute_angles_at_same_stance(self):
        # No separate mutable map: infer coverage from actual public views.
        self.mem.observations=[
            record('h90',90,pitch=0,step=9),
            record('h90repeat',90,pitch=0,step=10),
            record('down90',90,pitch=-30,step=11),
            record('down120',120,pitch=-30,step=12),
            record('other_epoch',180,pitch=-30,step=13,epoch=1),
            record('other_location',135,pitch=-30,step=14,z=.5)]
        self.mem.observations[1]['revisited_static_view']=True
        coverage=multi_pitch_coverage(self.mem,pose(90,pitch=0),15,30)
        self.assertEqual(coverage['unique_heading_pitch_cells'],3)
        self.assertEqual(coverage['repeated_samples'],1)
        levels={x['pitch_deg']:x for x in coverage['observed_pitch_levels']}
        self.assertEqual(levels[0]['observed_headings_deg'],[90.0])
        self.assertEqual(levels[-30]['observed_headings_deg'],[90.0,120.0])
        self.assertEqual(levels[0]['repeated_observation_samples'],1)
        self.assertEqual(coverage['current_pitch_checked_headings_deg'],[90.0])
        self.assertFalse(coverage['measurement'].endswith('path_clearance') is False)

    def test_repeated_look_and_waypoint_queries_trigger_information_intervention(self):
        self.mem.actions=[dict(action='TURN_LEFT',step=i) for i in range(12)]
        self.mem.observations=[]
        self.mem.tool_ledger=[]
        for i in range(10):
            obs=record('obs_loop_%d' % i,90,pitch=(0 if i%2 else -30),
                       step=i+1,motion='NEEDS_VIEW')
            obs['new_visual_information']=False
            obs['revisited_static_view']=bool(i>1)
            self.mem.observations.append(obs)
            self.mem.tool_ledger.append(dict(
                call=dict(tool='turn_and_observe' if i%2 else
                    'get_waypoint_candidates'),
                result=dict(status='NO_NEW_INFORMATION',
                    end_observation_id=obs['observation_id'],paths=[])))
        self.mem.current=self.mem.observations[-1]
        focus=observation_stagnation(self.mem,pose(90,pitch=0))
        self.assertTrue(focus['repeated_low_information'])
        self.assertEqual(focus['low_information_calls'],10)
        self.assertEqual(focus['same_stance_observation_calls'],10)
        # A measured translation or reset changes location and clears
        # the same-stance observation-loop warning.
        self.assertFalse(observation_stagnation(self.mem,pose(90,z=.6))[
            'repeated_low_information'])
        self.assertFalse(observation_stagnation(self.mem,pose(90,epoch=2))[
            'repeated_low_information'])
        # A new semantic goal at the SAME physical stance may reuse old
        # camera coverage but must not inherit the old goal's stall alert.
        self.mem.task.current_subgoal_index=1
        self.assertFalse(observation_stagnation(self.mem,pose(90,pitch=0))[
            'repeated_low_information'])
        self.assertEqual(multi_pitch_coverage(self.mem,pose(90,pitch=0),15,30)[
            'unique_heading_pitch_cells'],2)

    def test_pending_unknown_waypoint_does_not_claim_current_feasibility(self):
        self.mem.active_attempt=dict(unknown_motion_seen=True,
            forward_actions=1,forward_action_horizon=3)
        self.mem.tool_ledger.append(dict(
            call=dict(tool='navigate_local',args=dict(waypoint_id='path_old')),
            result=dict(tool='navigate_local',waypoint_id='path_old',
                status='NEEDS_VIEW',reason='unknown_depth_or_fov',
                end_observation_id='obs_0120')))
        runtime=SimpleNamespace(memory=self.mem,
            motion_controller=SimpleNamespace(pose=lambda:pose(120),
                target=(1.,1.),next_action=lambda:'MOVE_FORWARD'),
            active_subgoal=dict(action='descend',target='stairs'),
            active_waypoint_id='path_old',pitch_deg=-60,
            camera=SimpleNamespace(turn_angle=15,tilt_angle=30),
            config=dict(max_steps=150,max_scene_age_seconds=300),
            paths={})
        with patch('havln_agent.contexts.turn_evidence',
                   return_value=dict(requirement=None)):
            result=navigation_context(runtime)
        waypoint=result['local_navigation']
        self.assertEqual(waypoint['status'],'IN_PROGRESS')
        self.assertFalse(waypoint['endpoint_estimate_valid'])
        self.assertEqual(waypoint['last_failure_reason'],'unknown_depth_or_fov')
        self.assertEqual(waypoint['execution_feasibility'],
                         'REQUIRES_NEW_LIVE_SAFETY_EVIDENCE')
        self.assertIn('observation_coverage',result)
        self.assertIn('observation_stagnation',result)

    def test_navigation_context_preserves_collision_and_coverage(self):
        runtime=SimpleNamespace(memory=self.mem,
            motion_controller=SimpleNamespace(pose=lambda:pose(120),target=None),
            active_subgoal=dict(action='descend',target='stairs'),
            active_waypoint_id=None,pitch_deg=-60,
            camera=SimpleNamespace(turn_angle=15,tilt_angle=30),
            config=dict(max_steps=150,max_scene_age_seconds=300),
            paths={})
        with patch('havln_agent.contexts.turn_evidence',
                   return_value=dict(requirement=None)):
            result=navigation_context(runtime)
        block=result['recent_navigation_block']
        self.assertEqual(block['status'],'BLOCKED')
        self.assertEqual(block['measured_collision']['clearance'],.245)
        self.assertIn(90,result['local_exploration']['checked_absolute_headings_deg'])
        self.assertEqual(result['last_tool']['interrupted_status'],'BLOCKED')

    def agent(self):
        runtime=AgentRuntime.__new__(AgentRuntime)
        runtime.camera=SimpleNamespace(turn_angle=15,tilt_angle=30)
        runtime.config=dict(tool_max_actions=3,max_steps=150)
        runtime.memory=self.mem
        runtime.motion_controller=SimpleNamespace(
            pose=lambda:pose(120),heading=math.radians(120),
            pitch_deg=-60,target=None)
        runtime.done=False
        runtime._step=Mock(return_value=dict(status='EXECUTED'))
        runtime.source=Mock(return_value=dict(observation_id='source'))
        return runtime

    def test_runtime_suppresses_revisited_turn_without_spending_actions(self):
        runtime=self.agent()
        result=runtime._turn_and_observe(-30,'none')
        self.assertEqual(result['status'],'NO_NEW_INFORMATION')
        self.assertEqual(result['executed_actions'] if 'executed_actions' in result else 0,0)
        self.assertFalse(result['exploration']['motion_authorized'])
        runtime._step.assert_not_called()

    def test_runtime_allows_unseen_heading_after_obstacle(self):
        runtime=self.agent()
        states=iter([dict(observation_id='new1',new_visual_information=True),
                     dict(observation_id='new2',new_visual_information=True)])
        def step(action):
            runtime.memory.current=next(states)
            return dict(status='EXECUTED')
        runtime._step=Mock(side_effect=step)
        result=runtime._turn_and_observe(30,'none')
        self.assertEqual(result['status'],'OK')
        self.assertEqual(runtime._step.call_count,2)
        self.assertEqual(result['scan']['new_information_ids'],['new1','new2'])

    def test_active_observation_reuses_coverage_without_vlm_or_turn(self):
        runtime=self.agent()
        runtime.visual_perception=SimpleNamespace(search_view=Mock())
        runtime.memory.current=self.mem.observations[-1]
        result=ObservationController(runtime).scan('find stairs',
            ['stair_entry'],'right',30,checkpoint_degrees=30)
        self.assertEqual(result['status'],'NO_NEW_INFORMATION')
        self.assertEqual(result['outcome'],'REPEATED_SEARCH_NO_NEW_VIEW')
        runtime.visual_perception.search_view.assert_not_called()
        runtime._step.assert_not_called()

    def test_explicit_dynamic_recheck_is_model_directed_not_throttled(self):
        runtime=self.agent()
        values=iter([dict(observation_id='dynamic1',new_visual_information=False),
                     dict(observation_id='dynamic2',new_visual_information=False),
                     dict(observation_id='dynamic3',new_visual_information=False),
                     dict(observation_id='dynamic4',new_visual_information=False)])
        runtime.motion_controller.pose=lambda:pose(
            math.degrees(runtime.motion_controller.heading))
        def execute_turn(action):
            runtime.motion_controller.heading+=math.radians(-15 if action=='TURN_RIGHT' else 15)
            runtime.memory.current=next(values)
            return dict(status='EXECUTED')
        runtime._step=Mock(side_effect=execute_turn)
        # The event logs revisits without imposing a wall-clock lockout.
        runtime.memory.event=lambda kind,**kw:runtime.memory.events.append(dict(event=kind,**kw))
        first=runtime._turn_and_observe(-30,'none',purpose='dynamic_recheck')
        self.assertEqual(first['status'],'NO_NEW_INFORMATION') # no new static view
        self.assertEqual(runtime._step.call_count,2)
        # Return to the same camera heading without clearing the evidence.
        runtime.motion_controller.heading=math.radians(120)
        runtime._step.reset_mock()
        second=runtime._turn_and_observe(-30,'none',purpose='dynamic_recheck')
        self.assertEqual(second['status'],'NO_NEW_INFORMATION')
        self.assertEqual(runtime._step.call_count,2)
        self.assertEqual(runtime.memory.current['observation_id'],'dynamic4')


if __name__=='__main__':
    unittest.main()
