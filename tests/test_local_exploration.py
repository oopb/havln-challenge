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
    last_block, local_coverage, predicted_scan, same_stance, covered_view)
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
        self.mem.tool_ledger.append(dict(call_id='tool_moved',
            call=dict(tool='navigate_local'),
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

    def test_navigation_context_preserves_collision_and_coverage(self):
        runtime=SimpleNamespace(memory=self.mem,
            motion_controller=SimpleNamespace(pose=lambda:pose(120),target=None),
            active_subgoal=dict(action='descend',target='stairs'),
            active_waypoint_id=None,pitch_deg=-60,
            camera=SimpleNamespace(turn_angle=15),
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

    def test_explicit_dynamic_recheck_is_throttled(self):
        runtime=self.agent()
        values=iter([dict(observation_id='dynamic1',new_visual_information=False),
                     dict(observation_id='dynamic2',new_visual_information=False)])
        runtime._step=Mock(side_effect=lambda action:
            (setattr(runtime.memory,'current',next(values)) or dict(status='EXECUTED')))
        # Memory event() stores the rate-limiting receipt.
        runtime.memory.event=lambda kind,**kw:runtime.memory.events.append(dict(event=kind,**kw))
        first=runtime._turn_and_observe(-30,'none',purpose='dynamic_recheck')
        self.assertEqual(first['status'],'NO_NEW_INFORMATION') # no new static view
        self.assertEqual(runtime._step.call_count,2)
        runtime._step.reset_mock()
        second=runtime._turn_and_observe(-30,'none',purpose='dynamic_recheck')
        self.assertEqual(second['status'],'NO_NEW_INFORMATION')
        runtime._step.assert_not_called()


if __name__=='__main__':
    unittest.main()
