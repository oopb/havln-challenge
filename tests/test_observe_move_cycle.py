"""Observe/Move state transitions and estimated motion feedback."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from havln_agent.runtime import AgentRuntime
from havln_agent.memory_service import MemoryService
from havln_agent.navigation_agent import NavigationAgent


class ObserveMoveTests(unittest.TestCase):
    def test_navigation_dispatch_encloses_motion_in_move_then_observe(self):
        r=AgentRuntime.__new__(AgentRuntime)
        pose=dict(x=0.0,z=0.0,heading=0.0)
        r.done=False
        r.navigation_mode='OBSERVE'
        r.phase='NAVIGATING'
        r.registry={'navigate_local': {'permission':'NAVIGATION'}}
        r.memory=SimpleNamespace(
            current=dict(observation_id='obs_0001'),
            observations=[dict(observation_id='obs_0001',pose=pose)])
        r.motion_controller=SimpleNamespace(pose=lambda:pose)
        r.plan_revision=1
        r.active_subgoal_id='sg_0'
        r.event=Mock()
        r.refresh=Mock()
        r._authorized_action=False
        decision=dict(plan_revision=1,subgoal_id='sg_0',observation_id='obs_0001')
        with patch('havln_agent.runtime.ToolExecutor.dispatch',
                   return_value=dict(status='TARGET_REACHED')) as run:
            answer=r.dispatch(dict(tool='navigate_local',
                                   args=dict(waypoint_id='wp_0',max_atomic_steps=2)),decision)
            self.assertEqual(answer['status'],'TARGET_REACHED')
            self.assertEqual(r.navigation_mode,'OBSERVE')
            self.assertEqual(r.phase,'NAVIGATING')
            self.assertFalse(r._authorized_action)
            run.assert_called_once()
        events=[call.args[0] for call in r.event.call_args_list]
        self.assertEqual(events.count('navigation_mode_changed'),2)

    def test_task_review_continue_does_not_cancel_waypoint(self):
        r=AgentRuntime.__new__(AgentRuntime)
        r.memory=SimpleNamespace(task=SimpleNamespace(
            current_subgoal_index=0,subgoals=[dict(action='descend stairs')]))
        r.active_subgoal_id='sg_0'
        r.active_waypoint_id='wp_0'
        r.plan_revision=8
        r.task_review_requested=True
        r.event=Mock()
        r._cancel_local_navigation=Mock()
        r.set_semantic_goal(dict(intent='continue',subgoal='descend stairs'))
        self.assertEqual(r.active_waypoint_id,'wp_0')
        self.assertEqual(r.plan_revision,8)
        self.assertFalse(r.task_review_requested)
        r._cancel_local_navigation.assert_not_called()

    def test_navigation_summary_remembers_down_then_reverse(self):
        m=MemoryService.__new__(MemoryService)
        m.runtime=SimpleNamespace(semantic_motion_policy='DESCEND_STAIRS')
        m.task=SimpleNamespace(current_subgoal_index=0)
        m.actions=[]
        m.tool_ledger=[]
        for i,(direction,height) in enumerate([('DOWN',-.22),('DOWN',-.17),('UP',.14)]):
            m.actions.append(dict(step=i,action='MOVE_FORWARD',subgoal=0,
                pose=dict(heading=0.0 if direction=='DOWN' else math.pi),
                result=dict(observation_after='obs_%04d' % (i+1),
                    rgbd_motion=dict(status='ESTIMATED',vertical_motion=direction,
                        translation_m=dict(y_up=height),uncertainty_m=.02),
                    semantic_direction_mismatch=direction=='UP')))
        summary=m.navigation_summary()
        self.assertEqual(summary['expected_vertical_motion'],'DOWN')
        self.assertEqual(summary['last_progress_motion']['step'],1)
        self.assertEqual(summary['last_motion']['vertical_motion'],'UP')
        self.assertTrue(summary['last_motion']['semantic_direction_mismatch'])


    def test_goal_alignment_survives_compact_waypoint_context(self):
        from havln_agent.contexts import waypoint_summary, tool_summary
        association=dict(grounding_id='g_1',target='stairs_down',
                         distance_before_m=1.2,distance_after_m=0.7,
                         bearing_error_degrees=0.0,direction_compatibility='CONFIRMED')
        path=dict(id='wp_1',type='GOAL_WAYPOINT',first_step='EXECUTABLE',
                  bearing_deg=0,range_m=0.5,goal_association=[association],
                  failure_history=[dict(reason='occupied_next_step',
                                        basis='estimated_local_pose_and_heading',
                                        confidence='low')])
        compact=waypoint_summary(path)
        self.assertEqual(compact['goal_alignment']['target'],'stairs_down')
        self.assertEqual(compact['goal_alignment']['distance_after_m'],0.7)
        self.assertEqual(compact['failed_approaches'][0]['reason'],'occupied_next_step')
        result=tool_summary(dict(tool='get_waypoint_candidates',status='OK',
                                 paths=[path]))
        self.assertEqual(result['paths'][0]['goal_alignment']['grounding_id'],'g_1')

    def test_observe_mode_event_reflects_navigation_phase(self):
        r=AgentRuntime.__new__(AgentRuntime)
        pose=dict(x=0.0,z=0.0,heading=0.0)
        r.done=False
        r.navigation_mode='OBSERVE'
        r.phase='NAVIGATING'
        r.registry={'navigate_local':{'permission':'NAVIGATION'}}
        r.memory=SimpleNamespace(current=dict(observation_id='obs_0001'),
                                 observations=[dict(observation_id='obs_0001',pose=pose)])
        r.motion_controller=SimpleNamespace(pose=lambda:pose)
        r.plan_revision=1
        r.active_subgoal_id='sg_0'
        events=[]
        r.event=lambda name,**kwargs: events.append((name,r.navigation_mode,r.phase))
        r.refresh=Mock()
        r._authorized_action=False
        decision=dict(plan_revision=1,subgoal_id='sg_0',observation_id='obs_0001')
        with patch('havln_agent.runtime.ToolExecutor.dispatch',
                   return_value=dict(status='PARTIAL')):
            r.dispatch(dict(tool='navigate_local',
                            args=dict(waypoint_id='wp_0',max_atomic_steps=2)),
                       decision)
        self.assertEqual(events[-1],('navigation_mode_changed','OBSERVE','NAVIGATING'))

    def test_observation_decision_rejects_move_mode(self):
        runtime=SimpleNamespace(navigation_mode='MOVE')
        with self.assertRaisesRegex(RuntimeError,'requires_observe_mode'):
            NavigationAgent(runtime).decide()


if __name__=='__main__':
    unittest.main()
