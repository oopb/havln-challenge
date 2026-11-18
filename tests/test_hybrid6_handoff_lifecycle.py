"""Hybrid-6 regression: handoff lifecycle, action horizon and route evidence.

Uses the real router/contracts and synthetic action receipts, never Habitat or
external model calls. In particular, "after one atomic forward" must not
result in nine local forwards just because the scene remains visually clear.
"""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jsonschema import Draft7Validator

from havln_agent.contracts import (
    navigation_schema, navigation_choice_schema, registry)
from havln_agent.hybrid_router import HybridRouter
from havln_agent.local_agent import LocalAgent


def grant(limit=1):
    return dict(tool='navigate_relative',local_handoff=dict(
        goal='Reach the lower tiled landing',
        exit_condition='Return to cloud after one forward or at the landing',
        max_forward_actions=limit))


def forward_receipt():
    return dict(status='ACTIONS_COMPLETED',executed_actions=1,
                executed_motion=['MOVE_FORWARD'])


def runtime(heading=195):
    events=[]
    start=dict(action='MOVE_FORWARD',pose=dict(heading=math.radians(heading)),
               result=dict(visual_change=True))
    rt=SimpleNamespace(
        memory=SimpleNamespace(
            current=dict(observation_id='obs_0042',
                         pose=dict(heading=math.radians(heading))),
            actions=[start]),
        current_decision_id='navigation_agent_0032',
        event=lambda name,**kwargs:events.append(dict(event=name,**kwargs)))
    rt.events=events
    return rt


def local_forward(rt,heading=195):
    next_id='obs_%04d' % (42+len(rt.memory.actions))
    rt.memory.actions.append(dict(
        action='MOVE_FORWARD',pose=dict(heading=math.radians(heading)),
        result=dict(visual_change=True)))
    rt.memory.current=dict(
        observation_id=next_id,pose=dict(heading=math.radians(heading)))


def local_decision(rt,count=1,action='FORWARD',signal='NORMAL'):
    return dict(action=action,count=count,signal=signal,
                evidence_id=rt.memory.current['observation_id'])


class Hybrid6HandoffTests(unittest.TestCase):
    def test_schema_requires_explicit_integer_forward_horizon(self):
        tools=registry(None,100)
        schema=navigation_choice_schema(tools,allow_local_handoff=True)
        complete=navigation_schema(tools,allow_local_handoff=True)
        decision=dict(
            observed_evidence=['Current RGB shows the landing.'],
            decision_analysis='Take a short step onto flat floor.',
            tool='navigate_relative',
            local_handoff=grant(1)['local_handoff'])
        self.assertTrue(Draft7Validator(schema).is_valid(decision))
        self.assertTrue(Draft7Validator(complete).is_valid(dict(decision,args={
            'turn_direction':'NONE','turn_degrees':0,'distance_m':.25})))
        for invalid in [None,0,13,True,1.5,'1']:
            candidate=dict(decision)
            handoff=dict(decision['local_handoff'])
            if invalid is None:
                handoff.pop('max_forward_actions')
            else:
                handoff['max_forward_actions']=invalid
            candidate['local_handoff']=handoff
            with self.subTest(invalid=invalid):
                self.assertFalse(Draft7Validator(schema).is_valid(candidate))
        self.assertNotIn('local_handoff',
            navigation_choice_schema(tools)['properties'])

    def test_hybrid6_one_forward_means_one_forward_not_nine(self):
        rt=runtime()
        router=HybridRouter(dict(enabled=True))
        router.after_cloud(grant(1),forward_receipt(),rt,'obs_0040')
        self.assertEqual(router.remaining_forward_actions(),1)
        self.assertTrue(router.use_local())
        decision=local_decision(rt,1)
        self.assertIsNone(router.approve_local(
            decision,rt.memory.current['observation_id'],'CLEAR'))
        local_forward(rt)
        outcome=router.after_local(decision,forward_receipt(),rt)
        self.assertEqual(outcome,'delegated_forward_limit_reached')
        self.assertFalse(router.use_local())
        self.assertEqual(router.escalations,1)
        self.assertEqual(router.local_forward_actions,1)
        self.assertEqual(router.last_local_summary['executed_forward'],1)
        self.assertEqual(router.last_local_summary['handoff_forward_limit'],1)
        # No further local command is permitted until cloud delegates again.
        self.assertEqual(router.approve_local(
            local_decision(rt,1),rt.memory.current['observation_id'],'CLEAR'),
            'no_active_cloud_route')
        self.assertEqual(len(rt.memory.actions),2)  # cloud 1 + local 1

    def test_over_budget_local_command_is_rejected_not_clamped(self):
        rt=runtime()
        router=HybridRouter(dict(enabled=True))
        router.after_cloud(grant(1),forward_receipt(),rt,'obs_0040')
        proposal=local_decision(rt,2)
        self.assertEqual(router.approve_local(
            proposal,rt.memory.current['observation_id'],'CLEAR'),
            'delegated_forward_limit_would_be_exceeded')
        self.assertEqual(proposal['count'],2)
        self.assertEqual(router.local_forward_actions,0)
        self.assertEqual(len(rt.memory.actions),1)

    def test_cumulative_forward_receipts_and_goal_cue(self):
        rt=runtime()
        router=HybridRouter(dict(enabled=True))
        router.after_cloud(grant(3),forward_receipt(),rt,'obs_0040')
        two=local_decision(rt,2)
        self.assertIsNone(router.approve_local(
            two,rt.memory.current['observation_id'],'CLEAR'))
        local_forward(rt)
        local_forward(rt)
        receipt=dict(status='ACTIONS_COMPLETED',
                     executed_actions=2,
                     executed_motion=['MOVE_FORWARD','MOVE_FORWARD'])
        self.assertIsNone(router.after_local(two,receipt,rt))
        self.assertEqual(router.remaining_forward_actions(),1)
        self.assertEqual(router.approve_local(
            local_decision(rt,2),rt.memory.current['observation_id'],'CLEAR'),
            'delegated_forward_limit_would_be_exceeded')
        self.assertEqual(router.approve_local(
            local_decision(rt,0,action='ESCALATE',signal='GOAL_CUE'),
            rt.memory.current['observation_id'],'CLEAR'),
            'local_signal_goal_cue')
        router.escalate('local_signal_goal_cue',rt)
        self.assertEqual(router.last_local_summary['executed_forward'],2)

    def test_cloud_turn_does_not_erase_stair_exit_travel_reference(self):
        rt=runtime(195)
        router=HybridRouter(dict(enabled=True))
        router.after_cloud(grant(1),forward_receipt(),rt,'obs_0040')
        local_forward(rt,heading=195)
        router.after_local(local_decision(rt),forward_receipt(),rt)
        # Cloud turns 135 deg RIGHT, without any forward translation.
        for k in range(9):
            rt.memory.actions.append(dict(
                action='TURN_RIGHT',
                pose=dict(heading=math.radians(195-15*(k+1))),
                result=dict(visual_change=True)))
        rt.memory.current=dict(
            observation_id='obs_0060',pose=dict(heading=math.radians(60)))
        router.after_cloud(dict(tool='turn_and_observe'),
            dict(status='ACTIONS_COMPLETED',executed_motion=['TURN_RIGHT']*9),
            rt,'obs_0042')
        feedback=router.cloud_feedback(rt)
        self.assertEqual(feedback['local_execution_summary']['executed_forward'],1)
        self.assertEqual(feedback['local_execution_summary']['executed_right'],0)
        self.assertEqual(feedback['cloud_actions_since_local_return']['turns'],9)
        self.assertEqual(feedback[
            'last_local_forward_view_offset_deg_from_current'],135)
        self.assertTrue(feedback[
            'last_local_forward_is_historical_view_not_motor_target'])
        self.assertNotIn('heading_deg',str(feedback))
        # A subsequent cloud-only look retains exactly the same anchor.
        rt.memory.actions.append(dict(action='LOOK_UP',pose=dict(
            heading=math.radians(60))))
        router.after_cloud(dict(tool='turn_and_observe'),
            dict(status='ACTIONS_COMPLETED',executed_motion=['LOOK_UP']),
            rt,'obs_0060')
        feedback=router.cloud_feedback(rt)
        self.assertEqual(feedback['cloud_actions_since_local_return']['turns'],9)
        self.assertEqual(feedback['cloud_actions_since_local_return']['looks'],1)
        # A genuinely executed cloud forward supersedes the old route anchor.
        router.after_cloud(dict(tool='navigate_relative'),forward_receipt(),
                           rt,'obs_0060')
        self.assertIsNone(router.cloud_feedback(rt))

    def test_scheduler_reacquires_cloud_after_exactly_one_local_forward(self):
        """Integration check: the scheduler must actually switch tiers."""
        from havln_agent.scheduler import run
        from test_hybrid_routing import HybridSchedulerTests
        rt=HybridSchedulerTests.FakeRuntime()
        def cloud_decide(feedback):
            rt.cloud_calls+=1
            return dict(
                tool='navigate_relative',
                args=dict(turn_direction='NONE',turn_degrees=0,distance_m=.25),
                local_handoff=grant(1)['local_handoff'])
        rt.navigation_agent.decide=cloud_decide
        rt.assess=lambda result:setattr(
            rt,'done',len(rt.memory.actions)>=3)
        decisions=[]
        run(rt,lambda role,value:decisions.append(value))
        self.assertEqual(rt.local_calls,1)
        self.assertEqual(rt.hybrid_router.escalations,1)
        self.assertEqual(len(rt.executed),3)
        self.assertEqual([binding['actor'] for _,binding in rt.executed],
                         ['navigation_agent','local_agent','navigation_agent'])
        self.assertEqual(rt.hybrid_router.local_forward_actions,1)

    def test_local_model_receives_remaining_horizon_as_numeric_context(self):
        rt=runtime()
        router=HybridRouter(dict(enabled=True))
        router.after_cloud(grant(1),forward_receipt(),rt,'obs_0040')
        rt.hybrid_router=router
        rt.refresh=Mock()
        rt.forward_preflight=lambda stage:dict(status='CLEAR')
        rt.hybrid_settings=dict(keyframes_max=10,image_max_edge=448)
        rt.pitch_deg=-60
        rt.camera=SimpleNamespace(forward_step=.25,turn_angle=15)
        rt.client=SimpleNamespace(complete=Mock(return_value=dict(
            action='ESCALATE',count=0,signal='GOAL_CUE',
            evidence_id='obs_0042')))
        with patch('havln_agent.local_agent.select_local_frames',
                   return_value=([],[],None)):
            result,_=LocalAgent(rt).decide()
        context=rt.client.complete.call_args.args[3]
        self.assertEqual(context['remaining_forward_actions'],1)
        self.assertEqual(context['executed_forward_actions'],0)
        self.assertEqual(context['camera_pitch_deg'],-60)
        self.assertEqual(result['signal'],'GOAL_CUE')


if __name__=='__main__':
    unittest.main()
