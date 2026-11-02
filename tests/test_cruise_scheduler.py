"""Scheduler: one Agent-owned cruise persists across safe tool batches."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from havln_agent.scheduler import run


class CruiseSchedulerTests(unittest.TestCase):
    def test_finite_cruise_continues_without_extra_model_observation(self):
        calls=[]
        memory=SimpleNamespace(
            actions=[],current=dict(observation_id='obs_0000'),
            task=SimpleNamespace(current_subgoal_index=0,completed_subgoals=[]),
            write_hypothesis=Mock())
        runtime=SimpleNamespace(
            done=False,decisions=0,config=dict(max_steps=24,
                max_model_decisions=12,max_unchanged_evidence_decisions=8),
            memory=memory,task_review_requested=False,
            client=SimpleNamespace(availability=lambda:'available'),
            navigation_mode='OBSERVE',cruise_intent=None,
            active_subgoal_id='sg_0',plan_revision=1,
            current_decision_id='navigation_agent_0000',
            api_failure=None)
        runtime.set_semantic_goal=Mock()
        runtime.refresh=Mock()
        runtime.decision_id=Mock(return_value='navigation_agent_0000')
        runtime.event=Mock()
        runtime.assess=Mock()
        runtime.terminate=Mock(side_effect=lambda reason:setattr(runtime,'done',True))
        runtime.set_navigation_mode=Mock(side_effect=lambda mode,reason:
            setattr(runtime,'navigation_mode',mode))
        plan=[dict(tool='start_cruise',args=dict(
                  turn_deg=30,distance_m=1.,max_atomic_steps=2),
                  decision_reason='Travel toward the observed corridor'),
              dict(tool='request_finish',args=dict(
                  outcome='aborted',evidence='unit-test end'))]
        runtime.navigation_agent=SimpleNamespace(decide=Mock(side_effect=plan))
        def execute(call,binding):
            calls.append((call['tool'],dict(call['args']),dict(binding)))
            if call['tool']=='start_cruise':
                memory.actions.extend([dict(action='TURN_LEFT'),
                                       dict(action='TURN_LEFT')])
                runtime.navigation_mode='CRUISE'
                runtime.cruise_intent=dict(waypoint_id='wp_0',
                    max_atomic_steps=2,plan_revision=1,subgoal_id='sg_0',
                    decision_reason='Travel toward the observed corridor')
                return dict(tool='start_cruise',tool_call_id='tool_0000',
                    status='PARTIAL',reason='tool_action_limit')
            if call['tool']=='navigate_local':
                self.assertEqual(binding['actor'],'cruise_executor')
                self.assertEqual(call['args']['waypoint_id'],'wp_0')
                self.assertEqual(call['args']['max_atomic_steps'],2)
                memory.actions.extend([dict(action='MOVE_FORWARD'),
                                       dict(action='MOVE_FORWARD')])
                runtime.navigation_mode='OBSERVE'
                runtime.cruise_intent=None
                return dict(tool='navigate_local',tool_call_id='tool_0001',
                            status='TARGET_REACHED')
            self.assertEqual(call['tool'],'request_finish')
            runtime.done=True
            return dict(tool='request_finish',tool_call_id='tool_0002',
                        status='FINISHED')
        runtime.execute=Mock(side_effect=execute)
        events=[]
        run(runtime,lambda role,event:events.append((role,event)))
        self.assertEqual(runtime.navigation_agent.decide.call_count,2)
        self.assertEqual([v[0] for v in calls],
            ['start_cruise','navigate_local','request_finish'])
        self.assertEqual([a['action'] for a in memory.actions],
            ['TURN_LEFT','TURN_LEFT','MOVE_FORWARD','MOVE_FORWARD'])
        self.assertEqual(runtime.decisions,2)
        self.assertEqual(sum(event['event']=='cruise_execution_feedback'
            for _,event in events),1)
        self.assertIsNone(runtime.cruise_intent)


    def test_cruise_batch_shrinks_to_remaining_budget_without_llm(self):
        memory=SimpleNamespace(
            actions=[dict(action='TURN_LEFT') for _ in range(21)],
            current=dict(observation_id='obs_0021'),
            task=SimpleNamespace(current_subgoal_index=0,completed_subgoals=[]),
            write_hypothesis=Mock())
        rt=SimpleNamespace(
            done=False,decisions=12,
            config=dict(max_steps=24,max_model_decisions=12,
                        max_unchanged_evidence_decisions=8),
            memory=memory,task_review_requested=False,
            client=SimpleNamespace(availability=lambda:'unavailable'),
            navigation_mode='CRUISE',
            cruise_intent=dict(waypoint_id='wp_0',max_atomic_steps=6,
                plan_revision=1,subgoal_id='sg_0',
                decision_reason='Finite route authorized by model'),
            active_subgoal_id='sg_0',plan_revision=1,
            current_decision_id='navigation_agent_0000',
            api_failure=None,refresh=Mock(),event=Mock(),assess=Mock(),
            set_semantic_goal=Mock(),navigation_agent=SimpleNamespace(
                decide=Mock(side_effect=AssertionError('unexpected model call'))))
        rt.set_navigation_mode=Mock(side_effect=lambda mode,reason,**kw:
            setattr(rt,'navigation_mode',mode))
        rt.terminate=Mock(side_effect=lambda reason:setattr(rt,'done',True))
        def execute(call,binding):
            self.assertEqual(call['tool'],'navigate_local')
            self.assertEqual(call['args']['max_atomic_steps'],2)
            self.assertEqual(call['args']['waypoint_id'],'wp_0')
            memory.actions.extend([dict(action='MOVE_FORWARD'),
                                   dict(action='MOVE_FORWARD')])
            rt.navigation_mode='OBSERVE'
            rt.cruise_intent=None
            return dict(tool_call_id='tool_2',status='TARGET_REACHED',
                        tool='navigate_local')
        rt.execute=Mock(side_effect=execute)
        run(rt,lambda role,event:None)
        rt.execute.assert_called_once()
        rt.navigation_agent.decide.assert_not_called()
        rt.terminate.assert_called_once_with('action_budget_exhausted')
        self.assertEqual(len(memory.actions),23)

    def test_rejected_cruise_batch_hands_back_to_model(self):
        memory=SimpleNamespace(actions=[],current=dict(observation_id='obs_0'),
            task=SimpleNamespace(current_subgoal_index=0,completed_subgoals=[]),
            write_hypothesis=Mock())
        rt=SimpleNamespace(done=False,decisions=0,
            config=dict(max_steps=10,max_model_decisions=10,
                        max_unchanged_evidence_decisions=8),
            memory=memory,task_review_requested=False,
            client=SimpleNamespace(availability=lambda:'available'),
            navigation_mode='CRUISE',
            cruise_intent=dict(waypoint_id='wp_0',max_atomic_steps=4,
                plan_revision=1,subgoal_id='sg_0',decision_reason='continue'),
            active_subgoal_id='sg_0',plan_revision=1,
            current_decision_id='navigation_agent_0000',
            api_failure=None,refresh=Mock(),event=Mock(),assess=Mock(),
            set_semantic_goal=Mock(),decision_id=Mock(),
            navigation_agent=SimpleNamespace(
                decide=Mock(return_value=dict(tool='request_finish',
                    args=dict(outcome='aborted',evidence='no action')))))
        rt.set_navigation_mode=Mock(side_effect=lambda mode,reason,**kw:
            setattr(rt,'navigation_mode',mode))
        rt.terminate=Mock()
        observed=[]
        def execute(call,binding):
            observed.append(call['tool'])
            if call['tool']=='navigate_local':
                return dict(tool='navigate_local',
                    tool_call_id='bad0',status='INVALID_ARGUMENTS')
            rt.done=True
            return dict(tool='request_finish',
                tool_call_id='finish0',status='FINISHED')
        rt.execute=Mock(side_effect=execute)
        run(rt,lambda role,event:None)
        self.assertEqual(observed,['navigate_local','request_finish'])
        self.assertEqual(rt.navigation_mode,'OBSERVE')
        self.assertIsNone(rt.cruise_intent)
        rt.navigation_agent.decide.assert_called_once()

if __name__=='__main__':
    unittest.main()
