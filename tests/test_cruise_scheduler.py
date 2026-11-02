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


if __name__=='__main__':
    unittest.main()
