"""Smoke-500-17: zero-action decisions must not loop through 1000 paid calls."""
import unittest
from types import SimpleNamespace

from havln_agent.scheduler import EvidenceLivenessGuard, run


class FakeRuntime:
    def __init__(self, decision_limit=100, unchanged=3):
        self.config=dict(max_steps=150,max_model_decisions=decision_limit,
            max_unchanged_evidence_decisions=unchanged)
        self.memory=SimpleNamespace(
            current=dict(observation_id='obs_0097'),actions=[],
            task=SimpleNamespace(current_subgoal_index=0,completed_subgoals=[]))
        self.done=False
        self.decisions=0
        self.current_decision_id=None
        self.plan_revision=1
        self.active_subgoal_id='sg_0'
        self.task_review_requested=False
        self.review_reason=None
        self.navigation_liveness=None
        self.events=[]
        self.calls=[]
        self.reviews=0
        self.termination=None
        self.api_failure=None
        self.client=SimpleNamespace(availability=lambda:'available')
        self.navigation_agent=SimpleNamespace(decide=lambda feedback:dict(
            tool='turn_and_observe',args=dict(relative_turn_deg=0,look='up')))
        self.task_planner=SimpleNamespace(review=lambda feedback: self.review())

    def review(self):
        self.reviews+=1
        return dict(subgoal='descend stairs',success_criteria=['entry_confirmed'])

    def set_semantic_goal(self,value=None):
        self.task_review_requested=False
        if value is not None:self.plan_revision+=1

    def refresh(self):
        pass

    def decision_id(self,role):
        self.current_decision_id=role+'_'+str(self.decisions)

    def event(self,name,**fields):
        self.events.append(dict(event=name,**fields))

    def execute(self,call,binding):
        self.calls.append(call)
        return dict(status='NO_NEW_INFORMATION',tool=call['tool'],
            tool_call_id='tool_'+str(len(self.calls)),
            reason='repeated_camera_sector_after_measured_block',
            executed_actions=0,executed_motion=[])

    def assess(self,result):
        pass

    def terminate(self,reason):
        self.done=True
        self.termination=reason


class LivenessTests(unittest.TestCase):
    def test_zero_evidence_loop_reviews_once_then_finishes(self):
        runtime=FakeRuntime(unchanged=3)
        events=[]
        run(runtime,lambda role,event:events.append((role,event)))
        self.assertEqual(runtime.termination,'navigation_no_new_evidence')
        self.assertEqual(runtime.reviews,1)
        self.assertEqual(len(runtime.calls),6)
        self.assertEqual(runtime.decisions,7) # six navigations + one review
        self.assertEqual(runtime.navigation_liveness['consecutive_unchanged_decisions'],6)
        self.assertEqual(len([e for e in runtime.events if e['event']=='navigation_liveness_review']),1)
        self.assertEqual(len([e for e in runtime.events if e['event']=='navigation_liveness_exhausted']),1)

    def test_real_action_resets_stall_even_when_model_repeats_tool(self):
        runtime=FakeRuntime(unchanged=3)
        guard=EvidenceLivenessGuard(runtime)
        self.assertIsNone(guard.feedback(runtime))
        self.assertIsNone(guard.feedback(runtime))
        runtime.memory.actions.append(dict(action='MOVE_FORWARD'))
        runtime.memory.current=dict(observation_id='obs_0098')
        self.assertIsNone(guard.feedback(runtime))
        self.assertEqual(guard.actionless_decisions,0)
        self.assertFalse(guard.review_requested)
        self.assertIsNone(guard.feedback(runtime))

    def test_planner_rewriting_goal_without_new_evidence_does_not_reset(self):
        runtime=FakeRuntime(unchanged=2)
        guard=EvidenceLivenessGuard(runtime)
        self.assertIsNone(guard.feedback(runtime))
        self.assertEqual(guard.feedback(runtime),'REVIEW_NO_PROGRESS')
        runtime.plan_revision+=1
        self.assertIsNone(guard.feedback(runtime))
        self.assertEqual(guard.feedback(runtime),'NO_EVIDENCE_PROGRESS')

    def test_inference_budget_limits_tool_calls_even_if_no_evidence_limit_high(self):
        runtime=FakeRuntime(decision_limit=4,unchanged=100)
        run(runtime,lambda role,event:None)
        self.assertEqual(runtime.termination,'navigation_inference_budget_exhausted')
        self.assertEqual(runtime.decisions,4)
        self.assertEqual(len(runtime.calls),4)
        self.assertEqual(runtime.reviews,0)
        self.assertTrue(any(e['event']=='navigation_inference_budget_exhausted'
                            for e in runtime.events))


if __name__=='__main__':
    unittest.main()
