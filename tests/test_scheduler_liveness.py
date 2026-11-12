"""Offline scheduler regressions for bounded inference and evidence liveness."""
import unittest
from types import SimpleNamespace

from havln_agent.api_client import APIError
from havln_agent.scheduler import EvidenceLivenessGuard, run


class FakeMemory:
    def __init__(self):
        self.current = dict(observation_id='obs_0097')
        self.actions = []
        self.turns = []

    def note_navigation_assessment(self, proposal):
        return {}

    def note_navigation_turn(self, proposal, result, observation_id,
                             start_step, decision_id):
        self.turns.append((proposal['tool'], result['status']))


class FakeRuntime:
    def __init__(self, decision_limit=100, unchanged=8,
                 outcome='BLOCKED', bad_format=False):
        self.config=dict(max_steps=150,max_model_decisions=decision_limit,
            max_unchanged_evidence_decisions=unchanged)
        self.memory=FakeMemory()
        self.done=False
        self.decisions=0
        self.current_decision_id=None
        self.navigation_liveness=None
        self.events=[]
        self.calls=[]
        self.termination=None
        self.api_failure=None
        self.outcome=outcome
        self.bad_format=bad_format
        self.model_calls=0
        self.client=SimpleNamespace(availability=lambda:'available')
        self.navigation_agent=SimpleNamespace(decide=self.decide)

    def start_mission(self):
        pass

    def refresh(self):
        pass

    def decision_id(self,role):
        self.current_decision_id=role+'_'+str(self.model_calls)

    def decide(self,feedback):
        self.model_calls+=1
        if self.bad_format:
            raise APIError('Missing required fields: decision_analysis',
                           category='structured_field')
        return dict(
            tool='navigate_relative',
            args=dict(turn_direction='NONE',turn_degrees=0,distance_m=0.25))

    def event(self,name,**fields):
        self.events.append(dict(event=name,**fields))

    def execute(self,call,binding):
        self.calls.append(call)
        return dict(status=self.outcome,tool=call['tool'],
            reason=('measured_centerline_obstacle' if self.outcome=='BLOCKED'
                    else 'no_new_information'),
            tool_call_id='tool_'+str(len(self.calls)),
            executed_actions=0,executed_motion=[])

    def assess(self,result):
        pass

    def terminate(self,reason):
        self.done=True
        self.termination=reason


class LivenessTests(unittest.TestCase):
    def test_identical_physically_rejected_command_is_bounded(self):
        runtime=FakeRuntime(unchanged=8)
        run(runtime,lambda role,event:None)
        self.assertEqual(runtime.termination,'navigation_no_new_evidence')
        self.assertEqual(runtime.decisions,3)
        self.assertEqual(len(runtime.calls),3)
        self.assertEqual(runtime.navigation_liveness['identical_blocked_decisions'],3)
        self.assertTrue(any(e['event']=='navigation_liveness_exhausted'
                            for e in runtime.events))
        self.assertEqual(len([e for e in runtime.events
                              if e['event']=='navigation_repeated_blocked_command']),2)

    def test_generic_no_evidence_decisions_stop_at_liveness_limit(self):
        runtime=FakeRuntime(decision_limit=100,unchanged=4,
                            outcome='NO_NEW_INFORMATION')
        run(runtime,lambda role,event:None)
        self.assertEqual(runtime.termination,'navigation_no_new_evidence')
        self.assertEqual(runtime.decisions,4)
        self.assertEqual(len(runtime.calls),4)
        self.assertEqual(runtime.navigation_liveness['identical_blocked_decisions'],0)

    def test_real_new_observation_resets_liveness(self):
        runtime=FakeRuntime(unchanged=3)
        guard=EvidenceLivenessGuard(runtime)
        self.assertFalse(guard.feedback(runtime))
        self.assertFalse(guard.feedback(runtime))
        blocked=dict(status='BLOCKED',reason='measured_centerline_obstacle',
                     executed_actions=0)
        command=dict(tool='navigate_relative',
                     args=dict(turn_direction='NONE',turn_degrees=0,distance_m=.25))
        self.assertEqual(guard.repeated_blocked_command(runtime,command,blocked),1)
        self.assertEqual(guard.repeated_blocked_command(runtime,command,blocked),2)
        runtime.memory.actions.append(dict(action='TURN_RIGHT'))
        runtime.memory.current=dict(observation_id='obs_0098')
        self.assertFalse(guard.feedback(runtime))
        self.assertEqual(guard.actionless_decisions,0)
        self.assertEqual(guard.repeated_blocked_command(runtime,command,blocked),1)
        self.assertEqual(guard.repeated_blocked_command(runtime,command,
            dict(status='ACTIONS_COMPLETED',executed_actions=1)),0)

    def test_inference_budget_still_limits_even_nonblocked_commands(self):
        runtime=FakeRuntime(decision_limit=3,unchanged=100,
                            outcome='NO_NEW_INFORMATION')
        run(runtime,lambda role,event:None)
        self.assertEqual(runtime.termination,'navigation_inference_budget_exhausted')
        self.assertEqual(runtime.decisions,3)
        self.assertEqual(len(runtime.calls),3)

    def test_structural_error_has_one_bounded_regeneration_budget(self):
        runtime=FakeRuntime(unchanged=8,bad_format=True)
        run(runtime,lambda role,event:None)
        self.assertEqual(runtime.termination,'api_error')
        self.assertEqual(runtime.model_calls,3)
        self.assertEqual(len(runtime.calls),0)
        self.assertEqual(len([e for e in runtime.events
            if e['event']=='navigation_format_regeneration']),2)
        self.assertFalse(any(e['event']=='navigation_format_cycle_recovered'
                             for e in runtime.events))


if __name__=='__main__':
    unittest.main()
