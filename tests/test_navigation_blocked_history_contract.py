"""Regression for eval-batch4-25 worker KeyError('observation_id').

The mission_history projection normalizes physical obstacle records to
source_observation_id. NavigationAgent telemetry must read that exact key,
including after a real obstacle attempt has been recorded. This test uses
the actual NavigationAgent.decide() path, not only mission_evidence().
"""
import unittest
from unittest.mock import Mock

from havln_agent.mission_history import mission_evidence
from test_forward_preflight_v9 import test_runtime, geometry, center


class NavigationBlockedHistorySourceContract(unittest.TestCase):
    def test_model_decision_after_measured_forward_block_does_not_crash(self):
        r,_=test_runtime()
        r.geometry=Mock(return_value=geometry('EXECUTABLE'))
        source=r.source()
        r.memory.evidence.blocked_forward(
            source, r.motion_controller.pose(),
            reason='measured_centerline_obstacle',
            center=center(),place_id=r.memory.current.get('place_id'))
        history=mission_evidence(r.memory,r.motion_controller.pose())
        self.assertEqual(len(history['blocked_at_source_stances']),1)
        self.assertEqual(history['blocked_at_source_stances'][0][
            'source_observation_id'],source['observation_id'])
        self.assertNotIn('observation_id',
            history['blocked_at_source_stances'][0])
        emitted=[]
        r.event_callback=emitted.append
        proposal=r.navigation_agent.decide()
        self.assertEqual(proposal['tool'],'turn_and_observe')
        built=[e for e in emitted if e['event']=='mission_context_built']
        self.assertEqual(len(built),1)
        self.assertEqual(built[0]['blocked_source_ids'],
                         [source['observation_id']])
        sent_context=r.client.complete.call_args.args[3]
        self.assertEqual(sent_context['memory']['blocked_at_source_stances'][0][
            'source_observation_id'],source['observation_id'])

    def test_decision_after_stall_and_predecision_block_still_works(self):
        r,_=test_runtime()
        r.geometry=Mock(return_value=geometry('BLOCKED'))
        source=r.source()
        r.memory.evidence.stalled_forward(
            source,r.motion_controller.pose(),
            action_step=2,after_observation_id=source['observation_id'])
        events=[]
        r.event_callback=events.append
        proposal=r.navigation_agent.decide()
        self.assertEqual(proposal['tool'],'turn_and_observe')
        self.assertEqual(r.client.complete.call_count,1)
        history=r.client.complete.call_args.args[3]['memory']
        self.assertEqual(history['observed_front_blocks'][0][
            'source_observation_id'],source['observation_id'])
        self.assertEqual(history['stalled_forward_at_source_stances'][0][
            'source_observation_id'],source['observation_id'])
        self.assertEqual(
            [e['blocked_source_ids'] for e in events
             if e['event']=='mission_context_built'],[[]])

    def test_context_reads_pitch_from_same_pose_as_memory(self):
        # The existing SharedMissionHistory fixture is a legal minimal
        # controller exposing pose() but not a redundant pitch_deg attr.
        from types import SimpleNamespace
        from havln_agent.contexts import navigation_context
        from havln_agent.memory_records import MemoryRecords
        from test_mission_history_and_headings import cam, pose
        state=pose(heading=120)
        state['pitch_deg']=-30.
        memory=MemoryRecords(cam())
        memory.reset('1','Go down the stairs')
        memory.observations=[dict(observation_id='obs_0000',pose=dict(state))]
        memory.current=memory.observations[0]
        runtime=SimpleNamespace(
            memory=memory,
            motion_controller=SimpleNamespace(pose=lambda:dict(state)),
            camera=cam(),config={'max_steps':88},
            navigation_mode='OBSERVE',navigation_liveness=None)
        result=navigation_context(runtime)
        self.assertEqual(result['camera_controls']['current_pitch_deg'],-30.)
        self.assertEqual(result['memory']['current']['pitch_deg'],-30.)
        self.assertEqual(result['camera_controls']['allowed_look_now'],
                         ['none','up','down'])


if __name__=='__main__':
    unittest.main()
