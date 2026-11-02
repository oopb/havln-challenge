"""Model-led verification tests: physical receipts remain authoritative."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from jsonschema import Draft7Validator

from havln_agent.verification_agent import VerificationAgent, VERIFICATION_SCHEMA
from havln_agent.tools import ToolExecutor


def obs(i, epoch=0):
    return dict(observation_id='o%d' % i, step=i, subgoal=0,
        rgb_hash='rgb%d' % i, depth_hash='depth%d' % i,
        pose=dict(motion_epoch=epoch, heading=0., x=float(i)*.25,z=0.),
        captured_at=float(i), last_read_at=float(i))


def action(i, state='ESTIMATED', epoch=0):
    return dict(step=i,subgoal=0,action='MOVE_FORWARD',
        pose=dict(motion_epoch=epoch,heading=0.),
        result=dict(observation_before='o%d'%i,observation_after='o%d'%(i+1),
            visual_change=True,
            rgbd_motion=dict(status=state,
                vertical_motion='DOWN' if state=='ESTIMATED' else 'UNKNOWN',
                translation_m=dict(x_right=0.,y_up=-.2,z_forward=.25),
                uncertainty_m=.03,
                observation_pair=dict(
                    before_rgb_hash='rgb%d'%i,before_depth_hash='depth%d'%i,
                    after_rgb_hash='rgb%d'%(i+1),
                    after_depth_hash='depth%d'%(i+1)))))


def reply(operation='verdict',verdict='COMPLETED',steps=None,ids=None,terminal='o1',
          proof_basis='ACTION_TRANSITION'):
    if operation=='inspect_history':
        return dict(operation=operation,start_step=0)
    if operation=='inspect_frames':
        return dict(operation=operation,observation_ids=['o0'],
                    question='What changed?')
    if operation=='inspect_records':
        return dict(operation=operation,record_ids=['ground_0000'])
    return dict(operation='verdict',verdict=verdict,proof_basis=proof_basis,
        supported=['physical_and_visual_milestone_reached'],missing=[],
        contradictions=[],cited_observation_ids=ids if ids is not None else ['o0','o1'],
        cited_action_steps=steps if steps is not None else [0],
        terminal_observation_id=terminal,
        reasoning='Measured evidence and public views support the milestone.',
        next_evidence_action='')


class FakeClient:
    def __init__(self, replies):
        self.queue=copy.deepcopy(replies)
        self.calls=[]
    def complete(self,*args,**kwargs):
        self.calls.append((args,kwargs))
        return dict(message=self.queue.pop(0))


def fake_runtime(replies,actions=None,observations=None):
    actions=list(actions if actions is not None else [action(0)])
    observations=list(observations if observations is not None else [obs(0),obs(1)])
    task=SimpleNamespace(instruction='Walk through the doorway.',
        subgoals=[dict(action='Go through the doorway',landmark='doorway',
                       target='other side')],
        final_target='other side',constraints=[],current_subgoal_index=0,
        subgoal_started=0,subgoal_start_steps={'0':0},
        completed_subgoals=[],completion_evidence=[],progress_claims=[],
        provisional_transitions=[])
    mem=SimpleNamespace(task=task,actions=actions,observations=observations,
        groundings=[],scene_history=[dict(source=dict(observation_id='o1',
            rgb_hash='rgb1',depth_hash='depth1',pose=observations[-1]['pose']),
            scene=dict(description='Door opening is now behind the robot'))],
        tool_ledger=[],current=observations[-1],
        camera=SimpleNamespace(forward_step=.25),progress_feedback=None,
        progress_stage='IN_PROGRESS',event=Mock())
    rt=SimpleNamespace(memory=mem,client=FakeClient(replies))
    rt._vision_frame=lambda ident:(object(),dict(observation_id=ident,
                                                   image_scope='historical_keyframe'))
    rt.visual_perception=SimpleNamespace(scene=lambda image,source,question:
        dict(description='Door opening visible',landmarks=[]))
    return rt


class VerificationAgentTests(unittest.TestCase):
    def test_model_can_certify_non_template_goal_with_authentic_receipt(self):
        rt=fake_runtime([reply()])
        result=VerificationAgent(rt).verify('subgoal_completed','I crossed',
                                            ['o0','o1'])
        self.assertEqual(result['status'],'VERIFIED')
        self.assertEqual(rt.memory.task.completed_subgoals,[0])
        self.assertEqual(rt.memory.task.completion_evidence[0]['executed_steps'],[0])
        self.assertEqual(result['evidence_details']['terminal_observation_id'],'o1')

    def test_compact_disjoint_verifier_schema_accepts_only_relevant_fields(self):
        for operation in ('inspect_history','inspect_frames','inspect_records','verdict'):
            self.assertFalse(list(Draft7Validator(VERIFICATION_SCHEMA).iter_errors(
                dict(message=reply(operation=operation)))))
        invalid=dict(message=dict(reply(),extraneous_legacy_predicates=['target_nearby']))
        self.assertTrue(list(Draft7Validator(VERIFICATION_SCHEMA).iter_errors(invalid)))
        self.assertEqual(VERIFICATION_SCHEMA['type'],'object')

    def test_visual_state_can_be_certified_without_moving(self):
        answer=reply(steps=[],ids=['o0','o1'],proof_basis='VISUAL_STATE')
        rt=fake_runtime([answer],actions=[])
        rt.memory.scene_history=[
            dict(source=dict(observation_id='o0',rgb_hash='rgb0',depth_hash='depth0',
                             pose=rt.memory.observations[0]['pose']),
                 scene=dict(description='person present')),
            dict(source=dict(observation_id='o1',rgb_hash='rgb1',depth_hash='depth1',
                             pose=rt.memory.observations[1]['pose']),
                 scene=dict(description='person no longer present'))]
        result=VerificationAgent(rt).verify('subgoal_completed',
            'The watched state visibly changed',['o0','o1'])
        self.assertEqual(result['status'],'VERIFIED')

    def test_single_frame_visual_state_is_insufficient(self):
        answer=reply(steps=[],ids=['o1'],proof_basis='VISUAL_STATE')
        rt=fake_runtime([answer],actions=[])
        result=VerificationAgent(rt).verify('subgoal_completed','seen',['o1'])
        self.assertIn('visual_state_requires_two_reviewed_distinct_times',result['missing'])

    def test_visual_state_requires_authentic_source_pose(self):
        answer=reply(steps=[],ids=['o0','o1'],proof_basis='VISUAL_STATE')
        rt=fake_runtime([answer],actions=[])
        bad=dict(rt.memory.scene_history[0])
        bad['source']=dict(bad['source'],pose=dict(motion_epoch=99))
        rt.memory.scene_history=[bad]
        result=VerificationAgent(rt).verify('subgoal_completed','seen',['o0'])
        self.assertIn('visual_state_requires_two_reviewed_distinct_times',result['missing'])

    def test_motion_claim_cannot_succeed_without_a_prior_context_frame(self):
        answer=reply(ids=['o1'])
        rt=fake_runtime([answer])
        result=VerificationAgent(rt).verify('subgoal_completed','crossed',['o1'])
        self.assertIn('action_transition_missing_before_view',result['missing'])

    def test_unknown_forward_after_turn_invalidates_claimed_terminal(self):
        turn=dict(step=0,subgoal=0,action='TURN_RIGHT',
            pose=dict(motion_epoch=0,heading=0.),
            result=dict(visual_change=True))
        rt=fake_runtime([reply(steps=[0],ids=['o0','o2'],terminal='o2')],
            actions=[turn,action(1,'UNKNOWN')],
            observations=[obs(i) for i in range(3)])
        result=VerificationAgent(rt).verify('subgoal_completed','entered',['o0'])
        self.assertIn('unknown_motion_between_cited_steps',result['missing'])

    def test_unknown_translation_cannot_be_authenticated_by_model(self):
        rt=fake_runtime([reply()],actions=[action(0,'UNKNOWN')])
        result=VerificationAgent(rt).verify('subgoal_completed','I crossed',['o0'])
        self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE')
        self.assertIn('unregistered_motion_step:0',result['missing'])
        self.assertEqual(rt.memory.task.completed_subgoals,[])

    def test_fabricated_image_citation_is_rejected(self):
        rt=fake_runtime([reply(ids=['o0','invented'],terminal='invented')])
        result=VerificationAgent(rt).verify('subgoal_completed','I crossed',['o0'])
        self.assertIn('unknown_cited_observation_id:invented',result['missing'])
        self.assertFalse(result['completion_confirmed'])

    def test_completion_requires_semantic_review_of_terminal_frame(self):
        rt=fake_runtime([reply()])
        rt.memory.scene_history=[]
        result=VerificationAgent(rt).verify('subgoal_completed','I crossed',['o0'])
        self.assertIn('terminal_visual_semantics_not_reviewed',result['missing'])

    def test_historical_lookup_returns_exact_indexed_action_receipts(self):
        read=reply(operation='inspect_history')
        rt=fake_runtime([read,reply()])
        result=VerificationAgent(rt).verify('subgoal_completed','I crossed',['o0'])
        self.assertEqual(result['status'],'VERIFIED')
        inspection=rt.client.calls[-1][0][3]['inspections'][-1]
        self.assertEqual(inspection['operation'],'inspect_history')
        self.assertEqual(inspection['actions'][0]['motion_status'],'ESTIMATED')
        self.assertEqual(inspection['actions'][0]['observation_after'],'o1')

    def test_saved_rgb_query_is_allowed_without_motion(self):
        read=reply(operation='inspect_frames')
        read['observation_ids']=['o0']
        read['question']='Was the doorway entered?'
        rt=fake_runtime([read,reply()])
        result=VerificationAgent(rt).verify('subgoal_completed','I crossed',['o0'])
        self.assertEqual(result['status'],'VERIFIED')
        self.assertEqual(rt.client.calls[-1][0][3]['inspections'][-1]
                         ['views'][0]['scene']['description'],'Door opening visible')

    def test_cross_epoch_terminal_frame_is_not_valid_metric_endpoint(self):
        rt=fake_runtime([reply()],observations=[obs(0),obs(1,epoch=1)])
        result=VerificationAgent(rt).verify('subgoal_completed','I crossed',['o0'])
        self.assertIn('move_observation_pair_crosses_motion_epoch:0',result['missing'])

    def test_unknown_interior_forward_breaks_composed_displacement(self):
        rt=fake_runtime([reply(steps=[0,2],ids=['o0','o3'],terminal='o3')],
            actions=[action(0),action(1,'UNKNOWN'),action(2)],
            observations=[obs(i) for i in range(4)])
        result=VerificationAgent(rt).verify('subgoal_completed','I crossed',['o0'])
        self.assertIn('unknown_motion_between_cited_steps',result['missing'])

    def test_unknown_from_provisional_other_subgoal_breaks_metric_proof(self):
        # A provisional navigation switch cannot hide an UNKNOWN translation
        # from the physically continuous interval.
        uncertain=action(1,'UNKNOWN')
        uncertain['subgoal']=1
        rt=fake_runtime([reply(steps=[0,2],ids=['o0','o3'],terminal='o3')],
            actions=[action(0),uncertain,action(2)],
            observations=[obs(i) for i in range(4)])
        result=VerificationAgent(rt).verify('subgoal_completed','I crossed',['o0'])
        self.assertIn('unknown_motion_between_cited_steps',result['missing'])

    def test_claimed_completion_requires_terminal_frame(self):
        rt=fake_runtime([reply(terminal='o0')])
        result=VerificationAgent(rt).verify('subgoal_completed','I crossed',['o0'])
        self.assertIn('terminal_observation_precedes_claimed_actions',result['missing'])

    def test_incomplete_verdict_only_records_missing_evidence(self):
        incomplete=reply(verdict='IN_PROGRESS',steps=[],terminal='none')
        incomplete['missing']=['Need a view after the doorway threshold']
        rt=fake_runtime([incomplete])
        result=VerificationAgent(rt).verify('subgoal_completed','not sure',['o0'])
        self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE')
        self.assertEqual(rt.memory.task.completed_subgoals,[])
        self.assertIn('Need a view after the doorway threshold',result['missing'])

    def test_query_limit_never_autoverifies(self):
        rt=fake_runtime([reply(operation='inspect_history') for _ in range(7)])
        result=VerificationAgent(rt).verify('subgoal_completed','I crossed',['o0'])
        self.assertEqual(result['reason'],'verification_inspection_limit')
        self.assertFalse(result['completion_confirmed'])



class FinalFinishEvidenceTests(unittest.TestCase):
    @staticmethod
    def executor(certificates=None,verified=None,view='o1'):
        certificates=certificates if certificates is not None else [dict(
            subgoal=0,verification=dict(completion_confirmed=True,
                evidence_details=dict(verifier='model_led_sensor_guarded',
                    terminal_observation_id='o1')))]
        executor=ToolExecutor.__new__(ToolExecutor)
        executor.memory=SimpleNamespace(
            task=SimpleNamespace(subgoals=[dict(action='arrive at destination')],
                completed_subgoals=verified if verified is not None else [0],
                completion_evidence=certificates),
            current=dict(observation_id=view))
        executor.refresh=Mock()
        executor.terminate=Mock()
        return executor

    def test_final_semantic_proof_does_not_require_second_vlm_or_stair_rule(self):
        executor=self.executor()
        result=executor._finish_navigation('success','All goals certified')
        self.assertEqual(result['status'],'FINISHED')
        executor.refresh.assert_called_once()
        executor.terminate.assert_called_once_with('goal_completed',model_requested=True)

    def test_same_goal_previous_frame_cannot_authorize_stop(self):
        executor=self.executor(view='o2')
        result=executor._finish_navigation('success','done')
        self.assertEqual(result['reason'],'final_certified_view_changed_reverify')
        executor.terminate.assert_not_called()

    def test_missing_model_certificate_cannot_be_replaced_by_legacy_evidence(self):
        executor=self.executor(certificates=[dict(subgoal=0,verification=dict(
            completion_confirmed=True,supported=['multi_frame_level_landing_transition']))])
        result=executor._finish_navigation('success','done')
        self.assertEqual(result['reason'],'final_model_semantic_arrival_not_verified')
        executor.refresh.assert_not_called()
        executor.terminate.assert_not_called()

    def test_all_milestones_must_be_verified_even_if_final_is_certified(self):
        executor=self.executor(verified=[])
        result=executor._finish_navigation('success','done')
        self.assertEqual(result['reason'],'ordered_subgoals_not_verified')
        executor.terminate.assert_not_called()

    def test_model_abort_remains_available_without_completion(self):
        executor=self.executor(verified=[])
        result=executor._finish_navigation('aborted','blocked')
        self.assertEqual(result['status'],'FINISHED')
        executor.terminate.assert_called_once_with('model_aborted',model_requested=True)

    def test_final_certificate_selected_by_milestone_not_ledger_order(self):
        final=dict(subgoal=1,verification=dict(completion_confirmed=True,
            evidence_details=dict(verifier='model_led_sensor_guarded',
                                  terminal_observation_id='o1')))
        older=dict(subgoal=0,verification=dict(completion_confirmed=True))
        executor=self.executor(certificates=[final,older],verified=[1,0])
        executor.memory.task.subgoals=[dict(action='approach'),dict(action='enter')]
        result=executor._finish_navigation('success','done')
        self.assertEqual(result['status'],'FINISHED')
        executor.terminate.assert_called_once_with('goal_completed',model_requested=True)


if __name__=='__main__':
    unittest.main()
