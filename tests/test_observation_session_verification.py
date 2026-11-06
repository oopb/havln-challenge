"""Batch 4-17: VLM completion, full RGB-D history, retained route evidence."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jsonschema import Draft7Validator

from havln_agent.active_observation import ObservationController
from havln_agent.memory_records import MemoryRecords
from havln_agent.visual_perception import (
    VisualPerception, SCAN_GUIDANCE_SCHEMA)
from havln_agent.vision_protocol import coordinate_schema, normalize_coordinates
from test_active_observation import FakeRuntime
from test_vlm_led_observation import VLMGuide, guidance


class ObservationSessionVerificationTests(unittest.TestCase):
    def test_verify_confirmed_returns_source_backed_observation_without_forward(self):
        r=FakeRuntime([])
        answer=dict(guidance('finish'),
            verification_status='CONFIRMED',
            verification_evidence='Robot is physically at the level-floor landing',
            uncertainty='No additional stair tread visible')
        r.visual_perception=VLMGuide([answer])
        receipt=ObservationController(r).scan_vlm(
            'Confirm robot is at lower landing',['stairs','landing'],
            sweep_degrees=90,mode='VERIFY')
        self.assertEqual(receipt['status'],'OK')
        self.assertEqual(receipt['outcome'],'SUBGOAL_CONFIRMED')
        self.assertEqual(receipt['verification']['status'],'CONFIRMED')
        self.assertEqual(receipt['verification']['subgoal_index'],0)
        self.assertEqual(receipt['verification']['observation_ids'],['obs_0000'])
        self.assertEqual(r.step_calls,[])
        self.assertEqual(r.visual_perception.calls[0]['history'],[])

    def test_verify_not_yet_preserves_semantic_goal(self):
        r=FakeRuntime([])
        answer=dict(guidance('finish'),
            verification_status='NOT_YET',
            verification_evidence='Several descending treads remain below robot',
            uncertainty='')
        r.visual_perception=VLMGuide([answer])
        receipt=ObservationController(r).scan_vlm(
            'Are we at stair bottom?',['stairs'],90,mode='VERIFY')
        self.assertEqual(receipt['outcome'],'SUBGOAL_NOT_YET')
        self.assertEqual(r.memory.task.current_subgoal_index,0)
        self.assertNotIn('MOVE_FORWARD',r.step_calls)

    def test_real_source_required_before_advancing_ordered_subgoals(self):
        mem=MemoryRecords(SimpleNamespace(turn_angle=15.))
        mem.task.subgoals=[dict(action='descend'),dict(action='enter room')]
        mem.observations=[dict(observation_id='obs_0028',subgoal=0)]
        mem.current=mem.observations[0]
        verdict=dict(status='CONFIRMED',subgoal_index=0,
            source='vlm_real_rgbd_history',
            source_observation_id='obs_0028',
            observation_ids=['obs_0028'],
            evidence='Robot is on the bottom flat landing',uncertainty='')
        out=mem.confirm_subgoal_observation(verdict,'Check arrival at landing')
        self.assertEqual(out['status'],'CONFIRMED')
        self.assertEqual(out['next_subgoal'],1)
        self.assertEqual(mem.task.completed_subgoals,[0])
        self.assertEqual(mem.task.current_subgoal_index,1)
        self.assertEqual(mem.task.completion_evidence[0]['verification']
                         ['supported'],['vlm_observation_confirmed'])
        self.assertEqual(mem.confirm_subgoal_observation(verdict,
            'replay a stale answer')['status'],'INVALID_EVIDENCE')

    def test_last_milestone_keeps_valid_index_until_finish(self):
        mem=MemoryRecords(SimpleNamespace(turn_angle=15.))
        mem.task.subgoals=[dict(action='reach destination')]
        mem.observations=[dict(observation_id='obs_0099',subgoal=0)]
        mem.current=mem.observations[0]
        reply=dict(status='CONFIRMED',subgoal_index=0,
            source='vlm_real_rgbd_history',
            source_observation_id='obs_0099',
            observation_ids=['obs_0099'],
            evidence='Robot is physically inside requested destination')
        result=mem.confirm_subgoal_observation(reply,'Confirm final arrival')
        self.assertEqual(result['next_subgoal'],1)
        self.assertEqual(mem.task.current_subgoal_index,0)
        self.assertEqual(mem.task.completed_subgoals,[0])

    def test_unsupported_text_or_bad_observation_does_not_confirm(self):
        mem=MemoryRecords(SimpleNamespace(turn_angle=15.))
        mem.task.subgoals=[dict(action='descend')]
        mem.observations=[dict(observation_id='obs_0028',subgoal=0)]
        mem.current=mem.observations[0]
        for verdict in (
            dict(status='CONFIRMED',subgoal_index=0,
                 source='vlm_real_rgbd_history',source_observation_id='imaginary',
                 observation_ids=['imaginary'],evidence='done'),
            dict(status='CONFIRMED',subgoal_index=0,
                 source='vlm_real_rgbd_history',source_observation_id='obs_0028',
                 observation_ids=['obs_0028'],evidence='')):
            self.assertEqual(mem.confirm_subgoal_observation(
                verdict,'Check arrival')['status'],'INVALID_EVIDENCE')
        self.assertEqual(mem.task.completed_subgoals,[])

    def test_measured_descent_is_not_erased_by_later_blockage(self):
        mem=MemoryRecords(SimpleNamespace(turn_angle=15.))
        for step,dy in [(24,-.156),(25,-.156),(26,-.158),(27,-.164)]:
            mem.actions.append(dict(step=step,action='MOVE_FORWARD',subgoal=0,
                pose=dict(heading=math.radians(225),motion_epoch=0),
                result=dict(observation_before='obs_%04d'%step,
                    observation_after='obs_%04d'%(step+1),
                    rgbd_motion=dict(status='ESTIMATED',
                        observed_frame_change=True,
                        translation_m=dict(y_up=dy)))))
        mem.tool_ledger=[dict(subgoal_id='sg_0',
            call=dict(tool='navigate_relative'),
            request_context=dict(pose=dict(heading=math.radians(225))),
            result=dict(status='BLOCKED',reason='measured_centerline_obstacle',
                        end_observation_id='obs_0044'))]
        summary=mem.route_progress_summary()
        segment=summary['successful_segments'][0]
        self.assertEqual(segment['forward_steps'],4)
        self.assertAlmostEqual(segment['observed_height_change_m_est'],-.634,3)
        self.assertEqual(segment['heading_deg'],225)
        self.assertEqual(summary['blocked_at_prior_stances'][0]
                         ['source_observation_id'],'obs_0044')

    def test_multimodal_history_is_not_lost_between_view_requests(self):
        rt=SimpleNamespace(
            camera=SimpleNamespace(normalized=False),
            memory=SimpleNamespace(task=SimpleNamespace(
                current_subgoal_index=0,
                instruction='Descend stairs',
                subgoals=[dict(action='descend',target='landing',
                               landmark='stairway',semantic_description='descend')])))
        visual=VisualPerception(rt)
        result=dict(guidance('turn_left',45),box=None)
        recorded=[]
        def answer(*args,**kwargs):
            recorded.append(len(kwargs.get('conversation_messages') or []))
            return result
        visual.complete=Mock(side_effect=answer)
        session=[]
        src=lambda n:dict(observation_id='obs_%04d'%n,
            pose=dict(heading=math.radians(n*45),pitch_deg=0))
        frame=SimpleNamespace(rgb=None,depth='fake_depth')
        with patch('havln_agent.visual_perception.image_url',
                   side_effect=['rgb_0','rgb_1']),patch(
                   'havln_agent.visual_perception.depth_image_url',
                   side_effect=['depth_0','depth_1']):
            for n in (0,1):
                visual.guide_scan_view(frame,src(n),'Find descent',['stairs'],
                    [],90,session=session,mode='APPROACH',
                    geometry_evidence=dict(center_obstruction=dict(confirmed=False)),
                    route_history=dict(successful_segments=[]))
        self.assertEqual(recorded,[0,2])
        self.assertEqual(len(session),4)
        self.assertEqual(session[0]['content'][1]['image_url']['url'],'rgb_0')
        self.assertEqual(session[0]['content'][2]['image_url']['url'],'depth_0')
        self.assertEqual(session[2]['content'][1]['image_url']['url'],'rgb_1')
        self.assertEqual(session[1]['role'],'assistant')
        self.assertEqual(session[3]['role'],'assistant')

    def test_fixed_coordinate_contract_accepts_missing_redundant_unit(self):
        schema=coordinate_schema(SCAN_GUIDANCE_SCHEMA,'image_1000_xyxy',
                                 allow_missing_unit=True)
        report=dict(guidance('finish'),box=[100,200,700,900])
        self.assertTrue(Draft7Validator(schema).is_valid(report))
        fixed=normalize_coordinates(report,SCAN_GUIDANCE_SCHEMA,
                                   default_unit='image_1000_xyxy')
        self.assertEqual(fixed['box'],[.1,.2,.7,.9])


if __name__=='__main__':
    unittest.main()
