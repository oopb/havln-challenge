"""Source identity, bearing protocol and canonical mission history regressions.

Python-only (no Habitat runtime, API key, or simulation files required).
"""
import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from jsonschema import Draft7Validator
from havln_agent.heading import (
    signed_turn,relative_bearing,box_center_bearing,
    same_local_stance,direction_from_bearing)
from havln_agent.contracts import registry,navigation_schema
from havln_agent.memory_records import MemoryRecords
from havln_agent.mission_history import mission_evidence,selected_history_frames
from havln_agent.visual_perception import session_frame_ids
from havln_agent.runtime import AgentRuntime


def cam():
    return SimpleNamespace(
        turn_angle=15,tilt_angle=30,forward_step=.25,hfov=90)

def pose(heading=90,x=0,epoch=0):
    return dict(heading=math.radians(heading),pitch_deg=0.,
        x=float(x),z=0.,y=0.,motion_epoch=epoch,
        viewpoint_generation=0,position_coordinates_valid=True)


class DirectionProtocol(unittest.TestCase):
    def test_direction_is_explicit_not_sign_based(self):
        self.assertEqual(signed_turn('LEFT',30),30)
        self.assertEqual(signed_turn('RIGHT',30),-30)
        self.assertEqual(signed_turn('NONE',0),0)
        for direction,degrees in [('NONE',30),('LEFT',0),
                                  ('RIGHT',-30),('FRONT',45)]:
            with self.subTest(direction=direction,degrees=degrees):
                with self.assertRaises(ValueError):
                    signed_turn(direction,degrees)

    def test_model_schema_rejects_signed_old_motion(self):
        reg=registry(cam(),3)
        schema=navigation_schema(reg)
        Draft7Validator.check_schema(schema)
        self.assertTrue(Draft7Validator(schema).is_valid(dict(
            tool='navigate_relative',args=dict(
                turn_direction='RIGHT',turn_degrees=30,distance_m=.5))))
        self.assertFalse(Draft7Validator(schema).is_valid(dict(
            tool='navigate_relative',args=dict(turn_deg=-30,distance_m=.5))))
        self.assertTrue(Draft7Validator(reg['turn_and_observe']['parameters']).is_valid(
            dict(turn_direction='LEFT',turn_degrees=45,look='none')))
        self.assertNotIn('relative_turn_deg',
            reg['turn_and_observe']['parameters']['properties'])

    def test_executor_right_and_left_generate_correct_internal_yaw(self):
        r=AgentRuntime.__new__(AgentRuntime)
        r.camera=cam()
        r.memory=SimpleNamespace(tool_ledger=[],note_candidates=Mock())
        r.paths={}
        r.refresh=Mock()
        r.source=Mock(return_value=dict(observation_id='obs_0001',
            pose=pose(heading=90)))
        r._navigate_local=Mock(return_value=dict(status='ACTIONS_COMPLETED'))
        AgentRuntime._navigate_relative(r,'RIGHT',30,.5)
        self.assertEqual(r.paths['direct_0000']['bearing_deg'],-30)
        self.assertAlmostEqual(
            r.paths['direct_0000']['endpoint_xz_m'][0],
            -math.sin(math.radians(60))*.5)
        r.paths.clear()
        AgentRuntime._navigate_relative(r,'LEFT',45,.5)
        self.assertEqual(r.paths['direct_0000']['bearing_deg'],45)

    def test_camera_command_right_means_actual_turn_right(self):
        r=AgentRuntime.__new__(AgentRuntime)
        r.camera=cam()
        r.motion_controller=SimpleNamespace(pitch_deg=0.)
        r.memory=SimpleNamespace(current=dict(observation_id='obs_0001'))
        r._step=Mock(return_value=dict(status='EXECUTED'))
        result=AgentRuntime._turn_and_observe(
            r,'RIGHT',30,'none')
        self.assertEqual(result['actual_turn_direction'],'RIGHT')
        self.assertEqual([c.args[0] for c in r._step.call_args_list],
            ['TURN_RIGHT','TURN_RIGHT'])

    def test_bbox_center_bearing_uses_calibrated_hfov(self):
        right=box_center_bearing([.75,.2,.95,.6],90)
        left=box_center_bearing([.05,.2,.25,.6],90)
        self.assertLess(right,0)
        self.assertGreater(left,0)
        self.assertEqual(direction_from_bearing(right)[0],'RIGHT')
        self.assertEqual(direction_from_bearing(left)[0],'LEFT')
        self.assertEqual(box_center_bearing([.4,.2,.6,.7],90),0)
        self.assertAlmostEqual(relative_bearing(330,30),-60)


class SharedMissionHistory(unittest.TestCase):
    def make_memory(self):
        m=MemoryRecords(cam())
        m.reset('1','Descend stairs, find right hall, stop beside piano.')
        p=pose()
        m.observations=[
            dict(observation_id='obs_0001',pose=dict(p),place_id='P1'),
            dict(observation_id='obs_0002',pose=pose(heading=120),place_id='P1'),
            dict(observation_id='obs_0003',pose=pose(heading=60),place_id='P1')]
        m.current=m.observations[-1]
        m.observation_searches=[dict(mode='mission_vlm',
            objective='Find route',outcome='OBSERVATION_COMPLETED',
            start_observation_id='obs_0001',
            end_observation_id='obs_0003',
            checkpoints=[
                dict(observation_id='obs_0001',heading_deg=90,
                    pitch_deg=0,visual=dict(
                        candidate_approach=True,
                        observation_summary='Door opening to right.')),
                dict(observation_id='obs_0002',heading_deg=120,
                    pitch_deg=0,visual=dict(
                        candidate_approach=False,
                        observation_summary='Stone wall at front.'))],
            waypoints=[dict(source_observation_id='obs_0001',
                absolute_heading_deg=55,target_type='door',
                relative_bearing_deg_from_current=0,
                bearing_source='RGB_box',traversability='UNCONFIRMED')])]
        block=dict(observation_id='obs_0003',heading_deg=60,
            pitch_deg=0,motion_epoch=0,viewpoint_generation=0,
            central_support_pixels=45,nearest_along_m=.2,
            reason='measured_centerline_obstacle')
        m.evidence.obstacle_blocks=[dict(block),dict(block)]
        return m

    def test_source_ids_in_session_are_canonical_and_not_substrings(self):
        session=[dict(role='user',content=[dict(type='text',text=json.dumps(
            dict(source_observation_id='obs_0001',
                 historical_images=[dict(observation_id='obs_0002')])))]),
            dict(role='assistant',content='obs_9999')]
        current,historical=session_frame_ids(session)
        self.assertEqual(current,{'obs_0001'})
        self.assertEqual(historical,{'obs_0002'})
        session.append(dict(role='user',content=[dict(type='text',
            text=json.dumps(dict(source_observation_id='obs_0003',
                historical_keyframes=[dict(observation_id='obs_0001')])))]))
        current,historical=session_frame_ids(session)
        self.assertEqual(current,{'obs_0001','obs_0003'})
        self.assertEqual(historical,{'obs_0001','obs_0002'})

    def test_history_rebases_waypoint_and_counts_actual_blocks(self):
        m=self.make_memory()
        snapshot=mission_evidence(m,pose(heading=60))
        self.assertEqual(snapshot['contract'],'mission_evidence_v1')
        self.assertEqual(snapshot['progress']['model_revision_count'],0)
        wp=snapshot['observation_tasks'][0]['waypoints'][0]
        self.assertTrue(wp['source_stance_matches_current'])
        self.assertAlmostEqual(wp['relative_bearing_deg_from_current'],-5)
        self.assertEqual(snapshot['blocked_at_source_stances'][0][
            'repeated_same_frame_heading_attempts'],2)
        self.assertEqual(snapshot['blocked_at_source_stances'][0][
            'center_support_pixels'],45)
        self.assertFalse(snapshot['historical_clearance_authorizes_move'])
        frames=selected_history_frames(m,'obs_0003')
        self.assertEqual({v['observation_id'] for v in frames},
                         {'obs_0001','obs_0002'})

    def test_old_stance_removes_executable_relative_bearing(self):
        m=self.make_memory()
        snapshot=mission_evidence(m,pose(heading=60,x=1.0))
        wp=snapshot['observation_tasks'][0]['waypoints'][0]
        self.assertFalse(wp['source_stance_matches_current'])
        self.assertIsNone(wp['relative_bearing_deg_from_current'])
        self.assertFalse(same_local_stance(pose(epoch=0),pose(epoch=1)))

    def test_model_progress_audit_no_repeat_and_verified_ids_only(self):
        m=self.make_memory()
        update=dict(progress_narrative='I may have reached the landing.',
            navigation_intent='Look for hall.',
            uncertainties=['Not certain about final stair.'],
            evidence_observation_ids=['obs_0001','invented'])
        m.note_navigation_assessment(update)
        m.note_navigation_assessment(update)
        self.assertEqual(len(m.task.progress_history),1)
        self.assertEqual(m.task.evidence_observation_ids,['obs_0001'])
        entry=m.task.progress_history[0]
        self.assertEqual(entry['context_observation_id'],'obs_0003')
        self.assertFalse(entry['completion_verified'])
        self.assertEqual(m.note_observation_evidence(
            ['obs_0002','phantom']),['obs_0002'])
        self.assertEqual(m.task.evidence_observation_ids,
                         ['obs_0001','obs_0002'])


if __name__=='__main__':
    unittest.main()
