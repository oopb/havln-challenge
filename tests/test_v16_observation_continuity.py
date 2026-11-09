"""v16 no-simulator regressions for evidence coverage and route continuity.

Run: python -m unittest discover -s tests -p test_v16_observation_continuity.py -v
"""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

from havln_agent.active_observation import ObservationController
from havln_agent.memory_records import MemoryRecords
from havln_agent.mission_history import (
    mission_evidence,recent_views_at_stance)
from havln_agent.visual_perception import VisualPerception
from test_forward_preflight_v9 import camera,test_runtime,geometry
from test_v11_observation_review import visual_answer


def pose(heading=0, generation=0, pitch=0):
    return dict(heading=math.radians(heading),pitch_deg=pitch,
        x=None,y=None,z=None,motion_epoch=0,
        viewpoint_generation=generation,stance_id='0:'+str(generation),
        position_coordinates_valid=False)


def observed(identifier, heading=0, generation=0, pitch=0,
             rgb=None,depth=None):
    return dict(observation_id=identifier,
        pose=pose(heading,generation,pitch),
        rgb_hash=rgb or 'rgb_'+identifier,
        depth_hash=depth or 'depth_'+identifier)


class CompleteStanceCoverage(unittest.TestCase):
    def test_full_ring_survives_last_80_observations_and_detail_cap(self):
        memory=MemoryRecords(camera())
        memory.reset('ep','Find a passage')
        first=[observed('ring_%02d'%i,i*15)
               for i in range(24)]
        far=[observed('else_%03d'%i,0,generation=i+1)
             for i in range(90)]
        memory.observations=first+far+[observed('now',0)]
        memory.current=memory.observations[-1]
        coverage=recent_views_at_stance(memory,pose(0))
        self.assertEqual(coverage['distinct_heading_pitch_views'],24)
        self.assertEqual(len(coverage['latest_distinct_views']),16)
        self.assertTrue(coverage['older_views_may_be_omitted'])
        self.assertTrue(coverage['full_coverage_includes_old_views'])
        self.assertEqual(coverage['full_coverage_by_pitch'][0][
            'observed_heading_bins_deg'],list(range(0,360,15)))
        self.assertEqual(coverage['sampled_observations'],25)
        self.assertFalse(coverage['historical_depth_does_not_authorize_move'] is False)

    def test_forward_stance_is_never_same_coverage(self):
        memory=MemoryRecords(camera())
        memory.observations=[
            observed('initial',90,0),
            observed('turned',120,0),
            observed('moved',90,1)]
        coverage=recent_views_at_stance(memory,pose(90,1))
        self.assertEqual(coverage['distinct_heading_pitch_views'],1)
        self.assertEqual(coverage['full_coverage_by_pitch'][0][
            'observed_heading_bins_deg'],[90])


class SourceBoundCandidates(unittest.TestCase):
    def test_unboxed_route_survives_without_fabricated_bearing(self):
        memory=MemoryRecords(camera())
        memory.reset('ep','Follow the arch')
        memory.observations=[observed('obs_0001',90)]
        memory.current=memory.observations[0]
        proposal=dict(source_observation_id='obs_0001',
            target_type='archway',target_kind='APPROACH',confidence='medium',
            objective='Find a passage',evidence='Possible arch to the left',
            target_visible=True,absolute_heading_deg=None)
        candidate=memory.note_visual_route_candidate(proposal)
        self.assertIsNotNone(candidate)
        self.assertIsNone(candidate['absolute_heading_deg'])
        self.assertFalse(candidate['angular_bearing_grounded'])
        self.assertEqual(memory.note_visual_route_candidate(proposal),candidate)
        route=mission_evidence(memory,pose(90))['candidate_route_hypotheses'][0]
        self.assertTrue(route['source_stance_matches_current'])
        self.assertIsNone(route['relative_bearing_deg_from_current'])
        historical=mission_evidence(memory,pose(90,1))[
            'candidate_route_hypotheses'][0]
        self.assertFalse(historical['source_stance_matches_current'])
        self.assertIsNone(historical['relative_bearing_deg_from_current'])

    def test_boxed_visual_candidate_rebases_only_same_stance(self):
        memory=MemoryRecords(camera())
        memory.reset('ep','Find the hall')
        memory.observations=[observed('obs_0001',90)]
        memory.current=memory.observations[0]
        memory.note_visual_route_candidate(dict(
            source_observation_id='obs_0001',target_type='hall',
            evidence='Hallway on the left',objective='Find the hall',
            absolute_heading_deg=120,target_visible=True))
        turning=mission_evidence(memory,pose(105))[
            'candidate_route_hypotheses'][0]
        self.assertEqual(turning['relative_bearing_deg_from_current'],15)
        moved=mission_evidence(memory,pose(105,1))[
            'candidate_route_hypotheses'][0]
        self.assertIsNone(moved['relative_bearing_deg_from_current'])

    def test_last_successful_forward_is_a_receipt_not_forced_motion(self):
        memory=MemoryRecords(camera())
        memory.reset('ep','Find the hall')
        memory.observations=[observed('start',0)]
        memory.current=observed('after',0,1)
        memory.observations.append(memory.current)
        memory.actions=[dict(step=0,action='MOVE_FORWARD',
            pose=pose(0,1),result=dict(visual_change=True,
                observation_before='start',observation_after='after'))]
        decision=dict(tool='navigate_relative',args=dict(
            turn_direction='NONE',turn_degrees=0,distance_m=.25))
        memory.note_navigation_turn(decision,dict(status='ACTIONS_COMPLETED',
            executed_actions=1,executed_motion=['MOVE_FORWARD'],
            end_observation_id='after'),'start',0,'nav_0')
        report=mission_evidence(memory,pose(0,1))['last_motion_decision']
        self.assertEqual(report['executed_forward_actions'],1)
        self.assertTrue(report['last_motion_was_uninterrupted'])
        self.assertTrue(report['continuation_is_model_choice'])
        self.assertTrue(report['motion_is_evidence_not_command'])


class CrossSessionObservation(unittest.TestCase):
    def test_same_question_rgbd_and_stance_reuses_visual_not_turn(self):
        runtime,actions=test_runtime()
        runtime.geometry=Mock(return_value=geometry('EXECUTABLE'))
        runtime.visual_perception=SimpleNamespace(
            guide_scan_view=Mock(return_value=visual_answer('finish',0)))
        controller=ObservationController(runtime)
        first=controller.scan_mission('Locate a passage',sweep_degrees=90)
        second=controller.scan_mission('Locate a passage',sweep_degrees=90)
        self.assertEqual(first['status'],'OK')
        self.assertEqual(second['status'],'OK')
        self.assertEqual(runtime.visual_perception.guide_scan_view.call_count,1)
        self.assertFalse(actions)
        source=runtime.memory.observation_searches[-1]['checkpoints'][0]
        self.assertEqual(source['reused_visual_from_observation_id'],
                         first['evidence_observation_ids'][0])
        controller.scan_mission('Locate a different passage',sweep_degrees=90)
        self.assertEqual(runtime.visual_perception.guide_scan_view.call_count,2)

    def test_unboxed_candidate_is_recorded_after_observation(self):
        runtime,_=test_runtime()
        runtime.geometry=Mock(return_value=geometry('EXECUTABLE'))
        answer=visual_answer()
        answer.update(candidate_approach=True,target_visible=True,
            target_type='archway',target_kind='APPROACH',
            observation_summary='A possible archway is visible',box=None)
        runtime.visual_perception=SimpleNamespace(
            guide_scan_view=Mock(return_value=answer))
        result=ObservationController(runtime).scan_mission(
            'Find an opening',sweep_degrees=90)
        self.assertEqual(result['status'],'OK')
        self.assertEqual(result['waypoints'],[])
        self.assertEqual(len(runtime.memory.route_candidates),1)
        self.assertFalse(runtime.memory.route_candidates[0][
            'angular_bearing_grounded'])


class TargetedObstacleObservation(unittest.TestCase):
    def test_depth_roi_is_passed_in_existing_single_vlm_call(self):
        runtime,_=test_runtime()
        runtime.geometry=Mock(return_value=geometry('BLOCKED'))
        perception=VisualPerception(runtime)
        perception.complete=Mock(return_value=visual_answer())
        with patch('havln_agent.visual_perception.image_url',
                   return_value='rgb'),patch(
                   'havln_agent.visual_perception.depth_image_url',
                   return_value='depth'):
            perception.guide_scan_view(runtime.observation,runtime.source(),
                'Inspect the near obstruction',['passage'],[],90,
                session=[],view_coverage={})
        self.assertEqual(perception.complete.call_count,1)
        request=perception.complete.call_args.args[3]
        self.assertEqual(request['obstacle_focus'][
            'depth_region_xyxy_norm'],[.42,.34,.61,.78])
        self.assertEqual(request['obstacle_focus'][
            'source_observation_id'],runtime.source()['observation_id'])
        self.assertTrue(request['obstacle_focus'][
            'distant_route_walkability_unverified'])


if __name__=='__main__':
    unittest.main()
