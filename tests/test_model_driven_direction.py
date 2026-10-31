"""Regression: models choose motion direction; grounded-object heuristics do not veto."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from havln_agent.waypoint import goal_relation
from havln_agent.runtime import AgentRuntime
from havln_agent.contexts import tool_summary


class ModelDrivenDirectionTests(unittest.TestCase):
    def test_goal_relation_keeps_opposing_semantic_direction_as_evidence(self):
        result=goal_relation(
            dict(angle=math.radians(15),distance=.25),
            dict(grounding_id='g1',target='down stairs',semantic=dict(stair_direction='up'),
                 stair_direction='UP'),
            dict(bearing_deg=15.,x_right_m=-.2,z_forward_m=.9,uncertainty_m=.1),
            dict(action='descend stairs',target='stairs',landmark='stairs'),
            15.)
        self.assertEqual(result['direction_compatibility'],'CONFLICTING_EVIDENCE')
        self.assertNotIn('stair_direction_opposes_language_goal',result['rejection_reasons'])
        self.assertTrue(result['relevance_is_evidence_not_motion_permission'])

    def test_partial_grounding_reports_position_and_entry_reason(self):
        report=dict(tool='ground_landmark',status='PARTIAL',target='staircase',
            localization='LOCALIZED',entry_status='UNKNOWN',
            semantic=dict(uncertainty='entrance partly occluded'),
            position=dict(status='LOCALIZED',bearing_degrees=30,z_forward=1.1,x_right=-.2),
            entry=dict(status='UNKNOWN',reason='entry_region_not_at_standing_floor_height'))
        compact=tool_summary(report)['grounding']
        self.assertEqual(compact['position']['bearing_degrees'],30)
        self.assertEqual(compact['entry']['reason'],'entry_region_not_at_standing_floor_height')

    def runtime_with_path(self):
        r=AgentRuntime.__new__(AgentRuntime)
        r.motion_policy='LEVEL_ONLY'
        r.semantic_motion_policy='LEVEL_ONLY'
        r.active_waypoint_id=None
        r.memory=SimpleNamespace(task=SimpleNamespace(current_subgoal_index=0),
            note_candidates=Mock(),active_attempt=None)
        r.camera=SimpleNamespace(turn_angle=15,forward_step=.25)
        r.refresh=Mock()
        pose=dict(x=0,z=0,heading=0,pitch_deg=0)
        r.motion_controller=SimpleNamespace(pose=lambda:pose,target=None,
            next_step_state=Mock(return_value=dict(state='EXECUTABLE',reason=None)))
        r.geometry=Mock(return_value=object())
        r.paths={}
        return r,pose

    def test_ungrounded_approach_still_offers_executable_candidates(self):
        r,pose=self.runtime_with_path()
        r.memory.groundings=[]
        src=dict(subgoal=0,pose=pose,observation_id='obs_0000')
        values=[dict(id='wp_0',angle=0.,distance=.5,source=src,
                     first_step='EXECUTABLE',goal_association=[]),
                dict(id='wp_1',angle=math.radians(15),distance=.5,source=src,
                     first_step='EXECUTABLE',goal_association=[])]
        r.paths={v['id']:dict(v) for v in values}
        r._generate_waypoints=Mock(return_value=dict(
            status='OK',paths=values,inspected_directions=[]))
        with patch('havln_agent.runtime.local_coverage',return_value=dict(
                unexplored_directions=[])),patch(
                'havln_agent.runtime.forward_probe',return_value=dict(state='EXECUTABLE')):
            response=r._get_waypoint_candidates(target='partially localized landmark',
                                                 purpose='approach_landmark')
        self.assertEqual(response['status'],'OK')
        self.assertEqual(len(response['paths']),2)
        self.assertTrue(all(v['type']=='EXPLORATION_WAYPOINT' for v in response['paths']))
        self.assertEqual([p['id'] for p in response['paths']],['wp_0','wp_1'])

    def test_existing_goal_path_executes_without_fully_grounded_target(self):
        r,pose=self.runtime_with_path()
        r.paths={'wp_0':dict(id='wp_0',type='GOAL_WAYPOINT',
            first_step='EXECUTABLE',goal_association=[],
            source=dict(subgoal=0,pose=pose))}
        r._execute_waypoint=Mock(return_value=dict(status='TARGET_REACHED'))
        result=r._navigate_local('wp_0',max_atomic_steps=2)
        self.assertEqual(result['status'],'TARGET_REACHED')
        r._execute_waypoint.assert_called_once_with('wp_0',2)


if __name__=='__main__':
    unittest.main()
