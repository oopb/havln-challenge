"""PARTIAL visual direction can guide model choice without certifying path/target.

A broad depth box is not a localized door, and historical visual bearings
must not be carried into a changed RGB/pose/odometry epoch.
"""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from havln_agent.tools import ToolExecutor
from havln_agent.contexts import waypoint_summary


class PartialWaypointHintTests(unittest.TestCase):
    def executor(self, changed=False, confidence='medium', visible=True):
        e=ToolExecutor.__new__(ToolExecutor)
        e.camera=SimpleNamespace(turn_angle=15,hfov=90,forward_step=.25)
        e.config=dict(waypoint_distance=.5)
        e.refresh=Mock()
        e.paths={}
        source=dict(observation_id='obs_0010',rgb_hash='rgb-new',
            depth_hash='depth-new',pose=dict(x=0.,z=0.,heading=0.,motion_epoch=3))
        e.source=Mock(return_value=source)
        e.geometry=Mock(return_value=SimpleNamespace(human_status={}))
        e.estimate_goal=Mock(return_value=dict(
            requires_reconfirmation=True,reasons=['depth_spread_too_large']))
        grounding=dict(grounding_id='g1',
            target='far wall open doorway',
            source=dict(source,subgoal=1,image_scope='current'),
            semantic=dict(confidence=confidence,stair_direction='not_stairs'),
            semantic_presence='VISIBLE' if visible else 'NOT_VISIBLE',
            position=dict(status='PARTIAL',bearing_degrees=30.))
        if changed:
            grounding['source']['rgb_hash']='old-rgb'
        e.memory=SimpleNamespace(
            groundings=[grounding],tool_ledger=[],experiences=[],
            current=dict(revisit_matches=[]),
            task=SimpleNamespace(current_subgoal_index=1,subgoals=[
                dict(action='Turn left'),dict(action='Approach open doorway')]))
        candidate=dict(angle=math.radians(30),distance=.5,
                       first_step='EXECUTABLE')
        return e,candidate

    def test_current_partial_direction_annotates_but_never_certifies(self):
        e,candidate=self.executor()
        with patch('havln_agent.tools.generate_waypoints',
                   return_value=[candidate]):
            result=e._generate_waypoints('open doorway')
        self.assertEqual(result['status'],'OK')
        path=result['paths'][0]
        self.assertEqual(path['goal_association'],[])
        self.assertEqual(path['goal_status'],'GOAL_DIRECTION_UNKNOWN')
        hint=path['tentative_goal_bearings'][0]
        self.assertEqual(hint['source_observation_id'],'obs_0010')
        self.assertEqual(hint['bearing_deg'],30.)
        self.assertEqual(hint['angular_error_deg'],0.)
        self.assertFalse(hint['metric_distance_valid'])
        self.assertFalse(hint['target_identity_verified'])
        compact=waypoint_summary(path)
        self.assertEqual(compact['tentative_goal_bearings'][0]['bearing_deg'],30.)
        self.assertNotIn('goal_alignment',compact)

    def test_stale_rgb_and_weak_confidence_do_not_guide_path(self):
        for settings in (dict(changed=True),dict(confidence='low'),
                         dict(visible=False)):
            with self.subTest(settings=settings):
                e,candidate=self.executor(**settings)
                with patch('havln_agent.tools.generate_waypoints',
                           return_value=[candidate]):
                    path=e._generate_waypoints('open doorway')['paths'][0]
                self.assertFalse(path['tentative_goal_bearings'])

    def test_unrelated_bearing_not_attached_to_candidate(self):
        e,candidate=self.executor()
        candidate['angle']=math.radians(-15)
        with patch('havln_agent.tools.generate_waypoints',
                   return_value=[candidate]):
            path=e._generate_waypoints('open doorway')['paths'][0]
        self.assertFalse(path['tentative_goal_bearings'])


if __name__=='__main__':
    unittest.main()
