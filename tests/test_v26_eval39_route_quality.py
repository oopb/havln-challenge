"""Eval-39: retain route before/after RGB, distinguish endpoint wall from
six already executed forward steps, and keep relative-only model commands.

Offline regressions mirror the source IDs/angles from eval-batch4-39.rar.
No privileged pose, task-stage checker, forced MOVE or Habitat dependency.
"""
import copy
import json
import unittest

from havln_agent.contexts import compact_navigation_context
from havln_agent.memory_records import MemoryRecords
from havln_agent.mission_history import mission_evidence
from havln_agent.navigation_history import navigation_visual_anchors
from havln_agent.navigation_evidence import route_continuity
from havln_agent.navigation_agent import MISSION_PROMPT
from havln_agent.contracts import registry
from test_forward_preflight_v9 import camera
from test_v23_route_evidence import pose, frame


def eval39_memory():
    m=MemoryRecords(camera())
    m.reset('eval39-trajectory-repro',
            'Descend stairs, go through the lower foyer, find the piano.')
    original=[frame(19,255,-60,0)]
    original += [frame(19+i,255,-60,i) for i in range(1,7)]
    original += [frame(i,90,-60,6) for i in (44,48,78,84)]
    original[-1]['revisit_matches']=[
        dict(observation_id='obs_0044'),
        dict(observation_id='obs_0048'),
        dict(observation_id='obs_0078')]
    m.observations=original
    m.current=original[-1]
    m.task.progress_narrative='Still at stair top. Stairs not found.'
    for i in range(6):
        m.actions.append(dict(step=17+i,action='MOVE_FORWARD',
            pose=pose(255,-60,i+1),result=dict(
                visual_change=True,
                observation_before='obs_%04d'%(19+i),
                observation_after='obs_%04d'%(20+i),
                forward_depth_preflight=dict(status='CLEAR'))))
    # A read-only *later* obstruction at the six-step endpoint. No MOVE
    # failed. In the real trace this obstacle appeared AFTER the six moves.
    m.evidence.predecision_front_obstacles.append(dict(
        observation_id='obs_0025',heading_deg=255.0,
        motion_epoch=0,viewpoint_generation=6,stage='navigation_decision',
        seen_count=1,blocker=dict(nearest_along_m=.35)))
    return m


class Eval39RouteContext(unittest.TestCase):
    def test_endpoint_obstacle_is_not_past_segment_failure(self):
        m=eval39_memory()
        segments=m.route_progress_summary(
            max_successful_segments=None)['successful_segments']
        self.assertEqual(segments[0]['forward_steps'],6)
        result=route_continuity(m,m.current['pose'],segments)
        self.assertEqual(result['selected_historical_forward'][
            'start_observation_id'],'obs_0019')
        self.assertEqual(result['selected_historical_forward'][
            'end_observation_id'],'obs_0025')
        self.assertEqual(result['separation_from_historical_forward_deg'],165)
        endpoint=result['endpoint_front_obstacle']
        self.assertEqual(endpoint['kind'],
                         'FRONT_BLOCK_OBSERVED_NO_FORWARD_ATTEMPT')
        self.assertEqual(endpoint['source_observation_id'],'obs_0025')
        self.assertFalse(endpoint['attempted_subsequent_forward'])
        self.assertTrue(endpoint['previously_executed_forwards_not_rejected'])
        self.assertTrue(endpoint['measured_block_does_not_prove_earlier_route_failed'])
        self.assertTrue(result['executed_steps_do_not_prove_stair_descent_or_metric_displacement'])

    def test_old_stance_block_never_falsely_matches_endpoint(self):
        m=eval39_memory()
        m.evidence.predecision_front_obstacles[0]['observation_id']='obs_0019'
        seg=m.route_progress_summary(
            max_successful_segments=None)['successful_segments']
        result=route_continuity(m,m.current['pose'],seg)
        self.assertIsNone(result['endpoint_front_obstacle'])

    def test_route_images_come_before_duplicate_revisits(self):
        m=eval39_memory()
        selected=navigation_visual_anchors(m,'obs_0084',max_pairs=5)
        ids=[s['observation_id'] for s in selected]
        self.assertEqual(ids[:2],['obs_0019','obs_0025'])
        self.assertEqual(len(ids),len(set(ids)))
        # These three old images share yaw/pitch and stance with CURRENT
        # obs_0084. Never pay a history slot for the same current view.
        for repeated in ('obs_0044','obs_0048','obs_0078'):
            self.assertNotIn(repeated,ids)
        self.assertLessEqual(len(ids),5)
        self.assertEqual(selected[0]['selection_reason'],
                         'prior_forward_start_visual_evidence')
        self.assertEqual(selected[1]['selection_reason'],
                         'prior_forward_end_visual_evidence')
        # A strict three-image budget must ALSO prioritize authentic route
        # evidence: increasing the budget is not what fixes the regression.
        short=navigation_visual_anchors(m,'obs_0084',max_pairs=3)
        self.assertEqual([v['observation_id'] for v in short[:2]],
                         ['obs_0019','obs_0025'])

    def test_only_same_stance_candidates_have_relative_left_right(self):
        m=eval39_memory()
        m.route_candidates=[
            dict(candidate_id='old-stair',
                 source_observation_id='obs_0019',
                 absolute_heading_deg=210,target_type='STAIRS'),
            dict(candidate_id='current-gap',
                 source_observation_id='obs_0078',
                 absolute_heading_deg=210,target_type='PASSAGE')]
        raw=mission_evidence(m,m.current['pose'])
        copied=copy.deepcopy(raw)
        projected=compact_navigation_context(dict(memory=raw))['memory']
        self.assertEqual(raw,copied)
        candidates={v['candidate_id']:v for v in
                    projected['candidate_route_hypotheses']}
        self.assertEqual(candidates['current-gap']['relative_side_from_current'],
                         'LEFT')
        self.assertEqual(candidates['current-gap']['relative_turn_deg_from_current'],
                         120)
        self.assertTrue(candidates['current-gap']['source_stance_matches_current'])
        self.assertFalse(candidates['old-stair']['source_stance_matches_current'])
        self.assertNotIn('relative_side_from_current',candidates['old-stair'])
        self.assertNotIn('absolute_heading_deg',candidates['old-stair'])
        self.assertNotIn('absolute_heading_deg',candidates['current-gap'])
        self.assertLess(len(json.dumps(projected['route_continuity'])),2500)

    def test_navigation_prompt_prefers_short_progress_and_targeted_observation(self):
        self.assertIn('one plausible continuous passage',MISSION_PROMPT)
        self.assertIn('favor a short model-chosen forward exploration',MISSION_PROMPT)
        self.assertIn('Call observe_for_goal selectively',MISSION_PROMPT)
        self.assertIn('not failure of the completed steps',MISSION_PROMPT)
        self.assertNotIn('target_heading_deg',MISSION_PROMPT)
        self.assertLess(len(MISSION_PROMPT),5000)
        c=registry(camera(),100)
        for tool in ('navigate_relative','turn_and_observe'):
            self.assertNotIn('target_heading_deg',
                c[tool]['parameters']['properties'])

if __name__=='__main__':
    unittest.main()
