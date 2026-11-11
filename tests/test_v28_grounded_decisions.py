"""v28: replay eval-41 repeated Depth rejects without adding a navigation veto.

All tests use canonical-like source evidence and the existing v21 compact
context, not a second Memory or a privileged simulator pose.
"""
import copy
import json
import unittest

from havln_agent.contexts import compact_navigation_context
from havln_agent.model_angles import project_model_angles
from havln_agent.navigation_agent import MISSION_PROMPT
from test_v21_compact_navigation_context import large_request


def eval41_blocked_view():
    state=large_request()
    memory=state['memory']
    memory['current'].update(
        observation_id='obs_0080',heading_deg=60,pitch_deg=-60,
        stance_id='0:14')
    memory['blocked_at_source_stances']=[
        dict(kind='BLOCKED_BEFORE_FORWARD',source_observation_id='obs_0049',
             rgb_hash='old-rgb',depth_hash='old-depth',
             heading_deg=285,pitch_deg=-60,
             source_stance_matches_current=False,
             repeated_same_frame_heading_attempts=3),
        dict(kind='BLOCKED_BEFORE_FORWARD',source_observation_id='obs_0081',
             rgb_hash='new-rgb',depth_hash='new-depth',
             heading_deg=60,pitch_deg=-30,
             source_stance_matches_current=True,
             repeated_same_frame_heading_attempts=2),
        dict(kind='BLOCKED_BEFORE_FORWARD',source_observation_id='obs_0080',
             rgb_hash='current-rgb',depth_hash='current-depth',
             heading_deg=60,pitch_deg=-60,
             source_stance_matches_current=True,
             repeated_same_frame_heading_attempts=4,
             central_support_pixels=81,nearest_along_m=.345,
             reason='measured_centerline_obstacle'),
    ]
    state['front_depth_preflight']=dict(
        status='BLOCKED',source_observation_id='obs_0080',
        rgb_hash='current-rgb',depth_hash='current-depth',
        heading_deg=60,pitch_deg=-60,
        blocker=dict(central_support_pixels=81,nearest_along_m=.345))
    return state


class Eval41GroundedDecisions(unittest.TestCase):
    def test_current_identical_forward_rejections_are_prominent_without_duplicates(self):
        state=eval41_blocked_view()
        original=copy.deepcopy(state)
        projected=compact_navigation_context(state)
        self.assertEqual(state,original,'Canonical input must not mutate')
        current=projected['front_depth_preflight']
        self.assertEqual(current['previous_identical_view_forward_rejections'],4)
        self.assertEqual(current['previous_rejected_forward_actions_executed'],0)
        self.assertEqual(current['previous_rejection_observation_id'],'obs_0080')
        self.assertTrue(current['rejections_match_current_RGBD_stance_yaw_pitch'])
        history=projected['memory']['blocked_at_source_stances']
        self.assertEqual({x['source_observation_id'] for x in history},
                         {'obs_0049','obs_0081'})
        self.assertTrue(all(x['kind']=='BLOCKED_BEFORE_FORWARD' for x in history))
        self.assertFalse(any(x['source_observation_id']=='obs_0080'
                             for x in history))
        wire=project_model_angles(projected,60)
        self.assertEqual(wire['front_depth_preflight']['view_offset_deg_from_current'],0)
        self.assertEqual(wire['front_depth_preflight']['previous_identical_view_forward_rejections'],4)
        self.assertNotIn('heading_deg',json.dumps(wire))

    def test_new_pitch_has_its_own_rejection_count(self):
        state=eval41_blocked_view()
        state['front_depth_preflight'].update(
            source_observation_id='obs_0081',rgb_hash='new-rgb',
            depth_hash='new-depth',pitch_deg=-30)
        state['memory']['current'].update(
            observation_id='obs_0081',pitch_deg=-30)
        out=compact_navigation_context(state)
        self.assertEqual(out['front_depth_preflight'][
            'previous_identical_view_forward_rejections'],2)
        self.assertEqual(out['front_depth_preflight'][
            'previous_rejection_observation_id'],'obs_0081')
        self.assertEqual({x['source_observation_id']
                          for x in out['memory']['blocked_at_source_stances']},
                         {'obs_0049','obs_0080'})

    def test_clear_unknown_or_changed_sensor_are_not_past_rejections(self):
        for change in (dict(status='CLEAR'),dict(status='UNKNOWN'),
                       dict(rgb_hash='fresh-rgb'),
                       dict(depth_hash='fresh-depth'),
                       dict(heading_deg=75),dict(pitch_deg=-30)):
            state=eval41_blocked_view()
            state['front_depth_preflight'].update(change)
            out=compact_navigation_context(state)
            self.assertNotIn('previous_identical_view_forward_rejections',
                             out['front_depth_preflight'],
                             'Unexpected match for '+str(change))

    def test_old_stance_does_not_become_current_rejection(self):
        state=eval41_blocked_view()
        for item in state['memory']['blocked_at_source_stances']:
            item['source_stance_matches_current']=False
        out=compact_navigation_context(state)
        self.assertNotIn('previous_identical_view_forward_rejections',
                         out['front_depth_preflight'])

    def test_unattempted_predecision_block_is_not_rejected_forward(self):
        state=eval41_blocked_view()
        state['memory']['blocked_at_source_stances']=[]
        state['memory']['observed_front_blocks']=[dict(
            kind='OBSERVED_FRONT_DEPTH_BLOCK_NO_MOVE_ATTEMPT',
            source_observation_id='obs_0080',
            heading_deg=60,source_stance_matches_current=True)]
        out=compact_navigation_context(state)
        self.assertNotIn('previous_identical_view_forward_rejections',
                         out['front_depth_preflight'])
        self.assertEqual(out['front_depth_preflight']['status'],'BLOCKED')

    def test_navigation_policy_remains_advisory_and_compact(self):
        self.assertIn('previous_identical_view_forward_rejections',MISSION_PROMPT)
        self.assertIn('visible staircase does NOT prove descent',MISSION_PROMPT)
        self.assertIn('source-linked before/after observation IDs',MISSION_PROMPT)
        self.assertIn('favor a short model-chosen forward exploration',MISSION_PROMPT)
        self.assertIn('Every atomic forward rechecks fresh Depth',MISSION_PROMPT)
        self.assertLess(len(MISSION_PROMPT),5000)
        self.assertNotIn('forbid_forward',MISSION_PROMPT)
        self.assertNotIn('target_heading_deg',MISSION_PROMPT)


if __name__=='__main__':
    unittest.main()
