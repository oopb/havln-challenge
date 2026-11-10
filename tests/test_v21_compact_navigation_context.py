"""v21: compact NavigationAgent input over existing canonical mission evidence.

Regression is deliberately offline: no API connection, Habitat install or
navigation model success assumption. The original Memory is never mutated.
"""
import copy
import json
import unittest
from unittest.mock import Mock

from havln_agent.contexts import (
    navigation_context,compact_navigation_context)
from havln_agent.mission_history import mission_evidence
from test_forward_preflight_v9 import test_runtime,geometry


def size(value):
    return len(json.dumps(value,sort_keys=True,ensure_ascii=False))


def large_request():
    angles=list(range(0,360,15))
    current='obs_current'
    blocker=dict(
        central_support_pixels=654,
        nearest_along_m=.235,
        focus_region_xyxy_norm=[.4,.36,.6,.79],
        source_pixel_xy=[30,41],
        reason='measured_centerline_obstacle',
        diagnostic_blob='depth_profile_'*30)
    blocked=[dict(
        kind='BLOCKED_BEFORE_FORWARD',
        source_observation_id='obs_%04d'%i,
        heading_deg=i*15,pitch_deg=-30,
        source_stance_matches_current=i==4,
        repeated_same_frame_heading_attempts=i+2,
        central_support_pixels=654,
        nearest_along_m=.235,
        focus_region_xyxy_norm=[.4,.36,.6,.79],
        rgb_hash='f'*64,depth_hash='a'*64,
        reason='measured_centerline_obstacle',
        historical_depth_does_not_prove_current_obstacle=True)
        for i in range(5)]
    stalls=[dict(
        kind='STALLED_AFTER_FORWARD',
        source_observation_id='obs_stall_%s'%i,
        after_observation_id='obs_after_%s'%i,
        heading_deg=180,pitch_deg=-30,
        repeated_same_frame_heading_attempts=i+1,
        source_stance_matches_current=False,
        reason='executed_forward_same_RGBD_before_and_after',
        depth_hash='e'*64,rgb_hash='d'*64,
        collision_proven=False,displacement_verified=False)
        for i in range(4)]
    tasks=[dict(
        observation_task_id='observation_%03d'%i,
        objective='Find a visible path around the people and the railing.'*4,
        outcome='OBSERVATION_COMPLETED',
        end_observation_id='obs_session_%03d'%i,
        observation_summary='Person blocks the central approach.'*15,
        navigation_suggestion='Try the left corridor; no clearance guarantee.'*15,
        selected_views=[
            dict(observation_id='obs_%04d'%j,
                 observation_summary='Repeated same visual hypothesis.'*12)
            for j in range(3)],
        waypoints=[
            dict(source_observation_id='obs_%04d'%j,
                 absolute_heading_deg=180,
                 relative_bearing_deg_from_current=None)
            for j in range(3)])
        for i in range(4)]
    routes=[dict(
        candidate_id='visual_route_%04d'%i,
        source_observation_id='obs_%04d'%i,
        target_type='possible passage',target_kind='APPROACH',
        confidence='medium',route_status='UNCONFIRMED',
        source_stance_matches_current=False,
        absolute_heading_deg=None,
        evidence='Visible but occluded floor around a person. '*10,
        navigation_suggestion='Inspect floor left of the figure. '*12,
        uncertainty='Side gap width unknown.'*12)
        for i in range(6)]
    segments=[dict(
        first_step=i*10,last_step=i*10+1,forward_steps=2,
        heading_deg=120.,start_observation_id='obs_f_%04d'%i,
        end_observation_id='obs_f_%04d_end'%i,
        displacement_verified=False,
        observed_height_change_m_est=None,uncertainty_m_est=None,
        evidence='Executed MOVE and RGB-D changed, no metric translation')
        for i in range(5)]
    receipts=[dict(
        tool_call_id='tool_%04d'%i,tool='turn_and_observe',
        status='ACTIONS_COMPLETED',executed_actions=3,
        start_observation_id='obs_1',end_observation_id='obs_%04d'%i,
        args=dict(turn_direction='LEFT',turn_degrees=45,
                  look='none',purpose='Avoid the nearby wall.'*24),
        blocked_at=None,blocker=None,camera_pitch_limit=None)
        for i in range(5)]
    receipts[-1].update(status='CAMERA_PITCH_LIMIT',
                        reason='rejected pitch beyond camera limit')
    views=[dict(
        source_observation_id='obs_%04d'%i,heading_deg=i*15,
        pitch_deg=0,relative_bearing_deg_from_current=i*15,
        observed_only=True)
        for i in range(16)]
    coverage=dict(
        scope='same_executed_forward_stance_not_metric_position',
        sampled_observations=85,
        distinct_heading_pitch_views=24,
        latest_distinct_views=views,
        full_coverage_by_pitch=[
            dict(pitch_deg=0,observed_heading_bins_deg=angles,
                 distinct_headings=24)],
        turn_actions_since_last_forward=80,
        look_actions_since_last_forward=5,
        repeated_same_RGBD_heading_pitch_views=52,
        horizontal_360_coverage_observed=True,
        older_views_may_be_omitted=True,
        historical_depth_does_not_authorize_move=True)
    memory=dict(
        contract='mission_evidence_v1',
        instruction='Descend stairs, pass dancing figure and reach piano.',
        progress=dict(narrative='Current mission still in progress.',
            navigation_intent='Find passage.',
            uncertainties=['Cannot confirm a passage.' for _ in range(6)],
            source_observation_ids=['obs_%04d'%i for i in range(12)],
            prior_navigation_intent=dict(text='A potentially stale direction',
                source=dict(source_observation_id='obs_old'))),
        current=dict(observation_id=current,heading_deg=150.,pitch_deg=-30.,
                     stance_id='0:15'),
        blocked_at_source_stances=blocked,
        stalled_forward_at_source_stances=stalls,
        recent_forward_failures=blocked[:3]+stalls[:3],
        observed_front_blocks=[
            dict(kind='OBSERVED_FRONT_DEPTH_BLOCK_NO_MOVE_ATTEMPT',
                 source_observation_id=source,
                 heading_deg=150,
                 source_stance_matches_current=True,
                 blocker=blocker,repeat_observation_count=40)
            for source in (current,'obs_other','obs_earlier','obs_latest')],
        recent_tool_receipts=receipts,
        recent_camera_pitch_limits=[
            dict(current_pitch_deg=-60,rejected_look='down',
                 allowed_look=['up','none'],
                 repeated_same_invalid_look=4) for _ in range(3)],
        observation_tasks=tasks,
        candidate_route_hypotheses=routes,
        observed_forward_action_segments=segments,
        historical_frames=[
            dict(observation_id='obs_%04d'%i,
                 heading_deg=180.,pitch_deg=-30,
                 observation_summary='Historical image caption'*20,
                 navigation_suggestion='Move left'*20)
            for i in range(3)],
        last_motion_decision=dict(source_observation_id='obs_previous',
            end_observation_id='obs_older',
            status='ACTIONS_COMPLETED',executed_forward_actions=2,
            forward_actions_with_RGBD_change=2,last_heading_deg=165.,
            atomic_actions_since_motion=87,
            last_motion_was_uninterrupted=True,
            motion_is_evidence_not_command=True,
            current_heading_deg=150.,decision_id='agent_0001'),
        metric_pose_available=False,
        bearing_convention=dict(
            model_turn_commands='LEFT/RIGHT/NONE + nonnegative degrees'))
    return dict(memory=memory,
        front_depth_preflight=dict(
            status='BLOCKED',source_observation_id=current,blocker=blocker),
        liveness_diagnostic=dict(consecutive_unchanged_decisions=0),
        actions_remaining=20,
        current_rgb=dict(observation_id=current,image_attached=True),
        camera_controls=dict(allowed_look_now=['none','up']),
        actuator=dict(forward_step_m=.25,turn_step_deg=15))


class CompactNavigationEvidenceV21(unittest.TestCase):
    def test_compaction_does_not_mutate_canonical_record_or_create_new_summary(self):
        full=large_request()
        before=copy.deepcopy(full)
        compact=compact_navigation_context(full)
        self.assertEqual(full,before)
        self.assertEqual(compact['memory']['contract'],'mission_evidence_v1')
        self.assertNotIn('navigation_progress',compact)
        self.assertEqual(compact['memory']['instruction'],
                         full['memory']['instruction'])
        self.assertEqual(compact['memory']['current'],
                         full['memory']['current'])
        self.assertEqual(compact['front_depth_preflight'],
                         full['front_depth_preflight'])

    def test_realistic_full_history_is_substantially_smaller(self):
        full=large_request()
        compact=compact_navigation_context(full)
        self.assertLess(size(compact),size(full)*.55)
        self.assertLess(size(compact),12000)
        self.assertNotIn('recent_forward_failures',compact['memory'])
        self.assertNotIn('bearing_convention',compact['memory'])
        self.assertNotIn('liveness_diagnostic',compact)

    def test_complete_camera_coverage_and_true_turn_cost_survive(self):
        compact=compact_navigation_context(large_request())
        coverage=compact['memory']['recent_views_at_current_stance']
        self.assertEqual(coverage['turn_actions_since_last_forward'],80)
        self.assertEqual(coverage['look_actions_since_last_forward'],5)
        self.assertEqual(coverage['repeated_same_RGBD_heading_pitch_views'],52)
        self.assertTrue(coverage['horizontal_360_coverage_observed'])
        self.assertEqual(coverage['full_coverage_by_pitch'][0]
                         ['observed_heading_bins_deg'],list(range(0,360,15)))
        self.assertEqual(len(coverage['latest_distinct_views']),2)
        self.assertTrue(coverage['coverage_is_information_not_motion'])

    def test_blocked_stalled_and_unattempted_obstacles_remain_distinct(self):
        compact=compact_navigation_context(large_request())
        m=compact['memory']
        self.assertEqual(len(m['blocked_at_source_stances']),2)
        self.assertEqual(len(m['stalled_forward_at_source_stances']),2)
        self.assertEqual(len(m['observed_front_blocks']),2)
        self.assertNotEqual(m['observed_front_blocks'][0][
            'source_observation_id'],'obs_current')
        self.assertEqual(m['blocked_at_source_stances'][0]['kind'],
                         'BLOCKED_BEFORE_FORWARD')
        self.assertEqual(m['stalled_forward_at_source_stances'][0]['kind'],
                         'STALLED_AFTER_FORWARD')
        self.assertEqual(compact['front_depth_preflight']['status'],'BLOCKED')

    def test_last_motion_receipt_and_source_evidence_survive(self):
        compact=compact_navigation_context(large_request())
        m=compact['memory']
        self.assertEqual(m['last_motion_decision'][
            'atomic_actions_since_motion'],87)
        self.assertEqual(m['last_motion_decision'][
            'forward_actions_with_RGBD_change'],2)
        self.assertEqual(len(m['observed_forward_action_segments']),3)
        self.assertFalse(any(a['displacement_verified']
                             for a in m['observed_forward_action_segments']))
        self.assertEqual(len(m['observation_tasks']),2)
        self.assertEqual(len(m['candidate_route_hypotheses']),3)
        self.assertIn('source_observation_id',
                      m['candidate_route_hypotheses'][0])
        self.assertEqual(len(m['historical_frames']),3)
        self.assertEqual(len(m['recent_tool_receipts']),1)
        self.assertEqual(m['recent_tool_receipts'][0]['status'],
                         'CAMERA_PITCH_LIMIT')

    def test_liveness_is_not_stall_proof_but_nonzero_is_retained(self):
        state=large_request()
        state['liveness_diagnostic']['consecutive_unchanged_decisions']=3
        out=compact_navigation_context(state)
        self.assertEqual(out['liveness_diagnostic'][
            'consecutive_unchanged_decisions'],3)

    def test_live_navigation_api_receives_projection_not_raw_memory(self):
        runtime,_=test_runtime()
        runtime.geometry=Mock(return_value=geometry('EXECUTABLE'))
        # Use a long canonical stream without actually executing motion.
        original=mission_evidence(runtime.memory,
            runtime.motion_controller.pose())
        runtime.navigation_agent.decide()
        sent=runtime.client.complete.call_args.args[3]
        self.assertNotIn('recent_forward_failures',sent['memory'])
        self.assertNotIn('bearing_convention',sent['memory'])
        self.assertIn('recent_views_at_current_stance',sent['memory'])
        self.assertEqual(sent['front_depth_preflight']['status'],'CLEAR')
        still=mission_evidence(runtime.memory,
            runtime.motion_controller.pose())
        self.assertEqual(original['contract'],still['contract'])
        self.assertIn('recent_forward_failures',still)
        self.assertIn('bearing_convention',still)
        self.assertNotIn('navigation_progress',sent)


if __name__=='__main__':
    unittest.main()
