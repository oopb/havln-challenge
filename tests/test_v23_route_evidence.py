"""v23: source-linked route continuity, opposing views and useful LOOK memory.

Offline regression reproduces eval-37's 225-to-45-degree direction change
without trusting simulator height, model narratives or static route labels.
"""
import copy
import json
import math
import unittest

from havln_agent.contexts import compact_navigation_context
from havln_agent.memory_records import MemoryRecords
from havln_agent.mission_history import mission_evidence
from havln_agent.navigation_evidence import route_continuity, recent_look_evidence
from havln_agent.navigation_history import navigation_visual_anchors
from test_forward_preflight_v9 import camera


def pose(yaw, pitch=0, generation=0):
    return dict(heading=math.radians(yaw), pitch_deg=pitch,
        motion_epoch=0, viewpoint_generation=generation,
        stance_id='0:%d'%generation, position_coordinates_valid=False)


def frame(i, yaw, pitch, generation):
    return dict(observation_id='obs_%04d'%i,
        pose=pose(yaw,pitch,generation),rgb_hash='rgb_%04d'%i,
        depth_hash='depth_%04d'%i,place_id='p%d'%generation)


def turn(index, source, yaw, step, seen, claim, *, look='none',
         end_observation_id=None, purpose='inspect the current route'):
    executed=(['LOOK_UP' if look=='up' else 'LOOK_DOWN']
              if look!='none' else [])
    return dict(turn_index=index,source_observation_id=source,
        source_heading_deg=yaw,step_started=step,
        decision=dict(observed_evidence=[seen],progress_narrative=claim,
            tool='turn_and_observe',args=dict(look=look,purpose=purpose)),
        result=dict(end_observation_id=end_observation_id,
                    executed_motion=executed))


def fixture():
    memory=MemoryRecords(camera())
    memory.reset('eval37-synthetic','Descend stairs, pass the foyer, reach piano.')
    records=[frame(0,0,0,0),frame(11,225,-60,0),frame(12,225,-60,0)]
    records += [frame(12+i,225,-60,i) for i in range(1,8)]
    records += [frame(49,45,0,7),frame(50,45,0,8),
                frame(51,45,-30,8),frame(52,45,0,8)]
    memory.observations=records
    memory.current=records[-1]
    memory.task.progress_narrative=(
        'Still on upper level, stairs not yet located.')  # fallible model claim
    for i in range(7):
        memory.actions.append(dict(step=11+i,action='MOVE_FORWARD',
            pose=pose(225,-60,i),result=dict(visual_change=True,
                observation_before='obs_%04d'%(12+i),
                observation_after='obs_%04d'%(13+i))))
    memory.actions.append(dict(step=48,action='MOVE_FORWARD',
        pose=pose(45,0,7),result=dict(visual_change=True,
            observation_before='obs_0049',observation_after='obs_0050')))
    memory.actions.extend([
        dict(step=49,action='LOOK_DOWN',pose=pose(45,-30,8),
             result=dict(visual_change=True,observation_before='obs_0050',
                         observation_after='obs_0051')),
        dict(step=50,action='LOOK_UP',pose=pose(45,0,8),
             result=dict(visual_change=True,observation_before='obs_0051',
                         observation_after='obs_0052'))])
    memory.navigation_turns=[
        turn(2,'obs_0012',225,11,'Wooden stair treads descend ahead.',
             'At top, starting descent.'),
        turn(9,'obs_0019',225,18,'Wood floor and stone wall now in front.',
             'Possibly near bottom landing.'),
        turn(23,'obs_0049',45,48,'Stairs descend ahead.',
             'Found the stairs.'),
        turn(24,'obs_0050',45,49,'Stairs ascend ahead to a figure.',
             'This contradicts descent.',look='down',
             end_observation_id='obs_0051',purpose='Inspect treads.'),
        turn(25,'obs_0051',45,50,'Stairs recede upward.',
             'Need wider context.',look='up',
             end_observation_id='obs_0052',purpose='Recover distant view.')]
    return memory


class Eval37RouteEvidence(unittest.TestCase):
    def test_facts_and_conflicting_model_claims_both_survive(self):
        memory=fixture()
        segments=memory.route_progress_summary(
            max_successful_segments=None)['successful_segments']
        self.assertEqual(segments[0]['forward_steps'],7)
        summary=route_continuity(memory,memory.current['pose'],segments)
        self.assertEqual(summary['selected_historical_forward'][
            'start_observation_id'],'obs_0012')
        self.assertEqual(summary['selected_historical_forward'][
            'end_observation_id'],'obs_0019')
        self.assertEqual(summary['selected_historical_forward'][
            'executed_forward_with_RGBD_change'],7)
        self.assertEqual(summary['separation_from_historical_forward_deg'],180)
        self.assertTrue(summary['opposing_camera_heading'])
        self.assertTrue(summary['previous_forward_and_current_view_may_be_a_retrace_not_proven'])
        snapshots=summary['model_assessment_snapshots']
        self.assertEqual([s['source_observation_id'] for s in snapshots],
            ['obs_0012','obs_0019','obs_0049','obs_0050'])
        self.assertIn('descend',snapshots[2]['model_visual_description'])
        self.assertIn('ascend',snapshots[3]['model_visual_description'])
        self.assertTrue(all(s['interpretation_unverified'] for s in snapshots))
        self.assertTrue(summary['executed_steps_do_not_prove_stair_descent_or_metric_displacement'])

    def test_selects_prior_forward_before_after_not_generic_origin(self):
        memory=fixture()
        selected=navigation_visual_anchors(memory,'obs_0052',max_pairs=3)
        self.assertEqual([x['observation_id'] for x in selected],
            ['obs_0012','obs_0019','obs_0050'])
        self.assertNotIn('obs_0000',[v['observation_id'] for v in selected])
        self.assertEqual(selected[0]['selection_reason'],
            'prior_forward_start_visual_evidence')
        self.assertEqual(selected[1]['selection_reason'],
            'prior_forward_end_visual_evidence')

    def test_recent_look_evidence_includes_real_result_not_only_repeat_count(self):
        memory=fixture()
        checks=recent_look_evidence(memory,memory.current['pose'])
        self.assertEqual(checks['sampled_recent_look_decisions'],2)
        self.assertEqual(checks['first_time_yaw_pitch_views'],1)
        self.assertEqual(checks['revisited_yaw_pitch_views'],1)
        self.assertIn('upward',checks[
            'latest_questions_and_model_descriptions'][0][
                'following_model_visual_description'])
        # Rejected LOOK is not a real camera action.
        memory.navigation_turns[-1]['result']['executed_motion']=[]
        after=recent_look_evidence(memory,memory.current['pose'])
        self.assertEqual(after['sampled_recent_look_decisions'],1)

    def test_compact_model_input_keeps_evidence_without_changing_memory(self):
        memory=fixture()
        raw=mission_evidence(memory,memory.current['pose'])
        previous=copy.deepcopy(raw)
        self.assertEqual(raw['progress']['narrative'],
                         'Still on upper level, stairs not yet located.')
        compact=compact_navigation_context(dict(memory=raw))['memory']
        self.assertEqual(raw,previous)
        self.assertEqual(compact['route_continuity'][
            'selected_historical_forward']['heading_deg'],225.0)
        self.assertEqual(compact['recent_look_evidence'][
            'revisited_yaw_pitch_views'],1)
        self.assertLess(len(json.dumps(compact['route_continuity'])),2400)
        self.assertLess(len(json.dumps(compact['recent_look_evidence'])),1100)
        self.assertNotIn('recent_forward_failures',compact)

    def test_empty_episode_does_not_gain_fake_motion_or_look(self):
        memory=MemoryRecords(camera())
        memory.reset('empty','Find the exit.')
        memory.observations=[frame(0,0,0,0)]
        memory.current=memory.observations[0]
        compact=compact_navigation_context(dict(
            memory=mission_evidence(memory,memory.current['pose'])))['memory']
        self.assertNotIn('route_continuity',compact)
        self.assertNotIn('recent_look_evidence',compact)


if __name__=='__main__':
    unittest.main()
