"""Full independent completion path with only partially localized entrances."""
import unittest
import math
from types import SimpleNamespace
from unittest.mock import patch

from havln_agent.progress import _stair_evidence


def observed(index):
    return dict(observation_id='o%d' % index,step=index,
                pose=dict(x=0.,z=0.))


def action(index,dy,motion):
    return dict(action='MOVE_FORWARD',step=index,
        pose=dict(heading=0.),
        result=dict(observation_before='o%d'%index,
            observation_after='o%d'%(index+1),
            rgbd_motion=dict(status='ESTIMATED',
                vertical_motion=motion,uncertainty_m=.02,
                translation_m=dict(y_up=dy,z_forward=.25)),
            landing_support=dict(status='SUPPORTED' if motion=='LEVEL' else 'NOT_EVALUATED')))


def grounding(observation_id,landing=False):
    return dict(status='PARTIAL',target='stairs',
        semantic_presence='VISIBLE',
        semantic=dict(target_visible=True,stair_direction='down',
            confidence='high',landing_box=[.2,.2,.8,.8] if landing else None,
            landing_visible=landing,lower_landing_visible=landing),
        position=dict(status='LOCALIZED',bearing_degrees=0.,
                      x_right=0.,z_forward=1.0),
        entry=dict(status='UNKNOWN',reason='entry_not_localized'),
        landing_geometry=dict(status='SUPPORTED' if landing else 'UNKNOWN'),
        source=dict(observation_id=observation_id,subgoal=0,
                    pose=dict(motion_epoch=0,heading=0.)))


class FullTransitionRegression(unittest.TestCase):
    def test_early_scene_and_later_grounded_landing_can_verify_independent_transition(self):
        memory=SimpleNamespace(
            camera=SimpleNamespace(forward_step=.25),
            task=SimpleNamespace(current_subgoal_index=0,subgoal_started=0),
            current=dict(pose=dict(motion_epoch=0)),
            observations=[observed(i) for i in range(5)],
            scene_history=[dict(
                source=dict(observation_id='o0',subgoal=0),
                scene=dict(landmarks=[dict(name='Descending wooden staircase')]))],
            groundings=[grounding('o4',landing=True)],
            actions=[action(0,-.23,'DOWN'),action(1,-.23,'DOWN'),
                     action(2,0.,'LEVEL'),action(3,0.,'LEVEL')])
        goal=dict(action='descend stairs',landmark='stairs',target='stairs')
        with patch('havln_agent.progress.registered_action',return_value=True), \
             patch('havln_agent.progress.current_visual_evidence',return_value=True):
            stage,supported,missing,contradictions,details=_stair_evidence(
                memory,goal,{'o0','o4'},0.,20.,1.25)
        self.assertEqual(stage,'COMPLETED',msg=str((supported,missing,details)))
        self.assertTrue(details['independent_stair_to_landing_transition'])
        self.assertEqual(details['source_observation_ids'][0],'o0')
        self.assertIn('sensor_verified_stair_to_landing_transition',supported)

    def test_later_unknown_forward_does_not_erase_completed_stair_transition(self):
        observations=[observed(i) for i in range(6)]
        observations[4].update(rgb_hash='r4',depth_hash='d4',
                               pose=dict(motion_epoch=0,heading=0.))
        land=grounding('o4',landing=True)
        land['source'].update(image_scope='current',rgb_hash='r4',
                              depth_hash='d4',pose=dict(motion_epoch=0,heading=0.))
        unknown=action(4,0.,'UNKNOWN')
        unknown['result']['rgbd_motion']['status']='UNKNOWN'
        mem=SimpleNamespace(
            camera=SimpleNamespace(forward_step=.25),
            task=SimpleNamespace(current_subgoal_index=0,subgoal_started=0),
            current=dict(pose=dict(motion_epoch=1),observation_id='o5'),
            observations=observations,
            scene_history=[dict(source=dict(observation_id='o0',subgoal=0),
                scene=dict(landmarks=[dict(name='Descending staircase')]))],
            groundings=[land],
            actions=[action(0,-.23,'DOWN'),action(1,-.23,'DOWN'),
                     action(2,0.,'LEVEL'),action(3,0.,'LEVEL'),unknown])
        with patch('havln_agent.progress.registered_action',
                   side_effect=lambda a,obs:a['result']['rgbd_motion']['status']=='ESTIMATED'), \
             patch('havln_agent.progress.current_visual_evidence',return_value=False):
            stage,supported,missing,contradictions,details=_stair_evidence(
                mem,dict(action='descend stairs',landmark='stairs',target='stairs'),
                {'o0','o4'},0.,10.,1.25)
        self.assertEqual(stage,'COMPLETED',msg=str((supported,missing,details)))
        self.assertEqual(details['evidence_cutoff_step'],4)
        self.assertEqual(details['motion_steps'],[0,1,2,3])
        self.assertTrue(details['continuous_rgbd_motion_chain'])
        self.assertIn('sensor_verified_stair_to_landing_transition',supported)

    def test_later_camera_view_does_not_erase_proven_past_landing(self):
        # Regression: a historical landing's source hashes are authentic,
        # even if the robot later looks somewhere else before verification.
        observations=[observed(i) for i in range(5)]
        observations[-1].update(rgb_hash='rgb4',depth_hash='depth4',
            pose=dict(motion_epoch=0,heading=0.))
        landing=grounding('o4',landing=True)
        landing['source'].update(image_scope='current',rgb_hash='rgb4',
                                depth_hash='depth4',pose=dict(motion_epoch=0,heading=0.))
        mem=SimpleNamespace(
            camera=SimpleNamespace(forward_step=.25),
            task=SimpleNamespace(current_subgoal_index=0,subgoal_started=0),
            current=dict(pose=dict(motion_epoch=0,heading=math.pi),
                         observation_id='later_view'),
            observations=observations,
            scene_history=[dict(
                source=dict(observation_id='o0',subgoal=0),
                scene=dict(landmarks=[dict(name='Descending wooden staircase')]))],
            groundings=[landing],
            actions=[action(0,-.23,'DOWN'),action(1,-.23,'DOWN'),
                     action(2,0.,'LEVEL'),action(3,0.,'LEVEL')])
        goal=dict(action='descend stairs',landmark='stairs',target='stairs')
        with patch('havln_agent.progress.registered_action',return_value=True), \
             patch('havln_agent.progress.current_visual_evidence',return_value=False):
            stage,supported,missing,contradictions,details=_stair_evidence(
                mem,goal,{'o0','o4'},0.,20.,1.25)
        self.assertEqual(stage,'COMPLETED',msg=str((missing,contradictions)))
        self.assertEqual(details['landing_evidence_scope'],
                         'registered_rgbd_at_capture_not_current_position')

    def test_captured_landing_survives_later_camera_turn(self):
        from havln_agent.progress import _captured_landing_proof
        source=dict(observation_id='o4',image_scope='current',rgb_hash='r4',
                    depth_hash='d4',pose=dict(motion_epoch=0,heading=0.))
        original=dict(observation_id='o4',step=4,rgb_hash='r4',
                    depth_hash='d4',pose=dict(motion_epoch=0,heading=0.))
        self.assertTrue(_captured_landing_proof(
            dict(source=source),{'o4':original},1))
        self.assertFalse(_captured_landing_proof(
            dict(source=source),{'o4':original},4))
        original['depth_hash']='different'
        self.assertFalse(_captured_landing_proof(
            dict(source=source),{'o4':original},1))
        original['depth_hash']='d4'
        source['image_scope']='historical_keyframe'
        self.assertFalse(_captured_landing_proof(
            dict(source=source),{'o4':original},1))

    def test_partial_entry_and_later_landing_still_preserve_initial_motion(self):
        memory=SimpleNamespace(
            camera=SimpleNamespace(forward_step=.25),
            task=SimpleNamespace(current_subgoal_index=0,subgoal_started=0),
            current=dict(pose=dict(motion_epoch=0)),
            observations=[observed(i) for i in range(5)],
            groundings=[grounding('o0'),grounding('o4',landing=True)],
            actions=[action(0,-.23,'DOWN'),action(1,-.23,'DOWN'),
                     action(2,0.,'LEVEL'),action(3,0.,'LEVEL')])
        goal=dict(action='descend stairs',landmark='stairs',target='stairs')
        with patch('havln_agent.progress.registered_action',return_value=True), \
             patch('havln_agent.progress.current_visual_evidence',return_value=True):
            stage,supported,missing,contradictions,details=_stair_evidence(
                memory,goal,{'o0','o4'},0.,20.,1.25)
        self.assertEqual(stage,'COMPLETED',msg=str(dict(missing=missing,supported=supported,details=details,contradictions=contradictions)))
        self.assertEqual(details['source_observation_ids'][0],'o0')
        self.assertEqual(details['motion_steps'],[0,1,2,3])
        self.assertIn('sensor_verified_stair_to_landing_transition',supported)
        self.assertTrue(details['independent_stair_to_landing_transition'])
        self.assertFalse(missing)
        self.assertFalse(contradictions)


if __name__=='__main__':
    unittest.main()
