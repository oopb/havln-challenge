"""Full independent completion path with only partially localized entrances."""
import unittest
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
