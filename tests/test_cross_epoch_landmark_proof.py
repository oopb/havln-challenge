"""Landmark crossing must not use coordinates after RGB-D reanchor.

These synthetic receipts use exact per-frame hash pair references; only the
current local epoch and an uninterrupted measured forward chain may prove
that a semantic boundary has been passed.
"""
import math
import unittest
from types import SimpleNamespace

from havln_agent.progress import _passed_landmark


def observation(index,epoch):
    return dict(observation_id='obs_%04d'%index,step=index,
        pose=dict(x=0.,y=0.,z=index*.75,heading=0.,motion_epoch=epoch),
        rgb_hash='rgb%d'%index,depth_hash='depth%d'%index)


def action(index,epoch,valid=True):
    m=(dict(status='ESTIMATED',uncertainty_m=.02,
            translation_m=dict(x_right=0.,y_up=0.,z_forward=.75),
            observation_pair=dict(before_rgb_hash='rgb%d'%index,
              before_depth_hash='depth%d'%index,
              after_rgb_hash='rgb%d'%(index+1),
              after_depth_hash='depth%d'%(index+1)))
       if valid else dict(status='UNKNOWN',reason='feature_correspondence_failed'))
    return dict(step=index,subgoal=0,action='MOVE_FORWARD',
        pose=dict(heading=0.,motion_epoch=epoch),
        result=dict(observation_before='obs_%04d'%index,
                    observation_after='obs_%04d'%(index+1),
                    rgbd_motion=m))


def grounding(index,epoch,heading,bearing=0.):
    return dict(status='LOCALIZED',target='doorway',
        position=dict(status='LOCALIZED',x_right=0.,z_forward=.5,
                      bearing_degrees=bearing),
        source=dict(observation_id='obs_%04d'%index,subgoal=0,
            pose=dict(heading=heading,motion_epoch=epoch)))


def fixture(end=2,broken=False,cross_epoch=False):
    records=[observation(i,0 if i<end or not cross_epoch else 1)
             for i in range(end+1)]
    moves=[action(i,0,valid=(not broken or i!=1)) for i in range(end)]
    source=grounding(0,0,0.)
    dest=grounding(end,1 if cross_epoch else 0,math.pi)
    memory=SimpleNamespace(observations=records,
        groundings=[source,dest],actions=moves,
        task=SimpleNamespace(current_subgoal_index=0),
        camera=SimpleNamespace(robot_radius=.15))
    good=[a for a in moves if a['result']['rgbd_motion']['status']=='ESTIMATED']
    return memory,dest,good


class CrossEpochLandmarkTests(unittest.TestCase):
    def test_registered_same_epoch_crossing_supported(self):
        memory,dest,moves=fixture()
        self.assertTrue(_passed_landmark(memory,[dest],moves))

    def test_visual_behind_not_sufficient_after_epoch_reset(self):
        memory,dest,moves=fixture(cross_epoch=True)
        self.assertFalse(_passed_landmark(memory,[dest],moves))

    def test_unknown_intermediate_motion_breaks_metric_chain(self):
        memory,dest,moves=fixture(end=3,broken=True)
        self.assertFalse(_passed_landmark(memory,[dest],moves))

    def test_missing_cited_intermediate_motion_cannot_claim_crossing(self):
        memory,dest,moves=fixture(end=3)
        self.assertFalse(_passed_landmark(memory,[dest],[moves[0],moves[-1]]))

    def test_reverse_facing_without_motion_is_not_crossing(self):
        memory,dest,moves=fixture()
        self.assertFalse(_passed_landmark(memory,[dest],[]))


if __name__=='__main__':
    unittest.main()
