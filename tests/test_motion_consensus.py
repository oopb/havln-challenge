"""RGB-D motion consensus: recover distributed static matches, reject moving clusters.

Regression derived from smoke-500-15: 43 mutual matches / 23 well-distributed
inliers were rejected solely by the old 0.65 ratio threshold. These are
deterministic synthetic 3D correspondences, not simulator ground-truth poses.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from havln_agent.motion import estimate_translation


def synthetic_features(*, concentrated=False):
    size=43
    descriptors=np.eye(size)
    start=np.zeros((size,3),dtype=float)
    start[:,0]=np.linspace(-.7,.7,size)
    start[:,1]=np.linspace(-.4,.4,size)
    start[:,2]=np.linspace(1.,3.8,size)
    end=start.copy()
    # Static consensus in the previous-camera coordinate frame when looking
    # down 60 degrees. The other 20 matches describe a different, moving body.
    end[:23]-=np.array([.001,.337,-.095])
    end[23:]-=np.array([.14,-.11,.12])
    xx=np.linspace(.10,.90,23)
    yy=np.resize(np.array([.12,.35,.60,.90]),23)
    pixels=np.zeros((size,2),float)
    pixels[:23,0]=.48 if concentrated else xx
    pixels[:23,1]=.48 if concentrated else yy
    pixels[23:]=np.column_stack((np.linspace(.3,.7,20),np.linspace(.2,.8,20)))
    return (descriptors,start,pixels),(descriptors,end,pixels)


class MotionConsensusTests(unittest.TestCase):
    def estimate(self, *, concentrated=False):
        before=SimpleNamespace(rgb=np.zeros((4,4,3),np.uint8),
            depth=np.zeros((4,4),np.float32),pitch_deg=-60.,captured_at=10.)
        after=SimpleNamespace(rgb=np.ones((4,4,3),np.uint8),
            depth=np.ones((4,4),np.float32),pitch_deg=-60.,captured_at=11.)
        camera=SimpleNamespace(forward_step=.25,max_step_height=.2)
        with patch('havln_agent.motion._features',
                   side_effect=synthetic_features(concentrated=concentrated)):
            return estimate_translation(before,after,camera)

    def test_distributed_consensus_recovers_23_of_43_static_matches(self):
        motion=self.estimate()
        self.assertEqual(motion['matches'],43)
        self.assertEqual(motion['inliers'],23)
        self.assertAlmostEqual(motion['inlier_fraction'],23/43)
        self.assertEqual(motion['status'],'ESTIMATED')
        self.assertEqual(motion['consensus_mode'],'distributed_static')
        self.assertFalse(motion['position_verified'])
        self.assertGreater(motion['translation_m']['z_forward'],.15)
        self.assertLess(motion['residual_rms_m'],.02)

    def test_compact_motion_cluster_not_sufficient_for_fallback(self):
        motion=self.estimate(concentrated=True)
        self.assertEqual(motion['status'],'UNKNOWN')
        self.assertEqual(motion['consensus_mode'],'rejected')
        self.assertEqual(motion['reason'],'inconsistent_or_dynamic_correspondences')

    def test_absence_of_live_matches_fails_closed(self):
        before=SimpleNamespace(rgb=np.zeros((4,4,3),np.uint8),
            depth=np.zeros((4,4),np.float32),pitch_deg=0.,captured_at=10.)
        after=SimpleNamespace(rgb=np.ones((4,4,3),np.uint8),
            depth=np.ones((4,4),np.float32),pitch_deg=0.,captured_at=11.)
        with patch('havln_agent.motion._features',
                   return_value=(np.zeros((0,2)),np.zeros((0,3)),np.zeros((0,2)))):
            motion=estimate_translation(before,after,SimpleNamespace())
        self.assertEqual(motion['status'],'UNKNOWN')
        self.assertEqual(motion['reason'],'insufficient_static_correspondences')


if __name__=='__main__':
    unittest.main()
