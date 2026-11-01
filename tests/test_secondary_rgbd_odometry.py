"""Contract tests for optional secondary, depth-based RGB feature odometry.

No OpenCV dependency is required for these deterministic numerical tests.
The fallback must not turn commanded step lengths into displacement claims.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from havln_agent.motion import estimate_translation
from havln_agent.motion_fallback import _orb_metric_consensus


def synthetic_points():
    grid=np.array([(x,y) for x in np.linspace(.08,.92,10)
                         for y in np.linspace(.08,.92,6)])
    xyz=np.zeros((len(grid),3),float)
    xyz[:,0]=np.linspace(-.7,.7,len(grid))
    xyz[:,1]=np.linspace(-.4,.4,len(grid))
    xyz[:,2]=np.linspace(.9,4.2,len(grid))
    other=xyz.copy()
    # A measured camera-frame displacement; NOT a commanded step.
    other[:45]-=np.array([0.,.04,.25])
    other[45:]-=np.array([.3,-.3,-.15])  # moving object/outlier cluster
    return xyz,other,grid


class OptionalOdometryTests(unittest.TestCase):
    def setUp(self):
        self.camera=SimpleNamespace(
            forward_step=.25,max_step_height=.2)

    def test_broad_independent_static_consensus_can_recover_translation(self):
        a,b,uv=synthetic_points()
        result=_orb_metric_consensus(a,b,uv,-30.,self.camera)
        self.assertEqual(result['status'],'ESTIMATED',result)
        self.assertEqual(result['source'],'orb_mutual_rgbd_static_consensus')
        self.assertFalse(result['position_verified'])
        self.assertGreater(result['inliers'],35)
        self.assertGreater(result['translation_m']['z_forward'],.15)
        self.assertEqual(result['vertical_motion'],'DOWN')

    def test_near_zero_static_motion_is_not_fabricated_forward_progress(self):
        a,_,uv=synthetic_points()
        result=_orb_metric_consensus(a,a.copy(),uv,0.,self.camera)
        self.assertEqual(result['status'],'UNKNOWN')
        self.assertEqual(result['reason'],'no_resolvable_forward_translation')
        self.assertNotIn('translation_m',result)

    def test_narrow_spatial_matches_are_insufficient(self):
        a,b,uv=synthetic_points()
        uv[:]=.5
        result=_orb_metric_consensus(a,b,uv,-30.,self.camera)
        self.assertEqual(result['status'],'UNKNOWN')
        self.assertEqual(result['reason'],
                         'insufficient_distributed_orb_metric_consensus')
        self.assertNotIn('translation_m',result)

    def test_primary_good_measurement_never_replaced_by_optional_fallback(self):
        primary=dict(status='ESTIMATED',source='original',
                     observation_pair=dict(before_rgb_hash='x'))
        with patch('havln_agent.motion._estimate_patch_translation',
                   return_value=primary), \
             patch('havln_agent.motion_fallback.estimate_orb_translation') as fallback:
            output=estimate_translation(None,None,self.camera)
        self.assertIs(output,primary)
        fallback.assert_not_called()

    def test_fallback_keeps_authenticated_original_frame_pair(self):
        primary=dict(status='UNKNOWN',reason='inconsistent_or_dynamic_correspondences',
                     observation_pair=dict(before_rgb_hash='a',after_rgb_hash='b',
                                           before_depth_hash='c',after_depth_hash='d'))
        candidate=dict(status='ESTIMATED',source='orb_mutual_rgbd_static_consensus',
                       translation_m=dict(x_right=0.,y_up=0.,z_forward=.24))
        with patch('havln_agent.motion._estimate_patch_translation',
                   return_value=primary), \
             patch('havln_agent.motion_fallback.estimate_orb_translation',
                   return_value=candidate):
            output=estimate_translation(None,None,self.camera)
        self.assertEqual(output['status'],'ESTIMATED')
        self.assertEqual(output['observation_pair'],primary['observation_pair'])
        self.assertFalse(output['position_verified'])
        self.assertEqual(output['primary_estimator_failure_reason'],
                         'inconsistent_or_dynamic_correspondences')

    def test_stationary_primary_cannot_be_overridden(self):
        primary=dict(status='UNKNOWN',reason='no_resolvable_forward_translation',
                     observation_pair=dict(before_rgb_hash='a'))
        with patch('havln_agent.motion._estimate_patch_translation',
                   return_value=primary), \
             patch('havln_agent.motion_fallback.estimate_orb_translation') as fallback:
            output=estimate_translation(None,None,self.camera)
        self.assertEqual(output['status'],'UNKNOWN')
        fallback.assert_not_called()

    def test_unavailable_optional_backend_is_visible_in_diagnostics(self):
        primary=dict(status='UNKNOWN',reason='inconsistent_or_dynamic_correspondences',
                     observation_pair={})
        with patch('havln_agent.motion._estimate_patch_translation',
                   return_value=primary), \
             patch('havln_agent.motion_fallback.estimate_orb_translation',
                   return_value=dict(status='UNAVAILABLE',
                                     reason='opencv_python_not_available')):
            output=estimate_translation(None,None,self.camera)
        self.assertIs(output,primary)
        self.assertEqual(output['secondary_odometry']['status'],'UNAVAILABLE')
        self.assertEqual(output['secondary_odometry']['reason'],
                         'opencv_python_not_available')

    def test_missing_opencv_preserves_existing_unknown(self):
        primary=dict(status='UNKNOWN',reason='insufficient_static_correspondences',
                     observation_pair={})
        with patch('havln_agent.motion._estimate_patch_translation',
                   return_value=primary), \
             patch('havln_agent.motion_fallback.estimate_orb_translation',
                   return_value=None):
            output=estimate_translation(None,None,self.camera)
        self.assertIs(output,primary)


if __name__=='__main__':
    unittest.main()
