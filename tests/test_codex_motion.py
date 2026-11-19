"""Offline contracts for public RGB-D forward motion estimates."""
import unittest
from types import SimpleNamespace

import numpy as np

from havln_agent.codex_motion import assess_forward_motion


class CodexMotionTests(unittest.TestCase):
    def setUp(self):
        self.camera = SimpleNamespace(normalized=False, min_depth=0.0,
                                      max_depth=10.0, forward_step=0.25)
        self.rng = np.random.default_rng(18)
        self.rgb = self.rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        self.depth = np.full((256, 256), 2., dtype=np.float32)

    def observation(self, delta=0., noisy=False):
        rgb = self.rgb.copy()
        if noisy:
            rgb[90:120, 90:120] = self.rng.integers(
                0, 256, (30, 30, 3), dtype=np.uint8)
        return SimpleNamespace(rgb=rgb, depth=self.depth - delta, pitch_deg=0.)

    def test_small_real_motion_is_a_stall_even_with_scene_changes(self):
        before = self.observation()
        after = self.observation(0.015, noisy=True)
        result = assess_forward_motion(before, after, self.camera)
        self.assertEqual(result.state, "STALLED")
        self.assertEqual(result.reason, "rgbd_near_zero_translation")
        self.assertGreaterEqual(result.matched_regions, 9)

    def test_normal_forward_is_not_stalled(self):
        before = self.observation()
        after = self.observation(0.25, noisy=True)
        result = assess_forward_motion(before, after, self.camera)
        self.assertEqual(result.state, "PROGRESSED")

    def test_low_texture_is_uncertain_not_blocked(self):
        plain = np.full((224, 224, 3), 80, dtype=np.uint8)
        newer = plain.copy()
        newer[0, 0, 0] += 1
        result = assess_forward_motion(
            SimpleNamespace(rgb=plain, depth=self.depth, pitch_deg=0.),
            SimpleNamespace(rgb=newer, depth=self.depth, pitch_deg=0.),
            self.camera)
        self.assertEqual(result.state, "UNCERTAIN")

    def test_exact_unchanged_frame_still_detected(self):
        result = assess_forward_motion(self.observation(),
                                       self.observation(), self.camera)
        self.assertEqual(result.state, "STALLED")
        self.assertEqual(result.reason, "rgb_unchanged")

    def test_normalized_depth_uses_camera_scale(self):
        self.camera.normalized = True
        before = self.observation()
        after = self.observation(0.015, noisy=True)
        before.depth = before.depth / 10.
        after.depth = after.depth / 10.
        result = assess_forward_motion(before, after, self.camera)
        self.assertEqual(result.state, "STALLED")


if __name__ == "__main__":
    unittest.main()
