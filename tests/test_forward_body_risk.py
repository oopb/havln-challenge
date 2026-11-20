"""Advisory full-body public-Depth risk, independent of collision ground truth."""
import time
import unittest

import numpy as np

from havln_agent.perception import Camera, Geometry, Observation


def camera():
    return Camera(width=256, height=256, hfov=90,
                  min_depth=0, max_depth=10, normalized=True,
                  camera_height=1.25, robot_height=1.5,
                  robot_radius=0.1, forward_step=0.25, turn_angle=15)


def observed(depth, pitch=0):
    return Observation(
        rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        depth=depth, instruction="Cross the room.",
        captured_at=time.monotonic(), pitch_deg=pitch)


class ForwardBodyRiskTests(unittest.TestCase):
    def test_right_side_warning_without_centerline_obstruction(self):
        d = np.full((256, 256), 0.3, dtype=np.float32)
        # 0.4 m Depth at x≈+0.10 m, well outside the old ±0.025 m
        # center-only corridor but inside the robot radius + small margin.
        d[75:150, 156:168] = 0.04
        geometry = Geometry(observed(d), camera(), margin=0.0)
        self.assertFalse(geometry.central_forward_obstruction()["confirmed"])
        risk = geometry.forward_body_risk()
        self.assertIsNotNone(risk)
        self.assertEqual(risk["identity"], "unverified_obstacle")
        self.assertIn("right_edge", [r["region"] for r in
                                     risk["affected_regions"]])
        self.assertTrue(any(0.35 < r["nearest_forward_m"] < 0.45
                            for r in risk["affected_regions"]))

    def test_left_side_warning(self):
        d = np.full((256, 256), 0.3, dtype=np.float32)
        d[75:150, 87:99] = 0.04
        risk = Geometry(observed(d), camera(), margin=0).forward_body_risk()
        self.assertIn("left_edge", [r["region"] for r in
                                    risk["affected_regions"]])

    def test_far_open_depth_has_no_warning(self):
        d = np.full((256, 256), 0.4, dtype=np.float32)
        self.assertIsNone(
            Geometry(observed(d), camera(), margin=0).forward_body_risk())

    def test_isolated_noise_and_invalid_depth_do_not_claim_obstruction(self):
        d = np.zeros((256, 256), dtype=np.float32)
        d[107:110, 162:164] = 0.04
        self.assertIsNone(
            Geometry(observed(d), camera(), margin=0).forward_body_risk())
        self.assertIsNone(
            Geometry(observed(np.zeros_like(d)), camera(),
                     margin=0).forward_body_risk())


if __name__ == "__main__":
    unittest.main()
