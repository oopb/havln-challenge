"""smoke-500-10 regression: discrete waypoint bearings, rechecks, and route lifecycle."""
import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from havln_agent.waypoint import generate_waypoints, goal_relation
from havln_agent.local_motion_controller import LocalMotionController
from havln_agent.tools import ToolExecutor


class FakeGeometry:
    def __init__(self):
        self.camera = SimpleNamespace(turn_angle=15, hfov=90, forward_step=0.25)
        self.diagnose = Mock(return_value=dict(clearance=0.25))
        self.assess_step = Mock(side_effect=lambda angle=0: dict(
            state="EXECUTABLE", reason=None, depth=dict(clearance=0.25),
            support=dict(status="SUPPORTED")))


class NavigationLoopTests(unittest.TestCase):
    def test_generated_waypoints_use_discrete_turns_not_half_turns(self):
        g = FakeGeometry()
        candidates = generate_waypoints(g, 1.25)
        angles = [math.degrees(item["angle"]) for item in candidates]
        self.assertTrue(angles)
        self.assertTrue(all(abs(a/15 - round(a/15)) < 1e-8 for a in angles))
        self.assertNotIn(7.5, angles)
        self.assertEqual(next(c for c in candidates if c["angle"] == 0)["first_atomic_action"],
                         "MOVE_FORWARD")
        self.assertEqual(next(c for c in candidates if c["angle"] > 0)["first_atomic_action"],
                         "TURN_LEFT")
        self.assertTrue(all(c["requires_post_turn_recheck"] for c in candidates
                            if c["angle"] != 0))
        self.assertEqual(next(c for c in candidates if c["angle"] == 0)["distance"], 0.5)

    def test_goal_bearings_are_quantized_before_geometry(self):
        g = FakeGeometry()
        values = generate_waypoints(g, 1.0, angles=[-22.5, -7.5, 7.5, 22.5])
        # Halfway ties may be quantized either way, but all returned bearings
        # must be expressible by real 15-degree TURN actions.
        self.assertTrue(all(abs(math.degrees(c["angle"])/15 -
                                round(math.degrees(c["angle"])/15)) < 1e-8 for c in values))

    def test_descent_intent_matches_planner_classifier(self):
        semantic = dict(stair_direction="down")
        grounding = dict(grounding_id="g", target="stairs", semantic=semantic,
                         stair_direction="DOWN")
        estimate = dict(bearing_deg=15., x_right_m=-0.4, z_forward_m=1.0,
                        uncertainty_m=0.1)
        candidate = dict(angle=math.radians(15.), distance=0.5)
        relation = goal_relation(candidate, grounding, estimate,
                                 dict(action="Descent to the bottom of the stairs"),
                                 15)
        self.assertEqual(relation["direction_compatibility"], "CONFIRMED")

    def test_turn_rechecks_real_forward_bearing_before_motion(self):
        camera = SimpleNamespace(turn_angle=15, forward_step=.25)
        controller = LocalMotionController(camera)
        path = dict(id="path_test", angle=math.radians(15), distance=.5,
                    source=dict(subgoal=0, pose=controller.pose()))
        g = FakeGeometry()
        runtime = SimpleNamespace(
            memory=SimpleNamespace(task=SimpleNamespace(current_subgoal_index=0),
                                   note_validation=Mock()),
            refresh=Mock(),
            observation=SimpleNamespace(captured_at=time.monotonic()),
            config=dict(max_depth_age_seconds=3),
            geometry=Mock(return_value=g),
            source=lambda: dict(observation_id="obs_now"))
        first = controller.validate(runtime, path)
        self.assertEqual(first["status"], "OK")
        self.assertEqual(first["validation"]["next_atomic_action"], "TURN_LEFT")
        self.assertTrue(first["validation"]["requires_post_turn_recheck"])
        g.diagnose.assert_not_called()
        g.assess_step.assert_not_called()

        controller.target = tuple(first["original_target"])
        controller.accept_feedback("TURN_LEFT", None)
        second = controller.validate(runtime, path, resume=True)
        self.assertEqual(second["status"], "OK")
        self.assertEqual(second["validation"]["next_atomic_action"], "MOVE_FORWARD")
        self.assertEqual(second["validation"]["validated_forward_bearing_deg"], 0)
        g.assess_step.assert_called_once_with()
        g.diagnose.assert_called_once_with(0, .25)

    def test_unverified_motion_invalidates_pending_target_not_a_collision(self):
        executor = ToolExecutor.__new__(ToolExecutor)
        path = dict(id="p0")
        endpoint = (.1, .5)
        executor.motion_controller = SimpleNamespace(x=0, z=0, target=endpoint)
        executor.memory = SimpleNamespace(
            current=dict(observation_id="obs_now"),
            active_attempt=dict(interruptions=[]),
            finish_attempt=Mock())
        executor.executable_segment = None
        outcome = executor._navigation_feedback(
            path, dict(original_target=endpoint), "REQUIRES_ATTENTION",
            "motion_unverified_reanchor_current_rgbd")
        self.assertIsNone(executor.motion_controller.target)
        self.assertEqual(outcome["status"], "REQUIRES_ATTENTION")
        self.assertEqual(path["execution_status"], "REQUIRES_ATTENTION")
        executor.memory.finish_attempt.assert_called_once()


if __name__ == "__main__":
    unittest.main()
