"""Keep role prompts consistent with validated tool contracts and execution authority."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from havln_agent.navigation_agent import NavigationAgent
from havln_agent.task_planner import TaskPlanner
from havln_agent import perception, grounding


class CapturingClient:
    def __init__(self, response=None):
        self.response = response if response is not None else {}

    def complete(self, role, purpose, system, context, schema, **kwargs):
        self.role, self.purpose = role, purpose
        self.system, self.context, self.schema = system, context, schema
        self.kwargs = kwargs
        return self.response


class PromptContractTests(unittest.TestCase):
    def test_navigation_prompt_prioritizes_same_waypoint_and_real_safety(self):
        client = CapturingClient(dict(tool="inspect_geometry", args={}))
        runtime = SimpleNamespace(
            client=client, registry=None,
            make_registry=Mock(return_value={"inspect_geometry": {
                "parameters": {"type": "object", "properties": {},
                               "required": [], "additionalProperties": False},
                "description": "Current geometry", "permission": "READ_ONLY"}}),
            memory=SimpleNamespace(keyframes={}))
        with patch("havln_agent.navigation_agent.navigation_context", return_value={
            "candidate_waypoints": [{"id": "p0", "first_step": "EXECUTABLE"}],
            "local_navigation": {"status": "IN_PROGRESS", "waypoint_id": "p0"}}):
            NavigationAgent(runtime).decide()
        self.assertEqual(client.role, "navigation_agent")
        self.assertIn("SAME waypoint_id", client.system)
        self.assertIn("grounding is PARTIAL or UNKNOWN", client.system)
        self.assertIn("navigate_relative", client.system)
        self.assertIn("navigate_to_pixel", client.system)
        self.assertIn("NOT a movement prohibition", client.system)
        self.assertIn("Never bypass a measured collision", client.system)
        self.assertIn("candidate_waypoints", client.system)
        self.assertIn("fresh RGB-D", client.system)
        self.assertIn("NEEDS_VIEW", client.system)
        self.assertIn("REQUIRES_ATTENTION", client.system)
        self.assertIn("never request_finish", client.system)
        self.assertIn("MODE: OBSERVE", client.system)
        self.assertIn("navigation_memory", client.system)
        self.assertNotIn("LOOK_DOWN and assess/ground the entrance", client.system)
        self.assertTrue(client.kwargs["retry_safe"])

    def test_task_review_never_asks_model_to_generate_predicates(self):
        client = CapturingClient(dict(intent="continue", subgoal="Descend stairs"))
        goal = dict(action="Descent to the bottom of the stairs", target="landing",
                    landmark="stairs", success_criteria=["target_nearby"])
        task = SimpleNamespace(current_subgoal_index=0, subgoals=[goal])
        runtime = SimpleNamespace(client=client, memory=SimpleNamespace(task=task))
        with patch("havln_agent.task_planner.planner_context", return_value={}):
            value = TaskPlanner(runtime).review()
        self.assertIn("Do not generate success_criteria", client.system)
        self.assertIn("current semantic milestone", client.system.lower())
        self.assertEqual(value["success_criteria"], [
            "actual_descent_observed", "entry_confirmed", "final_arrival_visible",
            "target_nearby"])

    def test_scene_prompt_treats_stairs_as_approach_cues_not_authorization(self):
        scene = dict(description="Visible stair entry", landmarks=[],
                     humans=[], uncertain_humanoids=[], static_obstacles=[],
                     final_target_visible=False, final_target_evidence="",
                     uncertainty="partially obscured", instruction_evidence="", regions=[])
        client = CapturingClient(scene)
        observation = SimpleNamespace(rgb=None)
        with patch.object(perception, "image_url", return_value="data:image/png;base64,AA"):
            returned = perception.perceive(
                client, observation,
                {"mode": "scene", "question": "Where is the stair entry?",
                 "vision_contract": {"coordinate_space": "image_1000_xyxy"}},
                {"scope": "current", "observation_id": "obs_0001"})
        self.assertEqual(returned["description"], "Visible stair entry")
        self.assertIn("NOT a certified collision-free", client.system)
        self.assertIn("entire staircase", client.system)
        self.assertIn("one question", client.system.lower())

    def test_grounding_prompt_does_not_require_all_stairs_visible(self):
        import inspect
        text = inspect.getsource(grounding.ground)
        self.assertIn("even if other steps or the lower landing are occluded", text)
        self.assertIn("NOT reachability or permission to move", text)


if __name__ == "__main__":
    unittest.main()
