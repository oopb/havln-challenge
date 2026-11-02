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
    def test_navigation_prompt_separates_actionability_from_completion_and_observation(self):
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
            "local_navigation": {"status": "IN_PROGRESS", "waypoint_id": "p0"},
            "decision_feedback": {"turn_only_move_does_not_count_as_forward": True}}):
            NavigationAgent(runtime).decide()
        self.assertEqual(client.role, "navigation_agent")
        for phrase in (
            "ACTIONABILITY", "COMPLETION", "INFORMATION VALUE",
            "navigation_mode=OBSERVE", "DECIDE", "SAME waypoint_id",
            "navigate_relative", "navigate_to_pixel", "max_atomic_steps=1",
            "PARTIAL", "NEEDS_VIEW", "REQUIRES_ATTENTION", "BLOCKED",
            "decision_feedback", "TARGET_REACHED", "live",
            "Verify milestones", "Never request_finish(success)"):
            self.assertIn(phrase.lower(), client.system.lower(), phrase)
        self.assertIn('tool_action_limit',client.system)
        self.assertIn('different pitch alone', ' '.join(client.system.lower().split()))
        self.assertIn('same doorway',client.system)
        self.assertNotIn('EXECUTION-FOCUS REVIEW',client.system)
        self.assertNotIn('MODE: OBSERVE',client.system)
        self.assertNotIn('LOOK_DOWN and assess/ground the entrance',client.system)
        self.assertTrue(client.kwargs["retry_safe"])

    def test_navigation_strategy_does_not_force_scans_or_forward(self):
        client=CapturingClient(dict(tool='inspect_geometry',args={}))
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value={'inspect_geometry':dict(
                parameters=dict(type='object',properties={},
                                required=[],additionalProperties=False),
                description='Inspect live geometry',permission='READ_ONLY')}),
            memory=SimpleNamespace(keyframes={}))
        with patch('havln_agent.navigation_agent.navigation_context',
            return_value=dict(exploration_feedback=dict(
                scan_dominated_without_translation=True),
                target_identity_review=dict(candidates=[
                    dict(observation_id='o7',target='distant doorway')]))):
            NavigationAgent(runtime).decide()
        self.assertIn('You own observation choices',client.system)
        self.assertIn('do not just repeat',client.system)
        self.assertIn('no forced action',str(client.context).lower() if
                      'decision_feedback' in client.context else 'no forced action')
        self.assertNotIn('you MUST move forward',client.system)
        self.assertNotIn('EXECUTION-FOCUS REVIEW',client.system)

    def test_task_review_only_updates_semantic_description(self):
        client=CapturingClient(dict(intent='continue',subgoal='Descend stairs'))
        goal=dict(action='Descend to the bottom of the stairs',
                  target='landing',landmark='stairs',source_text='Go downstairs.')
        task=SimpleNamespace(current_subgoal_index=0,subgoals=[goal])
        runtime=SimpleNamespace(client=client,memory=SimpleNamespace(task=task))
        with patch('havln_agent.task_planner.planner_context',return_value={}):
            value=TaskPlanner(runtime).review()
        self.assertIn('Do not generate',client.system)
        self.assertIn('success criteria',client.system)
        self.assertIn('current semantic milestone',client.system.lower())
        self.assertNotIn('success_criteria',value)
        self.assertNotIn('required_success_criteria',client.context)

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
