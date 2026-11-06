"""Keep role prompts consistent with validated tool contracts and execution authority."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from havln_agent.navigation_agent import NavigationAgent
from havln_agent.api_client import APIError
from havln_agent.task_planner import TaskPlanner
from havln_agent import perception, grounding


class CapturingClient:
    def __init__(self, response=None):
        self.response = response if response is not None else {}

    def complete(self, role, purpose, system, context, schema, **kwargs):
        self.role, self.purpose = role, purpose
        self.system, self.context, self.schema = system, context, schema
        self.system_history = getattr(self, 'system_history', []) + [system]
        self.kwargs = kwargs
        return self.response


class PromptContractTests(unittest.TestCase):
    def test_navigation_prompt_delegates_full_RGBD_and_verification(self):
        client=CapturingClient(dict(tool='observe_for_goal',args=dict(
            objective='Has the robot reached the stair bottom?',
            target_types=['stairs'],mode='VERIFY')))
        specs=dict(observe_for_goal=dict(description='VLM observation',
            permission='NAVIGATION',parameters=dict(type='object',
                properties={},required=[],additionalProperties=True)))
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value=specs),
            memory=SimpleNamespace(keyframes={}))
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value=dict(goal=dict(target='stairs'))):
            NavigationAgent(runtime).decide()
        self.assertIn('mode=VERIFY',client.system)
        self.assertIn('real',client.system.lower())
        self.assertIn('CONFIRMED',client.system)
        self.assertIn('BLOCKED',client.system)
        self.assertIn('MODE: OBSERVE',client.system)
        self.assertNotIn('verify_subgoal',client.system)
        self.assertEqual(client.role,'navigation_agent')

    def test_observation_stagnation_requests_model_authored_direction_without_forced_motion(self):
        client=CapturingClient(dict(tool='navigate_relative',
            args=dict(turn_deg=30,distance_m=.5)))
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value={'navigate_relative':{
                'parameters':dict(type='object',properties={},
                                  required=[],additionalProperties=False),
                'description':'model chooses a short bearing',
                'permission':'NAVIGATION'}}),
            memory=SimpleNamespace(keyframes={}),event=Mock())
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value=dict(observation_stagnation=dict(
                       intervention='DIRECTIONAL_ACTION_DECISION_REQUIRED',
                       same_stance_observation_calls=10,
                       low_information_calls=9))):
            NavigationAgent(runtime).decide()
        self.assertIn('OBSERVATION-LOOP DECISION CHECKPOINT',client.system)
        self.assertIn('YOU choose the direction',client.system)
        self.assertIn('not',client.system.lower())
        self.assertIn('safety',client.system.lower())
        self.assertIn('navigate_relative',client.system)
        runtime.event.assert_called_once()
        self.assertFalse(runtime.event.call_args.kwargs['physically_forced'])
        self.assertEqual(client.role,'navigation_agent')

    def test_observation_intervention_is_one_shot_per_stance(self):
        client=CapturingClient(dict(tool='navigate_relative',args=dict(
            turn_deg=0,distance_m=.25)))
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value=dict(navigate_relative=dict(
                description='move',permission='NAVIGATION',
                parameters=dict(type='object',properties={},
                    required=[],additionalProperties=False)))),
            memory=SimpleNamespace(keyframes={}),event=Mock())
        def state():
            return dict(task_selection=dict(current_subgoal_index=0),
                pose=dict(motion_epoch=0,place_id='P0',
                    position_xz_m_est=[0.,0.]),
                observation_stagnation=dict(
                    intervention='BLOCKED_ACTIONABILITY_REASSESSMENT',
                    same_stance_observation_calls=16,
                    low_information_calls=15))
        with patch('havln_agent.navigation_agent.navigation_context',
                   side_effect=lambda _:state()):
            agent=NavigationAgent(runtime)
            agent.decide()
            agent.decide()
        self.assertIn('NO CONFIRMED SAMPLED BEARING',client.system_history[0])
        self.assertNotIn('NO CONFIRMED SAMPLED BEARING',client.system_history[1])
        self.assertEqual(runtime.event.call_count,1)
        runtime.plan_revision=2
        with patch('havln_agent.navigation_agent.navigation_context',
                   side_effect=lambda _:state()):
            agent.decide()
        self.assertIn('NO CONFIRMED SAMPLED BEARING',client.system_history[-1])
        self.assertEqual(runtime.event.call_count,2)

    def test_panorama_preserves_all_strategic_choices_and_only_advises_once(self):
        observing=dict(description='VLM full visual task',permission='NAVIGATION',
            parameters=dict(type='object',properties={},required=[],
                            additionalProperties=True))
        moving=dict(description='Model-directed finite motion',permission='NAVIGATION',
            parameters=dict(type='object',properties={},required=[],
                            additionalProperties=True))
        terminal=dict(description='End task',permission='TERMINAL',
            parameters=dict(type='object',properties={},required=[],
                            additionalProperties=True))
        client=CapturingClient(dict(tool='observe_for_goal',
            args=dict(mode='VERIFY',objective='Is destination reached?',
                      target_types=['landing'])))
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value=dict(
                observe_for_goal=observing,navigate_relative=moving,
                request_finish=terminal)),
            memory=SimpleNamespace(keyframes={},episode_id='ep-a',
                task=SimpleNamespace(instruction='Go down stairs')),
            plan_revision=1,event=Mock())
        state=dict(task_selection=dict(current_subgoal_index=0),
            pose=dict(motion_epoch=0,place_id='P0',heading_deg=0,
                      position_xz_m_est=[0.,0.]),
            panoramic_observation=dict(all_azimuths_observed=True,
                                      covered_yaw_bins=24),
            observation_stagnation=dict(repeated_low_information=False))
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value=state):
            agent=NavigationAgent(runtime)
            first=agent.decide()
            self.assertEqual(first['tool'],'observe_for_goal')
            self.assertIn('ONE-TIME COMPLETE-VIEW REVIEW',client.system)
            self.assertEqual(client.purpose,'navigation_decision')
            tools=[b['properties']['tool']['const'] for b in
                   client.schema['oneOf']]
            self.assertIn('observe_for_goal',tools)
            self.assertIn('request_finish',tools)
            self.assertIn('navigate_relative',tools)
            self.assertEqual(runtime.event.call_count,1)
            self.assertFalse(runtime.event.call_args.kwargs['physically_forced'])
            agent.decide()
            self.assertNotIn('ONE-TIME COMPLETE-VIEW REVIEW',client.system)
            self.assertEqual(runtime.event.call_count,1)
            runtime.plan_revision=2
            agent.decide()
            self.assertIn('ONE-TIME COMPLETE-VIEW REVIEW',client.system)
            self.assertEqual(runtime.event.call_count,2)

    def test_panorama_does_not_reexpose_legacy_turn_or_geometry(self):
        client=CapturingClient(dict(tool='navigate_relative',
                                    args=dict(turn_deg=30,distance_m=.25)))
        camera=SimpleNamespace(turn_angle=15,tilt_angle=30)
        from havln_agent.contracts import registry
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value=registry(camera,3)),
            memory=SimpleNamespace(keyframes={},episode_id='ep-a',
                task=SimpleNamespace(instruction='Find the exit')),
            plan_revision=1,event=Mock())
        state=dict(panoramic_observation=dict(all_azimuths_observed=True),
            task_selection=dict(current_subgoal_index=0),
            pose=dict(motion_epoch=0,place_id='P0',heading_deg=0,
                      position_xz_m_est=[0,0]))
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value=state):
            NavigationAgent(runtime).decide()
        for tool in ('turn_and_observe','inspect_geometry','verify_subgoal'):
            with self.assertRaises(APIError):
                client.kwargs['validate'](dict(tool=tool,args={}))

    def test_orientation_only_goal_offers_observation_verification(self):
        client=CapturingClient(dict(tool='observe_for_goal',args=dict(
            mode='VERIFY',objective='Is the robot facing the requested direction?',
            target_types=['orientation'])))
        spec=dict(description='observation',permission='NAVIGATION',
            parameters=dict(type='object',properties={},
                required=[],additionalProperties=True))
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value=dict(
                observe_for_goal=spec,
                navigate_relative=dict(spec,description='move'))),
            memory=SimpleNamespace(keyframes={},episode_id='ep-a'),event=Mock())
        state=dict(orientation=dict(requirement='turn_left'),
            task_selection=dict(current_subgoal_index=0),
            pose=dict(motion_epoch=0,position_xz_m_est=[0,0]))
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value=state):
            NavigationAgent(runtime).decide()
        self.assertIn('VERIFY',client.system)
        self.assertEqual(len(client.schema['oneOf']),2)
        runtime.event.assert_not_called()

    def test_navigation_prompt_explains_continuous_finite_motion(self):
        client=CapturingClient(dict(tool='navigate_local',
            args=dict(waypoint_id='direct_0040')))
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value={'navigate_local':{
                'parameters':dict(type='object',properties={},
                    required=[],additionalProperties=False),
                'description':'continuous finite waypoint','permission':'NAVIGATION'}}),
            memory=SimpleNamespace(keyframes={}),event=Mock())
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value=dict(observation_stagnation=dict(
                       repeated_low_information=False))):
            NavigationAgent(runtime).decide()
        self.assertNotIn('max_atomic_steps',client.system)
        self.assertIn('finite',client.system)
        self.assertNotIn('multiple bounded navigate_local batches',client.system)
        self.assertNotIn('MOVEMENT CONTINUITY',client.system)
        runtime.event.assert_not_called()

    def test_normal_observation_prompt_remains_original_baseline(self):
        client=CapturingClient(dict(tool='inspect_geometry',args={}))
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value={'inspect_geometry':{
                'parameters':dict(type='object',properties={},
                                  required=[],additionalProperties=False),
                'description':'measure depth', 'permission':'READ_ONLY'}}),
            memory=SimpleNamespace(keyframes={}))
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value=dict(observation_stagnation=dict(
                       repeated_low_information=False))):
            NavigationAgent(runtime).decide()
        self.assertNotIn('OBSERVATION-LOOP DECISION CHECKPOINT',client.system)
        self.assertIn('Observation Agent',client.system)

    def test_task_review_never_asks_model_to_generate_predicates(self):
        client = CapturingClient(dict(intent="continue", subgoal="Descend stairs"))
        goal = dict(action="Descent to the bottom of the stairs", target="landing",
                    landmark="stairs", success_criteria=["target_nearby"])
        task = SimpleNamespace(current_subgoal_index=0, subgoals=[goal])
        runtime = SimpleNamespace(client=client, memory=SimpleNamespace(task=task))
        with patch("havln_agent.task_planner.planner_context", return_value={}):
            value = TaskPlanner(runtime).review()
        self.assertIn("Legacy success_criteria", client.system)
        self.assertIn("completion_description", client.system)
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
