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
        client=CapturingClient(dict(tool='inspect_geometry',args={}))
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value={'inspect_geometry':{
                'parameters':dict(type='object',properties={},
                    required=[],additionalProperties=False),
                'description':'measure geometry','permission':'READ_ONLY'}}),
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
                   side_effect=lambda _: state()):
            agent=NavigationAgent(runtime)
            agent.decide()
            agent.decide()
        self.assertIn('NO CONFIRMED SAMPLED BEARING',client.system_history[0])
        self.assertNotIn('NO CONFIRMED SAMPLED BEARING',client.system_history[1])
        self.assertEqual(runtime.event.call_count,1)
        self.assertIn('other headings are UNMEASURED', client.system_history[0])
        self.assertIn('DO NOT take max_atomic_steps',client.system)
        # Same stance/subgoal but NEW plan revision must permit a new
        # checkpoint, because the semantic route requirement may have changed.
        runtime.plan_revision=2
        with patch('havln_agent.navigation_agent.navigation_context',
                   side_effect=lambda _: state()):
            agent.decide()
        self.assertIn('NO CONFIRMED SAMPLED BEARING',client.system_history[-1])
        self.assertEqual(runtime.event.call_count,2)

    @staticmethod
    def _panorama_fixture():
        headings=[0,90,180,270]
        viewpoints=[
            dict(observation_id='obs%d' % i,
                 absolute_heading_deg=a,
                 relative_turn_deg_from_current=((a+180)%360)-180,
                 pitch_deg=0,
                 historical_front_state='UNMEASURED')
            for i,a in enumerate(headings)]
        visual=SimpleNamespace(panorama_direction_evidence=Mock(return_value=dict(
            selected_observation_id='obs0',
            visible_evidence='a hallway in obs0',
            visual_cues=['visible entry'],uncertainty='medium',
            alternative_observation_ids=[])))
        return viewpoints,visual,Mock(
            side_effect=lambda ident:(SimpleNamespace(rgb=None),
                                      dict(observation_id=ident)))

    def test_full_panorama_requires_model_selected_finite_forward_attempt_once(self):
        moving=dict(description='Model selects bearing and distance',
            permission='NAVIGATION',
            parameters=dict(type='object',
                properties=dict(turn_deg=dict(type='number',minimum=-180,maximum=180),
                                distance_m=dict(type='number',minimum=.25,maximum=3.)),
                required=['turn_deg','distance_m'],additionalProperties=False))
        observation=dict(description='Camera-only turn',permission='NAVIGATION',
            parameters=dict(type='object',properties={},
                required=[],additionalProperties=False))
        client=CapturingClient(dict(tool='navigate_relative',
            args=dict(turn_deg=-60,distance_m=.5)))
        viewpoints,visual,frame_reader=self._panorama_fixture()
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value=dict(
                navigate_relative=moving,turn_and_observe=observation)),
            memory=SimpleNamespace(keyframes={},episode_id='ep-a',
                                   task=SimpleNamespace(instruction='Find the entry')),
            visual_perception=visual,_vision_frame=frame_reader,
            plan_revision=1,event=Mock())
        state=dict(task_selection=dict(current_subgoal_index=0),
            pose=dict(motion_epoch=0,place_id='P0',heading_deg=0,
                      position_xz_m_est=[0.,0.]),
            panoramic_observation=dict(all_azimuths_observed=True,
                                      observed_viewpoints=viewpoints,
                                      covered_yaw_bins=24),
            observation_stagnation=dict(repeated_low_information=True,
                intervention='BLOCKED_ACTIONABILITY_REASSESSMENT',
                same_stance_observation_calls=24,low_information_calls=8))
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value=state):
            agent=NavigationAgent(runtime)
            selected=agent.decide()
            self.assertEqual(selected['args']['turn_deg'],-60)
            self.assertIn('ONE-TIME PANORAMIC DIRECTION DECISION',client.system_history[0])
            self.assertNotIn('MODE: OBSERVE',client.system_history[0])
            self.assertEqual(client.purpose,'panorama_navigation_commit')
            self.assertEqual(len(client.schema['oneOf']),1)
            self.assertEqual(client.schema['oneOf'][0]['properties']['tool']['const'],
                             'navigate_relative')
            self.assertEqual(client.context['panorama_commitment']['allowed_tool'],
                             'navigate_relative')
            self.assertEqual(runtime.event.call_count,2)
            self.assertEqual([call.args[0] for call in runtime.event.call_args_list],
                             ['panorama_visual_memory_reviewed','panorama_direction_committed'])
            self.assertFalse(runtime.event.call_args.kwargs['physically_forced'])
            visual.panorama_direction_evidence.assert_called_once()
            agent.decide()
            self.assertNotIn('ONE-TIME PANORAMIC DIRECTION DECISION',client.system_history[1])
            self.assertEqual(len(client.schema['oneOf']),2)
            self.assertEqual(runtime.event.call_count,2)
            # Re-arm on a genuine new task/plan at the same location, not
            # on another model turn with identical physical and task evidence.
            runtime.plan_revision=2
            agent.decide()
            self.assertIn('ONE-TIME PANORAMIC DIRECTION DECISION',client.system_history[-1])
            self.assertEqual(runtime.event.call_count,4)
            self.assertEqual(visual.panorama_direction_evidence.call_count,2)

    def test_panorama_rejects_cheap_model_turn_loop_without_consuming_checkpoint(self):
        schema=dict(type='object',properties=dict(
            turn_deg=dict(type='number',minimum=-180,maximum=180),
            distance_m=dict(type='number',minimum=.25,maximum=3)),
            required=['turn_deg','distance_m'],additionalProperties=False)
        client=CapturingClient(dict(tool='turn_and_observe',
            args=dict(relative_turn_deg=60,look='none')))
        viewpoints,visual,frame_reader=self._panorama_fixture()
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value=dict(
                navigate_relative=dict(description='MOVE',permission='NAVIGATION',
                    parameters=schema),
                turn_and_observe=dict(description='LOOK',permission='NAVIGATION',
                    parameters=dict(type='object',properties={},
                        additionalProperties=False)))),
            memory=SimpleNamespace(keyframes={},episode_id='ep-a',
                                   task=SimpleNamespace(instruction='Find a passage')),
            visual_perception=visual,_vision_frame=frame_reader,
            event=Mock())
        state=dict(panoramic_observation=dict(
                all_azimuths_observed=True,observed_viewpoints=viewpoints),
            task_selection=dict(current_subgoal_index=0),
            pose=dict(motion_epoch=0,place_id='P0',heading_deg=0,
                      position_xz_m_est=[0,0]))
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value=state):
            agent=NavigationAgent(runtime)
            with self.assertRaises(APIError):
                agent.decide()
            # The VLM review exists, but a bad NavigationAgent tool may not
            # mark the model-owned direction commitment as executed.
            self.assertEqual(runtime.event.call_count,1)
            client.response=dict(tool='navigate_relative',
                                 args=dict(turn_deg=30,distance_m=.25))
            agent.decide()
            self.assertEqual(runtime.event.call_count,2)
            visual.panorama_direction_evidence.assert_called_once()

    def test_orientation_only_goal_does_not_require_unrelated_translation(self):
        client=CapturingClient(dict(tool='verify_subgoal',
                                    args=dict(claim='subgoal_completed',evidence='turn',
                                              observation_ids=['obs1'])))
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value=dict(
                navigate_relative=dict(description='MOVE',permission='NAVIGATION',
                    parameters=dict(type='object',properties={},
                                    additionalProperties=False)),
                verify_subgoal=dict(description='Verify turn',permission='READ_ONLY',
                    parameters=dict(type='object',properties={},
                                    additionalProperties=False)))),
            memory=SimpleNamespace(keyframes={},episode_id='ep-a'),event=Mock())
        state=dict(panoramic_observation=dict(all_azimuths_observed=True),
                   orientation=dict(requirement='turn_left'),
                   task_selection=dict(current_subgoal_index=0),
                   pose=dict(motion_epoch=0,position_xz_m_est=[0,0]))
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value=state):
            NavigationAgent(runtime).decide()
        self.assertNotIn('FULL-360 OBSERVATION COMPLETE',client.system)
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
        self.assertIn('DO NOT take max_atomic_steps',client.system)
        self.assertIn('continuously',client.system)
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
        self.assertIn('Prefer a measured goal-approaching direction',client.system)

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
