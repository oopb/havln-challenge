"""End-of-subgoal visual references are an observation question, not verification."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jsonschema import Draft7Validator

from havln_agent.navigation_agent import NavigationAgent, OBSERVE_PROMPT
from havln_agent.contracts import registry, navigation_schema
from havln_agent.visual_perception import VisualPerception, SCAN_GUIDANCE_SCHEMA, SCAN_GUIDANCE_PROMPT
from havln_agent.active_observation import ObservationController
from havln_agent.contexts import tool_summary, compact_navigation_context
from test_active_observation import FakeRuntime
from test_vlm_led_observation import VLMGuide, guidance
from test_role_prompt_contracts import CapturingClient


class VisualEndpointReferenceTests(unittest.TestCase):
    def test_navigation_delegates_endpoint_question_using_original_tool_contract(self):
        camera=SimpleNamespace(turn_angle=15,tilt_angle=30)
        specs=registry(camera,3)
        objective=('Inspect the lower staircase: do descending treads visibly '
                   'end at a flat landing or floor, or is the floor distant?')
        answer=dict(tool='observe_for_goal',
            args=dict(objective=objective,
                      target_types=['stairs','landing','floor'],
                      sweep_degrees=90))
        client=CapturingClient(answer)
        runtime=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value=specs),
            memory=SimpleNamespace(keyframes={},
                task=SimpleNamespace(instruction='Descend stairs then turn right')))
        state=dict(goal=dict(description='Descend the stairs'),
            progress_review=dict(next_subgoal=dict(action='Turn right',target='hallway')),
            task_selection=dict(current_subgoal_index=0),
            pose=dict(heading_deg=15),
            current_view=dict(observation_id='obs_0032'))
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value=state):
            result=NavigationAgent(runtime).decide()
        self.assertEqual(result,answer)
        self.assertIn('visible scene cues',client.system)
        self.assertIn('AT OR NEAR THE END',client.system)
        self.assertIn('NOT proof of arrival',client.system)
        self.assertIn('if an end cue is not visually',client.system)
        self.assertTrue(Draft7Validator(navigation_schema(specs)).is_valid(result))
        self.assertEqual(set(result['args']),{'objective','target_types','sweep_degrees'})
        self.assertNotIn('completion_visual_expectation',
                         specs['observe_for_goal']['parameters']['properties'])

    def test_vlm_receives_the_end_cue_in_objective_but_no_completion_contract(self):
        task=SimpleNamespace(instruction='Descend stairs and turn right',
            current_subgoal_index=0,
            subgoals=[dict(action='Descend',target='stairs')])
        vp=VisualPerception(SimpleNamespace(memory=SimpleNamespace(task=task),client=None))
        objective=('Look for descending treads transitioning to a flat landing; '
                   'report if the floor is only seen in the distance.')
        src=dict(observation_id='obs_0048',
                 pose=dict(heading=math.radians(90),pitch_deg=-30))
        response=dict(guidance(),target_type='wooden floor',
            evidence='Flat wooden surface, no visible tread-floor transition',
            uncertainty='Cannot tell where stairs end from this frame')
        with patch('havln_agent.visual_perception.image_url',
                   return_value='data:image/jpeg;base64,AA'):
            with patch.object(vp,'complete',return_value=response) as call:
                observed=vp.guide_scan_view(
                    SimpleNamespace(),src,objective,['stairs','landing','floor'],[],90)
        args,kwargs=call.call_args
        self.assertEqual(args[3]['objective'],objective)
        self.assertEqual(args[3]['observation_id'],'obs_0048')
        self.assertIn('ONLY as',args[2])
        self.assertIn('do not infer presence',args[2])
        self.assertIn('do not',SCAN_GUIDANCE_PROMPT)
        self.assertEqual(kwargs['result_schema'],SCAN_GUIDANCE_SCHEMA)
        self.assertNotIn('subgoal_completed',SCAN_GUIDANCE_SCHEMA['properties'])
        self.assertEqual(observed['evidence'],response['evidence'])
        self.assertEqual(call.call_count,1)

    def test_missing_visual_end_cue_does_not_block_observation_delivery(self):
        runtime=FakeRuntime([])
        runtime.visual_perception=VLMGuide([
            dict(guidance('finish'),target_type='stone wall',
                 evidence='Only a nearby wall is visible',
                 uncertainty='No stair-to-floor transition visible')])
        result=ObservationController(runtime).scan_vlm(
            'Look for the end of the stairs, if visible',
            ['stairs','landing'],sweep_degrees=90)
        self.assertEqual(result['outcome'],'VLM_SEARCH_FINISHED')
        self.assertIsNone(result['best_candidate'])
        summary=tool_summary(dict(result,tool='observe_for_goal'))
        report=summary['observation_task']['visual_reports'][0]
        self.assertEqual(report['evidence'],'Only a nearby wall is visible')
        self.assertEqual(report['observation_id'],'obs_0000')
        self.assertEqual(runtime.step_calls,[])
        state=dict(latest_observation_task=dict(mode='vlm_led',
            objective=result['objective'],outcome=result['outcome'],
            visual_reports=result['visual_reports']))
        compact=compact_navigation_context(state)
        self.assertEqual(compact['latest_observation_task']['visual_reports'][0]
                         ['uncertainty'],'No stair-to-floor transition visible')

    def test_non_stair_case_has_no_assumed_completion_or_extra_required_schema(self):
        self.assertIn('when passing a door',OBSERVE_PROMPT)
        self.assertIn('if an end cue is not visually',OBSERVE_PROMPT)
        self.assertIn('not preconditions for',OBSERVE_PROMPT)
        self.assertNotIn('subgoal_completed',SCAN_GUIDANCE_SCHEMA['required'])
        self.assertNotIn('completion_visual_expectation',
                         SCAN_GUIDANCE_SCHEMA['required'])


if __name__=='__main__':
    unittest.main()
