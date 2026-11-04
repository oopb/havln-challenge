"""NavigationAgent's model-facing tool contract is small, context-aware, and safe."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

from jsonschema import Draft7Validator

from havln_agent.api_client import APIError
from havln_agent.contracts import registry, navigation_schema
from havln_agent.navigation_agent import NavigationAgent
from havln_agent.tool_menu import (
    CORE_NAVIGATION_TOOLS,CONTEXTUAL_TOOLS,VLM_INTERNAL_TOOLS,
    navigation_tool_menu,contextual_tool_names)


def full_registry():
    return registry(SimpleNamespace(turn_angle=15,tilt_angle=30),3)


class NavigationToolMenuTests(unittest.TestCase):
    def test_full_executor_registry_kept_but_default_model_sees_nine(self):
        tools=full_registry()
        self.assertEqual(len(tools),20)
        self.assertEqual(len(CORE_NAVIGATION_TOOLS),9)
        offered=navigation_tool_menu(tools,{})
        self.assertEqual(tuple(offered),CORE_NAVIGATION_TOOLS)
        self.assertNotIn('analyze_scene',offered)
        self.assertNotIn('assess_entry',offered)
        self.assertNotIn('turn_and_observe',offered)
        self.assertNotIn('navigate_to_pixel',offered)
        self.assertEqual(len(navigation_schema(offered)['oneOf']),9)
        self.assertEqual(len(navigation_schema(tools)['oneOf']),20)
        self.assertIn('analyze_scene',tools)
        self.assertIn('turn_and_observe',tools)

    def test_vlm_owns_camera_search_without_mutating_legacy_schema(self):
        tools=full_registry()
        first=tools['observe_for_goal']['parameters']
        self.assertIn('direction',first['properties'])
        offered=navigation_tool_menu(tools,{})
        compact=offered['observe_for_goal']['parameters']
        self.assertNotIn('direction',compact['properties'])
        self.assertNotIn('checkpoint_degrees',compact['properties'])
        self.assertNotIn('look',compact['properties'])
        self.assertTrue(Draft7Validator(compact).is_valid(dict(
            objective='Find accessible stair entry',
            target_types=['stairs','stair_entry'],sweep_degrees=180)))
        self.assertFalse(Draft7Validator(compact).is_valid(dict(
            objective='Find stairs',target_types=['stairs'],
            direction='left')))
        self.assertIn('direction',first['properties'])
        self.assertIn('checkpoint_degrees',first['properties'])

    def test_camera_turn_is_only_an_orientation_task_or_legacy_executor(self):
        tools=full_registry()
        off=navigation_tool_menu(tools,{'last_tool':dict(
            information_options=['turn_and_observe','analyze_scene'])})
        self.assertNotIn('turn_and_observe',off)
        self.assertNotIn('analyze_scene',off)
        self.assertIn('turn_and_observe',
            navigation_tool_menu(tools,{},orientation=True))
        self.assertIn('turn_and_observe',tools)

    def test_fresh_vlm_candidate_enables_pixel_motion_but_not_historical_box(self):
        tools=full_registry()
        view={'current_view':dict(observation_id='obs_18'),
              'last_tool':dict(tool='observe_for_goal',
                  observation_task=dict(best_candidate=dict(
                      observation_id='obs_18',box=[.2,.3,.8,.9])))}
        self.assertIn('navigate_to_pixel',navigation_tool_menu(tools,view))
        view['last_tool']['observation_task']['best_candidate']['observation_id']='obs_17'
        self.assertNotIn('navigate_to_pixel',navigation_tool_menu(tools,view))

    def test_active_route_has_cancel_and_after_explicit_geometry_has_preview(self):
        tools=full_registry()
        state=dict(local_navigation=dict(status='IN_PROGRESS',waypoint_id='w2'),
            last_tool=dict(tool='inspect_geometry',status='OK'))
        offered=navigation_tool_menu(tools,state)
        self.assertIn('cancel_local_navigation',offered)
        self.assertIn('get_waypoint_candidates',offered)
        self.assertIn('navigate_local',offered)
        self.assertIn('navigate_relative',offered)
        # A fixed candidate route is not authorization: we only advertise it.
        self.assertEqual(offered['get_waypoint_candidates']['permission'],'READ_ONLY')
        self.assertEqual(offered['cancel_local_navigation']['permission'],'READ_ONLY')

    def test_ambiguous_historical_visual_memory_opens_retrieval_only_when_useful(self):
        tools=full_registry()
        self.assertNotIn('query_spatial_memory',navigation_tool_menu(tools,{}))
        state=dict(navigation_memory=dict(visual_target_memory=[
            dict(status='CUE_ONLY',selected_observation_id='obs_12')]))
        exposed=navigation_tool_menu(tools,state)
        self.assertIn('recall_observation',exposed)
        self.assertIn('query_spatial_memory',exposed)
        self.assertNotIn('query_task_memory',exposed)
        self.assertNotIn('read_memory_record',exposed)
        state=dict(last_tool=dict(tool='query_spatial_memory',
            memory=dict(routes=[dict(id='r1')])))
        self.assertIn('read_memory_record',navigation_tool_menu(tools,state))

    def test_rejected_verification_can_request_missing_evidence_tools(self):
        tools=full_registry()
        state=dict(last_tool=dict(tool='verify_subgoal',
            status='REJECTED',
            information_options=['analyze_scene','query_task_memory',
                                 'query_spatial_memory','compare_views']))
        offered=navigation_tool_menu(tools,state)
        self.assertIn('compare_views',offered)
        self.assertIn('query_task_memory',offered)
        self.assertIn('query_spatial_memory',offered)
        self.assertNotIn('analyze_scene',offered)
        self.assertIn('observe_for_goal',offered)

    def test_one_shot_panorama_retains_only_model_authored_motion(self):
        tools=full_registry()
        offered=navigation_tool_menu(tools,{'last_tool':dict(
            information_options=list(CONTEXTUAL_TOOLS))},
            full_circle_commit=True)
        self.assertEqual(list(offered),['navigate_relative'])

    def test_actual_navigation_request_has_nine_and_rejects_hidden_tool(self):
        specs=full_registry()
        client=Mock()
        client.complete.return_value=dict(tool='inspect_geometry',args={})
        rt=SimpleNamespace(client=client,registry=None,
            make_registry=Mock(return_value=specs),
            memory=SimpleNamespace(keyframes={},
                task=SimpleNamespace(instruction='Descend stairs')))
        state=dict(goal=dict(action='descend',target='stairs'),
            pose=dict(heading_deg=0),current_view=dict(
                observation_id='obs_1'),
            observation_stagnation=dict(repeated_low_information=False))
        with patch('havln_agent.navigation_agent.navigation_context',
                   return_value=state):
            result=NavigationAgent(rt).decide()
        self.assertEqual(result['tool'],'inspect_geometry')
        args,kwargs=client.complete.call_args
        exposed=[branch['properties']['tool']['const'] for branch in args[4]['oneOf']]
        self.assertEqual(exposed,list(CORE_NAVIGATION_TOOLS))
        self.assertEqual(len(rt.registry),20)
        self.assertNotIn('turn_and_observe',kwargs['protocol'])
        self.assertNotIn('query_spatial_memory',kwargs['protocol'])
        with self.assertRaises(APIError):
            kwargs['validate'](dict(tool='turn_and_observe',args=dict(
                relative_turn_deg=15,look='none')))
        kwargs['validate'](dict(tool='inspect_geometry',args={}))


if __name__=='__main__':
    unittest.main()
