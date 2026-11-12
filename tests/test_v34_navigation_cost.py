"""v34 offline tests: aggressive navigation context plus compatible prefix cache.

The navigation decision maker keeps real RGB, original instruction, selected
scene goal, stair reference and physical safety. No external model calls.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

from havln_agent.api_client import APIClient, model_endpoints
from havln_agent.contexts import (
    compact_navigation_context, cost_optimized_navigation_context,
    _cost_optimized_projection)
from havln_agent.contracts import compact_navigation_protocol,registry
from havln_agent.model_angles import project_model_angles
from havln_agent.navigation_agent import MISSION_PROMPT
from test_api_parameter_compat import accepted, rejected, settings
from test_forward_preflight_v9 import camera
from test_v21_compact_navigation_context import large_request
from test_v29_phase2_route_reference import fixture,update
from havln_agent.mission_history import mission_evidence


def chars(value):
    return len(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')))


class Phase2EfficientNavigationContext(unittest.TestCase):
    def test_wire_reduction_does_not_mutate_full_records(self):
        full=large_request()
        original=copy.deepcopy(full)
        legacy=compact_navigation_context(full)
        result=cost_optimized_navigation_context(full)
        self.assertEqual(full,original)
        self.assertEqual(result['memory']['instruction'],full['memory']['instruction'])
        self.assertEqual(result['front_depth_preflight']['status'],
                         full['front_depth_preflight']['status'])
        self.assertLess(chars(result),chars(legacy)*.80)
        self.assertEqual(result['memory']['contract'],'mission_evidence_v1')
        self.assertNotIn('recent_forward_failures',result['memory'])

    def test_keep_complete_view_coverage_and_separate_failures(self):
        result=cost_optimized_navigation_context(large_request())
        mem=result['memory']
        a=mem['recent_views_at_current_stance']
        self.assertEqual(a['turn_actions_since_last_forward'],80)
        self.assertEqual(a['look_actions_since_last_forward'],5)
        self.assertEqual(a['repeated_same_RGBD_heading_pitch_views'],52)
        self.assertTrue(a['horizontal_360_coverage_observed'])
        self.assertEqual(a['full_coverage_by_pitch'][0][
            'observed_heading_bins_deg'],list(range(0,360,15)))
        self.assertEqual(len(mem['blocked_at_source_stances']),2)
        self.assertEqual(len(mem['stalled_forward_at_source_stances']),2)
        self.assertEqual(len(mem['observed_front_blocks']),2)
        self.assertEqual(mem['blocked_at_source_stances'][0]['kind'],
                         'BLOCKED_BEFORE_FORWARD')
        self.assertEqual(mem['stalled_forward_at_source_stances'][0]['kind'],
                         'STALLED_AFTER_FORWARD')
        self.assertEqual(mem['last_motion_decision']['atomic_actions_since_motion'],87)
        self.assertEqual(len(mem['candidate_route_hypotheses']),3)

    def test_critical_scene_goal_and_instruction_route_survive_projection(self):
        memory=fixture()
        memory.task.instruction=('Descend stairs, turn LEFT to enter the '
                                 'bedroom, observe a person reading.')
        memory.note_route_reference(update())
        memory.task.scene_goal='Inspect chosen bedroom; reader may move.'
        memory.task.scene_goal_source_observation_id='obs_0019'
        raw=mission_evidence(memory,memory.current['pose'])
        out=cost_optimized_navigation_context(dict(memory=raw))
        projected=project_model_angles(out,45)
        mem=projected['memory']
        self.assertEqual(mem['instruction'],memory.task.instruction)
        self.assertEqual(mem['progress']['scene_goal'],memory.task.scene_goal)
        self.assertEqual(mem['progress']['scene_goal_source_observation_id'],
                         'obs_0019')
        self.assertEqual(mem['instruction_route_references'][0][
            'source_observation_id'],'obs_0019')
        self.assertIn('DESCENT APPROACH',
                      mem['instruction_route_references'][0]['instruction_relation'])
        self.assertNotIn('source_heading_deg',json.dumps(projected))
        self.assertIn('source_view_offset_deg_from_current',
                      mem['instruction_route_references'][0])

    def test_protocol_remains_strict_and_action_first(self):
        protocol=compact_navigation_protocol(registry(camera(),128))
        self.assertIn('FINAL executable decision',protocol)
        self.assertIn('Put tool parameters directly inside args',protocol)
        self.assertIn('route_reference_update',protocol)
        self.assertIn('scene_goal',protocol)
        self.assertIn('turn_direction',protocol)
        self.assertIn('look=up/down/none',protocol)
        self.assertLess(len(protocol),1650)
        self.assertIn('favor a short model-chosen forward exploration',MISSION_PROMPT)

    def test_no_unnecessary_source_removal_when_input_has_no_photos(self):
        full=large_request()
        original=copy.deepcopy(full)
        projected=_cost_optimized_projection(full)
        self.assertEqual(full,original)
        self.assertEqual(projected['memory']['historical_frames'],
                         original['memory']['historical_frames'])


class Phase2PrefixCache(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp=tmp.name
        env=patch.dict('os.environ',{'COMPAT_TEST_KEY':'offline-only'})
        env.start()
        self.addCleanup(env.stop)

    def client(self,cache='auto',url='https://api.openai.com/v1'):
        conf=settings(url)
        conf['temperature']=None
        conf['prompt_cache']=cache
        c=APIClient({'navigation_agent':conf,'visual_perception':conf},self.tmp)
        c.begin_episode('offline-cache',20,Path(self.tmp)/'api.jsonl')
        return c

    @staticmethod
    def ask(c,goal='A'):
        return c.complete('navigation_agent','cache_test',
            'Stable instruction '+('route, preserve source ID; '*210),
            {'dynamic_scene_goal':goal},
            dict(type='object',properties=dict(ok=dict(type='boolean')),
                 required=['ok'],additionalProperties=False))

    def test_fixed_prefix_cache_boundary_and_reported_hits(self):
        c=self.client()
        bodies=[]
        def send(request,timeout,record):
            b=json.loads(request.data)
            bodies.append(b)
            reply=accepted()
            reply['usage']=dict(prompt_tokens=2500,completion_tokens=10,
                total_tokens=2510,
                prompt_tokens_details=dict(cached_tokens=1450))
            return reply
        c._request=Mock(side_effect=send)
        self.ask(c,'first')
        self.ask(c,'second')
        self.assertEqual(len(bodies),2)
        self.assertEqual(bodies[0]['prompt_cache_options'],
                         {'mode':'explicit','ttl':'30m'})
        system=bodies[0]['messages'][0]['content']
        self.assertEqual(system,bodies[1]['messages'][0]['content'])
        self.assertEqual(system[0]['prompt_cache_breakpoint'],{'mode':'explicit'})
        self.assertNotEqual(bodies[0]['messages'][1]['content'],
                            bodies[1]['messages'][1]['content'])
        self.assertEqual(c.usage['cached_prompt_tokens'],2900)

    def test_unsupported_gateway_falls_back_once_and_is_remembered(self):
        c=self.client(url='https://relay.example/v1')
        bodies=[]
        def send(req,timeout,record):
            b=json.loads(req.data)
            bodies.append(b)
            if len(bodies)==1:
                rejected(req,record,'prompt_cache_options')
            return accepted()
        c._request=Mock(side_effect=send)
        self.assertEqual(self.ask(c),{'ok':True})
        self.assertEqual(c.calls,2)
        self.assertIn('prompt_cache_options',bodies[0])
        self.assertNotIn('prompt_cache_options',bodies[1])
        self.assertIsInstance(bodies[1]['messages'][0]['content'],str)
        self.ask(c,'new_view')
        self.assertNotIn('prompt_cache_options',bodies[2])
        self.assertEqual(c.request_modes[
            ('https://relay.example/v1','gpt-6-astra')]['omit_prompt_cache'],True)

    def test_cache_can_be_disabled_and_invalid_configuration_rejected(self):
        c=self.client(cache='off')
        bodies=[]
        c._request=Mock(side_effect=lambda req,timeout,record:
            (bodies.append(json.loads(req.data)),accepted())[1])
        self.ask(c)
        self.assertNotIn('prompt_cache_options',bodies[0])
        self.assertIsInstance(bodies[0]['messages'][0]['content'],str)
        bad=settings()
        bad['prompt_cache']='force'
        with self.assertRaises(ValueError):
            model_endpoints(bad)


if __name__=='__main__':
    unittest.main()
