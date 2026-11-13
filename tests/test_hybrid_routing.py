"""Offline regression tests for tool-free cloud-delegated navigation."""
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from havln_agent.api_client import APIClient, APIError
from havln_agent.hybrid_router import HybridRouter
from havln_agent.local_agent import LOCAL_ACTION_SCHEMA, local_to_registered_command
from havln_agent.local_frame_selector import select_local_frames
from havln_agent.memory_records import MemoryRecords
from havln_agent.observation import array_hash
from havln_agent.perception import Camera, Observation
from havln_agent.export import load_config


def camera():
    return Camera(width=32, height=24, hfov=90., min_depth=0.1,
                  max_depth=10., normalized=True, camera_height=1.5,
                  robot_height=1.5, robot_radius=0.1, forward_step=0.25,
                  turn_angle=15., tilt_angle=15.)


def frame(seed):
    rgb = np.full((24, 32, 3), seed, dtype=np.uint8)
    depth = np.full((24, 32), .4, dtype=np.float32)
    return Observation(rgb, depth, 'instruction', 1., 0.)


def record(identifier, image, index, heading=0., stance=None):
    pose = dict(heading=math.radians(heading), pitch_deg=0.,
                stance_id=stance if stance is not None else '%s:0' % index,
                motion_epoch=index, viewpoint_generation=index)
    return dict(observation_id=identifier, rgb_hash=array_hash(image.rgb),
                depth_hash=array_hash(image.depth), pose=pose)


def cloud_decision():
    return dict(tool='navigate_relative', local_handoff=dict(
        goal='Continue along the same corridor',
        exit_condition='The corridor ends or forks'))


def completed_move():
    return dict(status='ACTIONS_COMPLETED', executed_actions=1,
                executed_motion=['MOVE_FORWARD'])


class HybridRouterTests(unittest.TestCase):
    def make(self):
        r = HybridRouter(dict(enabled=True, max_turns_without_forward=4))
        events = []
        runtime = SimpleNamespace(
            current_decision_id='navigation_agent_0003',
            memory=SimpleNamespace(current=dict(observation_id='obs_0042'),
                actions=[dict(action='MOVE_FORWARD', result=dict(visual_change=True))]),
            event=lambda name, **kwargs: events.append((name, kwargs)))
        return r, runtime, events

    def test_only_completed_cloud_forward_grants_local(self):
        router, runtime, events = self.make()
        router.after_cloud(cloud_decision(), dict(status='PARTIAL',
            executed_motion=['MOVE_FORWARD']), runtime, 'obs_0041')
        self.assertFalse(router.use_local())
        router.after_cloud(cloud_decision(), completed_move(),
                           runtime, 'obs_0041')
        self.assertTrue(router.use_local())
        self.assertEqual(router.route['source_observation_id'], 'obs_0041')
        self.assertEqual(router.route['start_observation_id'], 'obs_0042')
        self.assertTrue(any(name == 'hybrid_route_delegated' for name, _ in events))
        router.after_cloud(dict(tool='turn_and_observe'), completed_move(),
                           runtime, 'obs_0043')
        self.assertFalse(router.use_local())

    def test_local_actions_cannot_override_route_or_execute_when_uncertain(self):
        router, runtime, _ = self.make()
        router.after_cloud(cloud_decision(), completed_move(),
                           runtime, 'obs_0041')
        valid = dict(action='FORWARD', count=2, signal='NORMAL',
                     evidence_id='obs_0042')
        self.assertIsNone(router.approve_local(valid, 'obs_0042', 'CLEAR'))
        self.assertEqual(router.approve_local(valid, 'obs_0042', 'BLOCKED'),
                         'live_front_depth_blocked')
        self.assertEqual(router.approve_local(dict(valid, signal='NEW_BRANCH'),
                         'obs_0042', 'CLEAR'), 'local_signal_new_branch')
        self.assertEqual(router.approve_local(dict(valid, evidence_id='old'),
                         'obs_0042', 'CLEAR'), 'local_evidence_id_mismatch')
        self.assertEqual(router.approve_local(dict(valid, count=4),
                         'obs_0042', 'CLEAR'), 'invalid_local_forward_count')
        self.assertEqual(router.approve_local(dict(valid, action='ESCALATE', count=0),
                         'obs_0042', 'CLEAR'), 'local_requested_escalation')

    def test_turn_loop_escalates_without_forcing_forward(self):
        router, runtime, events = self.make()
        router.after_cloud(cloud_decision(), completed_move(),
                           runtime, 'obs_0041')
        for _ in range(3):
            router.after_local(dict(action='LEFT'), dict(
                status='ACTIONS_COMPLETED', executed_actions=1,
                executed_motion=['TURN_LEFT']), runtime)
        self.assertEqual(router.approve_local(
            dict(action='LEFT', count=2, signal='NORMAL',
                 evidence_id='obs_0042'), 'obs_0042', 'CLEAR'),
            'repeated_turn_only_local')
        router.after_local(dict(action='LOOK_UP'), dict(
            status='ACTIONS_COMPLETED', executed_actions=1,
            executed_motion=['LOOK_UP']), runtime)
        self.assertFalse(router.use_local())
        self.assertEqual(router.last_escalation, 'local_observation_without_forward')
        self.assertTrue(any(name == 'hybrid_escalated' for name, _ in events))
        router.cloud_ready()
        self.assertEqual(router.state, 'CLOUD_DECIDE')


class LocalProtocolTests(unittest.TestCase):
    def test_local_schema_has_no_tools_or_global_progress(self):
        self.assertEqual(set(LOCAL_ACTION_SCHEMA['properties']),
                         {'action', 'count', 'signal', 'evidence_id'})
        self.assertNotIn('request_finish', str(LOCAL_ACTION_SCHEMA))
        from jsonschema import Draft7Validator
        good = dict(action='FORWARD', count=2, signal='NORMAL',
                    evidence_id='obs_0013')
        Draft7Validator(LOCAL_ACTION_SCHEMA).validate(good)
        self.assertFalse(Draft7Validator(LOCAL_ACTION_SCHEMA).is_valid(
            dict(good, tool='request_finish')))

    def test_atomic_mapping_uses_real_camera_calibration(self):
        c = camera()
        self.assertEqual(local_to_registered_command(dict(
            action='FORWARD', count=2), c)['args']['distance_m'], .5)
        self.assertEqual(local_to_registered_command(dict(
            action='LEFT', count=2), c)['args']['turn_degrees'], 30.)
        self.assertEqual(local_to_registered_command(dict(
            action='LOOK_DOWN', count=1), c)['args']['look'], 'down')
        with self.assertRaises(APIError):
            local_to_registered_command(dict(action='ESCALATE', count=0), c)

    def test_keyframes_keep_route_anchor_and_current_last(self):
        c = camera()
        memory = MemoryRecords(c)
        saved = {}
        for i in range(14):
            ident = 'obs_%04d' % i
            image = frame(i + 1)
            saved[ident] = image
            memory.observations.append(record(ident, image, i, heading=i * 15))
        memory.current = memory.observations[-1]
        runtime = SimpleNamespace(memory=memory, observation=saved['obs_0013'],
            load_observation=lambda row:saved[row['observation_id']])
        manifest, images, depth_idx = select_local_frames(runtime, dict(
            source_observation_id='obs_0001', start_observation_id='obs_0003'),
            max_frames=10, max_edge=224)
        self.assertEqual(manifest[-1]['kind'], 'CURRENT_RGB')
        self.assertEqual(manifest[-1]['observation_id'], 'obs_0013')
        self.assertEqual(depth_idx, len(images)-1)
        self.assertLessEqual(len(manifest), 10)
        self.assertEqual(len(images), len(manifest)+1)
        self.assertIn('obs_0001', [m['observation_id'] for m in manifest])
        self.assertEqual([m['image_index'] for m in manifest],
                         list(range(len(manifest))))
        self.assertTrue(all(img.startswith('data:image/png;base64,')
                            for img in images))

    def test_changed_current_pixels_are_rejected(self):
        c = camera()
        memory = MemoryRecords(c)
        current = frame(5)
        memory.current = record('obs_0001', current, 1)
        memory.observations = [memory.current]
        runtime = SimpleNamespace(memory=memory, observation=frame(6),
            load_observation=None)
        with self.assertRaises(APIError):
            select_local_frames(runtime, dict(), max_frames=10)

    def test_cloud_and_local_http_attempt_budgets_are_independent(self):
        settings = dict(provider='openai_compatible',
            base_url='http://127.0.0.1:8000/v1',
            api_key_env='OFFLINE_KEY', model='havln-local',
            response_format='auto', prompt_cache='off', reasoning=None,
            temperature=.1, timeout_seconds=4, max_retries=0,
            max_tokens=128, fallbacks=[])
        schema = dict(type='object', additionalProperties=False,
            properties=dict(ok=dict(type='boolean')), required=['ok'])
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'OFFLINE_KEY':'test-secret'}):
                client = APIClient(dict(navigation_agent=settings,
                    visual_perception=settings, local_control=settings), tmp)
                client.begin_episode('offline', 1, Path(tmp)/'api.jsonl',
                                     local_budget=2)
                client._request = Mock(return_value=dict(
                    choices=[dict(message=dict(content=json.dumps(dict(ok=True))),
                                  finish_reason='stop')],usage={}))
                client.complete('local_control', 'test', 'Return JSON', {}, schema)
                client.complete('navigation_agent', 'test', 'Return JSON', {}, schema)
                self.assertEqual(client.availability(), 'episode_api_budget_exhausted')
                self.assertEqual(client.availability('local_control'), 'available')
                client.complete('local_control', 'test', 'Return JSON', {}, schema)
                self.assertEqual(client.availability('local_control'),
                                 'episode_local_budget_exhausted')
                self.assertEqual(client.stats()['cloud_requests'], 1)
                self.assertEqual(client.stats()['local_requests'], 2)

    def test_default_config_does_not_require_local_credentials(self):
        # The old API-only configuration remains strictly compatible.
        root = Path(__file__).resolve().parents[1]
        env = dict(NAVIGATION_AGENT_BASE_URL='https://example.com/v1',
                   NAVIGATION_AGENT_MODEL='cloud-nav',
                   NAVIGATION_AGENT_API_KEY='key',
                   VISUAL_PERCEPTION_BASE_URL='https://example.com/v1',
                   VISUAL_PERCEPTION_MODEL='cloud-vision',
                   VISUAL_PERCEPTION_API_KEY='key',
                   LOCAL_CONTROL_API_KEY='local-key')
        with patch.dict(os.environ, env):
            old = load_config(root/'configs/agent.yaml')
            new = load_config(root/'configs/agent-hybrid.yaml')
        self.assertNotIn('local_control', old)
        self.assertTrue(new['hybrid']['enabled'])
        self.assertEqual(new['local_control']['model'], 'havln-local')


class HybridSchedulerTests(unittest.TestCase):
    """The real scheduler dispatches cloud and local through one executor."""

    class FakeRuntime:
        def __init__(self, local_action='FORWARD'):
            self.config = dict(max_steps=20, max_model_decisions=40,
                max_unchanged_evidence_decisions=8)
            self.memory = SimpleNamespace(
                current=dict(observation_id='obs_0000'), actions=[],
                note_navigation_assessment=lambda decision: {},
                note_navigation_turn=lambda *args:None)
            self.hybrid_router = HybridRouter(dict(enabled=True))
            self.local_action = local_action
            self.camera = SimpleNamespace(forward_step=.25, turn_angle=15.)
            self.client = SimpleNamespace(availability=lambda role=None:'available')
            self.navigation_agent = SimpleNamespace(decide=self.cloud_decide)
            self.local_agent = SimpleNamespace(decide=self.local_decide)
            self.done, self.decisions = False, 0
            self.cloud_calls, self.local_calls, self.current_decision_id = 0, 0, None
            self.navigation_liveness = None
            self.executed = []
            self.events = []

        def start_mission(self):
            pass

        def refresh(self):
            pass

        def decision_id(self, role):
            self.current_decision_id = role + '_' + str(self.decisions)

        def event(self, name, **fields):
            self.events.append(dict(event=name, **fields))

        def cloud_decide(self, feedback):
            self.cloud_calls += 1
            return dict(cloud_decision(), args=dict(
                turn_direction='NONE', turn_degrees=0, distance_m=.25))

        def local_decide(self):
            self.local_calls += 1
            action = self.local_action
            return (dict(action=action,
                         count=0 if action == 'ESCALATE' else 1,
                         signal='NEW_BRANCH' if action == 'ESCALATE' else 'NORMAL',
                         evidence_id=self.memory.current['observation_id']),
                    dict(status='CLEAR'))

        def execute(self, command, binding):
            self.executed.append((command, binding))
            if command['tool'] == 'navigate_relative':
                actions = ['MOVE_FORWARD']
            else:
                actions = ['TURN_LEFT']
            self.memory.actions.append(dict(
                action=actions[0], result=dict(visual_change=True)))
            self.memory.current = dict(
                observation_id='obs_%04d' % len(self.memory.actions))
            return dict(status='ACTIONS_COMPLETED',
                tool_call_id='tool_%04d' % len(self.executed),
                executed_motion=actions, executed_actions=1)

        def assess(self, result):
            if len(self.memory.actions) >= 2:
                self.done = True

        def terminate(self, reason):
            self.done = True

    def test_cloud_handoff_then_local_forward_uses_same_executor(self):
        from havln_agent.scheduler import run
        runtime = self.FakeRuntime()
        records = []
        run(runtime, lambda role, item:records.append(item))
        self.assertEqual(runtime.cloud_calls, 1)
        self.assertEqual(runtime.local_calls, 1)
        self.assertEqual(len(runtime.executed), 2)
        self.assertEqual(runtime.executed[0][1]['actor'], 'navigation_agent')
        self.assertEqual(runtime.executed[1][1]['actor'], 'local_agent')
        self.assertEqual(runtime.hybrid_router.local_decisions, 1)
        self.assertTrue(any(item.get('model_tier') == 'local' for item in records))

    def test_local_escalation_executes_no_local_command(self):
        from havln_agent.scheduler import run
        runtime = self.FakeRuntime(local_action='ESCALATE')
        run(runtime, lambda role, item:None)
        self.assertEqual(runtime.local_calls, 1)
        self.assertEqual(runtime.cloud_calls, 2)
        self.assertEqual(len(runtime.executed), 2)
        self.assertTrue(all(binding['actor'] == 'navigation_agent'
                        for _,binding in runtime.executed))
        self.assertEqual(runtime.hybrid_router.escalations, 1)


if __name__ == '__main__':
    unittest.main()
