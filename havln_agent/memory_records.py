"""Episode facts and bounded retrieval. Memory never selects or bans a route."""

from dataclasses import asdict, dataclass, field
import copy
import math
import re
import time

import numpy as np
from PIL import Image

from .observation import frame_info


def angle_error(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def task_tokens(text):
    return set(re.findall(r'[a-z]+', text.lower().replace('staircase', 'stairs'))) - {
        'the',
        'of',
        'a',
        'to',
        'at',
        'with',
        'near',
        'and',
    }


def frame_change(first, second):
    """Small RGB-D appearance comparison, not localization or measured progress."""
    rgb_error = float(np.abs(first[0] - second[0]).mean())
    common = first[2] & second[2]
    overlap = float(common.mean())
    depth_error = float(np.median(np.abs(first[1][common] - second[1][common]))) if overlap >= 0.4 else None
    mask_error = float(np.not_equal(first[2], second[2]).mean())
    depth_agrees = (depth_error is not None and depth_error < 0.2 or
                    not first[2].any() and not second[2].any())
    similar = rgb_error < 0.06 and depth_agrees and mask_error < 0.1
    return dict(rgb_error=rgb_error, depth_error_metres=depth_error, valid_overlap=overlap,
                depth_mask_error=mask_error, similar=similar,
                basis='downsampled_rgb_depth_appearance_not_localization')


@dataclass
class TaskMemory:
    instruction: str = ''
    subgoals: list = field(default_factory=list)
    final_target: str = ''
    constraints: list = field(default_factory=list)
    current_subgoal_index: int = 0
    completed_subgoals: list = field(default_factory=list)
    observed_landmarks: list = field(default_factory=list)
    completion_evidence: list = field(default_factory=list)
    progress_claims: list = field(default_factory=list)
    subgoal_started: int = 0


class MemoryRecords:
    def __init__(self, camera):
        self.camera = camera
        self.reset('', '')

    def reset(self, episode_id, instruction):
        self.episode_id = episode_id
        self.task = TaskMemory(instruction)
        self.observations, self.actions, self.experiences = [], [], []
        self.groundings, self.humans, self.events = [], [], []
        self.scene_memory, self.active_attempt = None, None
        self.historical_scenes = []
        self.scene_history, self.visual_changes = [], []
        self.human_max_age_seconds = getattr(self, 'human_max_age_seconds', 2)
        self.current, self.revisits = None, []
        self._features = []
        self.queries = 0
        self.tool_ledger = []
        self.progress_stage, self.progress_feedback = 'NOT_STARTED', None

    def event(self, kind, **data):
        self.events.append(dict(event=kind, step=len(self.actions), **data))

    def set_plan(self, plan):
        self.task.subgoals = plan['subgoals']
        self.task.final_target = plan.get('final_target', '')
        self.task.constraints = plan.get('constraints', [])

    def observe(self, observation, pose, diagnostics):
        info = frame_info(observation, self.camera)
        self.humans = [h for h in self.humans if observation.captured_at - h['source']['captured_at'] <= self.human_max_age_seconds]
        if (
            self.current
            and self.current['rgb_hash'] == info['rgb_hash']
            and self.current['depth_hash'] == info['depth_hash']
            and self.current['pose'] == pose
        ):
            self.current['last_read_at'] = observation.captured_at
            self.event('sensor_read',observation_id=self.current['observation_id'],read_at=observation.captured_at)
            self.current['directions'] = [
                {k: d[k] for k in ('angle_degrees', 'path_state', 'clearance', 'accepted',
                                   'rejection_reasons','motion_state','reason') if k in d} for d in diagnostics
            ]
            return self.current
        rgb = np.asarray(Image.fromarray(observation.rgb).resize((16, 16)), dtype=float) / 255
        metric = observation.depth
        if self.camera.normalized:
            metric = metric * (self.camera.max_depth - self.camera.min_depth) + self.camera.min_depth
        valid = np.isfinite(metric) & (metric > self.camera.min_depth) & (metric <= self.camera.max_depth)
        depth = np.asarray(
            Image.fromarray(np.where(valid, metric, 0).astype('float32')).resize((16, 16)), dtype=float
        )
        mask = np.asarray(Image.fromarray((valid * 255).astype('uint8')).resize((16, 16))) > 200
        features = (rgb, depth, mask)
        change = frame_change(features, self._features[-1]) if self._features else None
        similar_views = [o['observation_id'] for o, old in zip(self.observations, self._features)
                         if frame_change(features, old)['similar']]
        last_keyframe = next((i for i in range(len(self.observations) - 1, -1, -1)
                              if self.observations[i].get('representative_keyframe')), None)
        representative = last_keyframe is None or not frame_change(features, self._features[last_keyframe])['similar']
        matches = []
        for old, features in zip(self.observations[:-4], self._features[:-4]):
            if abs(angle_error(pose['heading'], old['pose']['heading'])) > math.radians(
                self.camera.turn_angle / 2
            ):
                continue
            rgb_error = float(np.abs(rgb - features[0]).mean())
            common = mask & features[2]
            coverage = float(common.mean())
            depth_error = (
                float(np.median(np.abs(depth[common] - features[1][common]))) if coverage >= 0.4 else None
            )
            structure_error = float(np.not_equal(mask, features[2]).mean())
            if rgb_error < 0.06 and depth_error is not None and depth_error < 0.2 and structure_error < 0.1:
                matches.append(
                    dict(
                        observation_id=old['observation_id'],
                        rgb_error=rgb_error,
                        depth_error_metres=depth_error,
                        valid_overlap=coverage,
                        confidence='moderate',
                        basis='rgb_depth_structure_and_heading',
                        localization_verified=False,
                        estimated_position_separation_m=math.hypot(pose['x'] - old['pose']['x'], pose['z'] - old['pose']['z']),
                        relative_pose_bound_m=pose.get('position_uncertainty', 0.05) + old['pose'].get('position_uncertainty', 0.05),
                    )
                )
        record = dict(
            info,
            observation_id='obs_%04d' % len(self.observations),
            step=len(self.actions),
            subgoal=self.task.current_subgoal_index,
            pose=dict(pose),
            captured_at=observation.captured_at,
            last_read_at=observation.captured_at,
            directions=[
                {k: d[k] for k in ('angle_degrees', 'path_state', 'clearance', 'accepted',
                                   'rejection_reasons','motion_state','reason') if k in d}
                for d in diagnostics
            ],
            revisit_matches=matches[-4:],
            visual_change=change,
            similar_view_ids=similar_views[-4:],
            new_visual_information=not similar_views,
            representative_keyframe=representative,
        )
        self.observations.append(record)
        self._features.append((rgb, depth, mask))
        self.current = record
        if matches:
            self.revisits.append(dict(observation_id=record['observation_id'], matches=matches[-4:]))
            self.event('possible_revisit', observation_id=record['observation_id'], matches=matches[-4:])
        self.event('observation_updated', observation_id=record['observation_id'])
        return record

    def record_historical_scene(self, scene, source):
        self.historical_scenes.append(dict(scene=scene, source=dict(source), evidence_state='model_hypothesis', usable_for_live_safety=False))
        self.scene_history.append(dict(scene=scene, source=dict(source), image_scope=source.get('image_scope')))
        self.task.observed_landmarks.extend(dict(item, source=dict(source), evidence_state='model_hypothesis') for item in scene['landmarks'])
        self.event('historical_scene_analyzed', source=source)

    def record_scene(self, scene, source):
        self.scene_memory = dict(scene=scene, source=dict(source), evidence_state='model_hypothesis')
        self.scene_history.append(dict(scene=scene, source=dict(source), image_scope='current_at_capture'))
        for landmark in scene['landmarks']:
            self.task.observed_landmarks.append(dict(landmark, source=dict(source), evidence_state='model_hypothesis'))
        self.humans = [
            h
            for h in self.humans
            if time.monotonic() - h['source']['captured_at'] <= self.human_max_age_seconds
            and not (
                math.hypot(
                    source['pose']['x'] - h['source']['pose']['x'],
                    source['pose']['z'] - h['source']['pose']['z'],
                )
                <= 0.12
                and abs(angle_error(source['pose']['heading'], h['source']['pose']['heading']))
                <= math.radians(self.camera.turn_angle / 2)
            )
        ]
        self.humans.extend(
            dict(item, kind=kind, source=dict(source), motion='unknown')
            for kind, key in [('person', 'humans'), ('uncertain', 'uncertain_humanoids')]
            for item in scene.get(key, []) if time.monotonic() - source['captured_at'] <= self.human_max_age_seconds
        )
        self.event('scene_updated', source=source)

    def record_tool(self, call, result, request_context):
        call,result,request_context=copy.deepcopy((call,result,request_context))
        successful=result['status'] in {'OK','EXECUTABLE','LOCALIZED','RECORDED','VERIFIED','FINISHED','CANCELLED','PARTIAL','TARGET_REACHED'}
        related=[r for r in self.tool_ledger if not r['resolved'] and r['call']==call]
        record=dict(call_id=result['tool_call_id'],actor=request_context['actor'],
            subgoal_id=request_context['subgoal_id'],call=call,result=result,request_context=request_context,
            resolved=successful,retry_of=[r['call_id'] for r in related],followups=[],recorded_at=time.monotonic())
        for previous in related:
            previous['followups'].append(record['call_id'])
            if successful:previous.update(resolved=True,resolution_call_id=record['call_id'])
        self.tool_ledger.append(record)
        self.event('tool_completed',tool_call_id=record['call_id'],tool=call['tool'],status=result['status'])
        return record

    def start_attempt(self, target, pose):
        self.active_attempt = dict(
            attempt_id='attempt_%03d' % len(self.experiences),
            target={k: v for k, v in target.items() if k not in ('execution_status', 'last_validation')},
            start_pose=dict(pose),
            subgoal=self.task.current_subgoal_index,
            step_started=len(self.actions),
            actions=[],
            interruptions=[],
            result='IN_PROGRESS',
        )
        self.experiences.append(self.active_attempt)
        self.event('attempt_started', attempt_id=self.active_attempt['attempt_id'], target=target)

    def finish_attempt(self, outcome, reason=None):
        if self.active_attempt:
            self.active_attempt.update(
                result=outcome,
                reason=reason,
                step_finished=len(self.actions),
                observation_after=self.current['observation_id'],
            )
            self.event('attempt_finished', attempt=dict(self.active_attempt))
            self.active_attempt = None

    def record_action(self, action, pose, feedback):
        entry = dict(
            step=len(self.actions),
            action=action,
            pose=dict(pose),
            result=feedback,
            subgoal=self.task.current_subgoal_index,
        )
        self.actions.append(entry)
        if self.active_attempt:
            self.active_attempt['actions'].append(entry)
        self.event('action_executed', action=action, result=feedback)

    def stats(self):
        outcomes = {}
        for a in self.experiences:
            outcomes[a['result']] = outcomes.get(a['result'], 0) + 1
        return dict(
            observations=len(self.observations),
            navigation_attempts=len(self.experiences),
            outcomes=outcomes,
            memory_query_count=self.queries,
            possible_revisits=len(self.revisits),
        )

    def snapshot(self):
        return dict(
            episode_id=self.episode_id,
            task=asdict(self.task),
            scene_memory=self.scene_memory,
            historical_scenes=self.historical_scenes,
            scene_history=self.scene_history,
            visual_changes=self.visual_changes,
            observations=self.observations,
            actions=self.actions,
            experiences=self.experiences,
            groundings=self.groundings,
            humans=self.humans,
            revisits=self.revisits,
            statistics=self.stats(),
            tool_ledger=self.tool_ledger,
            progress_state=dict(stage=self.progress_stage, feedback=self.progress_feedback),
        )
