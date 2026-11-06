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
    subgoal_start_steps: dict = field(default_factory=dict)
    provisional_transitions: list = field(default_factory=list)


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
        self.observation_searches = []
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
        self.task.subgoal_start_steps = {'0':len(self.actions)}
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
        # Compare against ANY previously seen same-pose view, not only the
        # immediately preceding keyframe. Up/down oscillation is not discovery.
        view_matches=[]
        for old, old_features in zip(self.observations, self._features):
            old_pose=old['pose']
            if (old.get('subgoal') != self.task.current_subgoal_index
                    or old_pose.get('motion_epoch') != pose.get('motion_epoch')
                    or old_pose.get('viewpoint_generation',0) != pose.get('viewpoint_generation',0)
                    or math.hypot(old_pose['x']-pose['x'],old_pose['z']-pose['z']) > .08
                    or abs(angle_error(old_pose['heading'],pose['heading'])) >
                        math.radians(self.camera.turn_angle/2)
                    or abs(old_pose.get('pitch_deg',0)-pose.get('pitch_deg',0)) > 1):
                continue
            if frame_change(features,old_features)['similar']:
                view_matches.append(old)
        representative = not any(o.get('representative_keyframe') for o in view_matches)
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
            revisited_view_ids=[o['observation_id'] for o in view_matches[-4:]],
            revisited_static_view=bool(view_matches),
            new_visual_information=not view_matches,
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
        successful=result['status'] in {'OK','EXECUTABLE','LOCALIZED','RECORDED','VERIFIED','FINISHED','CANCELLED','PARTIAL','TARGET_REACHED','ACTIONS_COMPLETED'}
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

    def route_progress_summary(self):
        """Reconstruct successful segments AND blocked endpoints from receipts.

        This is not a map, route permission, or inferred global odometry.
        Failed probes never erase earlier registered movement.
        """
        segments=[]
        for a in self.actions[-120:]:
            if a.get('action')!='MOVE_FORWARD':
                continue
            feedback=a.get('result') or {}
            motion=feedback.get('rgbd_motion') or {}
            if (motion.get('status')!='ESTIMATED'
                    or not motion.get('observed_frame_change',True)):
                continue
            translation=motion.get('translation_m') or {}
            if not isinstance(translation,dict):
                continue
            pose=a.get('pose') or {}
            if not isinstance(pose.get('heading'),(int,float)):
                continue
            yaw=round(math.degrees(pose['heading'])%360,1)
            dy=translation.get('y_up')
            epoch=pose.get('motion_epoch')
            previous=segments[-1] if segments else None
            consecutive=(previous and
                a['step']==previous['last_step']+1 and
                a.get('subgoal')==previous['subgoal'] and
                epoch==previous['motion_epoch'] and
                abs((yaw-previous['heading_deg']+180)%360-180)<=self.camera.turn_angle/2)
            if consecutive:
                previous['last_step']=a['step']
                previous['forward_steps']+=1
                previous['observed_height_change_m_est']=(
                    round(previous['observed_height_change_m_est']+dy,3)
                    if isinstance(dy,(int,float)) else
                    previous['observed_height_change_m_est'])
                previous['end_observation_id']=feedback.get('observation_after')
            else:
                segments.append(dict(subgoal=a.get('subgoal'),
                    motion_epoch=epoch,heading_deg=yaw,
                    first_step=a['step'],last_step=a['step'],
                    forward_steps=1,
                    observed_height_change_m_est=round(dy,3)
                        if isinstance(dy,(int,float)) else 0.,
                    start_observation_id=feedback.get('observation_before'),
                    end_observation_id=feedback.get('observation_after'),
                    evidence='registered_rgbd_motion_not_habitat_ground_truth'))
        blocked=[]
        for entry in self.tool_ledger[-50:]:
            call=entry.get('call') or {}
            if call.get('tool') not in (
                    'navigate_relative','navigate_local','navigate_to_pixel'):
                continue
            result=entry.get('result') or {}
            if (result.get('interrupted_status')!='BLOCKED' and
                    result.get('status')!='BLOCKED'):
                continue
            pose=(entry.get('request_context') or {}).get('pose') or {}
            heading=pose.get('heading')
            if not isinstance(heading,(int,float)):
                continue
            blocked.append(dict(
                heading_deg=round(math.degrees(heading)%360,1),
                source_observation_id=result.get('end_observation_id'),
                reason=result.get('reason'),
                subgoal=entry.get('subgoal_id'),
                evidence='observed_block_at_stance_not_historical_route_veto'))
        return dict(successful_segments=segments[-8:],
            blocked_at_prior_stances=blocked[-6:],
            current_subgoal=self.task.current_subgoal_index,
            interpretation='past_success_is_progress_evidence_not_live_motion_clearance')

    def confirm_subgoal_observation(self, verification, objective):
        """Commit only a source-backed VLM confirmation, not Python semantics."""
        task=self.task
        index=task.current_subgoal_index
        if index>=len(task.subgoals) or index in task.completed_subgoals:
            return dict(status='NOT_APPLICABLE',reason='no_active_unconfirmed_subgoal')
        if verification.get('status')!='CONFIRMED':
            return dict(status='NOT_CONFIRMED',reason='vlm_did_not_confirm')
        observed={o['observation_id']:o for o in self.observations}
        ids=list(dict.fromkeys(verification.get('observation_ids') or []))
        live=verification.get('source_observation_id')
        if (not ids or live not in ids or live not in observed
                or observed[live]['subgoal']!=index
                or not verification.get('evidence','').strip()):
            return dict(status='INVALID_EVIDENCE',
                reason='confirmation_requires_real_current_subgoal_RGBD')
        recorded=dict(subgoal=index,evidence=verification['evidence'],
            objective=objective,observation_ids=ids,
            executed_steps=[a['step'] for a in self.actions
                if a.get('subgoal')==index and a.get('action')=='MOVE_FORWARD'],
            verification=dict(status='VERIFIED',completion_confirmed=True,
                supported=['vlm_observation_confirmed'],
                evidence_origin='vlm_real_rgbd_history',
                source_observation_id=live,
                uncertainty=verification.get('uncertainty','')))
        task.completed_subgoals.append(index)
        task.completion_evidence.append(recorded)
        next_index=index+1
        while (next_index<len(task.subgoals) and
                next_index in task.completed_subgoals):
            next_index+=1
        if next_index<len(task.subgoals):
            task.current_subgoal_index=next_index
            task.subgoal_start_steps.setdefault(str(next_index),len(self.actions))
            task.subgoal_started=task.subgoal_start_steps[str(next_index)]
        self.progress_stage='COMPLETED'
        self.progress_feedback=recorded['verification']
        self.event('subgoal_observation_confirmed',
            subgoal=index,next_subgoal=next_index,observation_ids=ids)
        return dict(status='CONFIRMED',subgoal=index,
            next_subgoal=next_index,observation_ids=ids)

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
            observation_searches=copy.deepcopy(self.observation_searches),
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
