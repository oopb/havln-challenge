"""Episode facts and bounded retrieval. Memory never selects or bans a route."""

from dataclasses import asdict, dataclass, field
import copy
import math
import re
import time

import numpy as np
from PIL import Image

from .observation import frame_info
from .evidence_ledger import EvidenceLedger
from .heading import same_local_stance


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
    """One user instruction and revisable narrative state, not a task DAG."""
    instruction: str = ''
    progress_narrative: str = 'The mission has just begun; no destination is confirmed reached.'
    navigation_intent: str = 'Observe the surroundings and make progress toward the complete instruction.'
    uncertainties: list = field(default_factory=list)
    progress_history: list = field(default_factory=list)
    evidence_observation_ids: list = field(default_factory=list)
    observed_landmarks: list = field(default_factory=list)
    navigation_intent_anchor: dict = field(default_factory=dict)


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
        self.navigation_turns = []
        # Canonical, source-linked hypotheses and real motion receipts.
        # Neither field grants safety clearance or authorizes a move.
        self.route_candidates = []
        self.latest_motion_receipt = None
        self.human_max_age_seconds = getattr(self, 'human_max_age_seconds', 2)
        self.current, self.revisits = None, []
        self._features = []
        self.queries = 0
        self.tool_ledger = []
        self.evidence=EvidenceLedger()

    def event(self, kind, **data):
        entry=dict(event=kind,step=len(self.actions))
        entry.update(data)
        self.events.append(entry)

    def note_navigation_assessment(self, decision):
        """Revision log for model assessments, never a verified sensor fact.

        Each revision has an actual decision-frame identity even if the model
        omits citations. This context frame is not promoted to task evidence;
        only explicit, existing IDs and VLM checkpoint IDs become evidence.
        Unchanged repeated reports are not new historical events.
        """
        task=self.task
        allowed={o['observation_id'] for o in self.observations}
        cited=list(dict.fromkeys(o for o in
            (decision.get('evidence_observation_ids') or [])
            if isinstance(o,str) and o in allowed))
        previous=dict(progress_narrative=task.progress_narrative,
                      navigation_intent=task.navigation_intent,
                      uncertainties=list(task.uncertainties))
        changed=False
        for key,maximum in [('progress_narrative',1200),
                            ('navigation_intent',700)]:
            value=decision.get(key)
            if isinstance(value,str) and value.strip():
                proposed=value.strip()[:maximum]
                if getattr(task,key)!=proposed:
                    setattr(task,key,proposed)
                    changed=True
        if isinstance(decision.get('uncertainties'),list):
            proposed=[v.strip()[:220]
                      for v in decision['uncertainties'][:8]
                      if isinstance(v,str) and v.strip()]
            if proposed!=task.uncertainties:
                task.uncertainties=proposed
                changed=True
        # Each navigation intent is a CAMERA-RELATIVE MODEL HYPOTHESIS.
        # Never silently reuse "slightly right" after a turn or forward.
        context_record=self.current or {}
        context_pose=context_record.get('pose') or {}
        if (isinstance(decision.get('navigation_intent'),str)
                and decision['navigation_intent'].strip()
                and context_record.get('observation_id') in allowed):
            anchor=dict(
                source_observation_id=context_record['observation_id'],
                source_action_step=len(self.actions),
                source_heading_deg=round(
                    math.degrees(context_pose.get('heading',0))%360,2),
                source_pitch_deg=context_pose.get('pitch_deg'),
                motion_epoch=context_pose.get('motion_epoch'),
                viewpoint_generation=context_pose.get('viewpoint_generation'),
                stance_id=context_pose.get('stance_id'),
                direction_is_camera_relative_at_source=True)
            # Repeating the EXACT same free-text intent after movement
            # is not new grounding: it may be the stale "slightly right"
            # sentence copied forward verbatim. Only a genuinely revised
            # intent creates a new source binding; an unchanged phrase
            # retains its ORIGINAL camera reference until replaced.
            novel_intent=(
                task.navigation_intent!=previous['navigation_intent']
                or not task.navigation_intent_anchor)
            if novel_intent and anchor!=task.navigation_intent_anchor:
                task.navigation_intent_anchor=anchor
                changed=True
        new_refs=[]
        for identifier in cited:
            if identifier not in task.evidence_observation_ids:
                task.evidence_observation_ids.append(identifier)
                new_refs.append(identifier)
        if changed or new_refs:
            context_id=(self.current or {}).get('observation_id')
            entry=dict(
                step=len(self.actions),
                source='navigation_agent_model_assessment_not_sensor_fact',
                context_observation_id=(context_id
                    if context_id in allowed else None),
                prior=previous,
                progress_narrative=task.progress_narrative,
                navigation_intent=task.navigation_intent,
                navigation_intent_anchor=copy.deepcopy(task.navigation_intent_anchor),
                uncertainties=list(task.uncertainties),
                evidence_observation_ids=cited,
                cited_observation_ids_validated=True,
                completion_verified=False)
            task.progress_history.append(entry)
            # Retain every episode revision for causal navigation history.
            # The model receives a structured projection, not repeated raw
            # prompt texts or duplicated RGB base64 messages.
            self.event('mission_progress_updated',**entry)
        return dict(changed=changed,
                    evidence_observation_ids=cited,
                    new_evidence_observation_ids=new_refs,
                    context_observation_id=(self.current or {}).get('observation_id'))

    def note_navigation_turn(self, decision, result, source_observation_id,
                             step_started, decision_id=None):
        """Record EVERY completed model decision with its real tool receipt.

        Provenance is immutable. Raw visual media remains in observation
        records, never copied into navigation history or chat logs.
        """
        source=next((o for o in self.observations
                     if o.get('observation_id')==source_observation_id),None)
        pose=(source or {}).get('pose') or {}
        result=result or {}
        item=dict(
            turn_index=len(self.navigation_turns),
            decision_id=decision_id,
            source_observation_id=source_observation_id,
            source_heading_deg=round(math.degrees(
                pose.get('heading',0))%360,2),
            source_stance_id=pose.get('stance_id'),
            source_motion_epoch=pose.get('motion_epoch'),
            source_viewpoint_generation=pose.get('viewpoint_generation'),
            step_started=step_started,
            step_finished=len(self.actions),
            decision=copy.deepcopy(decision),
            tool_call_id=result.get('tool_call_id'),
            result=dict(
                status=result.get('status'),
                interrupted_status=result.get('interrupted_status'),
                reason=result.get('reason'),
                executed_actions=result.get('executed_actions'),
                executed_motion=copy.deepcopy(
                    result.get('executed_motion',[])),
                end_observation_id=result.get('end_observation_id'),
                blocked_at=copy.deepcopy(result.get('blocked_at')),
                motion_alert=copy.deepcopy(result.get('motion_alert')),
                observation_summary=result.get('observation_summary'),
                navigation_suggestion=result.get('navigation_suggestion'),
                evidence_observation_ids=copy.deepcopy(
                    result.get('evidence_observation_ids',[]))))
        self.navigation_turns.append(item)
        if decision.get('tool') in ('navigate_relative','navigate_local'):
            forward=[action for action in self.actions[step_started:]
                     if action.get('action')=='MOVE_FORWARD']
            status=result.get('interrupted_status') or result.get('status')
            end=(self.current or {}).get('observation_id')
            end_pose=(self.current or {}).get('pose') or {}
            self.latest_motion_receipt=dict(
                decision_id=decision_id,
                tool=decision['tool'],
                source_observation_id=source_observation_id,
                end_observation_id=end,
                last_heading_deg=round(
                    math.degrees(end_pose.get('heading',0))%360,2),
                last_stance_id=end_pose.get('stance_id'),
                status=status,
                reason=result.get('reason'),
                commanded_args=copy.deepcopy(decision.get('args') or {}),
                executed_forward_actions=len(forward),
                forward_actions_with_RGBD_change=sum(
                    bool((a.get('result') or {}).get('visual_change'))
                    for a in forward),
                observed_motion_not_verified_translation=True,
                action_step_finished=len(self.actions),
                continuation_is_model_choice=True,
                last_motion_was_uninterrupted=(
                    status in ('ACTIONS_COMPLETED','TARGET_REACHED')))
        return item

    def note_visual_route_candidate(self, value):
        """Remember a visual route hypothesis even when no target box exists.

        Only actual RGB-D source IDs are accepted. The optional PASSAGE
        box, not a semantic target/object box, supplies a source-relative
        bearing. Prose or an obstacle's semantic box never acquires
        a fabricated precise heading, distance or walkability certificate.
        """
        if not isinstance(value,dict):
            return None
        identifier=value.get('source_observation_id')
        source=next((o for o in reversed(self.observations)
                     if o.get('observation_id')==identifier),None)
        if source is None:
            return None
        evidence=str(value.get('evidence') or '').strip()[:420]
        suggestion=str(value.get('navigation_suggestion') or '').strip()[:350]
        if not (evidence or suggestion):
            return None
        pose=source.get('pose') or {}
        objective=str(value.get('objective') or '').strip()[:250]
        target_type=str(value.get('target_type') or 'possible passage')[:90]
        absolute=value.get('absolute_heading_deg')
        if not isinstance(absolute,(int,float)) or not math.isfinite(absolute):
            absolute=None
        for prior in reversed(self.route_candidates):
            old=next((o for o in reversed(self.observations)
                      if o.get('observation_id')==
                         prior.get('source_observation_id')),None)
            if (prior.get('target_type')==target_type
                    and prior.get('objective')==objective
                    and old is not None
                    and old.get('rgb_hash')==source.get('rgb_hash')
                    and old.get('depth_hash')==source.get('depth_hash')
                    and same_local_stance(old.get('pose') or {},pose)
                    and abs(angle_error(
                        (old.get('pose') or {}).get('heading',0),
                        pose.get('heading',0)))<math.radians(1)):
                return prior
        item=dict(
            candidate_id='visual_route_%04d'%len(self.route_candidates),
            source_observation_id=identifier,
            source_stance_id=pose.get('stance_id'),
            source_heading_deg=round(math.degrees(pose.get('heading',0))%360,2),
            target_type=target_type,confidence=value.get('confidence'),
            target_kind=value.get('target_kind'),
            objective=objective,
            evidence=evidence,navigation_suggestion=suggestion,
            uncertainty=str(value.get('uncertainty') or '')[:300],
            target_visible=bool(value.get('target_visible')),
            target_box=copy.deepcopy(value.get('target_box')),
            passage_box=copy.deepcopy(value.get('passage_box')),
            semantic_target_box=copy.deepcopy(
                value.get('semantic_target_box')),
            bearing_region_role=value.get('bearing_region_role') or (
                'legacy_unclassified_box' if absolute is not None else
                'unlocalized_semantic_route_hypothesis'),
            absolute_heading_deg=round(absolute%360,2)
                if absolute is not None else None,
            angular_bearing_grounded=absolute is not None,
            distance_m_hint=value.get('distance_m_hint')
                if absolute is not None else None,
            route_status=(
                'VISUAL_PASSAGE_REGION_NOT_VERIFIED_TRAVERSABILITY'
                if absolute is not None else
                'SEMANTIC_VISUAL_CUE_WITHOUT_LOCALIZED_PASSAGE'),
            motion_authorized=False)
        self.route_candidates.append(item)
        self.event('visual_route_candidate_recorded',
                   candidate_id=item['candidate_id'],
                   source_observation_id=identifier,
                   angular_bearing_grounded=absolute is not None)
        return item

    def note_observation_evidence(self, identifiers, objective=''):
        """Link genuine VLM checkpoints to Memory, not to task completion."""
        existing={o['observation_id'] for o in self.observations}
        added=[]
        for identifier in identifiers:
            if (identifier in existing and
                    identifier not in self.task.evidence_observation_ids):
                self.task.evidence_observation_ids.append(identifier)
                added.append(identifier)
        if added:
            self.event('observation_evidence_indexed',
                objective=str(objective)[:200],
                observation_ids=added,
                source='authentic_RGBD_observation_checkpoints',
                completion_verified=False)
        return added

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
            if (not same_local_stance(old_pose,pose)
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
                        estimated_position_separation_m=None,
                        relative_pose_bound_m=None,
                        same_stance_by_forward_receipts=same_local_stance(
                            old['pose'],pose),
                    )
                )
        record = dict(
            info,
            observation_id='obs_%04d' % len(self.observations),
            step=len(self.actions),
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
                same_local_stance(source['pose'],h['source']['pose'])
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
            call=call,result=result,request_context=request_context,
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
        )
        self.actions.append(entry)
        if self.active_attempt:
            self.active_attempt['actions'].append(entry)
        self.event('action_executed', action=action, result=feedback)

    def route_progress_summary(self, max_successful_segments=8):
        """Reconstruct changed-RGBD action segments and historical blocks.

        Never label a changed image as verified displacement or a traversed
        route. Consecutive executed-forward receipts are factual even when
        there is no metric robot pose.
        """
        segments=[]
        # Successful image change after an executed MOVE is useful history,
        # but is NOT metric translation or proven navigational arrival.
        for a in self.actions:
            if a.get('action')!='MOVE_FORWARD':
                continue
            feedback=a.get('result') or {}
            if not feedback.get('visual_change',False):
                continue
            pose=a.get('pose') or {}
            if not isinstance(pose.get('heading'),(int,float)):
                continue
            yaw=round(math.degrees(pose['heading'])%360,1)
            epoch=pose.get('motion_epoch')
            gen=pose.get('viewpoint_generation')
            previous=segments[-1] if segments else None
            consecutive=(previous
                and a['step']==previous['last_step']+1
                and epoch==previous['motion_epoch']
                and abs((yaw-previous['heading_deg']+180)%360-180)
                    <=self.camera.turn_angle/2)
            if consecutive:
                previous['last_step']=a['step']
                previous['forward_steps']+=1
                previous['end_observation_id']=feedback.get('observation_after')
                previous['last_stance_generation']=gen
            else:
                segments.append(dict(
                    motion_epoch=epoch,heading_deg=yaw,
                    first_step=a['step'],last_step=a['step'],
                    forward_steps=1,
                    first_stance_generation=gen,
                    last_stance_generation=gen,
                    observed_height_change_m_est=None,
                    uncertainty_m_est=None,
                    start_observation_id=feedback.get('observation_before'),
                    end_observation_id=feedback.get('observation_after'),
                    evidence='executed_forward_and_RGBD_changed_not_metric_displacement',
                    displacement_verified=False))
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
            # The executor's blocked_at pose is authoritative even if a
            # later keyframe was evicted. Legacy receipts fall back to the
            # actual end-of-call camera pose, not the pre-turn request pose.
            exact=result.get('blocked_at')
            if exact and exact.get('observation_id'):
                blocked.append(dict(exact,
                    source_observation_id=exact['observation_id'],
                    evidence='measured_block_at_post_turn_stance'))
                continue
            end_id=result.get('end_observation_id')
            end_record=next((o for o in reversed(self.observations)
                if o['observation_id']==end_id),None)
            pose=((end_record or {}).get('pose') or
                (entry.get('request_context') or {}).get('pose') or {})
            heading=pose.get('heading')
            if not isinstance(heading,(int,float)):
                continue
            blocked.append(dict(
                heading_deg=round(math.degrees(heading)%360,1),
                source_observation_id=result.get('end_observation_id'),
                local_pose_xz_m_est=None,
                place_id=(end_record or {}).get('place_id'),
                motion_epoch=pose.get('motion_epoch'),
                viewpoint_generation=pose.get('viewpoint_generation'),
                reason=result.get('reason'),
                evidence='observed_block_at_stance_not_historical_route_veto'))
        return dict(successful_segments=(segments if max_successful_segments is None
                                        else segments[-max_successful_segments:]),
            blocked_at_prior_stances=blocked[-6:],
            latest_exact_blocked_at=copy.deepcopy(
                self.evidence.obstacle_blocks[-4:]),
            interpretation='past_forward_actions_with_RGBD_change_not_verified_translation_or_live_clearance')

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
            navigation_turns=copy.deepcopy(self.navigation_turns),
            route_candidates=copy.deepcopy(self.route_candidates),
            latest_motion_receipt=copy.deepcopy(self.latest_motion_receipt),
            visual_changes=self.visual_changes,
            observations=self.observations,
            actions=self.actions,
            experiences=self.experiences,
            groundings=self.groundings,
            humans=self.humans,
            revisits=self.revisits,
            statistics=self.stats(),
            evidence_ledger=self.evidence.summary(),
            tool_ledger=self.tool_ledger,
            progress_state=dict(progress_narrative=self.task.progress_narrative,
                navigation_intent=self.task.navigation_intent,
                uncertainties=list(self.task.uncertainties)),
        )
