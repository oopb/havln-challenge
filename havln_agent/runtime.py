"""Single-writer Agent Runtime: dispatch, factual state, budgets and legal termination.

It contains no recovery ladder, scan quota, stagnation gate or automatic replan.
"""

import copy
import math
import re
import threading
import time

from .observation import array_hash
from .grounding import project_grounding
from .progress import current_visual_evidence, stair_motion_direction
from .motion import registered_action
from .tools import ToolExecutor
from .safety import forward_probe
from .waypoint import rank_candidates
from .spatial import validity
from .contracts import registry
from .active_observation import ObservationController
from .local_exploration import (covered_view, last_block, local_coverage,
                                predicted_scan, bearing_from_pose, wrap_heading, same_stance)
from .navigation_agent import NavigationAgent
from .task_planner import TaskPlanner, validate_conditions
from .visual_perception import VisualPerception


class AgentRuntime(ToolExecutor):
    allowed_actions = ('STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner = threading.get_ident()
        self.started = time.monotonic()
        self.phase, self.finish_reason = 'INITIALIZE', None
        self.plan_revision, self.active_subgoal_id, self.active_subgoal = 0, None, None
        self.active_waypoint_id = None
        self.navigation_mode = 'OBSERVE'
        self.task_review_requested, self.review_reason = True, 'initial_task'
        self.api_failure = None
        self.navigation_liveness = None
        self.event_callback, self.event_seq = None, 0
        self.task_planner = TaskPlanner(self)
        self.navigation_agent = NavigationAgent(self)
        self.visual_perception = VisualPerception(self)
        self.observation_controller = ObservationController(self)
        self._authorized_action = False
        # Pitch changes only after a successful public environment receipt.
        original_step, original_refresh = self.step_callback, self.refresh_callback
        def step(action):
            observation, done = original_step(action)
            delta = self.camera.tilt_angle if action == 'LOOK_UP' else -self.camera.tilt_angle if action == 'LOOK_DOWN' else 0
            self.motion_controller.pitch_deg += delta
            observation.pitch_deg = self.pitch_deg
            return observation, done
        def refresh():
            observation = original_refresh()
            observation.pitch_deg = self.pitch_deg
            return observation
        self.step_callback, self.refresh_callback = step, refresh

    @property
    def pitch_deg(self):
        return self.motion_controller.pitch_deg

    def make_registry(self):
        return registry(self.camera,min(self.config['tool_max_actions'],
            max(1,self.config['max_steps']-1-len(self.memory.actions))),self.pitch_deg)

    def assert_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('episode_state_has_one_synchronous_writer')

    def event(self, event_type, source='runtime', **data):
        self.assert_owner()
        record = dict(event=event_type, episode_id=self.memory.episode_id,
            step_index=len(self.memory.actions), event_seq=self.event_seq,
            observation_id=(self.memory.current or {}).get('observation_id'),
            plan_revision=self.plan_revision, timestamp=time.time(), run_status=self.phase, source=source, **data)
        self.event_seq += 1
        if self.event_callback:
            self.event_callback(record)

    def accept_observation(self, observation):
        self.assert_owner()
        observation.pitch_deg = self.pitch_deg
        old = self.memory.current
        record = super().accept_observation(observation)
        if old is not record:
            self.memory.note_observation(record)
            self.event('observe', source='public_rgbd', observation_ref=record.get('input_file'), pitch_deg=self.pitch_deg)
        return record

    def refresh(self):
        super().refresh()
        current = self.source()
        scene = self.memory.scene_memory
        if scene and scene['source'].get('image_scope', 'current') == 'current' and all(
                scene['source'][k] == current[k] for k in ('rgb_hash', 'pose')):
            origin = scene['source'].get('semantic_source_observation_id', scene['source']['observation_id'])
            scene['source'].update(current, semantic_source_observation_id=origin,
                revalidated_at=current['captured_at'])
        # Exact current RGB reconfirms a box's semantic referent; metric facts
        # must be projected again whenever the live Depth changes.
        latest = {}
        for grounding in self.memory.groundings:
            latest[grounding['target']] = grounding
        for grounding in latest.values():
            source = grounding['source']
            if (source.get('image_scope', 'current') == 'current'
                    and all(source[k] == current[k] for k in ('rgb_hash', 'pose'))
                    and source['depth_hash'] != current['depth_hash']):
                bound = dict(current, image_scope='current', semantic_source_observation_id=
                    source.get('semantic_source_observation_id', source['observation_id']))
                value = project_grounding(grounding['semantic'], super().geometry(), grounding['target'], bound)
                value['grounding_id'] = 'ground_%04d' % len(self.memory.groundings)
                self.memory.groundings.append(value)
                self.memory.note_semantics(bound, [grounding['target']+': '+value['semantic']['evidence']])
                self.event('grounding_reprojected', source='current_depth', grounding_id=value['grounding_id'],
                    semantic_source_observation_id=bound['semantic_source_observation_id'])

    def set_navigation_mode(self, mode, reason, **detail):
        if mode not in ('OBSERVE', 'MOVE'):
            raise ValueError('unknown_navigation_mode')
        previous = self.navigation_mode
        if previous != mode:
            self.navigation_mode = mode
            self.event('navigation_mode_changed', previous_mode=previous,
                       new_mode=mode, reason=reason, **detail)

    def state_view(self):
        self.assert_owner()
        scene = self.memory.scene_memory
        return dict(architecture='observe_move_v1', navigation_mode=self.navigation_mode,
            episode_id=self.memory.episode_id,
            observation_id=(self.memory.current or {}).get('observation_id'), plan_revision=self.plan_revision,
            active_subgoal=self.active_subgoal, pose=self.motion_controller.pose(), action_budget=self.config['max_steps'],
            actions_remaining=self.config['max_steps']-len(self.memory.actions),
            run_status=self.phase, finish_reason=self.finish_reason,
            progress_state=dict(stage=self.memory.progress_stage,
                completed_subgoals=list(self.memory.task.completed_subgoals)),
            scene_state=dict(validity=validity(scene['source'] if scene else None, self.memory.current,
                time.monotonic(), self.config['max_scene_age_seconds'])))

    def context(self):
        # Full local audit only. Role models use contexts directly.
        return dict(self.state_view(), memory=self.memory.snapshot())

    def set_semantic_goal(self, value=None):
        task = self.memory.task
        i = task.current_subgoal_index
        if value is not None and value.get('intent') == 'continue' and self.active_subgoal_id == 'sg_%d' % i:
            self.task_review_requested = False
            self.event('task_review_continue', source='task_planner', subgoal_id=self.active_subgoal_id)
            return
        if i >= len(task.subgoals):
            self.task_review_requested = False
            return
        goal = task.subgoals[i]
        criteria = list(value['success_criteria'] if value else goal['success_criteria'])
        validate_conditions(goal, criteria, i == len(task.subgoals)-1)
        self.plan_revision += 1
        self.active_subgoal_id = 'sg_%d' % i
        description = value['subgoal'] if value else goal['action']+' '+goal['target']
        goal.update(success_criteria=criteria, semantic_description=description)
        self.active_subgoal = dict(id=self.active_subgoal_id, description=description,
            landmark=goal['landmark'], success_criteria=criteria)
        action = goal['action']
        stair_direction = stair_motion_direction(goal)
        self.motion_policy = ('DESCEND_STAIRS' if stair_direction == 'DOWN' else
            'ASCEND_STAIRS' if stair_direction == 'UP' else 'LEVEL_ONLY')
        self.semantic_motion_policy = self.motion_policy
        self._cancel_local_navigation('semantic_task_revision')
        self.active_waypoint_id = None
        self.set_navigation_mode('OBSERVE', 'semantic_subgoal_changed')
        self.task_review_requested = False
        self.phase = 'NAVIGATING'
        self.event('task_plan', source='task_planner', goal=self.active_subgoal)

    def execute(self, call, decision=None):
        self.assert_owner()
        self.context_version = 'observe_move_v1:%d:%s:%d' % (self.plan_revision,
            self.memory.current['observation_id'], len(self.memory.actions))
        self.event('tool_call', source='registry', tool=call.get('tool'), args=call.get('args'))
        value = super().execute(call, decision)
        self.event('tool_result', source='registry', result=value)
        return value

    def dispatch(self, call, decision):
        if self.done:
            return dict(status='STOPPED', reason=self.stop_reason)
        if not decision or (decision.get('plan_revision'), decision.get('subgoal_id')) != (
                self.plan_revision, self.active_subgoal_id):
            return dict(status='STALE_TARGET', reason='task_revision_mismatch')
        if decision.get('observation_id') != self.memory.current['observation_id']:
            return dict(status='STALE_OBSERVATION', reason='decision_observation_mismatch')
        if self.registry[call['tool']]['permission'] == 'NAVIGATION':
            decided = next((o for o in self.memory.observations
                if o['observation_id'] == decision['observation_id']), None)
            self.refresh()
            if not decided or decided['pose'] != self.motion_controller.pose():
                return dict(status='STALE_TARGET', reason='decision_pose_changed')
            # New sensor data invalidates geometry authorization, not the semantic
            # intent. Motion tools independently recheck current Depth per step.
        moving = call['tool'] == 'navigate_local'
        if moving:
            self.set_navigation_mode('MOVE', 'local_waypoint_selected',
                                     waypoint_id=call['args'].get('waypoint_id'))
        self.phase = 'EXECUTE'
        self._authorized_action = True
        result = None
        try:
            result = super().dispatch(call, decision)
            return result
        finally:
            self._authorized_action = False
            if not self.done:
                self.phase = 'NAVIGATING'
            if moving:
                self.set_navigation_mode('OBSERVE', 'motion_batch_finished',
                                         status=result.get('status') if result else 'INTERRUPTED')

    def finalize_result(self, value, before, observation_before, call):
        moves = [a['result'].get('rgbd_motion') for a in self.memory.actions[before:]
                 if a['action'] == 'MOVE_FORWARD']
        known = bool(moves) and all(m and m['status'] == 'ESTIMATED' for m in moves)
        value.update(translation_m=sum(math.hypot(m['translation_m']['x_right'],
            m['translation_m']['z_forward']) for m in moves) if known else None,
            translation_confidence='estimated' if known else 'unknown')
        vertical=sum(m['translation_m']['y_up'] for m in moves) if known else None
        error=sum(m['uncertainty_m'] for m in moves) if known else None
        value.update(vertical_translation_m_est=vertical,vertical_motion=('DOWN' if vertical < -error else
            'UP' if vertical > error else 'LEVEL') if known else 'UNKNOWN',
            semantic_direction_mismatch=any(a['result'].get('semantic_direction_mismatch',False)
                for a in self.memory.actions[before:]))
        if value['status'] == 'BLOCKED' and value.get('reason') in (
                'terrain_support_unverified', 'human_localization_unverified', 'unknown_depth'):
            value.update(self.motion_controller.needs_view(value['reason']))
        if 'information_options' in value:
            value['information_options'] = [name for name in value['information_options'] if name in self.registry]
        return value

    def _step(self, action):
        self.assert_owner()
        if not self._authorized_action:
            raise ValueError('atomic_action_requires_runtime_authorization')
        value = super()._step(action)
        if value['status'] == 'EXECUTED':
            if action == 'MOVE_FORWARD':
                action_record = self.memory.actions[-1]
                valid = registered_action(action_record, {o['observation_id']:o for o in self.memory.observations})
                action_record['evidence_state'] = 'motion_verified' if valid else 'sensor_recorded'
                self.memory.note_motion()
            self.event('atomic_action', source='environment_receipt', action=action, feedback=value['feedback'])
        return value

    def _prepare_forward(self):
        geometry = self.geometry()
        if geometry.human_status['unlocalized']:
            return self.motion_controller.needs_view('human_localization_unverified')

    def _observe_for_goal(self, objective, target_types, direction, sweep_degrees,
                          checkpoint_degrees=30, look='none'):
        return self.observation_controller.scan(
            objective, target_types, direction, sweep_degrees, checkpoint_degrees, look)

    def _turn_and_observe(self, relative_turn_deg, look, purpose='explore'):
        count = round(abs(relative_turn_deg)/self.camera.turn_angle)
        actions = ['TURN_LEFT' if relative_turn_deg > 0 else 'TURN_RIGHT']*count
        if look != 'none':
            actions.append('LOOK_UP' if look == 'up' else 'LOOK_DOWN')
        if not actions:
            return dict(status='INVALID_ARGUMENTS', reason='choose_a_turn_or_look_runtime_already_refreshes')
        # Reserve room for the requested pitch action so LOOK is never silently dropped.
        if len(actions) > self.config['tool_max_actions']:
            return dict(status='INVALID_ARGUMENTS', reason='choose_1_to_3_actions_then_reconsider_new_views')
        final_pitch=self.pitch_deg+(self.camera.tilt_angle if look=='up' else -self.camera.tilt_angle if look=='down' else 0)
        if abs(final_pitch)>60:
            return dict(status='INVALID_ARGUMENTS',reason='camera_pitch_outside_calibrated_projection')
        if len(actions)>self.config['max_steps']-1-len(self.memory.actions):
            return dict(status='BUDGET_LIMIT',reason='whole_batch_exceeds_remaining_action_budget')
        # Post-collision, do not spend action budget replaying the same
        # absolute camera orientations with no new RGB-D evidence.
        if last_block(self.memory,self.motion_controller.pose()):
            planned=predicted_scan(self.memory,self.motion_controller.pose(),
                relative_turn_deg,look,self.camera.turn_angle,self.camera.tilt_angle)
            if not planned['new_view_expected']:
                exploration=local_coverage(self.memory,self.motion_controller.pose(),
                    self.camera.turn_angle,round(360/self.camera.turn_angle))
                # Turning to a formerly executable heading can still be useful
                # for a NEW live revalidation, even if that view is old.
                endpoint=planned['already_visited'][-1] if planned['already_visited'] else {}
                previous=covered_view(self.memory.observations,self.motion_controller.pose(),
                    self.memory.task.current_subgoal_index,
                    endpoint.get('heading_deg',0),endpoint.get('pitch_deg',0),
                    self.camera.turn_angle)
                previously_free=any(d.get('angle_degrees')==0 and
                    d.get('motion_state')=='EXECUTABLE'
                    for d in (previous or {}).get('directions',[]))
                dynamic_allowed=False
                if purpose=='dynamic_recheck':
                    # Dynamic obstacles may genuinely move. Permit an explicit
                    # revisit, but throttle equivalent sectors using wall time
                    # instead of blindly replaying every LLM request.
                    target_heading=endpoint.get('heading_deg',0)
                    target_pitch=endpoint.get('pitch_deg',self.pitch_deg)
                    now=time.monotonic()
                    previous_checks=[
                        e for e in self.memory.events if e.get('event')=='dynamic_sector_rechecked'
                        and abs((e.get('target_heading_deg',0)-target_heading+180)%360-180)<1
                        and abs(e.get('target_pitch_deg',0)-target_pitch)<1
                        and same_stance(e.get('pose'),self.motion_controller.pose())]
                    dynamic_allowed=(not previous_checks
                        or now-previous_checks[-1].get('time_monotonic',now)<0
                        or now-previous_checks[-1].get('time_monotonic',now)>=60)
                if not previously_free and not dynamic_allowed:
                    return dict(status='NO_NEW_INFORMATION',
                        reason='repeated_camera_sector_after_measured_block',
                        no_new_information_reason='all_requested_absolute_views_already_recorded',
                        scan=dict(frames=[],new_information_ids=[],revisited_view_ids=[
                            v['observation_id'] for v in planned['already_visited']]),
                        exploration=exploration,
                        suggestion='choose_unseen_absolute_heading_or_wait_for_dynamic_change_or_route_review')
        frames = []
        novel, revisited = [], []
        for action in actions:
            result = self._step(action)
            if result['status'] != 'EXECUTED':
                return dict(result,status='PARTIAL' if frames else result['status'],
                    interrupted_status=result['status'],scan=dict(frames=frames))
            current=self.memory.current
            frames.append(dict(observation_id=current['observation_id'],
                heading_deg=math.degrees(self.motion_controller.heading), pitch_deg=self.pitch_deg,
                new_static_information=bool(current.get('new_visual_information',True))))
            (novel if current.get('new_visual_information',True) else revisited).append(
                current['observation_id'])
            if self.done:
                return dict(status='PARTIAL',reason='environment_done',scan=dict(frames=frames))
        if purpose=='dynamic_recheck':
            pose=self.motion_controller.pose()
            self.memory.event('dynamic_sector_rechecked',
                target_heading_deg=bearing_from_pose(pose),target_pitch_deg=self.pitch_deg,
                pose=pose,time_monotonic=time.monotonic(),
                new_static_information=bool(novel))
        return dict(status=('OK' if novel else 'NO_NEW_INFORMATION'),
            reason=None if novel else 'scan_completed_without_new_static_view',
            source=self.source(), scan=dict(frames=frames,
            new_information_ids=novel, revisited_view_ids=revisited),
            exploration=local_coverage(self.memory,self.motion_controller.pose(),
                self.camera.turn_angle,round(360/self.camera.turn_angle)) if not novel else None)

    def _inspect_geometry(self):
        self.refresh()
        geometry=self.geometry()
        check=geometry.assess_step()
        directions=[]
        for degrees in range(-45,46,int(self.camera.turn_angle)):
            item=geometry.assess_step(math.radians(degrees))
            directions.append(dict(bearing_deg=degrees,status=item['state'],reason=item['reason'],
                clearance_m=item['depth']['clearance'],support=item['support']['status']))
        alternatives=[d for d in directions if d['status']=='EXECUTABLE']
        front=dict(check['depth']);front['depth_state']=front.pop('path_state')
        value=dict(status=check['state'],reason=check['reason'],
            geometry=dict(geometry.metadata(),front=dict(front,motion_state=check['state'],
                terrain_support=check['support'],human_evidence=geometry.human_status)),directions=directions,
            alternatives=alternatives,information_options=['get_waypoint_candidates','turn_and_observe'])
        value['exploration']=local_coverage(self.memory,self.motion_controller.pose(),
            self.camera.turn_angle,round(360/self.camera.turn_angle))
        if alternatives:
            value['suggestion']='request_waypoints_then_choose_a_measured_clear_approach_direction'
        elif value['exploration']['unexplored_directions']:
            value['suggestion']='inspect_a_new_absolute_heading_from_exploration; do_not_revisit_blocked_views'
        else:
            value['suggestion']='local_coverage_exhausted_request_route_or_semantic_review'
        return value

    def _compare_views(self, observation_ids, question):
        compared, frames = self.visual_perception.compare(
            [self._vision_frame(i) for i in observation_ids], question)
        sources = {source['observation_id']:source for _, source in frames}
        scenes = []
        for view in compared['views']:
            source = sources[view['observation_id']]
            self.memory.record_historical_scene(view['scene'], source)
            self.memory.note_semantics(source, [l['name'] for l in view['scene'].get('landmarks', [])])
            scenes.append(dict(view, source=source))
        self.memory.visual_changes.extend(compared['changes'])
        return dict(status='OK', scenes=scenes, changes=compared['changes'],
            image_scope='historical_keyframes', usable_for_live_safety=False)

    def _analyze_scene(self, scope='current', observation_ids=None, question=None):
        question = question or 'Which openings, stair entries, goal landmarks and nearby people are visible?'
        if scope == 'historical' and len(observation_ids) > 1:
            return self._compare_views(observation_ids, question)
        observation, source = self._vision_frame(observation_ids[0] if scope == 'historical' else None)
        scene = self.visual_perception.scene(observation, source, question)
        live = scope == 'current' and self._same_live_view(source)
        if live:
            self.memory.record_scene(scene, source)
            self.memory.note_scene(scene, source)
        else:
            self.memory.record_historical_scene(scene, source)
            self.memory.note_semantics(source, [l['name'] for l in scene.get('landmarks', [])])
        return dict(status='OK' if live or scope == 'historical' else 'STALE_OBSERVATION', scene=scene,
            source=source, image_scope='current' if live else 'historical_keyframe', usable_for_live_safety=live)

    def _same_live_view(self, source):
        self.refresh()
        current = self.source()
        if any(source[k] != current[k] for k in ('rgb_hash', 'pose')):
            return False
        source['semantic_source_observation_id'] = source['observation_id']
        source.update(current, image_scope='current', revalidated_at=current['captured_at'])
        return True

    def _ground_landmark(self, target, scope='current', observation_id=None):
        observation, source = self._vision_frame(observation_id if scope == 'historical' else None)
        value = self.visual_perception.grounding(observation, source, target)
        live = scope == 'current' and self._same_live_view(source)
        source['image_scope'] = 'current' if live else 'historical_keyframe'
        if live:
            value = project_grounding(value['semantic'], self.geometry(), target, source)
        value['source'] = dict(source)
        value['grounding_id'] = 'ground_%04d' % len(self.memory.groundings)
        self.memory.groundings.append(copy.deepcopy(value))
        self.memory.note_semantics(source, [target+': '+value['semantic']['evidence']])
        if scope == 'current' and not live:
            return dict(status='STALE_OBSERVATION', reason='view_changed_during_grounding', source=source)
        return value

    def _assess_entry(self, question):
        self.refresh()
        source = self.source()
        value = self.visual_perception.entry(question)
        live = self._same_live_view(source)
        self.memory.note_semantics(source, ['entry: '+str(value)])
        return dict(status='OK' if live else 'STALE_OBSERVATION', assessment=value, source=source)

    def _get_waypoint_candidates(self, target=None, purpose='approach_landmark'):
        original_policy = self.motion_policy
        if purpose == 'explore':
            self.motion_policy = 'LOCAL_TERRAIN'
        try:
            self.refresh()
            value = self._generate_waypoints(target)
            paths = value['paths']
            generated = list(paths)
            if purpose == 'explore':
                for path in paths:
                    distance=min(path['distance'],2*self.camera.forward_step)
                    heading=path['source']['pose']['heading']+path['angle']
                    path.update(type='EXPLORATION_WAYPOINT',distance=distance,
                        endpoint_xz_m=[path['source']['pose']['x']-math.sin(heading)*distance,
                            path['source']['pose']['z']+math.cos(heading)*distance],
                        goal_status='EXPLORATION_ONLY',
                        objective=dict(purpose='explore',goal='obtain a better view for the current subgoal',
                            completes_language_goal=False),
                        execution_requirements=['current_rgbd','live_human_safety','next_step_depth_collision_probe'])
            else:
                # Missing / partial grounding does not forbid short motion.
                # Return every depth-executable candidate for model selection.
                for path in paths:
                    path.update(
                        type=('GOAL_WAYPOINT' if path.get('goal_association')
                              else 'EXPLORATION_WAYPOINT'),
                        objective=dict(purpose='approach_landmark',target=target,
                            estimated_goal_association=bool(path.get('goal_association')),
                            completes_language_goal=False,
                            decision_authority='navigation_agent'))
            # Preserve genuine semantic goal progress in the shortlist. The
            # model still chooses one candidate; no autonomous route policy.
            paths = rank_candidates(paths, purpose)
            offered=paths[:5]
            # Do not replace a model-visible candidate with a fixed straight
            # choice. If useful, straight remains in the ranked options.
            geometry = self.geometry()
            next_step = self.motion_controller.next_step_state(geometry)
            for path in offered:
                path.update(bearing_deg=math.degrees(path['angle']), range_m=path['distance'],
                    source_observation_id=path['source']['observation_id'],
                    first_step=forward_probe(geometry, path['angle'])['state'])
                self.paths[path['id']].update(path)
            offered_ids = {p['id'] for p in offered}
            for path in generated:
                if path['id'] not in offered_ids:
                    self.paths.pop(path['id'], None)
            pose=self.motion_controller.pose()
            coverage=local_coverage(self.memory,pose,self.camera.turn_angle,
                round(360/self.camera.turn_angle))
            view_directions=[]
            revisited_view_directions=[]
            for d in value['inspected_directions']:
                if d.get('motion_state')!='NEEDS_VIEW' or (
                        'outside_observed_fov' not in d.get('rejection_reasons',[])):
                    continue
                absolute=wrap_heading(bearing_from_pose(pose)+d['angle_degrees'])
                previous=covered_view(self.memory.observations,pose,
                    self.memory.task.current_subgoal_index,
                    absolute,pose.get('pitch_deg',0),self.camera.turn_angle)
                entry=dict(bearing_deg=round(d['angle_degrees'],2),
                    absolute_heading_deg=absolute,
                    reason=d.get('reason'),
                    rejection_reasons=d.get('rejection_reasons',[]),
                    required_next='turn_and_observe_then_retry_live_depth_probe',
                    route_authorized=False)
                if previous:
                    entry.update(previous_observation_id=previous['observation_id'],
                        explored_before=True)
                    revisited_view_directions.append(entry)
                else:
                    view_directions.append(entry)
            value.update(paths=offered,purpose=purpose,
                view_directions=view_directions,
                revisited_view_directions=revisited_view_directions[:5],
                exploration=coverage)
            if not offered:
                relevant = [d for d in value['inspected_directions']
                            if 'outside_observed_fov' not in d['rejection_reasons']]
                unknown = not relevant or any(d['path_state'] == 'UNKNOWN' for d in relevant)
                if generated:
                    value.update(status='NEEDS_VIEW',reason='goal_grounding_unavailable',
                        suggestion='choose_get_waypoint_candidates_with_purpose_explore_for_geometric_exploration',
                        information_options=['get_waypoint_candidates','ground_landmark','analyze_scene'])
                elif view_directions:
                    value.update(self.motion_controller.needs_view('unknown_depth_or_fov'))
                    value.update(reason='alternate_directions_need_new_camera_view',
                        suggestion='turn_toward_one_unexplored_absolute_heading_then_recheck_live_geometry',
                        information_options=['turn_and_observe','inspect_geometry'])
                elif coverage['unexplored_directions']:
                    value.update(status='NEEDS_VIEW',reason='no_new_relative_view_candidates',
                        suggestion='use_exploration_unexplored_directions_absolute_heading_not_old_relative_bearings',
                        information_options=['turn_and_observe','inspect_geometry'])
                elif unknown:
                    value.update(status='NO_NEW_INFORMATION',
                        reason='local_view_coverage_exhausted_with_unknown_geometry',
                        suggestion='request_task_review_or_change_observation_pitch_not_repeat_same_scan',
                        information_options=['request_task_review','assess_entry'])
                else:
                    value.update(status='BLOCKED', reason='occupied_local_directions',
                        suggestion='no_new_executable_local_path; explore_distinct_heading_or_request_task_review',
                        information_options=['request_task_review','inspect_geometry'])
            elif not any(p['first_step'] == 'EXECUTABLE' for p in offered):
                value.update(self.motion_controller.needs_view(next_step['reason'] or 'observation_required_after_turn'))
            self.memory.note_candidates(offered)
            return value
        finally:
            self.motion_policy = original_policy

    def _revalidate_path(self, path, resume=False):
        return self.motion_controller.validate(self, path, resume)

    def _navigate_local(self, waypoint_id, max_atomic_steps):
        path = self.paths.get(waypoint_id)
        if path is None:
            return dict(status='INVALID_TARGET', reason='unknown_waypoint_id')
        if path.get('first_step') != 'EXECUTABLE' and self.motion_controller.target is None:
            return dict(status='NEEDS_VIEW', reason='waypoint_has_no_executable_first_step',
                suggestion='turn_to_observe_and_regenerate_waypoints_before_navigate_local')
        if path['source']['subgoal'] != self.memory.task.current_subgoal_index:
            return dict(status='STALE_TARGET', reason='language_goal_changed')
        # Target localization is a hint, not an independent motion veto.
        continuing_uncertain = (self.active_waypoint_id == waypoint_id
            and self.motion_controller.target is not None
            and self.memory.active_attempt is not None
            and self.memory.active_attempt.get('unknown_motion_seen', False))
        # The old global grounding loses localization after an uncertain
        # translation, but the already chosen STRAIGHT relative segment can
        # be finished within its original two-step command horizon. Never use
        # stale grounding to start a new route or certify goal completion.
        if path['type'] not in ('GOAL_WAYPOINT','EXPLORATION_WAYPOINT'):
            return dict(status='INVALID_TARGET',reason='unknown_waypoint_type')
        target = 'current' if self.active_waypoint_id == waypoint_id and self.motion_controller.target is not None else waypoint_id
        original = self.motion_policy
        if path['type']=='EXPLORATION_WAYPOINT':
            self.motion_policy = 'LOCAL_TERRAIN'
        try:
            result = self._execute_waypoint(target, max_atomic_steps)
            self.active_waypoint_id = (waypoint_id if self.motion_controller.target is not None else None)
            return dict(result,waypoint_type=path['type'],waypoint_arrival_completes_task=False)
        finally:
            self.motion_policy = original

    def _recall_observation(self, **args):
        return dict(status='OK', memory=self.memory.recall_observation(**args))

    def _query_spatial_memory(self, **args):
        return dict(status='OK', memory=self.memory.query_spatial_memory(**args))

    def _read_memory_record(self, record_id):
        return dict(status='OK', memory=self.memory.read_memory_record(record_id))

    def _query_task_memory(self, **args):
        return dict(status='OK', memory=self.memory.query_task_memory(**args))

    def _verify_subgoal(self, claim, evidence, observation_ids):
        value = self._update_progress(claim, evidence, observation_ids)
        if value['completion_confirmed']:
            self.active_waypoint_id = None
            self.task_review_requested = self.memory.task.current_subgoal_index < len(self.memory.task.subgoals)
            self.review_reason = 'verified_milestone'
        return value

    def _request_task_review(self, reason):
        if self.memory.task.current_subgoal_index >= len(self.memory.task.subgoals):
            return dict(status='REJECTED', reason='all_subgoals_verified_request_finish')
        self.task_review_requested, self.review_reason = True, reason
        return dict(status='OK', reason=reason, control_event='task_review_requested')

    def _request_finish(self, outcome, evidence):
        # The shared independent verifier alone can turn a success claim into STOP.
        return self._finish_navigation(outcome, evidence)

    def assess(self, value):
        if value['status'] in ('API_ERROR', 'API_UNAVAILABLE'):
            self.api_failure = value.get('api_failure', dict(category=value.get('category'), reason=value.get('reason')))
            self.terminate('api_error' if value['status'] == 'API_ERROR' else 'api_budget_exhausted')
        if value['status'] == 'TARGET_REACHED':
            self.active_waypoint_id = None
        # UNKNOWN, repeated views, failures and zero motion are feedback only.

    def terminate(self, reason, model_requested=False):
        self.assert_owner()
        self.finish_reason = ('goal_completed' if reason == 'goal_completed' else
            'action_budget_exhausted' if reason == 'action_budget_exhausted' else
            'navigation_no_new_evidence' if reason == 'navigation_no_new_evidence' else
            'navigation_inference_budget_exhausted' if reason == 'navigation_inference_budget_exhausted' else
            'model_aborted' if reason == 'model_aborted' else 'technical_failure')
        self._authorized_action = True
        try:
            super().terminate(reason, model_requested)
        finally:
            self._authorized_action = False
        self.active_waypoint_id = None
        self.phase = 'FINISH'
        self.event('finish_verdict', finish_reason=self.finish_reason, reason=reason,
            official_success='requires_independent_scorer')
