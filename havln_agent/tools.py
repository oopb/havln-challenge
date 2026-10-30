"""Model-invoked navigation tools; no autonomous exploration or recovery policy."""

import copy
import math
import time

from jsonschema import Draft7Validator, ValidationError

from .api_client import APIError, BudgetExceeded
from .contracts import registry, tool_result_schema
from .memory_records import task_tokens
from .memory_service import MemoryService
from .local_motion_controller import LocalMotionController
from .observation import array_hash, frame_info
from .perception import Geometry, Observation
from .progress import verify_progress, current_visual_evidence
from .safety import check_action
from .waypoint import generate_waypoints, goal_relation
from .motion import estimate_translation, landing_support
from .spatial import goal_estimate


class ToolExecutor:
    controller_type = LocalMotionController
    allowed_actions = ('STOP','MOVE_FORWARD','TURN_LEFT','TURN_RIGHT','LOOK_UP','LOOK_DOWN')

    def __init__(
        self,
        client,
        camera,
        config,
        step,
        refresh,
        record_step=None,
        record_tool=None,
        record_observation=None,
        load_observation=None,
    ):
        self.client, self.camera, self.config = client, camera, config
        self.step_callback, self.refresh_callback = step, refresh
        self.record_step, self.record_tool = record_step, record_tool
        self.record_observation = record_observation
        self.load_observation = load_observation
        self.frames = {}
        self.motion_controller = self.controller_type(camera)
        self.memory = MemoryService(self)
        self.memory.human_max_age_seconds = config.get('max_human_age_seconds', 2)
        self.registry = self.make_registry()
        self.observation = None
        self.paths = {}
        self.done, self.stop_reason = False, None
        self.tool_id = None
        self.decisions = 0
        self.context_version = None
        self.active_subgoal_id = None
        self.motion_policy = 'LEVEL_ONLY'
        self.executable_segment = None
        self.decision_counts = dict(task_planner=0, navigation_agent=0)
        self.current_decision_id = None

    def make_registry(self):
        return registry(self.camera, self.config['tool_max_actions'])

    def decision_id(self, role):
        identifier = '%s_%04d' % (role, self.decision_counts[role])
        self.decision_counts[role] += 1
        self.current_decision_id = identifier
        self.client.trace_context = dict(decision_id=identifier, subgoal_id=self.active_subgoal_id)
        return identifier

    def geometry(self):
        geometry = Geometry(self.observation, self.camera, self.config['safety_margin'])
        scene = self.memory.scene_memory
        if (
            scene
            and scene['source']['rgb_hash'] == array_hash(self.observation.rgb)
            and scene['source']['pose'] == self.motion_controller.pose()
            and scene['source'].get('image_scope', 'current') == 'current'
            and 0 <= time.monotonic()-self.observation.captured_at <= self.config['max_depth_age_seconds']
        ):
            geometry.set_humans(scene['scene'])
            # Classification is reused only on byte-identical, freshly read RGB, never on old positions.
            geometry.human_status.update(validated_observation_id=(self.memory.current or {}).get('observation_id'),
                semantic_source_observation_id=scene['source']['observation_id'],
                semantic_source_age_seconds=max(0,time.monotonic()-scene['source']['captured_at']),
                validated_rgbd_captured_at=self.observation.captured_at,
                evidence='exact_current_rgb_and_pose_match_with_current_depth')
        return geometry

    def accept_observation(self, observation):
        previous_frame, previous = self.observation, self.memory.current
        self.observation = observation
        pose = self.motion_controller.pose()
        # Sensor refresh is allowed to read the same RGB-D many times without
        # moving. Do not rerun 13 expensive floor/geometry sweeps when the
        # frame and pose are byte-identical to the already evaluated view.
        unchanged = (previous is not None and previous_frame is not None
            and previous['pose'] == pose
            and array_hash(previous_frame.rgb) == array_hash(observation.rgb)
            and array_hash(previous_frame.depth) == array_hash(observation.depth))
        if unchanged:
            diagnostics = previous.get('directions', [])
        else:
            diagnostics = []
            generate_waypoints(self.geometry(), self.config['waypoint_distance'], diagnostics)
        record = self.memory.observe(observation, pose, diagnostics)
        if self.record_observation and record is not previous:
            self.record_observation(record, observation)
        if self.load_observation is None and record is not previous:
            self.frames[record['observation_id']] = Observation(
                observation.rgb.copy(), observation.depth.copy(), observation.instruction, observation.captured_at,
                observation.pitch_deg)
        return record

    def source(self):
        return dict(
            observation_id=self.memory.current['observation_id'],
            rgb_hash=array_hash(self.observation.rgb),
            depth_hash=array_hash(self.observation.depth),
            captured_at=self.observation.captured_at,
            pose=self.motion_controller.pose(),
            subgoal=self.memory.task.current_subgoal_index,
        )

    def refresh(self):
        # Same-view public sensor read only: no free panorama, hidden pose or clock manipulation.
        try:
            observation = self.refresh_callback()
        except (AttributeError, RuntimeError, ValueError):
            raise APIError('current_public_observation_unavailable') from None
        self.accept_observation(observation)

    def estimate_goal(self, grounding):
        return goal_estimate(grounding, self.memory, self.motion_controller.pose(), time.monotonic(),
                             self.config['max_scene_age_seconds'],
                             self.config.get('max_static_landmark_age_seconds', 300),
                             self.config.get('max_human_age_seconds', 2))

    def execute(self, call, decision=None):
        started=time.monotonic()
        before=len(self.memory.actions)
        observation_before=self.memory.current['observation_id']
        self.tool_id='tool_%04d' % len(self.memory.tool_ledger)
        self.client.trace_context=dict(decision_id=self.current_decision_id,subgoal_id=self.active_subgoal_id,tool_call_id=self.tool_id)
        context=dict(context_version=self.context_version,observation_id=observation_before,
            pose=self.motion_controller.pose(),requested_at=started,subgoal_id=self.active_subgoal_id,
            actor=(decision or {}).get('actor','navigation_agent'),decision_id=self.current_decision_id)
        try:
            self.registry=self.make_registry()
            if set(call)!={'tool','args'} or call['tool'] not in self.registry:
                raise ValueError('unknown_tool_or_envelope')
            Draft7Validator(self.registry[call['tool']]['parameters']).validate(call['args'])
            result=self.dispatch(call,decision)
        except (ValidationError,ValueError) as error:
            result=dict(status='INVALID_ARGUMENTS',reason=str(error)[:300])
        except BudgetExceeded as error:
            result=dict(status='API_UNAVAILABLE',reason=str(error),category='budget_exhausted')
        except APIError as error:
            visual=call.get('tool') in ('analyze_scene','compare_views','ground_landmark','assess_entry')
            result=dict(status='TOOL_ERROR' if visual else 'API_ERROR',recoverable=visual,
                information_options=['inspect_geometry','get_waypoint_candidates','turn_and_observe'] if visual else [],
                reason=str(error),category=error.category,
                api_failure=getattr(error,'request_context',dict(error=str(error),category=error.category)))
        except (RuntimeError,OSError) as error:
            result=dict(status='TOOL_ERROR',reason=str(error)[:300],error_type=type(error).__name__)
        actions=[a['action'] for a in self.memory.actions[before:]]
        if actions and result['status'] not in ('OK','PARTIAL','TARGET_REACHED','FINISHED','EXECUTED'):
            result.update(interrupted_status=result['status'],status='PARTIAL')
        args=call.get('args',{})
        requested=args.get('max_atomic_steps',0)
        if call.get('tool')=='turn_and_observe':
            requested=round(abs(args.get('relative_turn_deg',0))/self.camera.turn_angle)+int(args.get('look','none')!='none')
        elif call.get('tool')=='observe_for_goal':
            # Upper bound: sweep TURN actions, optional initial LOOK and at
            # most one VLM-guided pitch refinement, all counted atomically.
            requested=round(args.get('sweep_degrees',0)/self.camera.turn_angle)+1+int(args.get('look','none')!='none')
        elif call.get('tool')=='request_finish':
            requested=1
        if type(requested) is not int or requested<0:requested=0
        result.update(tool_call_id=self.tool_id,tool=call.get('tool'),requested_actions=requested,
            executed_actions=len(actions),executed_motion=actions,start_observation_id=observation_before,
            end_observation_id=self.memory.current['observation_id'],seconds=time.monotonic()-started,
            request_context=context)
        result=self.finalize_result(result,before,observation_before,call)
        try:
            Draft7Validator(tool_result_schema()).validate(result)
        except ValidationError:
            result.update(status='TOOL_ERROR',reason='tool_output_schema_violation')
        record=self.memory.record_tool(call,result,context)
        if self.record_tool:self.record_tool(record)
        self.client.trace_context=dict(decision_id=self.current_decision_id,subgoal_id=self.active_subgoal_id)
        return result

    def dispatch(self, call, decision):
        return getattr(self, '_' + call['tool'])(**call['args'])

    def finalize_result(self, result, before, observation_before, call):
        return result

    def _prepare_forward(self):
        return None

    def _step(self, action):
        if action not in self.allowed_actions:
            raise ValueError('illegal_tool_action')
        if self.done:
            return dict(status='STOPPED', reason=self.stop_reason)
        if action != 'STOP' and len(self.memory.actions) >= self.config['max_steps'] - 1:
            return dict(status='BUDGET_LIMIT', reason='action_budget_reserved_for_stop')
        if action == 'MOVE_FORWARD':
            try:
                self.refresh()
            except (APIError, AttributeError, RuntimeError, ValueError):
                return dict(status='STALE_OBSERVATION', reason='fresh_public_depth_unavailable')
            if time.monotonic() - self.observation.captured_at > self.config['max_depth_age_seconds']:
                return dict(status='STALE_OBSERVATION', reason='depth_age_limit')
            pending = self._prepare_forward()
            if pending:
                return pending
        # A turn/look/STOP has no forward body sweep. Computing 13 waypoint
        # terrain reviews here used to burn CPU even for STOP. Candidates are
        # created by get_waypoint_candidates, not by the atomic executor.
        geometry = self.geometry() if action == 'MOVE_FORWARD' else None
        diagnostics, candidates = [], []
        safety_check = geometry.assess_step() if geometry is not None else None
        front = safety_check['depth'] if safety_check is not None else None
        allowed, reason = check_action(action, geometry)
        if allowed is None:
            self.memory.event('safety_rejected',requested_action=action,reason=reason,geometry=front)
            return dict(status=safety_check['state'],reason=reason,safety=dict(front,
                execution_allowed=False,execution_reason=reason,terrain_evidence=safety_check['support']))
        before, before_record = self.observation, dict(self.memory.current)
        pose = self.motion_controller.pose()
        checked_at = time.monotonic()
        after, done = self.step_callback(action)
        motion = estimate_translation(before, after, self.camera, geometry.human_mask) \
            if action == 'MOVE_FORWARD' else None
        changed_depth = array_hash(after.depth) != array_hash(before.depth)
        changed_rgb = array_hash(after.rgb) != array_hash(before.rgb)
        self.motion_controller.accept_feedback(action,motion)
        feedback = dict(
            visual_change=changed_depth or changed_rgb,
            depth_changed=changed_depth,
            rgb_changed=changed_rgb,
            translation_supported=bool(motion and motion['status'] == 'ESTIMATED'),
            displacement_verified=False,
            vertical_motion=motion.get('vertical_motion', 'UNKNOWN') if motion else 'UNKNOWN',
            observation_before=before_record['observation_id'],
            observation_after=None,
            rgbd_motion=motion,
            landing_support=landing_support(Geometry(after, self.camera, self.config['safety_margin']))
            if action == 'MOVE_FORWARD' else None,
            goal_association=(self.memory.active_attempt['target'].get('goal_association', [])
                              if self.memory.active_attempt else []),
        )
        self.memory.record_action(action, self.motion_controller.pose(), feedback)
        self.accept_observation(after)
        feedback['observation_after'] = self.memory.current['observation_id']
        if action == 'MOVE_FORWARD':
            policy = self.motion_policy
            direction = motion.get('vertical_motion', 'UNKNOWN')
            expected = {'LEVEL_ONLY': 'LEVEL', 'DESCEND_STAIRS': 'DOWN', 'ASCEND_STAIRS': 'UP','LOCAL_TERRAIN':direction}[policy]
            feedback['semantic_direction_mismatch']=direction not in ('LEVEL',expected)
            alert_reason=('motion_estimate_unknown' if motion['status']!='ESTIMATED' else
                'measured_vertical_conflicts_with_local_support' if abs(motion['translation_m']['y_up']-
                    (geometry.assess_step()['support'].get('end_height_m',0)-geometry.assess_step()['support'].get('initial_height_m',0)))
                    >self.camera.max_step_height+motion.get('uncertainty_m',0) else None)
            if alert_reason:
                alert = dict(action_step=len(self.memory.actions)-1, reason=alert_reason, motion_policy=policy,
                    observation_before=feedback['observation_before'], observation_after=feedback['observation_after'],
                    measurement=motion, displacement_verified=False)
                feedback['motion_alert'] = alert
                self.memory.event('motion_attention_required', **alert)
        self.done = bool(done) or action == 'STOP'
        record = dict(
            step=len(self.memory.actions) - 1,
            tool_call_id=self.tool_id,
            subgoal_id=self.active_subgoal_id,
            action=action,
            executed_action=action,
            proposed_action=action,
            pose=pose,
            pose_after=self.motion_controller.pose(),
            geometry=geometry.metadata() if geometry is not None else None,
            candidates=candidates,
            waypoint_diagnostics=diagnostics,
            observation=frame_info(before, self.camera),
            post_observation=frame_info(after, self.camera),
            observation_age_seconds=checked_at - before.captured_at,
            safety=dict(
                status='APPROVED',
                checked_action=action,
                front=front,
                captured_at=before.captured_at,
                checked_at=checked_at,
                scope=('current_depth_next_step_body_and_local_support' if geometry is not None
                       else 'no_forward_displacement'),
                human_semantics=geometry.human_status if geometry is not None else None,
                terrain_evidence=safety_check['support'] if safety_check is not None else None,
                dynamic_motion_prediction='UNAVAILABLE',
            ),
            scene=self.memory.scene_memory['scene'] if self.memory.scene_memory else None,
            scene_matches_current_rgb=bool(
                self.memory.scene_memory
                and self.memory.scene_memory['source']['rgb_hash'] == array_hash(before.rgb)
            ),
            action_pipeline=[
                dict(stage='tool_request' if self.tool_id else 'termination', action=action,
                     reason='llm_requested_tool' if self.tool_id else self.stop_reason),
                dict(
                    stage='safety_shield',
                    action=action,
                    reason='live_depth_approved' if action == 'MOVE_FORWARD' else 'turn_or_stop',
                ),
                dict(stage='executor', action=action, reason='environment_step_completed'),
            ],
            action_changes=[],
            stop_reason=self.stop_reason,
            done=self.done,
            motion_evidence=motion,
            terrain_evidence=feedback['landing_support'],
            decision_context_version=self.context_version,
        )
        if self.record_step:
            self.record_step(record, before, after)
        return dict(status='EXECUTED', reason=None, feedback=feedback)

    def terminate(self, reason, model_requested=False):
        self.stop_reason = reason
        if self.memory.active_attempt:
            self.paths[self.memory.active_attempt['target']['id']]['execution_status'] = 'INTERRUPTED'
        self.memory.finish_attempt('INTERRUPTED', reason)
        self.motion_controller.target = None
        if not model_requested:
            self.tool_id = None
        if not self.done:
            self._step('STOP')

    def _vision_frame(self, observation_id=None):
        if observation_id is None:
            self.refresh()
            return self.observation, self.source()
        record = next((o for o in self.memory.observations if o['observation_id'] == observation_id), None)
        if record is None:
            raise ValueError('unknown_saved_observation_id')
        observation = self.load_observation(record) if self.load_observation else self.frames[observation_id]
        source = {k: record[k] for k in ('observation_id', 'rgb_hash', 'depth_hash', 'captured_at', 'pose')}
        source.update(subgoal=record.get('subgoal'), image_scope='historical_keyframe')
        return observation, source

    def _generate_waypoints(self, target=None):
        self.refresh()
        geometry, diagnostics = self.geometry(), []
        source = self.source()
        groundings, excluded = [], []
        for original in self.memory.groundings:
            if original['source']['subgoal'] != self.memory.task.current_subgoal_index or not (
                target is None or target == original['grounding_id'] or
                task_tokens(target) & task_tokens(original['target'])
            ):
                continue
            if original['source'].get('image_scope','current')!='current':
                continue
            estimate = self.estimate_goal(original)
            if estimate['requires_reconfirmation']:
                excluded.append(dict(grounding_id=original['grounding_id'], reasons=estimate['reasons']))
            else:
                groundings.append(dict(original, current_estimate=estimate))
        angles = None
        if groundings:
            stride = self.camera.turn_angle
            base = list(range(-int(self.camera.hfov / 2 / stride), int(self.camera.hfov / 2 / stride) + 1))
            angles = [i * stride for i in base] + [g['current_estimate']['bearing_deg'] for g in groundings]
            angles = sorted(set(round(a, 4) for a in angles))
        candidates = generate_waypoints(geometry, self.config['waypoint_distance'], diagnostics, angles)
        # Offer extra depth-goal distance variants, without selecting one for the LLM.
        for candidate in list(candidates):
            for grounding in groundings:
                estimate = grounding['current_estimate']
                if abs(math.degrees(candidate['angle']) - estimate['bearing_deg']) > self.camera.turn_angle:
                    continue
                distance = math.floor(min(candidate['distance'], math.hypot(
                    estimate['x_right_m'], estimate['z_forward_m'])) / self.camera.forward_step) * self.camera.forward_step
                if self.camera.forward_step <= distance < candidate['distance']:
                    candidates.append(dict(candidate, distance=distance, distance_source='measured_goal_region'))
        paths = []
        # Descriptive facts are returned in angular order; no weighted policy ranking or blacklist.
        for candidate in candidates:
            identifier = 'path_%04d_%02d' % (len(self.memory.tool_ledger), len(paths))
            language_goal = (self.memory.task.subgoals[self.memory.task.current_subgoal_index]
                             if self.memory.task.current_subgoal_index < len(self.memory.task.subgoals) else {})
            relations = [goal_relation(candidate, g, g['current_estimate'], language_goal,
                                       self.camera.turn_angle) for g in groundings]
            associations = [relation for relation in relations if relation['relevant']]
            for association in associations:
                value = next(g for g in groundings if g['grounding_id'] == association['grounding_id'])
                association['stair_direction_hypothesis'] = value['semantic']['stair_direction']
                association['measured_height_slope'] = value['position'].get('height_slope')
                association['vertical_motion_verified'] = False
            failures = []
            heading = source['pose']['heading'] + candidate['angle']
            for attempt in self.memory.experiences[-24:]:
                previous = attempt['target']
                origin = previous['source']['pose']
                delta = (heading - origin['heading'] - previous['angle'] + math.pi) % (2 * math.pi) - math.pi
                if (
                    (attempt['result'] == 'BLOCKED' or any(e['status'] == 'BLOCKED' for e in attempt.get('interruptions', [])))
                    and abs(delta) <= math.radians(self.camera.turn_angle)
                    and math.hypot(origin['x'] - source['pose']['x'], origin['z'] - source['pose']['z'])
                    <= 0.35
                ):
                    failures.append(
                        dict(
                            attempt_id=attempt['attempt_id'],
                            reason=attempt['reason'],
                            basis='estimated_local_pose_and_heading',
                            confidence='low',
                            suppressed=False,
                        )
                    )
            path = dict(
                candidate,
                id=identifier,
                source=source,
                geometric_status='CANDIDATE',
                goal_association=associations,
                terrain_status='UNVERIFIED',
                connectivity='single_observed_straight_corridor',
                revisit_risk=self.memory.current['revisit_matches'],
                failure_history=failures[-4:],
                human_risk=geometry.human_status,
                execution_status='PROPOSED',
                goal_status='GOAL_RELEVANT' if associations else 'GOAL_DIRECTION_UNKNOWN',
                frame='source_robot_camera', angle_unit='radians',
                endpoint_xz_m=[source['pose']['x'] - math.sin(heading) * candidate['distance'],
                               source['pose']['z'] + math.cos(heading) * candidate['distance']],
                association_rejections=[r for r in relations if not r['relevant']] + excluded,
                execution_requirements=['reconfirm_uncertain_goal', 'current_rgbd', 'live_human_safety'],
            )
            self.paths[identifier] = path
            paths.append(copy.deepcopy(path))
        return dict(
            status='OK' if paths else 'NEEDS_VIEW',
            paths=paths,
            inspected_directions=diagnostics,
            target=target,
            source=source,
            unknown_space='outside_current_fov_and_unmeasured_depth',
            map_connectivity=self.memory.traversed_routes(),
            association_exclusions=excluded,
            safety_state=dict(visible_floor=landing_support(geometry), humans=geometry.human_status,
                authorization='requires_new_checks_before_every_move', dynamic_motion_prediction='UNAVAILABLE'),
            goal_status='GOAL_RELEVANT' if any(p['goal_association'] for p in paths) else 'GOAL_DIRECTION_UNKNOWN',
            goal_localization_status='GROUNDING_UNCERTAIN' if excluded else 'AVAILABLE' if groundings else 'UNKNOWN',
        )

    def _cancel_local_navigation(self, reason):
        self.active_waypoint_id = None
        if self.memory.active_attempt:
            path = self.paths.get(self.memory.active_attempt['target']['id'])
            if path:
                path['execution_status'] = 'CANCELLED'
        self.memory.finish_attempt('CANCELLED', reason)
        self.motion_controller.target = None
        self.executable_segment = None
        return dict(status='CANCELLED', reason=reason)

    def _navigation_feedback(self, path, checked, status, reason=None, **extra):
        endpoint = checked['original_target']
        attempt = self.memory.active_attempt
        uncertain = bool(attempt and attempt.get('unknown_motion_seen'))
        # The old endpoint's distance is not measurable once an UNKNOWN RGB-D
        # translation has broken the pose chain. Report None, not false meters.
        remaining = (None if uncertain else math.hypot(
            endpoint[0] - self.motion_controller.x,
            endpoint[1] - self.motion_controller.z))
        reached = status == 'TARGET_REACHED'
        path['execution_status'] = 'IN_PROGRESS' if status == 'PARTIAL' else status
        # The old endpoint is invalid after uncertain translation or a confirmed
        # collision. Keep PARTIAL/NEEDS_VIEW for same-path continuation, but do
        # not offer a stale waypoint as an executable route.
        if reached or status in ('INTERRUPTED','BLOCKED','REQUIRES_ATTENTION','STALE_TARGET'):
            self.memory.finish_attempt(status, reason)
            self.motion_controller.target = None
        elif self.memory.active_attempt:
            attempt.update(result='IN_PROGRESS' if status == 'PARTIAL' else status, reason=reason)
            if status != 'PARTIAL':
                attempt['interruptions'].append(dict(status=status, reason=reason,
                                                      observation_id=self.memory.current['observation_id']))
        if (uncertain and status == 'REQUIRES_ATTENTION'
                and reason == 'relative_forward_horizon_exhausted_without_verified_pose'):
            # Current post-action RGB-D is a NEW local anchor only; it does not
            # recover the old global displacement or certify endpoint arrival.
            anchor = self.memory.current['observation_id']
            established = self.motion_controller.anchor_relative_frame(anchor)
            if 'recovery' in extra:
                extra['recovery'].update(
                    local_anchor_observation_id=anchor if established else None,
                    coordinate_scope='new_local_relative_only',
                    old_endpoint_valid=False,
                    requires_new_waypoint=True,
                    missing_translation_still_unknown=True)
        return dict(status=status, reason=reason, target_reached=reached, displacement_verified=False,
                    endpoint_estimate_valid=not uncertain,
                    reach_basis=('unverified_relative_actions' if uncertain else
                                 'registered_rgbd_translation_estimate'),
                    original_target=list(endpoint),
                    pending_target=list(self.motion_controller.target) if self.motion_controller.target is not None else None,
                    remaining_distance=remaining,
                    remaining_path=dict(path_id=path['id'], original_endpoint_xz_m=list(endpoint),
                                        endpoint_estimate_valid=not uncertain,
                                        status=path['execution_status'], completed=reached),
                    executable_segment=self.executable_segment or {},
                    validation=checked.get('validation', {}), **extra)

    def _execute_waypoint(self, target, max_actions):
        if target != 'current':
            path = self.paths.get(target)
            if path is None:
                return dict(status='INVALID_TARGET', reason='unknown_tool_generated_path_id')
            if path.get('execution_status') == 'REQUIRES_ATTENTION':
                return dict(status='REQUIRES_ATTENTION', reason='previous_motion_requires_new_local_path_or_cancellation')
            if path.get('first_step') != 'EXECUTABLE':
                return dict(status='NEEDS_VIEW', reason='waypoint_requires_view_not_navigation',
                    suggestion='use_turn_and_observe_then_inspect_geometry_before_requesting_new_waypoints')
            checked = self._revalidate_path(path)
            if checked['status'] != 'OK':
                return checked
            self._cancel_local_navigation('model_selected_new_target')
            self.memory.start_attempt(path, self.motion_controller.pose())
            # Preserve the original selected endpoint; a checked fragment never replaces it.
            self.motion_controller.target = tuple(checked['original_target'])
            self.memory.active_attempt['original_endpoint_xz_m'] = list(self.motion_controller.target)
            # A waypoint is at most two atomic forward steps in the current
            # generator. A failed visual registration must never create an
            # unlimited blind-movement budget or fake measured distance.
            self.memory.active_attempt['forward_action_horizon'] = max(1, math.ceil(
                (path['distance'] - 1e-6) / self.camera.forward_step))
            self.memory.active_attempt['forward_actions'] = 0
            self.memory.active_attempt['unknown_motion_seen'] = False
            self.memory.active_attempt['unknown_observation_ids'] = []
        elif self.motion_controller.target is None:
            return dict(status='INVALID_TARGET', reason='no_pending_target')
        else:
            path = self.paths[self.memory.active_attempt['target']['id']]
            checked = dict(original_target=list(self.motion_controller.target))
            if path.get('execution_status') == 'REQUIRES_ATTENTION':
                return self._navigation_feedback(path, checked, 'REQUIRES_ATTENTION',
                    'previous_motion_requires_new_local_path_or_cancellation')
        for _ in range(max_actions):
            attempt = self.memory.active_attempt
            unverified = bool(attempt and attempt.get('unknown_motion_seen'))
            if unverified and attempt['forward_actions'] >= attempt['forward_action_horizon']:
                # We have performed the bounded relative route, but cannot
                # claim target arrival based on commanded displacement alone.
                return self._navigation_feedback(path, checked, 'REQUIRES_ATTENTION',
                    'relative_forward_horizon_exhausted_without_verified_pose',
                    recovery=dict(mode='REANCHOR_CURRENT_RGBD',
                        executed_forward_actions=attempt['forward_actions'],
                        horizon=attempt['forward_action_horizon'],
                        unknown_observation_ids=attempt['unknown_observation_ids'],
                        next_step='new_local_waypoints_from_current_rgbd_not_repeat_semantic_scan'))
            # After a failed visual translation, x/z are deliberately not
            # advanced. Continue only a straight, short-horizon intent under
            # NEW depth and collision checks; do not steer toward stale x/z.
            action = 'MOVE_FORWARD' if unverified else self.motion_controller.next_action()
            if action is None:
                return self._navigation_feedback(path, checked, 'TARGET_REACHED', 'original_endpoint_reached')
            validation = (self.motion_controller.validate(self, path, resume=True,
                uncertain_straight=True) if unverified else self._revalidate_path(path, resume=True))
            if validation['status'] != 'OK':
                checked.update(validation)
                self.executable_segment = None
                return self._navigation_feedback(path, checked, validation['status'], validation.get('reason'),
                                                 **({'safety': validation['safety']} if 'safety' in validation else {}))
            checked = validation
            self.executable_segment = checked['executable_segment']
            result = self._step(action)
            if result['status'] != 'EXECUTED':
                self.executable_segment = None
                return self._navigation_feedback(path, checked, result['status'], result.get('reason'),
                                                 **({'safety': result['safety']} if 'safety' in result else {}))
            if action == 'MOVE_FORWARD':
                attempt = self.memory.active_attempt
                if attempt:
                    attempt['forward_actions'] += 1
                alert = result['feedback'].get('motion_alert') or {}
                if alert and alert.get('reason') != 'motion_estimate_unknown':
                    self.executable_segment = None
                    return self._navigation_feedback(path, checked, 'REQUIRES_ATTENTION',
                        alert['reason'], motion_alert=alert)
                if not result['feedback']['translation_supported']:
                    if attempt:
                        attempt['unknown_motion_seen'] = True
                        attempt['unknown_observation_ids'].append(
                            self.memory.current['observation_id'])
                    # No hidden ground-truth pose and no assumed 0.25m update.
                    # The NEXT action, if any, gets a new live RGB-D safety
                    # validation, and the fixed horizon caps total forwards.
                    self.executable_segment = None
            if self.done:
                return self._navigation_feedback(path, checked, 'INTERRUPTED', 'environment_done')
        attempt = self.memory.active_attempt
        if attempt and attempt.get('unknown_motion_seen'):
            if attempt['forward_actions'] >= attempt['forward_action_horizon']:
                return self._navigation_feedback(path, checked, 'REQUIRES_ATTENTION',
                    'relative_forward_horizon_exhausted_without_verified_pose',
                    recovery=dict(mode='REANCHOR_CURRENT_RGBD',
                        executed_forward_actions=attempt['forward_actions'],
                        horizon=attempt['forward_action_horizon'],
                        unknown_observation_ids=attempt['unknown_observation_ids'],
                        next_step='new_local_waypoints_from_current_rgbd_not_repeat_semantic_scan'))
            return self._navigation_feedback(path, checked, 'PARTIAL',
                'tool_action_limit_after_unverified_translation',
                recovery=dict(mode='CONTINUE_BOUNDED_RELATIVE_MOTION',
                    executed_forward_actions=attempt['forward_actions'],
                    horizon=attempt['forward_action_horizon'],
                    remaining_safe_forward_attempts=(
                        attempt['forward_action_horizon']-attempt['forward_actions']),
                    requires_fresh_rgbd_before_next_step=True,
                    endpoint_estimate_valid=False))
        reached = self.motion_controller.next_action() is None
        return self._navigation_feedback(path, checked, 'TARGET_REACHED' if reached else 'PARTIAL',
                                         'original_endpoint_reached' if reached else 'tool_action_limit')

    def _update_progress(self, claim, evidence, observation_ids):
        self.refresh()
        result = verify_progress(
            self.memory,
            claim,
            evidence,
            observation_ids,
            self.config['max_scene_age_seconds'],
            self.config['waypoint_distance'],
            self.config.get('max_static_landmark_age_seconds', 300),
            self.config.get('max_human_age_seconds', 2),
        )
        if result['completion_confirmed']:
            self._cancel_local_navigation('verified_language_goal_completed')
        return result

    def _finish_navigation(self, outcome, evidence):
        if outcome == 'success' and len(self.memory.task.completed_subgoals) != len(
            self.memory.task.subgoals
        ):
            return dict(status='INSUFFICIENT_EVIDENCE', reason='ordered_subgoals_not_verified')
        if outcome == 'success':
            self.refresh()
            scene = self.memory.scene_memory
            if (
                not self.memory.task.subgoals
                or not scene
                or not current_visual_evidence(scene['source'], self.memory, time.monotonic(),
                                               self.config['max_scene_age_seconds'])
                or not scene['scene']['final_target_visible']
                or not scene['scene']['final_target_evidence'].strip()
            ):
                return dict(status='INSUFFICIENT_EVIDENCE', reason='fresh_final_arrival_evidence_required')
            final = self.memory.task.subgoals[-1]
            wanted = task_tokens(final['landmark'] + ' ' + final['target'])
            nearby = [
                g
                for g in self.memory.groundings
                if g['status'] == 'LOCALIZED'
                and current_visual_evidence(g['source'], self.memory, time.monotonic(),
                                            self.config['max_scene_age_seconds'])
                and wanted & task_tokens(g['target'])
                and g['position']['status'] == 'LOCALIZED'
                and not self.estimate_goal(g)['requires_reconfirmation']
                and math.hypot(g['position']['x_right'], g['position']['z_forward']) <= self.config['waypoint_distance']
            ]
            proof = self.memory.task.completion_evidence[-1].get('verification', {}) \
                if self.memory.task.completion_evidence else {}
            landing_arrival = (
                proof.get('completion_confirmed')
                and 'multi_frame_level_landing_transition' in proof.get('supported', [])
                and not any(a['action'] == 'MOVE_FORWARD' for a in self.memory.actions
                            if a['step'] >= self.memory.task.subgoal_started)
                and any(g.get('landing_geometry', {}).get('status') == 'SUPPORTED'
                        and current_visual_evidence(g['source'], self.memory, time.monotonic(),
                                                    self.config['max_scene_age_seconds'])
                        and wanted & task_tokens(g['target']) for g in self.memory.groundings)
            )
            if not nearby and not landing_arrival:
                return dict(status='INSUFFICIENT_EVIDENCE', reason='fresh_nearby_final_grounding_required')
        self.terminate('goal_completed' if outcome == 'success' else 'model_aborted', model_requested=True)
        return dict(status='FINISHED', outcome=outcome, evidence=evidence)
