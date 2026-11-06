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
from .safety import forward_probe
from .waypoint import generate_waypoints, goal_relation
from .motion import estimate_translation
from .spatial import goal_estimate, angle_delta


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
        # Passive RGB-D capture does NOT run geometry or collision detection.
        # Only a model-requested inspect_geometry/get_waypoint_candidates
        # tool can produce measured directional diagnostics.
        unchanged = (previous is not None and previous_frame is not None
            and previous['pose'] == pose
            and array_hash(previous_frame.rgb) == array_hash(observation.rgb)
            and array_hash(previous_frame.depth) == array_hash(observation.depth))
        diagnostics = previous.get('directions', []) if unchanged else []
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
        if actions and result['status'] not in ('OK','PARTIAL','TARGET_REACHED','ACTIONS_COMPLETED','FINISHED','EXECUTED','NO_NEW_INFORMATION'):
            result.update(interrupted_status=result['status'],status='PARTIAL')
        args=call.get('args',{})
        # Navigation has NO per-tool atomic budget: count actual authorized
        # simulator actions instead of pretending the model requested a cap.
        requested=len(actions) if call.get('tool') in (
            'navigate_relative','navigate_to_pixel','navigate_local') else 0
        if call.get('tool')=='turn_and_observe':
            requested=round(abs(args.get('relative_turn_deg',0))/self.camera.turn_angle)+int(args.get('look','none')!='none')
        elif call.get('tool')=='observe_for_goal':
            # Declared scope only; VLM controls camera headings and real
            # executed camera actions are counted separately.
            requested=round(args.get('sweep_degrees',360)/self.camera.turn_angle)+int(args.get('look','none')!='none')
        elif call.get('tool')=='request_finish':
            requested=1
        if type(requested) is not int or requested<0:requested=0
        result.update(tool_call_id=self.tool_id,tool=call.get('tool'),requested_actions=requested,
            executed_actions=len(actions),executed_motion=actions,start_observation_id=observation_before,
            end_observation_id=self.memory.current['observation_id'],seconds=time.monotonic()-started,
            request_context=context)
        if call.get('tool') in ('navigate_relative','navigate_to_pixel','navigate_local'):
            turns=sum(a in ('TURN_LEFT','TURN_RIGHT') for a in actions)
            forwards=actions.count('MOVE_FORWARD')
            pending_id=result.get('waypoint_id') or args.get('waypoint_id')
            active=(pending_id is not None and
                pending_id == getattr(self,'active_waypoint_id',None)
                and self.motion_controller.target is not None)
            attempt=self.memory.active_attempt or {}
            uncertain=bool(attempt.get('unknown_motion_seen'))
            remaining_turns=None
            forward_budget=None
            if active and attempt.get('target'):
                next_action=self.motion_controller.route_action(
                    attempt['target'],attempt.get('forward_actions',0),
                    attempt.get('forward_action_horizon',1))
                if next_action in ('TURN_LEFT','TURN_RIGHT'):
                    heading_goal=attempt['target']['source']['pose']['heading']+attempt['target']['angle']
                    heading_error=abs(angle_delta(
                        heading_goal,self.motion_controller.heading))
                    remaining_turns=max(0,math.ceil(
                        heading_error/math.radians(self.camera.turn_angle)-.5-1e-8))
                else:
                    remaining_turns=0
                forward_budget=remaining_turns+int(
                    attempt.get('forward_actions',0)<attempt.get('forward_action_horizon',0))
            result['movement_intent_feedback']=dict(
                executed_turn_actions=turns,
                executed_forward_actions=forwards,
                turn_only_no_forward_attempt=bool(turns and not forwards),
                actual_forward_attempted=bool(forwards),
                requested_forward_distance_m=(
                    (result.get('requested_motion') or {}).get('distance_m')
                    or args.get('distance_m')),
                waypoint_id=pending_id,
                same_waypoint_pending=bool(active),
                remaining_turn_actions_est=remaining_turns,
                min_atomic_actions_until_first_forward_attempt=forward_budget,
                first_forward_still_requires_live_safety=False,
                geometry_diagnostics_model_invoked_only=True,
                endpoint_estimate_valid=result.get('endpoint_estimate_valid',False),
                action_intent_completed=result.get('action_intent_completed',False),
                remaining_commanded_forward_actions=result.get(
                    'remaining_commanded_forward_actions'),
                movement_feedback_is_evidence_not_completion=True,
                interpretation=(
                    'only_turns_executed_forward_not_attempted'
                    if turns and not forwards else
                    'forward_was_attempted_registration_may_still_be_unknown'
                    if forwards else 'no_atomic_motion_executed'),
                model_decides_to_resume_replan_or_observe=True)
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
            self._last_forward_depth_probe = None
            try:
                self.refresh()
            except (APIError, AttributeError, RuntimeError, ValueError):
                return dict(status='STALE_OBSERVATION', reason='fresh_public_depth_unavailable')
            if time.monotonic() - self.observation.captured_at > self.config['max_depth_age_seconds']:
                return dict(status='STALE_OBSERVATION', reason='depth_age_limit')
            pending = self._prepare_forward()
            if pending:
                return pending
        # Runtime._prepare_forward() has already applied the conservative
        # live central-obstacle Depth probe for MOVE_FORWARD. Full terrain
        # geometry / other-bearing inspection remain NavigationAgent tools.
        # Do not run a second probe or override the model's route here.
        diagnostics, candidates = [], []
        geometry, safety_check, front = None, None, None
        live_preflight=(getattr(self,'_last_forward_depth_probe',None)
                        if action=='MOVE_FORWARD' else None)
        before, before_record = self.observation, dict(self.memory.current)
        pose = self.motion_controller.pose()
        checked_at = time.monotonic()
        after, done = self.step_callback(action)
        # RGB-D registration is odometry feedback, not collision permission.
        motion = estimate_translation(before, after, self.camera) \
            if action == 'MOVE_FORWARD' else None
        changed_depth = array_hash(after.depth) != array_hash(before.depth)
        changed_rgb = array_hash(after.rgb) != array_hash(before.rgb)
        if motion is not None:
            # Viewpoint separation is observable even when metric registration
            # fails. Never treat the commanded 0.25m as measured translation.
            motion = dict(motion, observed_frame_change=changed_depth or changed_rgb)
        self.motion_controller.accept_feedback(action,motion)
        feedback = dict(
            forward_depth_preflight=live_preflight,
            visual_change=changed_depth or changed_rgb,
            depth_changed=changed_depth,
            rgb_changed=changed_rgb,
            translation_supported=bool(motion and motion['status'] == 'ESTIMATED'),
            displacement_verified=False,
            vertical_motion=motion.get('vertical_motion', 'UNKNOWN') if motion else 'UNKNOWN',
            observation_before=before_record['observation_id'],
            observation_after=None,
            rgbd_motion=motion,
            # Geometry/terrain landing evidence is available only through an
            # explicit inspect_geometry call, not as an implicit MOVE side effect.
            landing_support=(dict(status='NOT_EVALUATED',
                reason='geometry_requires_model_requested_inspect_geometry',
                terrain_certified=False) if action == 'MOVE_FORWARD' else None),
            goal_association=(self.memory.active_attempt['target'].get('goal_association', [])
                              if self.memory.active_attempt else []),
        )
        self.memory.record_action(action, self.motion_controller.pose(), feedback)
        self.accept_observation(after)
        feedback['observation_after'] = self.memory.current['observation_id']
        if action == 'MOVE_FORWARD':
            policy = getattr(self, 'semantic_motion_policy', self.motion_policy)
            direction = motion.get('vertical_motion', 'UNKNOWN')
            expected = {'LEVEL_ONLY': 'LEVEL', 'DESCEND_STAIRS': 'DOWN', 'ASCEND_STAIRS': 'UP','LOCAL_TERRAIN':direction}[policy]
            feedback['semantic_direction_mismatch']=(motion['status']=='ESTIMATED' and
                direction in ('DOWN','UP') and direction != expected)
            # A measured direction which may oppose the current goal is an
            # observation for the model, not an automatic local-route veto.
            if feedback['semantic_direction_mismatch']:
                self.memory.event('semantic_motion_direction_observed',
                    observed_direction=direction, expected_direction=expected,
                    observation_before=feedback['observation_before'],
                    observation_after=feedback['observation_after'],
                    source='registered_rgbd_estimate_only')
            # Some Habitat MOVE_FORWARD receipts produce byte-identical RGB-D
            # before/after (e.g. 13 actions in the provided eval-batch4 trace).
            # Treat this as an explicit *no observed motion* stall rather than
            # an ordinary odometry gap. Never infer a collision or ground-truth
            # translation from this alone; but halt this local route so the
            # model can pick a different short motion/observation strategy.
            if motion['status'] != 'ESTIMATED':
                self.memory.event('odometry_diagnostic_unknown',
                    action_step=len(self.memory.actions)-1,
                    reason=motion.get('reason'),
                    observation_before=feedback['observation_before'],
                    observation_after=feedback['observation_after'],
                    does_not_reset_route=True)
            # Only a demonstrably unchanged frame can interrupt a route.
            # Low-consensus feature matching by itself is NOT a movement veto.
            alert_reason=('no_observed_frame_change_after_forward'
                          if not changed_rgb and not changed_depth else None)
            if alert_reason:
                alert = dict(action_step=len(self.memory.actions)-1, reason=alert_reason, motion_policy=policy,
                    observation_before=feedback['observation_before'], observation_after=feedback['observation_after'],
                    measurement=motion, displacement_verified=False)
                feedback['motion_alert'] = alert
                if alert_reason == 'no_observed_frame_change_after_forward':
                    alert['movement_evidence'] = 'identical_rgb_and_depth_hashes_after_executed_forward'
                    alert['collision_proven'] = False
                    self.memory.event('no_observed_forward_motion',
                        action_step=alert['action_step'],
                        observation_before=feedback['observation_before'],
                        observation_after=feedback['observation_after'],
                        motion_epoch=self.motion_controller.pose().get('motion_epoch'))
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
                status=('CENTRAL_DEPTH_PREFLIGHT_EXECUTED' if live_preflight
                        else 'NOT_CHECKED_MODEL_DIRECTED'),
                checked_action=action,
                front=front,
                captured_at=before.captured_at,
                checked_at=checked_at,
                scope=('central_depth_preflight' if live_preflight
                       else 'geometry_not_automatically_checked' if action == 'MOVE_FORWARD'
                       else 'no_forward_displacement'),
                forward_depth_preflight=live_preflight,
                human_semantics=geometry.human_status if geometry is not None else None,
                terrain_evidence=safety_check['support'] if safety_check is not None else None,
                dynamic_motion_prediction='UNAVAILABLE',
                geometry_invocation=('automatic_centerline_depth_only'
                    if live_preflight else 'explicit_model_tool_only'),
                physical_obstacle_proven=False,
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
                    stage='geometry_policy',
                    action=action,
                    reason=('central_depth_preflight' if live_preflight
                            else 'not_invoked_implicitly' if action == 'MOVE_FORWARD'
                            else 'turn_or_stop'),
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
        candidates = generate_waypoints(geometry, self.config['waypoint_distance'], diagnostics, angles, probe=True)
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
            # Include detours and uncertain bearings. These are factual
            # associations, not permission decisions or prefiltered routes.
            associations = relations
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
                goal_status='GOAL_EVIDENCE_AVAILABLE' if associations else 'GOAL_DIRECTION_UNKNOWN',
                frame='source_robot_camera', angle_unit='radians',
                endpoint_xz_m=[source['pose']['x'] - math.sin(heading) * candidate['distance'],
                               source['pose']['z'] + math.cos(heading) * candidate['distance']],
                association_rejections=[r for r in relations if not r['relevant']] + excluded,
                execution_requirements=['reconfirm_uncertain_goal','model_requested_geometry_optional'], 
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
            safety_state=dict(visible_floor=dict(status='DEFERRED',
                reason='inspect_geometry_for_full_terrain_diagnostics'), humans=geometry.human_status,
                authorization='no_implicit_checks_model_decides_when_to_inspect', dynamic_motion_prediction='UNAVAILABLE'),
            goal_status='GOAL_EVIDENCE_AVAILABLE' if any(p['goal_association'] for p in paths) else 'GOAL_DIRECTION_UNKNOWN',
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
        # Action completion is NOT spatial arrival. Even 100% executed command
        # steps cannot certify robot displacement in Habitat's collision world.
        attempt=self.memory.active_attempt or {}
        step_size=getattr(getattr(self,'camera',None),'forward_step',0.25)
        horizon=attempt.get('forward_action_horizon')
        if horizon is None:
            horizon=max(1,math.ceil((path.get('distance',step_size)-1e-6)/step_size))
        forwards=attempt.get('forward_actions',0)
        remaining_actions=max(0,horizon-forwards)
        action_complete=status=='ACTIONS_COMPLETED'
        path['execution_status']=('IN_PROGRESS' if status=='PARTIAL' else status)
        if status in ('ACTIONS_COMPLETED','TARGET_REACHED','INTERRUPTED',
                      'BLOCKED','REQUIRES_ATTENTION','STALE_TARGET'):
            self.memory.finish_attempt(status,reason)
            self.motion_controller.target=None
        elif self.memory.active_attempt:
            attempt.update(result='IN_PROGRESS' if status=='PARTIAL' else status,
                           reason=reason)
            if status!='PARTIAL':
                attempt.setdefault('interruptions',[]).append(dict(
                    status=status,reason=reason,
                    observation_id=self.memory.current['observation_id']))
        return dict(status=status,reason=reason,
            target_reached=False,action_intent_completed=action_complete,
            displacement_verified=False,endpoint_estimate_valid=False,
            reach_basis='atomic_action_receipts_not_verified_metric_arrival',
            original_target=list(checked['original_target']),
            pending_target=list(self.motion_controller.target)
                if self.motion_controller.target is not None else None,
            remaining_distance=None,
            remaining_commanded_forward_actions=remaining_actions,
            nominal_remaining_command_distance_m=round(
                remaining_actions*step_size,3),
            commanded_forward_actions=forwards,
            commanded_forward_horizon=horizon,
            odometry_unknown_during_route=bool(attempt.get('unknown_motion_seen')),
            unknown_observation_ids=list(attempt.get('unknown_observation_ids',[])),
            remaining_path=dict(path_id=path['id'],
                original_endpoint_xz_m=list(checked['original_target']),
                endpoint_estimate_valid=False,
                status=path['execution_status'],
                completed=False,action_intent_completed=action_complete),
            executable_segment=self.executable_segment or {},
            validation=checked.get('validation',{}),**extra)

    def _execute_waypoint(self, target):
        if target != 'current':
            path=self.paths.get(target)
            if path is None:
                return dict(status='INVALID_TARGET',reason='unknown_tool_generated_path_id')
            checked=self._revalidate_path(path)
            if checked['status']!='OK':
                return checked
            self._cancel_local_navigation('model_selected_new_target')
            self.memory.start_attempt(path,self.motion_controller.pose())
            # Retain the historical endpoint for display ONLY. Commanded
            # bearing and finite forward action count drive execution.
            self.motion_controller.target=tuple(checked['original_target'])
            attempt=self.memory.active_attempt
            attempt['original_endpoint_xz_m']=list(self.motion_controller.target)
            attempt['forward_action_horizon']=max(1,math.ceil(
                (path['distance']-1e-6)/self.camera.forward_step))
            attempt['forward_actions']=0
            attempt['unknown_motion_seen']=False
            attempt['unknown_observation_ids']=[]
        elif self.motion_controller.target is None or not self.memory.active_attempt:
            return dict(status='INVALID_TARGET',reason='no_pending_target')
        else:
            path=self.paths[self.memory.active_attempt['target']['id']]
            checked=dict(original_target=list(self.motion_controller.target))
        # The model-chosen finite action sequence is independent of optional
        # translation fitting. Only the global Episode budget limits actions.
        for _ in range(max(0,self.config['max_steps']-1-len(self.memory.actions))):
            attempt=self.memory.active_attempt
            action=self.motion_controller.route_action(
                path,attempt['forward_actions'],attempt['forward_action_horizon'])
            if action is None:
                return self._navigation_feedback(
                    path,checked,'ACTIONS_COMPLETED',
                    'model_requested_forward_actions_executed')
            validation=self._revalidate_path(path,resume=True)
            if validation['status']!='OK':
                checked.update(validation)
                self.executable_segment=None
                return self._navigation_feedback(
                    path,checked,validation['status'],validation.get('reason'))
            checked=validation
            self.executable_segment=checked['executable_segment']
            result=self._step(action)
            if result['status']!='EXECUTED':
                self.executable_segment=None
                return self._navigation_feedback(
                    path,checked,result['status'],result.get('reason'),
                    **{k:result[k] for k in (
                        'safety','obstacle_observation','information_options')
                        if k in result})
            if action=='MOVE_FORWARD':
                attempt=self.memory.active_attempt
                attempt['forward_actions']+=1
                if not result['feedback']['translation_supported']:
                    # Diagnostic fact only: neither steer to the stale metric
                    # endpoint nor reset the active model-selected route.
                    attempt['unknown_motion_seen']=True
                    attempt['unknown_observation_ids'].append(
                        self.memory.current['observation_id'])
                alert=result['feedback'].get('motion_alert') or {}
                if alert.get('reason')=='no_observed_frame_change_after_forward':
                    return self._navigation_feedback(
                        path,checked,'REQUIRES_ATTENTION',
                        alert['reason'],motion_alert=alert)
            if self.done:
                return self._navigation_feedback(
                    path,checked,'INTERRUPTED','environment_done')
        # Finite route may finish on the final non-STOP action.
        attempt=self.memory.active_attempt
        if (attempt and self.motion_controller.route_action(
                path,attempt['forward_actions'],
                attempt['forward_action_horizon']) is None):
            return self._navigation_feedback(path,checked,
                'ACTIONS_COMPLETED','model_requested_forward_actions_executed')
        return self._navigation_feedback(path,checked,'PARTIAL',
            'episode_action_budget_exhausted',
            remaining_episode_actions=max(
                0,self.config['max_steps']-1-len(self.memory.actions)))

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
        """Only real Observation-Agent-confirmed milestones authorize success.

        No second Python landmark/stair/floor verifier contradicts the VLM.
        The simulator still owns atomic action legality and STOP.
        """
        if outcome=='success' and (not self.memory.task.subgoals or
                set(self.memory.task.completed_subgoals)!=set(
                    range(len(self.memory.task.subgoals)))):
            return dict(status='INSUFFICIENT_EVIDENCE',
                reason='not_all_subgoals_confirmed_by_observation_agent')
        self.terminate('goal_completed' if outcome=='success'
                       else 'model_aborted',model_requested=True)
        return dict(status='FINISHED',outcome=outcome,evidence=evidence)
