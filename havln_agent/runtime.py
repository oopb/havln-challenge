"""Single-writer Agent Runtime: dispatch, factual state, budgets and legal termination.

It contains no recovery ladder, scan quota, stagnation gate or automatic replan.
"""

import copy
import math
import re
import threading
import time

from .observation import array_hash
from .api_client import APIError, BudgetExceeded
from .grounding import project_grounding
from .progress import current_visual_evidence
from .motion import registered_action, landing_support
from .tools import ToolExecutor
from .forward_preflight import sense_forward
from .evidence_ledger import physical_view_key
from .waypoint import rank_candidates
from .spatial import validity
from .contracts import (registry, camera_look_options,
                        CAMERA_PITCH_LIMIT_DEG)
from .active_observation import ObservationController
from .local_exploration import (covered_view, last_block, local_coverage,
                                predicted_scan, bearing_from_pose, wrap_heading, same_stance)
from .navigation_agent import NavigationAgent
from .heading import signed_turn
from .visual_perception import VisualPerception


class AgentRuntime(ToolExecutor):
    allowed_actions = ('STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner = threading.get_ident()
        self.started = time.monotonic()
        self.phase, self.finish_reason = 'INITIALIZE', None
        self.active_waypoint_id = None
        self.navigation_mode = 'OBSERVE'
        self.api_failure = None
        self.navigation_liveness = None
        self.event_callback, self.event_seq = None, 0
        # Runtime events are emitted to events.jsonl, not to MemoryRecords'
        # separate semantic-memory events list. Count physical event facts
        # here so summary.json matches the exported runtime event trace.
        self.runtime_event_counts={}
        self.semantic_motion_policy = 'LOCAL_TERRAIN'
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
        # The runtime owns canonical trace fields. Callers may annotate an
        # observation identity, but must never supply a second keyword with
        # the same name to dict(), nor overwrite the current-frame identity.
        record = dict(event=event_type, episode_id=self.memory.episode_id,
            step_index=len(self.memory.actions), event_seq=self.event_seq,
            observation_id=(self.memory.current or {}).get('observation_id'),
            timestamp=time.time(), run_status=self.phase, source=source)
        for key, value in data.items():
            if key in record:
                if value != record[key]:
                    record['reported_' + key] = value
            else:
                record[key] = value
        self.event_seq += 1
        if event_type=='forward_motion_stalled':
            k='forward_motion_stalled'
            self.runtime_event_counts[k]=self.runtime_event_counts.get(k,0)+1
        if event_type=='forward_side_depth_ambiguity':
            k='forward_side_depth_ambiguity'
            self.runtime_event_counts[k]=self.runtime_event_counts.get(k,0)+1
        elif (event_type=='pre_forward_obstacle_reported'
              and data.get('atomic_move_not_executed') is True):
            k='atomic_forward_confirmed_depth_block'
            self.runtime_event_counts[k]=self.runtime_event_counts.get(k,0)+1
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
        # Passive refresh never reprojects geometry. The model may explicitly
        # re-ground its target when depth or viewpoint changes.

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
        task=self.memory.task
        return dict(architecture='mission_narrative_v2',
            navigation_mode=self.navigation_mode,
            episode_id=self.memory.episode_id,
            observation_id=(self.memory.current or {}).get('observation_id'),
            pose=self.motion_controller.pose(),action_budget=self.config['max_steps'],
            actions_remaining=self.config['max_steps']-len(self.memory.actions),
            run_status=self.phase,finish_reason=self.finish_reason,
            mission=dict(instruction=task.instruction,
                progress_narrative=task.progress_narrative,
                navigation_intent=task.navigation_intent,
                uncertainties=list(task.uncertainties)))

    def context(self):
        return dict(self.state_view(),memory=self.memory.snapshot())

    def start_mission(self):
        self.phase='NAVIGATING'
        self.set_navigation_mode('OBSERVE','mission_started')
        self.event('mission_started',source='runtime',
                   instruction=self.memory.task.instruction)

    def execute(self, call, decision=None):
        self.assert_owner()
        self.context_version = 'mission_v2:%s:%d' % (
            self.memory.current['observation_id'], len(self.memory.actions))
        self.event('tool_call', source='registry', tool=call.get('tool'), args=call.get('args'))
        value = super().execute(call, decision)
        self.event('tool_result', source='registry', result=value)
        return value

    def dispatch(self, call, decision):
        if self.done:
            return dict(status='STOPPED', reason=self.stop_reason)
        if not decision:
            return dict(status='STALE_TARGET',reason='missing_navigation_decision')
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
        moving = call['tool'] in ('navigate_local','navigate_relative','navigate_to_pixel')
        if moving:
            self.set_navigation_mode('MOVE', 'model_selected_motion',
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
        # Do not report nominal command lengths as measured translation.
        # Route completion is execution of a finite action plan, not arrival.
        executed_forwards=[a['result'] for a in self.memory.actions[before:]
                           if a['action']=='MOVE_FORWARD']
        value.update(
            translation_m=None,
            translation_confidence='not_estimated',
            vertical_translation_m_est=None,
            vertical_motion='UNKNOWN',
            semantic_direction_mismatch=False,
            metric_odometry_enabled=False,
            forward_actions_executed=len(executed_forwards),
            forward_actions_with_RGBD_change=sum(
                bool(a.get('visual_change')) for a in executed_forwards),
            forward_actions_with_unchanged_RGBD=sum(
                not bool(a.get('visual_change')) for a in executed_forwards),
            physical_displacement_verified=False)
        if value['status'] == 'BLOCKED' and value.get('reason') in (
                'terrain_support_unverified', 'human_localization_unverified',
                'unknown_depth'):
            value.update(self.motion_controller.needs_view(value['reason']))
        if 'information_options' in value:
            value['information_options'] = [
                name for name in value['information_options']
                if name in self.registry]
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
                action_record['evidence_state'] = (
                    'executed_forward_with_RGBD_change_not_displacement_proof'
                    if valid else 'executed_forward_no_confirmed_displacement')
                self.memory.note_motion()
                alert=(value['feedback'].get('motion_alert') or {})
                if alert.get('reason')=='no_observed_frame_change_after_forward':
                    self.event('forward_motion_stalled',source='environment_receipt',
                        action_step=alert['action_step'],
                        before_observation_id=alert['observation_before'],
                        after_observation_id=alert['observation_after'],
                        repeated_same_RGBD_stance_heading=alert.get(
                            'repeat_count_same_RGBD_heading'),
                        collision_proven=False)
            self.event('atomic_action', source='environment_receipt', action=action, feedback=value['feedback'])
        return value

    def forward_preflight(self, stage='navigation_decision'):
        """Instantaneous, READ-ONLY current front Depth evidence for the model.

        Called before NavigationAgent inference AND from the atomic MOVE
        preflight. Same geometry and status semantics, but only an actual
        attempted atomic forward may add a BLOCKED receipt to EvidenceLedger.
        """
        self.assert_owner()
        if self.memory.current is None or self.observation is None:
            return dict(status='UNKNOWN',reason='no_live_RGBD',
                        source_observation_id=None,
                        camera_pitch_deg=self.pitch_deg)
        source=self.source()
        result=sense_forward(self.geometry(),source)
        result['stage']=stage
        result['camera_pitch_deg']=self.pitch_deg
        # Keep measured pre-decision obstacles even when the model chooses
        # TURN/OBSERVE and no forward was ever requested. This is not a
        # physical collision receipt or a globally forbidden bearing.
        if stage in ('navigation_decision','observation_view'):
            note=self.memory.evidence.note_predecision_obstacle(result)
            if note:
                result['repeat_count_same_predecision_view']=note['seen_count']
        view_key=physical_view_key(dict(
            observation_id=source['observation_id'],
            rgb_hash=source['rgb_hash'],
            depth_hash=source['depth_hash'],
            heading_deg=result['heading_deg'],
            pitch_deg=result['pitch_deg'],
            motion_epoch=result['motion_epoch'],
            viewpoint_generation=result['viewpoint_generation'],
            local_position_xz_m_est=result['local_position_xz_m_est']))
        result['repeat_count_prior_blocks']=sum(
            physical_view_key(v)==view_key
            for v in self.memory.evidence.obstacle_blocks)
        result['repeat_count_prior_stalls']=sum(
            physical_view_key(v)==view_key
            for v in self.memory.evidence.motion_stalls)
        self.event('forward_depth_preflight',source='live_depth',
            stage=stage,state=result['status'],
            preflight_source_observation_id=source['observation_id'],
            heading_deg=result['heading_deg'],pitch_deg=self.pitch_deg,
            center_confirmed=result['center_confirmed'],
            repeat_count_prior_blocks=result['repeat_count_prior_blocks'],
            obstacle_roi=(result['blocker'] or {}).get('focus_region_xyxy_norm'))
        return result

    def _center_camera_for_forward(self):
        """Once before the next forward step, return camera to level.

        No action when already level. LOOK actions consume real episode budget
        and are auditable, unlike silently overwriting pitch_deg. The pose
        heading and the model-authored route are not changed.
        """
        tilt=self.camera.tilt_angle
        required=int(round(abs(self.pitch_deg)/tilt))
        if abs(self.pitch_deg)<1e-4:
            return dict(status='OK',actions=[],pitch_deg=0.,
                        reason='already_level')
        if required>math.ceil(60/tilt) or abs(required*tilt-abs(self.pitch_deg))>1e-3:
            return dict(status='CAMERA_ALIGNMENT_UNCERTAIN',
                        reason='pitch_not_on_camera_tilt_grid',
                        pitch_deg=self.pitch_deg,actions=[])
        before=self.pitch_deg
        actions=[]
        for _ in range(required):
            direction='LOOK_UP' if self.pitch_deg<0 else 'LOOK_DOWN'
            feedback=self._step(direction)
            if feedback.get('status')!='EXECUTED':
                return dict(status=feedback.get('status','PARTIAL'),
                    reason=feedback.get('reason','camera_recenter_interrupted'),
                    pitch_before_deg=before,pitch_after_deg=self.pitch_deg,
                    actions=actions)
            actions.append(direction)
        if abs(self.pitch_deg)>1e-3:
            return dict(status='CAMERA_ALIGNMENT_UNCERTAIN',
                reason='camera_pitch_did_not_recenter',
                pitch_before_deg=before,pitch_after_deg=self.pitch_deg,
                actions=actions)
        self.event('camera_recentered_for_motion',source='action_receipts',
            actions=actions,pitch_before_deg=before,
            pitch_after_deg=self.pitch_deg)
        return dict(status='OK',actions=actions,
            pitch_before_deg=before,pitch_after_deg=self.pitch_deg,
            reason='physical_camera_look_receipts')

    def _prepare_forward(self):
        """Same calibrated Depth test as the pre-decision model input.

        This is called ONLY on the freshest public RGB-D after any turn and
        camera alignment, immediately before an actual atomic forward. Only
        measured central occupancy stops the pending atomic command.
        """
        result=self.forward_preflight(stage='atomic_forward')
        self._last_forward_depth_probe=dict(result,
            check_mode='live_same_schema_decision_and_atomic_forward')
        if result['status']!='BLOCKED':
            return None
        source=self.source()
        block=result['blocker']
        blocked_at=self.memory.evidence.blocked_forward(
            source,self.motion_controller.pose(),
            reason='measured_centerline_obstacle',
            center=block,place_id=self.memory.current.get('place_id'))
        repeat=blocked_at['repeat_count_at_recording']
        # Structure and ROI are identical in model predecision, tool feedback,
        # and historical Memory. A depth.center nesting mismatch was the v8 bug.
        combined=dict(
            source_observation_id=source['observation_id'],
            depth=dict(block,wide_probe_state=result['raw_probe_state'],
                       clearance_m=result['clearance_m'],
                       body_step_m=self.camera.forward_step),
            visual=dict(status='DEFERRED_TO_OBSERVATION_AGENT',
                description='',uncertainty='look_at_real_RGBD_and_depth_ROI'),
            visual_reused_for_identical_frame=False,
            repeated_same_RGBD_stance_heading=repeat,
            requested_followup='inspect_blocker_and_choose_different_bearing'
                if repeat>=2 else None,
            identity_is_visual_hypothesis_only=True,
            next_action_selected_by='navigation_agent')
        self.event('pre_forward_obstacle_reported',
            source='live_depth_for_navigation_agent',
            depth_observation_id=source['observation_id'],
            central_support_pixels=block['central_support_pixels'],
            focus_region_xyxy_norm=block.get('focus_region_xyxy_norm'),
            atomic_move_not_executed=True,
            repeated_same_RGBD_stance_heading=repeat,
            navigation_choice_unchanged=True)
        return dict(status='BLOCKED',reason='measured_centerline_obstacle',
            blocked_at=blocked_at,
            repeated_same_RGBD_stance_heading=repeat,
            blockage_interpretation=(
                'same_RGBD_heading_prior_block_repeated'
                if repeat>=2 else 'fresh_centerline_depth_occupancy'),
            forward_depth_preflight=result,
            obstacle_observation=combined,
            safety=dict(path_state=result['path_state'],
                clearance=result['clearance_m'],
                required_forward_step=result['required_forward_step_m'],
                center_confirmed=True,
                center_support_pixels=block['central_support_pixels'],
                geometry_invocation='automatic_live_central_depth_only'),
            information_options=['observe_for_goal','navigate_relative'])

    def _observe_for_goal(self, objective, sweep_degrees=180):
        """Delegate a natural-language request to full-session RGB-D observation."""
        result=self.observation_controller.scan_mission(
            objective,sweep_degrees=sweep_degrees)
        # Observations and propositions are saved as evidence inside Memory;
        # a visual hypothesis cannot advance or roll back a numbered state.
        return result

    def _turn_and_observe(self, turn_direction, turn_degrees, look,
                          purpose='explore'):
        """Model-selected LEFT/RIGHT/NONE and nonnegative turn magnitude."""
        try:
            delta=signed_turn(turn_direction,turn_degrees)
        except ValueError as error:
            return dict(status='INVALID_ARGUMENTS',reason=str(error))
        if look not in ('none','up','down'):
            return dict(status='INVALID_ARGUMENTS',reason='invalid_camera_look')
        steps=round(abs(delta)/self.camera.turn_angle)
        actual_turn=steps*self.camera.turn_angle
        direction='TURN_LEFT' if delta>=0 else 'TURN_RIGHT'
        actions=[direction]*steps
        if look!='none':
            # Check the WHOLE command before any TURN: an impossible LOOK
            # must not partially rotate the robot or trigger api_error STOP.
            requested_delta=(self.camera.tilt_angle if look=='up'
                             else -self.camera.tilt_angle)
            pitch_after=self.pitch_deg+requested_delta
            if not (-CAMERA_PITCH_LIMIT_DEG-1e-6 <= pitch_after <=
                    CAMERA_PITCH_LIMIT_DEG+1e-6):
                allowed=camera_look_options(self.pitch_deg,
                                           self.camera.tilt_angle)
                return dict(
                    status='CAMERA_PITCH_LIMIT',recoverable=True,
                    reason='requested_look_exceeds_camera_limits',
                    requested_look=look,
                    requested_turn_direction=turn_direction,
                    requested_turn_degrees=turn_degrees,
                    current_pitch_deg=self.pitch_deg,
                    requested_pitch_deg=pitch_after,
                    pitch_limits_deg=[-CAMERA_PITCH_LIMIT_DEG,
                                      CAMERA_PITCH_LIMIT_DEG],
                    tilt_step_deg=self.camera.tilt_angle,
                    allowed_look=allowed,
                    executed_camera_actions=[],
                    next_decision_owner='navigation_agent',
                    suggestion='Choose a valid LOOK, a different horizontal '
                               'bearing, observation, or navigation; the '
                               'rejected command did not execute.')
            actions.append('LOOK_UP' if look=='up' else 'LOOK_DOWN')
        executed=[]
        for action in actions:
            result=self._step(action)
            if result.get('status')!='EXECUTED':
                return dict(status='PARTIAL' if executed else result.get('status'),
                    reason=result.get('reason'),executed_camera_actions=executed)
            executed.append(action)
        return dict(status='ACTIONS_COMPLETED',executed_camera_actions=executed,
            requested_turn_direction=turn_direction,
            actual_turn_degrees=actual_turn,actual_turn_direction=turn_direction,
            observation_id=self.memory.current['observation_id'])

    def _revalidate_path(self, path, resume=False):
        return self.motion_controller.validate(self, path, resume)

    def _navigate_relative(self, turn_direction, turn_degrees,
                           distance_m, _source=None):
        """Direct model intent, without a preselected semantic candidate.

        The bearing is relative to the current heading, not a global pose or
        semantic permission. A TURN is always legal as a physical rotation;
        each subsequent MOVE independently validates fresh RGB-D.
        """
        try:
            requested_turn=signed_turn(turn_direction,turn_degrees)
        except ValueError as error:
            return dict(status='INVALID_ARGUMENTS',reason=str(error))
        if (isinstance(distance_m,bool) or not isinstance(distance_m,(int,float))
                or not math.isfinite(distance_m)
                or not 0.25 <= distance_m <= 3.0):
            return dict(status='INVALID_ARGUMENTS', reason='relative_motion_out_of_bounds')
        stride = self.camera.turn_angle
        step = self.camera.forward_step
        # Models express motion intent in continuous units; the executor
        # quantizes to the supported Habitat actuator increments and reports
        # what was actually requested of the simulator.
        actual_turn = round(requested_turn / stride) * stride
        actual_distance = max(1, round(distance_m / step)) * step
        actual_distance = min(actual_distance, math.floor(3.0 / step) * step)
        if _source is None:
            self.refresh()
            source = self.source()
        else:
            # The pixel and its derived bearing must refer to the *same*
            # RGB-D observation. validate() and _step() independently refresh
            # the depth frame before authorizing any physical forward step.
            source = _source
        angle = math.radians(actual_turn)
        heading = source['pose']['heading'] + angle
        path_id = 'direct_%04d' % len(self.memory.tool_ledger)
        path = dict(id=path_id, type='EXPLORATION_WAYPOINT', angle=angle,
            source_observation_id=source['observation_id'],
            bearing_deg=actual_turn,range_m=actual_distance,
            distance=actual_distance, source=source, goal_association=[],
            first_step='EXECUTABLE', execution_status='PROPOSED',
            objective=dict(purpose='model_directed_motion', completes_language_goal=False),
            direction_source='navigation_agent',
            requires_post_turn_recheck=bool(actual_turn),
            geometric_status='NOT_YET_VALIDATED',
            endpoint_xz_m=None,
            endpoint_estimate_valid=False,
            route_execution_basis='commanded_yaw_plus_finite_forward_receipts')
        # No automatic collision check: geometry is used only when the model
        # explicitly requests its inspection tool.
        self.paths[path_id] = path
        # Direct routes were absent from spatial memory because they skip
        # get_waypoint_candidates. Register the proposal, not a clearance
        # certificate. Action completion does NOT imply metric target arrival.
        self.memory.note_candidates([path])
        result = self._navigate_local(path_id)
        return dict(result, waypoint_id=path_id,
                    requested_motion=dict(
                        turn_direction=turn_direction,turn_degrees=turn_degrees,
                        distance_m=distance_m,quantized_turn_deg=actual_turn,
                        actual_turn_direction=('LEFT' if actual_turn>0 else
                            'RIGHT' if actual_turn<0 else 'NONE'),
                        quantized_turn_degrees=abs(actual_turn),
                        quantized_distance_m=actual_distance,
                        authorization='model_selected_atomic_actions_not_metric_arrival'))

    def _navigate_local(self, waypoint_id):
        path = self.paths.get(waypoint_id)
        if path is None:
            return dict(status='INVALID_TARGET', reason='unknown_waypoint_id')
        # A model-selected route may resume after a prior forward action:
        # its identity and finite horizon remain independent of metric pose.
        # Reusing an entirely new waypoint still requires its source stance
        # to match the current executed-forward generation.
        if path['type'] not in ('GOAL_WAYPOINT','EXPLORATION_WAYPOINT'):
            return dict(status='INVALID_TARGET',reason='unknown_waypoint_type')
        target = 'current' if self.active_waypoint_id == waypoint_id and self.motion_controller.target is not None else waypoint_id
        original = self.motion_policy
        if path['type']=='EXPLORATION_WAYPOINT':
            self.motion_policy = 'LOCAL_TERRAIN'
        try:
            result = self._execute_waypoint(target)
            self.active_waypoint_id = (waypoint_id if self.motion_controller.target is not None else None)
            return dict(result,waypoint_type=path['type'],waypoint_arrival_completes_task=False)
        finally:
            self.motion_policy = original

    def _read_memory_record(self, record_id):
        return dict(status='OK', memory=self.memory.read_memory_record(record_id))

    def _query_task_memory(self, **args):
        return dict(status='OK', memory=self.memory.query_task_memory(**args))

    def _request_finish(self, outcome, evidence):
        return self._finish_navigation(outcome, evidence)

    def assess(self, value):
        if value['status'] in ('API_ERROR', 'API_UNAVAILABLE'):
            self.api_failure = value.get('api_failure', dict(category=value.get('category'), reason=value.get('reason')))
            self.terminate('api_error' if value['status'] == 'API_ERROR' else 'api_budget_exhausted')
        if value['status'] in ('TARGET_REACHED','ACTIONS_COMPLETED'):
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
