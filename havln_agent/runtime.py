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
from .safety import forward_probe
from .waypoint import rank_candidates
from .spatial import validity
from .contracts import registry
from .active_observation import ObservationController
from .local_exploration import (covered_view, last_block, local_coverage,
                                predicted_scan, bearing_from_pose, wrap_heading, same_stance)
from .navigation_agent import NavigationAgent
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
        record = dict(event=event_type, episode_id=self.memory.episode_id,
            step_index=len(self.memory.actions), event_seq=self.event_seq,
            observation_id=(self.memory.current or {}).get('observation_id'),
            timestamp=time.time(), run_status=self.phase, source=source, **data)
        self.event_seq += 1
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
        """Fresh Depth -> (confirmed central obstacle) -> one RGB VLM -> NAV.

        Called once immediately before each atomic MOVE_FORWARD. Uncertain
        depth, side-only geometry and ordinary stair edges do not authorize
        an automatic veto. The VLM describes only the same current RGB frame;
        it never changes the motion command or selects a route.
        """
        geometry=self.geometry()
        probe=forward_probe(geometry)
        self._last_forward_depth_probe=dict(
            status=probe['state'],reason=probe.get('reason'),
            observation_id=self.memory.current['observation_id'],
            check_mode='automatic_central_near_field_depth',
            center_confirmed=False)
        if probe['state'] != 'BLOCKED':
            return None
        center=geometry.central_forward_obstruction()
        if not center['confirmed']:
            # The generic wide body sweep also sees nearby side walls/risers.
            # They are useful diagnostic data, not enough to suppress actual
            # model-authored motion; physical collision still belongs to Habitat.
            self.event('forward_side_depth_ambiguity',
                source='live_depth',geometry_state=probe['state'],
                reason=center['reason'],
                center_support_pixels=center['central_support_pixels'],
                physical_collision_proven=False)
            return None

        source=self.source()
        blocked_pose=self.motion_controller.pose()
        blocked_at=self.memory.evidence.blocked_forward(
            source,blocked_pose,reason='measured_centerline_obstacle',
            center=center,
            place_id=self.memory.current.get('place_id'))
        self._last_forward_depth_probe.update(
            status='BLOCKED',center_confirmed=True,
            blocked_at=blocked_at,
            reason='measured_centerline_obstacle')
        # A measured central obstacle stops the pending atomic step.
        # The NEXT Observation Agent OBSTACLE_REVIEW gets real RGB-D and
        # measured facts; never run a second, isolated obstacle VLM here.
        # The prior successful route remains in receipt-derived Memory.
        source=self.source()
        visual=dict(status='DEFERRED_TO_OBSERVATION_AGENT',
            description='',
            uncertainty='use_OBSTACLE_REVIEW_with_live_RGBD_and_geometry')
        reused=False
        depth=probe['depth']
        safety=dict(
            path_state=depth['path_state'],
            clearance=depth['clearance'],
            required_forward_step=depth['required_forward_step'],
            nearest_obstacle_distance=depth.get('nearest_obstacle_distance'),
            nearest_collision_travel=depth.get('nearest_collision_travel'),
            nearest_collision_point=depth.get('nearest_collision_point'),
            rejection_reasons=depth.get('rejection_reasons',[]),
            collision_points=depth.get('collision_points',0),
            unknown_sweep_pixels=depth.get('unknown_sweep_pixels',0),
            human_occupancy=depth.get('human_occupancy',{}),
            geometry_invocation='automatic_one_step_central_obstacle_only',
            model_selected_route_unchanged=True)
        combined=dict(
            source_observation_id=source['observation_id'],
            depth=dict(center,wide_probe_state=probe['state'],
                clearance_m=depth['clearance'],
                body_step_m=self.camera.forward_step),
            visual=visual,
            visual_reused_for_identical_frame=reused,
            identity_is_visual_hypothesis_only=True,
            next_action_selected_by='navigation_agent')
        self.event('pre_forward_obstacle_reported',
            source='live_depth_for_observation_agent',
            depth_observation_id=source['observation_id'],
            central_support_pixels=center['central_support_pixels'],
            visual_status=visual['status'],visual_reused=reused,
            atomic_move_not_executed=True,
            navigation_choice_unchanged=True)
        return dict(status='BLOCKED',reason='measured_centerline_obstacle',
            blocked_at=blocked_at,
            safety=safety,obstacle_observation=combined,
            information_options=['observe_for_goal','navigate_relative'])

    def _observe_for_goal(self, objective, sweep_degrees=180):
        """Delegate a natural-language request to full-session RGB-D observation."""
        result=self.observation_controller.scan_mission(
            objective,sweep_degrees=sweep_degrees)
        # Observations and propositions are saved as evidence inside Memory;
        # a visual hypothesis cannot advance or roll back a numbered state.
        return result

    def _turn_and_observe(self, relative_turn_deg, look, purpose='explore'):
        """Execute model-selected, mechanically calibrated camera movement."""
        if (not math.isfinite(relative_turn_deg) or abs(relative_turn_deg)>180
                or look not in ('none','up','down')):
            return dict(status='INVALID_ARGUMENTS',reason='invalid_camera_command')
        steps=round(abs(relative_turn_deg)/self.camera.turn_angle)
        actual_turn=steps*self.camera.turn_angle
        direction='TURN_LEFT' if relative_turn_deg>=0 else 'TURN_RIGHT'
        actions=[direction]*steps
        if look!='none':
            tilt=self.camera.tilt_angle if look=='up' else -self.camera.tilt_angle
            if abs(self.pitch_deg+tilt)>60:
                return dict(status='INVALID_ARGUMENTS',reason='camera_pitch_limit')
            actions.append('LOOK_UP' if look=='up' else 'LOOK_DOWN')
        executed=[]
        for action in actions:
            result=self._step(action)
            if result.get('status')!='EXECUTED':
                return dict(status='PARTIAL' if executed else result.get('status'),
                    reason=result.get('reason'),executed_camera_actions=executed)
            executed.append(action)
        return dict(status='ACTIONS_COMPLETED',
            executed_camera_actions=executed,
            actual_turn_deg=actual_turn if relative_turn_deg>=0 else -actual_turn,
            observation_id=self.memory.current['observation_id'])

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
        # Model-requested geometry: keep measurements only for this frame.
        self.memory.current['directions']=[dict(angle_degrees=d['bearing_deg'],
            motion_state=d['status'],reason=d['reason'],clearance=d['clearance_m'])
            for d in directions]
        # Landing analysis is now opt-in, not an automatic MOVE side effect.
        latest=self.memory.actions[-1] if self.memory.actions else None
        landing=None
        if (latest and latest.get('action')=='MOVE_FORWARD'
                and (latest.get('result') or {}).get('observation_after')==self.memory.current['observation_id']
                and ((latest.get('result') or {}).get('rgbd_motion') or {}).get('status')=='ESTIMATED'
                and ((latest.get('result') or {}).get('rgbd_motion') or {}).get('vertical_motion')=='LEVEL'
                and getattr(self,'semantic_motion_policy',self.motion_policy)
                    in ('ASCEND_STAIRS','DESCEND_STAIRS')):
            landing=landing_support(geometry)
            latest['result']['landing_support']=dict(landing,
                evidence_source='model_requested_inspect_geometry',
                inspection_observation_id=self.memory.current['observation_id'])
        front=dict(check['depth']);front['depth_state']=front.pop('path_state')
        value=dict(status=check['state'],reason=check['reason'],
            geometry=dict(geometry.metadata(),front=dict(front,motion_state=check['state'],
                terrain_support=check['support'],human_evidence=geometry.human_status)),directions=directions,
            alternatives=alternatives,landing_support=landing,
            geometry_invocation='explicit_model_requested_inspect_geometry',
            information_options=['get_waypoint_candidates','turn_and_observe'])
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

    def _revalidate_path(self, path, resume=False):
        return self.motion_controller.validate(self, path, resume)

    def _navigate_relative(self, turn_deg, distance_m, _source=None):
        """Direct model intent, without a preselected semantic candidate.

        The bearing is relative to the current heading, not a global pose or
        semantic permission. A TURN is always legal as a physical rotation;
        each subsequent MOVE independently validates fresh RGB-D.
        """
        if (not math.isfinite(turn_deg) or not math.isfinite(distance_m)
                or abs(turn_deg) > 180 or not 0.25 <= distance_m <= 3.0):
            return dict(status='INVALID_ARGUMENTS', reason='relative_motion_out_of_bounds')
        stride = self.camera.turn_angle
        step = self.camera.forward_step
        # Models express motion intent in continuous units; the executor
        # quantizes to the supported Habitat actuator increments and reports
        # what was actually requested of the simulator.
        actual_turn = round(turn_deg / stride) * stride
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
            endpoint_xz_m=[source['pose']['x'] - math.sin(heading)*actual_distance,
                           source['pose']['z'] + math.cos(heading)*actual_distance])
        # No automatic collision check: geometry is used only when the model
        # explicitly requests its inspection tool.
        self.paths[path_id] = path
        # Direct routes were absent from spatial memory because they skip
        # get_waypoint_candidates. Register the proposal, not a clearance
        # certificate. Action completion does NOT imply metric target arrival.
        self.memory.note_candidates([path])
        result = self._navigate_local(path_id)
        return dict(result, waypoint_id=path_id,
                    requested_motion=dict(turn_deg=turn_deg,distance_m=distance_m,
                        quantized_turn_deg=actual_turn,quantized_distance_m=actual_distance,
                        authorization='model_selected_atomic_actions_not_metric_arrival'))

    def _navigate_to_pixel(self, u, v, distance_m):
        """Turn toward a live measured RGB-D pixel; do not navigate to its surface."""
        if not (math.isfinite(u) and math.isfinite(v) and
                0 <= u <= 1 and 0 <= v <= 1):
            return dict(status='INVALID_ARGUMENTS',reason='invalid_normalized_pixel')
        self.refresh()
        geometry = self.geometry()
        x = min(self.camera.width-1, max(0, round(u*(self.camera.width-1))))
        y = min(self.camera.height-1, max(0, round(v*(self.camera.height-1))))
        import numpy as np
        y0,y1=max(0,y-2),min(self.camera.height,y+3)
        x0,x1=max(0,x-2),min(self.camera.width,x+3)
        valid = geometry.valid[y0:y1,x0:x1]
        if int(valid.sum()) < 5:
            return dict(status='NEEDS_VIEW',reason='pixel_target_depth_unmeasured',
                        pixel=dict(u=u,v=v),missing_evidence=['valid_local_depth'])
        xs = geometry.x[y0:y1,x0:x1][valid]
        zs = geometry.z[y0:y1,x0:x1][valid]
        x_m,z_m=float(np.median(xs)),float(np.median(zs))
        if z_m <= 0:
            return dict(status='NEEDS_VIEW',reason='pixel_target_not_forward_projectable',
                        pixel=dict(u=u,v=v))
        measured_bearing = math.degrees(math.atan2(-x_m,z_m))
        turn_deg = round(measured_bearing/self.camera.turn_angle)*self.camera.turn_angle
        if abs(turn_deg) > 180:
            return dict(status='INVALID_ARGUMENTS',reason='pixel_bearing_out_of_range')
        pixel_source = self.source()
        result = self._navigate_relative(turn_deg,distance_m,
                                         _source=pixel_source)
        return dict(result, pixel_target=dict(u=u,v=v,
            source_observation_id=pixel_source['observation_id'],
            measured_depth_m=round(float(np.median(geometry.depth[y0:y1,x0:x1][valid])),3),
            measured_bearing_deg=round(measured_bearing,2),
            selected_turn_deg=turn_deg,
            interpretation='direction_hint_only_pixel_not_walkability_certificate'))

    def _navigate_local(self, waypoint_id):
        path = self.paths.get(waypoint_id)
        if path is None:
            return dict(status='INVALID_TARGET', reason='unknown_waypoint_id')
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
            result = self._execute_waypoint(target)
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
