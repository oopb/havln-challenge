"""Short horizon geometry and feedback; never semantic route planning."""

import math
import time

from .spatial import angle_delta, relative_pose_error


class LocalMotionController:
    def __init__(self, camera):
        self.camera = camera
        self.x, self.y, self.z, self.heading = 0., 0., 0., 0.
        self.target = None
        self.position_uncertainty = .05
        self.motion_error_total, self.motion_epoch, self.local_reference = 0., 0, 0
        self.pitch_deg = 0.0
        # A failed RGB-D registration cannot be repaired by assuming an action
        # moved the robot 0.25m. It invalidates the old global-ish chain, but a
        # fresh public RGB-D image can anchor NEW local relative waypoints.
        self.local_anchor_observation_id = None
        self.global_chain_continuous = True
        # Command receipts are independent of optional RGB-D translation fitting.
        # A changed post-action view starts a new observed stance; failed
        # registration alone never resets a waypoint or its action budget.
        self.forward_actions_attempted = 0
        self.unknown_forward_actions = 0
        self.viewpoint_generation = 0

    def pose(self):
        return dict(x=self.x, y=self.y, z=self.z, heading=self.heading, pitch_deg=self.pitch_deg,
            source='optional_rgbd_translation_and_commanded_yaw',
            confidence=('estimated' if self.global_chain_continuous else 'action_only_unlocalized'),
            position_uncertainty=self.position_uncertainty, local_reference=self.local_reference,
            motion_error_total=self.motion_error_total, motion_epoch=self.motion_epoch,
            displacement_verified=False, uncertainty_source='registration_residual_not_ground_truth',
            forward_actions_attempted=self.forward_actions_attempted,
            unknown_forward_actions=self.unknown_forward_actions,
            viewpoint_generation=self.viewpoint_generation,
            position_coordinates_valid=self.global_chain_continuous,
            global_chain_continuous=self.global_chain_continuous,
            local_anchor_observation_id=self.local_anchor_observation_id,
            coordinate_scope=('episode_relative' if self.global_chain_continuous else
                              'action_only_after_unregistered_translation'))

    def accept_feedback(self, action, measurement):
        if action == 'TURN_LEFT':
            self.heading += math.radians(self.camera.turn_angle)
        elif action == 'TURN_RIGHT':
            self.heading -= math.radians(self.camera.turn_angle)
        elif action == 'MOVE_FORWARD':
            self.forward_actions_attempted += 1
            # Only a changed public frame supports treating the new view as
            # a different physical stance. Neither a receipt nor a command
            # certifies the nominal 0.25m of movement.
            if measurement and measurement.get('observed_frame_change', False):
                self.viewpoint_generation += 1
            if not measurement or measurement['status'] != 'ESTIMATED':
                self.unknown_forward_actions += 1
                self.global_chain_continuous = False
                # Preserve metric uncertainty as UNKNOWN (via pose confidence)
                # rather than adding an invented 0.25m uncertainty or
                # invalidating the model-selected route with a new motion_epoch.
                return
            motion=measurement['translation_m']
            x,z=motion['x_right'],motion['z_forward']
            self.x += x*math.cos(self.heading)-z*math.sin(self.heading)
            self.z += x*math.sin(self.heading)+z*math.cos(self.heading)
            self.y += motion['y_up']
            error=measurement['uncertainty_m']
            self.position_uncertainty += error
            self.motion_error_total += error
            self.local_reference += 1
        self.heading=(self.heading+math.pi)%(2*math.pi)-math.pi

    def anchor_relative_frame(self, observation_id):
        """Start a NEW relative waypoint chain from actual post-action RGB-D.

        No missing translation is inferred. Old global coordinates remain
        numerically present for historical bookkeeping, but are invalid across
        motion_epoch. Within this new chain only *deltas* to subsequent live
        frames are meaningful.
        """
        if self.global_chain_continuous or not observation_id:
            return False
        self.local_anchor_observation_id = observation_id
        return True

    def next_action(self):
        if self.target is None:
            return None
        dx,dz=self.target[0]-self.x,self.target[1]-self.z
        if math.hypot(dx,dz) < self.camera.forward_step*.6:
            return None
        angle=angle_delta(math.atan2(-dx,dz),self.heading)
        if abs(angle)>math.radians(self.camera.turn_angle)/2:
            return 'TURN_LEFT' if angle>0 else 'TURN_RIGHT'
        return 'MOVE_FORWARD'

    @staticmethod
    def needs_view(reason, **extra):
        suggestion = ('analyze_scene_for_current_people' if 'human' in reason else
                      'ground_a_visible_local_target_again' if 'objective' in reason or 'pose_uncertainty' in reason else
                      'turn_toward_target_and_observe' if 'fov' in reason else
                      'look_down_then_inspect_geometry')
        return dict(status='NEEDS_VIEW', reason=reason, missing_evidence=[reason],
                    suggestion=suggestion, information_options=['turn_and_observe', 'analyze_scene',
                        'assess_entry', 'ground_landmark', 'inspect_geometry'], **extra)

    def route_action(self, path, forwards_done, forward_horizon):
        """Return next atomic action for a finite model-authored bearing/step plan.

        Do not steer using unregistered x/z odometry. TURN uses commanded yaw,
        then the model's finite count of forward receipts is consumed.
        """
        target_heading = path['source']['pose']['heading'] + path['angle']
        yaw_error = angle_delta(target_heading, self.heading)
        if abs(yaw_error) > math.radians(self.camera.turn_angle) / 2:
            return 'TURN_LEFT' if yaw_error > 0 else 'TURN_RIGHT'
        if forwards_done < forward_horizon:
            return 'MOVE_FORWARD'
        return None

    def validate(self, runtime, path, resume=False, uncertain_straight=False):
        """Validate goal identity and fresh observation, NOT metric odometry."""
        r = runtime
        source, pose = path['source'], self.pose()
        if source['subgoal'] != r.memory.task.current_subgoal_index:
            return dict(status='STALE_TARGET', reason='language_goal_changed')
        if not resume:
            # Saved candidate waypoints from a different visual stance should
            # not be silently reused. An active finite route is still resumable
            # even if several RGB-D registrations failed mid-command.
            start_stance = source['pose'].get('viewpoint_generation')
            current_stance = pose.get('viewpoint_generation')
            if (start_stance is not None and current_stance is not None
                    and start_stance != current_stance):
                return dict(status='STALE_TARGET', reason='viewpoint_changed_since_waypoint_proposal')
            if math.hypot(source['pose']['x']-pose['x'],
                          source['pose']['z']-pose['z']) > self.camera.forward_step / 2:
                return dict(status='STALE_TARGET', reason='path_start_position_changed')
        r.refresh()
        if time.monotonic() - r.observation.captured_at > r.config['max_depth_age_seconds']:
            return self.needs_view('fresh_rgbd_required')
        attempt = getattr(r.memory, 'active_attempt', None) or {}
        same_path = (attempt.get('target') or {}).get('id') == path['id']
        forward_horizon = (attempt.get('forward_action_horizon')
                           if same_path else None)
        if forward_horizon is None:
            forward_horizon = max(1, math.ceil(
                (path['distance'] - 1e-6) / self.camera.forward_step))
        forwards_done = attempt.get('forward_actions', 0) if same_path else 0
        action = self.route_action(path, forwards_done, forward_horizon)
        target_heading = source['pose']['heading'] + path['angle']
        endpoint = self.target if resume and self.target is not None else (
            source['pose']['x']-math.sin(target_heading)*path['distance'],
            source['pose']['z']+math.cos(target_heading)*path['distance'])
        turn_only = action in ('TURN_LEFT','TURN_RIGHT')
        remaining_actions = max(0, forward_horizon - forwards_done)
        diagnostic = dict(scope='model_requested_geometry_only',
            status='NOT_CHECKED',collision_clearance_m=None)
        validation = dict(source=r.source(),geometry=diagnostic,
            relative_motion_only=True,
            endpoint_estimate_valid=False,
            next_atomic_action=action,
            validated_forward_bearing_deg=None,
            requires_post_turn_recheck=False,
            forward_evidence_state='NOT_CHECKED',
            commanded_forward_actions=forwards_done,
            commanded_forward_horizon=forward_horizon,
            scope='next_atomic_action_not_metric_arrival',
            geometry_policy='explicit_model_tool_only')
        path['last_validation'] = validation
        segment_distance = self.camera.forward_step if action == 'MOVE_FORWARD' else 0
        segment = dict(path_id=path['id'], origin_xz_m=None, endpoint_xz_m=None,
            distance_m=segment_distance, turn_only=turn_only,
            source=r.source(), geometry_checked=False,
            reach_basis='commanded_atomic_action_not_verified_displacement')
        result = dict(status='OK',
            candidate=dict(path, angle=angle_delta(target_heading, pose['heading']),
                           distance=remaining_actions*self.camera.forward_step),
            original_target=list(endpoint), executable_segment=segment,
            validation=validation)
        path['execution_status'] = 'REVALIDATED'
        r.memory.note_validation(path, result)
        return result
