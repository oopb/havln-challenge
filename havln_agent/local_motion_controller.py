"""Short horizon geometry and feedback; never semantic route planning."""

import math
import time

from .spatial import angle_delta


class LocalMotionController:
    """Action-receipt orientation and conservative physical-stance identity.

    No RGB-D translational registration feeds the navigation controller.
    x/y/z remain neutral, explicitly INVALID legacy compatibility fields;
    no caller may use them as coordinates, distance, route completion or
    collision evidence. Each executed MOVE_FORWARD starts a new stance epoch,
    even if the images are byte-identical: conservative separation is safer
    than mistakenly reusing a previous camera scan at a changed location.
    """

    def __init__(self, camera):
        self.camera = camera
        # Legacy numeric attributes for code that still serializes a pose.
        # They are never integrated, and pose marks them INVALID.
        self.x, self.y, self.z = 0., 0., 0.
        self.heading, self.pitch_deg = 0., 0.
        self.target = None  # active path ID, NOT an estimated x/z endpoint
        self.motion_epoch = 0
        self.viewpoint_generation = 0
        self.forward_actions_attempted = 0
        self.forward_actions_with_frame_change = 0
        self.forward_stall_actions = 0
        # Kept as a descriptive compatibility count, not a registration
        # failure or a control-flow gate: ALL translation is unmeasured.
        self.unknown_forward_actions = 0
        self.global_chain_continuous = False
        self.position_uncertainty = None
        self.motion_error_total = None
        self.local_reference = 0
        self.local_anchor_observation_id = None

    def pose(self):
        return dict(
            x=self.x, y=self.y, z=self.z,
            heading=self.heading, pitch_deg=self.pitch_deg,
            source='commanded_yaw_pitch_and_public_action_receipts',
            confidence='action_only_no_metric_localization',
            position_uncertainty=None, local_reference=self.local_reference,
            motion_error_total=None, motion_epoch=self.motion_epoch,
            displacement_verified=False,
            uncertainty_source='metric_translation_not_computed',
            forward_actions_attempted=self.forward_actions_attempted,
            forward_actions_with_frame_change=self.forward_actions_with_frame_change,
            forward_stall_actions=self.forward_stall_actions,
            unknown_forward_actions=self.unknown_forward_actions,
            viewpoint_generation=self.viewpoint_generation,
            stance_id='%d:%d' % (self.motion_epoch,self.viewpoint_generation),
            stance_identity_source='executed_forward_action_receipts',
            position_coordinates_valid=False,
            metric_position_valid=False,
            metric_position=None,
            legacy_xyz_are_placeholders=True,
            global_chain_continuous=False,
            local_anchor_observation_id=None,
            coordinate_scope='action_stance_only_without_metric_xyz')

    def accept_feedback(self, action, measurement):
        if action == 'TURN_LEFT':
            self.heading += math.radians(self.camera.turn_angle)
        elif action == 'TURN_RIGHT':
            self.heading -= math.radians(self.camera.turn_angle)
        elif action == 'MOVE_FORWARD':
            # Increment ONLY when Habitat actually executed the atomic action.
            # A Depth preflight veto never invokes accept_feedback.
            self.forward_actions_attempted += 1
            self.viewpoint_generation += 1
            self.unknown_forward_actions += 1
            changed=bool(measurement and
                         measurement.get('observed_frame_change',False))
            if changed:
                self.forward_actions_with_frame_change += 1
            else:
                self.forward_stall_actions += 1
            # Do not integrate, reanchor, extrapolate or mark x/y/z valid,
            # even if a legacy caller passes a numerical measurement.
        self.heading=(self.heading+math.pi)%(2*math.pi)-math.pi

    def anchor_relative_frame(self, observation_id):
        """Compatibility hook: local RGB-D frames are indexed by stance ID.

        No metric coordinate chain exists to reanchor.
        """
        return False

    def next_action(self):
        """No hidden metric waypoint steering: use route_action instead."""
        return None

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
        """Validate current source stance and sensor evidence, NOT task milestones."""
        r = runtime
        source, pose = path['source'], self.pose()
        if not resume:
            # A historical candidate is only reusable before any intervening
            # executed forward. No comparison of fake x/z placeholders.
            source_pose=source['pose']
            if (source_pose.get('motion_epoch') != pose.get('motion_epoch')
                    or source_pose.get('viewpoint_generation') !=
                       pose.get('viewpoint_generation')):
                return dict(status='STALE_TARGET',
                            reason='executed_forward_since_waypoint_proposal')
            # resume=True is the SAME actively executing model-chosen route;
            # its finite forward horizon is counted from action receipts.
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
        endpoint = None  # No metric position/endpoint can be inferred from receipts.
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
            original_target=None, executable_segment=segment,
            validation=validation)
        path['execution_status'] = 'REVALIDATED'
        r.memory.note_validation(path, result)
        return result
