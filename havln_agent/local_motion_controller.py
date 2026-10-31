"""Short horizon geometry and feedback; never semantic route planning."""

import math
import time

from .spatial import angle_delta, relative_pose_error
from .safety import forward_probe


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

    def pose(self):
        return dict(x=self.x, y=self.y, z=self.z, heading=self.heading, pitch_deg=self.pitch_deg,
            source='rgbd_translation_and_commanded_yaw', confidence='estimated',
            position_uncertainty=self.position_uncertainty, local_reference=self.local_reference,
            motion_error_total=self.motion_error_total, motion_epoch=self.motion_epoch,
            displacement_verified=False, uncertainty_source='registration_residual_not_ground_truth',
            global_chain_continuous=self.global_chain_continuous,
            local_anchor_observation_id=self.local_anchor_observation_id,
            coordinate_scope=('episode_relative' if self.global_chain_continuous else
                              'reanchored_local_relative' if self.local_anchor_observation_id else
                              'unanchored_after_unknown_translation'))

    def accept_feedback(self, action, measurement):
        if action == 'TURN_LEFT':
            self.heading += math.radians(self.camera.turn_angle)
        elif action == 'TURN_RIGHT':
            self.heading -= math.radians(self.camera.turn_angle)
        elif action == 'MOVE_FORWARD':
            if not measurement or measurement['status'] != 'ESTIMATED':
                self.position_uncertainty += self.camera.forward_step
                self.motion_epoch += 1
                self.global_chain_continuous = False
                self.local_anchor_observation_id = None
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
    def support(geometry):
        return geometry.assess_step()['support']

    @staticmethod
    def needs_view(reason, **extra):
        suggestion = ('analyze_scene_for_current_people' if 'human' in reason else
                      'ground_a_visible_local_target_again' if 'objective' in reason or 'pose_uncertainty' in reason else
                      'turn_toward_target_and_observe' if 'fov' in reason else
                      'look_down_then_inspect_geometry')
        return dict(status='NEEDS_VIEW', reason=reason, missing_evidence=[reason],
                    suggestion=suggestion, information_options=['turn_and_observe', 'analyze_scene',
                        'assess_entry', 'ground_landmark', 'inspect_geometry'], **extra)

    @staticmethod
    def next_step_state(geometry):
        return forward_probe(geometry)

    def validate(self, runtime, path, resume=False, uncertain_straight=False):
        r = runtime
        source, pose = path['source'], self.pose()
        if source['subgoal'] != r.memory.task.current_subgoal_index:
            return dict(status='STALE_TARGET', reason='language_goal_changed')
        if not resume and math.hypot(source['pose']['x']-pose['x'], source['pose']['z']-pose['z']) > self.camera.forward_step/2:
            return dict(status='STALE_TARGET', reason='path_start_position_changed')
        relative_error = relative_pose_error(source['pose'], pose)
        if uncertain_straight:
            # An UNKNOWN registration breaks global endpoint coordinates, but
            # not the direction of a short, already accepted straight segment.
            # Only ToolExecutor's bounded same-waypoint recovery may use this
            # mode; it never certifies arrival or old landmark geometry.
            commanded_heading = source['pose']['heading'] + path['angle']
            if not resume or self.target is None or math.isfinite(relative_error):
                return dict(status='REQUIRES_ATTENTION',
                    reason='relative_motion_recovery_not_applicable')
            if abs(angle_delta(commanded_heading, pose['heading'])) > math.radians(
                    self.camera.turn_angle) / 2:
                return dict(status='REQUIRES_ATTENTION',
                    reason='relative_motion_heading_changed')
        elif relative_error > .5:
            return self.needs_view('pose_uncertainty_requires_reconfirmation')
        r.refresh()
        if time.monotonic()-r.observation.captured_at > r.config['max_depth_age_seconds']:
            return self.needs_view('fresh_depth_required')
        heading = source['pose']['heading']+path['angle']
        endpoint = self.target if resume else (
            source['pose']['x']-math.sin(heading)*path['distance'],
            source['pose']['z']+math.cos(heading)*path['distance'])
        dx, dz = endpoint[0]-pose['x'], endpoint[1]-pose['z']
        if uncertain_straight:
            # Never interpret old x/z as a reliable distance to the target.
            # ToolExecutor bounds the number of commanded forward actions.
            angle, distance = 0., self.camera.forward_step
        else:
            angle, distance = angle_delta(math.atan2(-dx, dz), pose['heading']), math.hypot(dx, dz)
        turn_only = abs(angle) > math.radians(self.camera.turn_angle)/2
        geometry = r.geometry()
        # Keep the calibrated body sweep's visibility horizon; only one atomic
        # displacement is authorized, even if the selected endpoint is further.
        # Turning is not forward translation. After the real TURN, a new RGB-D
        # view must certify the actual straight-ahead 0-degree body sweep.
        diagnostic = (dict(scope='turn_only_no_forward_permission') if turn_only else
                      geometry.diagnose(0, self.camera.forward_step))
        next_action = ('TURN_LEFT' if angle > 0 else 'TURN_RIGHT') if turn_only else 'MOVE_FORWARD'
        validation = dict(source=r.source(), geometry=diagnostic,
            relative_motion_only=uncertain_straight,
            endpoint_estimate_valid=not uncertain_straight,
            next_atomic_action=next_action,
            validated_forward_bearing_deg=(0 if not turn_only else None),
            requires_post_turn_recheck=turn_only,
            scope='next_atomic_step_only', dynamic_safety='per_forward_step_required')
        path['last_validation'] = validation
        if not turn_only:
            check = self.next_step_state(geometry)
            if check['state'] == 'BLOCKED':
                return dict(status='BLOCKED', reason=check['reason'],
                    suggestion='inspect_geometry_if_obstacle_cause_is_unclear',
                    information_options=['inspect_geometry','turn_and_observe'],
                    safety=dict(diagnostic,terrain_evidence=check['support'],
                                check_mode=check.get('check_mode')),
                    validation=validation, original_target=list(endpoint))
            if check['state'] == 'NEEDS_VIEW':
                return self.needs_view(check['reason'],
                    safety=dict(diagnostic,terrain_evidence=check['support'],
                                check_mode=check.get('check_mode')),
                    validation=validation, original_target=list(endpoint))
        segment_distance = 0 if turn_only else min(distance, self.camera.forward_step)
        segment = dict(path_id=path['id'], origin_xz_m=[pose['x'], pose['z']],
            endpoint_xz_m=[pose['x']-math.sin(pose['heading'])*segment_distance,
                           pose['z']+math.cos(pose['heading'])*segment_distance],
            distance_m=segment_distance, turn_only=turn_only, source=r.source(), requires_live_safety=True)
        value = dict(status='OK', candidate=dict(path, angle=angle, distance=segment_distance),
            original_target=list(endpoint), executable_segment=segment, validation=validation)
        path['execution_status'] = 'REVALIDATED'
        r.memory.note_validation(path, value)
        return value
