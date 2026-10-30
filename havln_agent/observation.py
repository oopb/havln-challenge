"""Raw-array fingerprints and checks against the executed action estimate."""
import hashlib
import math

import numpy as np


def array_hash(array):
    data = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(data.dtype).encode())
    digest.update(str(data.shape).encode())
    digest.update(data.tobytes())
    return digest.hexdigest()


def frame_info(observation, camera):
    raw = observation.depth
    finite = raw[np.isfinite(raw)]
    metres = raw * (camera.max_depth - camera.min_depth) + camera.min_depth if camera.normalized else raw
    valid = np.isfinite(metres) & (metres > camera.min_depth) & (metres <= camera.max_depth)
    values = metres[valid]
    return dict(rgb_hash=array_hash(observation.rgb), depth_hash=array_hash(raw),
                rgb_shape=list(observation.rgb.shape), depth_shape=list(raw.shape),
                raw_depth_min=float(finite.min()) if finite.size else None,
                raw_depth_max=float(finite.max()) if finite.size else None,
                valid_depth_min_metres=float(values.min()) if values.size else None,
                valid_depth_max_metres=float(values.max()) if values.size else None,
                valid_depth_fraction=float(valid.mean()), normalized=camera.normalized)


class ObservationAudit:
    def __init__(self, camera):
        self.camera = camera
        self.previous = None
        self.previous_heading = None
        self.seen_views = set()
        self.seen_directions = set()
        self.repeated_views = 0
        self.repeated_directions = 0
        self.turn_anomalies = 0
        self.stalled_turns = 0

    def update(self, observation, pose, previous_action):
        info = frame_info(observation, self.camera)
        heading = pose['heading']
        delta = 0 if self.previous_heading is None else (heading - self.previous_heading + math.pi) % (2 * math.pi) - math.pi
        same_rgb = self.previous is not None and info['rgb_hash'] == self.previous['rgb_hash']
        same_depth = self.previous is not None and info['depth_hash'] == self.previous['depth_hash']
        turning = previous_action in ('TURN_LEFT', 'TURN_RIGHT')
        unchanged_turn = turning and abs(delta) > 1e-6 and same_rgb and same_depth
        expected = math.radians(self.camera.turn_angle) * (1 if previous_action == 'TURN_LEFT' else -1)
        heading_matches = not turning or math.isclose(delta, expected, abs_tol=1e-6)
        origin = (round(pose['x'], 6), round(pose['z'], 6))
        direction = origin + (round(heading / math.radians(self.camera.turn_angle)) % round(360 / self.camera.turn_angle),)
        view = direction + (info['rgb_hash'], info['depth_hash'])
        repeated_view = view in self.seen_views
        repeated_direction = direction in self.seen_directions
        self.repeated_views += int(repeated_view)
        self.repeated_directions += int(repeated_direction)
        self.turn_anomalies += int(unchanged_turn or not heading_matches)
        self.stalled_turns = self.stalled_turns + 1 if unchanged_turn or not heading_matches else 0
        self.seen_views.add(view)
        self.seen_directions.add(direction)
        info.update(heading=heading, heading_source='action_odometry', previous_action=previous_action,
                    heading_delta=delta, heading_matches_action=heading_matches,
                    rgb_changed=self.previous is None or not same_rgb,
                    depth_changed=self.previous is None or not same_depth,
                    unchanged_after_turn=unchanged_turn, repeated_view=repeated_view,
                    repeated_direction=repeated_direction,
                    update_confirmed=not unchanged_turn and heading_matches)
        self.previous, self.previous_heading = info, heading
        return info

    def stats(self):
        return dict(unique_views=len(self.seen_views), unique_pose_directions=len(self.seen_directions),
                    repeated_view_observations=self.repeated_views,
                    repeated_direction_observations=self.repeated_directions,
                    unchanged_turn_anomalies=self.turn_anomalies)
