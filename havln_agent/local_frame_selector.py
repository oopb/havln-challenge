"""Real RGB-D keyframes for a short-horizon local VLM, without new memory.

Frames are indexed by immutable Observation IDs. Historical orientation is a
signed camera-relative viewing offset, never a robot waypoint or free-space
certificate. The current RGB is always last among RGB frames.
"""
import math

from .api_client import APIError
from .navigation_history import navigation_visual_anchors
from .observation import array_hash
from .perception import image_url, depth_image_url


def _relative_degrees(heading, current):
    return round((math.degrees(heading-current)+180) % 360-180, 1)


def _view_key(record, camera):
    pose = record.get('pose') or {}
    yaw = pose.get('heading', 0.)
    pitch = pose.get('pitch_deg', 0.)
    return (pose.get('stance_id'),
            round(math.degrees(yaw) / camera.turn_angle),
            round(float(pitch) / camera.tilt_angle))


def select_local_frames(runtime, route, max_frames=10, max_edge=448):
    """Return (manifest, multimodal images). At most max_frames RGB + 1 Depth.

    Preserve route anchor, recent changed-forward evidence and diverse views;
    deduplicate identical stance/yaw/pitch, then order selected RGB chronologically.
    No cached image descriptions are treated as raw pixels.
    """
    memory = runtime.memory
    current = memory.current
    if current is None or runtime.observation is None:
        raise APIError('local_current_observation_missing', category='structured_field')
    if (array_hash(runtime.observation.rgb) != current.get('rgb_hash') or
            array_hash(runtime.observation.depth) != current.get('depth_hash')):
        raise APIError('local_current_observation_mismatch', category='structured_field')
    by_id = {row['observation_id']: row for row in memory.observations}
    current_id = current['observation_id']
    options = []
    def add(identifier, kind):
        if identifier in by_id and identifier != current_id:
            options.append((identifier, kind))

    add(route.get('source_observation_id'), 'cloud_route_source')
    add(route.get('start_observation_id'), 'cloud_route_start')
    # Consecutive true forward receipts are more useful than repeated turns.
    recent_forwards = [a for a in reversed(memory.actions[-64:])
                       if a.get('action') == 'MOVE_FORWARD' and
                       (a.get('result') or {}).get('visual_change')]
    for action in recent_forwards[:4]:
        feedback = action.get('result') or {}
        add(feedback.get('observation_before'), 'recent_forward_before')
        add(feedback.get('observation_after'), 'recent_forward_after')
    for candidate in navigation_visual_anchors(
            memory, current_id, max_pairs=max_frames * 2):
        add(candidate['observation_id'],
            candidate.get('selection_reason', 'historical_anchor'))
    # Preserve genuinely different recent rotation views if space remains.
    for item in reversed(memory.observations[-24:]):
        add(item['observation_id'], 'recent_view')

    selected = []
    used = {_view_key(current, memory.camera)}
    for identifier, kind in options:
        record = by_id[identifier]
        key = _view_key(record, memory.camera)
        if key in used:
            continue
        used.add(key)
        selected.append((record, kind))
        if len(selected) >= max_frames - 1:
            break
    order = {o['observation_id']: i for i, o in enumerate(memory.observations)}
    selected.sort(key=lambda pair: order[pair[0]['observation_id']])
    images, manifest = [], []
    current_yaw = (current.get('pose') or {}).get('heading', 0.)
    for record, kind in selected:
        try:
            frame = (runtime.load_observation(record) if runtime.load_observation
                     else runtime.frames[record['observation_id']])
            if (array_hash(frame.rgb) != record.get('rgb_hash') or
                    array_hash(frame.depth) != record.get('depth_hash')):
                continue
            encoded = image_url(frame, max_edge=max_edge)
        except (KeyError, OSError, ValueError, TypeError, AttributeError):
            continue
        pose = record.get('pose') or {}
        manifest.append(dict(image_index=len(images),
                             observation_id=record['observation_id'],
                             kind=kind, historical=True,
                             relative_yaw_deg=_relative_degrees(
                                 pose.get('heading', 0.), current_yaw),
                             pitch_deg=pose.get('pitch_deg'),
                             same_stance=(pose.get('stance_id') ==
                                          (current.get('pose') or {}).get('stance_id'))))
        images.append(encoded)
    pose = current.get('pose') or {}
    manifest.append(dict(image_index=len(images), observation_id=current_id,
                         kind='CURRENT_RGB', historical=False,
                         relative_yaw_deg=0., pitch_deg=pose.get('pitch_deg'),
                         same_stance=True))
    images.append(image_url(runtime.observation, max_edge=max_edge))
    # Current Depth is kept distinct from stale historical motion evidence.
    depth_index = len(images)
    images.append(depth_image_url(runtime.observation, memory.camera,
                                  max_edge=max_edge, display_max_m=5.0))
    return manifest, images, depth_index
