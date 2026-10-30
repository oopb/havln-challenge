"""Allowlisted RGB-D input, camera-calibrated geometry and low-frequency semantics."""
import base64
from dataclasses import asdict, dataclass
import io
import math
import re
import time

import numpy as np
from PIL import Image

from .api_client import FLAG, TEXT, object_schema
from .vision_protocol import coordinate_prompt, coordinate_schema, coordinate_unit, normalize_coordinates


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    hfov: float
    min_depth: float
    max_depth: float
    normalized: bool
    camera_height: float
    robot_height: float
    robot_radius: float
    forward_step: float
    turn_angle: float
    tilt_angle: float = 15.0
    camera_x: float = 0.0
    camera_z: float = 0.0
    max_step_height: float = 0.2

    def __post_init__(self):
        if not (self.width > 1 and self.height > 1 and 0 < self.hfov < 180 and
                0 <= self.min_depth < self.max_depth and self.camera_height > 0 and
                self.robot_height > 0 and self.robot_radius > 0 and
                self.forward_step > 0 and 0 < self.turn_angle <= 90 and 0 < self.tilt_angle <= 90 and 0 < self.max_step_height <= .2):
            raise ValueError('Unsupported camera or action calibration')


@dataclass
class Observation:
    rgb: np.ndarray
    depth: np.ndarray
    instruction: str
    captured_at: float
    pitch_deg: float = 0.0


def adapt_observation(raw, camera):
    # Deliberately copy only public inputs; raw can contain privileged fields.
    rgb = np.asarray(raw['rgb'])
    depth = np.asarray(raw['depth'])
    if depth.shape == (camera.height, camera.width, 1):
        depth = depth[..., 0]
    if depth.shape != (camera.height, camera.width):
        raise ValueError('Depth shape differs from runtime camera calibration')
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB image')
    instruction = raw['instruction']['text']
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Instruction sensor did not provide text')
    return Observation(rgb.copy(), depth.copy(), instruction, time.monotonic())


class Geometry:
    def __init__(self, observation, camera, margin):
        self.camera, self.margin = camera, margin
        self.pitch_deg = observation.pitch_deg
        raw = observation.depth.astype(np.float64)
        self.depth = raw * (camera.max_depth - camera.min_depth) + camera.min_depth if camera.normalized else raw
        self.valid = np.isfinite(raw) & (self.depth > camera.min_depth) & (self.depth <= camera.max_depth)
        if camera.normalized:
            self.valid &= (raw > 0) & (raw <= 1)
        self.depth = np.where(self.valid, self.depth, np.nan)
        self.fx = camera.width / (2 * math.tan(math.radians(camera.hfov) / 2))
        # Square pixels: fy == fx even when sensor height differs from width.
        v, u = np.indices(raw.shape)
        self.ray_x = (u - (camera.width - 1) / 2) / self.fx
        camera_ray_height = -(v - (camera.height - 1) / 2) / self.fx
        pitch = math.radians(observation.pitch_deg)
        self.ray_height = camera_ray_height * math.cos(pitch) + math.sin(pitch)
        self.ray_forward = math.cos(pitch) - camera_ray_height * math.sin(pitch)
        self.x = camera.camera_x + (u - (camera.width - 1) / 2) * self.depth / self.fx
        self.z = camera.camera_z + self.depth * self.ray_forward
        h = camera.camera_height + self.ray_height * self.depth
        self.height = h
        self.obstacles = self.valid & (h > 0.08) & (h < camera.robot_height)
        self.human_mask = np.zeros(raw.shape, dtype=bool)
        self.human_status = dict(source='current_depth_occupancy', semantic_identity='UNKNOWN', confirmed=0, uncertain=0, unlocalized=0)
        self._support_planes = None
        self._step_checks = {}

    def set_humans(self, scene):
        self.human_mask.fill(False)
        self._support_planes = None
        self._step_checks = {}
        people, uncertain = scene.get('humans', []), scene.get('uncertain_humanoids', [])
        self.human_status = dict(source='current_rgb_with_live_depth', confirmed=len(people),
                                 uncertain=len(uncertain), unlocalized=0)
        for entry in people + uncertain:
            box = entry.get('box')
            if box is None or not (box[0] < box[2] and box[1] < box[3]):
                self.human_status['unlocalized'] += 1
                continue
            x0, y0, x1, y1 = box
            self.human_mask[int(y0 * self.camera.height):int(math.ceil(y1 * self.camera.height)),
                            int(x0 * self.camera.width):int(math.ceil(x1 * self.camera.width))] = True

    def clearance(self, angle, limit):
        """Free distance along a body-width strip; positive bearing means left."""
        return self.diagnose(angle, limit)['clearance']

    def localize_human(self, box):
        """A box-depth estimate for memory; never an alternative collision test."""
        if box is None:
            return None
        left, top, right, bottom = box
        lo, hi = int(left * self.camera.width), int(math.ceil(right * self.camera.width))
        y0, y1 = int(top * self.camera.height), int(math.ceil(bottom * self.camera.height))
        depth = self.depth[y0:y1, lo:hi]
        valid = depth[np.isfinite(depth)]
        if not valid.size:
            return None
        distance = float(np.median(valid))
        x = ((left + right) * self.camera.width / 2 - (self.camera.width - 1) / 2) * distance / self.fx
        body_depth = self.z[y0:y1, lo:hi]
        return dict(x_right=x, z_forward=float(np.median(body_depth[np.isfinite(depth)])), frame='current_robot_camera',
                    source='rgb_box_depth_median', valid_fraction=float(valid.size / depth.size))

    def diagnose(self, angle, limit, obstacles=None):
        cam = self.camera
        radius = cam.robot_radius + self.margin
        limit = min(limit, cam.max_depth - radius)
        info = dict(angle=angle, angle_degrees=math.degrees(angle), clearance=0.0,
                    required_forward_step=cam.forward_step, robot_radius=cam.robot_radius,
                    safety_margin=self.margin, collision_strip_radius=radius,
                    inspected_distance=limit, depth_unit='metres', nearest_obstacle_distance=None,
                    collision_points=0, unknown_depth_pixels=0,
                    unknown_fraction=None, geometry_confidence='unknown', collision_strip=None,
                    path_state='UNKNOWN', unknown_sweep_pixels=0, unknown_outside_sweep_pixels=0,
                    unknown_ray_evidence=[],
                    collision_model='cylindrical_body_sweep', nearest_collision_travel=None,
                    body_height_band=[0.08, cam.robot_height], terrain_support='not_certified',
                    human_occupancy=dict(self.human_status, corridor_points=0, nearest_distance=None),
                    accepted=False, rejection_reasons=[])
        if limit <= 0:
            info['rejection_reasons'].append('depth_range_too_short')
            return info
        # Require the target footprint to lie within the observed horizontal FOV.
        if abs(angle) + math.atan2(radius, max(limit, cam.forward_step+radius)) >= math.radians(cam.hfov / 2):
            info['rejection_reasons'].append('outside_observed_fov')
            return info
        center = (cam.width - 1) / 2 - self.fx * math.tan(angle)
        half_width = self.fx * radius / max(cam.forward_step + radius, 0.5)
        lo, hi = max(0, int(center - half_width)), min(cam.width, int(center + half_width) + 1)
        # Invalid depths in the body-facing image region are unknown, never free.
        band = self.valid[:,lo:hi]
        info['unknown_depth_pixels'] = int((~band).sum())
        info['unknown_fraction'] = float((~band).mean()) if band.size else None
        info['collision_strip'] = dict(columns=[lo, hi], rows=[0,cam.height],
                                       radius_metres=radius, pixel_count=int(band.size))
        info['geometry_confidence'] = 'partial_depth' if not band.size or not band.all() else 'observed_depth'
        y0, y1 = 0,cam.height
        rows, columns = np.nonzero(~self.valid[y0:y1, lo:hi])
        rows, columns = rows + y0, columns + lo
        relevant, evidence = self._unknown_sweep(rows, columns, angle, min(limit, cam.forward_step))
        info['unknown_sweep_pixels'] = int(relevant.sum())
        info['unknown_outside_sweep_pixels'] = int(len(rows) - relevant.sum())
        info['unknown_ray_evidence'] = evidence
        info['sweep'] = dict(frame='candidate_local', along=[0, min(limit, cam.forward_step) + radius],
                             across=[-radius, radius], height=[0.08, cam.robot_height],
                             method='cylindrical_body_sweep_in_forward_camera_band',
                             known_current_body_radius=cam.robot_radius)
        along = -self.x * math.sin(angle) + self.z * math.cos(angle)
        across = self.x * math.cos(angle) + self.z * math.sin(angle)
        corridor = (self.obstacles if obstacles is None else obstacles) & (np.abs(across) <= radius) & (along > 0)
        info['collision_points'] = int(corridor.sum())
        if corridor.any():
            info['nearest_obstacle_distance'] = float(np.min(along[corridor]))
        # A circular footprint contacts each point according to its lateral
        # offset. Subtracting the full radius everywhere rejects empty corners
        # of the rectangular envelope, especially when looking obliquely.
        contact = along - np.sqrt(np.maximum(0, radius ** 2 - across ** 2))
        if corridor.any():
            info['nearest_collision_travel'] = float(np.min(contact[corridor]))
        humans = corridor & self.human_mask
        info['human_occupancy']['corridor_points'] = int(humans.sum())
        if humans.any():
            nearest_human = float(np.min(along[humans]))
            info['human_occupancy']['nearest_distance'] = nearest_human
            if float(np.min(contact[humans])) < cam.forward_step:
                info['rejection_reasons'].append('human_or_uncertain_humanoid_occupied')
        collision_travel = info['nearest_collision_travel']
        info['clearance'] = max(0.0, min(limit, collision_travel if collision_travel is not None else limit))
        occupied = collision_travel is not None and collision_travel < cam.forward_step
        if not band.size or relevant.any():
            info['clearance'] = 0.0
            info['rejection_reasons'].append('unknown_depth_in_collision_strip')
        if info['clearance'] < cam.forward_step:
            info['rejection_reasons'].append('clearance_below_forward_step')
        info['accepted'] = not info['rejection_reasons']
        info['path_state'] = ('DYNAMIC' if 'human_or_uncertain_humanoid_occupied' in info['rejection_reasons'] else
                              'OCCUPIED' if occupied else 'UNKNOWN' if not band.size or relevant.any() or
                              limit < cam.forward_step else 'FREE')
        return info

    def _unknown_sweep(self, rows, columns, angle, distance):
        """Intersect unknown pixel rays with the first-step cylindrical sweep.

        A ray has no measured range. Exclude only ranges inside the current robot
        body or provably outside the body-height swept volume; never fill holes.
        The existing forward sensor window remains a visibility-model limitation.
        """
        cam = self.camera
        rx, rh = self.ray_x[rows, columns], self.ray_height[rows, columns]
        rz = self.ray_forward[rows, columns]
        along = -rx * math.sin(angle) + rz * math.cos(angle)
        across = rx * math.cos(angle) + rz * math.sin(angle)
        # Invalid/clipped pixels may hide objects inside MIN_DEPTH. That blind
        # near field is not free merely because the sensor cannot measure it.
        offset_along=-cam.camera_x*math.sin(angle)+cam.camera_z*math.cos(angle)
        offset_across=cam.camera_x*math.cos(angle)+cam.camera_z*math.sin(angle)
        near = cam.robot_radius / np.sqrt(rz * rz + rx * rx) if not (cam.camera_x or cam.camera_z) else np.zeros(rx.shape)
        far = np.full(rx.shape, cam.max_depth)
        radius = cam.robot_radius + self.margin
        for slope, offset, lower, upper in ((along, offset_along, 0, distance + radius),
                                             (across, offset_across, -radius, radius),
                                             (rh, cam.camera_height, 0.08, cam.robot_height)):
            parallel = np.abs(slope) < 1e-12
            a = np.divide(lower - offset, slope, out=np.full(rx.shape, -np.inf), where=~parallel)
            b = np.divide(upper - offset, slope, out=np.full(rx.shape, np.inf), where=~parallel)
            near, far = np.maximum(near, np.minimum(a, b)), np.minimum(far, np.maximum(a, b))
            far[parallel & ((offset < lower) | (offset > upper))] = -np.inf
        # The box above bounds the height interval. Refine its horizontal front
        # corners to a capsule: a disk swept along [0, distance]. Rays originate
        # at the segment start, so there is one positive exit from this volume.
        norm2 = along ** 2 + across ** 2
        side_exit = np.divide(radius, np.abs(across), out=np.full(rx.shape, np.inf),
                              where=np.abs(across) > 1e-12)
        cap_exit = (distance * along + np.sqrt(np.maximum(
            0, norm2 * radius ** 2 - distance ** 2 * across ** 2))) / norm2
        horizontal_exit = np.where(along <= 0, radius / np.sqrt(norm2),
                                   np.where(along * side_exit <= distance, side_exit, cap_exit))
        if not (cam.camera_x or cam.camera_z):
            far = np.minimum(far, horizontal_exit)
        relevant = (far >= near) & (far > 0)
        evidence = [dict(row=int(rows[i]), column=int(columns[i]),
                         possible_depth_metres=[float(near[i]), float(far[i])],
                         possible_along_metres=[float(along[i] * near[i]), float(along[i] * far[i])],
                         reason='unknown_ray_intersects_first_step_swept_body')
                    for i in np.flatnonzero(relevant)[:16]]
        return relevant, evidence

    @property
    def support_planes(self):
        if self._support_planes is None:
            from .terrain import measured_planes
            self._support_planes=measured_planes(self)
        return self._support_planes

    def assess_step(self, angle=0.):
        from . import terrain
        key=round(float(angle),6)
        if key in self._step_checks:return self._step_checks[key]
        proof=terrain.review(self,angle)
        depth=self.diagnose(angle,self.camera.forward_step,terrain.obstacle_mask(self,proof))
        if depth['path_state'] in ('OCCUPIED','DYNAMIC'):
            state,reason='BLOCKED','dynamic_human' if depth['path_state']=='DYNAMIC' else 'occupied_next_step'
        elif proof['status']=='BLOCKED':
            state,reason='BLOCKED',proof['reason']
        elif not depth['accepted']:
            state,reason='NEEDS_VIEW','unknown_depth_or_fov'
        elif self.human_status['unlocalized']:
            state,reason='NEEDS_VIEW','human_localization_unverified'
        elif proof['status']!='SUPPORTED':
            state,reason='NEEDS_VIEW',proof['reason']
        else:
            state,reason='EXECUTABLE',None
        result=dict(state=state,reason=reason,depth=depth,support=proof)
        self._step_checks[key]=result
        return result

    def metadata(self):
        return dict(camera=asdict(self.camera), pitch_deg=self.pitch_deg, depth_unit='metres',
                    safety_margin=self.margin, valid_fraction=float(self.valid.mean()),
                    human_occupancy=self.human_status)


BOX = {'type': 'array', 'minItems': 4, 'maxItems': 4,
       'items': {'type': 'number', 'minimum': 0, 'maximum': 1}}
UNCERTAIN_OBJECT = object_schema({'description': TEXT, 'box': {'anyOf': [BOX, {'type': 'null'}]}})

SCENE_SCHEMA = object_schema({
    'description': TEXT,
    'landmarks': {'type': 'array', 'maxItems': 30, 'items': object_schema({
        'name': TEXT, 'direction': {'enum': ['left', 'front', 'right']}})},
    'humans': {'type': 'array', 'maxItems': 20, 'items': object_schema({
        'activity': TEXT,
        'box': BOX})},
    'uncertain_humanoids': {'type': 'array', 'maxItems': 20, 'items': UNCERTAIN_OBJECT},
    'static_obstacles': {'type': 'array', 'maxItems': 30, 'items': UNCERTAIN_OBJECT},
    'final_target_visible': FLAG,
    'final_target_evidence': TEXT})
SCENE_SCHEMA['properties'].update(
    uncertainty=TEXT, instruction_evidence=TEXT,
    regions={'type': 'array', 'maxItems': 20, 'items': object_schema(dict(
        name=TEXT, box={'anyOf': [BOX, {'type': 'null'}]},
        terrain={'enum': ['stairs_up', 'stairs_down', 'floor', 'unknown']}, evidence=TEXT))})


def normalize_scene(scene):
    scene = dict(scene)
    uncertain, static = list(scene['uncertain_humanoids']), []
    pattern = r'mannequin|humanoid|human[- ]shaped|person[- ]shaped'
    for entry in scene['static_obstacles']:
        (uncertain if re.search(pattern, entry['description'], re.I) else static).append(entry)
    descriptions = [scene['description']] + [item['name'] for item in scene['landmarks']]
    if not scene['humans'] and not uncertain and any(re.search(pattern, text, re.I) for text in descriptions):
        uncertain.append(dict(description='Human-shaped object mentioned without a localized box.', box=None))
    return dict(scene, uncertain_humanoids=uncertain, static_obstacles=static)


def image_url(observation):
    buffer = io.BytesIO()
    Image.fromarray(observation.rgb).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def perceive(client, observation, task_context, image_context):
    """Answer one question from one RGB image; all metric work remains local."""
    result = client.complete(
        'visual_perception', 'scene',
        'Describe only the supplied single camera image. image_context labels current versus historical '
        'keyframes; a historical image never describes live geometry or current humans. Report visible landmarks and '
        'people, preserving activities. Human boxes use the declared [left,top,right,bottom] image unit. '
        'Separate definite people in humans, uncertain human-shaped objects in uncertain_humanoids, '
        'and ordinary static objects in static_obstacles. Dark untextured simulated people may '
        'look like mannequins, humanoids or cropped torsos: retain them as uncertain humanoids, '
        'never silently omit them or classify them only as ordinary furniture. Use a visible-part '
        'box when possible, otherwise a null box for uncertain objects. '
        'Do not infer metric distances, hidden rooms, unobserved people or task success. '
        'Focus on current_task and the one question, plus people, temporary obstacles and walkable-area clues. '
        'When stopping_condition is absent, set final_target_visible=false and final_target_evidence empty; '
        'do not list absent final destinations. Mark final_target_visible only with specific visual evidence matching the '
        'instruction stopping condition; a visible landmark alone does not prove arrival. '
        'Include uncertainty, instruction_evidence and localized regions when available. '
        'Report unknown terrain/direction or null boxes when evidence is insufficient; never guess.' + coordinate_prompt(task_context),
        dict(task_context, image_context=image_context),
        coordinate_schema(SCENE_SCHEMA, coordinate_unit(task_context)),
        image_url=image_url(observation), cache=True,
        normalize=lambda value: normalize_coordinates(value, SCENE_SCHEMA), result_schema=SCENE_SCHEMA,
    )
    return normalize_scene(result)


MULTIVIEW_SCHEMA = object_schema(dict(
    views=dict(type='array', minItems=1, maxItems=4, items=object_schema(dict(
        observation_id=dict(type='string'), scene=SCENE_SCHEMA))),
    changes=dict(type='array', maxItems=20, items=object_schema(dict(
        from_observation_id=dict(type='string'), to_observation_id=dict(type='string'), target=TEXT,
        change=dict(enum=['APPEARED', 'DISAPPEARED', 'VIEWPOINT_CHANGE', 'UNCERTAIN']), evidence=TEXT))),
    uncertainty=TEXT))


def perceive_views(client, frames, instruction):
    """One bounded VLM request; image changes remain hypotheses, never motion measurements."""
    from .api_client import APIError
    sources = [source for _, source in frames]
    identifiers = [source['observation_id'] for source in sources]
    def validate(value):
        if [view['observation_id'] for view in value['views']] != identifiers:
            raise APIError('multiview_observation_ids_must_match_images_in_order')
        times = {source['observation_id']: source['captured_at'] for source in sources}
        for change in value['changes']:
            before, after = change['from_observation_id'], change['to_observation_id']
            if before not in times or after not in times or before == after or times[before] > times[after]:
                raise APIError('invalid_multiview_change_sources')
    value = client.complete('visual_perception', 'scene_multiview',
        'Analyze only these at most four historical RGB images in the exact supplied order and IDs. '
        'Give one scene per image, including landmarks, human/uncertain-humanoid boxes, obstacles and '
        'visible regions. Distinguish stair structures, entry, treads and landings with uncertainty. '
        'Describe bounded cross-image changes using actual observation IDs and capture chronology. '
        'All images are historical: never claim current humans/free space, metric motion, 3D coordinates, '
        'route choices or task completion. A changed viewpoint is not proof of robot displacement.' + coordinate_prompt(instruction),
        dict(mode='compare_views',question=instruction['question'],observations=[dict(observation_id=i) for i in identifiers],
             vision_contract=instruction.get('vision_contract',{})),
        coordinate_schema(MULTIVIEW_SCHEMA, coordinate_unit(instruction)),
        image_url=[image_url(observation) for observation, _ in frames], cache=True,
        normalize=lambda value: normalize_coordinates(value, MULTIVIEW_SCHEMA),
        result_schema=MULTIVIEW_SCHEMA, validate=validate)
    return dict(value, views=[dict(view, scene=normalize_scene(view['scene'])) for view in value['views']])
