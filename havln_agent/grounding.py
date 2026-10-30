"""Semantic boxes are hypotheses; all metric values come from measured depth."""

import math

import numpy as np

from .api_client import FLAG, TEXT, object_schema
from .perception import BOX, image_url
from .motion import landing_support
from .vision_protocol import coordinate_prompt, coordinate_schema, coordinate_unit, normalize_coordinates


GROUND_SCHEMA = object_schema(
    dict(
        target_visible=FLAG,
        box={'anyOf': [BOX, {'type': 'null'}]},
        entry_box={'anyOf': [BOX, {'type': 'null'}]},
        stair_direction={'enum': ['up', 'down', 'unknown', 'not_stairs']},
        lower_landing_visible=FLAG,
        evidence=TEXT,
        uncertainty=TEXT,
        confidence={'enum': ['low', 'medium', 'high']},
    )
)
GROUND_SCHEMA['properties']['landing_box'] = {'anyOf': [BOX, {'type': 'null'}]}
GROUND_SCHEMA['properties']['landing_visible'] = FLAG
GROUND_SCHEMA['properties'].update(
    components=dict(type='array', maxItems=20, items=object_schema(dict(
        kind=dict(enum=['stairs', 'handrail', 'tread', 'upper_landing', 'lower_landing',
                        'corridor', 'door', 'person', 'landmark']),
        visibility=dict(enum=['VISIBLE', 'POSSIBLE', 'UNKNOWN']),
        box={'anyOf': [BOX, {'type': 'null'}]}, evidence=TEXT))),
    entry_visible=FLAG, entry_evidence=TEXT, direction_evidence=TEXT,
    upper_landing_box={'anyOf': [BOX, {'type': 'null'}]},
    lower_landing_box={'anyOf': [BOX, {'type': 'null'}]},
    tread_boxes=dict(type='array', maxItems=6, items=BOX))


def project_box(geometry, box):
    if box is None or not (box[0] < box[2] and box[1] < box[3]):
        return dict(status='UNKNOWN', reason='no_valid_image_region')
    cam = geometry.camera
    x0, x1 = int(box[0] * cam.width), min(cam.width, int(math.ceil(box[2] * cam.width)))
    y0, y1 = int(box[1] * cam.height), min(cam.height, int(math.ceil(box[3] * cam.height)))
    valid = geometry.valid[y0:y1, x0:x1]
    if not valid.size or float(valid.mean()) < 0.5 or int(valid.sum()) < 8:
        return dict(status='UNKNOWN', reason='insufficient_region_depth')
    z = geometry.z[y0:y1, x0:x1][valid]
    x = geometry.x[y0:y1, x0:x1][valid]
    h = geometry.height[y0:y1, x0:x1][valid]
    dx, dz = float(np.median(x)), float(np.median(z))
    # Visible-point height profile, not a traversable floor or actual agent motion.
    bins = np.linspace(float(np.quantile(z, 0.1)), float(np.quantile(z, 0.9)), 6)
    profile = []
    for lo, hi in zip(bins, bins[1:]):
        selected = (z >= lo) & (z < hi)
        if int(selected.sum()) >= 8:
            profile.append([float(np.median(z[selected])), float(np.quantile(h[selected], 0.2))])
    slope = None
    if len(profile) >= 3 and profile[-1][0] - profile[0][0] >= 0.4:
        slope = float(np.polyfit(np.array(profile)[:, 0], np.array(profile)[:, 1], 1)[0])
    spread = float(np.quantile(z, 0.9) - np.quantile(z, 0.1))
    return dict(
        status='LOCALIZED' if spread < 0.6 else 'PARTIAL',
        frame='current_robot_camera',
        x_right=dx,
        z_forward=dz,
        bearing_degrees=math.degrees(math.atan2(-dx, dz)),
        valid_depth_fraction=float(valid.mean()),
        depth_spread_metres=spread,
        y_up=float(np.median(h) - cam.camera_height),
        height_spread_metres=float(np.quantile(h, 0.9) - np.quantile(h, 0.1)),
        uncertainty_m=max(0.05, spread / 2),
        source='rgb_region_and_measured_depth',
        visible_height_profile=profile,
        height_slope=slope,
        terrain_support='not_certified',
        limitation='Box may include occluders; a visible point is not a reachable navigation goal.',
    )


def ground_layers(result, geometry, position, entry):
    """Fuse model image regions with measured depth; never infer access from presence."""
    components = result.get('components', [])
    presence = 'VISIBLE' if result['target_visible'] else 'NOT_VISIBLE'
    if result['confidence'] == 'low' and result['target_visible']:
        presence = 'POSSIBLE'
    stairs = result['stair_direction'] != 'not_stairs'
    if stairs and components and not any(c['kind'] == 'stairs' and c['visibility'] == 'VISIBLE' for c in components):
        presence = 'POSSIBLE' if result['target_visible'] else presence
        entry = dict(status='UNKNOWN', reason='stair_entry_semantics_unconfirmed')
    entry_floor = (landing_support(geometry, result.get('entry_box'))
                   if stairs and result.get('entry_box') else dict(status='UNKNOWN', terrain_certified=False))
    if stairs and entry['status'] == 'LOCALIZED':
        height = entry.get('y_up', 0) + geometry.camera.camera_height
        if abs(height) > 0.25:
            entry = dict(entry, point_status='LOCALIZED', status='UNKNOWN',
                         reason='entry_region_not_at_standing_floor_height')
    if stairs:
        entry = dict(entry, floor_support=entry_floor)
    treads = [project_box(geometry, box) for box in result.get('tread_boxes', [])]
    reliable = sorted((p for p in treads if p['status'] == 'LOCALIZED'), key=lambda p: p['z_forward'])
    direction, reasons = 'UNKNOWN', []
    if len(reliable) >= 2 and presence == 'VISIBLE' and result.get('direction_evidence') and result['confidence'] != 'low':
        first, last = reliable[0], reliable[-1]
        dz, dy = last['z_forward'] - first['z_forward'], last['y_up'] - first['y_up']
        uncertainty = max(0.05, (first['height_spread_metres'] + last['height_spread_metres']) / 2)
        if dz >= 0.15 and abs(dy) > uncertainty:
            measured = 'DOWN' if dy < 0 else 'UP'
            if measured.lower() == result['stair_direction']:
                direction = measured
            else:
                reasons.append('visual_direction_conflicts_with_tread_depth')
    if direction == 'UNKNOWN' and stairs:
        reasons.append('insufficient_consistent_tread_direction_evidence')
    phases = (['SEE_STAIRS'] if stairs and presence == 'VISIBLE' else [])
    if stairs and entry['status'] == 'LOCALIZED':
        phases.append('FIND_STAIR_ENTRY')
    if direction == 'DOWN':
        phases.append('DESCENDING_STAIR_DIRECTION')
    access_supported, access_missing = [], ['terrain_traversability_not_certified']
    if stairs and entry_floor['status'] != 'SUPPORTED':
        access_missing.append('entry_floor_support_unverified')
    approach = dict(status='UNKNOWN')
    if entry['status'] == 'LOCALIZED':
        access_supported.append('depth_localized_entry')
        diagnostic = geometry.diagnose(math.radians(entry['bearing_degrees']), max(1.25, entry['z_forward']))
        approach = dict(status=diagnostic['path_state'], accepted=diagnostic['accepted'],
                        bearing_deg=entry['bearing_degrees'], clearance_m=diagnostic['clearance'])
        if diagnostic['accepted']:
            access_supported.append('geometrically_clear_approach_bearing')
        else:
            access_missing.extend(diagnostic['rejection_reasons'])
    else:
        access_missing.append('reliable_entry_region')
    if stairs and direction == 'UNKNOWN':
        access_missing.append('confirmed_stair_direction')
    return dict(semantic_presence=presence,
                localization='LOCALIZED' if position['status'] == 'LOCALIZED' else
                'PARTIAL' if position['status'] == 'PARTIAL' or entry['status'] == 'LOCALIZED' else 'UNKNOWN',
                entry_status=entry['status'], stair_direction=direction if stairs else 'NOT_STAIRS',
                traversable_access='APPROACH_ONLY' if approach.get('accepted') else 'UNVERIFIED',
                access_evidence=dict(supported=access_supported, missing=access_missing,
                                     approach_geometry=approach, terrain_certified=False),
                stair_evidence=dict(confirmed_phases=phases, direction_hypothesis=result['stair_direction'],
                                    measured_treads=treads, direction_rejection_reasons=reasons,
                                    measured_landings={key: project_box(geometry, result.get(key + '_box'))
                                                       for key in ('upper_landing', 'lower_landing')}),
                entry=entry)


def ground(client, observation, geometry, target, instruction, source, timeout=None):
    result = client.complete(
        'visual_perception',
        'grounding',
        'Locate only the requested landmark in the supplied image. Source marks current or historical. '
        'Distinguish stairs, handrails, treads, upper/lower landings, corridor, door and people as components. '
        'A handrail or lower room is not an accessible stair entry. Return a visible-part box in the declared image unit '
        '[left,top,right,bottom] and an entry/tread region if visible. Do not infer hidden targets, '
        'metric distances, 3D coordinates or task completion. Stairs up/down is a visual hypothesis, '
        'not actual motion. Use null boxes, low confidence and unknown when uncertain. '
        'Preserve uncertainty around dark simulated human occluders. '
        'Include landing_box for a visible flat landing/floor region, otherwise null, '
        'and landing_visible for a visibly supported upper or lower landing. '
        'Use entry_visible, entry_evidence, direction_evidence, tread_boxes (at most six visible treads), '
        'upper_landing_box and lower_landing_box. Leave direction unknown without unambiguous cues. '
        'A landing visible ahead is not proof the robot has reached it. Never claim task completion.' + coordinate_prompt(instruction),
        dict(target=target, instruction=instruction, image_context=dict(
            scope=source.get('image_scope', 'current'), observation_id=source.get('observation_id'),
            mode='grounding')),
        coordinate_schema(GROUND_SCHEMA, coordinate_unit(instruction)),
        normalize=lambda value: normalize_coordinates(value, GROUND_SCHEMA), result_schema=GROUND_SCHEMA,
        image_url=image_url(observation),
        cache=True,
        timeout_seconds=timeout,
    )
    return project_grounding(result,geometry,target,source)


def project_grounding(result,geometry,target,source):
    position = (
        project_box(geometry, result['box'])
        if result['target_visible']
        else dict(status='UNKNOWN', reason='target_not_visible')
    )
    if result['confidence'] == 'low':
        position = dict(status='UNKNOWN', reason='semantic_localization_uncertain')
    entry = (project_box(geometry, result['entry_box']) if result['target_visible'] and
             result.get('entry_visible', result['entry_box'] is not None) else
             dict(status='UNKNOWN', reason='target_entry_not_visible'))
    landing = (landing_support(geometry, result['landing_box']) if result.get('landing_box') else
               dict(status='UNKNOWN', reason='landing_region_not_observed', terrain_certified=False))
    if result['confidence'] == 'low':
        entry = dict(status='UNKNOWN', reason='semantic_localization_uncertain')
        landing = dict(status='UNKNOWN', reason='semantic_localization_uncertain', terrain_certified=False)
    layers = ground_layers(result, geometry, position, entry)
    entry = layers.pop('entry')
    return dict(
        target=target,
        source=dict(source),
        semantic=result,
        position=position,
        entry=entry,
        status=('LOCALIZED' if entry['status'] == 'LOCALIZED' else 'PARTIAL'
                if position['status'] in ('LOCALIZED', 'PARTIAL') or entry['status'] == 'PARTIAL'
                or landing['status'] == 'SUPPORTED' else 'UNKNOWN')
        if result['stair_direction'] != 'not_stairs' else position['status'],
        navigation_authorized=False,
        vertical_motion_evidence='UNKNOWN',
        landing_geometry=landing,
        region_evidence=dict(target_point=position['status'], entry_point=entry['status'],
                             landing_plane=landing['status']),
        **layers,
    )
