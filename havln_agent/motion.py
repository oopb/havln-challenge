"""Conservative translation evidence from public, consecutive RGB-D frames.

Only MOVE_FORWARD has fixed camera orientation. Mutual textured patch matches
must agree in measured 3D across the image; ambiguity returns UNKNOWN. This is
an estimate, never simulator pose or a substitute for the live Safety Shield.
"""

import numpy as np
from PIL import Image
from .observation import array_hash


def registered_action(action, observations):
    result = action['result']
    measurement = result.get('rgbd_motion') or {}
    pair = measurement.get('observation_pair', {})
    before, after = observations.get(result.get('observation_before')), observations.get(result.get('observation_after'))
    return bool(measurement.get('status') == 'ESTIMATED' and before and after and all(
        pair.get(prefix + '_' + kind + '_hash') == record[kind + '_hash']
        for prefix, record in (('before', before), ('after', after)) for kind in ('rgb', 'depth')))


def _features(observation, camera, excluded=None):
    width = min(96, observation.rgb.shape[1])
    height = round(width * observation.rgb.shape[0] / observation.rgb.shape[1])
    rgb = np.asarray(Image.fromarray(observation.rgb).resize((width, height)), dtype=float) / 255
    gray = rgb.mean(axis=2)
    gy, gx = np.gradient(gray)
    score = gx ** 2 + gy ** 2
    depth = observation.depth * (camera.max_depth - camera.min_depth) + camera.min_depth \
        if camera.normalized else observation.depth
    sampled = np.asarray(Image.fromarray(depth.astype('float32')).resize((width, height), Image.NEAREST))
    neighborhoods = np.stack([np.roll(np.roll(sampled, y, axis=0), x, axis=1)
                              for y in (-1, 0, 1) for x in (-1, 0, 1)])
    stable_depth = (np.isfinite(neighborhoods).all(axis=0) & (neighborhoods.min(axis=0) > camera.min_depth)
                    & (neighborhoods.max(axis=0) < camera.max_depth)
                    & (np.ptp(neighborhoods, axis=0) <= 0.12))
    score[~stable_depth] = 0
    if excluded is not None:
        mask = np.asarray(Image.fromarray(excluded.astype('uint8')).resize((width, height), Image.NEAREST))
        score[mask != 0] = 0
    points = []
    for top in range(3, height - 3, 8):
        for left in range(3, width - 3, 8):
            region = score[top:min(top + 8, height - 3), left:min(left + 8, width - 3)]
            for index in np.argsort(region, axis=None)[-2:]:
                y, x = np.unravel_index(index, region.shape)
                y, x = y + top, x + left
                if score[y, x] > 0.002:
                    points.append((x, y))
    descriptors, xyz, pixels = [], [], []
    fx = camera.width / (2 * np.tan(np.radians(camera.hfov / 2)))
    for x, y in points:
        u = (x + 0.5) * camera.width / width - 0.5
        v = (y + 0.5) * camera.height / height - 0.5
        column, row = round(u), round(v)
        patch = depth[max(0, row - 1):row + 2, max(0, column - 1):column + 2]
        if not np.isfinite(patch).all() or patch.min() <= camera.min_depth or patch.max() >= camera.max_depth:
            continue
        if np.ptp(patch) > 0.12:
            continue
        z = float(np.median(patch))
        descriptor = rgb[y - 2:y + 3, x - 2:x + 3].ravel()
        descriptor = descriptor - descriptor.mean()
        norm = np.linalg.norm(descriptor)
        if norm < 0.25:
            continue
        descriptors.append(descriptor / norm)
        xyz.append([(u - (camera.width - 1) / 2) * z / fx,
                    -(v - (camera.height - 1) / 2) * z / fx, z])
        pixels.append([x / width, y / height])
    return np.asarray(descriptors), np.asarray(xyz), np.asarray(pixels)


def estimate_translation(before, after, camera, excluded_before=None):
    result = dict(status='UNKNOWN', source='mutual_rgb_patches_and_measured_depth',
                  frame='previous_robot_camera', position_verified=False,
                  vertical_motion='UNKNOWN', reason='insufficient_static_correspondences')
    result['observation_pair'] = dict(before_rgb_hash=array_hash(before.rgb), after_rgb_hash=array_hash(after.rgb),
        before_depth_hash=array_hash(before.depth), after_depth_hash=array_hash(after.depth))
    if after.captured_at < before.captured_at:
        result['reason'] = 'non_chronological_rgbd_pair'
        return result
    a, points_a, pixels = _features(before, camera, excluded_before)
    b, points_b, _ = _features(after, camera)
    if len(a) < 12 or len(b) < 12:
        return result
    distances = np.maximum(0, 2 - 2 * a @ b.T)
    order = np.argsort(distances, axis=1)[:, :2]
    reverse = distances.argmin(axis=0)
    indices = np.arange(len(a))
    best, second = distances[indices, order[:, 0]], distances[indices, order[:, 1]]
    good = (best < 0.35) & (best < second * 0.7) & (reverse[order[:, 0]] == indices)
    differences = points_a[good] - points_b[order[good, 0]]
    matched_pixels = pixels[good]
    result['matches'] = len(differences)
    if len(differences) < 12:
        return result
    # Deterministic translation consensus, with no commanded displacement as evidence.
    errors = np.linalg.norm(differences[:, None] - differences[None, :], axis=2)
    inliers = errors[(errors < 0.05).sum(axis=1).argmax()] < 0.05
    translation = np.median(differences[inliers], axis=0)
    residual = np.linalg.norm(differences - translation, axis=1)
    inliers = residual < 0.04
    count, ratio = int(inliers.sum()), float(inliers.mean())
    result.update(inliers=count, inlier_fraction=ratio)
    if not count:
        result['reason'] = 'inconsistent_or_dynamic_correspondences'
        return result
    coverage = np.ptp(matched_pixels[inliers], axis=0)
    tiles = {tuple((p * 3).astype(int)) for p in matched_pixels[inliers]}
    depth_span = float(np.ptp(points_a[good][inliers, 2]))
    translation = np.median(differences[inliers], axis=0)
    sigma = max(0.005, float(np.sqrt(np.mean((differences[inliers] - translation) ** 2))))
    # The conventional 65% match-consensus threshold is useful when features
    # are sparse, but can reject a geometrically coherent STATIC background
    # when a moving humanoid contributes a minority of the matches. A relaxed
    # branch is allowed ONLY with strong independent coverage, depth diversity
    # and low metric residual. It never relies on the commanded step length.
    strong_consensus = count >= 12 and ratio >= 0.65
    distributed_consensus = (count >= 20 and ratio >= 0.52
        and float(coverage.min()) >= 0.65 and len(tiles) >= 7
        and depth_span >= 0.6 and sigma <= 0.02)
    result.update(consensus_mode=('strict' if strong_consensus else
                                  'distributed_static' if distributed_consensus else 'rejected'),
        spatial_tiles=len(tiles), spatial_coverage=list(map(float, coverage)),
        depth_span_m=depth_span, residual_rms_m=sigma)
    if not (strong_consensus or distributed_consensus):
        result['reason'] = 'inconsistent_or_dynamic_correspondences'
        return result
    if coverage.min() < 0.4 or len(tiles) < 6 or depth_span < 0.3:
        result['reason'] = 'insufficient_spatial_or_depth_coverage'
        return result
    x, y, z = map(float, translation)
    pitch = np.radians(before.pitch_deg)
    y, z = y*np.cos(pitch)+z*np.sin(pitch), z*np.cos(pitch)-y*np.sin(pitch)
    if abs(x) > camera.forward_step * 0.4 or not 0.02 < z < camera.forward_step * 1.6 \
            or abs(y) > camera.forward_step+camera.max_step_height:
        result['reason'] = 'inconsistent_with_forward_camera_motion'
        return result
    result.update(status='ESTIMATED', reason=None, translation_m=dict(x_right=x, y_up=y, z_forward=z),
                  uncertainty_m=3 * sigma, spatial_tiles=len(tiles),
                  uncertainty_kind='registration_residual_not_calibrated_ground_truth',
                  vertical_motion='UP' if y > 3 * sigma else 'DOWN' if y < -3 * sigma else 'LEVEL')
    return result


def landing_support(geometry, box=None):
    """A measured local level surface, separate from body collision permission."""
    from .terrain import review
    if box is None:
        proof=review(geometry)
        if proof['status']=='SUPPORTED' and proof['direction']=='LEVEL':return proof
        return dict(status='UNKNOWN',source='current_depth_local_support',terrain_certified=False,
            reason=proof.get('reason') or 'local_surface_contains_steps')
    x0,y0,x1,y1=box
    region=np.zeros_like(geometry.valid)
    region[int(y0*geometry.camera.height):int(np.ceil(y1*geometry.camera.height)),
        int(x0*geometry.camera.width):int(np.ceil(x1*geometry.camera.width))]=True
    for plane in geometry.support_planes:
        if (plane['mask'] & region).sum()>=32 and abs(plane['height_m'])<=geometry.camera.max_step_height:
            return dict(status='SUPPORTED',source='measured_local_level_patch',terrain_certified=False,
                height_offset_m=plane['height_m'],plane_slope=float(np.linalg.norm(plane['fit'][:2])),
                residual_p90_m=plane['residual_p90_m'],scope='visible_patch_not_motion_authorization')
    return dict(status='UNKNOWN',source='measured_local_level_patch',terrain_certified=False,
        reason='local_level_patch_not_measured')
