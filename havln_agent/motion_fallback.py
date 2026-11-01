"""Optional independent ORB RGB-D odometry when patch matching is inconclusive.

This module never estimates motion from the requested forward-step length:
all translations originate in matched 3-D feature coordinates. OpenCV is an
optional runtime accelerator; absence means the existing estimator remains
unchanged. Inliers must span multiple image sectors and several depth layers.
"""

import math
import numpy as np


def _feature_xyz(depth, uv, rgb_shape, camera):
    """Backproject one ORB keypoint through a small, stable measured-depth patch."""
    h, w = depth.shape
    rh, rw = rgb_shape[:2]
    x = int(np.clip(round((uv[0] + .5) * w / rw - .5), 0, w - 1))
    y = int(np.clip(round((uv[1] + .5) * h / rh - .5), 0, h - 1))
    if x < 1 or x >= w-1 or y < 1 or y >= h-1:
        return None
    patch = depth[y-1:y+2, x-1:x+2]
    if not np.isfinite(patch).all():
        return None
    patch = patch * (camera.max_depth - camera.min_depth) + camera.min_depth \
        if camera.normalized else patch
    if patch.min() <= max(camera.min_depth, .03) or patch.max() >= camera.max_depth:
        return None
    if np.ptp(patch) > .12:
        return None
    distance = float(np.median(patch))
    fx = camera.width / (2 * np.tan(np.radians(camera.hfov / 2)))
    u = (x + .5) * camera.width / w - .5
    v = (y + .5) * camera.height / h - .5
    return np.array([(u - (camera.width-1)/2) * distance / fx,
                     -(v - (camera.height-1)/2) * distance / fx, distance])


def _orb_metric_consensus(before_xyz, after_xyz, before_uv, pitch_deg, camera):
    """Conservative deterministic translation-only consensus on measured 3-D pairs.

    Does not estimate arbitrary camera rotation or chain poses across epochs.
    A high-confidence zero-motion cluster remains UNKNOWN, not a collision.
    """
    result = dict(status='UNKNOWN', source='orb_mutual_rgbd_static_consensus',
                  position_verified=False, vertical_motion='UNKNOWN',
                  reason='insufficient_orb_rgbd_correspondences')
    if len(before_xyz) < 35:
        return result
    before_xyz = np.asarray(before_xyz, dtype=float)
    after_xyz = np.asarray(after_xyz, dtype=float)
    before_uv = np.asarray(before_uv, dtype=float)
    diff = before_xyz - after_xyz
    rng = np.random.default_rng(22)
    seeds = diff[rng.choice(len(diff), min(len(diff), 150), replace=False)]
    errors = np.linalg.norm(diff[:, None, :] - seeds[None, :, :], axis=2)
    center = seeds[(errors < .065).sum(axis=0).argmax()]
    inliers = np.linalg.norm(diff - center, axis=1) < .065
    if int(inliers.sum()) < 35:
        return result
    translation = np.median(diff[inliers], axis=0)
    inliers = np.linalg.norm(diff - translation, axis=1) < .045
    count = int(inliers.sum())
    ratio = float(inliers.mean())
    if count < 35:
        return result
    p = before_uv[inliers]
    coverage = np.ptp(p, axis=0)
    tiles = len({tuple(np.minimum((v * 3).astype(int),2)) for v in p})
    depth_span = float(np.ptp(before_xyz[inliers, 2]))
    translation = np.median(diff[inliers], axis=0)
    residual = max(.005, float(np.sqrt(np.mean((diff[inliers]-translation)**2))))
    result.update(matches=len(diff), inliers=count, inlier_fraction=ratio,
                  spatial_coverage=list(map(float,coverage)), spatial_tiles=tiles,
                  depth_span_m=depth_span, residual_rms_m=residual)
    if not (count >= 35 and ratio >= .70 and min(coverage) >= .38
            and tiles >= 6 and depth_span >= .6 and residual <= .016):
        result['reason'] = 'insufficient_distributed_orb_metric_consensus'
        return result
    x, y, z = map(float,translation)
    pitch = math.radians(pitch_deg)
    y, z = y*math.cos(pitch)+z*math.sin(pitch), z*math.cos(pitch)-y*math.sin(pitch)
    result['measured_hypothesis_m']=dict(x_right=x,y_up=y,z_forward=z)
    if abs(x) <= .03 and abs(y) <= .035 and abs(z) <= .025:
        result['reason'] = 'no_resolvable_forward_translation'
        return result
    if (abs(x) > camera.forward_step * .4 or not .035 < z < camera.forward_step * 1.6
            or abs(y) > camera.forward_step + camera.max_step_height):
        result['reason'] = 'inconsistent_with_forward_camera_motion'
        return result
    sigma = max(.02, 3 * residual)
    result.update(status='ESTIMATED',reason=None,
                  translation_m=dict(x_right=x,y_up=y,z_forward=z),
                  uncertainty_m=sigma,
                  uncertainty_kind='orb_registration_residual_not_simulator_ground_truth',
                  vertical_motion=('UP' if y > sigma else 'DOWN' if y < -sigma else 'LEVEL'))
    return result


def estimate_orb_translation(before, after, camera, excluded_before=None):
    """Return an independent candidate, or None if ORB unavailable/insufficient."""
    try:
        import cv2
    except (ImportError, OSError):
        return None
    if after.captured_at < before.captured_at:
        return None
    try:
        gray_a = cv2.cvtColor(before.rgb, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(after.rgb, cv2.COLOR_RGB2GRAY)
        mask = None
        if excluded_before is not None:
            exclusion = cv2.resize(np.asarray(excluded_before,dtype='uint8'),
                                   (gray_a.shape[1],gray_a.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            mask = np.where(exclusion != 0, 0, 255).astype('uint8')
        detector = cv2.ORB_create(nfeatures=1200,scaleFactor=1.15,
                                  nlevels=8,edgeThreshold=12,fastThreshold=8)
        kpa, a = detector.detectAndCompute(gray_a,mask)
        kpb, b = detector.detectAndCompute(gray_b,None)
        if a is None or b is None or len(a) < 35 or len(b) < 35:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(a,b,k=2)
        reverse = {m.queryIdx:m.trainIdx for m in matcher.match(b,a)}
        matched = []
        for pair in pairs:
            if len(pair) != 2:
                continue
            first, second = pair
            if (first.distance > 50 or first.distance > .68 * second.distance
                    or reverse.get(first.trainIdx) != first.queryIdx):
                continue
            xyz_a = _feature_xyz(before.depth,kpa[first.queryIdx].pt,
                                 before.rgb.shape,camera)
            xyz_b = _feature_xyz(after.depth,kpb[first.trainIdx].pt,
                                 after.rgb.shape,camera)
            if xyz_a is None or xyz_b is None:
                continue
            pt = kpa[first.queryIdx].pt
            matched.append((xyz_a,xyz_b,
                            (pt[0]/before.rgb.shape[1],pt[1]/before.rgb.shape[0])))
        if len(matched) < 35:
            return None
        a3 = np.asarray([m[0] for m in matched])
        b3 = np.asarray([m[1] for m in matched])
        uv = np.asarray([m[2] for m in matched])
        return _orb_metric_consensus(a3,b3,uv,before.pitch_deg,camera)
    except (cv2.error, ValueError, TypeError):
        # This optional sensor estimator must not abort an otherwise legal
        # Habitat action; the primary conservative estimate remains UNKNOWN.
        return None
