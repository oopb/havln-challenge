"""Conservative, public RGB-D verification of a commanded forward action.

Never gates movement before it occurs. A false stall is more damaging than an
unknown result, so only well-supported *small* motion interrupts a batch.
No simulator pose, collision flag, reference path, or external model is read.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ForwardMotion:
    state: str  # STALLED, PROGRESSED, or UNCERTAIN
    reason: str
    estimated_m: Optional[float] = None
    matched_regions: int = 0


def _depth_m(raw, camera, size):
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim == 3:
        values = values[..., 0]
    if camera.normalized:
        valid = np.isfinite(values) & (values > 0) & (values <= 1)
        values = values * (camera.max_depth - camera.min_depth) + camera.min_depth
    else:
        valid = np.isfinite(values)
    valid &= (values > camera.min_depth) & (values < camera.max_depth)
    depth = np.where(valid, values, 0).astype(np.float32)
    # Depth is a different sensor resolution in some Habitat configurations.
    depth = np.asarray(Image.fromarray(depth, mode="F").resize(size, Image.Resampling.NEAREST))
    return depth


def assess_forward_motion(before, after, camera):
    """Estimate z-translation by matching textured static RGB patches in RGB-D.

    The known atomic command is straight forward (not a camera turn). We track
    the same visible patches and measure their *matched* Depth changes, not
    same-pixel changes; side walls alone cannot falsely imply a stall. When
    textures, Depth or agreement are insufficient, return UNCERTAIN and let the
    model keep control. Typical runtime is bounded by the fixed 96px grid.
    """
    if np.array_equal(before.rgb, after.rgb):
        return ForwardMotion("STALLED", "rgb_unchanged", 0.0)
    if before.pitch_deg != after.pitch_deg:
        return ForwardMotion("UNCERTAIN", "camera_pitch_changed")
    size = (96, 96)
    gray0 = np.asarray(Image.fromarray(before.rgb).convert("L").resize(size,
                       Image.Resampling.BILINEAR), dtype=np.float32)
    gray1 = np.asarray(Image.fromarray(after.rgb).convert("L").resize(size,
                       Image.Resampling.BILINEAR), dtype=np.float32)
    d0 = _depth_m(before.depth, camera, size)
    d1 = _depth_m(after.depth, camera, size)
    radius, patch = 9, 2
    estimates = []
    sectors = set()
    # Avoid the extreme periphery, dominated by occlusions/perspective flow.
    for y in range(15, 82, 9):
        for x in range(15, 82, 9):
            z0 = float(d0[y, x])
            if not 0.3 < z0 < min(5., camera.max_depth - 1e-3):
                continue
            source = gray0[y-patch:y+patch+1, x-patch:x+patch+1]
            if float(source.std()) < 9.0:
                continue
            # All candidate locations in one vectorized window.
            crop = gray1[y-radius-patch:y+radius+patch+1,
                         x-radius-patch:x+radius+patch+1]
            candidates = np.lib.stride_tricks.sliding_window_view(crop,
                                                               source.shape)
            costs = np.mean((candidates - source) ** 2, axis=(-2, -1))
            flat = costs.ravel()
            best = int(np.argmin(flat))
            best_cost = float(flat[best])
            if best_cost > 170.0:
                continue
            # Reject many ambiguous/repeated patterns, e.g. tiled floors.
            cy, cx = np.unravel_index(best, costs.shape)
            others = costs.copy()
            others[max(0, cy-2):cy+3, max(0, cx-2):cx+3] = np.inf
            if best_cost > 0.70 * float(others.min()):
                continue
            xx, yy = x + cx - radius, y + cy - radius
            z1 = float(d1[yy, xx])
            if not 0.2 < z1 < camera.max_depth:
                continue
            delta = z0 - z1
            if abs(delta) > camera.forward_step * 2.0 + 0.15:
                continue
            estimates.append(delta)
            sectors.add((int(x >= 48), int(y >= 48)))
    if len(estimates) < 9 or len(sectors) < 3:
        return ForwardMotion("UNCERTAIN", "insufficient_static_rgbd_matches",
                             matched_regions=len(estimates))
    samples = np.asarray(estimates)
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    if mad > max(0.055, camera.forward_step * 0.30):
        return ForwardMotion("UNCERTAIN", "inconsistent_scene_motion",
                             matched_regions=len(estimates))
    # A deliberately high-confidence small-motion test, never a CLEAR gate.
    if abs(median) < min(0.045, camera.forward_step * 0.18) and (
            np.mean(np.abs(samples) < 0.09) >= 0.7):
        return ForwardMotion("STALLED", "rgbd_near_zero_translation",
                             round(median, 3), len(estimates))
    if median >= camera.forward_step * 0.42:
        return ForwardMotion("PROGRESSED", "rgbd_forward_translation",
                             round(median, 3), len(estimates))
    return ForwardMotion("UNCERTAIN", "ambiguous_forward_translation",
                         round(median, 3), len(estimates))
