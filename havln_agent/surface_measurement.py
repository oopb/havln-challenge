"""Robust local depth-height measurement within VLM-proposed image regions.

This is not a navigability, stair-completion, or ground-truth validator.
The observed patch may be a distant landing, an intermediate tread or a wall.
"""
import math
import numpy as np


def measure_surface_region(geometry, box, observation_id):
    if (not isinstance(box,(list,tuple)) or len(box)!=4 or
            any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in box)
            or not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1)):
        raise ValueError('surface_box_requires_normalized_xyxy')
    height,width=geometry.depth.shape
    x0,x1=max(0,int(box[0]*width)),min(width,int(math.ceil(box[2]*width)))
    y0,y1=max(0,int(box[1]*height)),min(height,int(math.ceil(box[3]*height)))
    # Exclude boundaries that commonly contain railings or another surface.
    mx,my=int((x1-x0)*0.08),int((y1-y0)*0.08)
    x0,x1=x0+mx,x1-mx
    y0,y1=y0+my,y1-my
    if x1<=x0 or y1<=y0:
        return dict(status='INSUFFICIENT_DEPTH',observation_id=observation_id,
            reason='empty_region_after_boundary_trim',
            measured_completion=False)
    valid=geometry.valid[y0:y1,x0:x1] & np.isfinite(
        geometry.height[y0:y1,x0:x1])
    fraction=float(valid.mean())
    sample=geometry.height[y0:y1,x0:x1][valid]
    base=dict(observation_id=observation_id,
        box_xyxy=[float(v) for v in box],
        valid_depth_fraction=round(fraction,3),
        reference='robot_local_origin_not_ground_truth_floor',
        surface_identity='VLM_PROPOSED_UNVERIFIED',
        reachability='UNKNOWN',measured_completion=False)
    if sample.size<max(32,int(valid.size*0.05)):
        return dict(base,status='INSUFFICIENT_DEPTH',
            reason='too_few_valid_depth_samples')
    # Identify a dominant near-horizontal height cluster. Ordinary median
    # across a wall-and-floor box would report a meaningless halfway height.
    bins=np.round(sample/0.05).astype(np.int32)
    keys,counts=np.unique(bins,return_counts=True)
    center=float(keys[int(np.argmax(counts))])*0.05
    keep=valid & (np.abs(geometry.height[y0:y1,x0:x1]-center)<=0.09)
    ratio=float(keep.sum()/max(1,valid.sum()))
    if keep.sum()<32 or ratio<0.20:
        return dict(base,status='INSUFFICIENT_DEPTH',
            reason='no_dominant_near_horizontal_height_cluster',
            dominant_height_fraction=round(ratio,3))
    hh=geometry.height[y0:y1,x0:x1][keep]
    xx=geometry.x[y0:y1,x0:x1][keep]
    zz=geometry.z[y0:y1,x0:x1][keep]
    # Fit h=ax+bz+c in centered coordinates; a steep surface does not
    # become reliable merely because its narrow strip had similar heights.
    matrix=np.stack((xx-np.median(xx),zz-np.median(zz),
                     np.ones_like(xx)),axis=1)
    try:
        coeff=np.linalg.lstsq(matrix,hh,rcond=None)[0]
        residual=float(np.median(np.abs(matrix@coeff-hh)))
        slope=float(np.hypot(coeff[0],coeff[1]))
    except (ValueError,np.linalg.LinAlgError):
        return dict(base,status='INSUFFICIENT_DEPTH',reason='plane_fit_failed')
    tilt=math.degrees(math.atan(slope))
    quality=('HIGH' if fraction>=0.7 and ratio>=0.60 and
             residual<0.035 and tilt<12 else
             'MEDIUM' if fraction>=0.4 and ratio>=0.35 and
             residual<0.07 and tilt<20 else 'LOW')
    return dict(base,status='MEASURED',
        relative_height_m_est=round(float(np.median(hh)),3),
        plane_residual_m=round(residual,4),
        plane_tilt_deg=round(tilt,2),
        dominant_height_fraction=round(ratio,3),
        sample_pixels=int(keep.sum()),quality=quality,
        caution='measured_patch_height_not_identity_or_physical_arrival')
