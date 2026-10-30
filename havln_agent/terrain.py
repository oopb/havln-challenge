"""Measured local support for one body sweep; semantic stair labels grant no motion."""
import math
import numpy as np


def measured_planes(geometry):
    """Separate horizontal support patches from risers/walls using RGB-D normals."""
    g=geometry
    points=np.stack((g.x,g.height,g.z),axis=-1)
    dx=points[:,2:]-points[:,:-2]
    dy=points[2:]-points[:-2]
    normals=np.cross(dx[1:-1],dy[:,1:-1])
    length=np.linalg.norm(normals,axis=-1)
    flat=np.zeros_like(g.valid)
    flat[1:-1,1:-1]=(abs(normals[...,1])>=.98*np.maximum(length,1e-12))&(length>1e-9)
    flat &= g.valid & ~g.human_mask & (g.z>-.25) & (g.z<1.5) & (abs(g.height)<1.)
    # Neighbouring horizontal patches at different heights are fitted separately.
    levels=np.round(np.where(flat,g.height,0)/.02).astype(int)
    bins,counts=np.unique(levels[flat],return_counts=True)
    peaks=sorted(zip(bins,counts),key=lambda item:-item[1])
    planes=[]
    used=np.zeros_like(flat)
    for level,count in peaks:
        if count<24:continue
        mask=flat & ~used & (abs(g.height-level*.02)<=.035)
        if mask.sum()<32:continue
        x,z,h=g.x[mask],g.z[mask],g.height[mask]
        design=np.column_stack((x,z,np.ones_like(x)))
        fit,_,rank,_=np.linalg.lstsq(design,h,rcond=None)
        residual=abs(design@fit-h)
        if rank<3 or np.linalg.norm(fit[:2])>.12 or np.quantile(residual,.9)>.04:continue
        # Keep only pixels explained by this plane; never erase the rest of a box.
        support=g.valid & ~g.human_mask & (abs(g.height-(fit[0]*g.x+fit[1]*g.z+fit[2]))<=.035)
        support &= (g.x>=x.min()-.02)&(g.x<=x.max()+.02)&(g.z>=z.min()-.02)&(g.z<=z.max()+.02)
        used |= mask
        planes.append(dict(height_m=float(np.median(h)),fit=[float(v) for v in fit],
            x_min=float(x.min()),x_max=float(x.max()),z_min=float(z.min()),z_max=float(z.max()),
            residual_p90_m=float(np.quantile(residual,.9)),points=int(mask.sum()),mask=support))
    return planes


def review(geometry, angle=0.):
    g=geometry
    planes=g.support_planes
    radius=g.camera.robot_radius+g.margin
    step=g.camera.forward_step
    along=-g.x*math.sin(angle)+g.z*math.cos(angle)
    across=g.x*math.cos(angle)+g.z*math.sin(angle)
    support=np.zeros_like(g.valid)
    for plane in planes:support |= plane['mask']
    proof=dict(status='UNKNOWN',source='current_depth_local_support',terrain_certified=False,
        scope='next_atomic_step_only',reason='next_step_support_not_observed',
        planes=[{k:v for k,v in p.items() if k!='mask'} for p in planes],
        samples=[],angle_rad=float(angle),max_step_height_m=g.camera.max_step_height)
    # Start beyond the current physical footprint. Do not demand a 1-2 m route.
    stations=np.arange(g.camera.robot_radius+.025,step+radius+.001,.05)
    anchors=[p['height_m'] for p in planes if np.any(p['mask'] & (np.hypot(g.x,g.z)<=g.camera.robot_radius+.035))
        and abs(p['height_m'])<=g.camera.max_step_height]
    previous=min(anchors,key=abs) if anchors else 0.
    proof['initial_height_m']=float(previous)
    for station in stations:
        lateral=math.sqrt(max(0,radius**2-max(0,station-step)**2))
        region=(abs(along-station)<.035)&(abs(across)<=lateral+.025)&support
        heights=g.height[region]
        sample=dict(along_m=round(float(station),3),required_half_width_m=round(lateral,3),points=int(region.sum()))
        if heights.size<12:
            proof['samples'].append(sample)
            return proof
        height=float(np.median(heights))
        spread=float(np.quantile(heights,.98)-np.quantile(heights,.02))
        if spread>g.camera.max_step_height+.01:
            proof.update(status='BLOCKED',reason='measured_footprint_spans_dangerous_height_change')
            proof['samples'].append(dict(sample,height_spread_m=spread))
            return proof
        x=np.sort(across[region])
        sample.update(height_m=height,observed_across_m=[float(x.min()),float(x.max())])
        proof['samples'].append(sample)
        if x.min()>-lateral+.025 or x.max()<lateral-.025 or (len(x)>1 and np.diff(x).max()>.04):
            return proof
        if abs(height-previous)>g.camera.max_step_height+.01:
            proof.update(status='BLOCKED',reason='measured_step_or_drop_exceeds_collision_model')
            return proof
        previous=height
    proof.update(status='SUPPORTED',reason=None,direction='DOWN' if previous-proof['initial_height_m']<-.06 else 'UP' if previous-proof['initial_height_m']>.06 else 'LEVEL',
        end_height_m=previous)
    return proof


def obstacle_mask(geometry, proof=None):
    """Remove only depth-measured support and small risers between measured patches."""
    g=geometry
    heights=[s['height_m'] for s in (proof or {}).get('samples',[]) if 'height_m' in s]
    lo=min([0.]+heights);hi=max([0.]+heights)
    obstacles=g.valid & (g.height>lo+.08) & (g.height<hi+g.camera.robot_height)
    for plane in g.support_planes:
        # An elevated table or a distant lower floor cannot become standing ground.
        if abs(plane['height_m'])>g.camera.max_step_height+.01:continue
        region=(g.x>=plane['x_min'])&(g.x<=plane['x_max'])&(g.z>=plane['z_min'])&(g.z<=plane['z_max'])
        floor=plane['fit'][0]*g.x+plane['fit'][1]*g.z+plane['fit'][2]
        obstacles &= ~(region & ~g.human_mask & (g.height<=floor+.04)&(g.height>=floor-.04))
    if proof and proof['status']=='SUPPORTED':
        # The verified local footprint may span a single small riser. Walls and
        # people above it remain obstacles; no far support extends authorization.
        angle=proof.get('angle_rad',0.)
        along=-g.x*math.sin(angle)+g.z*math.cos(angle)
        across=g.x*math.cos(angle)+g.z*math.sin(angle)
        for sample in proof['samples']:
            region=(abs(along-sample['along_m'])<=.055)&(abs(across)<=sample['required_half_width_m'])
            obstacles &= ~(region & ~g.human_mask & (g.height<=sample['height_m']+.04)
                & (g.height>=min(0.,sample['height_m'])-.04))
    return obstacles
