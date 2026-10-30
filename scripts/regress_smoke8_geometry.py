"""Read saved public RGB-D and compare frozen/current geometry. No API or simulator."""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from havln_agent.perception import Camera,Observation,Geometry
from havln_agent import terrain
from havln_agent.observation import array_hash


def load_baseline(path):
    spec=importlib.util.spec_from_file_location('smoke8_baseline',path/'havln_agent/__init__.py',
        submodule_search_locations=[str(path/'havln_agent')])
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    from smoke8_baseline.perception import Camera as OldCamera,Observation as OldObservation,Geometry as OldGeometry
    from smoke8_baseline.local_motion_controller import LocalMotionController as OldController
    return OldCamera,OldObservation,OldGeometry,OldController


def collision_points(g,angle,mask):
    radius=g.camera.robot_radius+g.margin
    along=-g.x*math.sin(angle)+g.z*math.cos(angle)
    across=g.x*math.cos(angle)+g.z*math.sin(angle)
    selected=mask&(abs(across)<radius)&(along>0)
    contact=along-np.sqrt(np.maximum(0,radius**2-across**2))
    indices=np.flatnonzero(selected)
    indices=indices[np.argsort(contact.ravel()[indices])[:3]]
    rows=[]
    for index in indices:
        y,x=np.unravel_index(index,mask.shape)
        rows.append(dict(row=int(y),column=int(x),x_m=float(g.x[y,x]),z_m=float(g.z[y,x]),
            height_m=float(g.height[y,x]),depth_m=float(g.depth[y,x]),contact_travel_m=float(contact[y,x])))
    return rows


def regress(saved_run,baseline,output):
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh regression output directory')
    output.mkdir(parents=True,exist_ok=True)
    camera_values=json.loads(next(saved_run.glob('camera-*.json')).read_text())
    camera=Camera(**camera_values)
    OldCamera,OldObservation,OldGeometry,OldController=load_baseline(baseline)
    old_camera=OldCamera(**camera_values)
    trace=next((saved_run/'traces').iterdir())
    memory=json.loads((trace/'memory.json').read_text())
    reports=[]
    before_counts,after_counts=Counter(),Counter()
    for record in memory['observations']:
        with np.load(trace/record['input_file']) as frame:
            rgb,depth,pitch=frame['rgb'],frame['depth'],float(frame['pitch_deg'])
        scene=next((v['scene'] for v in reversed(memory['scene_history'])
            if v['source']['rgb_hash']==record['rgb_hash'] and v['source']['pose']==record['pose']),None)
        old=OldGeometry(OldObservation(rgb,depth,'',time.monotonic(),pitch),old_camera,.15)
        g=Geometry(Observation(rgb,depth,'',time.monotonic(),pitch),camera,.05)
        if scene:
            old.set_humans(scene)
            g.set_humans(scene)
        before=OldController.next_step_state(old)
        directions=[]
        for degrees in range(-30,31,15):
            angle=math.radians(degrees)
            check=g.assess_step(angle)
            before_counts[old.diagnose(angle,1.)['path_state']]+=1
            after_counts[check['state']]+=1
            directions.append(dict(bearing_deg=degrees,status=check['state'],reason=check['reason'],
                depth=check['depth'],support=check['support'],
                remaining_collision_points=collision_points(g,angle,terrain.obstacle_mask(g,check['support']))))
        reports.append(dict(observation_id=record['observation_id'],input_file=record['input_file'],pitch_deg=pitch,
            rgb_sha256=array_hash(rgb),depth_sha256=array_hash(depth),
            human_semantics='same_saved_rgb_and_pose' if scene else 'not_available',
            original_front=before,original_collision_points=collision_points(old,0,old.obstacles),directions=directions))
    executable=[dict(observation_id=r['observation_id'],bearings_deg=[d['bearing_deg'] for d in r['directions']
        if d['status']=='EXECUTABLE']) for r in reports]
    summary=dict(validation_scope='OFFLINE_SAVED_RGBD_NOT_EXECUTED_NAVIGATION',observations=len(reports),
        before_depth_states=dict(before_counts),after_motion_states=dict(after_counts),
        executable_views=[v for v in executable if v['bearings_deg']],
        safety_margin_before_m=.15,safety_margin_after_m=.05,physical_robot_radius_m=camera.robot_radius,
        max_step_height_m=camera.max_step_height,real_forward_actions=0)
    (output/'geometry.json').write_text(json.dumps(reports,indent=2))
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-run',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    regress(args.saved_run,args.baseline,args.output_dir)


if __name__=='__main__':main()
