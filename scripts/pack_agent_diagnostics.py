"""Build compact shareable run archives without duplicate RGB-D frames."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

TRACE_NAMES = ('api.jsonl','navigation_agent.jsonl','task_planner.jsonl',
    'visual_perception.jsonl','tools.jsonl','steps.jsonl','safety.jsonl',
    'progress.jsonl','events.jsonl','memory.jsonl','memory.json','episode.json',
    'summary.json','result.json','final_state.json','failure.json','state.jsonl')


def read(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def rows(path):
    try:
        return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    except OSError:
        return []


def select_frames(trace, interval):
    history = read(trace/'memory.json', {})
    observations = history.get('observations', [])
    if not observations:
        return set()
    hashes = {(o['rgb_hash'],o['depth_hash']):o['observation_id']
              for o in observations}
    chosen = {observations[0]['observation_id'],observations[-1]['observation_id']}
    steps = rows(trace/'steps.jsonl')
    indices = set()
    for i, action in enumerate(steps):
        if i%interval==0 or action.get('action')=='MOVE_FORWARD':
            indices.update((max(0,i-1),i,min(i+1,len(steps)-1)))
    for i in indices:
        for key in ('input_file','post_file'):
            filename = steps[i].get(key)
            if filename:
                chosen.add(Path(filename).stem)
        for key in ('observation','post_observation'):
            meta = steps[i].get(key) or {}
            ident = hashes.get((meta.get('rgb_hash'),meta.get('depth_hash')))
            if ident:chosen.add(ident)
    for tool in rows(trace/'tools.jsonl'):
        call, result = tool.get('call') or {}, tool.get('result') or {}
        if (call.get('tool') in ('navigate_local','inspect_geometry')
                and result.get('status') in ('BLOCKED','PARTIAL','REQUIRES_ATTENTION')):
            ident=result.get('end_observation_id')
            if ident:chosen.add(ident)
        candidate=result.get('best_candidate') or {}
        if candidate.get('observation_id'):
            chosen.add(candidate['observation_id'])
    return chosen


def pack(root, destination, frames='critical', interval=25):
    root, destination = Path(root), Path(destination)
    trace_root=root/'traces'
    if not trace_root.is_dir():
        raise ValueError('missing traces directory')
    episodes=sorted(x for x in trace_root.iterdir() if x.is_dir())
    if not episodes:raise ValueError('no episode directories')
    # Worker task JSON may expose per-model settings and is not needed for diagnosis.
    files={p for p in root.glob('*.json')
           if not (p.name.startswith('worker-') and p.name.endswith('.json')
                   and '.progress.' not in p.name)}
    files.update(root.glob('worker-*.log'))
    files.update((root/'shards').glob('*.json'))
    catalog={}
    for trace in episodes:
        files.update(trace/name for name in TRACE_NAMES if (trace/name).is_file())
        images=sorted((trace/'observations').glob('obs_*.npz'))
        selected=({x.stem for x in images} if frames=='all' else
                  select_frames(trace,interval) if frames=='critical' else set())
        files.update(x for x in images if x.stem in selected)
        catalog[trace.name]=dict(total_frames=len(images),
            included_frames=sum(x.stem in selected for x in images))
    destination.parent.mkdir(parents=True,exist_ok=True)
    manifest=dict(format='havln-diagnostic-v1',frames=frames,episodes=catalog,
        note='Only selected RGB-D frames are included; full traces remain on server.')
    with ZipFile(destination,'w',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(files):
            if file.is_file():
                z.write(file,root.name+'/'+str(file.relative_to(root)))
        z.writestr(root.name+'/diagnostic_manifest.json',json.dumps(manifest,indent=2))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--archive',type=Path)
    p.add_argument('--frames',choices=['critical','all','none'],default='critical')
    p.add_argument('--interval',type=int,default=25)
    a=p.parse_args()
    if a.interval<1:p.error('--interval must be positive')
    destination=a.archive or a.output_dir.with_name(a.output_dir.name+'-diagnostic.zip')
    info=pack(a.output_dir,destination,a.frames,a.interval)
    print(f'{destination}: {destination.stat().st_size/1048576:.1f} MiB')
    for key,value in info['episodes'].items():
        print(f'{key}: {value["included_frames"]}/{value["total_frames"]} RGB-D frames')


if __name__=='__main__':
    main()
