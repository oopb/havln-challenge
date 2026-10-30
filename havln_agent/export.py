"""Independent scan workers and episode resume, reusing official export checks."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import zipfile

import yaml

from .api_client import APIClient, canonical, PROMPT_VERSION, model_endpoints
from .episode_runner import run_episode
from .habitat_adapter import camera_from_config, environment_config, runtime_imports
from .progress_monitor import ProgressMonitor
from scripts.export_cma_submission import SOURCE_REVISION, SPLITS, atomic_json, check_rows, sha256

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / 'scripts/run_api_agent.py'
DEFAULT_MANIFEST = Path('/opt/havln-challenge/codabench/phase1/reference/public_reference_manifest.json')


def load_config(path):
    text = Path(path).read_text()
    variables = set(re.findall(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', text))
    missing = sorted(name for name in variables if not os.environ.get(name))
    if missing:
        raise ValueError('Missing configuration variables: ' + ', '.join(missing) +
                         '; use configs/agent.env.example and docker exec --env-file.')
    config = yaml.safe_load(os.path.expandvars(text))
    roles = ('task_planner', 'navigation_agent', 'visual_perception')
    if not isinstance(config, dict) or set(config) != set(roles) | {'runtime'}:
        raise ValueError('Expected task_planner, navigation_agent, visual_perception and runtime sections')
    for role in roles:
        model_endpoints(config[role])
    limits = config['runtime']
    required = {'max_steps', 'api_budget', 'waypoint_distance', 'safety_margin',
                'tool_max_actions', 'max_scene_age_seconds', 'max_depth_age_seconds',
                'max_static_landmark_age_seconds', 'max_human_age_seconds'}
    if not isinstance(limits, dict) or set(limits) != required:
        raise ValueError('Unexpected runtime fields')
    for name in ('max_steps', 'api_budget', 'tool_max_actions'):
        if type(limits[name]) is not int or limits[name] < 1:
            raise ValueError('Invalid runtime.%s=%r: expected a positive integer' % (name,limits[name]))
    for name in required - {'max_steps', 'api_budget', 'tool_max_actions'}:
        if type(limits[name]) not in (int, float) or not math.isfinite(limits[name]):
            raise ValueError('Invalid runtime.%s=%r: expected a finite number' % (name,limits[name]))
    checks = (
        ('max_steps', limits['max_steps'] <= 500, 'an integer in [1, 500]'),
        ('tool_max_actions', limits['tool_max_actions'] <= 3, 'an integer in [1, 3]'),
        ('waypoint_distance', 0 < limits['waypoint_distance'] <= 1.5, '0 < value <= 1.5'),
        ('safety_margin', 0 <= limits['safety_margin'] <= 1, '0 <= value <= 1'),
        ('max_depth_age_seconds', 0 < limits['max_depth_age_seconds'] <= 1, '0 < value <= 1'),
        ('max_scene_age_seconds', limits['max_scene_age_seconds'] > 0, 'value > 0'),
        ('max_static_landmark_age_seconds', limits['max_static_landmark_age_seconds'] > 0, 'value > 0'),
        ('max_human_age_seconds', limits['max_human_age_seconds'] > 0, 'value > 0'),
    )
    for name, valid, expected in checks:
        if not valid:
            raise ValueError('Invalid runtime.%s=%r: expected %s' % (name,limits[name],expected))
    return config


def build_jobs(data, splits, manifest, limit):
    from havln_scoring.submission_contract import load_manifest
    specs = {s.split: set(s.episode_ids) for s in load_manifest(manifest).splits}
    expected, jobs, datasets = {}, [], {}
    for split in splits:
        path = data / f'HA-R2R/{split}/{split}.json.gz'
        datasets[split] = sha256(path)
        with gzip.open(path, 'rt') as handle:
            episodes = json.load(handle)['episodes']
        ids = [str(e['episode_id']) for e in episodes]
        if len(ids) != len(set(ids)) or set(ids) != specs[split]:
            raise ValueError('Input dataset differs from the official manifest')
        episodes.sort(key=lambda e: int(e['episode_id']))
        if limit is not None:
            episodes = episodes[:limit]
        expected[split] = [str(e['episode_id']) for e in episodes]
        scans = {}
        for episode in episodes:
            episode_id = str(episode['episode_id'])
            if not re.fullmatch(r'[A-Za-z0-9_-]+', episode_id):
                raise ValueError('Unsafe episode ID for trace storage')
            scan = Path(episode['scene_id']).parent.name
            if not (data / 'scene_datasets/mp3d' / scan / (scan + '.glb')).is_file():
                raise ValueError('Missing licensed scene: ' + scan)
            scans.setdefault(scan, []).append(episode_id)
        jobs.extend(dict(key=split + '-' + scan, split=split, ids=ids)
                    for scan, ids in sorted(scans.items()))
    for relative in ('HAPS2_0', 'Multi-Human-Annotations/human_motion.json'):
        if not (data / relative).exists():
            raise ValueError('Missing simulator data: ' + relative)
    return jobs, expected, datasets


def completed_episode(directory, fingerprint, episode_id):
    result_path = directory / 'result.json'
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text())
    if result['fingerprint'] != fingerprint:
        raise ValueError('Completed episode provenance differs')
    row = result['row']
    check_rows([row], [episode_id])
    executed = [json.loads(line)['action'] for line in (directory / 'steps.jsonl').read_text().splitlines()]
    if executed != row['actions']:
        raise ValueError('Completed episode differs from its executed-action log')
    return row


def run_worker(task_path):
    task = json.loads(Path(task_path).read_text())
    import numpy as np
    from havln_scoring.runtime import quiet_runtime
    get_config, env_class = runtime_imports(Path(task['source']))
    random.seed(task['seed'])
    np.random.seed(task['seed'])
    data, output = Path(task['data']), Path(task['output'])
    client = APIClient(task['config'], output / 'cache' / str(task['gpu']))
    progress_file = output / ('worker-' + str(task['gpu']) + '.progress.json')
    for job in task['jobs']:
        shard = output / 'shards' / (job['key'] + '.json')
        rows, pending = [], []
        for episode_id in job['ids']:
            directory = output / 'traces' / (job['split'] + '-' + episode_id)
            row = completed_episode(directory, task['fingerprint'], episode_id)
            if row is None:
                api_log = directory / 'api.jsonl'
                if api_log.exists() and api_log.stat().st_size and not task['retry_incomplete']:
                    raise ValueError('Incomplete episode ' + episode_id +
                                     ' has API work; inspect it and pass --retry-incomplete to rerun')
                pending.append(episode_id)
            else:
                rows.append(row)
        if pending:
            atomic_json(progress_file, dict(phase='initializing',
                split=job['split'], episode_id=pending[0],
                max_steps=task['config']['runtime']['max_steps'],
                started_at_epoch=time.time()))
            with (nullcontext() if task['verbose'] else quiet_runtime()):
                config = environment_config(get_config, task['runtime_config'], data,
                                            job['split'], pending, task['gpu'], task['seed'])
                camera = camera_from_config(config)
                atomic_json(output / ('camera-' + job['key'] + '.json'), asdict(camera))
                with env_class(config=config) as env:
                    episodes = sorted(env.episodes, key=lambda e: int(e.episode_id))
                    if set(str(e.episode_id) for e in episodes) != set(pending):
                        raise ValueError('Runtime dataset episode coverage differs')
                    env.habitat_env.episode_iterator = iter(episodes)
                    for episode in episodes:
                        episode_id = str(episode.episode_id)
                        directory = output / 'traces' / (job['split'] + '-' + episode_id)
                        if directory.exists():
                            # Preserve failed attempts, including paid model outputs.
                            suffix = 1
                            while directory.with_name(directory.name + '.attempt-' + str(suffix)).exists():
                                suffix += 1
                            directory.rename(directory.with_name(directory.name + '.attempt-' + str(suffix)))
                        atomic_json(progress_file, dict(phase='running',
                            split=job['split'], episode_id=episode_id,
                            max_steps=task['config']['runtime']['max_steps'],
                            started_at_epoch=time.time()))
                        row = run_episode(env, episode_id, camera, client,
                                          task['config']['runtime'], directory,
                                          instruction=episode.instruction.instruction_text)
                        atomic_json(directory / 'result.json', dict(fingerprint=task['fingerprint'], row=row))
                        rows.append(row)
                        atomic_json(progress_file, dict(phase='completed',
                            split=job['split'], episode_id=episode_id,
                            max_steps=task['config']['runtime']['max_steps'],
                            finished_at_epoch=time.time(), actions=len(row['actions'])))
                        print('Completed ' + job['split'] + '/' + episode_id, flush=True)
        check_rows(rows, job['ids'])
        atomic_json(shard, dict(fingerprint=task['fingerprint'], rows=rows))
    atomic_json(progress_file, dict(phase='finished',split='all',episode_id='all',
        finished_at_epoch=time.time()))


def assemble(output, jobs, expected, fingerprint, diagnostic):
    filenames = []
    for split, ids in expected.items():
        rows = []
        for job in jobs:
            if job['split'] != split:
                continue
            payload = json.loads((output / 'shards' / (job['key'] + '.json')).read_text())
            if payload['fingerprint'] != fingerprint:
                raise ValueError('Shard provenance differs')
            check_rows(payload['rows'], job['ids'])
            rows.extend(payload['rows'])
        check_rows(rows, ids)
        rows.sort(key=lambda row: int(row['episode_id']))
        filename = split + ('.diagnostic.json' if diagnostic else '.json')
        atomic_json(output / filename, dict(format_version=1, split=split, episodes=rows))
        filenames.append(filename)
    if diagnostic:
        print('Diagnostic only: no submission ZIP is produced.')
        return
    if set(expected) != set(SPLITS):
        raise ValueError('Formal Phase 1 export requires both validation splits')
    partial = output / 'submission.zip.partial'
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in filenames:
            archive.write(output / filename, arcname=filename)
    subprocess.run(['havln-validate', str(partial)], check=True)
    os.replace(partial, output / 'submission.zip')


def export(args, config):
    data, output = args.data_root.resolve(), args.output_dir.resolve()
    if (output / 'export_metadata.json').exists():
        previous = json.loads((output / 'export_metadata.json').read_text())
        if previous.get('identity', {}).get('agent') != PROMPT_VERSION:
            raise ValueError('Output belongs to another pipeline; choose a fresh Agent directory')
    source = args.source_root.resolve()
    get_config, env_class = runtime_imports(source)
    jobs, expected, datasets = build_jobs(data, args.splits, args.manifest, args.episode_limit)
    import habitat
    import habitat_sim
    import numpy as np
    import PIL
    import jsonschema
    state = json.loads((source / '.release.json').read_text())
    if state['revision'] != SOURCE_REVISION or any(sha256(source / p) != digest for p, digest in state['files'].items()):
        raise ValueError('Pinned public sources have been modified')
    code = list((ROOT / 'havln_agent').rglob('*.py')) + [ENTRYPOINT, ROOT / 'scripts/export_cma_submission.py']
    try:
        revision = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True)
    except FileNotFoundError:
        revision = None
    identity = dict(agent=PROMPT_VERSION, config=config, datasets=datasets,
                    source_revision=state['revision'], source_manifest=sha256(source / '.release.json'),
                    git_revision=revision.stdout.strip() if revision is not None and revision.returncode == 0 else 'unavailable',
                    code={str(p.relative_to(ROOT)): sha256(p) for p in code}, seed=args.seed,
                    runtime_config=sha256(args.runtime_config), manifest=sha256(args.manifest),
                    resolved_runtime_config=hashlib.sha256(str(get_config(str(args.runtime_config))).encode()).hexdigest(),
                    simulator=sha256(Path(habitat.__file__).parent / 'sims/habitat_simulator/habitat_simulator.py'),
                    environment=sha256(sys.modules[env_class.__module__].__file__),
                    runtime=dict(habitat=habitat.__version__, habitat_sim=habitat_sim.__version__,
                                 numpy=np.__version__, pillow=PIL.__version__,
                                 jsonschema=getattr(jsonschema, '__version__', 'unknown')),
                    human_annotations=sha256(data / 'Multi-Human-Annotations/human_motion.json'),
                    data_root=str(data), splits=args.splits, episode_limit=args.episode_limit)
    fingerprint = hashlib.sha256(canonical(identity).encode()).hexdigest()
    receipt = output / 'export_metadata.json'
    if receipt.exists() and json.loads(receipt.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Output configuration/code/runtime differs; use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    # Validate credentials/config once, before launching any simulator worker.
    APIClient(config, output / 'cache/preflight')
    diagnostic = (args.episode_limit is not None or set(args.splits) != set(SPLITS) or
                  config['runtime']['max_steps'] != 500)
    metadata = dict(fingerprint=fingerprint, identity=identity, complete=False, diagnostic=diagnostic)
    atomic_json(receipt, metadata)
    processes, logs = [], []
    try:
        for index, gpu in enumerate(args.gpu_ids):
            task = dict(config=config, data=str(data), output=str(output), gpu=gpu, source=str(source),
                        runtime_config=str(args.runtime_config.resolve()), fingerprint=fingerprint,
                        seed=args.seed, retry_incomplete=args.retry_incomplete,
                        verbose=args.verbose, jobs=jobs[index::len(args.gpu_ids)])
            task_path = output / ('worker-' + str(gpu) + '.json')
            atomic_json(task_path, task)
            log = (output / ('worker-' + str(gpu) + '.log')).open('a')
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, str(ENTRYPOINT), '--worker-task', str(task_path)],
                                              stdout=log, stderr=subprocess.STDOUT))
        interval = getattr(args, 'progress_interval', 3.0)
        if interval > 0:
            monitor = ProgressMonitor(output, args.gpu_ids)
            print('[HA-VLN] Live terminal progress enabled; full worker output remains in worker-<GPU>.log',
                  flush=True)
            monitor.report()
            while any(process.poll() is None for process in processes):
                time.sleep(interval)
                monitor.report()
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError('Agent worker failed: ' + str(codes) + '; inspect worker logs')
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for log in logs:
            log.close()
    assemble(output, jobs, expected, fingerprint, diagnostic)
    metadata.update(complete=True, episodes={split: len(ids) for split, ids in expected.items()})
    atomic_json(receipt, metadata)
    print('Completed Agent export: ' + str(output))


def preflight(args):
    # Report missing dependencies without importing or modifying the simulator.
    report = {'dependencies': {name: importlib.util.find_spec(name) is not None
                              for name in ('habitat', 'habitat_sim', 'havln_scoring')},
              'runtime_config_exists': args.runtime_config.is_file(),
              'manifest_exists': args.manifest.is_file()}
    try:
        config = load_config(args.config)
        report['configuration'] = 'loaded'
        report['credentials_present'] = {role: all(bool(os.environ.get(endpoint['api_key_env']))
                    for endpoint in model_endpoints(config[role]))
                    for role in ('task_planner','navigation_agent','visual_perception')}
    except ValueError as error:
        report['configuration'] = str(error)
        report['credentials_present'] = {}
    print(json.dumps(report, indent=2))
    return (all(report['dependencies'].values()) and report['runtime_config_exists'] and
            report['manifest_exists'] and report['configuration'] == 'loaded' and
            bool(report['credentials_present']) and all(report['credentials_present'].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/agent.yaml')
    parser.add_argument('--data-root', type=Path, default=Path('/data/havln2'))
    parser.add_argument('--source-root', type=Path, default=Path('/workspace/cma-deps/HA-VLN'))
    parser.add_argument('--runtime-config', type=Path, default=Path('/app/config/challenge_submission.yaml'))
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=list(SPLITS))
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[0])
    parser.add_argument('--episode-limit', type=int)
    parser.add_argument('--max-steps', type=int, help='Diagnostic action budget, including executed STOP')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--preflight', action='store_true', help='Read-only checks; no API requests')
    parser.add_argument('--retry-incomplete', action='store_true', help='Authorize rerunning partial API work')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--progress-interval', type=float, default=3.0,
                        help='Terminal worker progress polling in seconds (default 3; 0 disables)')
    parser.add_argument('--worker-task', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_task:
        run_worker(args.worker_task)
        return
    if args.preflight:
        sys.exit(0 if preflight(args) else 1)
    if args.output_dir is None:
        parser.error('--output-dir is required')
    if not math.isfinite(args.progress_interval) or not 0 <= args.progress_interval <= 60:
        parser.error('--progress-interval must be a finite number between 0 and 60 seconds')
    if args.episode_limit is not None and args.episode_limit < 1:
        parser.error('--episode-limit must be positive')
    if len(set(args.splits)) != len(args.splits):
        parser.error('splits must be distinct')
    if len(set(args.gpu_ids)) != len(args.gpu_ids) or any(g < 0 for g in args.gpu_ids):
        parser.error('GPU IDs must be distinct and nonnegative')
    config = load_config(args.config)
    if args.max_steps is not None:
        if not 1 <= args.max_steps <= 500:
            parser.error('--max-steps must be between 1 and 500')
        config['runtime']['max_steps'] = args.max_steps
    export(args, config)
