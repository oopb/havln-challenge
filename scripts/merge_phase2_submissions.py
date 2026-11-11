"""Validate selected Phase 2 Agent run directories and build a submission ZIP.

No Habitat, GPU, network, or model API calls are made by this script.
Run directories are explicit inputs; incomplete batches and duplicate IDs fail
before any submission files are written.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

# Allow direct invocation via python /toolkit/scripts/merge_phase2_submissions.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_cma_submission import check_rows

TOTAL = 3408
ALL_IDS = [str(i) for i in range(1, TOTAL + 1)]


def spans(ids):
    """Compact missing/overlapping numeric Episode IDs for terminal feedback."""
    numbers = sorted(int(value) for value in ids)
    if not numbers:
        return 'none'
    ranges = []
    first = last = numbers[0]
    for value in numbers[1:]:
        if value != last + 1:
            ranges.append(str(first) if first == last else f'{first}-{last}')
            first = value
        last = value
    ranges.append(str(first) if first == last else f'{first}-{last}')
    return ', '.join(ranges)


def _read_json(path):
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def read_run(directory):
    """Read one completed Agent batch and verify its declared ID range."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f'Run directory does not exist: {directory}')

    receipt = directory / 'export_metadata.json'
    if not receipt.is_file():
        raise ValueError(f'Missing export_metadata.json: {directory}')
    metadata = _read_json(receipt)
    identity = metadata.get('identity') or {}
    if identity.get('phase') != 'phase2' or identity.get('splits') != ['test']:
        raise ValueError(f'Not a Phase 2 Test Agent run: {directory}')
    if metadata.get('complete') is not True:
        raise ValueError(f'Run is not marked complete: {directory}')
    fingerprint = metadata.get('fingerprint')
    data_hash = (identity.get('datasets') or {}).get('test')
    if not fingerprint or not data_hash:
        raise ValueError(f'Missing run fingerprint or Test dataset hash: {directory}')

    start = identity.get('episode_start', 1)  # compatible with older Phase 2 runs
    limit = identity.get('episode_limit')
    if type(start) is not int or not 1 <= start <= TOTAL:
        raise ValueError(f'Invalid stored episode_start in {directory}')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError(f'Invalid stored episode_limit in {directory}')
    end = min(TOTAL, start + limit - 1) if limit is not None else TOTAL
    expected_ids = [str(i) for i in range(start, end + 1)]

    filename = 'test.diagnostic.json' if metadata.get('diagnostic') else 'test.json'
    result_path = directory / filename
    if not result_path.is_file():
        raise ValueError(f'Missing assembled {filename}: {directory}')
    payload = _read_json(result_path)
    if payload.get('format_version') != 1 or payload.get('split') != 'test':
        raise ValueError(f'Invalid Phase 2 Test result header: {result_path}')
    rows = payload.get('episodes')
    if not isinstance(rows, list):
        raise ValueError(f'Expected episodes array: {result_path}')
    try:
        check_rows(rows, expected_ids)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Invalid actions/ID coverage in {result_path}: {exc}') from exc
    if (metadata.get('episodes') or {}).get('test') != len(expected_ids):
        raise ValueError(f'Export metadata episode count disagrees with results: {directory}')

    # If original traces were retained, independently cross-check the exported
    # actions against the worker's executed-action log. A minimal copy of only
    # metadata + assembled JSON is also a valid input.
    trace_verified = 0
    for row in rows:
        trace_dir = directory / 'traces' / ('test-' + row['episode_id'])
        episode_result = trace_dir / 'result.json'
        steps = trace_dir / 'steps.jsonl'
        if episode_result.exists() != steps.exists():
            raise ValueError(f'Partial retained Episode trace: {trace_dir}')
        if not episode_result.exists():
            continue
        stored = _read_json(episode_result)
        if stored.get('fingerprint') != fingerprint or stored.get('row') != row:
            raise ValueError(f'Exported actions differ from saved Episode result: {trace_dir}')
        try:
            executed = [json.loads(line)['action'] for line in steps.read_text().splitlines()]
        except (KeyError, ValueError) as exc:
            raise ValueError(f'Invalid executed-action log: {steps}: {exc}') from exc
        if executed != row['actions']:
            raise ValueError(f'Exported actions differ from executed-action log: {trace_dir}')
        trace_verified += 1

    return dict(directory=directory, rows=rows, data_hash=data_hash,
                start=start, end=end, traces_verified=trace_verified,
                fingerprint=fingerprint)


def collect_runs(directories):
    if not directories:
        raise ValueError('Select at least one run directory')
    resolved = [Path(item).expanduser().resolve() for item in directories]
    if len(resolved) != len(set(resolved)):
        raise ValueError('The same run directory was selected more than once')

    runs = [read_run(path) for path in resolved]
    hashes = {run['data_hash'] for run in runs}
    if len(hashes) != 1:
        raise ValueError('Selected runs use different Phase 2 Test datasets')

    rows_by_id = {}
    owners = {}
    for run in runs:
        for row in run['rows']:
            key = row['episode_id']
            if key in rows_by_id:
                raise ValueError(f'Duplicate Episode {key} in both '
                                 f'{owners[key]} and {run["directory"]}')
            rows_by_id[key] = row
            owners[key] = run['directory']
    missing = set(ALL_IDS) - set(rows_by_id)
    rows = [rows_by_id[key] for key in ALL_IDS if key in rows_by_id]
    print(f'Inputs: {len(runs)} completed directories; '
          f'Episodes: {len(rows)}/{TOTAL}; '
          f'executed-action traces verified: '
          f'{sum(run["traces_verified"] for run in runs)}/{len(rows)}')
    for run in runs:
        print(f'  {run["start"]}-{run["end"]}: {run["directory"]}')
    if missing:
        print(f'Uncovered Episode IDs: {spans(missing)}')
    else:
        check_rows(rows, ALL_IDS)
        print(f'PASS: exact coverage of all {TOTAL} Episode IDs; actions valid')
    if len({run['fingerprint'] for run in runs}) > 1:
        print('Note: selected runs have different run fingerprints; '
              'their original test dataset is identical.')
    return rows, missing, runs


def package(rows, destination, overwrite=False):
    """Stage files, reopen and verify ZIP, then publish without partial files."""
    destination = Path(destination).expanduser().resolve()
    payload = dict(format_version=1, split='test', episodes=rows)
    check_rows(rows, ALL_IDS)
    destination.mkdir(parents=True, exist_ok=True)
    test_file = destination / 'test.json'
    zip_file = destination / 'submission.zip'
    if not overwrite and (test_file.exists() or zip_file.exists()):
        raise ValueError('Output already contains test.json or submission.zip; '
                         'use a fresh --output-dir or --overwrite')

    with tempfile.TemporaryDirectory(prefix='.phase2-merge-', dir=destination) as temp:
        staging = Path(temp)
        staged_json = staging / 'test.json'
        staged_zip = staging / 'submission.zip'
        staged_json.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        with zipfile.ZipFile(staged_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(staged_json, arcname='test.json')
        with zipfile.ZipFile(staged_zip) as archive:
            if archive.namelist() != ['test.json'] or archive.testzip() is not None:
                raise ValueError('Submission ZIP structure or CRC verification failed')
            actual = json.loads(archive.read('test.json'))
        if actual != payload:
            raise ValueError('Submission ZIP contents differ from selected run results')
        check_rows(actual['episodes'], ALL_IDS)
        os.replace(staged_json, test_file)
        os.replace(staged_zip, zip_file)
    print(f'PASS: {test_file}')
    print(f'PASS: {zip_file} (root contains only test.json)')
    return zip_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dirs', nargs='+', type=Path, required=True,
                        help='Explicit completed Phase 2 Agent output directories')
    parser.add_argument('--output-dir', type=Path,
                        help='Directory for final test.json and submission.zip')
    parser.add_argument('--check-only', action='store_true',
                        help='Validate selected runs and report missing IDs without packaging')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace an existing test.json/submission.zip after validation')
    args = parser.parse_args(argv)
    if not args.check_only and args.output_dir is None:
        parser.error('--output-dir is required unless --check-only is specified')
    try:
        rows, missing, runs = collect_runs(args.run_dirs)
        if args.check_only:
            return 0  # Missing ranges are reported as an informational preview.
        if missing:
            raise ValueError(f'Cannot submit: {len(missing)} Episode IDs are missing')
        out = args.output_dir.expanduser().resolve()
        if any(out == run['directory'] or out.is_relative_to(run['directory'])
               for run in runs):
            raise ValueError('--output-dir must be outside selected run directories')
        package(rows, out, args.overwrite)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
