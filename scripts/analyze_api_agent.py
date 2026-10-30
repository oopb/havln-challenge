"""Summarize generation logs; never infer official SR/NE/CR from model STOP."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from havln_agent.api_client import latency_stats, usage_counts


def request_repetition(records):
    """Exact physical payload repeats, based only on actual wire fingerprints."""
    signatures, sources = Counter(), Counter()
    physical = [r for r in records if not r.get('cache_hit')]
    for record in physical:
        source = 'request_body_sha256' if record.get('request_body_sha256') else 'unavailable'
        sources[source] += 1
        signature = record.get('request_body_sha256')
        if signature:
            signatures[(record.get('role'), record.get('purpose'), record.get('model'), signature)] += 1
    count = sum(signatures.values())
    repeated = sum(n-1 for n in signatures.values())
    return dict(physical_requests=len(physical), fingerprinted_requests=count,
        duplicate_physical_requests=repeated, duplicate_request_rate=repeated/count if count else None,
        fingerprint_sources=dict(sources))


def api_groups(records, field):
    """Count the global journal once, preserving unlabeled records."""
    groups = {}
    for record in records:
        key = record.get(field) or ('outside_subgoal' if field == 'subgoal_id' else 'unlabeled')
        if field == 'subgoal_id' and record.get('trace_episode'):
            key = record['trace_episode'] + '/' + key
        group = groups.setdefault(key, dict(requests=0, cache_hits=0, seconds=0, retries=0,
            errors=0, timeouts=0, usage={}, context_chars=[], input_text_sizes=[], system_sizes=[], image_attachments=0,
            request_records=[]))
        if 'context_chars' in record:
            group['context_chars'].append(record['context_chars'])
        if record.get('cache_hit'):
            group['cache_hits'] += 1
            continue
        group['requests'] += 1
        group['request_records'].append(record)
        messages=record.get('messages', [])
        text_sizes=[len(m['content']) if isinstance(m.get('content'),str) else
                    sum(len(c.get('text','')) for c in m.get('content',[])) for m in messages]
        if messages:
            group['input_text_sizes'].append(record.get('input_text_chars',sum(text_sizes)))
            group['system_sizes'].append(record.get('system_chars',sum(n for n,m in zip(text_sizes,messages) if m['role']=='system')))
        image_hashes=messages[1].get('image_sha256',[]) if len(messages)>1 else []
        group['image_attachments']+=record.get('image_count',len(image_hashes) if isinstance(image_hashes,list) else int(bool(image_hashes)))
        group['seconds'] += record['seconds']
        group['retries'] += int(record.get('attempt', 0) > 0 or record.get('endpoint_index', 0) > 0 or record.get('format_retry', 0) > 0)
        group['errors'] += int('error' in record)
        group['timeouts'] += int(record.get('error') == 'request_timeout')
        for name, value in usage_counts(record.get('usage')).items():
            group['usage'][name] = group['usage'].get(name, 0) + value
    for group in groups.values():
        group['request_repetition'] = request_repetition(group.pop('request_records'))
        for key,destination in [('input_text_sizes','input_text_characters'),('system_sizes','system_characters')]:
            values=group.pop(key)
            group[destination]=dict(count=len(values),total=sum(values),mean=sum(values)/len(values) if values else None,
                                    maximum=max(values) if values else None)
        sizes = group.pop('context_chars')
        group['context_characters'] = dict(count=len(sizes), mean=sum(sizes)/len(sizes) if sizes else None,
                                          maximum=max(sizes) if sizes else None)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    metadata_path = args.output_dir / 'export_metadata.json'
    identity = json.loads(metadata_path.read_text()).get('identity', {}) if metadata_path.exists() else {}
    summaries = [json.loads(path.read_text()) for path in
                 args.output_dir.glob('traces/*/summary.json') if '.attempt-' not in path.parent.name]
    if not summaries:
        parser.error('No episode summaries found')
    reasons = Counter(s['stop_reason'] for s in summaries)
    usage = Counter()
    for summary in summaries:
        usage.update(summary['api']['usage'])
    requests = [dict(trace_episode=path.parent.name, **json.loads(line)) for path in args.output_dir.glob('traces/*/api.jsonl')
                if '.attempt-' not in path.parent.name for line in path.read_text().splitlines()]
    physical = [r for r in requests if not r.get('cache_hit')]
    ledgers = [json.loads(path.read_text()) for path in args.output_dir.glob('traces/*/tool_ledger.json')
               if '.attempt-' not in path.parent.name]
    worlds = [json.loads(line) for path in args.output_dir.glob('traces/*/state.jsonl')
              if '.attempt-' not in path.parent.name for line in path.read_text().splitlines()]
    by_purpose = {purpose: latency_stats([r['seconds'] for r in physical if r['purpose'] == purpose])
                  for purpose in sorted({r['purpose'] for r in physical})}
    print(json.dumps(dict(episodes=len(summaries), stop_reasons=dict(reasons),
                         validation_scope=identity.get('validation_scope', 'UNLABELED_SAVED_TRACE'),
                         mean_steps=sum(s['steps'] for s in summaries) / len(summaries),
                         mean_seconds=sum(s['seconds'] for s in summaries) / len(summaries),
                         api_requests=sum(s['api']['requests'] for s in summaries),
                         api_seconds=sum(s['api']['seconds'] for s in summaries),
                         api_budgets=[s['api'].get('budget') for s in summaries],
                         tool_calls=sum(s.get('tool_calls', 0) for s in summaries),
                         state_snapshots=len(worlds),
                         tool_ledger_episodes=len(ledgers),
                         scene_validity_states=dict(Counter(w['scene_state']['validity']['status'] for w in worlds)),
                         unresolved_tool_calls=sum(not r['resolved'] for ledger in ledgers for r in ledger) if ledgers else None,
                         path_revalidation_calls=sum('validation' in r['result'] for ledger in ledgers for r in ledger) if ledgers else None,
                         tool_statuses=dict(sum((Counter(s.get('tool_outcomes', {})) for s in summaries), Counter())),
                         safety_rejections=sum(s.get('safety_rejections', 0) for s in summaries),
                         finish_reasons=dict(Counter(s['finish_reason'] for s in summaries)),
                         navigation_success='requires_independent_scorer',
                         latency=latency_stats([r['seconds'] for r in physical]),
                         latency_by_purpose=by_purpose,
                         by_role=api_groups(requests, 'role'),
                         by_model=api_groups(requests, 'model'),
                         by_endpoint=api_groups([dict(r,endpoint=r['role']+'/'+r.get('endpoint_id','primary'))
                             for r in requests], 'endpoint'),
                         by_subgoal=api_groups(requests, 'subgoal_id'),
                         retry_count=sum(r.get('attempt', 0) > 0 or r.get('endpoint_index', 0) > 0 or r.get('format_retry', 0) > 0 for r in physical),
                         failed_requests=sum('error' in r for r in physical),
                         timeout_scopes=dict(Counter(r.get('timeout_scope', 'unrecorded') for r in physical
                                                     if r.get('error') == 'request_timeout')),
                         failed_request_phases=dict(Counter(r.get('request_phase', 'unrecorded') for r in physical
                                                            if 'error' in r)),
                         cache_calls_saved=sum(bool(r.get('cache_hit')) for r in requests),
                         request_repetition=request_repetition(requests),
                         token_usage=dict(usage)), indent=2))


if __name__ == '__main__':
    main()
