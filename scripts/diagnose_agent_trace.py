"""Read canonical role decisions, tool evidence and STOP attribution without APIs."""
import argparse
import json
from pathlib import Path


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def diagnose(directory, subgoal_id=None, call_id=None):
    summary=json.loads((directory/'summary.json').read_text())
    ledger=json.loads((directory/'tool_ledger.json').read_text())
    selected=[r for r in ledger if (subgoal_id is None or r['subgoal_id']==subgoal_id)
        and (call_id is None or r['call_id']==call_id)]
    calls={r['call_id'] for r in selected}
    decisions={r['request_context'].get('decision_id') for r in selected}
    filtered=subgoal_id is not None or call_id is not None
    roles={}
    for role in ('task_planner','navigation_agent','visual_perception'):
        rows=read_lines(directory/(role+'.jsonl'))
        roles[role]=[r for r in rows if not filtered or r.get('tool_call_id') in calls
            or r.get('decision_id') in decisions or (subgoal_id is not None and r.get('subgoal_id')==subgoal_id)]
    return dict(scope='trace_facts_not_navigation_metrics',episode_id=summary['episode_id'],
        termination=dict(reason=summary['stop_reason'],finish_reason=summary['finish_reason']),
        api=summary['api'],role_events=roles,tool_calls=selected,
        safety=[r for r in read_lines(directory/'safety.jsonl') if not filtered or r.get('tool_call_id') in calls],
        progress=[r for r in read_lines(directory/'progress.jsonl') if subgoal_id is None
            or 'sg_%d' % r['result']['subgoal']==subgoal_id])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir',type=Path)
    parser.add_argument('--episode',default='val_seen-1')
    parser.add_argument('--subgoal-id')
    parser.add_argument('--call-id')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    directory=args.output_dir/'traces'/args.episode
    if not (directory/'summary.json').is_file():parser.error('Episode summary not found: '+str(directory))
    result=json.dumps(diagnose(directory,args.subgoal_id,args.call_id),indent=2,ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(result+'\n')
    else:print(result)


if __name__=='__main__':main()
