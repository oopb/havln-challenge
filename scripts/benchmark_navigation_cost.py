"""Compare v34 request text with a saved diagnostic ZIP without model calls.

Usage:
    python scripts/benchmark_navigation_cost.py /workspace/agent-runs/phase2-test-6-diagnostic.zip
This is a JSON character count, NOT a tokenizer or proof of provider savings.
"""
import argparse
import json
import pathlib
import zipfile
from havln_agent.contexts import _cost_optimized_projection


def compact_size(value):
    return len(json.dumps(value,sort_keys=True,ensure_ascii=False,
                          separators=(',',':')))


def run(path):
    by_ep={}
    with zipfile.ZipFile(path) as bundle:
        for name in bundle.namelist():
            if not name.endswith('/api.jsonl') or '/traces/' not in name:
                continue
            ep=name.split('/traces/',1)[1].split('/',1)[0]
            rows=[]
            for line in bundle.read(name).splitlines():
                call=json.loads(line)
                if call.get('role')!='navigation_agent' or not call.get('success'):
                    continue
                user=call['messages'][1]['content']
                text=user[0]['text'] if isinstance(user,list) else user
                context=json.loads(text)
                # Logged inputs already passed the old compact_projection
                # and absolute-to-relative angle conversion. Apply ONLY the
                # additional cost projection, without mutating the log.
                before=compact_size(context)
                after=compact_size(_cost_optimized_projection(context))
                rows.append((before,after))
            if rows:
                by_ep[ep]=dict(calls=len(rows),
                    before_chars=sum(a for a,b in rows),
                    projected_chars=sum(b for a,b in rows),
                    reduction_pct=round(100*(1-sum(b for a,b in rows)/
                                             sum(a for a,b in rows)),2))
    total_before=sum(x['before_chars'] for x in by_ep.values())
    total_after=sum(x['projected_chars'] for x in by_ep.values())
    return dict(episodes=by_ep,overall=dict(
        calls=sum(x['calls'] for x in by_ep.values()),
        before_chars=total_before,projected_chars=total_after,
        reduction_pct=round(100*(1-total_after/total_before),2)
                      if total_before else None),
        note='Character reduction only; actual billed tokens and cache hits '
             'require a new live evaluation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('diagnostic_zip',type=pathlib.Path)
    args=parser.parse_args()
    print(json.dumps(run(args.diagnostic_zip),indent=2,ensure_ascii=False))
