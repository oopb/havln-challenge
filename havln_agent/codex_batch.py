"""Stdlib-only batch selection, consolidation and legal diagnostic exports.

This module runs inside the Habitat container AFTER independent Codex workers
exit. It never opens Habitat or calls a model. The batch root owns one manifest,
one aggregate status and at most one external diagnostic ZIP.
"""
import argparse
import gzip
import json
from pathlib import Path
import random

from .submission_contract import atomic_json, check_rows

PHASE2_EPISODES = 3408


def select_ids(ids, episode_start, episode_limit, random_episodes=False, seed=0):
    """Select unique *original* dataset IDs, not zero-based dataset offsets."""
    if type(episode_start) is not int or episode_start < 1:
        raise ValueError("episode-start must be a positive original ID")
    if type(episode_limit) is not int or episode_limit < 1:
        raise ValueError("episode-limit must be a positive integer")
    available = sorted(set(int(n) for n in ids if int(n) >= episode_start))
    if len(available) != len([n for n in ids if int(n) >= episode_start]):
        raise ValueError("Dataset contains duplicate episode IDs")
    if episode_limit > len(available):
        raise ValueError("Requested episodes exceed available IDs from episode-start")
    if random_episodes:
        return sorted(random.Random(seed).sample(available, episode_limit))
    return available[:episode_limit]


def dataset_ids(data_root, split):
    if split not in ("test", "val_seen", "val_unseen"):
        raise ValueError("unsupported split")
    path = Path(data_root) / "HA-R2R" / split / (split + ".json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as reader:
        episodes = json.load(reader)["episodes"]
    ids = [int(episode["episode_id"]) for episode in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate episode ids in dataset")
    if split == "test" and set(ids) != set(range(1, PHASE2_EPISODES + 1)):
        raise ValueError("Phase 2 test IDs must be exactly 1..3408")
    return ids


def _load(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def aggregate(root):
    """Aggregate all requested workers without fabricating missing consumption.

    Never sum overlapping account quota windows: only the host batch runner
    has reliable non-overlapping before/after snapshots.
    """
    root = Path(root)
    plan = _load(root / "batch_plan.json")
    worker_data = _load(root / "batch_worker_results.json")
    if plan.get("architecture") != "codex_batch_v1":
        raise ValueError("Missing or invalid batch plan")
    ids = plan.get("selected_episode_ids")
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise ValueError("Missing or duplicate selected episode IDs")
    split = plan["split"]
    worker_results = worker_data.get("episodes") or {}
    rows = []
    episodes = []
    total = dict(input_tokens=0, cached_input_tokens=0, output_tokens=0,
                 reasoning_output_tokens=0, total_tokens=0)
    token_missing = False
    token_partial = False
    for ident in ids:
        episode_id = str(int(ident))
        path = root / "traces" / (split + "-" + episode_id)
        status = _load(path / "codex" / "run_status.json")
        usage = _load(path / "codex" / "codex_usage.json").get("tokens") or {}
        summary = _load(path / "summary.json")
        result = _load(path / "result.json")
        worker = worker_results.get(episode_id, {})
        row = result.get("row")
        valid = False
        if isinstance(row, dict) and row.get("episode_id") == episode_id:
            try:
                check_rows([row], [episode_id])
                valid = bool(summary.get("completed"))
            except (ValueError, AssertionError, TypeError):
                pass
        if valid:
            rows.append(row)
        numbers = {key: usage.get(key) for key in total}
        complete_token = (usage.get("status") == "complete" and
                          all(type(value) is int and value >= 0 for value in numbers.values()))
        if not complete_token:
            token_partial = True
        for key, value in numbers.items():
            if type(value) is int and value >= 0:
                total[key] += value
            else:
                token_missing = True
        episodes.append(dict(episode_id=episode_id,
                             worker_exit_code=worker.get("returncode"),
                             codex_exit_code=status.get("codex_exit_code"),
                             completed=valid, habitat_summary=summary or None,
                             token_status=usage.get("status", "unavailable"),
                             total_tokens=usage.get("total_tokens"),
                             worker_log=worker.get("log_file")))
    expected = [str(int(i)) for i in ids]
    all_valid = len(rows) == len(expected)
    any_worker_failed = any(e["worker_exit_code"] != 0 for e in episodes)
    tokens = {
        "status": ("unavailable" if token_missing and not any(
            e.get("total_tokens") is not None for e in episodes) else
                   "partial" if token_partial or token_missing else "complete"),
        **{key: (None if token_missing and not any(
            e.get("total_tokens") is not None for e in episodes) else value)
           for key, value in total.items()},
        "note": ("Totals cover available per-episode turn.completed records; "
                 "missing/incomplete sessions are not assumed to consume zero. "
                 "Cached input is included in input_tokens."),
    }
    summary = dict(
        schema_version=1, architecture="codex_batch_v1", phase=plan["phase"],
        split=split, selected_episode_ids=expected, episode_count=len(expected),
        completed_episodes=len(rows), failed_episodes=len(expected) - len(rows),
        worker_errors=any_worker_failed, success=all_valid and not any_worker_failed,
        official_submission=False, episodes=episodes, tokens=tokens,
        archive_requested=plan.get("auto_archive", True))
    atomic_json(root / "batch_status.json", summary)
    atomic_json(root / "run_status.json", {
        "agent_path": "codex_batch_v1", "success": summary["success"],
        "completed_episodes": len(rows), "selected_episodes": len(expected),
        "official_submission": False})
    atomic_json(root / "codex_usage.json", {
        "schema_version": 1, "scope": "batch", "selected_episodes": len(expected),
        "tokens": tokens, "quota": _load(root / "batch_quota.json")})
    # Diagnostic rows are valid but PARTIAL runs are not official submissions.
    if rows:
        atomic_json(root / (split + ".diagnostic.json"), {
            "format_version": 1, "split": split, "episodes": rows,
            "selected_ids": expected, "partial": not all_valid,
            "not_an_official_submission": True})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
