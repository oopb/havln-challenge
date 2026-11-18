"""Run multiple independent Codex/MCP HA-VLN episodes into ONE batch output.

No Codex conversation or Habitat environment is shared across episodes.
--batch is processes per GPU (like the API runner), NOT a single shared model
session. Every result stays under --output-dir/traces/<split>-<id>, and only the
parent may create a timestamped sibling diagnostic ZIP.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from havln_agent.codex_batch import select_ids
from havln_agent.codex_usage import compare_rate_limits, read_rate_limits
from scripts.run_codex_episode import archive_codex_run

HERE = Path(__file__).resolve().parent


def _docker(args, command, **options):
    return subprocess.run([args.docker_bin, "exec", "-w", "/toolkit",
                           args.container, *command], **options)


def discover_ids(args, split):
    """Read real dataset IDs once, validating Phase 2 official coverage."""
    program = (
        "import json,sys;from havln_agent.codex_batch import dataset_ids;"
        "print(json.dumps(dataset_ids(sys.argv[1],sys.argv[2])))"
    )
    result = _docker(args, ["python", "-c", program, args.data_root, split],
                     check=True, capture_output=True, text=True, timeout=120)
    return json.loads(result.stdout)


def validate_options(parser, args):
    if not 1 <= args.max_steps <= 500:
        parser.error("--max-steps must be in 1..500")
    if not 1 <= args.timeout_seconds <= 14400:
        parser.error("--timeout-seconds must be in 1..14400")
    if not 1 <= args.batch <= 32:
        parser.error("--batch must be 1..32 processes per GPU")
    if not args.gpu_ids or len(set(args.gpu_ids)) != len(args.gpu_ids) or min(args.gpu_ids) < 0:
        parser.error("--gpu-ids must be distinct nonnegative integers")
    if not Path(args.output_dir).is_absolute() or args.output_dir == "/":
        parser.error("--output-dir must be a new absolute container path")
    if args.archive_dir and not Path(args.archive_dir).is_absolute():
        parser.error("--archive-dir must be an absolute container path")
    split = args.split or ("test" if args.phase == "phase2" else "val_seen")
    if args.phase == "phase2" and split != "test":
        parser.error("Phase 2 supports only test")
    if args.phase == "phase1" and split not in ("val_seen", "val_unseen"):
        parser.error("Phase 1 supports val_seen or val_unseen")
    return split


def preflight(args):
    probes = [
        ("Codex installed", [args.codex_bin, "--version"]),
        ("Codex logged in", [args.codex_bin, "login", "status"]),
        ("Habitat MCP server", [args.docker_bin, "exec", args.container,
                               "test", "-f", "/toolkit/scripts/havln_mcp_server.py"]),
        ("Batch summarizer", [args.docker_bin, "exec", args.container,
                              "test", "-f", "/toolkit/havln_agent/codex_batch.py"]),
    ]
    for name, command in probes:
        result = subprocess.run(command, capture_output=True,
                                text=True, timeout=45, check=False)
        if result.returncode:
            raise RuntimeError(name + " failed: " +
                               (result.stderr or result.stdout)[:300])
        print("PASS", name, flush=True)


def reserve_output_directory(args):
    """Atomic exclusive mkdir inside the container before any paid inference."""
    program = (
        "import pathlib,sys;root=pathlib.Path(sys.argv[1]);"
        "root.parent.mkdir(parents=True,exist_ok=True);root.mkdir()"
    )
    _docker(args, ["python", "-c", program, args.output_dir],
            check=True, capture_output=True, text=True, timeout=45)


def copy_json(args, destination, payload, tempdir):
    filename = Path(tempdir) / (Path(destination).name + "-" +
                                str(time.monotonic_ns()) + ".json")
    filename.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    subprocess.run([args.docker_bin, "cp", str(filename),
                    args.container + ":" + destination],
                   check=True, capture_output=True, text=True, timeout=90)


def build_child_command(args, split, episode_id, gpu):
    return [
        sys.executable, str(HERE / "run_codex_episode.py"),
        "--phase", args.phase, "--split", split,
        "--episode-id", str(episode_id),
        "--max-steps", str(args.max_steps),
        "--timeout-seconds", str(args.timeout_seconds),
        "--gpu-id", str(gpu), "--seed", str(args.seed),
        "--model", args.model, "--reasoning-effort", args.reasoning_effort,
        "--docker-bin", args.docker_bin, "--codex-bin", args.codex_bin,
        "--container", args.container, "--output-dir", args.output_dir,
        "--data-root", args.data_root,
        "--batch-child", "--no-auto-archive",
    ]


def run_batch(args, split, ids, tempdir):
    """Use threads only to supervise fully isolated child processes."""
    locks = {gpu: threading.BoundedSemaphore(args.batch) for gpu in args.gpu_ids}
    processes = {}
    guard = threading.Lock()
    aborted = threading.Event()
    completed = {}

    def execute(index, episode_id):
        gpu = args.gpu_ids[index % len(args.gpu_ids)]
        log_name = "worker-" + split + "-" + str(episode_id) + ".log"
        local_log = Path(tempdir) / log_name
        returncode = 127
        started = time.monotonic()
        with locks[gpu]:
            if aborted.is_set():
                return dict(returncode=130, log_file=log_name,
                            gpu_id=gpu, aborted_before_start=True)
            with local_log.open("w", encoding="utf-8") as log:
                log.write("Episode " + str(episode_id) + " | GPU " + str(gpu) + "\n")
                log.flush()
                try:
                    with subprocess.Popen(
                            build_child_command(args, split, episode_id, gpu),
                            stdout=log, stderr=subprocess.STDOUT) as process:
                        with guard:
                            processes[episode_id] = process
                        returncode = process.wait()
                except OSError as error:
                    log.write("Unable to launch child: " + str(error) + "\n")
                finally:
                    with guard:
                        processes.pop(episode_id, None)
                    log.write("Worker exit: " + str(returncode) + "\n")
            try:
                subprocess.run([args.docker_bin, "cp", str(local_log),
                                args.container + ":" + args.output_dir +
                                "/" + log_name], check=True, timeout=90,
                               capture_output=True)
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                print("WARN: cannot save worker log for", episode_id, str(error),
                      flush=True)
            return dict(returncode=returncode, log_file=log_name,
                        gpu_id=gpu, seconds=round(time.monotonic() - started, 3))

    workers = args.batch * len(args.gpu_ids)
    print("Codex batch:", len(ids), "episodes |", workers,
          "maximum simultaneous Codex sessions | GPU IDs:", args.gpu_ids,
          flush=True)
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {executor.submit(execute, i, ident): ident
                   for i, ident in enumerate(ids)}
        try:
            for future in as_completed(futures):
                ident = futures[future]
                try:
                    outcome = future.result()
                except Exception as error:
                    outcome = dict(returncode=127, error=str(error)[:300])
                completed[str(ident)] = outcome
                print("Episode", ident, ":", "OK" if outcome["returncode"] == 0
                      else "FAILED", "|", len(completed), "/", len(ids),
                      "| GPU", outcome.get("gpu_id"), flush=True)
        except KeyboardInterrupt:
            aborted.set()
            print("Interrupt received: stopping active Codex episode processes...",
                  file=sys.stderr, flush=True)
            for future in futures:
                future.cancel()
            with guard:
                live = list(processes.values())
            # SIGINT is delivered to the Python child, which in turn
            # terminates its independent Codex + docker exec process group.
            for process in live:
                if process.poll() is None:
                    try:
                        process.send_signal(signal.SIGINT)
                    except ProcessLookupError:
                        pass
            for future, ident in futures.items():
                if future.cancelled():
                    completed[str(ident)] = dict(returncode=130, cancelled=True)
                else:
                    try:
                        completed[str(ident)] = future.result(timeout=30)
                    except Exception:
                        completed[str(ident)] = dict(returncode=130,
                                                     interrupted=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("phase1", "phase2"), default="phase2")
    parser.add_argument("--split", choices=("test", "val_seen", "val_unseen"))
    parser.add_argument("--episode-start", type=int, default=1,
                        help="Inclusive ORIGINAL dataset episode ID")
    parser.add_argument("--episode-limit", type=int, default=1,
                        help="Number of DISTINCT episodes to evaluate")
    parser.add_argument("--random-episodes", action="store_true",
                        help="Sample unique original IDs from [episode-start, split end]")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random selection seed and Habitat seed")
    parser.add_argument("--batch", type=int, default=1,
                        help="Concurrent Codex/Habitat subprocesses PER GPU")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", default="medium",
                        choices=("low", "medium", "high", "xhigh"))
    parser.add_argument("--container", default="havln-cma")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--data-root", default="/data/havln2",
                        help="Dataset location INSIDE container")
    parser.add_argument("--output-dir", required=True,
                        help="Fresh absolute container run root, e.g. /workspace/agent-runs/codex-batch-1")
    parser.add_argument("--archive-dir", default=None,
                        help="Container archive root (default: output-dir sibling diagnostic-archives)")
    parser.add_argument("--archive-frames",
                        choices=("auto", "all", "critical", "none"),
                        default="auto",
                        help="auto: all for one episode, critical for multiple")
    parser.add_argument("--no-auto-archive", action="store_true",
                        help="Keep raw run files without creating a diagnostic ZIP")
    parser.add_argument("--preflight", action="store_true",
                        help="Check prerequisites and episode selection only; no paid model calls")
    args = parser.parse_args(argv)
    split = validate_options(parser, args)
    try:
        preflight(args)
        candidates = discover_ids(args, split)
        selected = select_ids(candidates, args.episode_start,
                              args.episode_limit, args.random_episodes, args.seed)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as error:
        parser.error("Batch preflight failed: " + str(error))
    print("Episode selection:", "random" if args.random_episodes else "sequential",
          "| count:", len(selected), "| IDs:", selected, flush=True)
    if args.preflight:
        return 0

    with tempfile.TemporaryDirectory(prefix="havln-codex-batch-") as tempdir:
        try:
            reserve_output_directory(args)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            parser.error("Output directory is not fresh or cannot be created: " + str(error))
        plan = {
            "architecture": "codex_batch_v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "phase": args.phase, "split": split,
            "episode_start": args.episode_start,
            "episode_limit": args.episode_limit,
            "random_episodes": args.random_episodes,
            "selection_seed": args.seed, "selected_episode_ids": selected,
            "batch_per_gpu": args.batch, "gpu_ids": args.gpu_ids,
            "max_steps": args.max_steps, "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "auto_archive": not args.no_auto_archive,
        }
        copy_json(args, args.output_dir + "/batch_plan.json", plan, tempdir)
        copy_json(args, args.output_dir + "/run_status.json",
                  {"agent_path": "codex_batch_v1", "success": False,
                   "status": "in_progress", "official_submission": False}, tempdir)
        before = read_rate_limits(args.codex_bin)
        worker_results = {}
        try:
            worker_results = run_batch(args, split, selected, tempdir)
        finally:
            after = read_rate_limits(args.codex_bin)
            quota = {"before": before, "after": after,
                     "delta": compare_rate_limits(before, after),
                     "note": ("Account-wide overlapping batch window; "
                              "never sum individual workers' rate limit deltas.")}
            copy_json(args, args.output_dir + "/batch_quota.json", quota, tempdir)
            copy_json(args, args.output_dir + "/batch_worker_results.json",
                      {"episodes": worker_results}, tempdir)

        summary = None
        try:
            result = _docker(args, ["python", "-m", "havln_agent.codex_batch",
                                   "--output-dir", args.output_dir],
                             check=True, capture_output=True, text=True, timeout=120)
            summary = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as error:
            print("ERROR: batch aggregation failed:", error, file=sys.stderr)
        archive = None
        if not args.no_auto_archive:
            archive_frames = (("all" if len(selected) == 1 else "critical")
                              if args.archive_frames == "auto" else args.archive_frames)
            archive_args = argparse.Namespace(
                docker_bin=args.docker_bin, container=args.container,
                output_dir=args.output_dir, archive_frames=archive_frames,
                archive_dir=args.archive_dir)
            try:
                archive = archive_codex_run(archive_args)
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired) as error:
                print("ERROR: diagnostic archive failed:", error, file=sys.stderr)
        print("Run output:", args.output_dir)
        print("Diagnostic ZIP:", archive if archive else
              ("disabled" if args.no_auto_archive else "FAILED"))
        if summary:
            print("Episode results:", summary["completed_episodes"], "/",
                  summary["episode_count"], "valid action exports")
            print("Codex total tokens:", summary["tokens"]["total_tokens"],
                  "| status:", summary["tokens"]["status"])
        return 0 if summary and summary["success"] and (
            args.no_auto_archive or archive) else 2


if __name__ == "__main__":
    sys.exit(main())
