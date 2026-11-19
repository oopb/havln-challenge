"""Launch one fresh Codex CLI session against one fresh Habitat MCP episode.

Host-side script; requires codex, Docker access and the existing havln-cma
container. It does not change the cloud/hybrid API agent configuration.
"""
import argparse
from datetime import datetime, timezone
import json
import os
import signal
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

# Executing this script by path does not always place the checkout root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from havln_agent.codex_usage import (compare_rate_limits, format_quota_summary, read_rate_limits,
                                     summarize_token_usage)


PROMPT = """You are the sole navigation agent for ONE HA-VLN episode.
Use ONLY the havln MCP tools: observe and step. Begin by calling observe.
Observe returns RGB; Depth appears for a near obstacle or after a detected forward stall. Depth is evidence, never clearance authorization.
Navigate using visual evidence and the six permitted actions. On open paths, use longer MOVE_FORWARD batches; near people, stairs, doorways or uncertain passages use short movements with fresh observations.
step may report NO_VISUAL_PROGRESS (verified stall), MOTION_UNCERTAIN (repeated weak motion), or MOTION_MONITOR_ERROR (the forward step WAS executed, but the local monitor failed). These skip remaining batch actions; use the executed receipt, re-observe, and decide the next move. Never retry the same failed approach without new evidence.
When searching, turn in short batches and observe. If motion_hint reports REPEATED_SCAN, compare the supplied historical views and their angles relative to your CURRENT heading; revisit a promising opening rather than circling again. Do not force forward into unsafe space.
observe pitch_deg is commanded tilt from the initial forward view (negative=down). For uncertain near-foot clearance, LOOK_DOWN then observe; restore pitch near zero with LOOK_UP before forward motion. Avoid repeated floor-only inspections without new evidence.
Follow landmarks in order: interpret left/right at the place specified in the route, not at a later exploratory viewpoint.
People can move between views: keep a safe gap, avoid their immediate path, and use stable surroundings to identify locations.
Near a final landmark use short movements; when the requested stopping place is visibly confirmed, STOP rather than moving solely to improve image alignment.
If Depth appears: dark=near, bright=far, magenta=invalid. Its distance and orientation may be outdated after turning.
When the goal has been reached, execute STOP using step.
Do not use shell commands, read source files, inspect hidden simulator state,
look up reference trajectories, or attempt to bypass the action interface.
Do not rely on state from any previous episode. A STOP is not proof of success.
"""

START_PROMPT = "Begin navigation by calling havln.observe()."

# Import the same public controller that the MCP server imports, using the
# container's interpreter rather than the (often newer) host/CI Python.
MCP_IMPORT_PROBE = ("import sys; "
                    "from havln_agent.codex_episode import CodexEpisode; "
                    "print('MCP controller import OK, Python', sys.version.split()[0])")


def require_mcp_runtime(args):
    """Fail before paid Codex inference with the real container traceback."""
    command = [args.docker_bin, "exec", "-w", "/toolkit", args.container,
               "python", "-c", MCP_IMPORT_PROBE]
    try:
        result = subprocess.run(command, capture_output=True,
                                text=True, timeout=45, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Cannot run MCP import probe: " + str(error)) from error
    if result.returncode != 0:
        raise RuntimeError(
            "Habitat MCP Python import failed inside container; check installed "
            "versions and source mount:\n" +
            (result.stderr or result.stdout or "No stderr returned")[-2400:])
    return result.stdout.strip()


def load_episode_instruction(args, split):
    """Read only the public dataset instruction before launching Codex."""
    program = (
        "import gzip,json,sys;"
        "p=sys.argv[1]+'/HA-R2R/'+sys.argv[2]+'/'+sys.argv[2]+'.json.gz';"
        "rows=json.load(gzip.open(p,'rt',encoding='utf-8'))['episodes'];"
        "matches=[e['instruction']['instruction_text'] for e in rows "
        "if str(e['episode_id'])==sys.argv[3]];"
        "assert len(matches)==1 and isinstance(matches[0],str) and matches[0].strip(),"
        "'Expected one nonempty instruction';"
        "print(json.dumps(matches[0],ensure_ascii=False))"
    )
    completed = subprocess.run(
        [args.docker_bin, "exec", "-w", "/toolkit", args.container,
         "python", "-c", program, args.data_root, split, str(args.episode_id)],
        capture_output=True, text=True, check=True, timeout=90)
    instruction = json.loads(completed.stdout)
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Dataset instruction is empty or malformed")
    return instruction


def build_command(args, final_message, system_prompt_file):
    split = args.split or ("test" if args.phase == "phase2" else "val_seen")
    server_args = [
        args.docker_bin, "exec", "-i", "-w", "/toolkit", args.container,
        "python", "-u", "/toolkit/scripts/havln_mcp_server.py",
        "--phase", args.phase, "--split", split,
        "--episode-id", str(args.episode_id),
        "--max-steps", str(args.max_steps),
        "--gpu-id", str(args.gpu_id),
        "--seed", str(args.seed),
        "--output-dir", args.output_dir,
        "--data-root", getattr(args, "data_root", "/data/havln2"),
    ]
    if getattr(args, "batch_child", False):
        server_args.append("--batch-child")
    # JSON arrays/strings are valid TOML inline config values. Unlike
    # permanent MCP registration, these overrides apply only to this run.
    overrides = [
        'mcp_servers.havln.command=' + json.dumps(server_args[0]),
        'mcp_servers.havln.args=' + json.dumps(server_args[1:]),
        'mcp_servers.havln.required=true',
        'mcp_servers.havln.enabled_tools=["observe","step"]',
        'mcp_servers.havln.default_tools_approval_mode="approve"',
        'mcp_servers.havln.supports_parallel_tool_calls=false',
        'mcp_servers.havln.startup_timeout_sec=180',
        'mcp_servers.havln.tool_timeout_sec=180',
        'model_reasoning_effort=' + json.dumps(args.reasoning_effort),
        # Actual system-level model instructions, not an ordinary user prompt.
        'model_instructions_file=' + json.dumps(str(system_prompt_file)),
    ]
    command = [args.codex_bin]
    for override in overrides:
        command.extend(["-c", override])
    command.extend(["exec", "--ephemeral", "--ignore-user-config",
                    "--skip-git-repo-check",
                    "--sandbox", "read-only", "--json",
                    "--output-last-message", str(final_message)])
    if args.model:
        command.extend(["--model", args.model])
    command.append(START_PROMPT)
    return command


def preflight(args, split):
    """Read-only host/container checks; never start a Codex inference session."""
    probes = [
        ("Codex CLI", [args.codex_bin, "--version"]),
        ("Codex authentication", [args.codex_bin, "login", "status"]),
        ("Docker access", [args.docker_bin, "exec", args.container,
                           "test", "-f", "/toolkit/scripts/havln_mcp_server.py"]),
        ("Habitat 0.1.7", [args.docker_bin, "exec", args.container, "python",
                           "-c", "import habitat, habitat_sim; assert habitat.__version__ == habitat_sim.__version__ == '0.1.7'"]),
        ("MCP Python controller import", [args.docker_bin, "exec", "-w",
                                         "/toolkit", args.container, "python",
                                         "-c", MCP_IMPORT_PROBE]),
        ("Episode dataset ID", [args.docker_bin, "exec", args.container,
                                "python", "-c",
                                ("import gzip, json, sys; "
                                 "path='/data/havln2/HA-R2R/'+sys.argv[1]+'/'+sys.argv[1]+'.json.gz'; "
                                 "data=json.load(gzip.open(path,'rt')); "
                                 "assert any(str(e['episode_id'])==sys.argv[2] "
                                 "for e in data['episodes']), 'episode missing from dataset'"),
                                split, str(args.episode_id)]),
    ]
    ok = True
    for title, command in probes:
        try:
            completed = subprocess.run(command, capture_output=True,
                                       text=True, timeout=45, check=False)
            passed = completed.returncode == 0
            note = (completed.stdout or completed.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            passed, note = False, str(error)
        print(("PASS " if passed else "FAIL ") + title +
              (": " + note[:500] if note else ""))
        ok = ok and passed
    return 0 if ok else 2


def summarize_codex_events(path):
    """Count events without assuming a fixed Codex JSON event protocol."""
    counts = {}
    malformed = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    malformed += 1
                    continue
                kind = event.get("type", "UNKNOWN")
                if not isinstance(kind, str):
                    kind = "UNKNOWN"
                counts[kind] = counts.get(kind, 0) + 1
            except ValueError:
                malformed += 1
    return {"events_by_type": counts, "malformed_events": malformed}


def run_codex_cli(command, workdir, stdout, stderr, timeout_seconds):
    """Bound one CLI process tree, including its stdio Docker MCP child."""
    with subprocess.Popen(command, cwd=workdir, stdout=stdout, stderr=stderr,
                          start_new_session=True) as process:
        try:
            return process.wait(timeout=timeout_seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            interrupted = isinstance(error, KeyboardInterrupt)
            reason = ("interrupted by user" if interrupted else
                      "exceeded wall-time limit: " + str(timeout_seconds) +
                      " seconds")
            stderr.write("Codex episode " + reason + "\n")
            stderr.flush()
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            return 130 if interrupted else 124


def require_fresh_container_output(args):
    """Fail before paid inference rather than append to another run."""
    # Batch children share the run root but must never reuse an episode trace.
    target = (args.output_dir.rstrip("/") + "/traces/" +
              (args.split or ("test" if args.phase == "phase2" else "val_seen")) +
              "-" + str(args.episode_id)) if getattr(args, "batch_child", False) else args.output_dir
    probe = subprocess.run(
        [args.docker_bin, "exec", args.container, "test", "-e", target],
        text=True, capture_output=True, check=False, timeout=30)
    if probe.returncode == 0:
        raise ValueError("Output directory already exists in container: " +
                         target + ". Choose a new --output-dir.")
    if probe.returncode != 1 or probe.stderr.strip():
        raise RuntimeError("Cannot verify new output directory via Docker: " +
                           (probe.stderr.strip() or str(probe.returncode)))


def collect_habitat_summary(args, trace):
    """Return a checked simulator summary, or an explicit diagnostics warning."""
    try:
        exists = subprocess.run(
            [args.docker_bin, "exec", args.container, "test", "-d", trace],
            text=True, capture_output=True, timeout=30, check=False)
        if exists.returncode != 0:
            return False, None, "No Habitat trace directory"
        result = subprocess.run(
            [args.docker_bin, "exec", args.container, "cat", trace + "/summary.json"],
            text=True, capture_output=True, timeout=30, check=False)
        if result.returncode != 0:
            return True, None, "No finalized Habitat summary"
        summary = json.loads(result.stdout)
        if not isinstance(summary, dict) or not isinstance(summary.get("completed"), bool):
            return True, None, "Malformed Habitat summary"
        return True, summary, None
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return False, None, str(error)


def collect_codex_audit(args, staged_dir, trace_exists, trace):
    """Copy the ONE staged Codex directory to the requested container root.

    A partial Habitat trace still receives its CLI history. On MCP startup
    failure, use output-dir/codex and preserve the startup error in the ZIP.
    """
    parent = trace if trace_exists or getattr(args, "batch_child", False) else args.output_dir
    subprocess.run([args.docker_bin, "exec", args.container,
                    "mkdir", "-p", parent], check=True, timeout=60)
    subprocess.run([args.docker_bin, "cp", str(staged_dir),
                    args.container + ":" + parent], check=True, timeout=120)
    if not getattr(args, "batch_child", False):
        subprocess.run([args.docker_bin, "cp", str(staged_dir / "run_status.json"),
                        args.container + ":" + args.output_dir + "/run_status.json"],
                       check=True, timeout=60)


def archive_codex_run(args):
    """Use the EXACT automatic archive rule of the original API exporter.

    Default single-episode policy: all frames. Place timestamped archives in
    <output-dir parent>/diagnostic-archives, never inside the run directory.
    """
    frames = "all" if args.archive_frames == "auto" else args.archive_frames
    program = (
        "import sys; from havln_agent.export import auto_archive; "
        "path = auto_archive(sys.argv[1], "
        "archive_dir=(sys.argv[3] if len(sys.argv)>3 else None), "
        "frames=sys.argv[2]); "
        "assert path is not None, 'No diagnostic evidence to archive'; "
        "print('CODEX_ARCHIVE_PATH=' + str(path))"
    )
    extra = ([args.archive_dir] if getattr(args, "archive_dir", None) else [])
    result = subprocess.run(
        [args.docker_bin, "exec", "-w", "/toolkit", args.container,
         "python", "-c", program, args.output_dir, frames] + extra,
        check=True, capture_output=True, text=True, timeout=600)
    archive = next(
        (line.partition("=")[2] for line in result.stdout.splitlines()
         if line.startswith("CODEX_ARCHIVE_PATH=")), None)
    if not archive:
        raise RuntimeError("Archive created but expected path was not reported")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("phase1", "phase2"), default="phase2")
    parser.add_argument("--split", choices=("val_seen", "val_unseen", "test"))
    parser.add_argument("--episode-id", required=True, type=int)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=int, default=2400,
                        help="Max total Codex session wall time; 124 on timeout")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="gpt-6-astra",
                        help="Must be available to your Codex CLI account")
    parser.add_argument("--reasoning-effort", default="medium",
                        choices=("low", "medium", "high", "xhigh"))
    parser.add_argument("--container", default="havln-cma")
    parser.add_argument("--data-root", default="/data/havln2",
                        help="Container-side HA-VLN dataset root")
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--output-dir", required=True,
                        help="Single fresh absolute CONTAINER output directory")
    parser.add_argument("--archive-frames", choices=("auto", "critical", "all", "none"),
                        default="auto",
                        help="Same as API exporter: auto=all frames for one episode")
    parser.add_argument("--archive-dir",
                        help="Optional container archive directory; defaults to sibling diagnostic-archives")
    parser.add_argument("--no-auto-archive", action="store_true",
                        help="Keep run traces only; do not create diagnostic ZIP")
    parser.add_argument("--batch-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--preflight", action="store_true",
                        help="Check Codex/Docker/Habitat without model inference or output files")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 500 or args.episode_id < 1:
        parser.error("episode-id must be positive and max-steps must be 1..500")
    if not 1 <= args.timeout_seconds <= 14400:
        parser.error("timeout-seconds must be 1..14400")
    split = args.split or ("test" if args.phase == "phase2" else "val_seen")
    if (args.phase == "phase2" and split != "test") or (
            args.phase == "phase1" and split == "test"):
        parser.error("phase2 requires test; phase1 requires val_seen/val_unseen")
    if not Path(args.output_dir).is_absolute() or args.output_dir == "/":
        parser.error("--output-dir must be a dedicated absolute container path")
    if args.preflight:
        return preflight(args, split)
    if not args.batch_child:
        try:
            print(require_mcp_runtime(args), flush=True)
        except RuntimeError as error:
            parser.error(str(error))
    try:
        require_fresh_container_output(args)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.error(str(error))

    try:
        instruction = load_episode_instruction(args, split)
    except (OSError, ValueError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as error:
        parser.error("Cannot read public episode instruction: " + str(error))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace = args.output_dir.rstrip("/") + "/traces/" + split + "-" + str(args.episode_id)
    # Temporary host files are removed AFTER successful container collection.
    # If Docker disappears, preserve them as an exceptional recovery fallback.
    staging_root = Path(tempfile.mkdtemp(prefix="havln-codex-audit-"))
    staged_dir = staging_root / "codex"
    staged_dir.mkdir()
    system_prompt = (PROMPT.rstrip() + "\n\nRoute instruction for this episode:\n" +
                     instruction + "\n")
    system_prompt_file = staged_dir / "prompt.txt"
    system_prompt_file.write_text(system_prompt, encoding="utf-8")
    command = build_command(args, staged_dir / "final_message.txt",
                            system_prompt_file)
    (staged_dir / "command.json").write_text(
        json.dumps({"started_utc": stamp, "command": command,
                    "episode_id": args.episode_id, "phase": args.phase,
                    "split": split, "container_output_dir": args.output_dir},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        # Read-only quota queries are best-effort and never gate model execution.
        quota_before = (read_rate_limits(args.codex_bin)
                        if not args.batch_child else
                        {"status": "unavailable", "reason": "use batch-level quota snapshots"})
        with tempfile.TemporaryDirectory(prefix="havln-codex-session-") as workdir:
            with (staged_dir / "codex_events.jsonl").open("w", encoding="utf-8") as stdout, \
                 (staged_dir / "codex_stderr.log").open("w", encoding="utf-8") as stderr:
                try:
                    exit_code = run_codex_cli(
                        command, workdir, stdout, stderr, args.timeout_seconds)
                except OSError as error:
                    stderr.write("Failed to launch Codex: " + str(error) + "\n")
                    exit_code = 127

        quota_after = (read_rate_limits(args.codex_bin)
                       if not args.batch_child else
                       {"status": "unavailable", "reason": "use batch-level quota snapshots"})
        tokens = summarize_token_usage(staged_dir / "codex_events.jsonl")
        # Interrupted or failed sessions may have consumed additional tokens
        # without a turn.completed usage receipt. Do not claim full coverage.
        if exit_code != 0 and tokens["status"] == "complete":
            tokens["status"] = "partial"
        usage = {
            "schema_version": 1,
            "episode_id": str(args.episode_id),
            "phase": args.phase,
            "split": split,
            "codex_exit_code": exit_code,
            "tokens": tokens,
            "quota": {
                "before": quota_before,
                "after": quota_after,
                "delta": compare_rate_limits(quota_before, quota_after),
            },
        }
        (staged_dir / "codex_usage.json").write_text(
            json.dumps(usage, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8")

        trace_exists, summary, warning = collect_habitat_summary(args, trace)
        run_status = {
            "prompt_kind": "model_instructions_file",
            "codex_usage_file": (("traces/" + split + "-" + str(args.episode_id) +
                                  "/codex/codex_usage.json")
                                 if trace_exists or args.batch_child else
                                 "codex/codex_usage.json"),
            "agent_path": "codex_mcp_v1",
            "codex_exit_code": exit_code,
            "wall_time_limit_seconds": args.timeout_seconds,
            "timed_out": exit_code == 124,
            "container_trace_dir": trace,
            "codex_events": summarize_codex_events(
                staged_dir / "codex_events.jsonl"),
            "habitat_summary": summary,
            "export_valid": bool(summary and summary.get("completed")),
        }
        if warning:
            run_status["collection_warning"] = warning
        run_status["success"] = bool(
            exit_code == 0 and run_status["export_valid"])
        (staged_dir / "run_status.json").write_text(
            json.dumps(run_status, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

        try:
            collect_codex_audit(args, staged_dir, trace_exists, trace)
        except (OSError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as error:
            print("Unable to store Codex logs in --output-dir: " + str(error),
                  file=sys.stderr)
            print("Recovery logs retained at " + str(staged_dir), file=sys.stderr)
            return 2

        # The single selected output directory is now the durable copy.
        shutil.rmtree(staging_root)
        archive = None
        archive_failed = False
        if not args.no_auto_archive and not args.batch_child:
            try:
                archive = archive_codex_run(args)
            except (OSError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired) as error:
                print("Automatic diagnostic archive failed: " + str(error),
                      file=sys.stderr)
                print("Raw traces and Codex logs remain in " + args.output_dir,
                      file=sys.stderr)
                archive_failed = True

        if summary:
            print("Habitat episode:", summary["stop_reason"],
                  "| actions:", summary["steps"],
                  "| legal action export:", summary["completed"])
        else:
            print("Habitat episode has no finalized summary: " +
                  str(warning or "unknown"), file=sys.stderr)
        print("Output:", args.output_dir)
        if archive:
            print("Diagnostic ZIP:", archive,
                  "(frames=" + args.archive_frames + ")")
        else:
            print("Diagnostic ZIP: disabled (raw traces retained)")
        print("Codex exit code:", exit_code)
        print("Codex token usage:", tokens["status"],
              "| input:", tokens["input_tokens"],
              "| cached:", tokens["cached_input_tokens"],
              "| output:", tokens["output_tokens"],
              "| total:", tokens["total_tokens"])
        if not args.batch_child:
            for line in format_quota_summary(usage["quota"]):
                print(line)
        return 0 if run_status["success"] and not archive_failed else 2
    finally:
        # If an unexpected error occurs before collection, do not silently
        # delete the sole copy of the Codex model/tool transcript.
        if staging_root.exists():
            print("Host recovery audit: " + str(staged_dir), file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
