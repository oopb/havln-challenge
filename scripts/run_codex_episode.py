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


PROMPT = """You are the sole navigation agent for ONE HA-VLN episode.
Use ONLY the havln MCP tools: observe and step. Begin by calling observe.
The tool will show the task instruction and an actual RGB camera image.
Navigate using visual evidence and the six permitted simulator actions.
Call step with ordered action batches; use observe when the view is needed.
Reason about stairs, crossings, dynamic people and the route instruction.
When the goal has been reached, execute STOP using step.
Do not use shell commands, read source files, inspect hidden simulator state,
look up reference trajectories, or attempt to bypass the action interface.
Do not rely on state from any previous episode. A STOP is not proof of success.
If a tool reports failure, adapt your plan rather than inventing progress.
"""


def build_command(args, final_message):
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
    ]
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
    command.append(PROMPT)
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
        except subprocess.TimeoutExpired:
            stderr.write("Codex episode exceeded wall-time limit: " +
                         str(timeout_seconds) + " seconds\n")
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
            return 124


def require_fresh_container_output(args):
    """Fail before paid inference rather than append to another run."""
    probe = subprocess.run(
        [args.docker_bin, "exec", args.container, "test", "-e", args.output_dir],
        text=True, capture_output=True, check=False, timeout=30)
    if probe.returncode == 0:
        raise ValueError("Output directory already exists in container: " +
                         args.output_dir + ". Choose a new --output-dir.")
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
    parent = trace if trace_exists else args.output_dir
    subprocess.run([args.docker_bin, "exec", args.container,
                    "mkdir", "-p", parent], check=True, timeout=60)
    subprocess.run([args.docker_bin, "cp", str(staged_dir),
                    args.container + ":" + parent], check=True, timeout=120)
    subprocess.run([args.docker_bin, "cp", str(staged_dir / "run_status.json"),
                    args.container + ":" + args.output_dir + "/run_status.json"],
                   check=True, timeout=60)


def archive_codex_run(args):
    """Create one portable diagnostic ZIP *inside* --output-dir."""
    archive = args.output_dir.rstrip("/") + "/diagnostic.zip"
    command = [
        args.docker_bin, "exec", args.container, "python",
        "/toolkit/scripts/pack_agent_diagnostics.py",
        "--output-dir", args.output_dir,
        "--archive", archive,
        "--frames", args.archive_frames,
        "--allow-empty",
    ]
    subprocess.run(command, check=True, timeout=600)
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
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--output-dir", required=True,
                        help="Single fresh absolute CONTAINER output directory")
    parser.add_argument("--archive-frames", choices=("critical", "all", "none"),
                        default="critical",
                        help="Automatic diagnostic ZIP frame policy (default: critical)")
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
    try:
        require_fresh_container_output(args)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.error(str(error))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace = args.output_dir.rstrip("/") + "/traces/" + split + "-" + str(args.episode_id)
    # Temporary host files are removed AFTER successful container collection.
    # If Docker disappears, preserve them as an exceptional recovery fallback.
    staging_root = Path(tempfile.mkdtemp(prefix="havln-codex-audit-"))
    staged_dir = staging_root / "codex"
    staged_dir.mkdir()
    command = build_command(args, staged_dir / "final_message.txt")
    (staged_dir / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    (staged_dir / "command.json").write_text(
        json.dumps({"started_utc": stamp, "command": command,
                    "episode_id": args.episode_id, "phase": args.phase,
                    "split": split, "container_output_dir": args.output_dir},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        with tempfile.TemporaryDirectory(prefix="havln-codex-session-") as workdir:
            with (staged_dir / "codex_events.jsonl").open("w", encoding="utf-8") as stdout, \
                 (staged_dir / "codex_stderr.log").open("w", encoding="utf-8") as stderr:
                try:
                    exit_code = run_codex_cli(
                        command, workdir, stdout, stderr, args.timeout_seconds)
                except OSError as error:
                    stderr.write("Failed to launch Codex: " + str(error) + "\n")
                    exit_code = 127

        trace_exists, summary, warning = collect_habitat_summary(args, trace)
        run_status = {
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
        try:
            archive = archive_codex_run(args)
        except (OSError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as error:
            print("Automatic diagnostic archive failed: " + str(error),
                  file=sys.stderr)
            print("Raw traces and Codex logs remain in " + args.output_dir,
                  file=sys.stderr)
            return 2

        if summary:
            print("Habitat episode:", summary["stop_reason"],
                  "| actions:", summary["steps"],
                  "| legal action export:", summary["completed"])
        else:
            print("Habitat episode has no finalized summary: " +
                  str(warning or "unknown"), file=sys.stderr)
        print("Output:", args.output_dir)
        print("Diagnostic ZIP:", archive,
              "(frames=" + args.archive_frames + ")")
        print("Codex exit code:", exit_code)
        return 0 if run_status["success"] else 2
    finally:
        # If an unexpected error occurs before collection, do not silently
        # delete the sole copy of the Codex model/tool transcript.
        if staging_root.exists():
            print("Host recovery audit: " + str(staged_dir), file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
