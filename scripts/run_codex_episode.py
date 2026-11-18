"""Launch one fresh Codex CLI session against one fresh Habitat MCP episode.

Host-side script; requires codex, Docker access and the existing havln-cma
container. It does not change the cloud/hybrid API agent configuration.
"""
import argparse
from datetime import datetime, timezone
import json
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
        'mcp_servers.havln.startup_timeout_sec=180',
        'mcp_servers.havln.tool_timeout_sec=180',
        'model_reasoning_effort=' + json.dumps(args.reasoning_effort),
    ]
    command = [args.codex_bin]
    for override in overrides:
        command.extend(["-c", override])
    command.extend(["exec", "--skip-git-repo-check",
                    "--sandbox", "read-only", "--json",
                    "--output-last-message", str(final_message)])
    if args.model:
        command.extend(["--model", args.model])
    command.append(PROMPT)
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("phase1", "phase2"), default="phase2")
    parser.add_argument("--split", choices=("val_seen", "val_unseen", "test"))
    parser.add_argument("--episode-id", required=True, type=int)
    parser.add_argument("--max-steps", type=int, default=128)
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
                        help="Absolute CONTAINER path; one fresh run root")
    parser.add_argument("--log-dir", type=Path,
                        help="Host directory for CLI events, prompt and stderr")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 500 or args.episode_id < 1:
        parser.error("episode-id must be positive and max-steps must be 1..500")
    split = args.split or ("test" if args.phase == "phase2" else "val_seen")
    if (args.phase == "phase2" and split != "test") or (
            args.phase == "phase1" and split == "test"):
        parser.error("phase2 requires test; phase1 requires val_seen/val_unseen")
    if not Path(args.output_dir).is_absolute():
        parser.error("--output-dir must be an absolute container path")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = args.log_dir or (Path("codex-runs") /
                                (args.phase + "-" + str(args.episode_id) + "-" + stamp))
    log_dir = log_dir.resolve()
    if log_dir.exists():
        parser.error("log directory already exists; do not overwrite a previous run")
    log_dir.mkdir(parents=True)
    final = log_dir / "final_message.txt"
    command = build_command(args, final)
    (log_dir / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    (log_dir / "command.json").write_text(json.dumps({
        "started_utc": stamp, "command": command, "episode_id": args.episode_id,
        "phase": args.phase, "split": split, "container_output_dir": args.output_dir,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # An empty workdir prevents project-local AGENTS.md and previous episode
    # artifacts from appearing in this session's Codex working directory.
    with tempfile.TemporaryDirectory(prefix="havln-codex-session-") as workdir:
        with (log_dir / "codex_events.jsonl").open("w", encoding="utf-8") as stdout, \
             (log_dir / "codex_stderr.log").open("w", encoding="utf-8") as stderr:
            try:
                result = subprocess.run(command, cwd=workdir,
                                        stdout=stdout, stderr=stderr, check=False)
                exit_code = result.returncode
            except OSError as error:
                stderr.write("Failed to launch Codex: " + str(error) + "\n")
                exit_code = 127

    trace = args.output_dir.rstrip("/") + "/traces/" + split + "-" + str(args.episode_id)
    exists = subprocess.run([args.docker_bin, "exec", args.container,
                             "test", "-d", trace], check=False)
    if exists.returncode == 0:
        subprocess.run([args.docker_bin, "exec", args.container, "mkdir", "-p",
                        trace + "/codex"], check=True)
        for name in ("prompt.txt", "command.json", "codex_events.jsonl",
                     "codex_stderr.log", "final_message.txt"):
            path = log_dir / name
            if path.exists():
                subprocess.run([args.docker_bin, "cp", str(path),
                                args.container + ":" + trace + "/codex/" + name],
                               check=True)
        status = subprocess.run(
            [args.docker_bin, "exec", args.container, "cat", trace + "/summary.json"],
            check=False, text=True, capture_output=True)
        if status.returncode == 0:
            summary = json.loads(status.stdout)
            print("Habitat episode:", summary["stop_reason"],
                  "| actions:", summary["steps"],
                  "| valid action export:", summary["completed"])
            print("Container trace:", trace)
            print("Host Codex log:", log_dir)
            return 0 if exit_code == 0 and summary["completed"] else 2
    print("Codex exit code:", exit_code, "| logs:", log_dir,
          "| Habitat trace not finalized", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
