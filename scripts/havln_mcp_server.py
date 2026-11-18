"""Run exactly one HA-VLN episode as a Codex-controlled MCP stdio server.

Launch with docker exec -i (never -t). A fresh process must be used for each
episode so the Habitat state AND the Codex conversation cannot leak across runs.
"""
import argparse
from contextlib import redirect_stdout
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from havln_agent.codex_episode import CodexEpisode
from havln_agent.codex_mcp import MCPServer
from havln_agent.habitat_adapter import (camera_from_config, environment_config,
                                         runtime_imports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("phase1", "phase2"), default="phase2")
    parser.add_argument("--split", choices=("val_seen", "val_unseen", "test"))
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=Path("/data/havln2"))
    parser.add_argument("--runtime-config", type=Path,
                        default=Path("/app/config/challenge_submission.yaml"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Container-side run root (contains traces/)")
    args = parser.parse_args()
    split = args.split or ("test" if args.phase == "phase2" else "val_seen")
    if (args.phase == "phase2" and split != "test") or (
            args.phase == "phase1" and split == "test"):
        parser.error("phase2 uses test; phase1 uses val_seen/val_unseen")
    if not args.episode_id.isdigit():
        parser.error("episode-id must be numeric")
    if not 1 <= args.max_steps <= 500:
        parser.error("max-steps must be between 1 and 500")

    # Save a dedicated protocol descriptor, then send ALL regular stdout
    # (including native Habitat library prints) to stderr. MCP owns stdout.
    protocol_stdout = os.fdopen(os.dup(sys.stdout.fileno()),
                                "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    controller = None
    try:
        with redirect_stdout(sys.stderr):
            get_config, env_type = runtime_imports(args.source_root)
            config = environment_config(get_config, args.runtime_config,
                                        args.data_root, split,
                                        [args.episode_id], args.gpu_id, args.seed)
            camera = camera_from_config(config)
            with env_type(config=config) as env:
                episodes = [e for e in env.episodes
                            if str(e.episode_id) == args.episode_id]
                if len(episodes) != 1:
                    raise ValueError("Episode ID not uniquely present in selected split")
                env.habitat_env.episode_iterator = iter(episodes)
                controller = CodexEpisode(env, camera, args.episode_id, split,
                                          args.max_steps, args.output_dir)
                MCPServer(controller, writer=protocol_stdout).serve()
    finally:
        if controller is not None:
            controller.finish()
        protocol_stdout.close()


if __name__ == "__main__":
    main()
