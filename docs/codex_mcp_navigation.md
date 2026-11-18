# Codex CLI as an independent HA-VLN navigation agent (v1)

This is an **independent experimental path**, not a replacement for
run_api_agent.py. One episode = one fresh Codex CLI process + one fresh
Habitat MCP stdio process. **Never use codex exec resume between episodes.**

## Architecture and boundaries

- Host: scripts/run_codex_episode.py launches Codex from an empty temporary
  working directory with episode-scoped MCP config. No permanent codex mcp add.
- Docker: scripts/havln_mcp_server.py opens the released Habitat environment
  for exactly one requested split/episode.
- Controller: havln_agent/codex_episode.py owns public RGB-D captures,
  legal six-action stepping, budgets, replay receipts and STOP/export checks.
- MCP: havln_agent/codex_mcp.py exposes ONLY observe() and step(actions).
  observe returns the instruction, fresh RGB PNG as an MCP image, and budget.
  step executes an ordered list of allowed atomic actions and returns receipts.
- The agent sees **RGB only**. Depth is saved for replay but not supplied
  to Codex. No ground-truth pose, reference route or goal coordinates exposed.

The existing cloud/hybrid scheduler, model API clients and config YAML
are untouched. The CLI model selection is independent of .env API settings.

## Before launching

On the server HOST, ensure:

1. codex is installed and authenticated: codex --version; codex login status.
2. The selected model is available to that Codex account (the default is
   gpt-6-astra; override --model if needed).
3. The user can run docker exec without an interactive sudo password.
4. havln-cma is running with the released Habitat 0.1.7 image, licensed
   data under /data/havln2, and the current repository mounted at /toolkit.
5. The target episode has not already been written in the selected output
   directory. The controller refuses to overwrite old traces.

Check container visibility after updating the host repository:

    docker exec havln-cma test -f /toolkit/scripts/havln_mcp_server.py
    docker exec havln-cma python -c "import habitat, habitat_sim; print(habitat.__version__, habitat_sim.__version__)"

If the first check fails, update the container's /toolkit checkout or bind
mount, not just the host repository. Avoid launching a stale Docker checkout.

## Free preflight before the first inference

Run the same command with --preflight first. It checks the installed Codex
CLI, Codex login, your direct Docker permission, the mounted MCP server
file, Habitat 0.1.7 imports and the selected split's episode ID. No Codex model is
called, no Habitat episode is started, and no output directory is created.

    python scripts/run_codex_episode.py \
      --phase phase2 --episode-id 1 --max-steps 128 \
      --output-dir /workspace/agent-runs/codex-p2-episode1 \
      --preflight

PASS only means dependencies and file paths are reachable. It does not
confirm the selected model is enabled for your account or that
the MCP tool loop has actually run.

## First real test: Phase 2, episode 1, 128 atomic actions

Run on the HOST, not inside havln-cma:

    cd ~/ccx/havln-challenge
    python scripts/run_codex_episode.py \
      --phase phase2 --episode-id 1 --max-steps 128 \
      --timeout-seconds 2400 \
      --model gpt-6-astra --reasoning-effort medium \
      --container havln-cma \
      --output-dir /workspace/agent-runs/codex-p2-episode1

This calls a single codex exec with --json, --ephemeral,
--ignore-user-config, read-only sandbox and an
ephemeral MCP config. It starts one Habitat server via docker exec -i.
Do not use docker exec -t: a TTY corrupts the stdio MCP protocol.

**One output directory (no --log-dir needed)**

The runner writes all persistent results under the container-side
`--output-dir`. The host uses a temporary staging directory only while
Codex is running; it deletes that temporary copy after the audit is safely
copied to the container. If Docker collection fails, the host recovery
location is printed and retained instead of silently losing the logs.

```text
/workspace/agent-runs/codex-p2-episode1/
├── run_status.json                  # Codex exit status + Habitat/export status
├── diagnostic.zip                   # Automatically generated, critical frames
└── traces/
    └── test-1/
        ├── episode.json            # Exact route instruction
        ├── summary.json            # Executed action counts / stop reason
        ├── result.json             # Legal action row, only if completed
        ├── failure.json            # Only when Habitat episode incomplete
        ├── steps.jsonl             # Atomic action receipts and frame references
        ├── tools.jsonl             # observe/step arguments and results
        ├── replay/events.jsonl     # Habitat timeline
        ├── observations/obs_*.npz  # Canonical RGB-D observations
        └── codex/
            ├── prompt.txt
            ├── command.json
            ├── codex_events.jsonl  # Codex session model/tool events
            ├── codex_stderr.log
            ├── final_message.txt
            └── run_status.json
```

If the simulator does not initialize, the same session audit instead lives
under `output-dir/codex/`. The automatic ZIP is still created with
`--allow-empty` so the Codex error is not lost. In a partially initialized
episode, the CLI audit is placed next to the partial trace.

The archive defaults to **critical** frames, including the first/last
frame, movement evidence, and **every frame actually returned to Codex by
`observe()`**. All canonical RGB-D observations remain in the uncompressed
run directory. Choose `--archive-frames all` when a fully portable replay
with every atomic action frame is needed; use `none` for metadata only.

Replay Studio now has a **Codex 会话** filter that renders CLI events,
their raw model/tool records, and the exact initial Prompt. It preserves
the normal Habitat **原子动作** and tool views with canonical RGB-D before
and after each action. The CLI event stream and Habitat action stream are
**not** assigned artificial cross-stream timestamps or per-event frame
matches: Codex events are grouped separately. Only Habitat events have
precise recorded causal order. We can add exact cross-stream correlation
IDs later after validating real CLI events. Scores are NOT
computed from STOP or model claims.

The result.json single-episode row is **not** a full Phase 2 submission.
The official test split requires its complete episode coverage and separate
scoring. Do not upload a one-episode result as a full submission.

The action budget includes STOP. For diagnostic budgets below 500, if Codex
uses the final available movement slot without stopping, the controller
executes a recorded budget_guard STOP so the resulting sequence remains legal.
This is reported as budget_guard_stop, NEVER as confirmed navigation success.
Abnormal CLI termination without STOP remains explicitly incomplete.
The host launcher has a 2400-second default wall-clock limit, configurable
with --timeout-seconds. On timeout it sends SIGTERM (then SIGKILL if needed)
to the Codex process group, including the docker exec MCP child; the host
run_status.json records exit 124/timed_out=true. Even if Habitat cannot
finalize its trace, the host audit logs are retained.

## Replay and archive

For the existing replay studio, point to the container output's corresponding
host-mounted directory as described in docs/replay_studio.md, or copy that
directory with docker cp and open it locally:

    docker cp havln-cma:/workspace/agent-runs/codex-p2-episode1 /tmp/codex-p2-episode1
    python scripts/serve_agent_replay.py --output-dir /tmp/codex-p2-episode1

After each run, **diagnostic.zip is automatically generated inside
--output-dir**, whether Codex succeeded or returned an error (provided Docker
remains accessible). It contains the model/tool transcript, prompt,
termination diagnostics and critical RGB-D evidence. You no longer need to
run pack_agent_diagnostics.py manually.

For a shareable *all-frame* archive, select `--archive-frames all` before
starting the episode, or use the existing packer manually afterwards.

## Offline tests

Without Docker, Codex login or Habitat:

    python -m unittest discover -s tests -p 'test_codex_navigation.py' -v

These tests exercise MCP initialization/tools, real PNG wire content,
atomic action ordering, budget STOP, failed input handling, the
one-session CLI command, no-inference preflight, Codex event viewing,
diagnostic ZIP retention and fail-closed simulator exceptions. A green offline test does not prove that an actual
Habitat episode or Codex account completed the route.

## Next phase (not in v1)

A sequential/batch harness can launch an independent codex exec for each
episode, assemble official split files after validating all episode rows,
and add Codex model/tool events to Replay Studio. Do not reuse sessions or
persistent simulator state between episodes.
