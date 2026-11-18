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

## First real test: Phase 2, episode 1, 128 atomic actions

Run on the HOST, not inside havln-cma:

    cd ~/ccx/havln-challenge
    python scripts/run_codex_episode.py \
      --phase phase2 --episode-id 1 --max-steps 128 \
      --model gpt-6-astra --reasoning-effort medium \
      --container havln-cma \
      --output-dir /workspace/agent-runs/codex-p2-episode1 \
      --log-dir ./codex-runs/phase2-episode1-first

This calls a single codex exec with --json, read-only sandbox and an
ephemeral MCP config. It starts one Habitat server via docker exec -i.
Do not use docker exec -t: a TTY corrupts the stdio MCP protocol.

**What gets recorded**

HOST (the selected log-dir):

- prompt.txt: exact instruction/policy sent to Codex; the episode's route
  instruction is supplied by observe and separately stored in episode.json.
- command.json: exact CLI arguments, model, split and output paths.
- codex_events.jsonl: streaming CLI events including available tool activity.
- codex_stderr.log: diagnostics (not MCP protocol data).
- final_message.txt: final agent reply if Codex emits one.

DOCKER (output-dir/traces/test-1):

- episode.json: actual navigation instruction, camera and budget.
- observations/obs_XXXXXX.npz: canonical RGB-D frame after every atomic
  action and explicit observation. No duplicate image files for the CLI.
- tools.jsonl: observe/step call arguments and structured results.
- steps.jsonl: executed actions with before/after frame references.
- replay/events.jsonl: ordered action/observation/tool timeline.
- summary.json: termination, action counts and *action-export* validity.
- result.json: a single checked row when legal, OR failure.json if incomplete.
- codex/: host CLI logs copied here by the launcher for future replay tooling.

The initial replay viewer can display actions, RGB and depth from these
canonical files. It does NOT yet present the Codex CLI event stream as native
API/model-call cards; inspect codex_events.jsonl for that. Scores are NOT
computed from STOP or model claims.

The result.json single-episode row is **not** a full Phase 2 submission.
The official test split requires its complete episode coverage and separate
scoring. Do not upload a one-episode result as a full submission.

The action budget includes STOP. For diagnostic budgets below 500, if Codex
uses the final available movement slot without stopping, the controller
executes a recorded budget_guard STOP so the resulting sequence remains legal.
This is reported as budget_guard_stop, NEVER as confirmed navigation success.
Abnormal CLI termination without STOP remains explicitly incomplete.

## Replay and archive

For the existing replay studio, point to the container output's corresponding
host-mounted directory as described in docs/replay_studio.md, or copy that
directory with docker cp and open it locally:

    docker cp havln-cma:/workspace/agent-runs/codex-p2-episode1 /tmp/codex-p2-episode1
    python scripts/serve_agent_replay.py --output-dir /tmp/codex-p2-episode1

For a standalone RGB-D diagnostic archive inside the container:

    docker exec havln-cma python /toolkit/scripts/pack_agent_diagnostics.py \
      --output-dir /workspace/agent-runs/codex-p2-episode1 --frames all

The existing diagnostic packer handles the canonical episode frames and logs;
retain the HOST codex-runs log directory separately until Codex-specific
event packaging is added.

## Offline tests

Without Docker, Codex login or Habitat:

    python -m unittest discover -s tests -p 'test_codex_navigation.py' -v

These tests exercise MCP initialization/tools, real PNG wire content,
atomic action ordering, budget STOP, failed input handling and the
one-session CLI command. A green offline test does not prove that an actual
Habitat episode or Codex account completed the route.

## Next phase (not in v1)

A sequential/batch harness can launch an independent codex exec for each
episode, assemble official split files after validating all episode rows,
and add Codex model/tool events to Replay Studio. Do not reuse sessions or
persistent simulator state between episodes.
