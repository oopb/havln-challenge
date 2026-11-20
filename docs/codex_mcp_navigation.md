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
- MCP exposes `observe(include_depth)` and `step(actions)`. The
  `include_depth` option is optional.
  observe returns fresh RGB, observation ID, action budget and `pitch_deg`
  (commanded angle relative to the initial forward view; negative is down).
  The episode's route instruction is already in one-time system instructions.
  Depth is sent for a measured near obstacle across the robot's **whole
  forward body width**, once after a stalled forward in the same blocked
  period, on the first observation at a changed heading or tilt, or when
  the caller passes `{"include_depth": true}`. Other observations return RGB
  only. `near_body_risk` gives at most two coherent measured regions with
  forward distances. `depth_reason` records the first applicable reason in
  order, `front_obstacle`, `near_body_risk`, `post_stall_inspection`,
  `requested`, or `view_changed`.
  `front_obstacle_m` is included only when the legacy centerline detector
  confirms central obstruction. The regions describe measured obstacle
  locations. Use RGB to identify what occupies those regions.
  `forward_blocked_by_depth` reports that the observed one-step sweep contains
  a measured obstruction. `forward_clearance_unknown` reports missing Depth
  in a sweep with nearby risk. Both describe the current view. The forward
  action still checks live Depth before moving.
  step executes an ordered list of allowed atomic actions and returns receipts.
  A list containing both forward and turn or tilt actions is rejected.
  After any turn or tilt, call observe and wait for the new view before
  submitting a separate forward step call.
  It refreshes RGB-D before a tool call's initial forward and checks each
  subsequent forward against the preceding action's new frame. Coherent Depth
  inside the nominal body's one-step sweep yields `FORWARD_BLOCKED` before the
  action is issued. Nearer-term advisory risk outside that sweep allows one
  forward step per call while the risk persists. `REOBSERVE_FORWARD` ends the
  batch before another forward. Only executed actions consume budget.
  For each attempted forward, the controller uses the robot's own planar
  displacement along the initial heading. Less than 20% of the nominal
  step yields `LOW_FORWARD_PROGRESS` and interrupts the batch. A retry in
  that heading requires a new `observe()`. When pose is unavailable, exact
  unchanged RGB still yields `NO_VISUAL_PROGRESS`. `step()` returns text.
  Only when useful, `step()` also returns `motion_hint`: cumulative and net
  turn degrees since the last forward that made progress, plus batch turn
  degrees if the batch alone turned >=180 degrees.
  Right turns are positive, left turns negative. Up to two recent exact-RGB
  forward stalls (within 16 actions) are expressed as angles **relative to
  the current view**.
  After two measured forward steps in the same direction, a turn of at least
  150° from that direction prompts `ROUTE_RECHECK`. The next two successful
  forward moves each require a fresh observation and a separate batch. Verify
  the next route landmark before committing to the new direction.
- `look_down_for_clearance` suggests inspecting the low edge of a currently
  measured near obstacle. Use one `LOOK_DOWN` and observe the new RGB-D
  view. Tilt further if more evidence is needed, then restore pitch.
  At most one straight step can reuse the low-view check. Check Depth on
  both sides and at the nearest corner before entering a narrow gap.
  Near people or crowded passages, observe between single forward steps.
  Open corridors still allow longer batches.
- `pitch_deg` tracks commanded LOOK_UP/LOOK_DOWN actions. The controller uses
  robot pose internally and gives Codex relative motion feedback. Depth with
  missing or out-of-view evidence can leave obstacles undetected.

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

The runner keeps run output under the container-side
`--output-dir`, exactly like the original `run_api_agent.py`. The host uses
temporary staging only while Codex runs; it removes that copy after saving
the audit to Docker. If collection fails, it prints and retains the host
recovery directory.

The original exporter also puts automatic archives in a **sibling**
`diagnostic-archives/` directory and defaults to **all frames for one
episode**. Codex now calls the same `auto_archive()` function, preserving
its timestamped, collision-resistant, atomic ZIP naming convention.

```text
/workspace/agent-runs/codex-p2-episode1/
├── run_status.json                  # Codex exit status + Habitat/export status
├── test.diagnostic.json             # Valid single-episode action export
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
            ├── codex_usage.json    # Token totals + quota snapshots
            ├── codex_stderr.log
            ├── final_message.txt
            └── run_status.json
```

If the simulator does not initialize, the same session audit instead lives
under `output-dir/codex/`. The automatic ZIP is still created with
`--allow-empty` so the Codex error is not lost. In a partially initialized
episode, the CLI audit is placed next to the partial trace.

The automatic archive is stored outside the run directory, following the
original structure:

```text
/workspace/agent-runs/
├── codex-p2-episode1/
│   ├── test.diagnostic.json
│   ├── run_status.json
│   └── traces/test-1/...
└── diagnostic-archives/
    └── codex-p2-episode1-YYYYMMDDTHHMMSSZ-diagnostic.zip
```

For this single-episode runner, `--archive-frames auto` (default) retains
**all frames**. Explicit `--archive-frames critical` keeps the first and
last images, movement evidence, and every image returned by `observe()`.
A critical ZIP may omit frames required for per-action before/after replay;
use `all` for complete offline playback. The uncompressed run directory
always retains every original frame.

Replay Studio correlates the Codex MCP tool stream to the exact Habitat
`tools.jsonl` and `steps.jsonl` sequence only when tool names and arguments
agree; mismatches are left unlinked. On completed `observe()` events it
shows **the actual RGB returned to Codex**, plus Depth marked as a model
input only when the completed observe response actually included it. On `step()` events it shows the true
before/after frames. Initial/thread-level Codex events remain unlinked.
The viewer omits duplicate base64 image bytes in JSON responses, while the
original complete Codex event stream remains in the saved ZIP.

Before launching Codex, the host reads only the public route instruction
for the selected episode from the matching HA-R2R dataset JSON. The route
instruction and existing short navigation rules are written once to
`codex/prompt.txt` and injected via the per-run
`-c model_instructions_file=<absolute-path>` override, as Codex's system-level
model instructions. This intentionally replaces Codex's built-in base
instructions for the isolated navigation session. The initial user prompt
only asks Codex to call `observe()`. Each `observe()` returns RGB and a
short metadata text block, with optional Depth and `front_obstacle_m`; it
does **not** repeat the route. `step()` accepts only an action list and
returns a short text receipt without any image.
Codex manages the continuous model context internally: the CLI event log
does not record the complete hidden per-model-call system/developer
message envelopes.

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

After each run, a timestamped ZIP is automatically generated in
`--output-dir`'s **sibling `diagnostic-archives/` directory**, even if Codex
failed (provided Docker remains reachable). A single episode defaults to
`--archive-frames auto` = `all`. You no longer need to pack manually.

For earlier critical-only archives, missing frames cannot be recovered from
the ZIP itself. Repackage from the original uncompressed run directory with
`--frames all` if it still exists.

## Codex usage accounting (per single-episode run)

Each `run_codex_episode.py` invocation automatically saves
`traces/<split>-<episode>/codex/codex_usage.json` (or
`output-dir/codex/codex_usage.json` if Habitat never started). It is
included in the standard sibling diagnostic ZIP and shown in Replay Studio.

- `tokens`: actual `turn.completed.usage` counters from
  `codex exec --json`; input already includes cached input, so
  `total_tokens = input_tokens + output_tokens`. Reasoning tokens, when
  reported, are a subset of output. Codex does not expose per-MCP-tool-call
  token bills; `mcp_tool_calls_completed` is a tool event count only.
- `quota.before` / `quota.after`: optional account-wide read-only
  `account/rateLimits/read` snapshots from `codex app-server`.
  These requests do **not** create inference turns. Each query has a
  bounded timeout and is non-fatal when unavailable.
- `quota.delta`: changes in *percentage points* per returned quota
  window. Comparisons require the same limit ID, window duration, and
  reset timestamp, and cannot claim an exact task quota charge: other
  simultaneous Codex usage affects the same account.
- `unavailable`, `partial`, or `not_comparable` mean unknown,
  **not zero consumption**. Missing 5-hour windows are not fabricated
  if the account reports only weekly limits.
- On interruption or startup failure, reported Token counters are
  retained where available. In-flight work may be unreported, so
  the Token status is marked partial or unavailable.

Each finished single-episode or batch runner also prints the **actual post-run
account quota remaining percentage**, used percentage, and comparable
before/after change per reported window (for example 5-hour or weekly).
Batch windows are account-wide and are never summed per child; these
percentages are not per-episode billing. If quota is unavailable, it says so.

To inspect an archived result, open Replay Studio or read
`codex_usage.json` directly; no additional CLI flags are needed.

## Concurrent multi-episode evaluation (host)

`codex exec --ephemeral` can run in **independent CLI processes** at the
same time. The batch runner launches one fresh Codex process, fresh Habitat
MCP process, and fresh trace per episode. Model contexts and simulator
states never leak between jobs. Concurrent sessions share the same account
quota and available GPU/CPU resources. Start with `--batch 2` on one GPU,
not `--batch 32` without memory testing.

A recommended Phase 2 sample, preserving original episode IDs:

```bash
cd ~/ccx/havln-challenge
python scripts/run_codex_batch.py \\
  --phase phase2 --episode-start 1 --episode-limit 20 \\
  --batch 2 --gpu-ids 0 --max-steps 128 \\
  --model gpt-6.2-sol --reasoning-effort low \\
  --output-dir /workspace/agent-runs/codex-p2-batch20
```

To sample 20 **distinct** random test IDs reproducibly from the original
ID range `[100, 3408]`:

```bash
python scripts/run_codex_batch.py \\
  --phase phase2 --episode-start 100 --episode-limit 20 \\
  --random-episodes --seed 42 --batch 2 --gpu-ids 0 \\
  --model gpt-6.2-sol --reasoning-effort low \\
  --output-dir /workspace/agent-runs/codex-p2-random20
```

`--episode-start` is an **inclusive original dataset ID**, not a
0-based row number. Without `--random-episodes`, the runner selects the
next `--episode-limit` original IDs sequentially. Random mode samples
without replacement, records the chosen IDs in `batch_plan.json`, and
uses `--seed` for reproducibility. By default the number of processes
per GPU is `--batch 1`. `--gpu-ids 0 1` uses both GPUs with up to
`batch * gpu_count` independent Codex sessions. The model and reasoning
effort remain configurable; the single-episode defaults are unchanged.

Use `--preflight` to verify Codex login, Docker, dataset IDs and selected
samples without starting model inference or creating output. Every run must
use a **fresh** output directory; existing batches are never silently
overwritten. Failed individual episodes do not prevent the other selected
episodes from running. Partial batches remain marked non-official.

### Batch output and diagnostic archive

By default a batch generates **one** timestamped diagnostic ZIP in the
same sibling archive directory as the existing API runner:

```text
/workspace/agent-runs/
├── codex-p2-batch20/
│   ├── batch_plan.json                # exact selected IDs/settings
│   ├── batch_worker_results.json      # one result per child
│   ├── batch_status.json              # validated exports and token summary
│   ├── batch_quota.json               # account-wide before/after limits
│   ├── codex_usage.json               # aggregated per-episode token usage
│   ├── run_status.json
│   ├── test.diagnostic.json           # legal completed rows, NOT official
│   ├── worker-test-<id>.log
│   └── traces/test-<id>/
│       ├── summary.json
│       ├── steps.jsonl
│       ├── observations/obs_*.npz
│       └── codex/codex_events.jsonl
└── diagnostic-archives/
    └── codex-p2-batch20-YYYYMMDDTHHMMSSZ-diagnostic.zip
```

The container's `/workspace/agent-runs` is the existing mounted workspace
used by the API runner (typically `havln-workspace/agent-runs` on the host,
depending on the Docker bind mount). Use `--no-auto-archive` to retain raw
runs **without** generating the test diagnostic ZIP. Use
`--archive-frames auto|all|critical|none` to control which RGB-D
frames the ZIP contains; auto means **all for one episode** and
**critical for multi-episode** batch runs. All model-seen observe frames
are always retained logically in critical mode. Exact duplicate NPZ
snapshots are stored once with alias metadata, so Replay Studio can open
every original observation ID, including from a ZIP. The full unmodified
run directory stays on the server.

Run a batch ZIP in Replay Studio just like a single-episode ZIP, selecting
any episode from the dropdown. If the MCP server fails before producing
traces, a startup-only diagnostic ZIP can still show the original Codex
CLI error events and logs. The Replay UI now separates MCP request
arguments from completed tool outputs and retains the full raw event tab;
Codex does **not** expose full internal model request envelopes.

The root quota snapshots describe the whole batch time window. Individual
Codex quota percentages overlap in parallel execution and must **never**
be added; token totals, by contrast, are aggregated from disjoint episode
event streams. No partial run is mislabeled as a full official Phase 2
submission.

## Offline tests

Without Docker, Codex login or Habitat:

    python -m unittest discover -s tests -p 'test_codex_navigation.py' -v
    python -m unittest discover -s tests -p 'test_codex_usage.py' -v
    python -m unittest discover -s tests -p 'test_codex_batch.py' -v

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


## Retained reliability fixes (without changing the best-performing policy)

The navigation policy, observation cadence, exact-RGB stall behavior, optional
Depth policy, action batching and system prompt are restored to the
`2cae614b` baseline. The following later fixes are retained because they
protect correctness without adding a new navigation heuristic:

- After Habitat successfully executes an atomic action, record it before
  evaluator/capture work. If post-action processing fails, preserve the
  executed action and return exact `executed` / `skipped_count` receipts.
  Fail closed: an action whose post-STOP capture failed must not be exported
  as a valid submission.
- Use the shared `pillow_compat.RESAMPLING` constants for image resizing;
  Habitat's legacy Pillow runtime does not expose `Image.Resampling`.
- Before paid Codex inference, probe the actual container-side
  `CodexEpisode` import and report the full traceback on failure. Batch
  preflight performs the same probe once before launching workers.
- Keep independent Habitat metric capture and Replay Studio scoring/reporting.
  These are offline diagnostics and never enter the navigation agent's
  observation or tool protocol.
