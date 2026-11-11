<h1 align="center">🤖 RoboWorld 2026 Track 2: HA-VLN<br>Human-Aware Vision-and-Language Navigation</h1>

<div align="center" markdown="1">

**Official participant toolkit for [RoboWorld 2026](https://roboworld2026.github.io/) Track 2**

*Built on [HA-VLN 2.0](https://uwmilab.github.io/HA-VLN-webpage/) and affiliated with the [RoboPAD Workshop at NeurIPS 2026](https://robotpad2026.github.io/)*

[![RoboWorld](https://img.shields.io/badge/RoboWorld-2026-blue)](https://roboworld2026.github.io/)
[![Track 2](https://img.shields.io/badge/Track_2-HA--VLN-green)](https://roboworld2026.github.io/track2)
[![CodaBench](https://img.shields.io/badge/CodaBench-Submit-purple)](https://www.codabench.org/competitions/18135/)
[![RoboPAD](https://img.shields.io/badge/Affiliated_with-RoboPAD_2026-red)](https://robotpad2026.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-2503.14229-b31b1b)](https://arxiv.org/abs/2503.14229)

<p align="center">
  <img src="assets/media/track2-havln-poster.png" alt="RoboWorld 2026 Track 2: HA-VLN Poster" width="460" />
</p>

**🏆 Awards: Official Certificates for Top 5 Teams & NeurIPS 2026 RoboPAD Workshop Oral Presentations**

</div>

## 🌍 Challenge Overview

HA-VLN evaluates embodied agents on **human-aware vision-and-language navigation**
in continuous 3D environments populated by dynamic people. Unlike static VLN
benchmarks, [HA-VLN 2.0](https://arxiv.org/abs/2503.14229) requires agents to
ground human-referenced route instructions, adapt to moving bystanders, and
reach the goal without human collisions. Official scoring replays submitted
action sequences in the released HA-VLN 2.0 Habitat 0.1.7 runtime, and any
policy, planner, world model, or VLA method that outputs valid actions is welcome.

<p align="center">
  <img src="assets/media/teaser.webp" alt="HA-VLN task overview" width="90%" />
</p>

### 🎯 Task Definition

| Component | Description |
|:--|:--|
| Input | Episode ID, initial position and yaw, navigation instruction, and released human-aware episode inputs. |
| Output | One sequence made from six discrete VLN-CE actions. |
| Actions | `STOP`, `MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`, `LOOK_UP`, `LOOK_DOWN`. |
| Budget | 1–500 actions; all six actions count toward the limit. |
| Evaluation | Trusted simulator replay measuring navigation and human-aware safety. |

## 📅 Competition Details

- **Event:** [RoboWorld Challenge 2026, Track 2](https://roboworld2026.github.io/track2).
- **Affiliated workshop:** [RoboPAD Workshop at NeurIPS 2026](https://robotpad2026.github.io/).
- **Registration:** register through the [Google Form](https://roboworld2026.github.io/#registration) (registration opens Oct 08, 2026) to be eligible for the leaderboard, certificates, and awards.
- **Submission platform:** [CodaBench — HA-VLN](https://www.codabench.org/competitions/18135/).
- **Submission limits:** five per day and 100 per phase; the best score is retained.

### 🗓️ Timeline

| Event | Date |
|:--|:--|
| Registration opens | October 08, 2026 (via [Google Form](https://roboworld2026.github.io/#registration)) |
| Data, baselines & servers online (Phase 1 opens) | October 15, 2026 |
| Phase 1 deadline / Phase 2 opens | October 30, 2026 |
| Final submission deadline (Phase 2 closes) | November 30, 2026 |
| Award decision announcement | December 12, 2026 |

*Follow the [competition page](https://www.codabench.org/competitions/18135/) and [track website](https://roboworld2026.github.io/track2) for any updates.*

### 🗂️ Phases

| Phase | Duration | Submission | Leaderboard role |
|:--|:--|:--|:--|
| Phase 1: Validation | Oct 15 – Oct 30, 2026 | Released `val_seen` and `val_unseen` action sequences | Reports both splits; ranks by full-precision `val_unseen` Score. |
| Phase 2: Final Test | Oct 30 – Nov 30, 2026 | Action sequences for the held-out phase bundle | Determines final ranking and awards. |

Both phases use the same six-action JSON contract and Score. Phase 1 and Phase
2 results are not combined.

### 🏆 Awards & Recognition

| Recognition | Description |
|:--|:--|
| 📜 **Certificates of Recognition** | Official certificates awarded to the **Top 5 teams** in Track 2 |
| 💡 **Best Innovative Solution** | Certificate recognizing outstanding creativity and technical innovation |
| 🎤 **Oral Presentations** | Selected top-performing teams will be invited to give oral presentations at the **RoboPAD Workshop @ NeurIPS 2026** |

> **Active implementation (v17):** [Continuous natural-language mission navigation](docs/mission_narrative_navigation.md) retains the model-selected camera pitch during MOVE (no automatic LOOK_UP recenter), runs calibrated live Depth preflight before each atomic forward, and emits model decisions as **observed evidence → concise analysis → final tool + args**. The earlier detailed design notes below are historical and are **not** the active agent protocol.

For the API-driven HA-VLN Agent, see [evidence-grounded multimodal
navigation](docs/evidence_grounded_navigation.md) and
[active observation](docs/active_observation.md).
The canonical configuration is `configs/agent.yaml`; real Habitat testing is run manually.

**Post-obstacle exploration:** A `PARTIAL` waypoint that advanced once and
then returned `interrupted_status=BLOCKED` now retains its measured
obstacle evidence at the actual post-action pose. Local camera coverage uses
absolute heading, pitch, subgoal and odometry epoch to distinguish unvisited
directions from repeated relative turns. `turn_and_observe` and
`observe_for_goal` return `NO_NEW_INFORMATION` for redundant camera
sweeps instead of consuming atomic actions. An intentional
`turn_and_observe(purpose="dynamic_recheck")` can revisit a potentially
dynamic obstruction. Collision/terrain geometry is now **model-invoked only**;
normal motor actions no longer call a depth-body safety shield automatically. See
[post-obstacle exploration](docs/post_block_exploration.md).


**Active navigation variant:** The eval-32/v17 mission strategy is retained with requested PR #90 (model-chosen LEFT/RIGHT/LOOK actions; conclusive-only front Depth in NavigationAgent requests) and PR #91 (read-only deduplicated compact navigation evidence). The factual same-stance TURN/LOOK counters are restored without any forced-motion rule. Raw Memory, per-atomic Depth safety, visual passage grounding, Replay Studio and Score remain intact. See [rollback and retention notes](docs/eval35_rollback_v17.md).

**Eval-41 targeted decision repair (v28):** Without changing motion control or Depth thresholds, `front_depth_preflight` now carries the number of physically denied forward requests from exactly the same current RGB-D/stance/yaw/pitch, when a matching source-backed rejection exists. The duplicate historical rejection row is removed from the compact input, not from Memory. NavigationAgent gets concise guidance to avoid identical rejected retries and compare current upward/downward stair cues with earlier route frames through a focused optional ObservationAgent call; open single passages still favor short autonomous forward progress. No forced observation, new validator, or rigid rule. See [v28 eval-41 regression](docs/eval41_v28_decision_repair.md).

**Current-camera-relative model bearings (v27):** World yaw remains **internal only** (Memory, physical motion, tests and logs); every NavigationAgent / ObservationAgent model-facing structured context is a read-only projection of source-bearing fields into signed offsets from the CURRENT robot RGB (LEFT positive, RIGHT negative, forward=0). This includes current and historical views, route segment/endpoint Depth evidence, scan coverage bins, waypoint snapshots, tool results and panorama inputs. Prior VLM dialogue is reprojected on each actual camera turn from immutable raw session requests, so an old relative bearing is not mislabeled as current. Cross-stance historical target bearings never become executable movement suggestions. Tool output stays relative LEFT/RIGHT/NONE with nonnegative turn_degrees. See [v27 audit and tests](docs/v27_relative_model_angles.md).

**Eval-39 navigation continuity fix (v26):** Relate a confirmed current/endpoint Depth block to its preceding authentic multi-step forward segment without rewriting earlier moves as failures. History RGB now prioritizes the segment's actual start/end, de-duplicates repeated same-stance yaw/pitch views, and allows up to five historical RGB images **when useful**; ObservationAgent uses the same source-backed keyframe selection for up to three RGB-D pairs. Same-stance candidate route bearings are already expressed as `LEFT/RIGHT/NONE + angle` in compact memory; across stances they remain historical clues only. NavigationAgent is encouraged to take a short forward exploration along one visible plausible passage without a confirmed obstacle, and to call ObservationAgent only to resolve a specific route-critical visual ambiguity. No forced motion, sg0/sg1, new validator, arbitrary Depth veto or absolute heading output. See [v26 design and regression](docs/eval39_route_continuity_v26.md).

**Eval-39 rollback (relative-only):** The v24 near_field_proximity prompt input, optional model person boxes, human-buffer veto, automatic near-person re-observe, and related config are rolled back to the v23/eval-38 behavior. As explicitly requested, the **ONLY model-facing turn format remains `turn_direction=LEFT/RIGHT/NONE` plus `turn_degrees`**, not `target_heading_deg`; absolute yaw remains source-labelled history only. No replacement navigation/safety behavior is included in this rollback. See [eval-39 analysis and staged future-fix plan](docs/eval39_rollback_and_repair_plan.md).

**Eval-37 route-memory repair (v23):** NavigationAgent now receives source-linked authentic forward-history anchors (including relevant early segments and opposing-heading evidence) alongside bounded, explicitly unverified earlier model visual claims; it no longer relies solely on the last mutable progress narrative. It also receives recent LOOK questions and their actual subsequent model-view descriptions plus first-time/revisited yaw/pitch counts. History RGB selection prioritizes actual before/after forward images when comparing an opposing heading instead of always attaching the episode-origin frame. No stage verifier, fake odometry, forced motion, or persistent second memory. See [eval-37 evidence-design notes](docs/eval37_route_memory_repair.md).

**Eval-36 observation + heading repair (v22):** NavigationAgent actively weighs informative LOOK_UP/LOOK_DOWN at existing yaw (near stair/floor geometry versus distant routes); ObservationAgent gets matching pitch-view guidance. The experimental absolute-heading model output was **removed**; both navigation tools now use relative LEFT/RIGHT/NONE only. Historical yaw labels are not motor commands. This is model-owned choice, not forced scanning or hardcoded route verification. The #90 confirmed Depth-only input, #91 compact Memory, atomic Depth veto and official scoring are unchanged. See [v22 design and regression notes](docs/eval36_pitch_heading_repair.md).

**Offline replay studio:** Inspect completed real RGB-D trajectories, model
requests/responses, observation sessions, tools and atomic actions in a
read-only browser, without starting Habitat or calling an API:

```bash
python scripts/serve_agent_replay.py --output-dir /workspace/agent-runs/eval-batch4-30
# Or: --archive /path/to/diagnostic.zip
```

Open `http://127.0.0.1:8765` (use SSH port forwarding on a remote server).
New runs include byte-verified model-input media references; existing sensor
NPZ frames are reused instead of copied. For a portable *complete* replay
archive, use `pack_agent_diagnostics.py --frames all`; default critical
archives intentionally omit some frames. See [Replay Studio](docs/replay_studio.md).

The viewer also displays the [official HA-VLN 2.0 composite Score formula](https://roboworld2026.github.io/track2)
and computes Navigation, Social and Score from independently evaluated
`SR/NE/CR/TCR`. Add `--metrics-file /path/to/score-metrics.json` to the
viewer command or place a single `score_metrics.json` in the run root.
If ground-truth goal distance or human-collision metrics are unavailable,
Score is marked **unavailable** (not zero); you can still use the
collapsible four-metric **formula preview** without modifying the run.
The viewer never infers official success from STOP, action counts or RGB-D.

**Compact diagnostics:** Episode tracing now stores one canonical `observations/obs_XXXX.npz`
per RGB-D frame and references it from `steps.jsonl`, instead of tripling
the data with `input-NNN.npz` and `post-NNN.npz`. Model requests are logged
in one `api.jsonl` (with `role` included); duplicate per-role API logs
and the second full tool ledger are not written. To share a run without
uploading all sensor frames:

```bash
python scripts/pack_agent_diagnostics.py \\
  --output-dir /workspace/smoke-500-17 \\
  --archive /workspace/smoke-500-17-diagnostic.zip
```

The archive contains all model decisions, tool receipts, actions, memory and
key RGB-D evidence around motion and obstacles. It deliberately omits API
cache, duplicate files and most repetitive scan frames. For frame-by-frame
image analysis, add `--frames all`; for logs only use `--frames none`.
The original run directory remains intact on the server.


**Action-driven motion with optional odometry:** NavigationAgent selects
a finite yaw and forward action count. The executor completes the requested
calibrated atomic actions without relying on an RGB-D metric position estimate.
An UNKNOWN translation never resets motion_epoch or stops an otherwise changing
RGB-D route; identical pre/post RGB-D still interrupts to avoid blind repeated
forward attempts. Completed plans return `ACTIONS_COMPLETED`, *not* physical
`TARGET_REACHED`: a nominal command distance is never fabricated as measured
motion. Genuine stair descent and subgoal completion still require independent
registered visual/motion evidence. See
[action-driven motion and optional odometry](docs/odometry_recovery.md).


**Batch 4-13 direct VLM relay:** VLM-led observation no longer drops
visual hypotheses because the free-form `target_type` differs from the
requested `target_types`. Both positive and negative VLM statements, including
low-confidence and missing-box observations, reach NavigationAgent as
source-labeled `visual_reports` (up to eight compact recent frames; complete
reports in diagnostics). Main-agent planning and movement authority stays
unchanged. Real observation provenance, JSON schemas, camera action limits
and independent task completion checks remain in place.

**Batch 4-12 visual/progress follow-up:** VLM may now actually execute its
requested LOOK/TURN after finding a visual candidate (up to three follow-up
image checkpoints) instead of automatically terminating the scan. Expensive
reverse sweeps back to already-visited yaw are returned to VLM for selection
of a more useful view. The navigation context carries chronological, strictly
registered local vertical movement evidence, observed direction reversals,
next ordered subgoal and recent source-labeled VLM hypotheses in a bounded
`progress_review`. The strategic model may independently call
`verify_subgoal`, `ground_landmark` or `select_subgoal` provisionally,
but no heuristic automatically marks a subgoal complete or chooses the
robot's direction. Camera pitch conventions and visual-versus-hypothetical
target classifications are explicit. Verify in Habitat on the same episode;
passing CI is not evidence of successful descent or task completion.

**Batch 4-11 P0/P1 hardening:** NavigationAgent now emits only the validated
`tool` and `args` object; provider-hallucinated optional `memory_delta` fields
(even malformed object/list shapes) are silently discarded *after* strict
validation of actual tool parameters. VLM and tool receipts own persistent
memory. During one VLM-led observation, a duplicated yaw/pitch request is
rejected without physical TURN and the VLM gets up to two same-image correction
opportunities with real coverage/rejection metadata. VLM view evidence is also
persisted in the same typed `visual_targets` pool as panoramic hypotheses,
including authentic frame ID, bearing, camera pitch, box and evidence status.
Such targets never certify safe traversal or completed tasks.

**Navigation tool-menu refactor (20 registered, 9 core):**
The ToolExecutor registry still contains all 20 validated tools, so archived
tool calls and specialized visual/history utilities remain executable.
Ordinary NavigationAgent requests advertise only these nine strategic
capabilities: `observe_for_goal`, `navigate_relative`, `navigate_local`,
`ground_landmark`, `inspect_geometry`, `verify_subgoal`,
`select_subgoal`, `request_task_review`, and `request_finish`.
Extra tools become visible only when their purpose is grounded in current
evidence: a fresh grounded target enables `navigate_to_pixel`; an active
route enables cancel; explicit geometry feedback enables waypoint previews;
ambiguous visual memory enables contextual retrieval; verification gaps enable
targeted task/history inspection. Legacy `turn_and_observe` is reserved for
orientation-only tasks and existing executor callers, never routine camera
search. `analyze_scene` and `assess_entry` are internal VLM utilities rather
than competing main-agent observation tools. The panorama checkpoint still
offers only `navigate_relative`. The API trace records `offered_tool_count`
and `offered_tool_names` for each navigation request. **No core motion or
safety behavior is inferred from reducing the tool menu.**

**VLM-led observation and smaller navigation requests:** NavigationAgent
specifies WHAT visual evidence to find using `observe_for_goal`, then
VisualPerception chooses successive real TURN/LOOK checkpoints without
invoking the navigation LLM after every 15-degree camera turn. It distinguishes
entries and approaches from distant visual cues and returns evidence-backed
observation IDs and angles. Normal navigation requests now use compact goal,
pose and target-memory summaries rather than repeatedly sending full historical
panorama/geometry records. Detailed records remain queryable by tool. API
logging includes prompt/protocol/context character counts. Legacy scripted
observation remains available for archived calls. See
[VLM-led active observation](docs/active_observation.md).


### Geometry mode: model-invoked only

The API navigation agent now executes model-selected finite MOVE/OBSERVE actions
without hidden automatic `Geometry.diagnose`, `forward_probe`, terrain fitting,
or human-occupancy vetoes. `inspect_geometry` and `get_waypoint_candidates`
run measured geometry only when NavigationAgent explicitly calls those tools;
`ground_landmark` still permits model-requested RGB-D coordinate projection.
Measured obstacle evidence is advisory to the model, not an automatic executor
gate. Passive camera refreshes and `observe_for_goal` visual checkpoints do
not perform collision probes.

The Habitat simulator still enforces its own action validity and physics.
Executed MOVE_FORWARD actions still yield real RGB-D receipts for odometry,
stalls, pose uncertainty, task progress, and the global episode budget. For
stair landing verification, an explicit `inspect_geometry` immediately after
a registered level forward can contribute measured post-step terrain evidence.
**This is a simulation-only experimental policy; it would not be a safe
physical-robot default**, especially around moving people.

### Live terminal progress for API Agent runs

`scripts/run_api_agent.py` now prints progress during quiet Habitat worker execution.
By default it polls every 3 seconds and prints on action/tool/API-phase changes,
plus a heartbeat every 15 seconds during slow model requests. Example:

```bash
python -u scripts/run_api_agent.py --config configs/agent.yaml \\
  --output-dir /workspace/smoke-100-live --splits val_seen \\
  --episode-limit 1 --max-steps 100
```

Each line includes the split/episode, actions used/max, forward/turn/look
counts, total tool calls, latest tool name/result, completed HTTP attempts,
current API request phase, and elapsed time. This output is emitted by the
parent process; full worker logs and trace JSONL are still saved unchanged.
Use `--progress-interval 1` for faster refresh or `--progress-interval 0`
to disable live polling. `--terminal-log off` disables all normal terminal
progress and startup/completion messages but retains error diagnostics on stderr,
worker logs, the full per-episode trace and automatic archives.

### Recover an already-completed parallel run after shard assembly failure

Before fixing the batch assembler, `--batch 4` could successfully finish all
workers but then try to open a nonexistent pre-split shard such as
`val_seen-s8pcmisQ38h.json`. The individual `-partNNNN.json` shards and
the original worker task manifests can be reused **without rerunning Habitat
or reissuing any model requests**:

```bash
python /toolkit/scripts/run_api_agent.py \\
  --output-dir /workspace/agent-runs/eval-batch4 \\
  --assemble-only --terminal-log on --archive-frames critical
```

This reads the **original** `export_metadata.json` and `worker-<ID>.json`
manifests, verifies their stored run fingerprint and exact episode coverage,
checks each shard's action sequence, writes split diagnostic JSON, and only then
marks the export complete. No `--config`, `--batch`, credentials, GPU or
simulator initialization is required for `--assemble-only`. A missing shard
still signals an incomplete/failed worker rather than silently skipping it.
A new ordinary run after code changes must use a new output directory because
its code fingerprint differs; use `--assemble-only` to salvage existing work.

### Parallel batch evaluation on one or more GPUs

`--batch N` launches **N independent Habitat worker processes per GPU**,
each with a separate simulator, model client, API cache, progress marker and log.
Episodes are split into unique shards; if there are fewer episodes than worker
slots, surplus slots are not launched. It is NOT simultaneous actions within
a single episode and does not shorten the intrinsic latency of one episode.

```bash
# H100 GPU 0: 4 concurrent episodes, if the split contains >=4 episodes
python -u scripts/run_api_agent.py --config configs/agent.yaml \\
  --splits val_seen --episode-limit 8 --max-steps 128 \\
  --gpu-ids 0 --batch 4 --terminal-log off \\
  --output-dir /workspace/eval-batch4

# Two visible GPUs, two processes each (up to four independent episodes)
python -u scripts/run_api_agent.py --config configs/agent.yaml \\
  --gpu-ids 0 1 --batch 2 --terminal-log on \\
  --output-dir /workspace/eval-two-gpus
```

With `--batch 1` the legacy one-process-per-GPU behavior and worker filenames
are preserved. For `--batch 4 --gpu-ids 0`, files are named
`worker-0-b0.log` through `worker-0-b3.log`, with corresponding progress
markers and isolated caches. The software accepts 1-32 processes **per GPU**;
32 is an argument validation ceiling, not a claim that a GPU can run 32 Habitat
instances. Start at 2 on an 80GB H100; try 4 and 8 only after measuring actual
per-worker peak GPU memory, host RAM/CPU pressure and provider API throttling.
Avoid using the same output directory from two simultaneous parent launchers.
A rerun with changed --batch preserves completed episode traces, but uses new
worker shards and filenames.

`worker-<ID>.progress.json` and the per-episode
`api-inflight.json` are small live state markers, not proof of task success.


## 🚀 Getting Started

This walkthrough runs the released CMA checkpoint, records the actions actually
executed, and produces a validated submission ZIP. You may instead use any agent
that follows the [submission contract](#-submission-format).

### 1. Download data and the CMA checkpoint

Use Python 3 and `curl` on a Linux/WSL filesystem to download from
[Hugging Face](https://huggingface.co/datasets/fly1113/HA-VLN).
The helper pins resource revisions, verifies SHA-256 checksums, resumes interrupted
downloads, and extracts HAPS 2.0 into the required layout:

```bash
DATA_ROOT=/absolute/path/to/havln2-data
bash scripts/download_data.sh --destination "$DATA_ROOT" --target all
```

Use a Linux/WSL filesystem for HAPS 2.0 asset extraction. Place your licensed
Matterport3D scenes under `$DATA_ROOT/scene_datasets/mp3d/<scan>/<scan>.glb`.

### 2. Start a development container and install CMA dependencies

From this toolkit's root directory, create a named container with persistent
data and output mounts:

```bash
IMAGE=ghcr.io/jostarxiong/havln-challenge-2026@sha256:e1a0544f66beaf5218cc9df63da51b4a1a22a6bf0471ca0c2f75e81ee02a6556
WORK_ROOT=/absolute/path/to/havln-workspace
mkdir -p "$DATA_ROOT" "$WORK_ROOT"
docker pull "$IMAGE"
docker run --name havln-cma --gpus all -it --shm-size 16g \
  --mount type=bind,source="$DATA_ROOT",target=/data/havln2 \
  --mount type=bind,source="$WORK_ROOT",target=/workspace \
  --mount type=bind,source="$(pwd)",target=/toolkit,readonly \
  "$IMAGE" bash
```

Inside the container:

```bash
bash /toolkit/scripts/setup_cma.sh
havln-check-data
```

The setup script retrieves pinned public policy sources and installs the inference
dependencies while retaining the image's Habitat core. Re-enter with
`docker start -ai havln-cma` after exiting.

### 3. Run CMA and export the executed actions (Optional)

You do not need to use the CMA baseline for your competition submission; any model or planning approach that outputs valid action sequences is welcome. We provide this reference baseline walkthrough to illustrate how to export and verify a valid submission archive.

To run CMA inference and export action predictions for the validation splits:

```bash
python /toolkit/scripts/export_cma_submission.py \
  --output-dir /workspace/cma-submission
```

For multiple GPUs, append `--gpu-ids 0 1` (container-visible GPU indices). The exporter retains completed scan shards, so the same command can resume an interrupted run. Local multi-GPU subprocess logs and export metadata are saved alongside the results.

Completing all **778 `val_seen` and 1,839 `val_unseen` episodes** automatically packages `val_seen.json`, `val_unseen.json`, and `submission.zip`.

### 4. Validate and submit

Inside the container, verify your submission package:

```bash
havln-validate /workspace/cma-submission/submission.zip
```

Once validated, upload `$WORK_ROOT/cma-submission/submission.zip` on your host to [CodaBench](https://www.codabench.org/competitions/18135/).

## 📊 Dataset

The challenge is built on released HA-VLN 2.0 resources: HA-R2R navigation episodes,
HAPS 2.0 dynamic human assets, and licensed Matterport3D scenes. Challenge datasets
are mounted externally into the container.

| HA-R2R split | Episodes |
|:--|--:|
| Train | 10,819 |
| Validation Seen (`val_seen`) | 778 |
| Validation Unseen (`val_unseen`) | 1,839 |
| Test | 3,408 |
| **Total** | **16,844** |

- **HA-R2R**: 16,844 instructions across 90 scenes with multi-human interaction annotations.
- **HAPS 2.0**: 486 dynamic 3D human motion sequences across 172 activities.
- **Matterport3D**: Licensed indoor scene assets. Obtain access via the [Matterport3D project](https://niessner.github.io/Matterport/) and place extracted meshes under `$DATA_ROOT/scene_datasets/mp3d/<scan>/<scan>.glb` (see [FAQ Q7](#-frequently-asked-questions)).

## 🧠 Baseline Model

We provide **HA-VLN-CMA** as the reference baseline for Track 2. For model architecture and training details, refer to the official [HA-VLN repository](https://github.com/UWMILab/HA-VLN). Participants are free to use any model, planner, or VLA approach that outputs valid discrete action sequences.

Evaluating the released CMA validation checkpoint produces reference results:

| Split | SR ↑ | NE ↓ | CR ↓ | TCR ↓ | Score ↑ |
|:--|--:|--:|--:|--:|--:|
| `val_seen` | 0.180 | 6.223 | 0.623 | 13.726 | 16.509450 |
| `val_unseen` | 0.129 | 6.387 | 0.684 | 17.536 | 12.945387 |

*Note: Component metrics are rounded for display. All official rankings and leaderboard scores are determined by CodaBench replay evaluation.*

## 📏 Evaluation

| Metric | Direction | Meaning |
|:--|:--|:--|
| SR (Success Rate) | Higher | Collision-free success rate over all episodes. |
| NE (Navigation Error) | Lower | Mean final navigation error in metres. |
| CR (Collision Rate) | Lower | Collision-episode rate over human-influenced episodes. |
| TCR (Total Collision Rate) | Lower | Mean adjusted human-collision count over all episodes. |

### 🏁 Composite Score

The composite Score combines navigation and social compliance metrics (higher is better):

$$
\begin{aligned}
\mathrm{Navigation} &= 0.80 \times \mathrm{SR} + 0.20 \times \frac{3}{3 + \mathrm{NE}}, \\
\mathrm{Social} &= 0.75 \times (1 - \mathrm{CR}) + 0.25 \times \frac{1}{1 + \mathrm{TCR}}, \\
\mathrm{Score} &= 100 \times \mathrm{Navigation} \times (0.70 + 0.30 \times \mathrm{Social}).
\end{aligned}
$$

Higher Score ranks first. Exact ties use higher SR, lower NE, lower CR, lower
TCR, then earlier submission time. Calculation and comparison retain full
precision.

## 📥 Submission Format

Phase 1 requires one valid prediction JSON for each validation split. The
following ZIP is the recommended layout, not a filename restriction:

```text
submission.zip
├── val_seen.json
└── val_unseen.json
```

Each file follows this schema (the invented ID is only an illustration):

```json
{
  "format_version": 1,
  "split": "val_unseen",
  "episodes": [
    {
      "episode_id": "synthetic-example-001",
      "actions": ["MOVE_FORWARD", "TURN_LEFT", "LOOK_UP", "LOOK_DOWN", "STOP"]
    }
  ]
}
```

Include every official episode ID exactly once. `STOP`, if present, occurs once
at the end; sequences shorter than 500 actions require it. The examples in
[`examples/submission`](examples/submission) demonstrate schema only and do not
have complete manifest coverage.

The validator identifies predictions by each JSON object's `split` value, so
you may rename the files and include harmless extra root files. It rejects
missing or duplicate required splits, invalid prediction content, unsafe ZIP
members, and nested paths. Phase 2 follows the same rule with its single
required `test` split when that phase opens.

## ❓ Frequently Asked Questions

**Q1. Must I use CMA or a particular architecture?**

No. Any method is eligible if it exports legal official action sequences.

**Q2. Do I submit code, weights, or a container?**

No code, weights, or containers:
- **Standard Submission:** Upload only the required JSON action-sequence files in a single ZIP.
- **Award Winners:** Teams qualifying for awards (Top 5, Best Innovative Solution, or RoboPAD presentation) will be expected to contribute to a technical report and may be asked to provide code or models for verification.

**Q3. Are `LOOK_UP` and `LOOK_DOWN` valid?**

Yes. Both actions are valid and count toward the 500 task steps.

**Q4. Does an action advance a human animation frame?**

No. Human animation follows released wall-clock behavior independently.

**Q5. Does local validation compute a Score?**

`havln-validate` checks structure and coverage only. Local replay can compute
metrics for development, but only the official replay result published on
CodaBench for the submitted archive determines leaderboard scores and awards.
Results from another machine are not accepted as official scores.

**Q6. Why did my ZIP fail validation?**

`SUBMISSION REJECTED:` means the ZIP needs correction: check for a parent
folder, missing or duplicate split, duplicate/missing episode IDs (778
`val_seen` and 1,839 `val_unseen` are required), invalid action names, or
incorrect `STOP` placement. `PASS:` means validation or replay succeeded.
`SCORING DATA ERROR (organizer):` indicates a scoring-side problem; report
the submission identifier rather than changing valid predictions.

**Q7. Can the image download Matterport3D for me?**

No. Request access through the [Matterport3D dataset page](https://niessner.github.io/Matterport/):
sign its Terms of Use and email the form to `matterport3d@googlegroups.com`.
After approval, use the [HA-VLN 2.0 VLN-CE scene guide](https://github.com/UWMILab/HA-VLN/blob/main/agent/VLN-CE/README.md#scenes-matterport3d),
and place the licensed scenes under your host data root's
`scene_datasets/mp3d/`. That root can be anywhere on your machine. When using
the challenge Docker image, mount it at `/data/havln2` inside the container.

## 🔗 Resources and Contact

| Resource | Link |
|:--|:--|
| HA-VLN 2.0 | [Project page](https://uwmilab.github.io/HA-VLN-webpage/) & [GitHub](https://github.com/UWMILab/HA-VLN) |
| RoboWorld 2026 | [Challenge website](https://roboworld2026.github.io/) |
| Track 2 HA-VLN | [Track website](https://roboworld2026.github.io/track2) |
| CodaBench Platform | [Submissions and leaderboard](https://www.codabench.org/competitions/18135/) |
| RoboPAD Workshop | [RoboPAD @ NeurIPS 2026](https://robotpad2026.github.io/) |

For technical support, use [GitHub Issues](https://github.com/roboworld2026/track2/issues).
For event and registration questions, email [roboworld2026@gmail.com](mailto:roboworld2026@gmail.com).

### 💬 Community & Discussion

- **Discord:** [Join the RoboWorld Track 2 Discord](https://discord.gg/S8s8JcxtT)
- **WeChat Group:** [Join the RoboWorld Track 2 WeChat Group](https://github.com/roboworld2026/roboworld2026.github.io/blob/main/wechat_track2.JPG)

<a href="https://github.com/roboworld2026/roboworld2026.github.io/blob/main/wechat_track2.JPG" target="_blank"><img src="https://raw.githubusercontent.com/roboworld2026/roboworld2026.github.io/main/wechat_track2.JPG" alt="Track 2 WeChat Group QR Code" width="220" /></a>

## 📄 License and Terms

HA-VLN 2.0, HA-R2R, HAPS2.0, pretrained weights, and source assets retain their upstream
licenses. Matterport3D requires separate authorized access. Participation is
governed by the Terms displayed on the official CodaBench competition.

## 📚 Citation

If you use HA-VLN 2.0 or this challenge toolkit, cite the benchmark paper:

```bibtex
@inproceedings{dong2026havln,
  author    = {Dong, Yifei and Wu, Fengyi and He, Qi and Kong, Lingdong and Li, Heng and Li, Minghan and Cheng, Zebang and Zhou, Yuxuan and Sun, Jingdong and Dai, Qi and Hauptmann, Alexander G. and Cheng, Zhi-Qi},
  title     = {{HA-VLN 2.0: An Open Benchmark and Leaderboard for Human-Aware Navigation in Discrete and Continuous Environments with Dynamic Multi-Human Interactions}},
  booktitle = {2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
}

@misc{roboworld2026track2,
  title={Track 2 | HA-VLN: Human-Aware Vision-and-Language Navigation},
  author={RoboWorld Challenge 2026 Organizers},
  year={2026},
  howpublished={https://roboworld2026.github.io/track2}
}
```

## 🤝 Acknowledgements

The track is organized by the RoboWorld Challenge 2026 team and affiliated with
the RoboPAD Workshop at NeurIPS 2026. We thank the HA-VLN 2.0 and VLN-CE authors,
dataset and simulator contributors, Matterport3D, and CodaBench.
