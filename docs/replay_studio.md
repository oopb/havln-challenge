# HA-VLN Replay Studio

A read-only, offline trajectory/debugging interface for **completed**
`run_api_agent.py` episodes. No API keys, inference, Habitat simulator,
or GPU are needed to view a run.

## Launch from the server host (recommended)

Use the **host path** corresponding to the persisted Docker `/workspace`
mount and a Python environment with NumPy, Pillow and the existing agent
dependencies. Launching on the host makes SSH port forwarding straightforward:

```bash
cd ~/ccx/havln-challenge
python scripts/serve_agent_replay.py \
  --output-dir /HOST/PERSISTENT/WORKSPACE/agent-runs/eval-batch4-30 \
  --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` in the host's browser. For remote servers,
use an SSH port forward instead of binding the viewer publicly:

```bash
ssh -L 8765:127.0.0.1:8765 USER@SERVER
```

Alternatively, open an archived run without extracting it. If you launch the viewer **inside** an existing Docker container, its localhost is not the server host's localhost; arrange a container port publish or other private forwarding first:

```bash
python /toolkit/scripts/serve_agent_replay.py \
  --archive /workspace/diagnostic-archives/RUN-diagnostic.zip
```

To generate an **all-frame replay archive** for sharing:

```bash
python /toolkit/scripts/pack_agent_diagnostics.py \
  --output-dir /workspace/agent-runs/eval-batch4-30 \
  --archive /workspace/agent-runs/eval-batch4-30-replay.zip \
  --frames all
```

The default automatic `--archive-frames critical` diagnostic ZIP is
intentionally sparse. It may omit views that were available in the original
run directory. The viewer reports missing frames rather than synthesizing
or silently replacing them. Old traces lack full observation-session
conversation messages: the viewer marks them as incomplete.

## Interaction

- Select a split/episode, then choose **Model calls**, **Atomic actions**,
  **All events**, or **Issues**. Previous/next and arrow keys move through
  the selected filtered sequence. The slider and numbered jump reach any
  entry.
- Main pane shows **real recorded RGB**, visualized calibrated Depth,
  and, for an atomic action, its actual post-action RGB.
- Model-call pane shows the original system prompt and user context from
  `api.jsonl`, full historical user/assistant messages, complete response
  schema / request parameters, current and prior attached media, raw model
  response, validated result, provider diagnostics, retry/fallback errors,
  and cache hits. Historical media are clearly distinguished from current
  sensor views.
- Tool entries show the requested arguments and actual feedback; raw
  underlying rows remain inspectable.
- In a newly recorded run, `replay/events.jsonl` gives the exact order
  in which records were committed. Old runs lack that clock; their
  cross-stream ordering is explicitly marked **approximate**, with each
  original log stream retaining its own order.

## RoboWorld Track 2 score calculation

The viewer includes the exact published [Track 2 composite score rule](https://roboworld2026.github.io/track2):

```text
Navigation = 0.80 * SR + 0.20 * 3/(3+NE)
Social     = 0.75 * (1-CR) + 0.25 * 1/(1+TCR)
Score      = 100 * Navigation * (0.70 + 0.30*Social)
```

**Score is a split-level aggregate, not a mean of per-episode scores.**
Compute `SR, NE, CR, TCR` over the evaluated episodes **before**
applying the nonlinear formula. NE is in metres, SR/CR are fractions
(not percentages), and TCR is the adjusted mean collision count.
The scoreboard shows Score, four metrics, Navigation, Social and the
social penalty factor; when separately evaluated episode evidence is
available, it also displays the selected episode's true navigation
error, no-extra-collision success, and adjusted collisions.

The existing `summary.json`, `steps.jsonl`, `api.jsonl` and RGB-D
frames **cannot** establish ground-truth goal distance, simulator
collision events or the reference-route collision allowance. They
are never silently treated as SR/NE/CR/TCR; a missing metric report
produces **not available**, not zero. CodaBench alone determines official
rankings, and even a complete local evaluation is marked offline.

### Import independently evaluated metrics (no copy required)

Pass a read-only scorer result alongside an existing run:

```bash
python scripts/serve_agent_replay.py \
  --output-dir /HOST/WORKSPACE/agent-runs/eval-batch4-30 \
  --metrics-file /HOST/WORKSPACE/metrics/validation-score.json
```

Alternatively, place a single `score_metrics.json` at the **run root**
(the same level as `traces/`). The viewer discovers it automatically,
and the diagnostic packer already includes root-level JSON files.
Never store a second copy if `--metrics-file` already points to the
original evaluation report.

Supported format A: metrics already computed independently by a scorer:

```json
{
  "splits": {
    "val_seen": {
      "metrics": {"SR": 0.180, "NE": 6.223, "CR": 0.623, "TCR": 13.726}
    },
    "val_unseen": {
      "metrics": {"SR": 0.129, "NE": 6.387, "CR": 0.684, "TCR": 17.536}
    }
  }
}
```

*Numbers above illustrate the published CMA baseline's rounded **display**
values, not an evaluation of your agent. Recomputing a baseline Score from
rounded inputs will not reproduce the official full-precision Score.*

Supported format B: per-episode **independently evaluated** metrics:

```json
{
  "split": "val_seen",
  "episodes": [
    {
      "episode_id": "3",
      "navigation_error_m": 1.9,
      "success": true,
      "collisions_count": 4,
      "reference_collision_count": 2,
      "human_influenced": true
    }
  ]
}
```

Alternatively each episode can provide `adjusted_collision_count` in
place of both raw collision counts. The scorer computes per-episode
adjusted TCR = `max(0, collisions_count - reference_collision_count)`,
SR = `success AND adjusted_TCR==0`, CR indicator = `adjusted_TCR>0`,
then aggregates all valid episodes. For CR, when each episode has a
`human_influenced` label, the denominator is the number of marked
episodes. Otherwise the official full-split beta coefficient is used
for an **approximate small-sample denominator**, and the UI labels the
result diagnostic/provisional. The source values must come from the
independent simulator/scorer, not the NavigationAgent's own success
claim or a geometry obstacle report. For 778 `val_seen`, 1,839
`val_unseen` and 3,408 `test` tasks, separate splits are evaluated
independently; Phase 2 ranks by the test Score.

Users can expand **评分拆解、单条任务依据与公式试算** and enter four
metric values to evaluate the formula immediately. The preview is
calculated by the server, is never written to the run, and is explicitly
marked non-official. Enter `SR=0.18`, not `18` for 18% success.

Run the pure score unit tests with:

```bash
python -m unittest discover -s tests -p 'test_competition_scoring.py' -v
```

## Storage contract

The canonical, already-existing files remain authoritative:

```text
traces/EPISODE/
  observations/obs_XXXX.npz   # one RGB-D frame, original sensor arrays
  steps.jsonl                 # one actual atomic-action ledger
  api.jsonl                   # one API request/result ledger
  tools.jsonl                 # one tool-call ledger
  episode.json                # includes camera calibration for new runs
  replay/events.jsonl         # small sequence of references (no copied rows)
  replay/media/SHA256.png     # only images that cannot match canonical NPZ
```

Every `api.jsonl` request has a `replay` object in **new** runs. The
already logged `messages` field remains the source for system/current-user
text. The new data adds only the prior dialogue, current image refs and
request parameters. Each image is verified byte-for-byte against derived
PNG renditions of the canonical observation (RGB original / RGB 640 /
Depth 640). Exact matches are stored as references to the NPZ and a
rendering recipe, **not a duplicate PNG**. Unmatched media are saved
once by SHA-256, without embedding repeated Base64 in logs. The viewer
never fabricates true XYZ coordinates or physical displacement from the
action sequence.

The replay layer never changes model prompts, schema validation, tool
decisions, movement execution, or official action export. Unavailable
replay data is a diagnostic limitation, not a navigation policy.

## Test

```bash
python -m unittest discover -s tests -p 'test_replay_studio.py' -v
python -m unittest discover -s tests -p 'test_compact_diagnostics.py' -v
```

For frontend manual acceptance, load a new full trace and confirm every
actual API attempt (including failed requests), every forward/turn/look,
and every tool feedback is indexed; inspect multiple VLM dialogue turns,
jump to an arbitrary action and compare its pre/post images. Contrast
the same run's complete archive with its original directory to confirm
that no frame is copied unnecessarily.
