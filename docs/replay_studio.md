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
