# Two navigation modes (one existing YAML)

There are TWO independent modes:

| Mode | Command-line override | Navigation prompt | Execution |
| --- | --- | --- | --- |
| Pure cloud (legacy) | `--model-mode cloud` | Original `MISSION_PROMPT` (or original staged variants), untouched | All navigation actions and final STOP remain cloud-owned; no local calls or handoff fields |
| Hybrid cloud + local | `--model-mode hybrid` | Dedicated `hybrid_cloud_prompt.py` prompts | Cloud selects routes and resolves ambiguity; local follows explicitly delegated route segments and escalates |

Omitting `--model-mode` uses the existing `hybrid.enabled` in `configs/agent.yaml`
(default: `true`). No second YAML file is introduced. The override applies
to this run only, including all worker processes and output provenance, and
does not edit `agent.yaml`.

**Pure cloud** does not require `LOCAL_CONTROL_*` environment variables or
a running local vLLM service. **Hybrid** requires them. In both modes the
cloud navigation endpoint must accept current RGB image content.

Configure all three endpoints in the same `.env` for hybrid runs:

```dotenv
NAVIGATION_AGENT_BASE_URL=https://your-provider.example/v1
NAVIGATION_AGENT_MODEL=your-cloud-navigation-model
NAVIGATION_AGENT_API_KEY=your-cloud-navigation-key
VISUAL_PERCEPTION_BASE_URL=https://your-provider.example/v1
VISUAL_PERCEPTION_MODEL=your-cloud-vision-model
VISUAL_PERCEPTION_API_KEY=your-cloud-vision-key
LOCAL_CONTROL_BASE_URL=http://172.17.0.1:8000/v1
LOCAL_CONTROL_MODEL=havln-local
LOCAL_CONTROL_API_KEY=havln-local-key
```

The local address and model ID above are examples for the existing vLLM
deployment. They are not hardcoded in Python or YAML. Copy the additional
variables into your private `.env` file, keeping credentials out of Git.
To compare against the original API-only behavior without editing YAML,
pass `--model-mode cloud`. Alternatively set `hybrid.enabled: false` in the
existing YAML; the local endpoint is then ignored even if not configured.

## API-specific JSON formats

`configs/agent.yaml` is the only model configuration. Each of
`navigation_agent`, `visual_perception`, and `local_control` has an
independent `response_format` setting. `auto` now makes the **first**
request using a deterministic endpoint profile:

| Endpoint/model | First wire format | Navigation execution |
| --- | --- | --- |
| Official DeepSeek API or model ID containing DeepSeek | `json_object` | Two-stage tool selection and argument generation |
| Native-schema OpenAI / GPT-compatible model | `json_schema` | Native full tool union, staged only if independently shown unreliable |
| Local vLLM Qwen-VL | `json_schema` | Native full tool union, followed by strict local validation |

To override an endpoint, set its existing `response_format` in
`agent.yaml` to `json_schema`, `json_object`, or `omit`.
`omit` sends no `response_format` parameter but still requires valid
JSON and local Schema validation. Provider errors remain a last-resort
compatibility safeguard, **not the initial format selection strategy**.
A proxy with opaque model aliases should use an explicit format setting.

The navigation protocol requires top-level `progress_narrative`,
`scene_goal`, and `uncertainties`, not a nested `progress` object.
The previously observed DeepSeek nested form is normalized only for
known descriptive keys, with unknown/conflicting fields and executable
arguments still strictly validated.

The local model receives an observable cloud-issued route goal, about 10
source-verified RGB frames (one CURRENT RGB and up to nine historical anchors),
a CURRENT Depth visualization and the six most recent action receipts. Images
are referenced by immutable observation IDs, chronological RGB order, and
signed yaw relative to the CURRENT camera. The local model NEVER sees tool
definitions, chooses global routes, revises mission progress or issues STOP.

Outputs have only action/count/signal/evidence_id. The adapter converts short
FORWARD/TURN/LOOK commands to the existing calibrated runtime tools; the
current per-atomic Depth preflight and real simulator feedback are unchanged.
ESCALATE and all non-NORMAL signals bypass the executor and cause the next
cloud NavigationAgent turn. Repeated no-progress turns, blocked attempted
motion, malformed output and local inference errors also escalate.

Only the hybrid cloud prompt may produce this OPTIONAL top-level handoff:

```json
"local_handoff": {
  "goal": "Continue along the currently selected hallway",
  "exit_condition": "The hallway ends or another branch requires selection"
}
```
The router activates this handoff only after the cloud-proposed
`navigate_relative` actually completes a forward action. Merely observing a
scene, an incomplete move, a speculative waypoint, or a success/STOP claim
never grants local autonomy. Future cloud decisions revoke previous handoffs.

The hybrid cloud uses three dedicated prompts: complete native JSON,
staged tool selection, and staged typed arguments. In pure cloud all
three original prompts are preserved. The current local controller contract
stays `action/count/signal/evidence_id`; it cannot call tools or STOP.

On escalation, cloud receives the previous delegated route and an action
receipt summary: executed local forward/turn/look counts, source/current
observation IDs, and number of forwards with observed RGB-D change. These
receipts do NOT certify displacement, stair descent or semantic progress.

## Hybrid-3 route continuity and local control repair

The hybrid cloud now receives a small `hybrid_route_review` field **only
when** its current camera bearing differs by at least 105° from a salient
earlier **executed, changed-RGBD** forward segment. The field includes
real source observation IDs, a camera-relative view offset, and a caution
to compare the matching before/after RGB frames. It is not localization,
a motion veto, a verified retrace or a new persistent state machine.
Pure-cloud requests do not include this field.

The dedicated hybrid cloud prompts require positive current evidence that
descending treads actually ended on a connected landing before declaring
the stair descent complete or applying the instruction's following LEFT
turn. An instruction-relative exit turn is referenced to the travel
direction at that exit, **not** to arbitrary LOOK rotations made earlier.
When uncertain, the cloud retains control and rechecks the real images;
a local `UNCERTAIN` or `GOAL_CUE` signal does not prove arrival.

The local model is asked not to propose FORWARD when the current
`front_depth_status` is BLOCKED. For an obviously straight, stable route
it may use `count=2/3`; near stairs, people, corners, obstacles or unclear
clearance it should use `count=1`. It may make a small same-route
LEFT/RIGHT correction, but never pick a new branch. Every atomic forward
still undergoes independent live Depth preflight regardless of `count`.

These are evidence-presentation and prompt repairs; they do **not** prove
that the robot will correctly find the stair bottom. Compare real
closed-loop traces from the same Episode IDs and use separate output dirs
for each run. The cloud `reasoning: none` setting from PR #125 is retained.

## Example

The existing vLLM service from the local experiments is reachable from the
Habitat container through the Docker bridge:
```sh
CUDA_VISIBLE_DEVICES=0 vllm serve /home/cc/ccx/datadisk/models/Qwen/Qwen2.5-VL-7B-Instruct \
  --served-model-name havln-local --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --max-model-len 16384 \
  --limit-mm-per-prompt '{"image":16,"video":0}' \
  --api-key havln-local-key
```

With all three model endpoints configured in the same private `.env` file:
```sh
docker exec --env-file /path/to/.env -w /toolkit havln-cma \
  python /toolkit/scripts/run_api_agent.py \
  --config /toolkit/configs/agent.yaml --model-mode hybrid \
  --splits val_seen --episode-limit 4 --max-steps 128 \
  --gpu-ids 0 --batch 4 --terminal-log off \
  --output-dir /workspace/agent-runs/hybrid-eval
```

The local model is reached via the address configured in
`LOCAL_CONTROL_BASE_URL`. Adjust that `.env` value if the Docker
bridge address differs. No separate YAML edit is required.
The cloud and local request counts have independent budgets, and the
`api.jsonl` log contains both role-tagged requests (local uses role
`local_control`). `navigation_agent.jsonl` and `events.jsonl` retain the
cloud/local decisions and transition reasons. Summary includes
`api.cloud_requests`, `api.local_requests`, and `hybrid` counters.

Test on saved local traces and a small closed-loop subset before full Phase 2
evaluation. This change is not proof of navigation accuracy or latency gains.

### Pure cloud A/B comparison

With the same cloud credentials but without any local-model endpoint variables:

```sh
docker exec --env-file /path/to/cloud-only.env -w /toolkit havln-cma \
  python /toolkit/scripts/run_api_agent.py \
  --config /toolkit/configs/agent.yaml --model-mode cloud \
  --splits val_seen --episode-limit 4 --max-steps 128 \
  --gpu-ids 0 --batch 4 --terminal-log off \
  --output-dir /workspace/agent-runs/cloud-eval
```

Keep `--output-dir` separate between cloud/hybrid runs. Switching modes
changes the run fingerprint, so reusing prior outputs is intentionally rejected.
The default `hybrid.enabled` value can still be changed in the original
YAML; `--model-mode` takes precedence for the current invocation.

Suggested regression commands (no Habitat or paid API required):

```sh
python -m unittest discover -s tests -p 'test_navigation_structured_protocol.py' -v
python -m unittest discover -s tests -p 'test_hybrid_routing.py' -v
python -m unittest discover -s tests -p 'test_v32_phase2_prompt_api_regression.py' -v
```
