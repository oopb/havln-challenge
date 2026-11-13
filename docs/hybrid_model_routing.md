# Hybrid cloud/local navigation (single configuration)

Use the existing `configs/agent.yaml` for both cloud and local models;
no separate hybrid YAML is needed. This configuration enables hybrid
routing by default. The cloud model must support multimodal RGB and the
local VLM must accept OpenAI-compatible Chat Completions with `image_url`
content. Configure all three model roles using the same `.env` file:

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
To compare against API-only behavior, set `hybrid.enabled: false` in
`configs/agent.yaml`; leave the endpoint variables configured.

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

A cloud navigation proposal may include optional:
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
  --config /toolkit/configs/agent.yaml \
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
