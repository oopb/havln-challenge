# Hybrid cloud/local navigation (opt-in)

Use `configs/agent-hybrid.yaml` rather than replacing the proven API-only
`configs/agent.yaml`. The cloud model must support multimodal RGB and the
local `havln-local` model must accept Chat Completions with image_url content.
Set `LOCAL_CONTROL_API_KEY` alongside the two existing cloud API keys.

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

With the normal existing cloud endpoint environment variables set:
```sh
docker exec --env-file /path/to/.env -w /toolkit havln-cma \
  python /toolkit/scripts/run_api_agent.py \
  --config /toolkit/configs/agent-hybrid.yaml \
  --splits val_seen --episode-limit 4 --max-steps 128 \
  --gpu-ids 0 --batch 4 --terminal-log off \
  --output-dir /workspace/agent-runs/hybrid-eval
```

The loopback model is reached via `172.17.0.1:8000` in this example.
Override the file's local_control.base_url if that host is inaccessible.
The cloud and local request counts have independent budgets, and the
`api.jsonl` log contains both role-tagged requests (local uses role
`local_control`). `navigation_agent.jsonl` and `events.jsonl` retain the
cloud/local decisions and transition reasons. Summary includes
`api.cloud_requests`, `api.local_requests`, and `hybrid` counters.

Test on saved local traces and a small closed-loop subset before full Phase 2
evaluation. This change is not proof of navigation accuracy or latency gains.
