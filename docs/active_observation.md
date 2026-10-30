# Active observation subtask (RoboWorld HA-VLN Track 2)

## Purpose

Previously, `turn_and_observe` returned to Navigation Agent after every 1–3
atomic turns or looks. On the stair episode this caused repeated scans of the
same camera positions and dozens of high-latency navigation LLM decisions before
the first forward step.

`observe_for_goal` delegates **a bounded visual search**, not movement.
Navigation Agent chooses the objective, static target types, left/right direction
and sweep area. ObservationController executes legal Habitat TURN actions and
collects every RGB-D frame. VLM only checks keyframes after 30 or 45 degrees,
with at most one VLM-suggested LOOK refinement per task. A medium/high-confidence
visible target with a bounding box and concrete evidence ends the task early.

No single VLM call can certify human safety, ground support or traversability.
Navigation Agent must use fresh Geometry/Terrain and local-waypoint checks before
any `MOVE_FORWARD`. An in-progress local navigation goal must be explicitly
cancelled before starting a wide scan; scanning does not silently undo alignment.

## Request

```json
{
  "tool": "observe_for_goal",
  "args": {
    "objective": "Find an accessible-looking downward stair entry",
    "target_types": ["stairs_down", "stair_entry"],
    "direction": "left",
    "sweep_degrees": 90,
    "checkpoint_degrees": 30,
    "look": "down"
  }
}
```

The two optional parameters default to `checkpoint_degrees=30`, `look=none`.
Allowed sweep sizes are 30, 60, 90 and 120 degrees (two to eight atomic
TURN actions on the current 15-degree camera calibration). Every TURN/LOOK
still counts toward `max_steps`; the underlying actuator reserves one STOP.
The regular `tool_max_actions=3` continues to constrain existing short
action tools; this macro has its own angle and refinement bounds.

## Response and lifecycle

The tool responds `status=OK` and
`outcome=TARGET_CANDIDATE_FOUND` with a `best_candidate` containing:
`target_type`, `confidence`, `box` (normalized `[left,top,right,bottom]`),
`evidence`, `observation_id`, `heading_deg`, `pitch_deg`,
`geometry.forward_status` and `usable_for_live_safety=false`.

If none is localized within the sweep, returns
`outcome=SEARCH_REGION_COVERED_NO_MATCH` with `checkpoints` and precise
`covered_turn_deg`; this does **not** stop the robot or prevent a later task.
Budget exhaustion or visual API failure returns `status=PARTIAL` with an
`INTERRUPTED` or `VISION_UNAVAILABLE` outcome, preserving executed actions.

Observation history and checkpoint evidence are recorded in `memory.json`,
tool ledger and episode traces. Revisited static views can reuse prior
vision hypotheses **only for the same objective and target types** within
the same subgoal and comparable pose. Reused evidence is explicitly marked
historical and never authorizes movement.

## Boundaries

- Task Planner: semantic sequence and milestone verification.
- Navigation Agent: which target to look for; where to scan; how to navigate
  afterwards.
- ObservationController: TURN/LOOK batching, coverage, novelty and VLM
  checkpoint timing. Never moves forward.
- Visual Perception: one frame → typed visual candidate and evidence,
  optional one-time pitch refinement request.
- Local Motion Controller and runtime safety: local target execution and
  **fresh** Depth / support / person checks before every forward step.
- Memory Service: view-history reuse and already-covered directions.

## Measure improvement

Compare equal-budget and same-seed diagnostic runs against smoke-500-14:
Navigation Agent decisions per 100 atomic actions, repeat-view share,
time/steps to first `MOVE_FORWARD`, safe forward actions,
localized target coverage, and verified stair descent.
A shorter run or fewer VLM calls alone is **not** task success.

## Diagnostic command

```bash
cd /toolkit
python -u scripts/run_api_agent.py --config configs/agent.yaml \\
  --output-dir /workspace/smoke-active-observation-100 \\
  --splits val_seen --episode-limit 1 --max-steps 100 --progress-interval 2
```

Use a fresh `--output-dir` after code changes. All real Habitat
end-to-end validation remains to be done on the user's simulator machine.
