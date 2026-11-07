# v11 — observation reliability and model input/output review

**Baseline preserved:** The 88-step `eval-batch4-25` episode was completed with
20 MOVE_FORWARD actions, 63 turns (31 from VLM-guided scans, 22 from
NavigationAgent's turn_and_observe, 10 as part of relative navigation),
and no identical before/after RGB-D on executed forward actions. This
version intentionally preserves Depth forward preflight, camera
re-centering, and NavigationAgent's authority over movement and STOP.

## VLM JSON output fix
Six of 22 observation API attempts in this episode returned duplicate
JSON keys. Five duplicated `target_visible` with the **same** value;
one duplicated `uncertainty` with **different** text. Previous strict
decoding rejected all six.

The v11 parser continues to reject duplicate keys by default. Only
`role=visual_perception` and
`purpose=vlm_guided_observation` may normalize identical duplicate
values. The values must have the same Python/JSON type *and* the same
canonical JSON serialization. This prevents Python's `True == 1`
comparison from admitting boolean/integer conflicts. Other roles,
conflicting duplicates, malformed JSON, and any result failing the
existing independent schema/coordinate/role validation are **rejected**.
No rejected model action is dispatched. The audit logs each benign
normalization as `response_normalization=identical_duplicate_keys[...]`.
Existing one-shot VLM format repair remains for truly conflicting output,
with a more precise correction instruction.

## Observation input and turn-budget feedback
Both agents receive a read-only
`mission_memory.recent_views_at_current_stance`: latest distinct actual
yaw/pitch observations at the same local estimated stance and motion
epoch, with source observation IDs and relative bearing from the current
heading. It is bounded to 16 distinct views in the latest 80 records.
It is **not** a route ban, an automatic motion command, or a safety
certificate. A pose change/uncertain odometry cannot be assumed to be the
same stance.

During a VLM observation session,
`request.view_coverage` additionally reports number of inspected
views, actual TURN degrees consumed, direction-reversal count and
remaining sweep allowance. ObservationAgent's prompt is asked to
reuse observed bearings and return `next_action=finish` when a concrete
route hypothesis is available, rather than rescanning by default.
NavigationAgent's prompt similarly treats scans as costly, asks for
short, model-selected forward progress where supported by current
evidence, and does **not** claim stair completion from height estimates.
No hardcoded stairs, forced-forward quotas or Python steering added.

A camera scan with no remaining yaw budget now offers `finish` and
physically possible LOOK options, **not** `turn_left/right` with an
impossible zero-degree turn.

## Multimodal conversation and logging audit
`VisualPerception.guide_scan_view` already supplies current RGB and
Depth, optionally three source-backed historical RGB-D pairs on the
first turn, and resends the full earlier task-local conversation on
subsequent VLM requests. v11 preserves that behavior. It corrects
`api.jsonl` input-size reporting to count *prior conversation* text
and images, while continuing to store only safe metadata and hashes,
never replayed image base64. Added
`prior_conversation_turns`, `prior_conversation_text_chars` and
`prior_conversation_image_count` metrics.

## Regression checks
```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_v11_observation_review.py' -v
python -m unittest discover -s tests -p 'test_mission_navigation_architecture.py' -v
python -m unittest discover -s tests -p 'test_mission_history_and_headings.py' -v
python -m unittest discover -s tests -p 'test_recoverable_camera_pitch.py' -v
python -m unittest discover -s tests -p 'test_navigation_blocked_history_contract.py' -v
```

Use a new `eval-batch4-26` output directory to avoid mixing code
fingerprints. v11 only claims *protocol/observation* repairs until the
new simulator run verifies the effect on action mix, useful forward
motion, observation calls, and official completion.

---

# v10: recoverable camera pitch limits (eval-batch4-24)

**Root cause:** At -60° pitch the v9 navigator twice emitted a structurally
valid `turn_and_observe(...,look="down")`. The dynamically generated
`look` enum had removed `down`, so the response was incorrectly treated
as an API/schema failure; the scheduler exhausted its single format retry
and terminated with `api_error` after only 22 atomic actions.

**Fix:** `turn_and_observe` now always exposes `look=none|up|down`, regardless
of pitch. `navigation_context().camera_controls` independently reports
the **current** allowed LOOK directions, pitch (-60..+60), and calibrated
tilt step. Tool execution checks the requested pitch **before any model-
requested TURN** or LOOK. An out-of-range LOOK returns:

```json
{
  "status": "CAMERA_PITCH_LIMIT",
  "recoverable": true,
  "reason": "requested_look_exceeds_camera_limits",
  "requested_look": "down",
  "current_pitch_deg": -60,
  "requested_pitch_deg": -90,
  "pitch_limits_deg": [-60, 60],
  "allowed_look": ["none", "up"],
  "executed_camera_actions": []
}
```

The requested pitch in this example assumes a 30° tilt step. No physical
action is executed, and the simulator pose and action budget remain
unchanged. The scheduler sees a **normal tool receipt**, not an API error,
and calls NavigationAgent again for a model-owned decision. Identical
pitch-limit errors are counted in bounded `recent_camera_pitch_limits` and
also visible in `recent_tool_receipts.camera_pitch_limit`; no Python
steering or automatic camera commands are introduced.

Truly malformed JSON, missing fields, and illegal enum values remain
schema errors, now with more specific field details when available.
Observation Agent's independent camera scan schema is unchanged.

**Regression:**
```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_recoverable_camera_pitch.py' -v
```

Tests cover both pitch bounds, unchanging schema, **zero action** on
`RIGHT 30 + down` at -60°, valid LOOK at bounds, source-backed error
memory, and navigation-scheduler continuation past a rejected command.

**Run ID:** use `eval-batch4-25` or a fresh directory for a comparison.
The v10 prompt protocol is
`havln-agent-v10-recoverable-camera-pitch`.
This is a focused robustness change; it does not alter v9 Depth or
forward-motion heuristics.

---

# v9: pre-decision Depth, motion camera alignment and physical failures

This update addresses the `eval-batch4-23` failure mode: multiple accepted
`MOVE_FORWARD` commands with byte-identical before/after RGB-D, too many
rotations, and repeated attempts at the same local obstacle. It does not infer
Habitat collision states from RGB-D, force a route, or add stair-specific
rules.

## Runtime decision and execution order

```text
Refresh current public RGB-D
    -> forward_preflight(stage=navigation_decision) [no action, no VLM]
    -> NavigationAgent receives current RGB, CLEAR/BLOCKED/UNKNOWN,
       measured centerline obstacle ROI and the unified Memory projection
    -> NavigationAgent independently selects movement / observation / turn
    -> Movement tool executes model-authored TURN steps (if any)
    -> Before each MOVE_FORWARD:
        - Physically LOOK_UP/LOOK_DOWN toward 0° iff camera is tilted
          (actual Habitat actions and budget, no synthetic angle change)
        - Refresh public RGB-D
        - Reuse identical forward_preflight logic (stage=atomic_forward)
        - BLOCKED central body evidence: do not issue MOVE; return ROI
        - CLEAR or UNKNOWN: ordinary model-requested motion can attempt
    -> After real MOVE: compare actual RGB and Depth
        - unchanged RGB-D: STALLED_AFTER_FORWARD, no collision claimed;
          interrupt current finite route and return to model
        - changed RGB-D: registration estimates remain uncertain evidence
```

The pre-decision scan is only valid for the **current** heading and pitch.
It cannot certify future headings. Each post-turn step uses a new live
Depth reading. A CLEAR reading does not guarantee Habitat can translate,
and UNKNOWN is not treated as blocked. High-confidence central obstacle
pixels still conservatively veto the attempted atomic forward; a side-only
wide-depth warning alone does not.

## One measured ROI schema in both places

`havln_agent/forward_preflight.py` emits the exact same:
`status`, `source_observation_id`, RGB/Depth hashes, yaw and pitch,
`blocker.focus_region_xyxy_norm`, `blocker.source_pixel_xy`,
`blocker.nearest_along_m`, `clearance_m` and `path_state`.
NavigationAgent gets it under `front_depth_preflight` before inference.
Observation Agent gets `current_front_depth_preflight` with its real RGB-D.
Executed tools return the same blocker under
`obstacle_observation.depth.*`, and `mission_history.py` now reads this
**flat** Depth dictionary rather than incorrectly indexing `depth.center`.

## Physical failure Memory

`EvidenceLedger` separately records:

- `BLOCKED_BEFORE_FORWARD`: measured central obstacle; simulator forward
  **not issued**.
- `STALLED_AFTER_FORWARD`: simulator accepted `MOVE_FORWARD` but RGB and
  Depth were byte-identical before/after; **not** proof of a wall/collision.

Additionally, `observed_front_blocks` stores a **third, read-only sensor fact**:
front Depth blocked during a NavigationAgent or Observation Agent preflight,
even when no forward action was attempted. It is not labeled as an
executed collision or a rejected forward action. It remains source-frame
specific and is capped and deduplicated.

Histories deduplicate repeated identical RGB/Depth hashes **plus** local
yaw, pitch, estimated position, motion epoch and viewpoint generation,
not just `observation_id`. Distinctive visual and positional failures remain
separate. Models see repeat counts, source frame and actual evidence without
automatic Python steering/recovery rules. The original whole-mission
navigation and Observation Agent roles are unchanged.

Camera alignment occurs only if camera pitch is nonzero when the finite
model-authored route reaches a MOVE_FORWARD step. Subsequent forward steps
have no alignment overhead if already level. LOOKs count toward the
competition's atomic action budget. If camera alignment cannot complete,
motion is not attempted and the model receives an interruption.

## Local smoke test

```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_mission_navigation_architecture.py' -v
python -m unittest discover -s tests -p 'test_mission_history_and_headings.py' -v
python -m unittest discover -s tests -p 'test_forward_preflight_v9.py' -v
```

Use a new output directory (for example `eval-batch4-24`) and verify
`forward_depth_preflight`, `camera_recentered_for_motion`,
`pre_forward_obstacle_reported`, and `forward_motion_stalled` events,
plus per-role API costs, valid forwards, and mission completion. The
protocol identity is `havln-agent-v9-live-forward-depth-preflight`.

---

# v8: explicit turn directions and canonical mission history

This update fixes problems observed in `eval-batch4-22` after the v7
whole-mission refactor. It does **not** introduce sg/VERIFY stages or
automatic navigation steering. It changes the model/tool interface and the
provenance of memory supplied to both agents.

## Source of truth and what each model receives

```text
Habitat public RGB-D + atomic action receipts
  -> MemoryRecords / MemoryService (observations, motions, blocks)
  -> mission_history.mission_evidence() [one bounded evidence index]
        |-> contexts.navigation_context() -> NavigationAgent + current RGB
        |-> visual_perception.guide_scan_view() -> Observation Agent + RGB-D
        |-> query_task_memory() -> on-demand expanded same index
```

- **Facts:** RGB-D observation IDs, source headings and pitch, current local
  pose/motion epoch, actual accepted actions, registered motion estimates,
  and measured near-field obstacle ROI. Historical Depth is not authorization
  for current movement; registration is not Habitat ground truth.
- **Visual hypotheses:** Observation Agent checkpoint summaries and box
  proposals are linked to real source observation IDs and camera poses.
  A waypoint's heading is computed from calibrated image-box center and
  camera horizontal field of view, not merely the camera centerline.
- **Navigation assessments:** `progress_narrative`, `navigation_intent`,
  uncertainty and model-declared citations are *revisable*. The context
  observation ID is recorded as the frame at which the model decided; it
  is **not** automatically counted as verified task-completion evidence.
- **Observation history:** A single observation task preserves the complete
  turn-by-turn RGB-D conversation. New tasks use a diverse, bounded selection
  of actual saved historical frames across previous observation tasks.
  Historical frames appear with source ID, yaw/pitch, local motion epoch,
  and summary; the selection never implies physical reachability.
- **Deduplication:** Same-frame, same-stance blocked-forward receipts are
  summarized once with a repeat count. Unchanged narrative revisions are
  not added as new memory records. Every observation task gets a stable
  `observation_XXX` memory ID for explicit retrieval and forensic replay.

## Unambiguous model turn API

NavigationAgent tools now use:

```json
{"tool":"navigate_relative","args":{
  "turn_direction":"RIGHT","turn_degrees":30,"distance_m":0.5}}
```

and for camera-only turns:

```json
{"tool":"turn_and_observe","args":{
  "turn_direction":"LEFT","turn_degrees":45,"look":"none"}}
```

`LEFT/RIGHT` require a positive, nonnegative magnitude;
`NONE` requires exactly zero. The executor maps directions to internal
signed yaw. These are **breaking tool-schema changes** from
`navigate_relative(turn_deg=...)` and
`turn_and_observe(relative_turn_deg=...)`; do not reuse old cached runs.

The new pipeline identity is
`havln-agent-v8-consistent-mission-evidence`. Use a fresh output directory
and compare forward actions, blocker repetition, VLM calls and objective
scoring across at least val_seen and val_unseen.

## Local checks

```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_mission_navigation_architecture.py' -v
python -m unittest discover -s tests -p 'test_mission_history_and_headings.py' -v
```

Observe `mission_context_built` events to confirm the exact evidence
observation IDs and source task IDs included with each NavigationAgent
decision. Each Observation Agent request separately logs its source
observation ID and prior conversation-turn count in API diagnostics.

---

# HA-VLN: continuous natural-language mission navigation (v7)

This is the active inference architecture after the complete task-control refactor.
It replaces the previous numbered `sg_0` / `sg_1`, TaskPlanner, forced
`VERIFY` and stage-completion/reopening protocols. The official external
HA-VLN scorer remains unchanged.

## Roles and authority

- **NavigationAgent:** receives the entire instruction and current authentic
  RGB, pose/actuator state, natural-language progress and intent, recent
  registered RGB-D motion evidence, authentic obstacle evidence, and recent
  observations/waypoints. It decides when to observe, execute a finite motion
  segment, continue a partial segment, or STOP. It can revise its own progress
  narrative without reopening task IDs.
- **Observation Agent:** receives the entire instruction, current narrative,
  and a **natural-language observation delegation**. During one observation
  session, every vision call retains all previous authentic RGB-D views,
  previous VLM replies, absolute headings and pitch. It autonomously performs
  legal TURN/LOOK, but never MOVE or STOP. It returns narrative observation,
  navigation suggestion and optional source-anchored local waypoint hypotheses.
- **Memory:** the single home for sensor facts, model interpretations,
  observation history and movement feedback. Sensor frames/measurements and
  navigation hypotheses remain distinguished by provenance. A candidate
  waypoint is never an independent safety certificate or task-completion
  verdict.
- **ToolExecutor / Habitat:** executes model-chosen atomic actions within the
  official action budget and actuator limits. It may interrupt a blocked
  near-field forward step using current central Depth evidence. No goal-specific
  stair rules, automatic post-step milestone verification, `candidate_id`
  prediction scoreboard, or automatic reopening of historical tasks exist.

## Example mission and delegation

Original instruction:

> Go downstairs, find the hall to the right, cross the group of people,
> and stop beside the piano.

Example revisable Memory:

```json
{
  "progress_narrative": "I appear to have descended the staircase; the hall entrance is not yet confirmed.",
  "navigation_intent": "Find and enter the right-side hall, then approach the piano.",
  "uncertainties": ["Whether the last staircase tread has been passed"],
  "evidence_observation_ids": ["obs_0066"]
}
```

Observation request:

> The complete mission is to descend, find the right-side hall, cross the
> group of people and stop at the piano. I believe the descent is mostly
> complete. Inspect the current RGB-D layout and identify a useful direction
> toward the hall, or point out remaining descending stairs.

Observation response (illustrative, **not** a measured route):

```json
{
  "observation_summary": "A doorway is visible to the right; distance remains uncertain.",
  "navigation_suggestion": "Try a short approach toward the opening.",
  "waypoints": [{
    "source_observation_id": "obs_0066",
    "heading_at_capture_deg": 90,
    "absolute_heading_deg": 90,
    "relative_bearing_deg_from_current": -45,
    "distance_m_hint": null,
    "traversability": "UNCONFIRMED",
    "motion_authorized": false
  }]
}
```

The final relative bearing must always be recalculated against the robot's
**current** heading; saved historical views cannot grant live clearance.

## Execution

The entry point is `scripts/run_api_agent.py`. Two independent vision-capable
roles are configured: `navigation_agent` (current real RGB on each decision)
and `visual_perception` (real RGB and calibrated Depth). The retired
`task_planner` API key and YAML section are not required.

Pipeline identity: `havln-agent-v7-mission-narrative`. Its output directories
must not be mixed with earlier v6 outputs. Existing batch, GPU, terminal-log,
critical-frame capture and diagnostic packaging options remain in place.

To test Python imports/schema and new architecture contracts:

```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_mission_navigation_architecture.py' -v
```

For a real Habitat smoke test, use a **fresh output directory**:

```bash
python /toolkit/scripts/run_api_agent.py \
  --config /toolkit/configs/agent.yaml --splits val_seen \
  --episode-limit 1 --max-steps 88 --gpu-ids 0 --batch 1 \
  --terminal-log on --archive-frames critical \
  --output-dir /workspace/agent-runs/mission-v7-smoke
```

The new Memory records and action trace establish what was executed, **not**
that the full destination was reached. Run `val_seen` and `val_unseen`
on the actual simulator and compare full-mission scoring, forward fraction,
collision metrics, API use and early STOP before asserting performance gains.

The old tests dedicated to numbered milestones, `VERIFY`, or Planner
decomposition test a retired contract; they need to be migrated before
treating the entire legacy test suite as an acceptance gate.
