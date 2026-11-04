# VLM-led active observation — RoboWorld HA-VLN 2.0

## Responsibility split

**NavigationAgent** is the strategic decision-maker: choose the current
language goal, decide whether new visual evidence is needed, specify the
observation objective and open-vocabulary targets, then decide if/where/how
far to navigate or whether to request an explicit geometry inspection.

**VisualPerception (VLM)** owns the **whole camera observation subtask**.
At each checkpoint it receives the current genuine RGB image, task and
goal, absolute current heading/pitch, remaining turn budget, and a short
history of previously inspected real views and prior target hypotheses.
It chooses the **next camera action** (turn left/right by 15–90 calibrated
degrees, look up/down, or finish), then inspects the actual fresh post-action
image. There is **no NavigationAgent inference between camera steps**.

**ObservationController** applies calibrated TURN/LOOK actions, persists
real frame/angle receipts, enforces pitch/turn/episode limits and prevents
returning to an already-inspected heading/pitch within the same task.
No forward movement, simulator ground-truth reads or automatic geometry
probes occur inside observation.

## Recommended tool call

```json
{
  "tool": "observe_for_goal",
  "args": {
    "objective": "Locate an actual approach or entry to the downward stairs",
    "target_types": ["stairs", "stair_entry"],
    "sweep_degrees": 180
  }
}
```

`direction` and `checkpoint_degrees` should be **omitted**. With direction
omitted, VLM chooses all next view bearings and tilt actions. Omitted
`sweep_degrees` defaults to **360°**, though VLM can stop as soon as it has
meaningful visual evidence. A legacy explicit `look` field may initialize pitch only through executor compatibility; ordinary NavigationAgent no longer sees it, so VLM owns every view/pitch decision.

For backward compatibility, explicit `direction=left|right` still uses the
legacy scripted sweep at fixed checkpoints. Ordinary NavigationAgent decisions
do not advertise this legacy variant or the standalone
`turn_and_observe` micro-turn tool (except orientation-only tasks).

## VLM output

`vlm_guided_observation` returns one structured visual checkpoint:

- `target_visible`, `target_type`, `confidence`, normalized `box`
- `evidence`, `uncertainty`, and `target_kind`:
  `ENTRY`, `APPROACH`, `TARGET` (localized object), `CUE`, `UNKNOWN`
- `next_action`: `turn_left`, `turn_right`, `look_up`,
  `look_down`, or `finish`; for turns, `turn_deg` is 15–90
- a short `reason` explaining the visual decision

A staircase seen beyond a railing can be a **CUE** without a verified
accessible entrance. Cues are preserved but do **not** automatically
yield `TARGET_CANDIDATE_FOUND`. The VLM should refine the view
when an entry remains uncertain. A localized, evidence-backed actionable
target yields the named observation ID, bounding box, heading, pitch and
current-relative bearing, with `usable_for_live_safety=false`.

Finding a candidate no longer ignores the VLM's `next_action`. If VLM
asks for another real view, the controller executes the legal LOOK/TURN
and revisits the image. This continuation is bounded to three extra VLM
checkpoints after the first candidate; the best source-labeled candidate is
preserved if VLM finishes or reaches that budget. This is an observation-cost
bound only, not a limit on NavigationAgent's MOVE actions.

A large return sweep (90 degrees or greater) that simply reverses a prior
sweep toward a yaw physically visited in the SAME observation subtask is
returned to VLM for correction when the cumulative turn is at least 180
degrees. Its history labels *physically visited yaw* separately from
*VLM-inspected image*, so intermediate TURN frames do not become invented
semantic evidence. The model may pick a new pitch, new view, or finish.

## Returned result and safety

`observe_for_goal` returns one consolidated tool receipt with
`outcome`, `best_candidate` or `best_cue`, real view references,
`covered_turn_deg` and `observation_owner=visual_perception_vlm`.
A found ENTRY, APPROACH, TARGET or CUE is stored persistently in shared
`visual_targets` with the genuine observation ID, image box, absolute heading,
camera pitch, VLM evidence and a historical/uncertain evidence label. The next
NavigationAgent decision receives **only compact evidence**,
not all raw checkpoint JSON. Full details are retained in diagnostics
and requestable memory records.

VLM has no power to invoke `MOVE_FORWARD`, `STOP`, physical safety
clearance or semantic task completion. Geometry and terrain checks remain
an **explicit NavigationAgent tool choice** in this simulator-specific
experimental pipeline, and Habitat remains responsible for its physics.

A repeated requested view triggers up to two same-frame VLM correction calls
with explicit rejected-action feedback and the inspected yaw/pitch map. This
never calls NavigationAgent or spends a TURN for the rejected camera action.
Repeated failed corrections, budget exhaustion or unavailable visual API
return a structured non-success receipt rather than making up target coordinates
or starting a forced movement. Existing in-progress navigation targets
are preserved by camera observation.

## Token/cost measurement

The API ledger now reports `system_chars`,
`tool_protocol_chars` and `context_chars_before_encoding`, alongside
provider token usage per role. Evaluate the same episode and compare:

- NavigationAgent requests per 100 atomic actions
- navigation input tokens / successful forward attempt
- VLM requests and visual input tokens
- repeated views, first forward time/step, observed stair entry evidence
- physical progress, obstacle collisions and verified task completion

Fewer model calls alone do **not** imply better navigation.

## Verification

```bash
python -m unittest discover -s tests -p 'test_vlm_led_observation.py' -v
python -m unittest discover -s tests -p 'test_active_observation.py' -v
```

Closed-loop Habitat measurements still require rerunning on the user's
simulation host.
