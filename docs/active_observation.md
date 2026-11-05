# VLM-led active observation — RoboWorld HA-VLN 2.0

## Responsibility split

**NavigationAgent** is the strategic decision-maker: choose the current
language goal, decide whether new visual evidence is needed, specify the
observation objective and open-vocabulary targets, then decide if/where/how
far to navigate or whether to request an explicit geometry inspection.

**VisualPerception (VLM)** owns the **whole camera observation subtask**.
At each checkpoint it receives the current genuine RGB image AND a calibrated
rendering of PUBLIC Depth, task and
goal, absolute current heading/pitch, remaining turn budget, and a short
history of previously inspected real views and prior target hypotheses.
It chooses the **next camera action** (turn left/right by 15–90 calibrated
degrees, look up/down, or finish), then inspects the actual fresh post-action
image. There is **no NavigationAgent inference between camera steps**.

**ObservationController** applies calibrated TURN/LOOK actions, persists
real frame/angle receipts, enforces pitch/turn/episode limits and prevents
returning to an already-inspected heading/pitch within the same task.
No forward movement or simulator ground-truth reads occur inside observation.
A **model-proposed approach heading** does trigger one local, conservative
PUBLIC Depth central-occupancy probe on that CURRENT viewing bearing.
This is not full terrain or route geometry certification; only confirmed
centerline occupancy is returned to the VLM as a blocked candidate.

## Recommended tool call

```json
{
  "tool": "observe_for_goal",
  "args": {
    "objective": "Find the descending stairs and inspect whether the visible treads end at a flat landing or floor. Distinguish a nearby stair-to-floor transition from a distant floor view; report uncertainty.",
    "target_types": ["stairs", "landing", "floor"],
    "sweep_degrees": 180
  }
}
```

The `objective` now states **both** the current visual question and the
visible cue(s) one might expect near the END of the current semantic milestone,
if a meaningful endpoint is visually identifiable. NavigationAgent writes
this reference in the **existing objective string**; the new RGB-D approach
agent reuses this same tool signature, but adds a model-visible
`candidate_approach` flag to its internal scan output. For a descent,
the cue might be descending treads transitioning to level flooring; for a
doorway, the visible opening/interior beyond the door. The VLM checks only
the current real RGB-D evidence and reports absence or uncertainty rather than
assuming the endpoint is present. Seeing a distant floor or room is NOT proof
that the robot has reached it. This also must NOT make the VLM suppress other
useful visual facts or unilaterally change the goal. Only NavigationAgent
interprets the results with actual motion history.

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
when an entry remains uncertain. **Direct VLM evidence relay (Batch 4-13):** In VLM-led scans,
`target_type` is a **free-form visual description**, not an enum that must
exactly match NavigationAgent's `target_types`. No intermediate string
match, minimum confidence, or required bounding-box predicate can suppress a
VLM report. ObservationController records structured `visual_reports`
from real inspected frames: observation ID, yaw, pitch, target visibility,
free-form type, target kind, confidence, box (or null), visual evidence,
uncertainty, and the suggested camera action. The latest eight compact
reports appear in NavigationAgent's `latest_observation_task.visual_reports`
with a current-relative bearing; all original reports remain in diagnostics.
Negative and possibly mistaken visual hypotheses are also relayed. Best
candidate/cue fields are optional conveniences, not information gates.
Reports never prove physical clearance or subgoal completion.

A localized, evidence-backed actionable
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
an **explicit NavigationAgent tool choice** for FULL diagnostics. A small
central public-Depth candidate probe runs inside observation only after VLM
proposes the CURRENT robot yaw as a possible approach; Habitat remains
responsible for its physics.

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


## RGB-D local approach-search loop (November 2026)

Every genuine VLM checkpoint receives two ordered images:

1. Current RGB.
2. Current public Depth, rendered as grayscale in **meters** over [0, 5] m
   (near=black; far=white; farther than 5 m clips to white; missing/invalid
   pixels are magenta). RGB/Depth are related by normalized image coordinates;
   do NOT assume identical pixel resolution or mistake bright far flooring for
   directly walkable free space. This is only a visualization of Depth, not a
   semantic collision certificate.

The VLM still receives the current subgoal, instruction, absolute yaw/pitch,
up to 12 prior in-task VLM observation summaries, and up to 2 earlier target
hypotheses. **Raw prior images and full API conversation are not replayed**.

The VLM can request a calibrated TURN/LOOK action, or nominate the current
camera heading as a short-range motion *hypothesis* with
`candidate_approach=true`. The controller invokes ONLY
`Geometry.central_forward_obstruction()` for this candidate against the
same current public Depth (not `assess_step()` or `inspect_geometry`).
It performs **no forward action**.

- If the centerline probe finds confirmed measured occupancy: persist its
  observation ID, absolute bearing, depth hash, measured near-field evidence
  and BLOCKED status. Retry the VLM on the SAME RGB-D frame with an explicit
  `view_correction`; the VLM can choose a new camera bearing/pitch and inspect
  the resulting REAL RGB-D frame **inside the same observation task**.
  Re-proposing the same blocked bearing on unchanged Depth terminates
  `NO_RELIABLE_CANDIDATE` without another camera action.
- If no confirmed center obstruction exists, hand off the VLM's bearing as
  `APPROACH_CANDIDATE_FOUND` with `probe.status=UNCONFIRMED`. This
  deliberately does **not** mean path clearance has been certified.
- If there is no useful different viewing direction after a blocker, the
  VLM may finish `NO_RELIABLE_CANDIDATE`, with rejected bearings preserved.
  Re-asking exactly the same observation objective at the same place and
  starting RGB is deduplicated; no forced forward/steering, extra Agent or
  hard-coded staircase rule is introduced.

The result keeps `best_approach`, `approach_candidates`,
`blocked_approaches`, existing `best_candidate`, `best_cue` and all VLM
`visual_reports`. Direction records carry **absolute yaw and observation
ID** and remain available to NavigationAgent through compact context after
subsequent tool calls. NavigationAgent alone selects MOVEMENT direction,
distance, subgoal changes and stopping.

The existing conservative pre-MOVE Depth check remains as a fresh last-step
safeguard against changed live obstacles. Full `inspect_geometry` is still
available only on deliberate NavigationAgent request, but not needed for
ordinary VLM candidate-search. Do not repeatedly use full stair-terrain
`BLOCKED` results as evidence that all navigation directions are impossible.

Run the new regression:
```bash
python -m unittest discover -s tests -p 'test_rgbd_vlm_approach_search.py' -v
```
