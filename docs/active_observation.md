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


## Batch 4-17: complete session and VLM-owned verification (Nov 2026)

The NavigationAgent calls one of SEARCH, APPROACH, OBSTACLE_REVIEW or VERIFY
through the existing `observe_for_goal` tool (`mode` is optional and defaults
to APPROACH). All four modes share camera action control and the complete
observation-session message history. On each VLM API call the entire previous
session (real RGB and metric Depth images, prior assistant replies, and
geometry feedback included in the previous request text) is transmitted
alongside the new current RGB-D. The provider is stateless: messages are
explicitly rebuilt, not presumed to live on the server. The session is
task-local, not persisted as raw base64 in JSON diagnostics. Actual checkpoint
records and provenance remain in Memory.

At every checkpoint, current public Depth measurements are supplied to VLM:
the coherent near-field center occupancy plus relevant swept-body Depth
diagnostics. These are measured facts with unknown regions and sensor
limitations, not an independent Python routing decision or clearance
certificate. Only the reliable fresh central obstacle preflight remains in
MOVE to prevent a planned atomic forward into a measured obstacle.
After blocking, NavigationAgent delegates OBSTACLE_REVIEW and does not
re-run a separate `inspect_geometry` tool. The registry retains older
internal/archival tools; they are NOT advertised to NavigationAgent.

The receipt-derived `route_progress` exposes grouped registered forward
segments (heading, source and destination observation IDs, estimated
height changes) and independently recorded obstacles at particular stances.
A successful downward segment is never erased by a later BLOCKED direction;
historical progress also never grants permanent clearance to move. Both
NavigationAgent and Observation Agent see this summary when choosing routes
or interpreting apparent up/down staircase orientation.

When NavigationAgent suspects a semantic milestone has completed it requests
mode VERIFY with a concrete question: for example, "Confirm that the ROBOT
is at the flat bottom landing, not merely seeing flooring beyond steps."
VLM returns CONFIRMED, NOT_YET or UNCERTAIN with cited real RGB-D sources.
Only a nonempty, valid and current-subgoal-confirmed observation may append
to `completed_subgoals` and advance to the next ordered milestone.
NOT_YET and UNCERTAIN do not switch the task. The old separate
`verify_subgoal` and `select_subgoal` calls are unadvertised, and the
final STOP success gate checks only that every milestone has been confirmed
through Observation Agent; no second stair-specific Python verifier is run.

VLM-led approach results are de-duplicated by same RGB/Depth stance,
observation mode, semantic goal and target types, even if NavigationAgent
merely rewords its objective. Genuine viewpoint, Depth, goal or question-mode
changes remain eligible for a new observation. `force_recheck` is an
explicit override, not an automatic retry.

Run focused regressions:

```bash
python -m unittest discover -s tests -p 'test_observation_session_verification.py' -v
python -m unittest discover -s tests -p 'test_rgbd_vlm_approach_search.py' -v
python -m unittest discover -s tests -p 'test_navigation_tool_menu.py' -v
python -m unittest discover -s tests -p 'test_pre_forward_obstacle_perception.py' -v
```

A real Habitat batch (same episode with Batch 4-17 starting pose) is still
needed to validate success rate, forward proportion, and token/latency cost.


## Agent I/O cleanup after PR #65 (2026-11-06)

The Observation Agent now distinguishes **current viewpoint evidence** from
**historical motion evidence** and reports both with real observation IDs.

- SEARCH, APPROACH and OBSTACLE_REVIEW return multiple local approach
  hypotheses with VLM-authored \`expected_progress\`
  (\`ADVANCE\`/\`REVERSE\`/\`DETOUR\`/\`UNKNOWN\`). The label is not a movement
  permit or Python semantic gate. NavigationAgent still chooses actual motion.
- A non-blocked candidate is saved, but **does not automatically finish** the
  observation. The VLM may choose another real turn/look to compare candidates,
  or choose \`finish\` with the accumulated list. Candidate refinement remains
  bounded by the existing episode and observation budgets.
- In VERIFY, tentative CONFIRMED is not task completion if a camera action
  is still requested, or if a task is interrupted. Only the final VLM
  \`finish\` produces a commit-eligible verification receipt. The receipt
  contains \`evidence_observation_ids\` when model-provided; invalid IDs are
  rejected. \`missing_evidence\` is conveyed to the NavigationAgent.
- A new observation task can preload at most three genuine prior RGB-D
  motion views (last successful segment's start/end and last local obstacle).
  Each is labeled with a frame ID, original heading, pitch and historical scope.
  Missing/evicted images are skipped, never synthesized. Task-local history
  still replays ALL earlier messages and images on every VLM call.
- When the VLM needs to correct a decision on the identical observation ID,
  the previously transmitted RGB-D stays in the replayed session and is not
  encoded or attached twice. The request explicitly references the earlier
  frame. The available turn angles are provided with every request and
  reflected in its schema, preventing pointless impossible 90-degree turns
  when only 30 degrees of sweep remain.
- Route evidence includes estimated motion uncertainty. An obstacle report
  includes the stance's place, local x/z, motion epoch and yaw, never an
  episode-global blocked direction.
- Repeat-observation caching requires the actual objective text, target
  labels, VLM mode, physical stance, RGB and Depth to match. A second question
  about "stairs" cannot accidentally reuse the answer to a different
  question just because both share the same target type.
- Complete angular observation is a **one-shot NavigationAgent advisory**.
  It never removes VERIFY or OBSERVE from the tool menu and never forces
  an unmeasured forward command. A fresh task/stance may issue the advisory
  again.

TaskPlanner now requests a natural-language \`completion_description\`
for each milestone and passes it to the VLM. Old deterministic
\`success_criteria\` remain only for archived legacy paths and diagnostics,
not for VLM semantic completion.

Regression tests live in \`tests/test_agent_io_contracts.py\`, alongside the
existing observation, navigation and task-planning contract suites.

The model-visible transcript replays prior messages rather than repeating route memory
and visual summaries in every newly appended turn. The first turn holds the
relevant motion context; later turns retain compact covered-yaw data and fresh
geometry measurements. All historical images remain available in the transcript.
