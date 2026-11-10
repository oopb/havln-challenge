# v21 — reuse existing state, trim redundant NavigationAgent evidence

The already-implemented `mission_evidence_v1` projection contains complete
same-stance yaw/pitch coverage, repeated observations, actual TURN/LOOK
counts and last successful-forward receipts. v21 does **not** add any new
persisted navigation state, novel task phases, forced forward, arbitrary
navigation lock, recovery policy or a second validator.

Observed eval-batch4-34 API trace (194 NavigationAgent messages):
- context characters: first 6,444; median 24,118.5; last 37,451
- `memory` characters: first ~3,358; median ~19,490; last ~33,231
- the `recent_forward_failures` list exactly repeats the separately
  supplied `blocked_at_source_stances` and
  `stalled_forward_at_source_stances`. Multiple other fields repeat
  the same Depth ROI and motion receipts. This made prior input growth
  expensive and diluted the useful causal signal.

The new `compact_navigation_context` keeps the canonical `memory`
contract and essential existing field names, but copies only the
*NavigationAgent* model request and projects a bounded set of:
- complete, exact same-stance yaw/pitch heading bins, repeat count,
  actual TURN and LOOK since last executed forward, 2 recent view IDs;
- 3 recent changed-RGB-D forward segments, model-selected last motion
  status and actual motion receipts (never claim measured translation);
- 2 distinct executed Depth blocks, 2 actual unchanged-RGB-D stalls,
  2 passive front-block observations, and at most 2 exceptional tool
  receipts; current live BLOCKED ROI takes precedence over duplicate
  passive blocks;
- 2 recent concise VLM observation sessions, 3 provenance-rich route
  candidate hypotheses and 3 lightweight historical frame descriptors;
- full original mission instruction and source-anchored progress.

It removes redundant `recent_forward_failures`, repeated standard
TURN tools, long `purpose` strings in old tool calls, duplicate VLM
checkpoint prose, irrelevant source hashes in historical obstacle
summaries, and verbose constant flags. The full Memory, VLM raw session,
original tool receipts, replay logs, and RGB-D images are unchanged and
remain individually retrievable.

This is only an input projection change: no new format requirements for
model output; no changes to tilted forward, Depth safety, VLM passage
segmentation, user-selected routes or agent control. The prompt now
explicitly distinguishes observed camera rotation from executed forward
progress without requiring a fixed threshold or full-circle scan.

Regression:
```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_v21_compact_navigation_context.py' -v
python -m unittest discover -s tests -v
```

---

# v20 — direction-word decisions and confirmed Depth evidence only

v20 only implements the two approved changes from the eval-batch4-34
review; **no change** is made to the proposed historical/stagnation
memory redesign (item 2), the observation strategy, or physical motion.

1. The navigator chooses the **direction word** `LEFT`/`RIGHT`/`NONE`
   plus a nonnegative `turn_degrees`; the separate `look` value is
   `up`/`down`/`none`. Existing schema, model autonomy, and
   the `tool+args` executor contract remain stable. The existing
   `signed_turn()` helper converts direction and magnitude to an
   internal motor angle. No turn-heading arithmetic reference table
   is sent to the LLM; prompt and tool descriptions no longer ask it
   to reason about motor angle signs. `turn_and_observe` still means
   turn without translating and `navigate_relative` still means turn
   and execute a *positive-distance* finite segment.
2. The navigation model receives `front_depth_preflight` only when
   the current-source state is `CLEAR` or `BLOCKED`. When the state
   is `UNKNOWN` or any other inconclusive value the **entire field is
   omitted**; do not insert a substitute string, a fake CLEAR, or
   extra uncertain-path warnings. The live Depth preflight is still
   computed for trace/debug data, and the atomic forward executor
   still runs an independent *fresh* calibrated Depth check and
   blocks confirmed occupancy. ObservationAgent continues to receive
   the actual RGB-D and existing independent Depth information.
3. Current same-stance coverage, route candidate evidence, historical
   image selection, Memory architecture, and liveness intervention
   remain as in v19. A proposed solution for the repeated-turn
   issue is documented for user review separately, **not implemented**.

Run:
```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_v20_direction_depth.py' -v
python -m unittest discover -s tests -p 'test_v19_passage_heading.py' -v
python -m unittest discover -s tests -v
```

Compare a true v20 same-episode 500-action run against eval34; monitor
`turn_and_observe` count, actual forwards, unknown-status omissions from
NavigationAgent API input, confirmed blocks, and percent useful observed views.

---

# v19 — source-grounded passage regions, exact turn references and scan receipts

eval-batch4-33 had 367 turns and 109 atomic forwards, including 119
actions without forward at one executed-forward stance. In three VLM
answers at obs_0222/0224/0225, `box` marked the *human blocking the
centerline*, while the prose suggested passing to the human's LEFT.
The old harness computed ~72-degree waypoints from the human boxes.

## Behavior changes

- The VLM's `box` remains the semantic object region. A separate,
  optional `passage_box` is exclusively the visible potential floor /
  doorway / stair passage region. Only `passage_box` can ground an
  angle through calibrated image-HFOV projection. No passage region
  => natural-language route hypothesis only, **no numeric bearing**.
  A passage box still does not certify physical clearance. Existing
  boxed target models remain valid; their legacy object boxes no
  longer turn into route headings.
- ObservationSession keeps source `semantic_target_region` and
  `passage_region` distinct; memory stores both and labels the numeric
  source as `visually_proposed_passage_not_obstacle`. Exact observation
  ID/stance is preserved.
- Following an actual forward stance change, the *compact*
  navigation memory no longer exposes historical absolute candidate
  headings as if they were current targets. Source records remain
  retrievable. The model must re-ground old sightings in CURRENT RGB-D.
- NavigationAgent sees a deterministic, model-only table of executed
  LEFT/RIGHT turn results from the current yaw (all 15-degree increments
  on the competition's calibrated camera), including explicit 150→90
  RIGHT 60 not LEFT 60. There is NO direction_intent wrapper and the
  runtime never rewrites or vetoes a syntactically valid model turn.
- The same-stance coverage summary now adds actual atomic TURN/LOOK
  counts since the last forward and a measured complete horizontal
  360-degree sweep flag. It **does not** automatically choose a direction
  or impose a scan cap. The prompt asks the model to reconsider stale
  candidates/actual forward receipts before revisiting identical views.
- Selected historical images no longer prioritize an obstacle-only
  `candidate_approach` when no passage region was grounded.

Retains v17 camera pitch during forward, per-atomic live Depth,
v18 format recovery, and the existing tool+args movement authority.

## Regression

```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_v19_passage_heading.py' -v
python -m unittest discover -s tests -v
```

A real eval-batch4-34 run should compare forward progress, same-stance
turn loops, VLM person/passages, heading arithmetic and measured blocks.
No official success is inferred from action count or the presence of a
semantic target in an image.

---

# v18 — bounded output recovery, continuous cruise, and local visual passages

The eval-batch4-31 (2026-11-09T140359Z) run made 20 forwards but stopped
technically after the navigation model emitted
`navigate_relative(turn=RIGHT 30°, distance_m=0)`. Other replies exceeded
the 220-character per-evidence limit.

## Changes

1. `observed_evidence` length is now 600 chars per entry (still 1–5
   source-grounded entries). The same evidence → short analysis → final
   `tool, args` ordering is retained.
2. `turn_and_observe` is explicitly the pure turn/LOOK tool. Distance zero
   remains **invalid** for `navigate_relative` and is never silently
   rounded to 0.25m. Argument validation produces field-specific feedback
   before the generic tagged-union schema error.
3. Each decision gets up to three *fresh* role regenerations for formatting
   errors, with concrete field feedback. If still invalid, the scheduler
   permits up to two further decision cycles with newly refreshed sensor
   evidence. Invalid model output never reaches the motor executor; after
   the bounded limit the episode can still fail with an explicit api_error.
   Transport, authorization and budget errors are not reinterpreted as
   regenerable format failures.
4. NavigationAgent can choose 0.25–3m finite segments, and is reminded that
   it may choose 0.5–1m when a visible corridor is stable. This is **advice,
   not a hard-coded minimum**. The existing atomic forward depth and RGB-D
   stall checks remain active.
5. The VLM receives the specific visual question from NavigationAgent and
   is told to compare candidate **floor/entry corridors** around people,
   rails, stairs and doorways instead of targeting the centroid of a broad
   semantic-object box. Hypotheses never authorize motion.
6. The `last_motion_decision` context is projected from canonical receipts
   into a compact, source-bound summary rather than copied with unused
   command arguments and duplicated raw action events.

All v17 behavior remains: actual camera pitch is preserved on each forward;
no automatic LOOK_UP; central Depth blockage is still checked. No forced
forward, automatic observer invocation, direction_intent or validator agent.

## Regression

```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_v18_navigation_recovery.py' -v
python -m unittest discover -s tests -v
```

A comparable `val_seen` run should use the same episode, model and 128-step
budget as eval-batch4-31. Compare abort type, observed evidence length errors,
valid pure turns, measured blocks, actual forwards and locally useful VLM calls.

---

# v16 — complete observation evidence and model-owned route continuity

Addresses the eval-batch4-29 (2026-11-08T151735Z) failure mode of
88 turns / 22 forwards and repeated visually identical observations at
the same physical-action stance. It retains the v15 `tool + args`
protocol and the v12 movement executor.

## Runtime changes

1. `recent_views_at_stance` includes **every** observation at the
   current motion epoch and executed-forward stance, even across >80
   frames and separate observation sessions. It serializes complete
   quantized yaw/pitch bins compactly, with a small sample of detailed
   source-labelled views and exact RGB-D repeat counts.
2. An ObservationController session checks prior **same-question,
   same-RGBD, same-action-stance, same-yaw/pitch** VLM checkpoints.
   Matching visual evidence is reused without a paid API call, but
   *camera turns are never replayed*; changed frames, stance or
   questions always require a fresh VLM decision.
3. All candidate-approach evidence, including cues with no target box,
   is recorded in canonical Memory. Unboxed cues have NO exact heading.
   Grounded box-centre headings are only reprojected at the original
   action stance; moving forward invalidates their relative bearing.
4. The latest model-selected motion **receipt** is saved for next-turn
   route continuity. It records executed forwards, actual RGB-D change
   and interruption status. It never triggers automatic continuation.
5. The existing VLM-guided RGB-D request embeds a focused near-field
   ROI question when the live Depth preflight confirms an obstacle.
   No second VLM call, separate obstacle validator, semantic stair rule,
   automatic turn, or Depth safety bypass is introduced.
6. NavigationAgent prompt uses complete same-stance coverage, route
   candidates and last motion receipts. The model retains authority
   over another scan, route continuation, STOP and alternatives.

`PROMPT_VERSION=havln-agent-v16-evidence-coverage-route-continuity`.

## Verification

```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_v16_observation_continuity.py' -v
python -m unittest discover -s tests -p 'test_v11_observation_review.py' -v
python -m unittest discover -s tests -p 'test_mission_history_and_headings.py' -v
python -m unittest discover -s tests -p 'test_v13_navigation_context.py' -v
python -m unittest discover -s tests -p 'test_forward_preflight_v9.py' -v
```

Compare a fresh val_seen single-episode 128-step run with eval-batch4-29.
Evaluate invalid API responses, actual action receipts, longest no-forward
run, repeated identical RGB-D observations, obstacle ROI interpretation,
and route continuation at the staircase, not just the final forward ratio.
No claimed official success until real simulator measurements confirm it.

---

# v15 — remove direction_intent and restore model-owned navigation turns

In eval-batch4-29, the navigator made 0 forward actions and stopped after
malformed responses repeatedly nested `direction_intent` inside `args`.
That wrapper was a v13 patch for a contradictory visual description/turn,
not evidence that a target was correctly localized.

- Model output is again a single `{"tool": "...", "args": {...}}` object,
  with optional progress and uncertainty fields. There is no structured
  direction-intent mode, target bounding-box requirement, or Python semantic
  direction check. Tool arguments still use legal LEFT/RIGHT/NONE and
  nonnegative magnitudes; the executor checks physical action legality.
- Current RGB, at most 3 source-verified historical RGB images, v12
  bounded mission evidence, pre-decision Depth, and source-linked stale
  navigation-intent handling remain as in v14.
- NavigationAgent chooses the route, observation and turn autonomously.
  A mismatch between a visual phrase and a turn is a model decision issue
  to inspect in trajectory diagnostics, not a reason to require a second
  schema envelope that can abort an otherwise valid navigation command.
- Scheduler no longer forwards the removed field in tool bindings, and
  internal episode-history projections no longer emit it. Stored older
  episode files can still contain their historical v13 decisions.
- No motion executor, Depth policy, forced direction rules or metric
  odometry changes. This change addresses schema compliance; it does not
  by itself guarantee useful forward motion or external API reliability.

`PROMPT_VERSION=havln-agent-v15-simple-navigation-tool-protocol`.

Run `python -m compileall -q havln_agent` and
`python -m unittest discover -s tests -v` locally; compare next
`eval-batch4-30` with the same episode and 128-step budget.

---

# v14 — rollback NavigationAgent input to v12 + selected historical RGB

This revision **reverts only the model-facing NavigationAgent INPUT**, not
its navigation output schema or robot motion/Depth checks. Real
`eval-batch4-28` had overlarge text history and 10 image attachments per
decision, coinciding with JSON/format failures and early API termination.

## Per-navigation-decision payload

- Restore exactly the v12-style bounded `navigation_context` projection:
  `memory=mission_evidence_v1` (recent observation tasks, task progress,
  local stance view coverage, recent tool receipts, blocked bearings, forward
  action segments and a few historical frame descriptors), `actuator`,
  `camera_controls`, `actions_remaining`, `navigation_mode`, and fresh
  `front_depth_preflight`.
- **Do not include** `episode_history.timeline`, a full RGB-D catalog,
  historical per-action event records or repeated verbose log fragments in
  the routine navigation prompt. The complete canonical records remain
  persisted in Memory and accessible through `query_task_memory` and
  `read_memory_record` if the model explicitly requests them.
- Attach **one CURRENT RGB** (as in v12) and at most **three genuine
  HISTORICAL RGB images** selected to cover a suspected revisit, episode
  origin, recent candidate route, or recent changed-forward scene.
  Historical Depth images and a separate current Depth image are **not
  sent**. The real current Depth preflight remains present as structured
  measured blocker evidence; pre-forward safety is unchanged.
- One small `history_media.manifest` maps every attached image to its
  source `observation_id`, original heading/pitch, action stance ID,
  selection reason, and image index. Media bytes are hash checked against
  their recorded RGB-D sources; missing historic images are skipped safely.
  Source-relative historical left/right text is not current navigation
  permission or a live camera bearing.
- The v13 *output* contract (model-authored `direction_intent` for
  movement, guarded `ALIGN_VISIBLE_TARGET` image-box turn consistency
  versus explicit `EXPLORE_ALTERNATIVE`) **is kept unchanged**. This
  change reduces input load only and is NOT a claim that all JSON output
  compatibility errors have been repaired. If malformed outputs remain,
  debug output protocol and limited recovery separately.

`PROMPT_VERSION=havln-agent-v14-compact-v12-memory-with-history-rgb`.

## Check before live eval

```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_v13_navigation_context.py' -v
python -m unittest discover -s tests -p 'test_forward_preflight_v9.py' -v
python -m unittest discover -s tests -p 'test_mission_history_and_headings.py' -v
python -m unittest discover -s tests -p 'test_v12_action_stance.py' -v
```

Use a NEW `eval-batch4-29` directory for controlled comparison with the
v13 `eval-batch4-28` failure. Verify `image_count` <=4 for every
NavigationAgent call (unless transport adds unrelated images), **no
`episode_history` field** in navigation request context, correct
source-frame manifest, actual token usage, valid JSON rate and
forward/turn distribution. No online metric-odometry integration returns.

---

# v13 — complete navigation history, grounded turn intent and prompt consolidation

**Real failure case:** `eval-batch4-27` observation `obs_0095`.
NavigationAgent described inspecting the stone arch to the **left** of a
person but returned `turn_and_observe(RIGHT,30)`; Habitat correctly
executed two right turns. The previous decision at `obs_0091` had
recorded `navigation_intent="Follow the opening slightly right"` at a
different yaw (195 degrees), whereas `obs_0095` was at 180 degrees and
a different executed-forward stance. That camera-relative sentence was
still being copied verbatim into current navigation context.
This is a source-binding/decision-consistency flaw, **not** a demonstrated
left/right sign error in the atomic executor.

## 1. Full ordered context without duplicate prompt histories

Every NavigationAgent API request now receives
`context_contract=navigation_context_v2_complete_causal_history` with
`episode_history` containing **all** earlier:
- navigation decisions: tool name, exact args, source observation ID,
  source heading/stance, direction intent, progress and uncertainties
- results: status, executed actions, associated tool call ID, error or
  blocked evidence, terminal observation and visual guidance
- atomic actions: every TURN, LOOK, FORWARD and STOP with exact source
  observation IDs, current stance, heading, RGB-D change and live
  Depth preflight status
- VLM observation tasks: all session objectives, observation checkpoints,
  model hypotheses, visible target boxes, suggestions, source frames and
  camera actions
- RGB-D frame catalog: all recorded observation IDs, step, heading,
  stance generation, public frame hashes, possible static-structure
  revisit matches, explicitly **not** verified physical localization

The timeline is ordered by **model-turn causality**, not just atomic
action count, so zero-action turns and prior tool results appear before
the following decision. Full history is sourced from canonical Memory;
the old three-task/four-receipt/five-motion overlapping mini-summaries
are removed from the NavigationAgent projection. The VLM's separate
mission_evidence format remains backward-compatible.

The visual payload includes the **current RGB and calibrated Depth pair**
and up to **4 diverse historical RGB-D pairs**, with a manifest that maps
every image position to its source `observation_id`, original heading,
stance and selection reason. Historical frames may include initial
scene, visually similar revisit, route-candidate checkpoint and
recent changed-forward view. A hash mismatch or missing sparse archived
frame means skip the attachment, not fabricate an image or crash; all
historical records remain available as source IDs in the text catalog.
Historical Depth **never** grants current collision clearance. This
limits per-turn media cost while preserving complete textual chronology.

## 2. Historical relative intent validity

Each updated `navigation_intent` is stored with:
`source_observation_id, source_heading_deg, source_pitch_deg,
motion_epoch, viewpoint_generation, stance_id, source_action_step`.
After any TURN changing yaw or any MOVE changing the stance, the old
free-text intent remains as `progress.prior_navigation_intent` but
`progress.navigation_intent` becomes null until the model reissues an
intent at the current camera view. Free-text "left/right" must never
silently become a current bearing.

The pre-decision Depth scan may refresh the RGB-D sensor; the context
is now constructed **after** this fresh scan and scheduler binds tool
execution to the actual post-refresh observation ID used for inference,
preventing a stale-source decision/execute mismatch.

## 3. Model-owned grounded direction intent

`navigate_relative` and `turn_and_observe` require structured
`direction_intent` with `mode` and `reason`:
- `ALIGN_VISIBLE_TARGET`: a **current** observation ID and a normalized
  `target_box_xyxy_norm` are mandatory. The calibrated center ray (left
  positive, right negative) must agree in sign with the declared turn
  when outside the actuator half-step ambiguity zone. A mismatch causes
  `APIError(category=structured_field)` and the scheduler's existing
  one-time corrective model retry — never an automatic Python turn.
- `EXPLORE_ALTERNATIVE`: the model may intentionally turn in another
  direction to avoid an obstacle if its reason explains the choice.
- `FOLLOW_CURRENT_EVIDENCE`, `REORIENT_USING_HISTORY` and
  `CONTINUE_CURRENT_ROUTE` remain model-selected modes. Historical
  direction text is not a current camera-relative target box.

There is deliberately no keyword heuristic matching English "left" to
LEFT, no hard-coded stair rules, no forced forward action and no hidden
steering. A bounding box supplied by a model is a *claim*: geometric
consistency can be checked, but object presence/identity still requires
real RGB inspection. Depth and safety remain controlled by the existing
current-frame preflight.

## 4. Prompt and audit

The former 6.5k-character NavigationAgent prompt has been consolidated
to role, provenance, evidence and decision principles; detailed tool
argument syntax is defined once by `compact_navigation_protocol` and
validated by the registered JSON schema. The old redundant recent
history fragments are not duplicated in the new full timeline.
`PROMPT_VERSION=havln-agent-v13-causal-history-grounded-turns`.
Every completed model decision and source-bound tool receipt is also
persisted in `memory.json` under `navigation_turns`, enabling direct
review of the historical interpretation and execution.

## 5. Verification

```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_v13_navigation_context.py' -v
python -m unittest discover -s tests -p 'test_mission_navigation_architecture.py' -v
python -m unittest discover -s tests -p 'test_mission_history_and_headings.py' -v
python -m unittest discover -s tests -p 'test_forward_preflight_v9.py' -v
python -m unittest discover -s tests -p 'test_v12_action_stance.py' -v
```

Please run a new `eval-batch4-28` on the same val_seen episode and
128-step budget. Monitor API context characters / images / token usage
(the history is larger), visual directional intent contradictions,
navigation forward ratio, observation turns, and whether the arch-to-
window route reversal recurs. This code has not yet proven a navigation
score improvement; only a real simulator run can demonstrate that.

---

# v12 — action-receipt stance identity (no metric VO in navigation)

**Problem addressed:** In `eval-batch4-26`, 14/33 executed forward
actions lacked reliable RGB-D registration. The first failed registration
marked `position_coordinates_valid=false` for the remainder of the
episode. v11's same-stance view coverage then stayed empty for most
NavigationAgent requests, encouraging redundant observation and turns.
The navigation pipeline had kept integrating the 19 estimated moves,
even though the coordinate chain was no longer valid.

## New single source of truth

Every public `pose()` includes:

- `heading`: yaw updated only by executed TURN_LEFT/RIGHT receipts
- `pitch_deg`: camera tilt updated by executed LOOK_UP/DOWN receipts
- `motion_epoch`: episode-level local reference compatibility
- `viewpoint_generation` / `stance_id`: **increments on every actually
  executed MOVE_FORWARD**, regardless of whether RGB-D changed
- `forward_actions_attempted`, `forward_actions_with_frame_change`,
  `forward_stall_actions`: actual action/image observations
- `x=null, y=null, z=null, position_uncertainty=null,
  position_coordinates_valid=false, metric_position_valid=false`

No default 0.25m motion or static x/z placeholder may be interpreted as
the robot position. The original `estimate_translation` is retained
**only as an offline research diagnostic** in `havln_agent/motion.py`;
the online executor never calls it. All online `rgbd_motion` receipts
have `status=NOT_ESTIMATED`, `observed_frame_change`, source-linked
RGB/Depth hashes, and `translation_m=null`. There is no binary VO
success/failure gate in the model's execution loop.

## Physical action correctness (unchanged)

A navigation tool still uses model-selected turn direction and finite
nominal distance; distance is quantized into simulator action counts.
`LocalMotionController.route_action` uses the command yaw and remaining
forward receipts, not metric endpoint distances. A resumed ACTIVE command
is not canceled by a new generation. Any newly proposed saved waypoint
is rejected if its original generation differs from the current one.
`target` is the active waypoint **ID**. `original_target`,
`pending_target`, `endpoint_xz_m`, measured displacement and
vertical-motion estimates are null. `ACTIONS_COMPLETED` means the
intended commands executed, not that the metric target was reached.

Every MOVE still uses fresh current Depth preflight, real camera
recentering when needed, and calibrated legal Habitat actions. `BLOCKED`
preflight produces zero MOVE and zero stance-generation changes.
Byte-identical after-forward RGB-D triggers the existing
`REQUIRES_ATTENTION` feedback, not a fabricated collision. Even for
such a receipt, the stance generation advances conservatively because
we cannot prove the robot did not physically move.

## Entire memory and observation contract

Both `heading.same_local_stance` and
`local_exploration.same_stance` compare actual
`motion_epoch+viewpoint_generation` when coordinates are invalid.
TURN/LOOK remain the same stance; any executed MOVE isolates prior
view coverage. Old history is still available as historical semantic
evidence, but NOT as a current camera view or live path clearance.

- Place memory groups real observations by stance generation, not
  estimated x/y/z; `position` and `height_y_m_est` are null.
- ObservationAudit and EvidenceLedger dedup by generation plus real
  RGB/Depth hashes / heading, never by a zero-filled origin.
- Source-linked route progress is a record of executed forwards whose
  images changed, NOT a measured translation or successful descent.
- Legacy `TRAVERSED/motion_verified` states are not emitted from an
  unverified action. Route graph edges are marked
  `ACTION_SEQUENCE_EXECUTED` with explicit unverified displacement.
- A same-stance grounded point may still produce a new candidate
  **bearing** by rotating source-local RGB-D geometry using commanded yaw.
  An executed MOVE invalidates that bearing's source location, requiring
  fresh visual grounding.
- NavigationAgent and ObservationAgent prompts explicitly distinguish
  nominal forward commands, image change and measured displacement.

## Provenance, summaries and compatibility

`summary.json` reports `net_displacement_m_est=null`,
`translation_m_est=null`, `metric_odometry_enabled=false`,
`final_stance_generation`, and actual forward frame-change/stall counts.
`motion_registration_states` becomes `NOT_ESTIMATED` by design.
The complete tool/Memory/observation histories remain source-linked,
including Depth blocker ROI and recoverable camera pitch limits.
`PROMPT_VERSION` changes to
`havln-agent-v12-action-stance-no-metric-odometry` so old run outputs
cannot be incorrectly combined. This is a protocol refactor, NOT a
guarantee of improved route-finding or official navigation success.

## Checks

```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_v12_action_stance.py' -v
python -m unittest discover -s tests -p 'test_forward_preflight_v9.py' -v
python -m unittest discover -s tests -p 'test_v11_observation_review.py' -v
python -m unittest discover -s tests -p 'test_mission_history_and_headings.py' -v
python -m unittest discover -s tests -p 'test_recoverable_camera_pitch.py' -v
python -m unittest discover -s tests -p 'test_navigation_blocked_history_contract.py' -v
```

Start a NEW `eval-batch4-27` output directory. Do not overwrite v11
diagnostic artifacts. Compare at the same 128-step budget, model IDs,
initial episode and Depth policy. Key checks: now nonempty same-stance
history after many forward actions, fewer redundant observation turns,
unchanged robust Depth blocking, and unchanged action count semantics.

---

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


## v17 — retain tilted camera during motion and evidence-first navigation output

Atomic forward movements no longer trigger `LOOK_UP` to level the camera.
The model-selected pitch is retained across TURN and MOVE; `_step`
still refreshes live RGB-D and runs the same calibrated central Depth
obstacle preflight before *each* forward. `Geometry` projects each pixel's
ray using `observation.pitch_deg`; missing FOV evidence remains UNKNOWN,
never a claim of safe clearance. The model can explicitly LOOK when the
camera angle is uninformative. No new hard-coded stairs mode is introduced.

Navigation responses now have a three-part autoregressive structure:
1. `observed_evidence`: 1–5 short, grounded observations (required)
2. optional narrative progress notes and `decision_analysis`: short
   visible rationale based on current and historical evidence (required)
3. `tool`, `args`: the only executable choice, always written last

The underlying tool envelope remains `tool + args` and the scheduler
continues to project only these fields to the executor. A concise rationale
is an inspectable decision summary, not hidden internal chain-of-thought,
movement authorization, or a new direction-intent validator. The outgoing
wire JSON no longer alphabetizes schema keys, so model-facing property order
is preserved for JSON-schema-mode providers. No provider is assumed to
guarantee response key order; model instruction and schema order guide it.

Sample:

```json
{
  "observed_evidence": [
    "Current RGB shows a descending stair tread ahead.",
    "Live Depth reports UNKNOWN rather than a confirmed centerline block."
  ],
  "decision_analysis": "The stair direction is visible, but there is no certified clearance; try a short controlled move and reassess the next frame.",
  "tool": "navigate_relative",
  "args": {"turn_direction": "NONE", "turn_degrees": 0, "distance_m": 0.25}
}
```

Regression: `python -m unittest discover -s tests -p 'test_forward_preflight_v9.py' -v`
and `test_v13_navigation_context.py`, `test_mission_history_and_headings.py`,
`test_mission_navigation_architecture.py`.
