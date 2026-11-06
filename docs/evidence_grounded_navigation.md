# Evidence-grounded multimodal navigation (2026-11)

## Single source of truth

NavigationAgent receives the actual **current RGB** using the same
\`perception.image_url\` as the Observation Agent. The image, accepted
\`observation_id\`, RGB hash, Depth hash and motion-controller pose must match.
A mismatch is a structured error instead of pretending that a stale image
belongs to the current heading. The API request attaches one 640-pixel-long-edge
RGB per decision; it does **not** attach Depth to NavigationAgent. VLM
Observation Agent still receives both genuine RGB and calibrated Depth.

**Deployment prerequisite:** \`NAVIGATION_AGENT_MODEL\` and all role fallback
endpoints must accept image_url input in Chat Completions AND return valid
JSON tools. If a configured model is text-only, replace it with an available
vision-capable model. Do not catch a multimodal API error and silently
re-run the decision without the current image. This change intentionally
adds image input cost to NavigationAgent.

## Candidate -> execution -> measurement

- ObservationAgent captures real proposals; each saved route hypothesis has
  a stable \`candidate_id\` (the existing \`visual_target_XXXX\` key).
- \`navigate_relative\` and \`navigate_to_pixel\` accept an **optional**
  \`candidate_id\`. It is evidence-correlation metadata, not movement
  authorization. Unrecognized IDs are logged but never veto movement.
- Every actual forward step associated with an explicit candidate produces
  a source-backed \`route_prediction_feedback\` entry: candidate's original
  frame, prediction, executed heading, before/after frame IDs, RGB-D estimated
  height change, uncertainty, measured direction and whether an ADVANCE
  prediction agreed with the task's expected vertical direction.
- A \`DETOUR\` can legitimately move sideways or briefly against the
  overall task direction, so it is never marked as CONTRADICTED simply
  because a single vertical step differs. A failed/UNKNOWN registration is
  INCONCLUSIVE, not a false proof of collision.
- The actual \`blocked_at\` result uses the **post-turn** frame, yaw, pitch,
  local x/z, place, motion epoch and Depth hash at the attempted forward.
  Prior collision evidence has no global veto authority.

## Calibrated surface evidence and observation strategy

During VERIFY the VLM may optionally choose a normalized xyxy patch
\`candidate_surface_box\` for a visible relevant platform/floor. A helper
uses the existing \`Geometry.height\` projection and reports a dominant
planar height, patch reliability, residual, camera geometry frame and
UNKNOWN reachability. A second VLM response evaluates the **same RGB-D**
together with that measurement; no simulator ground truth, independent
stair-rule validator or automatic completion gate is introduced.

The VLM also receives a two-dimensional \`view_coverage\` summary of
actual checkpoints (yaw AND pitch), so a prolonged -60-degree close-up
sweep can be recognized and deliberately replaced with a legal look-up
or an explicit uncertain finish. No mandatory camera action is imposed.

Each finished VERIFY records the relevant motion evidence revision. Exact
repeats are cached; a subsequent NOT_YET can be reused when there is
neither new registered motion, new planar evidence, nor a genuinely novel
view. Changed target evidence and new motion still permit verification.
Old results carry a \`subgoal_generation\` and are not current after reopening.

## Completion review and auditable reopening

Normal \`observe_for_goal(mode=VERIFY)\` checks ONLY the active milestone.
It never silently switches back to another milestone. When newer evidence
credibly conflicts with an **already confirmed** milestone, NavigationAgent
may explicitly call \`review_completed_subgoal\` with the prior \`sg_N\`,
a concrete reason, and actual observation IDs.

The Observation Agent receives the original confirmation's real RGB-D,
the newer RGB-D, registered motion history and any relevant local surface
measurements. It returns \`UPHOLD\`, \`REOPEN\` or \`UNCERTAIN\` with source
citations. Only an explicit REOPEN triggers a state transaction in
MemoryRecords:

- Original confirmation and dependent completion entries become
  \`SUPERSEDED\`, never deleted.
- Earlier physical steps and RGB-D receipts remain immutable. A new
  generation's current-task motion starts at the reopening action count;
  earlier success is still available in full route_history, but cannot
  masquerade as movement newly executed after reopening.
- The previous milestone becomes the active subgoal; a new generation
  invalidates old cached VERIFY reports and same-task route candidates.
- Runtime cancels stale local waypoints and invokes the existing
  \`set_semantic_goal\` transition.
- A repeated review of identical original/current evidence is rejected
  without further API cost.

Seeing an old landmark in a new camera direction **alone** does not
prove that a previous confirmation was false. Only the VLM makes the
semantic choice, informed by real RGB-D and measured movement. The
runtime enforces task identity and evidence provenance, not stair-specific
behavior.

## Regressions and closed-loop checks

\`\`\`bash
python -m compileall -q havln_agent
python -m unittest discover -s tests \
  -p 'test_evidence_grounded_navigation.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v
\`\`\`

In a fresh diagnostic batch inspect: navigation image IDs/hashes;
route prediction disagreement vs actual movement; blocker yaw after
turn; surface local height and uncertainty; distinct yaw/pitch views;
VERIFY re-use; review UPHOLD/REOPEN/UNCERTAIN; superseded receipts;
and actual downstream task success. Physical Habitat success **cannot**
be inferred from unit tests or local RGB-D height estimation alone.
