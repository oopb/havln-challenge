# Evidence-backed current perception (interim rollback)

This branch removes the two over-complicated additions introduced in PR #69:

- No independent route `candidate_id` → predicted-versus-actual motion ledger.
- No `review_completed_subgoal` / `REOPEN` transition, dependent milestone
  invalidation, or generation-specific cache state.

The present program **still uses legacy numbered task milestones**. This
rollback does not yet implement whole-instruction natural-language
navigation: that redesign needs separate, explicit implementation.

## Preserved capabilities

- NavigationAgent receives current authentic RGB through the existing
  image_url transport, with pose, RGB hash and observation ID integrity checks.
  The configured NavigationAgent model **must support image input**.
- Observation Agent receives current RGB-D and full same-session observation
  dialogue, and returns structured visual suggestions. These suggestions
  are hypotheses, not collision clearance or navigation commands.
- Geometry returns a measured `blocked_at` record at the actual *post-turn*
  heading and pose, not the request's prior bearing.
- VLM can point to a visible surface ROI during VERIFY; local Geometry can
  measure its relative plane height and quality and return it to the same
  VLM for interpretation. A visible floor is not an automatic arrival verdict.
- Observation Agent receives yaw/pitch coverage and may decide to look up
  rather than continue rotating while looking down.
- Repeated VERIFY with no novel sensor/motion evidence can reuse its previous
  result. The ordinary visual `evidence` field accepts up to 1200 characters.
- All sensor records and VLM reports remain inside episode Memory. Blocked
  positions, surface measurements and verification summaries are memory
  evidence, **not independent executive agents or validators**.

## Startup acceptance

Run in the same container as the evaluation worker:

```bash
python -m compileall -q /toolkit/havln_agent
python -m unittest discover -s /toolkit/tests \
  -p 'test_evidence_grounded_navigation.py' -v
python -m unittest discover -s /toolkit/tests \
  -p 'test_navigation_tool_menu.py' -v
```

On the first real run verify that `registry(camera, ...)` completes without
TypeError, and that the NavigationAgent endpoint accepts actual image_url
content. No simulator success claim is implied by schema/contract tests.
