# Eval-batch4-35 rollback to v17 navigation policy (2026-11-10)

## Evidence
- eval-32 ran v17 / PR #86 (merge 6184d6995ef43e130c52e1999339af564d7926fe).
- eval-35 ran v21: 200 actions, 182 turns, 4 looks, 13 forwards; first forward action index 108; 56 navigation model decisions and one VLM observation; zero API errors.
- The model repeatedly searched headings without reaching the intended arc, then interpreted ascending-looking stairs as its instruction to descend; no valid vertical metric localization is available.

## Rollback policy
- Restore the v17 NavigationAgent prompt, navigation state projection, historical intent and depth context, and motion policy. Keep Replay Studio and competition score visualization from PR #85/#87.
- Keep PR #88 non-semantic fault tolerance: bounded model JSON regeneration, typed feedback, no synthetic movement for invalid replies.
- Keep PR #89 object-vs-passage separation: a human or obstruction box cannot become a numeric passage waypoint; only a separate source-grounded passage_box can do so.
- Exclude v19 model-facing heading table/scan heuristics; exclude v20 omission of UNKNOWN front Depth; exclude v21 compact context projection.
- No additional forced movement rules, verifier, metric odometry or auto-progress assertion.

## Validation caveat
This is the v17 decision POLICY with narrow robustness fixes, not byte-identical v17. It has not yet been validated on the Habitat simulator. Replay the same val_seen episode. Compare first-forward index, longest turn-only streak, motion and RGB-D direction, whether stair descent truly happens, and externally measured score. Agent frame changes alone are not verified displacement.

Suggested smoke tests: python -m compileall -q havln_agent ; python -m unittest discover -s tests -p 'test_mission_navigation_architecture.py' -v

## Follow-up: retain PR #90 and #91 (2026-11-10)

The subsequent user request explicitly retains both changes on top of the previously merged policy rollback.

- PR #90: the NavigationAgent chooses LEFT/RIGHT/NONE, nonnegative turn_degrees and look=up/down/none; the executor owns signed motor conversion. Keep logging fresh pre-decision Depth, but send front_depth_preflight only if the actual measured state is CLEAR or BLOCKED. Absent field is inconclusive, not guaranteed clearance or a veto. Atomic forward still rechecks live calibrated Depth.
- PR #91: project existing mission_evidence_v1 into a bounded, source-linked NavigationAgent request without mutating canonical Memory, manufacturing translation or adding a second summary. Retain 360-degree yaw/pitch coverage, distinct real action TURN/LOOK counts since last executed forward, block/stall event distinction, last motion receipts, and historical frame identifiers. For this requirement, only v19's factual scan-count measurement is ported back to v17; no v19 heading arithmetic table, automatic scan rule, new verifier or forced movement.
- v17 mission strategy remains, with v18 malformed JSON recovery, v19 visual object/passage isolation, Replay Studio and score visualization retained. Tests for both PR #90 and #91 are reintroduced in the CI workflow.
- This is not byte-identical eval-32; same-episode Habitat validation is required.
