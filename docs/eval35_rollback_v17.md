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