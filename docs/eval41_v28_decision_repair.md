# v28: precise eval-41 decision repair (same RGB-D rejected forward + stair ambiguity)

## Why

The real eval-41/v27 trajectory has 21 forwards, 70 turns, 57 NavigationAgent calls and zero ObservationAgent tasks. Navigation calls #34–40 around `obs_0080/0081` include six physically refused forward requests on a central Depth BLOCKED step; four at one -60° camera pitch, then two more at -30°. The raw evidence ledger already retains exact blocked-frame source RGB/Depth hashes, yaw, pitch, stance and repeated rejection counts, but model compact context distributes the counter, Depth result and tool receipts (the exceptional receipt list is empty after deduplication).

The robot also performed four forward steps toward a visually rising stair flight around `obs_0075`, while describing it as the downward route. Both issues are reasoning/input prioritization problems, not evidence for changing physical Depth thresholds or relative turn math.

## Code scope (two runtime files)

- `havln_agent/contexts.py` adds `_same_current_forward_rejection`, a strict **read-only match** for actual `BLOCKED_BEFORE_FORWARD` evidence against the current conclusive preflight: RGB hash, Depth hash, source stance, horizontal yaw and vertical pitch must match. Never treat `OBSERVED_FRONT_DEPTH_BLOCK_NO_MOVE_ATTEMPT` as a rejected move; never reuse other-stance, other-pitch or other-frame blocks. The existing `front_depth_preflight` is enriched with `previous_identical_view_forward_rejections`, `previous_rejected_forward_actions_executed=0`, the source observation ID and an explicit source-equivalence flag. The same matched history block is excluded from the **compact** historical block list to avoid repeated evidence; canonical Memory and the actual Depth measurement remain unchanged. NO second state table, veto, threshold or controller action.
- `havln_agent/navigation_agent.py` changes only focused model guidance: treat a repeated same-view physical rejection as zero actions executed; choosing a different requested distance without a changed view/route does not resolve it. If current stair treads appear to rise versus descend or contradict the source-backed route frames, request `observe_for_goal` with specific historical start/end observation IDs as a **selective option**; a visible staircase alone never proves descent. The existing short-forward bias for a single plausible CLEAR/UNKNOWN passage and every-atomic Depth preflight remains intact. Redundant global-yaw Prompt guidance was condensed; entire prompt < 5,000 characters.

## Regression

`tests/test_v28_grounded_decisions.py` reproduces the eval-41 source identities and different-pitch attempt counts, and asserts: nonmutation; same RGB-D rejection count (4 for `obs_0080` and 2 for `obs_0081`); no duplicate raw block in compact context; yaw/pitch/Depth/RGB mismatch and CLEAR/UNKNOWN never cause a match; observed-only Depth BLOCKED never counts as an executed attempted-forward rejection; output passes the v27 current-camera-relative angle projection. Tests also check the continuing selective ObservationAgent and short forward guidance.

```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_v28_grounded_decisions.py' -v
python -m unittest discover -s tests -p 'test_v27_model_relative_angles.py' -v
python -m unittest discover -s tests -p 'test_v26_eval39_route_quality.py' -v
python -m unittest discover -s tests -p 'test_v21_compact_navigation_context.py' -v
```

## Evaluation, not a hard controller constraint

Re-run the same first `val_seen` episode (100 actions and same model/API settings) and compare with eval-40/v26 and eval-41/v27. Measure changed-RGB-D forward steps, turn/LOOK counts, navigation/observation calls, duplicate denied same-view forward requests, whether a rising stair is misinterpreted as descent, and actual foyer progress. An advisory goal is zero repeated identical-view denied moves; the Python executor is NEVER set to disallow a model-chosen retry. VLM may still choose not to run; visual semantic accuracy and actual Habitat success require live evaluation.
