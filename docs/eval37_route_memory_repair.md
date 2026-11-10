# v23: eval-37 route evidence and observation novelty

## Root cause observed in eval-37

- The robot executed **seven forwards at 225°** (atomic steps 11–17), with genuine RGB-D changes along the stair route. Later it faced 45° and advanced back along a nearly opposite bearing.
- The latest free-text progress had changed from "descending the stairs / near the landing" to "still on the upper level, stairs not found". These are conflicting model interpretations, not verified sensor positions.
- A generic episode-origin image occupied one of just three historical RGB slots in most requests. At the opposite view, the important *before* stair image was often missing.
- Later the model spent many LOOK steps revisiting existing pitch/yaw views even though source-labelled coverage counts were available.

## Fix, without a new state machine

1. **Factual route continuity over canonical Memory:** `route_progress_summary(max_successful_segments=None)` reads all existing changed-RGBD forward receipts, not only the latest eight for read-only projections. `route_continuity()` picks one substantial source-linked historical segment (longest, with relevant heading tie-break) and latest segment, and computes only the angular separation from current camera yaw. Opposing headings suggest a *possible* retrace; they do NOT establish spatial coincidence, real descent/ascent, net travel or task completion.
2. **Confront older conflicting model claims:** The same compact field includes the original before/after RGB observation IDs, the earlier model's visual-description excerpt and progress claim, plus two early observations looking approximately back along that historical travel heading. All remain explicitly `interpretation_unverified=true`. The latest `progress.narrative` stays mutable but no longer hides key old assessments; no semantic verifier or sg0/sg1 label is created.
3. **Relevant historical RGB, not fixed origin:** `navigation_visual_anchors` prioritizes authentic before/after images of the salient previous forward segment when the current camera yaw is roughly opposite; then a recent changed-forward frame, current visual route evidence, and finally the episode origin as fallback. Historical images remain capped at **three**, hash-checked and labelled by original stance/yaw/pitch; they are never live Depth clearance.
4. **Concrete evidence for repeated LOOK:** `recent_look_evidence()` reports the last eight *executed* NavigationAgent LOOK decisions at the current action stance, their first-time vs revisited camera yaw/pitch bins, and up to two original LOOK questions linked to the following navigation model's actual visual description. Angle novelty is explicitly NOT semantic information gain. Rejected LOOKs do not count. This is read-only input, no scan lock or forced MOVE.
5. **Compact model context and prompts:** #91's projection trims the added historical claims and LOOK descriptions, omits empty new fields and deep-copies the original Memory. Prompts tell both agents to compare source IDs, not mistake an earlier model hypothesis for a verified stage, and prefer reasoning from answered questions rather than endless pitch alternation. All v22 heading calculations, #90 conclusive Depth exposure, atomic collision preflight, and replay/scoring components remain unchanged.

## Offline regression and deployment

```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_v23_route_evidence.py' -v
python -m unittest discover -s tests -p 'test_v21_compact_navigation_context.py' -v
python -m unittest discover -s tests -p 'test_v22_active_pitch_heading.py' -v
```

Regression fixtures recreate source IDs `obs_0012`, `obs_0019`, `obs_0049`, `obs_0050` and their opposing 225°/45° headings without claiming ground-truth height. An identical Habitat episode should be run to measure actual avoidance of self-retracing, redundant LOOK, real forward steps and official Score.

## Limitation

An executed MOVE_FORWARD with RGB-D change is a factual motion/action receipt, not trusted metric localization. Without verified 3D poses or high-confidence viewpoint re-identification, one cannot deterministically certify descending, ascending or revisiting a particular stair flight merely from opposite headings.
