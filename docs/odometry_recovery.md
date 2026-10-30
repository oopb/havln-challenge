# Robust short-horizon navigation and odometry gaps

## Motivation

In smoke-500-15 a second real MOVE_FORWARD was followed by an UNKNOWN
RGB-D translation. The estimator found 43 mutual matches, including a
23-match 3D consensus spread across 7 spatial tiles, with ~1.2 cm metric
residual, but rejected it because the consensus was 53.5% rather than the
strict 65% threshold. The old executor then discarded the remaining waypoint,
causing repeated semantic scans instead of local geometric movement.

## Live per-step control

1. Navigation Agent selects a short Waypoint (0.5 m / two calibrated
   0.25 m forward actions in the current geometry generator).
2. Each TURN/forward action is executed by Local Motion Controller. Before
   every forward action, Runtime reads current public RGB-D and verifies
   geometry, floor support and people. The depth check is never skipped.
3. Translation is estimated from public consecutive RGB-D frames, not from
   commanded distance or simulator ground-truth position.
4. A strict estimate still requires >=65% match consensus. A second
   distributed-static branch requires **all**: >=20 inliers, >=52% ratio,
   image coverage >=65% on both axes, >=7 spatial tiles, >=0.6 m depth
   span, <=0.02 m metric residual. Camera motion plausibility checks and
   terrain-direction alerts remain unchanged.
5. If both fail, one UNKNOWN does not immediately discard a short Waypoint.
   Existing bounded relative continuation may execute only its remaining
   forward attempts at the same heading, with fresh independent checks.
   It cannot increase the original two-step horizon or claim measured arrival.
6. If an UNKNOWN remains at the horizon, the tool reports
   `REQUIRES_ATTENTION` (possibly surfaced as outer `PARTIAL` because
   actions already executed), `endpoint_estimate_valid=false` and
   `remaining_distance=null`; the missing translation is still **UNKNOWN**.
   The post-action current public RGB-D becomes a *new relative waypoint
   anchor*; no missing distance is filled in. Old geometry and landmark
   anchors remain invalid across the changed `motion_epoch`.
7. Navigation Agent must now request new locally measured waypoints instead
   of re-identifying the same static landmark. It retains high-level control
   of choosing the next route; the local controller never self-selects a
   new semantic destination.

## Safety limitations

Distributed consensus provides an estimate, not ground truth: nearby moving
people or repetitive texture can still cause motion registration errors.
The fallback is deliberately stricter on coverage and residual than the
ordinary 65% match path. Neither route continuation nor local reanchoring
can authorize MOVE_FORWARD without current Depth and terrain checks.

After an unknown translation, numerical x/z history is not a global
localization; its meaning is local within the new epoch only. Do not infer
stair descent or semantic subgoal completion from commanded movement.

## Verification

```bash
python -m unittest discover -s tests -p 'test_motion_consensus.py' -v
python -m unittest discover -s tests -p 'test_odometry_gap_navigation.py' -v
```

These regressions validate both a distributed 23-of-43 match scene and a
concentrated moving-object distractor, along with bounded safe motion,
unknown-distance feedback and new local Waypoint validation. Real Habitat
success requires rerunning the same episode and comparing verified forwards,
model decisions, action budgets, and semantic descent evidence.
