# Finite action plans with optional RGB-D odometry

## Design goal

The Habitat simulator accepts calibrated TURN and MOVE_FORWARD actions. RGB-D
translation registration is **not** needed to issue those actions. A failed
feature-matching estimate is missing metric evidence, not proof that an action
failed. In batch4-10, repeatedly treating UNKNOWN estimates as lost pose
caused avoidable reanchoring and re-observation.

## Model-owned execution

1. NavigationAgent chooses the target bearing and finite distance, such as
   `navigate_relative(turn_deg=-30, distance_m=1.0)`.
2. LocalMotionController quantizes turns to 15-degree actions and distance to
   calibrated forward steps (typically 0.25m). The route is a **finite action
   sequence**, never an inferred world-coordinate endpoint.
3. The executor performs the selected turns and forward actions, collecting
   public RGB-D after each action. The model may explicitly request geometry
   inspection; automatic geometry/terrain/human vetoes remain disabled in this
   simulation configuration.
4. `estimate_translation` remains an optional diagnostic, recording registered
   3D translation and vertical motion when measurements meet consensus checks.
   `UNKNOWN` is recorded as UNKNOWN without adding an invented 0.25m
   uncertainty or incrementing `motion_epoch`. It does **not** stop the
   model's finite action plan.
5. If the public RGB-D is byte-identical before and after a forward attempt,
   the current route yields `REQUIRES_ATTENTION` and a no-motion alert.
   This indicates no observable change, **not** proven collision.
6. Otherwise, after the requested forward count is executed, the executor
   returns **`ACTIONS_COMPLETED`**, `action_intent_completed=true`,
   `target_reached=false`, `endpoint_estimate_valid=false`, and
   `remaining_distance=null`. The nominal commanded distance is never
   reported as verified traversal.
7. If the global Episode budget interrupts a route, its commanded forward
   count and remaining steps persist. NavigationAgent may resume the *same*
   model-authored bearing through `navigate_local` without needing an
   old metric x/z endpoint.
8. A changed public frame after forward increments `viewpoint_generation`.
   This keeps separate visual stances distinct even without reliable metric
   translation; UNKNOWN alone no longer resets the navigation `motion_epoch`.

## Evidence versus control

`position_coordinates_valid=false` and a null model-visible
`position_xz_m_est` mean that metric coordinates are not reliable following
an unregistered forward. Old projected landmark positions cannot be used as
verified new destinations. New current-frame images, visual target hypotheses,
command action history and yaw still inform model decisions.

Independent progress verification **continues to require** registered motion,
visual grounding, observed level landing support and task evidence as
appropriate. Executing 4 nominal 0.25m commands cannot by itself prove 1m
translation or stair descent. The verifier intentionally does not replace
unknown displacement with the nominal command length.

## Validation

```bash
python -m unittest discover -s tests -p 'test_direct_motion_execution.py' -v
python -m unittest discover -s tests -p 'test_odometry_gap_navigation.py' -v
python -m unittest discover -s tests -p 'test_progress_full_transition.py' -v
python -m unittest discover -s tests -p 'test_motion_consensus.py' -v
```

Habitat closed-loop performance and collision metrics require a new episode
evaluation. This is a simulation-specific experimental policy and should not
be directly applied to a real robot without an independent safety system.
