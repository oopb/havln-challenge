# Evidence liveness guard (smoke-500-17)

## Observed failure

The 150-step val_seen-1 test was manually interrupted after ~1h26m.
The saved archive contains 96 executed atomic actions (2 MOVE_FORWARD,
68 TURN and 26 LOOK), but 1,060 successful tool records, 1,064
Navigation Agent decisions and 872 NO_NEW_INFORMATION tool outcomes.
Even when every camera heading was already visited, the LLM kept
calling turn_and_observe with a previously rejected static view.

The previous observation deduplication patch stopped *physical TURNs*,
not paid model inference. The scheduler loop was guarded solely by
`len(memory.actions) >= max_steps-1` plus the much larger physical API
budget. A tool returning NO_NEW_INFORMATION did not use an atomic action,
so it could be invoked indefinitely.

## Fix

`EvidenceLivenessGuard` tracks a *physical evidence fingerprint*:
executed action count, current RGB-D observation ID, subgoal index, and
verified subgoal completion. Rewording the Task Planner text does not
create a new fingerprint. New scene observations, ANY real atomic robot
action, and verified progress reset the no-evidence streak.

After eight decision/tool calls without changed physical evidence
(configurable `max_unchanged_evidence_decisions`), the scheduler
explicitly requests one high-level Task Planner review of the
unproductive route. If another eight calls do not produce a new
observation/action/subgoal transition, the scheduler records
`navigation_liveness_exhausted` and terminates with
`finish_reason=navigation_no_new_evidence`: this is an unsuccessful
run, not an invented success, and never bypasses collision checks.

A separate `max_model_decisions` ceiling (default
`2 * effective max_steps`) prevents a model from making unbounded paid
calls even if tool calls vary. It is distinct from robot action budget
and does not authorize or reject individual MOVE_FORWARD actions. These
numbers are not a 12-step movement/scan quota: the watchdog resets
upon any actual action or new RGB-D evidence.

The LLM receives `decision_liveness` after early no-information
decisions so it can choose a real novel action, a route review or a
safe failure rather than incurring the watchdog exit.

## Physical geometry remains unresolved

The first MOVE_FORWARD was followed by measured occupancy:
required .25 m; clearance .136 m. The second MOVE_FORWARD was followed
by `measured_step_or_drop_exceeds_collision_model` despite a depth
body sweep with clearance .25 m. Both RGB-D translations estimated
vertical UP rather than verified descent. This issue is separate from
the unbounded inference loop. The terrain step/drop proof (height
samples and maximum supported step height) now propagates into the
Navigation Agent's measured_collision summary; it does not turn a
blocked stair into a false free route.

## Tests and diagnosis

```bash
python -m unittest discover -s tests -p 'test_scheduler_liveness.py' -v
python -m unittest discover -s tests -p 'test_smoke50012_feedback.py' -v
```

After rerunning a Habitat scene, compare:
- `summary.json`: `navigation_liveness`, `finish_reason`, verified forwards
- `events.jsonl`: `navigation_liveness_review` and `navigation_liveness_exhausted`
- `tools.jsonl`: `measured_collision.terrain` step/drop evidence
- `api.jsonl`: repeated tool selection before/after last RGB-D/action change.

A legitimate new observation/action resets the guard; these tests do
not establish that the physical environment has a safe traversable path.
