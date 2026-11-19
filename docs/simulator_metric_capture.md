# Default Habitat metric capture and local score (API + Codex)

All **new** API Agent and Codex/MCP episodes capture real simulator
measurements by default. No extra API call or user flag is needed. This does
not retrofit earlier traces.

## Isolation and measurement sources

- habitat_adapter.environment_config enables Habitat **measurement** modules
  in addition to existing public RGB/Depth and instruction sensors.
- val_seen / val_unseen enable DISTANCE_TO_GOAL, SUCCESS, SPL, ORACLE_SUCCESS,
  PATH_LENGTH, STEPS_TAKEN, COLLISIONS, and COLLISIONS_DETAIL.
- Held-out test enables PATH_LENGTH, STEPS_TAKEN, COLLISIONS, and
  COLLISIONS_DETAIL. Goal-dependent metrics are **not requested** when
  scoring truth may be hidden. Absent test goals or reference collision
  allowances never become synthetic zeroes.
- DISTANCE_TO_HUMAN is intentionally omitted: the upstream implementation
  temporarily applies an agent movement while computing it.
- After each actual environment action, the evaluator records the Habitat
  metrics dictionary and real simulator robot position in a separate file,
  without feeding those fields back to the NavigationAgent, VLM, Codex,
  MCP status, or tool responses.

These are **same-run live Habitat measurements**, not submission replay
metrics. The dynamic-human timeline may differ on CodaBench.

## Files and availability

Every trace contains simulator_metrics.jsonl (reset plus one sample per
executed atomic action) and simulator_metrics.json (final true values,
reference collision allowance if published, complete/partial flags).
Incomplete episodes retain captured data but are not scored.

The run root contains score_metrics.json with per-split evidence and
observed component means where available: end distance, raw navigation
arrival success, raw collision count, path length and simulator step count.
For episodes with the full required evidence it also computes adjusted
SR, NE, CR, TCR and composite Score using competition_scoring.py.

Partial-batch results explicitly carry requested_episodes,
fully_scored_episodes, missing_or_partial, and a diagnostic scope.
An available subset Score is **not a leaderboard score**. For Phase 2,
the raw metrics normally exist, but full local Score usually does not.

Replay Studio can import the score_metrics.json root file automatically.
Archives now include the evaluation trace files without duplicating RGB-D.

## Verify a new run

~~~bash
cd /toolkit
python -m unittest discover -s tests -p 'test_simulator_evaluation.py' -v
cat /workspace/agent-runs/EXAMPLE/score_metrics.json
cat /workspace/agent-runs/EXAMPLE/traces/val_seen-1/simulator_metrics.json
~~~

Do not pass the collected simulator truth to the inference Agent or MCP
tools. Official scores come only from CodaBench replay, not live inference
traces; real-time human animation can produce replay differences.
