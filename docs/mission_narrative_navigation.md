# HA-VLN: continuous natural-language mission navigation (v7)

This is the active inference architecture after the complete task-control refactor.
It replaces the previous numbered `sg_0` / `sg_1`, TaskPlanner, forced
`VERIFY` and stage-completion/reopening protocols. The official external
HA-VLN scorer remains unchanged.

## Roles and authority

- **NavigationAgent:** receives the entire instruction and current authentic
  RGB, pose/actuator state, natural-language progress and intent, recent
  registered RGB-D motion evidence, authentic obstacle evidence, and recent
  observations/waypoints. It decides when to observe, execute a finite motion
  segment, continue a partial segment, or STOP. It can revise its own progress
  narrative without reopening task IDs.
- **Observation Agent:** receives the entire instruction, current narrative,
  and a **natural-language observation delegation**. During one observation
  session, every vision call retains all previous authentic RGB-D views,
  previous VLM replies, absolute headings and pitch. It autonomously performs
  legal TURN/LOOK, but never MOVE or STOP. It returns narrative observation,
  navigation suggestion and optional source-anchored local waypoint hypotheses.
- **Memory:** the single home for sensor facts, model interpretations,
  observation history and movement feedback. Sensor frames/measurements and
  navigation hypotheses remain distinguished by provenance. A candidate
  waypoint is never an independent safety certificate or task-completion
  verdict.
- **ToolExecutor / Habitat:** executes model-chosen atomic actions within the
  official action budget and actuator limits. It may interrupt a blocked
  near-field forward step using current central Depth evidence. No goal-specific
  stair rules, automatic post-step milestone verification, `candidate_id`
  prediction scoreboard, or automatic reopening of historical tasks exist.

## Example mission and delegation

Original instruction:

> Go downstairs, find the hall to the right, cross the group of people,
> and stop beside the piano.

Example revisable Memory:

```json
{
  "progress_narrative": "I appear to have descended the staircase; the hall entrance is not yet confirmed.",
  "navigation_intent": "Find and enter the right-side hall, then approach the piano.",
  "uncertainties": ["Whether the last staircase tread has been passed"],
  "evidence_observation_ids": ["obs_0066"]
}
```

Observation request:

> The complete mission is to descend, find the right-side hall, cross the
> group of people and stop at the piano. I believe the descent is mostly
> complete. Inspect the current RGB-D layout and identify a useful direction
> toward the hall, or point out remaining descending stairs.

Observation response (illustrative, **not** a measured route):

```json
{
  "observation_summary": "A doorway is visible to the right; distance remains uncertain.",
  "navigation_suggestion": "Try a short approach toward the opening.",
  "waypoints": [{
    "source_observation_id": "obs_0066",
    "heading_at_capture_deg": 90,
    "absolute_heading_deg": 90,
    "relative_bearing_deg_from_current": -45,
    "distance_m_hint": null,
    "traversability": "UNCONFIRMED",
    "motion_authorized": false
  }]
}
```

The final relative bearing must always be recalculated against the robot's
**current** heading; saved historical views cannot grant live clearance.

## Execution

The entry point is `scripts/run_api_agent.py`. Two independent vision-capable
roles are configured: `navigation_agent` (current real RGB on each decision)
and `visual_perception` (real RGB and calibrated Depth). The retired
`task_planner` API key and YAML section are not required.

Pipeline identity: `havln-agent-v7-mission-narrative`. Its output directories
must not be mixed with earlier v6 outputs. Existing batch, GPU, terminal-log,
critical-frame capture and diagnostic packaging options remain in place.

To test Python imports/schema and new architecture contracts:

```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_mission_navigation_architecture.py' -v
```

For a real Habitat smoke test, use a **fresh output directory**:

```bash
python /toolkit/scripts/run_api_agent.py \
  --config /toolkit/configs/agent.yaml --splits val_seen \
  --episode-limit 1 --max-steps 88 --gpu-ids 0 --batch 1 \
  --terminal-log on --archive-frames critical \
  --output-dir /workspace/agent-runs/mission-v7-smoke
```

The new Memory records and action trace establish what was executed, **not**
that the full destination was reached. Run `val_seen` and `val_unseen`
on the actual simulator and compare full-mission scoring, forward fraction,
collision metrics, API use and early STOP before asserting performance gains.

The old tests dedicated to numbered milestones, `VERIFY`, or Planner
decomposition test a retired contract; they need to be migrated before
treating the entire legacy test suite as an acceptance gate.
