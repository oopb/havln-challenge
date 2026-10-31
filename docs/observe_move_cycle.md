# OBSERVE → MOVE navigation

Keep TaskPlanner's ordered semantic subgoals. NavigationAgent runs **only in OBSERVE**:
it examines the latest RGB-D geometry, current or remembered VLM target evidence,
and a concise navigation history, then chooses exactly one tool. Visual/geometry
tools may refine the next short move; a `navigate_local` call performs **MOVE**
of up to three atomic actions using the existing LocalMotionController. The
runtime then returns to OBSERVE for the next local decision. There is no separate
route-level planner, route ID, or new navigation tool.

## Execution

1. OBSERVE: reuse a relevant pending waypoint or choose a feasible short
   candidate toward the current semantic subgoal.
2. MOVE: execute the chosen `navigate_local` call. Fresh depth, terrain,
   human and collision checks remain active for every forward action.
3. Store the real execution receipt and RGB-D motion estimate. A semantic
   direction contradiction (for example, estimated UP motion during descent)
   ends the batch with `REQUIRES_ATTENTION` and is visible to the next OBSERVE.
4. Repeat until the independent progress verifier confirms the current
   semantic subgoal. TaskPlanner `continue` reviews do not cancel the waypoint.

The `navigation_mode` field and `navigation_mode_changed` events appear in
state/trace logs. OBSERVE may perform in-place TURN/LOOK actions to obtain a
missing view; this does not mean navigating a new global route. MOVE does not
make an extra NavigationAgent model request.

## Memory

`navigation_memory` is derived from the existing MemoryService action and
tool ledgers. It exposes recent forward direction, vertical motion and its
uncertainty, the last motion consistent with the subgoal, and recent failed
waypoints. The full memory and per-step evidence remain in the run traces.
These are RGB-D estimates, not simulator ground-truth position or task proof.

## Automatic diagnostic archives

The top-level `scripts/run_api_agent.py` process now calls the existing
diagnostic packer automatically after normal completion **and after a failed
worker run**, provided a trace directory exists. Worker subprocesses never
pack individually.

Default output: `<output-dir-parent>/diagnostic-archives/<run-name>-<UTC>-diagnostic.zip`.
This folder is on the same host-mounted volume as `--output-dir`.
Archives contain API/tool/step/progress/safety/memory/worker logs and selected
RGB-D frames; cache and worker task configuration files are omitted.

- `--archive-frames auto` (default): include ALL RGB-D frames for
  `--episode-limit 1`, only critical frames for larger runs.
- `--archive-frames all`: keep every RGB-D frame.
- `--archive-frames critical`: compact selection plus motion/obstacle evidence.
- `--archive-dir /workspace/agent-runs/diagnostic-archives`: override location.
- `--no-auto-archive`: explicitly disable automatic pack.

`--worker-task` and `--preflight` do not archive. Failed simulator startup
before any trace directory exists cannot produce a useful archive.

Audit follow-up: waypoint summaries preserve goal-distance and direction alignment; successful and failed entrypoint runs are covered by tests; startup worker logs are archived even without an episode trace. Architecture IDs match in episode summaries and runtime state.


## On-demand geometry diagnostics (post-audit)

During ordinary OBSERVE/MOVE operation the robot performs a fast **live
RGB-D body-depth collision probe** for the next 0.25 m (or turn-aligned
short waypoint). It no longer requires fitting stair support planes as a
precondition to each short forward action or each waypoint query. This avoids
the failure in `tool_0026`: plain depth was FREE and tread support SUPPORTED,
but the separate terrain-masked body sweep classified the path BLOCKED.

The probe still returns BLOCKED for measured near-body occupancy and NEEDS_VIEW
for insufficient depth or unknown human localization. These reports end MOVE
and return to NavigationAgent. The agent decides when to call
`inspect_geometry`, which preserves the full terrain/step-height analysis as
ON-DEMAND diagnostics, not an automatic motion veto. RGB-D translation
estimation and post-step landing evidence remain available for subgoal
verification. A probe saying FREE is not proof of safe stair traversal or
task completion; actual Habitat outcome must be checked in smoke tests.
