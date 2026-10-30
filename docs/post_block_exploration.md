# Post-obstacle exploration contract

## Root cause (smoke-500-16)

The robot made three measured `MOVE_FORWARD` actions, then its next
0.25 m body sweep was BLOCKED by a nearby measured depth obstacle. Its
`navigate_local` tool reported outer `status=PARTIAL` because one action
executed, with real `interrupted_status=BLOCKED`. Old
`navigation_context` checked only outer BLOCKED and compared the PRE-move
pose against the current (post-move) location. Consequently 99 Navigation
Agent decisions contained no `recent_navigation_block` evidence.

Afterward, 21 waypoint checks had zero executable paths; 66
`turn_and_observe` calls moved the camera in only a few repeatedly
visited heading/pitch pairs. Relative `NEEDS_VIEW +/-30` bearings were
reissued after each return to an old absolute orientation. Previous
`revisited_static_view` warnings were merely suggestions.

## New contract

- `local_exploration.last_block` recognizes outer BLOCKED or
  `PARTIAL+interrupted_status=BLOCKED` and compares current location with
  `end_observation_id.pose`, NOT the tool's pre-execution pose.
  Harmless subsequent scans/queries do not erase the collision. A later
  completed navigation, physical displacement, or new odometry epoch
  invalidates stale obstruction context.
- `local_exploration.local_coverage` aggregates current-subgoal observed
  absolute yaw and pitch at the same estimated local stance. This history
  is **never a permit to move**, never a permanent obstacle blacklist, and
  does not pretend to be globally localized.
- `get_waypoint_candidates` returns `exploration.unexplored_directions`,
  `view_directions` excluding already-visited absolute camera views, and
  separate `revisited_view_directions` diagnostics. Once current-pitch
  search is exhausted, it reports the condition rather than issuing the
  same `NEEDS_VIEW +/-30` again.
- `turn_and_observe` refuses post-obstacle camera batches with no new
  heading/pitch coverage, returning `NO_NEW_INFORMATION` without
  spending any action budget. Turning toward a previously measured
  executable direction is still permitted for a NEW live recheck.
- `observe_for_goal` refuses similarly redundant sweeps before VLM or
  atomic TURNs. A different pitch or unvisited sector still works.
- For a potentially moving human, `turn_and_observe` accepts optional
  `purpose="dynamic_recheck"`. Previously visited camera sectors can
  be reobserved only on an explicit request, at most once per 60 seconds
  at the same stance, heading and pitch. A new pose/epoch resets this.
  Neither static history nor VLM guesses authorize forward motion.
- Navigation Agent receives the last measured block, view coverage
  history and next possible absolute bearings in its context; it should
  explore truly new directions, evaluate known safe *candidate* directions
  with live geometry, or request higher-level route review. It must not
  repeatedly scan two blocked headings.

## Verify

```bash
python -m unittest discover -s tests -p 'test_local_exploration.py' -v
```

For the actual Habitat episode, compare the first post-collision
`get_waypoint_candidates` result, percentage of repeated TURN/LOOK
frames, number of `NO_NEW_INFORMATION` refusals, API request count,
verified forwards, new absolute headings, and subgoal progress.
This change does not prove a safe geometric path exists or allow a
blocked MOVE_FORWARD.
