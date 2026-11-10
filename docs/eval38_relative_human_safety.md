# v24: Restore relative turns and use current RGB-D for human clearance

## Why
The model-facing `target_heading_deg` alternative was introduced in v22 to bypass incorrect model heading arithmetic. It created two incompatible ways for NavigationAgent to express the same motor action. The user has explicitly selected **relative LEFT/RIGHT/NONE** as the only model motion interface. Internal numeric heading labels remain sensor/history metadata, not motion commands.

In eval-38 (episode val_seen-1) the robot navigated the stairway and foyer with more continuity but approached humans too closely. Measured Depth remained mostly UNKNOWN (not a clearance certificate), and side-body obstacles were rejected as central-block evidence to avoid false stair/rail vetoes. Raw occupancy near the person was not consistently correlated with authentic, current human semantic regions.

## Changes
- Both `navigate_relative` and `turn_and_observe` now require `turn_direction` and `turn_degrees`; neither accepts `target_heading_deg`. `signed_turn` alone converts relative model decisions into actuator signs. The previous absolute-heading resolver was removed; camera yaw in Memory and VLM visual candidate metadata is **not** an action field.
- One **optional** `visible_people_boxes` (0..4 normalized [left,top,right,bottom] entries) attaches to the NavigationAgent response as CURRENT-RGB model evidence. The model must describe humans from the actual first image, not old historical RGB. Geometry applies these boxes **only** when the RGB hash AND source pose match exactly, then measures Depth using the camera's actual pitch.
- A confirmed model-person ROI intersecting the next forward sweep plus `human_clearance_margin_m=0.45` yields an instantaneous measured person-buffer veto. The default robot radius 0.10m plus existing safety_margin 0.05m gives 0.60m effective path-buffer radius, grounded in **measured Depth pixels** rather than hallucinated person range. Absence of matched boxes, invalid Depth or ordinary UNKNOWN **does not** block the step.
- In a human-visible near range (default 3m), the executor returns control to NavigationAgent after a physically executed forward, rather than proceeding with later commands against stale human boxes. If the model changes yaw first, old regions are invalid; request fresh RGB instead of applying old regions to a new angle. This is a live people-recheck, not a global one-step limit.
- Separate source-labelled `near_field_proximity` reports coherent measured tall occupancy from real current Depth (lateral offset, forward range), explicitly unclassified as human; positive facts can be shown alongside current RGB even when wide Depth status is UNKNOWN. This never creates a generic UNKNOWN veto.
- Existing central obstruction preflight, #90 conclusive-only Depth result, #91 source-linked compact Memory, v23 route/LOOK continuity, ObservationAgent and official scorer are preserved. No new model API per atom.

## Test and compare
```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_v22_active_pitch_heading.py' -v
python -m unittest discover -s tests -p 'test_v24_person_clearance.py' -v
python -m unittest discover -s tests -p 'test_forward_preflight_v9.py' -v
python -m unittest discover -s tests -p 'test_v20_direction_depth.py' -v
```

Compare the same eval-38 episode: actual minimum person distance from calibrated RGB-D in the specific model/person ROI, positive human-buffer `BLOCKED`, forward recheck events, near-field warnings, no new stair-loop, total forward actions and task Score. People boxes are **model hypotheses** and may be absent; therefore the buffer is a conditional evidence-based safety control, NOT a certified human detector. Official collision clearance and motion success remain independently evaluated.
