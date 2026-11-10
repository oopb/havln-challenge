# v22: eval-36 pitch observation and reliable heading resolution

## Diagnosed failure
Eval-36 spent nearly all of 150 atomic actions turning horizontally, with only one LOOK_DOWN and no LOOK_UP. In model decisions, e.g. current yaw 270 and intended yaw 180, the model sometimes chose LEFT instead of the shortest RIGHT turn. The previous pure-turn example also exclusively used look=none, while the v17 rollback dropped concrete ObservationAgent instructions for pedestrians/rails/doorways.

## Implementation
- NavigationAgent now explicitly treats current yaw with a different pitch as a potentially useful new observation: LOOK_DOWN for floor/stair edge details, LOOK_UP for distant hallway/doorway context when the camera is tilted down. This is based on a question and sensor evidence, never a mandatory scan or forced recenter.
- In ambiguous people/rail/stair/doorway settings, the navigator can ask ObservationAgent to inspect near-floor, far passage and left/right possible gaps across useful pitch views. The VLM session Prompt agrees and still controls its LOOK_UP/LOOK_DOWN decisions.
- Two exclusive model turn forms are allowed for `navigate_relative` and `turn_and_observe`:
  1. Model selects absolute global yaw `target_heading_deg` (0..360), optionally with `look` or positive `distance_m`; no `turn_direction` or `turn_degrees`.
  2. Legacy image-relative `turn_direction=LEFT/RIGHT/NONE` and `turn_degrees>=0` with optional LOOK or positive distance; no `target_heading_deg`.
- Absolute yaw is converted to existing legacy motor arguments immediately after the NavigationAgent model response, before scheduler/tool dispatch. The converter uses current controller heading and the calibrated turn angle; it has no semantic route veto. 270 to 180 -> RIGHT 90; 300 to 30 -> LEFT 90; 150 to 90 -> RIGHT 60. Zero or 360 is normalized, 180-degree ties resolve RIGHT. All resolution decisions are recorded in `navigation_absolute_heading_resolved` events. API traces preserve original model output.
- Actual simulator action safety is unchanged: every atomic MOVE_FORWARD still uses fresh live calibrated Depth, confirmed BLOCKED vetoes, and UNKNOWN is inconclusive. #90 omits inconclusive predecision Depth from NavigationAgent requests; #91 still creates a read-only compact projection of canonical Memory.
- No new tool name, second semantic validator, forced LOOK, hard-coded up/down goal confirmation, synthetic metric pose, or automatic pitch reset.

## Offline regression
```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_v22_active_pitch_heading.py' -v
python -m unittest discover -s tests -p 'test_v20_direction_depth.py' -v
python -m unittest discover -s tests -p 'test_v21_compact_navigation_context.py' -v
python -m unittest discover -s tests -p 'test_recoverable_camera_pitch.py' -v
```

## Same-episode acceptance
With a fixed model/config and the same val_seen episode, compare #93 versus v22:
1. LOOK_UP / LOOK_DOWN count, changes in pitch actually executed and VLM `observe_for_goal` calls (not simple maximization of LOOK count).
2. Repeated same-yaw/pitch views, consecutive horizontal turns without forward, and actual forward count and unchanged-RGBD stalls.
3. Heading error after each `target_heading_deg` conversion and resulting ordered atomic actions.
4. Side effect on API token/length, confirmed BLOCKED safety and official Score when ground-truth metrics are available.

Offline tests cannot prove improved navigation success; run Habitat in the configured H100 environment.
