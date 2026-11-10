# v27: All model-visible heading evidence is relative to CURRENT camera

## Core contract

**The model does not receive 0–360° absolute robot heading values.** The runtime still retains those exact angles in canonical Memory, movement, geometry, saved trace, replay and scoring. The model-facing copy uses a signed offset relative to the CURRENT RGB camera: forward=0°, positive means LEFT, negative means RIGHT. Exactly 180° is represented as -180° (tie is ambiguous, not a preferred side).

For example, with internal robot camera yaw 90°:
- historical RGB at internal yaw 210° → `source_view_offset_deg_from_current: 120`;
- earlier forward at internal yaw 255° → `view_offset_deg_from_current: 165`;
- earlier block at internal yaw 45° → `view_offset_deg_from_current: -45`;
- current RGB at internal yaw 90° → `view_offset_deg_from_current: 0`.

These are **orientation differences**. A photo captured at an earlier executed-forward stance is NOT a current route target even when its orientation difference is computable. Only same-stance, source-verified passage hypotheses can include a usable relative candidate bearing. Model actions always remain `turn_direction=LEFT|RIGHT|NONE`, `turn_degrees>=0` (and optional independent pitch LOOK).

## Exact source audit and changes

1. `havln_agent/model_angles.py` (new): a bounded, pure, deep-copied projection for model text JSON. Converts `heading_deg`, `source_heading_deg`, `current_heading_deg`, `last_heading_deg`, `heading_at_capture_deg`, `absolute_heading_deg`, yaw/azimuth keys, pose heading radians and 360-degree scan bins. It also converts **explicitly heading-labelled numbers in model-authored historical prose** (for example `heading 255`, `yaw:210°`) to current-relative LEFT/RIGHT. It preserves calibrated tilt, relative signed turn commands, unqualified angular distances, sweep budget and source IDs. Text conventions describing 0–360° world headings are dropped. Never mutates canonical Memory.
2. `havln_agent/navigation_agent.py`: applies that projection LAST, after current RGB and up to five verified historical RGB images are aligned. Model never receives the original absolute `history_media.manifest`, current pose angle, historical segment/block headings or old view coverage bins. Prompt now explains signed offsets, not world heading arithmetic.
3. `havln_agent/visual_perception.py`: every VLM call through VisualPerception.complete receives the same projection, including current/old RGB-D checkpoints, observation history/mission memory, obstacle ROI descriptions, grounding and panoramic memory. `project_observation_conversation` separately reprojects the **raw previously stored VLM user requests** to this turn's new current camera orientation before constructing the stateless provider request. Raw session messages, images, assistant decisions and source hashes stay unchanged; old same-stance target bearings are recomputed from stored internal source yaw.
4. `havln_agent/panorama_memory.py`: the additional direct NavigationAgent panorama call also projects its model context; tool description now matches LEFT/RIGHT/NONE without obsolete signed-angle actions. The VLM first attempt and JSON format retry both use the converted request, not an unsanitized first attempt.
5. Tests cover conversion of real eval-39-style headings, wraparound, history/depth/coverage/pose branches, no cross-stance motor bearing, source unchanged, assistant-response preservation, repeated VLM turn rebase, and NavigationAgent's actual model context.

## Cautions

Do NOT change source Memory's yaw into relative offsets; the next turn would double-convert history and break simulator geometry. Do NOT transform signed `turn_degrees`, robot pitch, visual HFOV or sector/sweep widths as if they were absolute headings. Source-bound natural-language mission text is preserved verbatim, so arbitrary numbers quoted by a human are NOT silently rewritten as robot headings. Model-authored freeform past narrative may contain mistakes; projection only guarantees structured model-provided orientation metadata, not the truth of historical VLM descriptions.

## Regressions

```bash
python -m compileall -q havln_agent
python -m unittest discover -s tests -p 'test_v27_model_relative_angles.py' -v
python -m unittest discover -s tests -p 'test_v26_eval39_route_quality.py' -v
python -m unittest discover -s tests -p 'test_v23_route_evidence.py' -v
python -m unittest discover -s tests -p 'test_v13_navigation_context.py' -v
```

After deployment inspect an actual `api.jsonl` request. Confirm absence of raw structured `heading_deg`, `source_heading_deg`, `absolute_heading_deg`, `observed_heading_bins_deg` fields in BOTH navigation and visual wire contexts, with correct signed model-relative offsets and original source IDs. Inspect session-turn N+1 input: all previously transmitted raw session user messages must be reprojected relative to view N+1, including same-stance visual target bearings. Compare action continuity, LEFT/RIGHT mistakes and turn budget to the eval-40 reference episode.
