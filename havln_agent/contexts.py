"""Single source-linked context projection for entire-mission navigation.

No independent second summary of observations, waypoints, failed moves or
natural-language progress is kept here. Both agents read mission_evidence()
from the same immutable Memory records.
"""
import copy
from .mission_history import mission_evidence
from .navigation_history import navigation_episode_history
from .contracts import CAMERA_PITCH_LIMIT_DEG, camera_look_options

def fields(value,names):
    return {k:copy.deepcopy(value[k]) for k in names
            if isinstance(value,dict) and k in value}

def navigation_context(runtime):
    # The controller pose is the canonical source for pitch and heading.
    # Read it once so camera_controls and mission_evidence cannot disagree,
    # and so lightweight/mock controllers need not expose duplicate fields.
    pose=runtime.motion_controller.pose()
    pitch_deg=pose.get('pitch_deg',0.0)
    evidence=mission_evidence(runtime.memory,pose,
                              max_observation_tasks=3)
    return dict(
        memory=evidence,
        episode_history=navigation_episode_history(runtime.memory,pose),
        context_contract='navigation_context_v2_complete_causal_history',
        actuator=dict(
            forward_step_m=runtime.camera.forward_step,
            turn_step_deg=runtime.camera.turn_angle,
            max_direct_segment_m=3.0),
        camera_controls=dict(
            current_pitch_deg=pitch_deg,
            pitch_limits_deg=[-CAMERA_PITCH_LIMIT_DEG,CAMERA_PITCH_LIMIT_DEG],
            tilt_step_deg=runtime.camera.tilt_angle,
            allowed_look_now=camera_look_options(
                pitch_deg,runtime.camera.tilt_angle),
            tool_look_enum_stable=['none','up','down'],
            pitch_limit_is_recoverable_tool_feedback=True),
        actions_remaining=max(0,runtime.config['max_steps']-
                              len(runtime.memory.actions)-1),
        navigation_mode=getattr(runtime,'navigation_mode','OBSERVE'),
        liveness_diagnostic=copy.deepcopy(
            getattr(runtime,'navigation_liveness',None)),
        current_rgb_image_attached_separately=True)

def compact_navigation_context(state):
    """Remove duplicate slices; keep ALL events in the canonical timeline.

    The former projection repeated the last three observation tasks, four
    tool receipts, three historical image descriptions and five motion
    segments in addition to the full episode log. Those are now accessed
    through episode_history.timeline and its complete frame catalog.
    Keep only *derived current* evidence in memory, such as live stance
    coverage, blocker ROIs and directional intent provenance.
    """
    output=copy.deepcopy(state)
    if 'episode_history' in output:
        memory=output.get('memory') or {}
        for field in (
                'observation_tasks','historical_frames',
                'recent_tool_receipts','observed_forward_action_segments'):
            memory.pop(field,None)
        memory['full_chronology_path']='episode_history.timeline'
        memory['all_frame_index_path']='episode_history.rgbd_observation_catalog'
    return output

def planner_context(runtime):
    return dict(instruction=runtime.memory.task.instruction)
