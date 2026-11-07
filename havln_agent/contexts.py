"""Single source-linked context projection for entire-mission navigation.

No independent second summary of observations, waypoints, failed moves or
natural-language progress is kept here. Both agents read mission_evidence()
from the same immutable Memory records.
"""
import copy
from .mission_history import mission_evidence
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
    # Projection is bounded centrally; do NOT add a second rollup here.
    return copy.deepcopy(state)

def planner_context(runtime):
    return dict(instruction=runtime.memory.task.instruction)
