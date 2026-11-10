"""Single source-linked context projection for entire-mission navigation.

No independent second summary of observations, waypoints, failed moves or
natural-language progress is kept here. Both agents read mission_evidence()
from the same immutable Memory records.
"""
import copy
import math
from .mission_history import mission_evidence
from .contracts import CAMERA_PITCH_LIMIT_DEG, camera_look_options

def fields(value,names):
    return {k:copy.deepcopy(value[k]) for k in names
            if isinstance(value,dict) and k in value}

def heading_turn_reference(pose, camera):
    """Exact action-result headings, not an automatic steering suggestion.

    Positive LEFT yaw and negative RIGHT yaw are computed once from the
    controller pose; no inferred metric position or language-parser gate.
    """
    current=round(math.degrees(pose.get('heading',0))%360,2)
    step=float(camera.turn_angle)
    degrees=[round(i*step,2) for i in range(
        1, int(180//step)+1)]
    return dict(
        current_heading_deg=current,
        convention='LEFT increases yaw; RIGHT decreases yaw; 0..360 wrap',
        action_results=[
            dict(turn_deg=amount,
                 left_result_deg=round((current+amount)%360,2),
                 right_result_deg=round((current-amount)%360,2))
            for amount in degrees],
        reference_only_model_chooses_motion=True)


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
        turn_heading_reference=heading_turn_reference(pose,runtime.camera),
        actions_remaining=max(0,runtime.config['max_steps']-
                              len(runtime.memory.actions)-1),
        navigation_mode=getattr(runtime,'navigation_mode','OBSERVE'),
        liveness_diagnostic=copy.deepcopy(
            getattr(runtime,'navigation_liveness',None)),
        current_rgb_image_attached_separately=True)

def compact_navigation_context(state):
    """v12 bounded mission_evidence context, without a duplicated episode log.

    Full historical source records remain in Memory and query_task_memory,
    but are not serialised on each navigation decision.
    """
    return copy.deepcopy(state)

def planner_context(runtime):
    return dict(instruction=runtime.memory.task.instruction)
