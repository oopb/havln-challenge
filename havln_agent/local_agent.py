"""Tool-free Qwen-VL short-horizon controller for a CLOUD-delegated route."""
from .api_client import APIError
from .local_frame_selector import select_local_frames

LOCAL_FOLLOW_PROMPT = (
    'You control one short robot motion on an ALREADY SELECTED route. '
    'Follow only local_goal, not the whole mission. Review the real RGB frames '
    '(last RGB is CURRENT) and CURRENT Depth. Historical images are NOT current clearance. '
    'LEFT/RIGHT are relative to CURRENT RGB; count is calibrated atomic steps. '
    'Choose FORWARD, LEFT, RIGHT, LOOK_UP, LOOK_DOWN or ESCALATE. '
    'If the route branches, conflicts with local_goal, appears finished, '
    'or cannot be identified, choose ESCALATE. No global replanning or STOP. '
    'Signal NORMAL only when the chosen short action clearly follows local_goal. '
    'For uncertainty or new route questions use ESCALATE and a non-NORMAL signal. '
    'Return only one JSON object, no explanations.'
)
LOCAL_LOOK_PROMPT = (
    'You are checking the view for an ALREADY SELECTED local route. '
    'Use the latest RGB and current Depth with the historical RGB frames. '
    'Choose a short LEFT, RIGHT, LOOK_UP, LOOK_DOWN, FORWARD, or ESCALATE. '
    'LEFT/RIGHT mean directions in CURRENT RGB, not historical view angles. '
    'Never choose a new route, global goal or STOP. '
    'When a branch, route change, possible arrival, or conflict needs reasoning, '
    'choose ESCALATE with a non-NORMAL signal. '
    'Return one JSON object only.'
)
LOCAL_ACTION_SCHEMA = dict(
    type='object', additionalProperties=False,
    properties=dict(
        action=dict(enum=['FORWARD', 'LEFT', 'RIGHT', 'LOOK_UP',
                          'LOOK_DOWN', 'ESCALATE']),
        count=dict(type='integer', minimum=0, maximum=3),
        signal=dict(enum=['NORMAL', 'UNCERTAIN', 'NEW_BRANCH', 'CONFLICT',
                          'GOAL_CUE', 'BLOCKED']),
        evidence_id=dict(type='string', minLength=1, maxLength=64)),
    required=['action', 'count', 'signal', 'evidence_id'])


def local_to_registered_command(decision, camera):
    """Deterministic action mapping; model cannot name tools or plan segments."""
    action, count = decision['action'], decision['count']
    if action == 'FORWARD':
        distance = round(count * camera.forward_step, 6)
        if not (0.25 <= distance <= 3.0):
            raise APIError('local_forward_distance_outside_executor_contract',
                           category='structured_field')
        return dict(tool='navigate_relative', args=dict(
            turn_direction='NONE', turn_degrees=0, distance_m=distance))
    if action in ('LEFT', 'RIGHT'):
        degrees = count * camera.turn_angle
        if not (0 < degrees <= 180):
            raise APIError('local_turn_outside_executor_contract',
                           category='structured_field')
        return dict(tool='turn_and_observe', args=dict(
            turn_direction=action, turn_degrees=degrees, look='none'))
    if action in ('LOOK_UP', 'LOOK_DOWN'):
        return dict(tool='turn_and_observe', args=dict(
            turn_direction='NONE', turn_degrees=0,
            look='up' if action == 'LOOK_UP' else 'down'))
    raise APIError('local_action_is_not_executable', category='structured_field')


class LocalAgent:
    def __init__(self, runtime):
        self.runtime = runtime

    def decide(self):
        r = self.runtime
        router = r.hybrid_router
        if not router.use_local():
            raise APIError('local_decision_without_cloud_delegation',
                           category='structured_field')
        r.refresh()
        preflight = r.forward_preflight(stage='navigation_decision')
        settings = r.hybrid_settings
        manifest, images, depth_index = select_local_frames(
            r, router.route, max_frames=settings.get('keyframes_max', 10),
            max_edge=settings.get('image_max_edge', 448))
        current = r.memory.current
        recent = r.memory.actions[-6:]
        context = dict(
            mode=router.state,
            local_goal=router.route['goal'],
            exit_condition=router.route['exit_condition'],
            route_id=router.route['route_id'],
            current_observation_id=current['observation_id'],
            front_depth_status=preflight.get('status', 'UNKNOWN'),
            camera_pitch_deg=r.pitch_deg,
            forward_step_m=r.camera.forward_step,
            turn_step_deg=r.camera.turn_angle,
            recent_actions=[dict(action=a['action'],
                changed_rgbd=(a.get('result') or {}).get('visual_change'))
                for a in recent],
            frames=manifest,
            current_depth_image_index=depth_index,
            relative_yaw_sign='positive LEFT, negative RIGHT',
            historical_depth_is_not_clearance=True)
        prompt = (LOCAL_LOOK_PROMPT if router.state == 'LOCAL_LOOK'
                  else LOCAL_FOLLOW_PROMPT)
        result = r.client.complete(
            'local_control', 'local_motion', prompt, context,
            LOCAL_ACTION_SCHEMA, image_url=images, cache=False,
            retry_safe=False)
        # The generated schema alone cannot enforce calibrated action counts
        # or binding to the actual sensor frame.
        if result.get('evidence_id') != current['observation_id']:
            raise APIError('local_evidence_id_mismatch',
                           category='structured_field')
        r.event('local_visual_context', source='local_agent',
                route_id=router.route['route_id'],
                frame_ids=[f['observation_id'] for f in manifest],
                image_count=len(images), depth_image_index=depth_index,
                local_mode=router.state, model_tier='local')
        return result, preflight
