"""Cloud-owned route delegation with bounded, evidence-based local execution.

The router never chooses a navigation bearing, verifies semantic completion or
turns historical sensor observations into collision clearance. Only a cloud
NavigationAgent decision can delegate a single currently selected route.
"""
from copy import deepcopy
from math import degrees, isfinite
from .heading import relative_bearing

LOCAL_STATES = frozenset(('LOCAL_FOLLOW', 'LOCAL_LOOK'))
SIMPLE_ACTIONS = frozenset(('FORWARD', 'LEFT', 'RIGHT', 'LOOK_UP', 'LOOK_DOWN'))


def hybrid_route_review(projected_context):
    """Flag source-linked historical motion that conflicts with current view.

    This is a READ-ONLY model attention cue, not a navigation veto, inferred
    physical retrace, new state machine, or permission to turn. It operates on
    the already camera-relative model projection so no world yaw leaks out.
    """
    memory = projected_context.get('memory') or {}
    continuity = memory.get('route_continuity') or {}
    previous = continuity.get('selected_historical_forward') or {}
    separation = continuity.get('separation_from_historical_forward_deg')
    if (type(separation) not in (int, float) or
            not 105 <= separation <= 180 or
            not isinstance(previous, dict) or
            not all(isinstance(previous.get(key), str) and previous[key]
                    for key in ('start_observation_id', 'end_observation_id'))):
        return None
    # Real atomic forward receipts, not a model's narrative of movement.
    executed = previous.get('executed_forward_with_RGBD_change')
    if type(executed) is not int or executed < 1:
        return None
    comparison = dict(
        historical_start_observation_id=previous['start_observation_id'],
        historical_end_observation_id=previous['end_observation_id'],
        prior_forward_actions_with_RGBD_change=executed,
        heading_difference_deg=round(separation, 1),
        opposing_view_does_not_prove_retrace=True,
        instruction_turn_reference='If the task says turn after descending, '
            'interpret LEFT/RIGHT relative to actual travel direction at '
            'the stair exit, not the yaw of a later camera scan.',
        review='Compare CURRENT RGB with the source-linked forward start/end '
            'images and actual motion receipts before claiming continued '
            'descent, confirming the stair bottom, or handing off that route.')
    offset = previous.get('view_offset_deg_from_current')
    if type(offset) in (int, float) and -180 <= offset <= 180:
        comparison['historical_forward_view_offset_deg_from_current'] = offset
    return comparison


class HybridRouter:
    def __init__(self, settings=None):
        settings = settings or {}
        self.enabled = settings.get('enabled', False) is True
        self.state = 'CLOUD_DECIDE'
        self.route = None
        self.last_escalation = None
        self.last_route = None
        self.last_local_proposal = None
        self.local_forward_actions = 0
        self.last_local_summary = None
        self.last_forward_heading_rad = None
        self.last_return_action_step = None
        self.local_decisions = 0
        self.cloud_decisions = 0
        self.escalations = 0
        self.turns_since_forward = 0
        self.max_turns_without_forward = settings.get('max_turns_without_forward', 4)

    def use_local(self):
        return self.enabled and self.route is not None and self.state in LOCAL_STATES

    def remaining_forward_actions(self):
        """Remaining executed atomic forwards in this explicit cloud lease."""
        if not self.use_local():
            return 0
        return max(0, self.route['max_forward_actions'] -
                   self.local_forward_actions)

    def after_cloud(self, proposal, receipt, runtime, source_observation_id):
        """Authorize a new local segment only after a successful CLOUD motion.

        Do not interpret text claims of arrival as real sensor measurements.
        A fresh route anchors to the executed cloud decision and its resulting
        observation; old local grants never survive a cloud replanning turn.
        """
        if not self.enabled:
            return
        self.cloud_decisions += 1
        self.state = 'CLOUD_DECIDE'
        self.route = None
        self.turns_since_forward = 0
        # A new real forward supersedes a returned handoff anchor.
        # A cloud-only TURN or LOOK must not erase its travel reference.
        if 'MOVE_FORWARD' in (receipt.get('executed_motion') or ()):
            self.last_route = None
            self.last_escalation = None
            self.last_local_proposal = None
            self.last_local_summary = None
            self.last_forward_heading_rad = None
            self.last_return_action_step = None
        grant = proposal.get('local_handoff')
        if (not isinstance(grant, dict) or
                proposal.get('tool') != 'navigate_relative' or
                receipt.get('status') != 'ACTIONS_COMPLETED' or
                not any(a == 'MOVE_FORWARD'
                        for a in receipt.get('executed_motion', ()))):
            return
        goal = grant.get('goal')
        until = grant.get('exit_condition')
        limit = grant.get('max_forward_actions')
        if not (isinstance(goal, str) and goal.strip() and
                isinstance(until, str) and until.strip() and
                type(limit) is int and 1 <= limit <= 12):
            return
        current_id = (runtime.memory.current or {}).get('observation_id')
        if not current_id:
            return
        self.route = dict(
            route_id=str(runtime.current_decision_id),
            source_observation_id=source_observation_id,
            start_observation_id=current_id,
            # Exclude the cloud's initial movement from LOCAL execution feedback.
            start_action_step=len(runtime.memory.actions),
            goal=goal.strip()[:200],
            exit_condition=until.strip()[:200],
            max_forward_actions=limit)
        self.local_forward_actions = 0
        self.state = 'LOCAL_FOLLOW'
        runtime.event('hybrid_route_delegated', source='hybrid_router',
                      route=deepcopy(self.route), model_tier='cloud')

    def approve_local(self, decision, current_id, front_state):
        """Return a reason to escalate, or None for an eligible local action."""
        if not self.use_local():
            return 'no_active_cloud_route'
        if self.remaining_forward_actions() == 0:
            return 'delegated_forward_limit_reached'
        if decision.get('evidence_id') != current_id:
            return 'local_evidence_id_mismatch'
        action, count, signal = (decision.get('action'), decision.get('count'),
                                  decision.get('signal'))
        if signal != 'NORMAL':
            return 'local_signal_' + str(signal).lower()
        if action == 'ESCALATE':
            return 'local_requested_escalation'
        if action not in SIMPLE_ACTIONS or type(count) is not int:
            return 'invalid_local_command'
        if action == 'FORWARD':
            if not 1 <= count <= 3:
                return 'invalid_local_forward_count'
            if count > self.remaining_forward_actions():
                # Never silently shorten a model-selected command.
                return 'delegated_forward_limit_would_be_exceeded'
            if front_state == 'BLOCKED':
                return 'live_front_depth_blocked'
        elif action in ('LEFT', 'RIGHT'):
            if not 1 <= count <= 2:
                return 'invalid_local_turn_count'
            if self.turns_since_forward + count > self.max_turns_without_forward:
                return 'repeated_turn_only_local'
        elif count != 1:
            return 'invalid_local_look_count'
        if (action in ('LOOK_UP', 'LOOK_DOWN') and
                self.turns_since_forward + count > self.max_turns_without_forward):
            return 'repeated_look_only_local'
        return None

    def after_local(self, decision, receipt, runtime):
        """One short local step, not a metric arrival or proof of progress."""
        self.local_decisions += 1
        status = receipt.get('status')
        if status not in ('ACTIONS_COMPLETED', 'EXECUTED', 'OK'):
            return self.escalate('local_action_' + str(status).lower(), runtime)
        if not receipt.get('executed_actions'):
            return self.escalate('local_no_action_executed', runtime)
        actions = receipt.get('executed_motion') or []
        self.local_forward_actions += sum(
            action == 'MOVE_FORWARD' for action in actions)
        if 'MOVE_FORWARD' in actions:
            self.turns_since_forward = 0
            # An executed forward receipt may still have unchanged RGB-D.
            attempts = [a for a in runtime.memory.actions[-len(actions):]
                        if a.get('action') == 'MOVE_FORWARD']
            if any(not (a.get('result') or {}).get('visual_change')
                   for a in attempts):
                return self.escalate('local_forward_no_rgbd_change', runtime)
            self.state = 'LOCAL_FOLLOW'
            if self.remaining_forward_actions() == 0:
                return self.escalate('delegated_forward_limit_reached', runtime)
        else:
            self.turns_since_forward += sum(a in ('TURN_LEFT', 'TURN_RIGHT',
                                                   'LOOK_UP', 'LOOK_DOWN')
                                            for a in actions)
            self.state = 'LOCAL_LOOK'
            if self.turns_since_forward >= self.max_turns_without_forward:
                return self.escalate('local_observation_without_forward', runtime)
        return None

    @staticmethod
    def _execution_summary(runtime, route):
        """Snapshot actual local receipts, excluding future cloud turns."""
        if not isinstance(route, dict):
            return None, None
        start=route.get('start_action_step')
        actions=getattr(getattr(runtime,'memory',None),'actions',None)
        if type(start) is not int or not isinstance(actions,list) or not (
                0 <= start <= len(actions)):
            return None, None
        segment=actions[start:]
        count=lambda kind:sum(a.get('action')==kind for a in segment)
        forward=[a for a in segment if a.get('action')=='MOVE_FORWARD']
        current=(getattr(runtime.memory,'current',None) or {})
        heading=None
        for action in reversed(forward):
            angle=(action.get('pose') or {}).get('heading')
            if type(angle) in (int,float) and isfinite(angle):
                heading=float(angle)
                break
        summary=dict(
            start_observation_id=route.get('start_observation_id'),
            latest_observation_id=current.get('observation_id'),
            executed_forward=count('MOVE_FORWARD'),
            executed_left=count('TURN_LEFT'),
            executed_right=count('TURN_RIGHT'),
            executed_look_up=count('LOOK_UP'),
            executed_look_down=count('LOOK_DOWN'),
            forward_receipts_with_RGBD_change=sum(
                (item.get('result') or {}).get('visual_change') is True
                for item in forward),
            handoff_forward_limit=route.get('max_forward_actions'),
            action_receipts_only_no_verified_displacement=True)
        return summary,heading

    def escalate(self, reason, runtime):
        if not self.enabled:
            return reason
        previous=self.state
        route=deepcopy(self.route)
        self.last_local_summary,self.last_forward_heading_rad=(
            self._execution_summary(runtime,route))
        self.last_return_action_step=len(runtime.memory.actions)
        self.route=None
        self.state='ESCALATE'
        self.last_route=route
        self.last_escalation=reason
        self.escalations+=1
        runtime.event('hybrid_escalated',source='hybrid_router',
            from_state=previous,reason=reason,route=route,
            model_tier='cloud_next',
            executed_forward_actions=self.local_forward_actions)
        return reason

    def cloud_ready(self):
        """Escalation is processed by the next ordinary cloud navigation turn."""
        if self.enabled and self.state == 'ESCALATE':
            self.state = 'CLOUD_DECIDE'

    def finish(self):
        if self.enabled:
            self.state = 'FINISHED'
            self.route = None

    def cloud_feedback(self, runtime=None):
        """Read-only handoff evidence retained through cloud-only scans.

        Preserve last executed forward orientation as a RELATIVE historical
        view offset, never an absolute heading or a movement command.
        """
        if not self.last_escalation:
            return None
        current=(getattr(getattr(runtime,'memory',None),'current',None) or {})
        result=dict(
            escalation_reason=self.last_escalation,
            prior_delegated_route=deepcopy(self.last_route),
            last_local_proposal=deepcopy(self.last_local_proposal),
            local_execution_summary=deepcopy(self.last_local_summary),
            current_observation_id=current.get('observation_id'),
            local_proposal_is_unverified=True,
            selected_route_requires_cloud_reassessment=True)
        if runtime is not None:
            actions=getattr(runtime.memory,'actions',None)
            start=self.last_return_action_step
            if isinstance(actions,list) and type(start) is int and (
                    0 <= start <= len(actions)):
                newer=actions[start:]
                result['cloud_actions_since_local_return']=dict(
                    forward=sum(a.get('action')=='MOVE_FORWARD' for a in newer),
                    turns=sum(a.get('action') in ('TURN_LEFT','TURN_RIGHT')
                              for a in newer),
                    looks=sum(a.get('action') in ('LOOK_UP','LOOK_DOWN')
                              for a in newer))
            angle=(current.get('pose') or {}).get('heading')
            if angle is None:
                controller=getattr(runtime,'motion_controller',None)
                if controller is not None:
                    angle=controller.pose().get('heading')
            if (type(angle) in (int,float) and isfinite(angle)
                    and self.last_forward_heading_rad is not None):
                result['last_local_forward_view_offset_deg_from_current']=(
                    relative_bearing(degrees(self.last_forward_heading_rad),
                                     degrees(angle)))
                result['last_local_forward_is_historical_view_not_motor_target']=True
        return result

    def stats(self):
        return dict(enabled=self.enabled, state=self.state,
                    active_route=deepcopy(self.route),
                    cloud_decisions=self.cloud_decisions,
                    local_decisions=self.local_decisions,
                    escalations=self.escalations,
                    last_escalation=self.last_escalation)
