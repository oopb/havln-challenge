"""Cloud-owned route delegation with bounded, evidence-based local execution.

The router never chooses a navigation bearing, verifies semantic completion or
turns historical sensor observations into collision clearance. Only a cloud
NavigationAgent decision can delegate a single currently selected route.
"""
from copy import deepcopy

LOCAL_STATES = frozenset(('LOCAL_FOLLOW', 'LOCAL_LOOK'))
SIMPLE_ACTIONS = frozenset(('FORWARD', 'LEFT', 'RIGHT', 'LOOK_UP', 'LOOK_DOWN'))


class HybridRouter:
    def __init__(self, settings=None):
        settings = settings or {}
        self.enabled = settings.get('enabled', False) is True
        self.state = 'CLOUD_DECIDE'
        self.route = None
        self.last_escalation = None
        self.local_decisions = 0
        self.cloud_decisions = 0
        self.escalations = 0
        self.turns_since_forward = 0
        self.max_turns_without_forward = settings.get('max_turns_without_forward', 4)

    def use_local(self):
        return self.enabled and self.route is not None and self.state in LOCAL_STATES

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
        grant = proposal.get('local_handoff')
        if (not isinstance(grant, dict) or
                proposal.get('tool') != 'navigate_relative' or
                receipt.get('status') != 'ACTIONS_COMPLETED' or
                not any(a == 'MOVE_FORWARD'
                        for a in receipt.get('executed_motion', ()))):
            return
        goal = grant.get('goal')
        until = grant.get('exit_condition')
        if not (isinstance(goal, str) and goal.strip() and
                isinstance(until, str) and until.strip()):
            return
        current_id = (runtime.memory.current or {}).get('observation_id')
        if not current_id:
            return
        self.route = dict(
            route_id=str(runtime.current_decision_id),
            source_observation_id=source_observation_id,
            start_observation_id=current_id,
            goal=goal.strip()[:200],
            exit_condition=until.strip()[:200])
        self.state = 'LOCAL_FOLLOW'
        runtime.event('hybrid_route_delegated', source='hybrid_router',
                      route=deepcopy(self.route), model_tier='cloud')

    def approve_local(self, decision, current_id, front_state):
        """Return a reason to escalate, or None for an eligible local action."""
        if not self.use_local():
            return 'no_active_cloud_route'
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
        if 'MOVE_FORWARD' in actions:
            self.turns_since_forward = 0
            # An executed forward receipt may still have unchanged RGB-D.
            attempts = [a for a in runtime.memory.actions[-len(actions):]
                        if a.get('action') == 'MOVE_FORWARD']
            if any(not (a.get('result') or {}).get('visual_change')
                   for a in attempts):
                return self.escalate('local_forward_no_rgbd_change', runtime)
            self.state = 'LOCAL_FOLLOW'
        else:
            self.turns_since_forward += sum(a in ('TURN_LEFT', 'TURN_RIGHT',
                                                   'LOOK_UP', 'LOOK_DOWN')
                                            for a in actions)
            self.state = 'LOCAL_LOOK'
            if self.turns_since_forward >= self.max_turns_without_forward:
                return self.escalate('local_observation_without_forward', runtime)
        return None

    def escalate(self, reason, runtime):
        if not self.enabled:
            return reason
        previous = self.state
        route = deepcopy(self.route)
        self.route = None
        self.state = 'ESCALATE'
        self.last_escalation = reason
        self.escalations += 1
        runtime.event('hybrid_escalated', source='hybrid_router',
                      from_state=previous, reason=reason,
                      route=route, model_tier='cloud_next')
        return reason

    def cloud_ready(self):
        """Escalation is processed by the next ordinary cloud navigation turn."""
        if self.enabled and self.state == 'ESCALATE':
            self.state = 'CLOUD_DECIDE'

    def stats(self):
        return dict(enabled=self.enabled, state=self.state,
                    active_route=deepcopy(self.route),
                    cloud_decisions=self.cloud_decisions,
                    local_decisions=self.local_decisions,
                    escalations=self.escalations,
                    last_escalation=self.last_escalation)
