"""Event schedule: semantic reviews are explicit; exploration never has a stall quota."""
from .api_client import APIError,BudgetExceeded


class EvidenceLivenessGuard:
    """Prevent unbounded paid inference with no changed robot/evidence state.

    This is NOT a quota on turning, exploration, or safe forward steps.
    Every executed atomic action, changed RGB-D observation or verified
    subgoal transition resets the guard. A Task Planner text revision alone
    cannot masquerade as new physical evidence.
    """

    def __init__(self, runtime):
        self.limit = max(2,int(runtime.config.get('max_unchanged_evidence_decisions',8)))
        self.max_decisions = max(1,int(runtime.config.get(
            'max_model_decisions',2*runtime.config['max_steps'])))
        self.reference = self.fingerprint(runtime)
        self.actionless_decisions = 0
        self.review_requested = False

    @staticmethod
    def fingerprint(runtime):
        memory=runtime.memory
        task=memory.task
        return (len(memory.actions),
            (memory.current or {}).get('observation_id'),
            task.current_subgoal_index,
            tuple(str(item) for item in task.completed_subgoals))

    def feedback(self, runtime):
        current = self.fingerprint(runtime)
        if current != self.reference:
            self.reference = current
            self.actionless_decisions = 0
            self.review_requested = False
            return None
        self.actionless_decisions += 1
        if self.actionless_decisions >= 2*self.limit and self.review_requested:
            return 'NO_EVIDENCE_PROGRESS'
        if self.actionless_decisions >= self.limit and not self.review_requested:
            self.review_requested = True
            return 'REVIEW_NO_PROGRESS'
        return None

    def diagnostic(self, runtime):
        return dict(consecutive_unchanged_decisions=self.actionless_decisions,
            max_unchanged_evidence_decisions=self.limit,
            max_model_decisions=self.max_decisions,
            model_decisions=runtime.decisions,
            review_requested=self.review_requested,
            observation_id=(runtime.memory.current or {}).get('observation_id'),
            atomic_actions=len(runtime.memory.actions),
            reason='no_new_rgbd_or_atomic_action_or_verified_subgoal')


def request(runtime,role,operation,append):
    feedback=None
    for attempt in range(2):
        if runtime.memory.current is not None:runtime.refresh()
        runtime.decision_id(role)
        try:
            return operation(feedback)
        except BudgetExceeded:
            raise
        except APIError as error:
            context=getattr(error,'request_context',dict(error=str(error),category=error.category))
            runtime.event('api_error',source=role,error=str(error),category=error.category,request_context=context)
            append(role,dict(event='api_error',request_context=context))
            if attempt or error.category not in ('structured_field','json_syntax','json_schema'):
                runtime.api_failure=context
                raise
            # A completed malformed response may be followed by ONE new decision.
            # It gets fresh state and a fresh request deadline, without replaying
            # the malformed action list or changing any navigation facts.
            feedback=dict(error=str(error),category=error.category,
                detail=context.get('error_detail','Return exactly one schema-valid JSON object.')[:400])


def run(runtime,append):
    runtime.set_semantic_goal()
    liveness=EvidenceLivenessGuard(runtime)
    while not runtime.done:
        if runtime.decisions >= liveness.max_decisions:
            diagnostic=liveness.diagnostic(runtime)
            runtime.navigation_liveness=diagnostic
            runtime.event('navigation_inference_budget_exhausted',
                source='scheduler',diagnostic=diagnostic)
            runtime.terminate('navigation_inference_budget_exhausted')
            break
        if len(runtime.memory.actions)>=runtime.config['max_steps']-1:
            runtime.terminate('action_budget_exhausted')
            break
        if runtime.client.availability()!='available':
            runtime.terminate('api_budget_exhausted')
            break
        try:
            if runtime.task_review_requested:
                value=request(runtime,'task_planner',runtime.task_planner.review,append)
                runtime.decisions+=1
                runtime.set_semantic_goal(value)
                append('task_planner',dict(event='task_review',decision=value))
            binding={}
            def decide(feedback):
                binding.update(plan_revision=runtime.plan_revision,subgoal_id=runtime.active_subgoal_id,
                    observation_id=runtime.memory.current['observation_id'],actor='navigation_agent')
                return runtime.navigation_agent.decide(feedback)
            proposal=request(runtime,'navigation_agent',decide,append)
            runtime.decisions+=1
            runtime.event('navigation_decision',source='navigation_agent',decision=proposal)
            append('navigation_agent',dict(event='navigation_decision',decision_id=runtime.current_decision_id,
                subgoal_id=runtime.active_subgoal_id,decision=proposal))
            result=runtime.execute(dict(tool=proposal['tool'],args=proposal['args']),binding)
            for note in proposal.get('memory_delta',[]):runtime.memory.write_hypothesis(**note)
            append('navigation_agent',dict(event='tool_feedback',tool_call_id=result['tool_call_id'],
                decision_id=runtime.current_decision_id,subgoal_id=binding['subgoal_id'],feedback=result))
            runtime.assess(result)
            if runtime.done:
                continue
            outcome=liveness.feedback(runtime)
            runtime.navigation_liveness=liveness.diagnostic(runtime)
            if outcome=='REVIEW_NO_PROGRESS':
                # Offer the high-level planner ONE evidence-linked opportunity
                # to revise the route. Do not reset merely because it rewrites
                # the same semantic goal in different words.
                runtime.task_review_requested=True
                runtime.review_reason='repeated_decisions_without_new_sensor_or_action_evidence'
                runtime.event('navigation_liveness_review',source='scheduler',
                    diagnostic=runtime.navigation_liveness,
                    last_tool=result.get('tool'),last_status=result.get('status'))
            elif outcome=='NO_EVIDENCE_PROGRESS':
                runtime.event('navigation_liveness_exhausted',source='scheduler',
                    diagnostic=runtime.navigation_liveness,
                    last_tool=result.get('tool'),last_status=result.get('status'))
                # Explicit failed navigation, not a claimed goal completion or
                # an unsafe fallback MOVE_FORWARD.
                runtime.terminate('navigation_no_new_evidence')
        except BudgetExceeded:
            runtime.terminate('api_budget_exhausted')
        except APIError as error:
            runtime.api_failure=getattr(error,'request_context',dict(error=str(error),category=error.category))
            runtime.terminate('api_error')
