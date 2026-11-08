"""Synchronous mission scheduler: one navigation decision at a time.

No TaskPlanner calls, indexed milestone review or automatic VERIFY loops.
"""
from .api_client import APIError,BudgetExceeded

class EvidenceLivenessGuard:
    """Resource diagnostic only. It never chooses headings or forces STOP."""
    def __init__(self,runtime):
        self.limit=max(2,int(runtime.config.get(
            'max_unchanged_evidence_decisions',8)))
        self.max_decisions=max(1,int(runtime.config.get(
            'max_model_decisions',2*runtime.config['max_steps'])))
        self.reference=self.fingerprint(runtime)
        self.actionless_decisions=0

    @staticmethod
    def fingerprint(runtime):
        m=runtime.memory
        return (len(m.actions),
                (m.current or {}).get('observation_id'))

    def feedback(self,runtime):
        current=self.fingerprint(runtime)
        if current!=self.reference:
            self.reference=current
            self.actionless_decisions=0
        else:
            self.actionless_decisions+=1
        return self.actionless_decisions>=self.limit

    def diagnostic(self,runtime):
        return dict(consecutive_unchanged_decisions=self.actionless_decisions,
            max_unchanged_evidence_decisions=self.limit,
            max_model_decisions=self.max_decisions,
            model_decisions=runtime.decisions,
            observation_id=(runtime.memory.current or {}).get('observation_id'),
            atomic_actions=len(runtime.memory.actions),
            reason='no_new_RGBD_or_atomic_action',
            decision_owner='navigation_agent',
            no_automatic_semantic_review=True)

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
            context=getattr(error,'request_context',
                dict(error=str(error),category=error.category))
            runtime.event('api_error',source=role,error=str(error),
                          category=error.category,request_context=context)
            append(role,dict(event='api_error',request_context=context))
            if attempt or error.category not in (
                'structured_field','json_syntax','json_schema'):
                runtime.api_failure=context
                raise
            feedback=dict(error=str(error),category=error.category,
                detail=context.get('error_detail',
                    'Return a schema-valid tool+args JSON object.')[:400])

def run(runtime,append):
    runtime.start_mission()
    liveness=EvidenceLivenessGuard(runtime)
    while not runtime.done:
        if runtime.decisions>=liveness.max_decisions:
            runtime.navigation_liveness=liveness.diagnostic(runtime)
            runtime.event('navigation_inference_budget_exhausted',
                source='scheduler',diagnostic=runtime.navigation_liveness)
            runtime.terminate('navigation_inference_budget_exhausted')
            break
        if len(runtime.memory.actions)>=runtime.config['max_steps']-1:
            runtime.terminate('action_budget_exhausted')
            break
        if runtime.client.availability()!='available':
            runtime.terminate('api_budget_exhausted')
            break
        try:
            binding={}
            turn_start_step=len(runtime.memory.actions)
            def decide(feedback):
                proposal=runtime.navigation_agent.decide(feedback)
                # NavigationAgent may refresh the RGB-D frame during its
                # pre-decision Depth check. Bind the tool to the ACTUAL
                # image used by the completed model decision, not a stale
                # observation ID sampled before inference.
                binding.update(
                    observation_id=runtime.memory.current['observation_id'],
                    actor='navigation_agent')
                return proposal
            proposal=request(runtime,'navigation_agent',decide,append)
            runtime.decisions+=1
            runtime.event('navigation_decision',
                source='navigation_agent',decision=proposal)
            append('navigation_agent',dict(
                event='navigation_decision',
                decision_id=runtime.current_decision_id,decision=proposal))
            # Commit the model's proposed narrative before executing its
            # delegated Observation task. Observation Agent must see the most
            # recent natural-language progress from THIS decision, too.
            # This updates hypotheses only; it does not write sensor facts.
            memory_update=runtime.memory.note_navigation_assessment(proposal)
            # Include the model's declared target/turn semantics in the tool
            # audit without granting Python any steering authority.
            binding['direction_intent']=proposal.get('direction_intent')
            # Only the registered executable command reaches ToolExecutor.
            command=dict(tool=proposal['tool'],args=proposal['args'])
            result=runtime.execute(command,binding)
            runtime.memory.note_navigation_turn(
                proposal,result,binding.get('observation_id'),
                turn_start_step,runtime.current_decision_id)
            append('navigation_agent',dict(event='tool_feedback',
                tool_call_id=result['tool_call_id'],
                decision_id=runtime.current_decision_id,
                feedback=result,memory_update=memory_update))
            runtime.assess(result)
            if runtime.done:
                continue
            repeated=liveness.feedback(runtime)
            runtime.navigation_liveness=liveness.diagnostic(runtime)
            if repeated:
                runtime.event('navigation_liveness_advisory',
                    source='scheduler',diagnostic=runtime.navigation_liveness,
                    last_tool=result.get('tool'),last_status=result.get('status'))
        except BudgetExceeded:
            runtime.terminate('api_budget_exhausted')
        except APIError as error:
            runtime.api_failure=getattr(error,'request_context',
                dict(error=str(error),category=error.category))
            runtime.terminate('api_error')
