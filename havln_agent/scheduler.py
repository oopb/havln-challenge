"""Synchronous mission scheduler: one navigation decision at a time.

No TaskPlanner calls, indexed milestone review or automatic VERIFY loops.
"""
from .api_client import APIError,BudgetExceeded,canonical

class EvidenceLivenessGuard:
    """Stop evidence-free inference without prescribing a physical action."""
    def __init__(self,runtime):
        self.limit=max(2,int(runtime.config.get(
            'max_unchanged_evidence_decisions',8)))
        self.max_decisions=max(1,int(runtime.config.get(
            'max_model_decisions',2*runtime.config['max_steps'])))
        self.reference=self.fingerprint(runtime)
        self.actionless_decisions=0
        self.last_blocked_command=None
        self.identical_blocked_count=0

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

    def repeated_blocked_command(self,runtime,command,result):
        """Count identical rejected commands at one unchanged sensor stance."""
        if (result.get('status')!='BLOCKED'
                or result.get('reason')!='measured_centerline_obstacle'
                or result.get('executed_actions')!=0):
            self.last_blocked_command=None
            self.identical_blocked_count=0
            return 0
        signature=(self.fingerprint(runtime),canonical(command))
        if signature==self.last_blocked_command:
            self.identical_blocked_count+=1
        else:
            self.last_blocked_command=signature
            self.identical_blocked_count=1
        return self.identical_blocked_count

    def diagnostic(self,runtime):
        return dict(consecutive_unchanged_decisions=self.actionless_decisions,
            max_unchanged_evidence_decisions=self.limit,
            max_model_decisions=self.max_decisions,
            model_decisions=runtime.decisions,
            identical_blocked_decisions=self.identical_blocked_count,
            max_identical_blocked_decisions=3,
            observation_id=(runtime.memory.current or {}).get('observation_id'),
            atomic_actions=len(runtime.memory.actions),
            reason='no_new_RGBD_or_atomic_action',
            decision_owner='navigation_agent',
            no_automatic_semantic_review=True)

FORMAT_ERROR_CATEGORIES=frozenset(
    ('structured_field','json_syntax','json_schema'))
# Each navigation decision has one bounded, exact-role regeneration budget.
# Invalid commands never enter the executor; no outer recovery cycle restarts
# the same decision and multiplies unsuccessful API calls.
FORMAT_REGENERATIONS_PER_DECISION=3


def format_repair_feedback(error,context):
    detail=str(context.get('error_detail') or str(error))[:350]
    field=context.get('validation_field') or []
    if isinstance(field,str):
        field=[field]
    field_text='.'.join(map(str,field))
    pure_turn=('distance_m' in detail or 'distance_m' in field_text)
    return dict(
        category=error.category,
        failed_field=field_text or None,
        error_detail=detail,
        repair_instruction=(
            'No physical action executed. Preserve the intended goal '
            'and observations, then regenerate ONE legal JSON object '
            'with observed_evidence, decision_analysis, tool and args. '
            'Pure rotation: choose turn_and_observe with look=none, '
            'never navigate_relative(distance_m=0). '
            'A navigate_relative call MUST request distance_m 0.25..3m.'
            if pure_turn else
            'No physical action executed. Reconsider the specific invalid '
            'field and regenerate ONE legal evidence-first JSON decision. '
            'Keep observed_evidence entries concise (max 600 characters). '
            'Do not add unregistered fields or change model intent just '
            'to meet a numeric constraint.'))


def request(runtime,role,operation,append):
    feedback=None
    for attempt in range(FORMAT_REGENERATIONS_PER_DECISION):
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
            if (attempt+1>=FORMAT_REGENERATIONS_PER_DECISION
                    or error.category not in FORMAT_ERROR_CATEGORIES):
                runtime.api_failure=context
                raise
            feedback=format_repair_feedback(error,context)
            runtime.event('navigation_format_regeneration',
                source=role,attempt=attempt+1,
                failed_field=feedback.get('failed_field'),
                action_executed=False)

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
            unchanged=liveness.feedback(runtime)
            identical=liveness.repeated_blocked_command(runtime,command,result)
            runtime.navigation_liveness=liveness.diagnostic(runtime)
            if identical>=2:
                runtime.event('navigation_repeated_blocked_command',
                    source='scheduler',diagnostic=runtime.navigation_liveness,
                    action_executed=False,command=command)
            if unchanged or identical>=3:
                runtime.event('navigation_liveness_exhausted',
                    source='scheduler',diagnostic=runtime.navigation_liveness,
                    last_tool=result.get('tool'),last_status=result.get('status'))
                runtime.terminate('navigation_no_new_evidence')
        except BudgetExceeded:
            runtime.terminate('api_budget_exhausted')
        except APIError as error:
            runtime.api_failure=getattr(error,'request_context',
                dict(error=str(error),category=error.category))
            # request() owns the only bounded format retry budget.
            runtime.terminate('api_error')
