"""Event schedule: semantic reviews are explicit; exploration never has a stall quota."""
from .api_client import APIError,BudgetExceeded


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
    while not runtime.done:
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
        except BudgetExceeded:
            runtime.terminate('api_budget_exhausted')
        except APIError as error:
            runtime.api_failure=getattr(error,'request_context',dict(error=str(error),category=error.category))
            runtime.terminate('api_error')
