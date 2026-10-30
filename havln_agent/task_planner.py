"""Low-frequency semantic planning; completion predicates are checked locally."""
import re

from .api_client import APIError, TEXT, object_schema
from .contracts import CONDITIONS, TASK_REVIEW
from .contexts import planner_context
from .progress import turn_requirement, requires_landmark_crossing

PLAN_SCHEMA=object_schema(dict(subgoals=dict(type='array',minItems=1,maxItems=30,items=
    object_schema(dict(action=TEXT,landmark=TEXT,target=TEXT,success_criteria=CONDITIONS,
        source_sentence_ids=dict(type='array',minItems=1,uniqueItems=True,items=dict(type='integer',minimum=1))))),
    final_target=dict(type='string',minLength=1,maxLength=1000),
    constraints=dict(type='array',maxItems=30,items=TEXT)))


def required_conditions(goal, final=False):
    action=goal['action'].lower()
    if turn_requirement(goal):
        required={'orientation_completed'}
    elif re.search(r'descend|下楼',action):
        required={'entry_confirmed','actual_descent_observed','target_nearby'}
    elif re.search(r'ascend|上楼',action):
        required={'entry_confirmed','actual_ascent_observed','target_nearby'}
    elif re.match(r'stop|wait|停',action):
        required={'target_nearby'}
    else:
        required={'landmark_visible','registered_translation','target_nearby'}
        if requires_landmark_crossing(goal):
            required.add('landmark_passed')
    if final:required.add('final_arrival_visible')
    return required


def validate_conditions(goal, criteria, final=False):
    missing=required_conditions(goal,final)-set(criteria)
    if missing:
        error=APIError('missing_required_success_criteria for action='+repr(goal['action'][:160])+': '
            +', '.join(sorted(missing)),category='structured_field')
        error.validation_field=['success_criteria']
        raise error


class TaskPlanner:
    def __init__(self,runtime):
        self.runtime=runtime

    @staticmethod
    def parse(client,instruction,format_feedback=None):
        if not isinstance(instruction,str) or not instruction.strip():
            raise APIError('missing_instruction_text',category='structured_field')
        sentences=[p.strip() for p in re.split(r'(?<=[.!?])\s+|(?<=[。！？])',instruction) if p.strip()]
        def validate(plan):
            covered=set()
            for index,goal in enumerate(plan['subgoals']):
                ids=goal['source_sentence_ids']
                if ids!=sorted(set(ids)) or ids[-1]>len(sentences):
                    raise APIError('invalid_instruction_source_ids',category='structured_field')
                covered.update(ids)
                try:
                    validate_conditions(goal,goal['success_criteria'],index==len(plan['subgoals'])-1)
                except APIError as error:
                    error.validation_field=['subgoals',index,'success_criteria']
                    raise
            if covered!=set(range(1,len(sentences)+1)):
                raise APIError('instruction_source_coverage_incomplete',category='structured_field')
        plan=client.complete('task_planner','task_initial_plan',
            'You are Task Planner. Decompose the instruction into ordered semantic navigation milestones. '
            'Separate an in-place language turn from translation. Preserve human activities as landmark '
            'attributes. Merge descriptive sentences with their navigation milestone; never create an '
            'abstract recognize/note task merely to cover prose. Cite every numbered source sentence '
            'across the plan. Return only semantic outcomes and typed success_criteria, never atomic '
            'action commands, coordinates or waypoints. A plan is intent and never proves completion. '
            'Use entry_confirmed+actual_descent_observed/actual_ascent_observed+target_nearby for stairs; '
            'orientation_completed for a pure turn/orientation change; landmark_visible+registered_translation+'
            'target_nearby for ordinary movement. Add landmark_passed only for pass/through/enter; '
            'proceed TO a destination does not mean passing it. '
            'The final milestone also needs final_arrival_visible. Return exactly one JSON object.',
            dict(instruction=instruction,source_sentences=[dict(id=i+1,text=s) for i,s in enumerate(sentences)],
                **({'format_feedback':format_feedback} if format_feedback else {})),
            PLAN_SCHEMA,validate=validate,cache=True,format_retry=int(bool(format_feedback)),retry_safe=True)
        return dict(plan,subgoals=[dict(g,source_text=' '.join(sentences[i-1] for i in g['source_sentence_ids']))
            for g in plan['subgoals']])

    def review(self,format_feedback=None):
        r=self.runtime
        task=r.memory.task
        index=task.current_subgoal_index
        goal=task.subgoals[index]
        def validate(value):
            validate_conditions(goal,value['success_criteria'],index==len(task.subgoals)-1)
        return r.client.complete('task_planner','task_review',
            'You are Task Planner. Update only the current semantic outcome and its typed completion '
            'conditions. Preserve the original ordered instruction and human activity landmarks. '
            'Never choose coordinates, waypoints or atomic actions. Your text cannot verify progress. '
            'Keep required_success_criteria and optionally add predicates that require more evidence. '
            'Return one intent/subgoal/success_criteria JSON object.',
            dict(planner_context(r),required_success_criteria=sorted(required_conditions(goal,index==len(task.subgoals)-1)),
                **({'format_feedback':format_feedback} if format_feedback else {})),TASK_REVIEW,
            validate=validate,format_retry=int(bool(format_feedback)),retry_safe=True)
