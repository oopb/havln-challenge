"""Low-frequency semantic planning; completion predicates are checked locally."""
import re

from .api_client import APIError, TEXT, object_schema
from .contracts import CONDITIONS, TASK_REVIEW
from .contexts import planner_context
from .progress import turn_requirement, requires_landmark_crossing, stair_motion_direction

# Runtime owns proof requirements; the legacy field is accepted but optional.
GOAL_SCHEMA=object_schema(dict(action=TEXT,landmark=TEXT,target=TEXT,
    source_sentence_ids=dict(type='array',minItems=1,uniqueItems=True,items=dict(type='integer',minimum=1))))
GOAL_SCHEMA['properties']['success_criteria']=CONDITIONS

PLAN_SCHEMA=object_schema(dict(subgoals=dict(type='array',minItems=1,maxItems=30,items=GOAL_SCHEMA),
    final_target=dict(type='string',minLength=1,maxLength=1000),
    constraints=dict(type='array',maxItems=30,items=TEXT)))


def required_conditions(goal, final=False):
    """Runtime-owned evidence contract, never a free-form planner decision."""
    action=goal['action'].lower()
    direction=stair_motion_direction(goal)
    if direction=='DOWN':
        required={'entry_confirmed','actual_descent_observed','target_nearby'}
    elif direction=='UP':
        required={'entry_confirmed','actual_ascent_observed','target_nearby'}
    elif turn_requirement(goal):
        required={'orientation_completed'}
    elif re.match(r'\s*(?:stop\b|wait\b|remain\s+(?:stop|wait)|停|等候)',action):
        required={'target_nearby'}
    else:
        required={'landmark_visible','registered_translation','target_nearby'}
        if requires_landmark_crossing(goal):
            required.add('landmark_passed')
    if final:required.add('final_arrival_visible')
    return required


def normalize_conditions(goal, final=False):
    """Ignore model-supplied proof requirements; derive them from task intent.

    Model values are hypotheses about desired outcomes, not evidence policy.
    In particular, do not let invented extra predicates prevent navigation.
    """
    return sorted(required_conditions(goal, final))


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
                # Success predicates are normalized locally after structural
                # validation; missing model-generated predicates are not errors.
            if covered!=set(range(1,len(sentences)+1)):
                raise APIError('instruction_source_coverage_incomplete',category='structured_field')
        plan=client.complete('task_planner','task_initial_plan',
            'You are Task Planner. Decompose the instruction into ordered semantic navigation milestones. '
            'Separate an in-place language turn from translation. Preserve human activities as landmark '
            'attributes. Merge descriptive sentences with their navigation milestone; never create an '
            'abstract recognize/note task merely to cover prose. Cite every numbered source sentence '
            'across the plan. Return semantic outcomes only; Runtime assigns success_criteria. Never return atomic '
            'action commands, coordinates or waypoints. A plan is intent and never proves completion. '
            'Do not invent completion evidence predicates. Preserve boundary crossing versus arrival: '
            'proceeding TO a destination does not mean passing it. Return exactly one JSON object.',
            dict(instruction=instruction,source_sentences=[dict(id=i+1,text=s) for i,s in enumerate(sentences)],
                **({'format_feedback':format_feedback} if format_feedback else {})),
            PLAN_SCHEMA,validate=validate,cache=True,format_retry=int(bool(format_feedback)),retry_safe=True)
        return dict(plan,subgoals=[dict(g,
            success_criteria=normalize_conditions(g,index==len(plan['subgoals'])-1),
            source_text=' '.join(sentences[i-1] for i in g['source_sentence_ids']))
            for index,g in enumerate(plan['subgoals'])])

    def review(self,format_feedback=None):
        r=self.runtime
        task=r.memory.task
        index=task.current_subgoal_index
        goal=task.subgoals[index]
        value = r.client.complete('task_planner','task_review',
            'You are Task Planner. Update only the current semantic outcome and its typed completion '
            'conditions. Preserve the original ordered instruction and human activity landmarks. '
            'Never choose coordinates, waypoints or atomic actions. Your text cannot verify progress. '
            'Runtime owns evidence predicates. Return one intent/subgoal JSON object.',
            dict(planner_context(r),required_success_criteria=sorted(required_conditions(goal,index==len(task.subgoals)-1)),
                **({'format_feedback':format_feedback} if format_feedback else {})),TASK_REVIEW,
            format_retry=int(bool(format_feedback)),retry_safe=True)
        return dict(value, success_criteria=normalize_conditions(goal,index==len(task.subgoals)-1))
