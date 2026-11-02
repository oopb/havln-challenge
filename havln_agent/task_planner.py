"""Low-frequency task decomposition: semantic intent is not a proof policy."""
import re

from .api_client import APIError, TEXT, object_schema
from .contracts import TASK_REVIEW
from .contexts import planner_context

GOAL_SCHEMA = object_schema(dict(
    action=TEXT, landmark=TEXT, target=TEXT,
    source_sentence_ids=dict(type='array', minItems=1, uniqueItems=True,
                             items=dict(type='integer', minimum=1))))
PLAN_SCHEMA = object_schema(dict(
    subgoals=dict(type='array', minItems=1, maxItems=30, items=GOAL_SCHEMA),
    final_target=TEXT,
    constraints=dict(type='array', maxItems=30, items=TEXT)))


class TaskPlanner:
    def __init__(self, runtime):
        self.runtime = runtime

    @staticmethod
    def parse(client, instruction, format_feedback=None):
        if not isinstance(instruction, str) or not instruction.strip():
            raise APIError('missing_instruction_text', category='structured_field')
        sentences = [p.strip() for p in re.split(
            r'(?<=[.!?])\s+|(?<=[。！？])', instruction) if p.strip()]

        def validate(plan):
            covered = set()
            for goal in plan['subgoals']:
                ids = goal['source_sentence_ids']
                if ids != sorted(set(ids)) or ids[-1] > len(sentences):
                    raise APIError('invalid_instruction_source_ids',
                                   category='structured_field')
                covered.update(ids)
            missing = sorted(set(range(1, len(sentences)+1))-covered)
            if missing and not format_feedback:
                # The first failure should be precise enough for the planner
                # to repair its own source mapping on the normal API retry.
                raise APIError(
                    'instruction_source_coverage_incomplete; missing_source_ids='
                    +str(missing)+'; missing_source_sentences='
                    +repr([sentences[i-1][:180] for i in missing]),
                    category='structured_field')
            if missing:
                # If the second model attempt still omits purely narrative
                # sentences, keep ALL original instructions in the evidence
                # chain instead of losing the entire physical episode.
                # This changes only source attribution, NEVER the model's
                # milestone actions, their ordering or completion status.
                # Downstream agents see an explicit repair marker.
                goals = plan['subgoals']
                for i in missing:
                    closest=min(range(len(goals)), key=lambda j: (
                        min(abs(i-n) for n in goals[j]['source_sentence_ids']),
                        j))
                    goals[closest]['source_sentence_ids']=sorted(
                        set(goals[closest]['source_sentence_ids'])|{i})
                plan['source_coverage_repaired_ids']=missing

        plan = client.complete(
            'task_planner', 'task_initial_plan',
            'Decompose the instruction into ordered semantic navigation milestones. '
            'Separate in-place turns from translation, retain location qualifiers '
            'and human activities in the correct milestone. Cite every numbered '
            'source sentence, including descriptive and contextual sentences. '
            'An omitted source sentence prevents using the first plan. '
            'Return semantic goals ONLY, not success predicates, '
            'atomic actions, coordinates or waypoints. The read-only verifier '
            'decides semantic completion from actual cited evidence. Do not invent '
            'a predicate checklist or treat the plan as completion proof. '
            'Distinguish approaching a target from passing or entering it. '
            'Return exactly one JSON object.',
            dict(instruction=instruction, source_sentences=[
                dict(id=i+1, text=s) for i,s in enumerate(sentences)],
                **({'format_feedback':format_feedback} if format_feedback else {})),
            PLAN_SCHEMA, validate=validate, cache=True,
            format_retry=int(bool(format_feedback)), retry_safe=True)
        return dict(plan, subgoals=[
            dict(g, source_text=' '.join(sentences[i-1] for i in g['source_sentence_ids']))
            for g in plan['subgoals']])

    def review(self, format_feedback=None):
        r = self.runtime
        task = r.memory.task
        goal = task.subgoals[task.current_subgoal_index]
        return r.client.complete(
            'task_planner', 'task_review',
            'Review the CURRENT semantic milestone using new observations and '
            'verified history. intent=continue keeps the same milestone; '
            'intent=revise improves its short semantic description, without '
            'changing the ordered instruction or inventing a new action. '
            'Keep location qualifiers such as "at the bottom" and distinguish '
            'approach from crossing. Do not generate success criteria or a '
            'task-specific verification predicate: independent model-led '
            'verification owns completion. Return exactly one JSON object.',
            dict(planner_context(r), current_goal=dict(
                action=goal['action'], target=goal['target'],
                landmark=goal['landmark'], source_text=goal.get('source_text')),
                **({'format_feedback':format_feedback} if format_feedback else {})),
            TASK_REVIEW, format_retry=int(bool(format_feedback)), retry_safe=True)
