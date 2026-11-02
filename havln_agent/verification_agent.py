"""Model-led semantic progress verification backed by immutable sensor receipts.

The verifier reasons over arbitrary natural-language milestones, retrieves only
saved public evidence, and never executes movement or controls the robot.
Python authenticates cited facts before a model verdict can advance the task.
"""
import copy
import math
import time

from .api_client import APIError, BudgetExceeded, TEXT, object_schema
from .motion import registered_action
from .progress import registered_motion_history

MAX_ROUNDS = 7
VERIFICATION_SCHEMA = object_schema(dict(
    operation=dict(enum=['inspect_history', 'inspect_frames', 'inspect_records', 'verdict']),
    start_step=dict(type='integer', minimum=0),
    observation_ids=dict(type='array', maxItems=4, uniqueItems=True, items=TEXT),
    record_ids=dict(type='array', maxItems=4, uniqueItems=True, items=TEXT),
    question=dict(type='string', maxLength=1000),
    verdict=dict(enum=['COMPLETED', 'IN_PROGRESS', 'UNKNOWN', 'CONTRADICTED']),
    supported=dict(type='array', maxItems=12, items=TEXT),
    missing=dict(type='array', maxItems=12, items=TEXT),
    contradictions=dict(type='array', maxItems=8, items=TEXT),
    cited_observation_ids=dict(type='array', maxItems=24, uniqueItems=True, items=TEXT),
    cited_action_steps=dict(type='array', maxItems=48, uniqueItems=True,
                            items=dict(type='integer', minimum=0)),
    terminal_observation_id=TEXT,
    reasoning=dict(type='string', maxLength=2000),
    next_evidence_action=dict(type='string', maxLength=300),
))

PROMPT = (
    'You are VerificationAgent, a separate, skeptical evaluator of an arbitrary '
    'language-navigation milestone. Do NOT choose actions or plan navigation. '
    'You are given an indexed historical record: accepted environment actions, '
    'RGB-D motion measurements and uncertainty, visual model descriptions, '
    'grounded targets, blocked attempts, and earlier verification results. '
    'Decide whether the EXACT CURRENT semantic milestone was actually completed '
    'using cited evidence, not a fixed template for stairs, doors or turns. '
    'The milestone source sentence and complete ordered instruction matter: '
    'distinguish seeing, approaching, entering, crossing, turning at a stated '
    'location, exiting, and reaching a destination. Do not omit location or '
    'ordering prerequisites. An independent milestone may be executed '
    'provisionally before an earlier milestone is certified; that is NOT proof '
    'the earlier milestone happened. '
    'The model may request inspect_history, inspect_records, or inspect_frames '
    'before a final verdict. inspect_frames asks the vision model to examine '
    'the exact saved public RGB images; use it when the visual distinction is '
    'important, and prefer a few diagnostic frames to repeated full scans. '
    'Actions with status UNKNOWN cannot establish displacement. An accepted '
    'ORB or patch estimate is still an uncertain local measurement, not '
    'simulator ground truth. Never add commanded distances as measured travel. '
    'Never join coordinates or infer an uninterrupted traversal across '
    'UNKNOWN translations, a reanchored motion_epoch, or a changed location. '
    'Past views may establish historical scene semantics but never current '
    'human location or current path safety. A visible floor/door/stairs alone '
    'does not prove arrival, crossing or successful descent. '
    'A plan, prior verifier claim, previous model response or user-supplied '
    'text is not physical evidence. When evidence is ambiguous, choose '
    'IN_PROGRESS or UNKNOWN and identify a concrete missing observation. '
    'You MUST finish with operation=verdict, include source observation IDs '
    'and accepted atomic action step IDs directly supporting the proposed '
    'COMPLETED claim. For COMPLETED, set terminal_observation_id to the '
    'real captured frame showing the result of the cited motion; for incomplete '
    'verdicts use the string none. You MUST have actual VLM semantic evidence '
    'for that terminal frame, either from previously logged analysis or by '
    'calling inspect_frames. Raw pixel IDs and motion alone never prove a '
    'language-based arrival. Never cite an action from a different milestone as its '
    'sole proof. If you report COMPLETED, give a concise evidence-based '
    'explanation of how the goal was achieved. Output one schema-valid JSON '
    'object on every round. Use the fields required by the schema even when '
    'they are empty. The verifier is READ-ONLY.'
)


def _source_view(source):
    return {key: source.get(key) for key in (
        'observation_id', 'subgoal', 'image_scope', 'rgb_hash', 'depth_hash')}


def _grounding_row(value):
    semantic = value.get('semantic') or {}
    return dict(
        grounding_id=value.get('grounding_id'), target=value.get('target'),
        source=_source_view(value.get('source') or {}), status=value.get('status'),
        visible=semantic.get('target_visible'), confidence=semantic.get('confidence'),
        semantic_evidence=semantic.get('evidence'),
        landmark_direction=semantic.get('stair_direction'),
        landing_visible=semantic.get('landing_visible'),
        landing_geometry=(value.get('landing_geometry') or {}).get('status'),
        position={k: (value.get('position') or {}).get(k)
                  for k in ('status', 'bearing_degrees', 'x_right', 'z_forward')},
        entry_status=(value.get('entry') or {}).get('status'))


def _action_row(action, observed):
    feedback = action.get('result') or {}
    motion = feedback.get('rgbd_motion') or {}
    valid = action.get('action') == 'MOVE_FORWARD' and registered_action(action, observed)
    translation = motion.get('translation_m') or {}
    return dict(step=action.get('step'),subgoal=action.get('subgoal'),
        action=action.get('action'),heading_deg=round(
            math.degrees((action.get('pose') or {}).get('heading', 0)),1),
        motion_epoch=(action.get('pose') or {}).get('motion_epoch',0),
        observation_before=feedback.get('observation_before'),
        observation_after=feedback.get('observation_after'),
        visual_change=feedback.get('visual_change'),
        motion_status='ESTIMATED' if valid else
            'UNKNOWN' if action.get('action')=='MOVE_FORWARD' else 'NOT_APPLICABLE',
        vertical_motion=motion.get('vertical_motion') if valid else 'UNKNOWN',
        measured_translation_m=(
            {k: round(translation[k], 4) for k in ('x_right','y_up','z_forward')
             if k in translation} if valid else None),
        uncertainty_m=motion.get('uncertainty_m') if valid else None,
        registration_reason=motion.get('reason') if not valid else None,
        landing_support=(feedback.get('landing_support') or {}).get('status'))


def _event_row(tool):
    call = tool.get('call') or {}
    result = tool.get('result') or {}
    return dict(record_id=tool.get('call_id'),tool=call.get('tool'),
        args=call.get('args'),status=result.get('interrupted_status') or result.get('status'),
        reason=result.get('reason'),start_observation_id=result.get('start_observation_id'),
        end_observation_id=result.get('end_observation_id'),
        executed_motion=result.get('executed_motion'))


class VerificationAgent:
    def __init__(self, runtime):
        self.runtime = runtime

    def _overview(self, claim, evidence, requested_ids):
        memory = self.runtime.memory
        task = memory.task
        index = task.current_subgoal_index
        observed = {v['observation_id']: v for v in memory.observations}
        # Complete compact chronological manifest for context size, with
        # detailed measured FORWARD receipts. Turn and look receipts can be
        # expanded by inspect_history without sending huge payloads every round.
        actions = [_action_row(a,observed) for a in memory.actions]
        manifest=[dict(step=a['step'],action=a['action'],
            subgoal=a['subgoal'],motion_epoch=a['motion_epoch']) for a in actions]
        detailed=[a for a in actions if a['action']=='MOVE_FORWARD'
                  or a['step']>=len(memory.actions)-8]
        goal = task.subgoals[index]
        scenes = []
        for entry in getattr(memory, 'scene_history', []):
            src = entry.get('source') or {}
            scene = entry.get('scene') or {}
            scenes.append(dict(observation_id=src.get('observation_id'),
                subgoal=src.get('subgoal'),
                description=str(scene.get('description',''))[:240],
                landmarks=[str(x.get('name',''))[:100]
                           for x in scene.get('landmarks',[])[:8]]))
        return dict(
            instruction=task.instruction,
            ordered_milestones=[dict(index=i,action=g.get('action'),
                target=g.get('target'),landmark=g.get('landmark'),
                source_text=g.get('source_text'),completed=i in task.completed_subgoals)
                for i,g in enumerate(task.subgoals)],
            current_subgoal_index=index, current_goal=goal,
            claim=claim, proposed_evidence=evidence,
            proposed_observation_ids=requested_ids,
            current_observation_id=(memory.current or {}).get('observation_id'),
            current_motion_epoch=((memory.current or {}).get('pose') or {}).get('motion_epoch'),
            history_action_manifest=manifest,
            history_actions=detailed,
            history_action_detail_scope='all_forward_receipts_plus_recent_eight_actions; '
                'inspect_history(start_step) expands every action in any 40-step window',
            registered_history={k: v for k,v in registered_motion_history(memory).items()
                                if k != 'rows'},
            observation_index=[dict(observation_id=o['observation_id'],
                step=o.get('step'),subgoal=o.get('subgoal'),
                motion_epoch=(o.get('pose') or {}).get('motion_epoch'))
                for o in memory.observations],
            scene_history=scenes[-35:],
            groundings=[_grounding_row(g) for g in memory.groundings][-35:],
            recent_tool_results=[_event_row(v) for v in memory.tool_ledger[-18:]],
            previous_progress_feedback=copy.deepcopy(memory.progress_feedback),
            provisional_transitions=copy.deepcopy(task.provisional_transitions[-8:]),
            cited_evidence_is_not_automatically_true=True,
            sensor_based_truth='public_rgbd_and_accepted_environment_actions_only')

    def _inspection(self, request, observations):
        memory = self.runtime.memory
        operation = request['operation']
        if operation == 'inspect_history':
            start = request['start_step']
            selected=[a for a in memory.actions if start<=a['step']<start+40]
            return dict(operation=operation,start_step=start,
                actions=[_action_row(a,observations) for a in selected],
                next_step=(selected[-1]['step']+1 if selected else None))
        if operation == 'inspect_records':
            records=[]
            for ident in request['record_ids']:
                try:
                    record=memory.read_memory_record(ident)
                    # Keep large tool payloads out of the prompt: model can
                    # request individual identifiers after seeing the index.
                    records.append(dict(id=ident,value=record))
                except (ValueError, KeyError, StopIteration):
                    records.append(dict(id=ident,error='unknown_record'))
            return dict(operation=operation,records=records)
        if operation == 'inspect_frames':
            requested=request['observation_ids']
            if not requested:
                return dict(operation=operation,error='specify_saved_observation_ids')
            pairs=[]
            for ident in requested:
                if ident not in observations:
                    return dict(operation=operation,error='unknown_observation_id',id=ident)
                pairs.append(self.runtime._vision_frame(ident))
            question=request['question'] or (
                'What visible changes support or contradict the current semantic milestone? '
                'Distinguish approach from actual arrival or crossing.')
            if len(pairs)==1:
                image, source=pairs[0]
                result=self.runtime.visual_perception.scene(image,source,question)
                return dict(operation=operation,views=[dict(
                    observation_id=source['observation_id'],scene=result)],
                    historical_images_not_live_safety=True)
            compared,frames=self.runtime.visual_perception.compare(pairs,question)
            return dict(operation=operation,views=compared.get('views'),
                changes=compared.get('changes'),historical_images_not_live_safety=True)
        return dict(error='unknown_verification_operation')

    @staticmethod
    def _proof_guard(memory, index, decision, inspected_frames=()):
        observed={v['observation_id']:v for v in memory.observations}
        by_step={a['step']:a for a in memory.actions}
        image_ids=decision['cited_observation_ids']
        steps=decision['cited_action_steps']
        errors=[]
        if not image_ids or not steps:
            errors.append('requires_cited_sensor_observation_and_executed_action')
        # Citing genuine records is necessary but not sufficient: at least one
        # cited image must refer to the current milestone's own evidence.
        if not any(observed.get(i,{}).get('subgoal')==index for i in image_ids):
            errors.append('no_observation_from_claimed_milestone')
        for ident in image_ids:
            if ident not in observed:
                errors.append('unknown_cited_observation_id:'+ident)
        terminal=decision['terminal_observation_id']
        endpoint=observed.get(terminal)
        if terminal not in image_ids or endpoint is None:
            errors.append('terminal_observation_not_cited_or_missing')
        # A genuine sensor frame is not itself a semantic interpretation.
        # Require VLM analysis on the claimed endpoint, logged previously or
        # explicitly requested during this independent verification session.
        reviewed=set(inspected_frames)
        for entry in list(getattr(memory,'scene_history',[])) + list(memory.groundings):
            src=entry.get('source') or {}
            frame=observed.get(src.get('observation_id'))
            if (frame and src.get('rgb_hash')==frame.get('rgb_hash')
                    and src.get('depth_hash')==frame.get('depth_hash')):
                reviewed.add(src['observation_id'])
        if terminal not in reviewed:
            errors.append('terminal_visual_semantics_not_reviewed')
        registered=[]
        for step in steps:
            action=by_step.get(step)
            if action is None or action.get('subgoal')!=index:
                errors.append('unattributed_action_step:'+str(step))
                continue
            if action['action']=='MOVE_FORWARD':
                if not registered_action(action,observed):
                    errors.append('unregistered_motion_step:'+str(step))
                else:
                    epoch=(action.get('pose') or {}).get('motion_epoch',0)
                    before=observed.get((action.get('result') or {}).get('observation_before')) or {}
                    after=observed.get((action.get('result') or {}).get('observation_after')) or {}
                    if (before.get('pose',{}).get('motion_epoch',0)!=epoch
                            or after.get('pose',{}).get('motion_epoch',0)!=epoch):
                        errors.append('move_observation_pair_crosses_motion_epoch:'+str(step))
                    else:
                        registered.append(action)
            elif action['action'].startswith('TURN_'):
                if not (action.get('result') or {}).get('visual_change'):
                    errors.append('turn_without_observed_change:'+str(step))
                else:
                    registered.append(action)
            else:
                errors.append('nonprogress_action_step:'+str(step))
        if steps and not registered:
            errors.append('no_authenticated_motion_or_turn')
        # Never bridge an UNKNOWN movement or motion epoch reset to claim one
        # continuous displacement chain. Non-metric TURN citations may coexist.
        moves=[a for a in registered if a['action']=='MOVE_FORWARD']
        if endpoint and registered and endpoint['step'] <= max(a['step'] for a in registered):
            errors.append('terminal_observation_precedes_claimed_actions')
        if moves:
            epochs={(a.get('pose') or {}).get('motion_epoch',0) for a in moves}
            if endpoint and endpoint['pose'].get('motion_epoch',0) not in epochs:
                errors.append('terminal_observation_outside_cited_motion_epoch')
            if len(epochs)>1:
                errors.append('cross_epoch_displacement_chain')
            for action in memory.actions:
                if (min(a['step'] for a in moves) < action['step']
                        < (endpoint['step'] if endpoint else max(a['step'] for a in moves))
                        and action['action']=='MOVE_FORWARD'
                        and action.get('subgoal')==index
                        and not registered_action(action,observed)):
                    errors.append('unknown_motion_between_cited_steps')
                    break
        return sorted(set(errors))

    def verify(self, claim, evidence, observation_ids):
        r=self.runtime
        memory=r.memory
        task=memory.task
        index=task.current_subgoal_index
        known={o['observation_id'] for o in memory.observations}
        base=dict(status='INSUFFICIENT_EVIDENCE',
            reason='verification_model_evidence_incomplete',subgoal=index,
            stage='IN_PROGRESS',supported=[],missing=[],contradictions=[],
            completion_confirmed=False,vertical_motion='UNKNOWN',
            evidence_details=dict(verifier='model_led_sensor_guarded'),
            confirmed_phases=[],confirmation_questions=[],information_options=[])
        if index>=len(task.subgoals):
            return dict(base,status='REJECTED',reason='all_subgoals_already_completed')
        if not evidence.strip() or not set(observation_ids)<=known:
            return dict(base,reason='unknown_observation_or_empty_evidence',
                missing=['valid_observation_citations'])
        overview=self._overview(claim,evidence,observation_ids)
        observed={v['observation_id']:v for v in memory.observations}
        inspections=[]
        inspected_frames=set()
        decision=None
        for number in range(MAX_ROUNDS):
            try:
                request_context=dict(overview=overview,inspections=inspections[-4:],
                    round_index=number,rounds_remaining=MAX_ROUNDS-number)
                try:
                    response=r.client.complete(
                        'navigation_agent','verification_decision',PROMPT,
                        request_context,VERIFICATION_SCHEMA,retry_safe=True)
                except APIError as format_error:
                    if format_error.category not in (
                            'json_syntax','structured_field','json_schema'):
                        raise
                    response=r.client.complete(
                        'navigation_agent','verification_decision',PROMPT,
                        dict(request_context,format_feedback=dict(
                            error=str(format_error)[:240],
                            instruction='Regenerate one complete schema-valid JSON. '
                                'Use every required field, no duplicate keys.')),
                        VERIFICATION_SCHEMA,retry_safe=True,format_retry=1)
            except BudgetExceeded:
                raise
            except APIError as exc:
                return dict(base,reason='verification_model_unavailable',
                    missing=['retry_independent_verification'],
                    evidence_details=dict(base['evidence_details'],error=str(exc)[:250]))
            if response['operation']=='verdict':
                decision=response
                break
            try:
                inspection=self._inspection(response,observed)
            except (APIError,ValueError,KeyError,RuntimeError) as exc:
                inspection=dict(operation=response['operation'],
                    error='inspection_failed',detail=str(exc)[:300])
            inspections.append(inspection)
            if (inspection.get('operation')=='inspect_frames'
                    and not inspection.get('error')):
                inspected_frames.update(view.get('observation_id') for view in
                    inspection.get('views',[]) if view.get('observation_id'))
        if decision is None:
            return dict(base,reason='verification_inspection_limit',
                missing=['verifier_must_return_cited_verdict'],
                evidence_details=dict(base['evidence_details'],inspection_rounds=len(inspections)))
        guard=self._proof_guard(memory,index,decision,inspected_frames) if (
            claim=='subgoal_completed' and decision['verdict']=='COMPLETED') else []
        complete=(claim=='subgoal_completed' and decision['verdict']=='COMPLETED'
                  and not guard and not decision['contradictions'] and not decision['missing'])
        status=('VERIFIED' if complete else
                'RECORDED' if claim!='subgoal_completed'
                and decision['verdict'] in ('COMPLETED','IN_PROGRESS') else
                'INSUFFICIENT_EVIDENCE')
        stage='COMPLETED' if complete else ('POSSIBLY_COMPLETED' if decision['verdict']=='COMPLETED'
            else 'IN_PROGRESS' if decision['verdict']=='IN_PROGRESS' else 'NOT_STARTED')
        result=dict(base,status=status,stage=stage,
            reason=('model_verdict_sensor_evidence_verified' if complete
                    else 'model_verdict_fails_sensor_proof_guard' if guard
                    else 'model_verdict_not_completed'),
            supported=decision['supported'],missing=list(dict.fromkeys(
                decision['missing']+guard)),contradictions=decision['contradictions'],
            completion_confirmed=complete,
            evidence_details=dict(
                verifier='model_led_sensor_guarded',
                model_verdict=decision['verdict'],
                reasoning=decision['reasoning'],
                cited_observation_ids=decision['cited_observation_ids'],
                cited_action_steps=decision['cited_action_steps'],
                terminal_observation_id=decision['terminal_observation_id'],
                inspection_rounds=len(inspections),
                next_evidence_action=decision['next_evidence_action'],
                proof_guard_errors=guard,
                source='public_observations_and_authenticated_action_receipts'))
        result['confirmation_questions']=[
            'Collect evidence for: '+x for x in result['missing']]
        result['information_options']=[
            'query_task_memory','read_memory_record','compare_views',
            'ground_landmark','get_waypoint_candidates']
        if complete:
            task.completed_subgoals.append(index)
            task.completion_evidence.append(dict(
                subgoal=index,evidence=evidence,
                observation_ids=decision['cited_observation_ids'],
                executed_steps=decision['cited_action_steps'],
                verification=copy.deepcopy(result)))
            following=index+1
            while following<len(task.subgoals) and following in task.completed_subgoals:
                following+=1
            if following==len(task.subgoals):
                unresolved=[i for i in range(len(task.subgoals))
                            if i not in task.completed_subgoals]
                if unresolved: following=unresolved[0]
            task.current_subgoal_index=following
            task.subgoal_start_steps.setdefault(str(following),len(memory.actions))
            task.subgoal_started=task.subgoal_start_steps[str(following)]
        elif status=='RECORDED':
            task.progress_claims.append(dict(
                claim=claim,evidence=evidence,observation_ids=observation_ids,
                subgoal=index,step=len(memory.actions),
                model_verdict=decision['verdict']))
        memory.progress_stage='NOT_STARTED' if complete and (
            task.current_subgoal_index<len(task.subgoals)) else stage
        memory.progress_feedback=result
        memory.event('progress_checked',claim=claim,evidence=evidence,
            observation_ids=observation_ids,result=result)
        return result
