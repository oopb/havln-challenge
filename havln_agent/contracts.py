"""Navigation and Observation tool contracts for one continuous natural-language mission.

No numbered milestones, VERIFY state transitions, route-prediction ledgers, or
implicit semantic completion predicates are part of the execution protocol.
"""
import copy
from .api_client import TEXT, object_schema

def _optional(properties, required=()):
    schema=object_schema(properties)
    schema['required']=list(required)
    return schema

# The model-facing tool shape MUST NOT change with camera pitch. A pitch
# limit is a recoverable *execution condition*, not an invalid JSON tool
# argument. Only sensor/actuator evidence determines whether a LOOK can run.
CAMERA_PITCH_LIMIT_DEG = 60.0
LOOK_ACTIONS = ('none', 'up', 'down')


def camera_look_options(pitch_deg, tilt_step_deg):
    """Return currently physically available LOOK actions, for model feedback.

    The protocol itself always accepts all three LOOK values. Never mutate
    that protocol based on current pitch.
    """
    legal=['none']
    if pitch_deg + tilt_step_deg <= CAMERA_PITCH_LIMIT_DEG + 1e-6:
        legal.append('up')
    if pitch_deg - tilt_step_deg >= -CAMERA_PITCH_LIMIT_DEG - 1e-6:
        legal.append('down')
    return legal


def registry(camera, max_actions, pitch_deg=0.):
    # Retain the pitch_deg argument for existing callers, but DO NOT use it to
    # shrink the tool enum: such schema drift triggered v9's api_error STOP.
    # Direction and magnitude are separate: no signed-angle ambiguity.
    direction=dict(enum=['LEFT','RIGHT','NONE'])
    turn=dict(type='number',minimum=0,maximum=180)
    # One model-facing RELATIVE turn contract. Heading is sensor history only.
    motion=object_schema(dict(
        turn_direction=direction,turn_degrees=turn,
        distance_m=dict(type='number',minimum=0.25,maximum=3.0)))
    observe=_optional(dict(
        objective=dict(TEXT,minLength=1,maxLength=1500),
        sweep_degrees=dict(type='integer',enum=[30,60,90,120,180,270,360])),
        ('objective',))
    specs={
        'observe_for_goal':(
            'Ask Observation Agent a specific route-critical question for the entire mission. '
            'It sees current RGB-D, source-linked history and prior frames from the same '
            'observation session. For competing room entrances, ask whether a distinctive '
            'person/activity is actually visible; furniture or a doorway is not proof of '
            'the requested room. It cannot walk to another room, choose movement or '
            'certify task completion.',
            observe),
        'navigate_relative':(
            'Move along one model-chosen finite relative segment (up to 3m). '
            'Choose turn_direction LEFT/RIGHT/NONE relative to the robot CURRENT '
            'view and nonnegative turn_degrees (NONE=0). LEFT rotates the robot '
            'toward its left image side, RIGHT toward its right image side. '
            'Do not output absolute headings; the executor applies the chosen '
            'physical relative motor turn. '
            'distance_m is a NOMINAL command length quantized into a finite '
            'number of forward action receipts, NOT measured robot displacement. '
            'Distance MUST be at least 0.25m; for a pure turn without forward '
            'motion use turn_and_observe(look=none), never distance_m=0. '
            'Clear stable space may justify 0.5–1m or more; near obstacles '
            'a shorter segment may be appropriate. The model chooses. '
            'Every actual forward step still gets live Depth preflight. '
            'No metric robot position is available. '
            'Then execute forward steps until the command finishes or actual '
            'safety/sensor interruption; no milestone authorization.',
            motion),
        'turn_and_observe':(
            'Use for TURN and/or LOOK with ZERO forward motion. Choose '
            'turn_direction LEFT/RIGHT/NONE relative to CURRENT robot view, '
            'and nonnegative turn_degrees (NONE=0). No absolute target heading. '
            'For a new vertical view at the same yaw use turn_direction=NONE, '
            'turn_degrees=0, look=up or look=down. Check lower stairs/floor '
            'versus upper doorway/distant path when their evidence is missing. '
            'Never express pure turning as navigate_relative(distance_m=0). '
            'The LOOK enum is '
            'always none/up/down. Camera limits (-60..60 degrees) are checked '
            'at execution; out-of-range LOOK returns recoverable CAMERA_PITCH_LIMIT '
            'with the current allowed_look choices and NO atomic actions. '
            'Read camera_controls.allowed_look_now before choosing.',
            _optional(dict(
                turn_direction=direction,turn_degrees=turn,
                look=dict(enum=list(LOOK_ACTIONS)),purpose=TEXT),
                ('turn_direction','turn_degrees','look'))),
        'navigate_local':(
            'Resume a previously selected, still-active local motion command.',
            object_schema(dict(waypoint_id=TEXT))),
        'cancel_local_navigation':(
            'Cancel the pending finite motion command; never change semantic mission progress.',
            object_schema(dict(reason=TEXT))),
        'query_task_memory':(
            'Retrieve complete task wording, narrative progress, intent, uncertainties, and evidence references.',
            object_schema({})),
        'read_memory_record':(
            'Inspect a specific saved sensor/observation/tool evidence record by ID.',
            object_schema(dict(record_id=TEXT))),
        'request_finish':(
            'Request STOP for whole-mission success only with current evidence of the '
            'actual FINAL destination, or abort. For a room distinguished by an occupant '
            'or activity, reaching one furnished doorway while another plausible room '
            'is unchecked does not establish the correct room. Official scoring is external.',
            object_schema(dict(outcome=dict(enum=['success','aborted']),evidence=dict(TEXT,minLength=1)))),
    }
    return {name:dict(description=desc,parameters=params,
        permission=('TERMINAL' if name=='request_finish' else
                    'NAVIGATION' if name in ('navigate_relative','navigate_local',
                        'turn_and_observe','observe_for_goal') else 'READ_ONLY'))
        for name,(desc,params) in specs.items()}

# Output is intentionally authored in autoregressive reading order:
# authentic visual/tool observations -> short explicit decision analysis
# -> final tool/args. This is a visible decision summary, not hidden chain
# of thought or a separate validator authorizing the movement.
_EVIDENCE=dict(
    observed_evidence=dict(type='array',minItems=1,maxItems=8,
        items=dict(type='string',minLength=1,maxLength=600)),
)
_DECISION_ANALYSIS=dict(
    decision_analysis=dict(type='string',minLength=1,maxLength=1000),
)

_PROGRESS=dict(
    progress_narrative=dict(type='string',maxLength=1200),
    navigation_intent=dict(type='string',maxLength=700),
    scene_goal=dict(type='string',maxLength=350),  # optional semantic spatial target
    uncertainties=dict(type='array',maxItems=8,
                       items=dict(type='string',maxLength=220)),
    evidence_observation_ids=dict(type='array',maxItems=12,uniqueItems=True,
                                  items=dict(type='string',maxLength=64)),
    local_handoff=_optional(dict(
        goal=dict(type='string', minLength=1, maxLength=200),
        exit_condition=dict(type='string', minLength=1, maxLength=200)),
        ('goal', 'exit_condition')),
    route_reference_update=dict(
        type='object',additionalProperties=False,
        properties=dict(
            source_observation_id=dict(type='string',minLength=1,maxLength=64),
            place_description=dict(type='string',minLength=1,maxLength=180),
            instruction_relation=dict(type='string',minLength=1,maxLength=300),
            evidence_status=dict(enum=['candidate','supported','contradicted']),
            route_attempt_outcome=dict(type='string',minLength=1,maxLength=300),
            outcome_observation_id=dict(type='string',minLength=1,maxLength=64)),
        # The source frame and original instruction relation are the only
        # essential facts. Labels and candidate confidence may be omitted.
        required=['source_observation_id','instruction_relation']))


def normalize_navigation_route_update(decision, known_observation_ids=None):
    """Normalize optional *descriptive* route metadata, never motor actions.

    Some JSON-object providers paraphrase route-reference field names even
    while returning a valid tool decision. Canonicalize a small set of
    semantically equivalent names observed in real Phase 2 diagnostics.
    Discard an ungrounded/ambiguous OPTIONAL reference rather than forcing
    another paid completion or fabricating a waypoint. This does not bypass
    the registered executable tool schema.
    """
    if not isinstance(decision, dict) or 'route_reference_update' not in decision:
        return decision
    original=decision['route_reference_update']
    output=dict(decision)
    if not isinstance(original, dict):
        output.pop('route_reference_update', None)
        return output

    def choose(*names):
        values=[original.get(k) for k in names
                if isinstance(original.get(k), str) and original[k].strip()]
        if not values or len(set(values)) != 1:
            # Conflicting explicit canonical/alias facts are not repairable.
            return None
        return values[0].strip()

    source=choose('source_observation_id')
    relation=choose('instruction_relation','instruction_relative_relation')
    if (not source or len(source)>64 or not relation or len(relation)>300
            or (known_observation_ids is not None
                and source not in known_observation_ids)):
        output.pop('route_reference_update', None)
        return output

    normalized=dict(source_observation_id=source,instruction_relation=relation)
    description=choose('place_description','junction_description','description')
    if description and len(description)<=180:
        normalized['place_description']=description
    status=original.get('evidence_status')
    if status in ('candidate','supported','contradicted'):
        normalized['evidence_status']=status
    else:
        normalized['evidence_status']='candidate'

    attempt=original.get('tried_route')
    # Explicit canonical outcome takes precedence; otherwise recover a
    # single cited trial, never flatten an ambiguous list of alternatives.
    outcome=original.get('route_attempt_outcome')
    outcome_id=original.get('outcome_observation_id')
    if not isinstance(outcome,str) and isinstance(attempt,dict):
        outcome=attempt.get('outcome') or attempt.get('description')
        outcome_id=attempt.get('outcome_observation_id')
    if (isinstance(outcome,str) and outcome.strip()
            and len(outcome.strip())<=300
            and isinstance(outcome_id,str) and len(outcome_id)<=64
            and (known_observation_ids is None
                 or outcome_id in known_observation_ids)):
        normalized['route_attempt_outcome']=outcome.strip()
        normalized['outcome_observation_id']=outcome_id
    output['route_reference_update']=normalized
    return output

def _navigation_progress_fields(allow_local_handoff=False):
    result = copy.deepcopy(_PROGRESS)
    if not allow_local_handoff:
        result.pop('local_handoff', None)
    return result


def navigation_schema(tools, allow_local_handoff=False):
    branches=[]
    for name,spec in tools.items():
        branch=_optional(dict(
            **copy.deepcopy(_EVIDENCE),
            **_navigation_progress_fields(allow_local_handoff),
            **copy.deepcopy(_DECISION_ANALYSIS),
            tool=dict(const=name),args=spec['parameters']),
            ('observed_evidence','decision_analysis','tool','args'))
        branch['description']=spec['description']
        branches.append(branch)
    return dict(oneOf=branches)

def navigation_envelope(tools, allow_local_handoff=False):
    return _optional(dict(
        **copy.deepcopy(_EVIDENCE),
        **_navigation_progress_fields(allow_local_handoff),
        **copy.deepcopy(_DECISION_ANALYSIS),
        tool=dict(enum=list(tools)),args=dict(type='object')),
        ('observed_evidence','decision_analysis','tool','args'))

def navigation_generation_schema(tools, allow_local_handoff=False):
    """A discriminated executable union: tool *and* its own required args.

    Each branch is a complete JSON object with a distinct required tool const.
    This avoids the dangerous flat-args union, where XGrammar could legally
    generate navigate_relative without turn_direction or turn_degrees.
    There are no siblings alongside oneOf (some backends ignore those).
    Native JSON dialect projection is performed separately by APIClient;
    the authoritative execution validator stays navigation_schema(tools).
    """
    return navigation_schema(tools, allow_local_handoff=allow_local_handoff)


def navigation_choice_schema(tools, allow_local_handoff=False):
    """The non-executable first stage if an endpoint cannot enforce unions.

    Tool selection and narrative are separate from executable arguments.
    The second stage uses tools[choice['tool']]['parameters'] directly, so
    neither stage can generate unrelated tool arguments.
    """
    return _optional(dict(
        **copy.deepcopy(_EVIDENCE),
        **_navigation_progress_fields(allow_local_handoff),
        **copy.deepcopy(_DECISION_ANALYSIS),
        tool=dict(enum=list(tools))),
        ('observed_evidence', 'decision_analysis', 'tool'))


def compact_navigation_protocol(tools):
    import json
    def signature(s):
        if 'enum' in s:
            return '|'.join(map(str,s['enum']))
        if 'const' in s:
            return json.dumps(s['const'])
        return s.get('type','object')
    result=[]
    for name,spec in tools.items():
        args=spec['parameters'].get('properties',{})
        req=set(spec['parameters'].get('required',[]))
        result.append(name+'('+', '.join(k+('' if k in req else '?')+':'+signature(v)
            for k,v in args.items())+')')

    return ('Return exactly one JSON object with fields in order: '
        'observed_evidence (1-2 brief source-grounded facts, not repeated '
        'history); optional progress_narrative, navigation_intent, '
        'scene_goal, uncertainties, evidence_observation_ids and '
        'route_reference_update (only if changed); decision_analysis '
        '(one brief evidence-based sentence); tool; args (FINAL executable '
        'decision). Put tool parameters directly inside args. '
        'scene_goal is a persistent scene/room goal, NOT a motor angle, '
        'person position or verified arrival. route_reference_update uses '
        'source_observation_id (real RGB ID) and instruction_relation '
        '(ORIGINAL instruction-relative direction); optional '
        'place_description, evidence_status=candidate|supported|contradicted, '
        'route_attempt_outcome and outcome_observation_id. '
        'Turn/LOOK is relative to CURRENT RGB using '
        'turn_direction LEFT/RIGHT/NONE and nonnegative turn_degrees. '
        'look=up/down/none; no pitch change is automatic. '
        'Use navigate_relative for forward (distance_m 0.25..3m); '
        'use turn_and_observe for pure TURN/LOOK. '
        'The FINAL executable decision is tool + args. '
        'All progress text is revisable model opinion, not verified fact. '
        'Tools: '+'; '.join(result))


def tool_result_schema():
    return dict(type='object',properties=dict(
        status=dict(type='string'),tool=dict(type='string'),
        tool_call_id=dict(type='string'),
        requested_actions=dict(type='integer',minimum=0),
        executed_actions=dict(type='integer',minimum=0),
        executed_motion=dict(type='array',items=dict(type='string')),
        start_observation_id=dict(type='string'),
        end_observation_id=dict(type='string'),
        seconds=dict(type='number',minimum=0)),
        required=['status','tool','tool_call_id','requested_actions',
                  'executed_actions','executed_motion','start_observation_id',
                  'end_observation_id','seconds'])
