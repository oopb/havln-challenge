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
    # Two EXCLUSIVE model-facing turn forms; normalize absolute targets to
    # the existing relative LEFT/RIGHT actuator contract before dispatch.
    # No new tool or semantic movement validator is introduced.
    heading=dict(type='number',minimum=0,maximum=360)
    def with_heading_choice(schema):
        schema['oneOf']=[
            dict(required=['target_heading_deg'],
                 **{'not':dict(anyOf=[
                     dict(required=['turn_direction']),
                     dict(required=['turn_degrees'])])}),
            dict(required=['turn_direction','turn_degrees'],
                 **{'not':dict(required=['target_heading_deg'])})]
        return schema
    motion=with_heading_choice(_optional(dict(
        turn_direction=direction,turn_degrees=turn,
        target_heading_deg=heading,
        distance_m=dict(type='number',minimum=0.25,maximum=3.0)),
        ('distance_m',)))
    observe=_optional(dict(
        objective=dict(TEXT,minLength=1,maxLength=1500),
        sweep_degrees=dict(type='integer',enum=[30,60,90,120,180,270,360])),
        ('objective',))
    specs={
        'observe_for_goal':(
            'Ask Observation Agent in natural language what to look for to advance the ENTIRE mission; '
            'it receives full instruction and narrative progress, current RGB-D, relevant movement '
            'history and retains the full RGB-D conversation within this observation session. '
            'It returns observations and optional waypoint proposals, NEVER a task-completion verdict.',
            observe),
        'navigate_relative':(
            'Move along one model-chosen finite relative segment (up to 3m). '
            'For a numerical absolute goal bearing, provide ONLY target_heading_deg '
            '(0..360, using the current reported pose heading): the executor 
            'computes the shortest calibrated LEFT/RIGHT turn from CURRENT yaw. '
            'For a directly visible relative turn, instead provide ONLY '
            'turn_direction LEFT/RIGHT/NONE and nonnegative turn_degrees '
            '(NONE=0). Never combine the two turn formats or mentally convert '
            'an absolute heading into LEFT/RIGHT yourself. '
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
            'Use for TURN and/or LOOK with ZERO forward motion. Choose EITHER '
            'target_heading_deg (absolute yaw; executor resolves shortest turn) '
            'OR turn_direction plus nonnegative turn_degrees (relative), not both. '
            'For a new vertical view at the same yaw use turn_direction=NONE, '
            'turn_degrees=0, look=up or look=down. Check lower stairs/floor '
            'versus upper doorway/distant path when their evidence is missing. '
            'Never express pure turning as navigate_relative(distance_m=0). '
            'The LOOK enum is 
            'always none/up/down. Camera limits (-60..60 degrees) are checked '
            'at execution; out-of-range LOOK returns recoverable CAMERA_PITCH_LIMIT '
            'with the current allowed_look choices and NO atomic actions. '
            'Read camera_controls.allowed_look_now before choosing.',
            with_heading_choice(_optional(dict(
                turn_direction=direction,turn_degrees=turn,
                target_heading_deg=heading,
                look=dict(enum=list(LOOK_ACTIONS)),purpose=TEXT),
                ('look',)))),
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
            'Request STOP for whole-mission success (only with direct current evidence of final arrival), '
            'or abort. The independent official scorer determines whether the episode was successful.',
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
    observed_evidence=dict(type='array',minItems=1,maxItems=5,
        items=dict(type='string',minLength=1,maxLength=600)),
)
_DECISION_ANALYSIS=dict(
    decision_analysis=dict(type='string',minLength=1,maxLength=700),
)

_PROGRESS=dict(
    progress_narrative=dict(type='string',maxLength=1200),
    navigation_intent=dict(type='string',maxLength=700),
    uncertainties=dict(type='array',maxItems=8,
                       items=dict(type='string',maxLength=220)),
    evidence_observation_ids=dict(type='array',maxItems=12,uniqueItems=True,
                                  items=dict(type='string',maxLength=64)))

def navigation_schema(tools):
    branches=[]
    for name,spec in tools.items():
        branch=_optional(dict(
            **copy.deepcopy(_EVIDENCE),
            **copy.deepcopy(_PROGRESS),
            **copy.deepcopy(_DECISION_ANALYSIS),
            tool=dict(const=name),args=spec['parameters']),
            ('observed_evidence','decision_analysis','tool','args'))
        branch['description']=spec['description']
        branches.append(branch)
    return dict(oneOf=branches)

def navigation_envelope(tools):
    return _optional(dict(
        **copy.deepcopy(_EVIDENCE),**copy.deepcopy(_PROGRESS),
        **copy.deepcopy(_DECISION_ANALYSIS),
        tool=dict(enum=list(tools)),args=dict(type='object')),
        ('observed_evidence','decision_analysis','tool','args'))

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
        if name in ('navigate_relative','turn_and_observe'):
            turn_args='(target_heading_deg:0..360 OR turn_direction:LEFT|RIGHT|NONE + turn_degrees:0..180)'
            tail=(', distance_m:0.25..3m' if name=='navigate_relative'
                  else ', look:none|up|down, purpose?:string')
            result.append(name+'('+turn_args+tail+')')
        else:
            result.append(name+'('+', '.join(k+('' if k in req else '?')+':'+signature(v)
                for k,v in args.items())+')')
    return ('Return EXACTLY one JSON object. Generate its fields in this '
        'reasoning-friendly order: (1) observed_evidence: 1..5 short '
        'source-grounded observations from CURRENT RGB, history, Depth and '
        'tool receipts; distinguish seen facts from hypotheses; '
        'each item should be concise (max 600 characters); '
        '(2) optional progress_narrative/navigation_intent/uncertainties/'
        'evidence_observation_ids: revisable model assessments; '
        '(3) decision_analysis: concise evidence-based explanation of '
        'the next navigation choice, including current-vs-historical '
        'heading and obstacle uncertainty where relevant; '
        '(4) tool; (5) args: the FINAL executable decision, always LAST. '
        'observed_evidence and decision_analysis are REQUIRED, not '
        'hidden reasoning or a second motion authorization. '
        'Put tool parameters directly inside args, not another wrapper. '
        'Do not put any fields after args. '
        'Historical left/right text is NOT a current camera bearing. '
        'For a numerical ABSOLUTE goal yaw use target_heading_deg:0..360 '
        'alone as the turn form; executor resolves shortest LEFT/RIGHT. '
        'For a visually relative turn use turn_direction:LEFT/RIGHT/NONE '
        'and nonnegative turn_degrees (NONE=0). Never mix turn forms. '
        'For camera tilt use look=up/down/none; no pitch change is automatic. '
        'Stationary LOOK_UP example: {"tool":"turn_and_observe","args":'
        '{"turn_direction":"NONE","turn_degrees":0,"look":"up"}}. '
        'Goal yaw example: {"tool":"turn_and_observe","args":'
        '{"target_heading_deg":90,"look":"none"}}. '
        'For MOVE use navigate_relative(distance_m:0.25..3m); '
        'a pure TURN/LOOK must use turn_and_observe. '
        'Optional progress fields are '
        'model assessments, not sensor proof. '
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
