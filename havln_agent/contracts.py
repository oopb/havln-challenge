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

def registry(camera, max_actions, pitch_deg=0.):
    legal=['none']
    if pitch_deg+camera.tilt_angle<=60:
        legal.append('up')
    if pitch_deg-camera.tilt_angle>=-60:
        legal.append('down')
    # Direction and magnitude are separate: no signed-angle ambiguity.
    direction=dict(enum=['LEFT','RIGHT','NONE'])
    turn=dict(type='number',minimum=0,maximum=180)
    motion=object_schema(dict(turn_direction=direction,
        turn_degrees=turn,distance_m=dict(type='number',minimum=0.25,maximum=3.0)))
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
            'Explicit turn_direction LEFT or RIGHT with nonnegative turn_degrees; '
            'NONE requires 0 degrees; LEFT/RIGHT require >0. Robot heading increases '
            'when turning LEFT, decreases when turning RIGHT. E.g. turn right 30 '
            '=> RIGHT,30; turn left 60 => LEFT,60. No signed angles. '
            'Then execute forward steps until the command finishes or actual '
            'safety/sensor interruption; no milestone authorization.',
            motion),
        'turn_and_observe':(
            'Execute calibrated rotation/LOOK for a purposeful viewpoint, without a forced forward step.',
            _optional(dict(turn_direction=direction,turn_degrees=turn,
                look=dict(enum=legal),purpose=TEXT),
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
            'Request STOP for whole-mission success (only with direct current evidence of final arrival), '
            'or abort. The independent official scorer determines whether the episode was successful.',
            object_schema(dict(outcome=dict(enum=['success','aborted']),evidence=dict(TEXT,minLength=1)))),
    }
    return {name:dict(description=desc,parameters=params,
        permission=('TERMINAL' if name=='request_finish' else
                    'NAVIGATION' if name in ('navigate_relative','navigate_local',
                        'turn_and_observe','observe_for_goal') else 'READ_ONLY'))
        for name,(desc,params) in specs.items()}

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
        branch=_optional(dict(tool=dict(const=name),
                              args=spec['parameters'],**copy.deepcopy(_PROGRESS)),
                         ('tool','args'))
        branch['description']=spec['description']
        branches.append(branch)
    return dict(oneOf=branches)

def navigation_envelope(tools):
    return _optional(dict(tool=dict(enum=list(tools)),
                          args=dict(type='object'),**copy.deepcopy(_PROGRESS)),
                     ('tool','args'))

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
    return ('For turn_direction choose LEFT, RIGHT or NONE and specify '
        'turn_degrees as a NONNEGATIVE magnitude. NONE means 0; LEFT/RIGHT '
        'means positive. Never encode left/right using a signed angle. '
        'Return exactly one JSON object with tool and args; optionally include '
        'progress_narrative, navigation_intent, uncertainties, evidence_observation_ids. '
        'These optional fields are model assessments saved in Memory, not sensor proof. '
        'No numbered subgoals or VERIFY calls. Tools: '+'; '.join(result))

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
