"""Current semantic plans, one-tool decisions and trusted execution feedback."""
from .api_client import TEXT, object_schema

PREDICATES = ('landmark_visible', 'entry_confirmed', 'actual_descent_observed',
    'actual_ascent_observed', 'registered_translation', 'orientation_completed',
    'landmark_passed', 'target_nearby', 'final_arrival_visible')
CONDITIONS = dict(type='array', minItems=1, maxItems=len(PREDICATES), uniqueItems=True,
                  items=dict(enum=list(PREDICATES)))
TASK_REVIEW = object_schema(dict(intent=dict(enum=['continue', 'revise']), subgoal=TEXT))
# Backward compatible with legacy model output, but no longer required or trusted.
TASK_REVIEW['properties']['success_criteria'] = CONDITIONS
HYPOTHESIS = object_schema(dict(finding=TEXT, source_observation_ids=dict(type='array',
    minItems=1, maxItems=4, uniqueItems=True, items=TEXT)))


def registry(camera, max_actions, pitch_deg=0.):
    text = dict(TEXT, minLength=1)
    ids = dict(type='array', minItems=1, maxItems=4, uniqueItems=True, items=text)
    top = dict(type='integer', minimum=1, maximum=4)
    history = dict(scope=dict(const='historical'), observation_ids=ids)
    turn = []
    for look, available in [('none', max_actions), ('up', max_actions-1), ('down', max_actions-1)]:
        if available < 0 or (look=='up' and pitch_deg+camera.tilt_angle>60) or (look=='down' and pitch_deg-camera.tilt_angle<-60):
            continue
        turn.append(dict(object_schema(dict(relative_turn_deg=dict(type='number',
            minimum=-available*camera.turn_angle, maximum=available*camera.turn_angle,
            enum=[step*camera.turn_angle for step in range(-available,available+1) if step or look!='none'],
            description='degrees, positive turns left'), look=dict(const=look),
            purpose=dict(enum=['explore','dynamic_recheck','intentional_recheck']))),
            required=['relative_turn_deg','look']))
    specs = {
        'turn_and_observe': ('Execute up to 3 calibrated TURN/LOOK actions. Duplicates are reported; purpose=intentional_recheck explicitly allows re-observation. Prior views never authorize current motion.', dict(oneOf=turn)),
        'analyze_scene': ('Ask one visual question about current openings/landmarks/people, or explicitly select historical observations.',
            dict(oneOf=[dict(object_schema(dict(question=text)), required=[]),
                dict(object_schema(dict(history, question=text)), required=['scope', 'observation_ids'])])),
        'observe_for_goal': ('Bounded active observation with OPEN-VOCABULARY target types, optional force_recheck and paused-waypoint preservation. Never moves forward and does not implicitly cancel a pending route.',
            dict(object_schema(dict(
                objective=dict(TEXT,minLength=1),
                target_types=dict(type='array',minItems=1,maxItems=3,uniqueItems=True,
                    items=dict(type='string',minLength=1,maxLength=80)),
                direction=dict(enum=['left','right']),
                sweep_degrees=dict(enum=[30,60,90,120]),
                checkpoint_degrees=dict(enum=[30,45]),
                look=dict(enum=['none','up','down']),
                force_recheck=dict(type='boolean'))),
                required=['objective','target_types','direction','sweep_degrees'])),
        'compare_views': ('Compare current and/or saved RGB keyframes for one question. Historical views never authorize motion.',
            object_schema(dict(observation_ids=ids, question=text))),
        'ground_landmark': ('Locate a visible target/entry/treads; metric projection uses local Depth.',
            dict(oneOf=[object_schema(dict(target=text)), object_schema(dict(target=text,
                scope=dict(const='historical'), observation_id=text))])),
        'assess_entry': ('Ask where a stair entry may lie relative to a railing/opening; answer is a hypothesis.',
            object_schema(dict(question=text))),
        'inspect_geometry': ('ON-DEMAND full geometry diagnosis after an obstacle, unclear drop or repeated block; normal MOVE uses a fast live depth probe. Inspection provides evidence and does not itself authorize motion.', object_schema({})),
        'get_waypoint_candidates': ('Optional short candidate bearings with current-depth evidence; not mandatory before navigate_relative or navigate_to_pixel. Every forward step gets fresh safety checks.',
            dict(object_schema(dict(target=text, purpose=dict(enum=['approach_landmark', 'explore']))), required=[])),
        'navigate_relative': ('Model-directed movement: select a calibrated relative bearing and desired segment length directly, without an approved semantic waypoint. Executes a short atomic batch, rechecks live collision before each forward step, and returns a resumable waypoint_id. Use navigate_local with that ID to continue.',
            object_schema(dict(turn_deg=dict(type='number',minimum=-180,maximum=180),
                distance_m=dict(type='number',minimum=0.25,maximum=3.0),
                max_atomic_steps=dict(type='integer',minimum=1,maximum=max_actions)))),
        'navigate_to_pixel': ('Model-directed visual bearing: select normalized target image pixel and intended short travel distance. Uses live RGB-D for a bearing hypothesis; the pixel is NOT certified walkable floor. Live collision checks still apply, returning a resumable waypoint_id.',
            object_schema(dict(u=dict(type='number',minimum=0,maximum=1),
                v=dict(type='number',minimum=0,maximum=1),
                distance_m=dict(type='number',minimum=0.25,maximum=3.0),
                max_atomic_steps=dict(type='integer',minimum=1,maximum=max_actions)))),
        'navigate_local': ('Execute or resume a returned waypoint ID in up to 3 atomic actions, revalidating fresh RGB-D and people before each forward step.',
            object_schema(dict(waypoint_id=text, max_atomic_steps=dict(type='integer', minimum=1, maximum=max_actions)))),
        'cancel_local_navigation': ('Cancel a pending local waypoint without changing the semantic task.', object_schema(dict(reason=text))),
        'recall_observation': ('Retrieve up to four short keyframe summaries by place, heading or landmark.',
            dict(object_schema(dict(place_id=text, heading_deg=dict(type='number'), landmark=text, top_k=top)), required=['place_id'])),
        'query_spatial_memory': ('Retrieve up to four relevant landmark, route or recent/expired-person summaries.',
            dict(object_schema(dict(place_id=text, target=text, top_k=top)), required=['place_id'])),
        'query_task_memory': ('Retrieve current task, typed conditions, missing evidence and short verified milestones.',
            dict(object_schema(dict(subgoal_id=text)), required=[])),
        'read_memory_record': ('Open one explicitly requested observation/tool/grounding/task record from local history.', object_schema(dict(record_id=text))),
        'verify_subgoal': ('Propose evidence for independent task verification; text alone never completes a task.',
            object_schema(dict(claim=dict(enum=['landmark_found','entered','bottom_visible','subgoal_completed']),
                evidence=text, observation_ids=ids))),
        'request_task_review': ('Request a semantic review for a specific task change or major new evidence.', object_schema(dict(reason=text))),
        'request_finish': ('Propose verified completion, or explicitly abort. Runtime owns STOP and completion checks.',
            object_schema(dict(outcome=dict(enum=['success','aborted']), evidence=text))),
    }
    return {name:dict(description=description, parameters=parameters,
        permission='NAVIGATION' if name in ('turn_and_observe','navigate_local','navigate_relative','navigate_to_pixel','observe_for_goal') else
                   'TERMINAL' if name=='request_finish' else 'READ_ONLY')
        for name,(description,parameters) in specs.items()}


def navigation_schema(tools):
    branches=[]
    for name,spec in tools.items():
        branch=object_schema(dict(tool=dict(const=name), args=spec['parameters'],
            memory_delta=dict(type='array', maxItems=4, items=HYPOTHESIS)))
        branch['required']=['tool','args']
        branch['description']=spec['description']
        branches.append(branch)
    return dict(oneOf=branches)


def navigation_envelope(tools):
    """Malformed messages are API errors; valid calls get runtime argument feedback."""
    value=object_schema(dict(tool=dict(enum=list(tools)),args=dict(type='object'),
        memory_delta=dict(type='array',maxItems=4,items=HYPOTHESIS)))
    value['required']=['tool','args']
    return value


def compact_navigation_protocol(tools):
    """Derive short argument signatures from the same strictly validated registry."""
    import json
    def kind(schema):
        if 'const' in schema:
            return json.dumps(schema['const'],separators=(',',':'))
        if 'enum' in schema:
            return '|'.join(json.dumps(v,separators=(',',':')) for v in schema['enum'])
        value=schema.get('type','object')
        if value=='array':
            return kind(schema['items'])+'['+str(schema.get('minItems',0))+'..'+str(schema.get('maxItems',''))+']'
        if value in ('integer','number'):
            return value+'['+str(schema.get('minimum',''))+'..'+str(schema.get('maximum',''))+']'
        return 'str' if value=='string' else value
    def signatures(schema):
        if 'oneOf' in schema:
            return ' OR '.join(signatures(branch) for branch in schema['oneOf'])
        required=schema.get('required',[])
        return '('+', '.join(name+('' if name in required else '?')+':'+kind(value)
            for name,value in schema.get('properties',{}).items())+')'
    return ('Return exactly {"tool":"name","args":{...}}. Optional memory_delta is at most 4 '
        'objects {"finding":"text","source_observation_ids":["existing ID"]}; hypotheses only. '
        'Use only listed tools/args. ? means optional; all text <=1000 characters, IDs nonempty, '
        'arrays unique. Tool signatures and purpose:\n'+'\n'.join(
            name+signatures(spec['parameters'])+' '+spec['description'] for name,spec in tools.items()))


def tool_result_schema():
    return dict(type='object', properties=dict(
        status=dict(enum=['OK','PARTIAL','NEEDS_VIEW','BLOCKED','INVALID_ARGUMENTS',
            'INVALID_TARGET','STALE_TARGET','STALE_OBSERVATION','API_ERROR','API_UNAVAILABLE',
            'TARGET_REACHED','INSUFFICIENT_EVIDENCE','RECORDED','VERIFIED','REJECTED',
            'FINISHED','CANCELLED','LOCALIZED','UNKNOWN','EXECUTED','EXECUTABLE','STOPPED','BUDGET_LIMIT',
            'INTERRUPTED','TOOL_ERROR','REQUIRES_ATTENTION','NO_NEW_INFORMATION']),
        tool=dict(type='string'), tool_call_id=dict(type='string'),
        requested_actions=dict(type='integer',minimum=0), executed_actions=dict(type='integer',minimum=0),
        executed_motion=dict(type='array',items=dict(enum=['STOP','MOVE_FORWARD','TURN_LEFT','TURN_RIGHT','LOOK_UP','LOOK_DOWN'])),
        start_observation_id=dict(type='string'), end_observation_id=dict(type='string'),
        seconds=dict(type='number',minimum=0)),
        required=['status','tool','tool_call_id','requested_actions','executed_actions','executed_motion',
                  'start_observation_id','end_observation_id','seconds'])
