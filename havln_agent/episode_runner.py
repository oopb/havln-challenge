"""Public environment I/O, six-component runtime and official action export."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from .api_client import APIError, BudgetExceeded
from .habitat_adapter import read_current_observation
from .perception import Observation, adapt_observation
from .runtime import AgentRuntime
from .scheduler import request, run
from .task_planner import TaskPlanner
from scripts.export_cma_submission import atomic_json, check_rows


def canonical_step_frame_refs(directory, feedback):
    """Use the same RGB-D frames as memory and historical image lookup."""
    refs = {}
    for key, obs_id in (('input_file', feedback['observation_before']),
                        ('post_file', feedback['observation_after'])):
        filename = 'observations/' + obs_id + '.npz'
        if not (Path(directory) / filename).is_file():
            raise RuntimeError('missing_canonical_observation_for_step')
        refs[key] = filename
    return refs


def run_episode(env, episode_id, camera, client, config, directory, instruction=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('Episode trace directory already contains files; use a fresh output directory')
    for name in ('api', 'task_planner', 'navigation_agent', 'visual_perception',
                 'steps', 'tools',
                 'safety', 'progress', 'state', 'events'):
        (directory / (name+'.jsonl')).write_text('')
    client.begin_episode(episode_id, config['api_budget'], directory / 'api.jsonl')
    started = time.monotonic()
    environment_done = False

    def append(name, record):
        with (directory / (name+'.jsonl')).open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False)+'\n')

    def step(action):
        nonlocal environment_done
        raw, _, environment_done, info = env.step(action)
        if isinstance(info, dict) and info.get('error'):
            raise RuntimeError('environment_action_rejected')
        return adapt_observation(raw, camera), environment_done

    def refresh():
        return read_current_observation(env, camera, runtime.memory.task.instruction)

    def record_step(record, before, after):
        # MemoryRecords already saves each distinct RGB-D frame once under
        # observations/obs_XXXX.npz. The former input-NNN/post-NNN copies
        # tripled disk usage without adding diagnostic information.
        feedback = runtime.memory.actions[record['step']]['result']
        record.update(canonical_step_frame_refs(directory, feedback))
        append('steps', record)
        append('safety', dict(step=record['step'], tool_call_id=record['tool_call_id'], **record['safety']))

    def record_observation(record, observation):
        (directory / 'observations').mkdir(exist_ok=True)
        filename = 'observations/'+record['observation_id']+'.npz'
        np.savez_compressed(directory / filename, rgb=observation.rgb, depth=observation.depth,
                            pitch_deg=observation.pitch_deg)
        record['input_file'] = filename

    def load_observation(record):
        with np.load(directory / record['input_file']) as saved:
            return Observation(saved['rgb'], saved['depth'], runtime.memory.task.instruction,
                record['captured_at'], float(saved['pitch_deg']))

    def record_tool(record):
        append('tools', record)
        if record['call']['tool'] in ('analyze_scene', 'compare_views', 'ground_landmark', 'assess_entry'):
            append('visual_perception', dict(tool_call_id=record['call_id'],
                requester=record['actor'], subgoal_id=record['subgoal_id'],
                tool=record['call']['tool'], args=record['call']['args'], result=record['result']))

    runtime = AgentRuntime(client, camera, config, step, refresh, record_step, record_tool,
                           record_observation, load_observation)

    def record_event(record):
        append('events', record)
        if record['event'] in ('navigation_decision', 'task_plan', 'finish_verdict',
                               'navigation_mode_changed'):
            append('state', runtime.state_view())
    runtime.event_callback = record_event
    try:
        startup_error, plan = None, None
        # Plan before reset when possible, so slow model work cannot age the initial frame.
        if instruction is not None:
            runtime.memory.reset(episode_id, instruction)
            try:
                plan = request(runtime, 'task_planner',
                    lambda feedback: TaskPlanner.parse(client, instruction, feedback), append)
            except (APIError, BudgetExceeded) as error:
                startup_error = error
        observation = adapt_observation(env.reset(), camera)
        if str(env.current_episode.episode_id) != episode_id:
            raise ValueError('Reset resolved the wrong episode')
        if instruction is not None and observation.instruction != instruction:
            raise ValueError('Instruction sensor differs from the planned episode text')
        if instruction is None:
            runtime.memory.reset(episode_id, observation.instruction)
            try:
                plan = request(runtime, 'task_planner',
                    lambda feedback: TaskPlanner.parse(client, observation.instruction, feedback), append)
            except (APIError, BudgetExceeded) as error:
                startup_error = error
        if plan:
            runtime.memory.set_plan(plan)
            append('task_planner', dict(event='instruction_plan', plan=plan))
        runtime.accept_observation(observation)
        atomic_json(directory / 'episode.json', dict(episode_id=episode_id,
            instruction=observation.instruction, plan=plan))
        if startup_error:
            runtime.api_failure = getattr(startup_error, 'request_context',
                dict(error=str(startup_error), category=getattr(startup_error, 'category', 'budget_exhausted')))
            runtime.terminate('api_budget_exhausted' if isinstance(startup_error, BudgetExceeded) else 'api_error')
        else:
            run(runtime, append)
        if not environment_done:
            raise ValueError('Environment did not terminate after the executed STOP/budget')
        actions = [a['action'] for a in runtime.memory.actions]
        row = dict(episode_id=episode_id, actions=actions)
        check_rows([row], [episode_id])
        pose = runtime.motion_controller.pose()
        summary = dict(episode_id=episode_id, architecture='observe_move_v1', steps=len(actions),
            stop_reason=runtime.stop_reason, finish_reason=runtime.finish_reason,
            api_failure=runtime.api_failure, navigation_liveness=runtime.navigation_liveness,
            seconds=time.monotonic()-started,
            api=client.stats(), pose=pose, decisions=runtime.decisions,
            decisions_by_role=runtime.decision_counts, task_reviews=max(0,runtime.plan_revision-1),
            completed_subgoals=len(runtime.memory.task.completed_subgoals),
            move_forward_steps=actions.count('MOVE_FORWARD'),
            turn_steps=actions.count('TURN_LEFT')+actions.count('TURN_RIGHT'),
            look_steps=actions.count('LOOK_UP')+actions.count('LOOK_DOWN'),
            first_forward_step=next((i for i,a in enumerate(actions) if a=='MOVE_FORWARD'),None),
            net_displacement_m_est=(pose['x']**2+pose['z']**2)**.5,
            translation_m_est=sum(r['result'].get('translation_m') or 0 for r in runtime.memory.tool_ledger),
            translation_source='rgbd_estimate_only_not_physical_ground_truth',
            waypoint_completed=sum(r['call']['tool']=='navigate_local' and r['result']['status']=='TARGET_REACHED'
                for r in runtime.memory.tool_ledger),
            waypoint_candidate_queries=sum(r['call']['tool']=='get_waypoint_candidates'
                for r in runtime.memory.tool_ledger),
            waypoint_queries_without_executable_path=sum(
                r['call']['tool']=='get_waypoint_candidates' and not r['result'].get('paths')
                for r in runtime.memory.tool_ledger),
            local_navigation_blocked_calls=sum(
                r['call']['tool']=='navigate_local' and (
                    r['result'].get('reason')=='occupied_next_step'
                    or r['result'].get('interrupted_status')=='BLOCKED'
                    or r['result'].get('status')=='BLOCKED')
                for r in runtime.memory.tool_ledger),
            actions_since_last_forward=(len(actions)-1-max(
                (i for i,a in enumerate(actions) if a=='MOVE_FORWARD'),default=-1)),
            tool_calls=len(runtime.memory.tool_ledger),
            tools_by_name=dict(Counter(r['call']['tool'] for r in runtime.memory.tool_ledger)),
            tool_outcomes=dict(Counter(r['result']['status'] for r in runtime.memory.tool_ledger)),
            memory=runtime.memory.stats(),
            safety_rejections=sum(e['event']=='safety_rejected' for e in runtime.memory.events),
            motion_registration_states=dict(Counter((a['result'].get('rgbd_motion') or {}).get('status','UNKNOWN')
                for a in runtime.memory.actions if a['action']=='MOVE_FORWARD')),
            geometry_rejection_reasons=dict(Counter(d.get('reason') for r in runtime.memory.tool_ledger
                for d in r['result'].get('directions',[]) if d.get('status')!='EXECUTABLE')),
            visual_tool_errors=sum(r['result']['status']=='TOOL_ERROR' and r['result'].get('recoverable',False)
                for r in runtime.memory.tool_ledger),
            actual_navigation_success=None, collision_metrics=None,
            validation_scope='execution_trace_requires_independent_environment_and_scoring_verification')
        runtime.event('export', source='action_exporter', actions=len(actions), validated_by='check_rows')
        atomic_json(directory / 'summary.json', summary)
        return row
    except Exception as error:
        runtime.memory.finish_attempt('INTERRUPTED','episode_execution_error')
        atomic_json(directory / 'failure.json', dict(episode_id=episode_id,
            error_type=type(error).__name__, error=str(error),
            executed_actions=[a['action'] for a in runtime.memory.actions]))
        raise
    finally:
        atomic_json(directory / 'memory.json', runtime.memory.snapshot())
        # All tool receipts already appear in tools.jsonl and memory.json.
        if runtime.memory.current is not None:
            atomic_json(directory / 'final_state.json', runtime.state_view())
        with (directory / 'memory.jsonl').open('w') as handle:
            for event in runtime.memory.events:
                handle.write(json.dumps(event,allow_nan=False)+'\n')
                if event['event']=='progress_checked':append('progress',event)
                if event['event']=='safety_rejected':append('safety',event)
