"""Evidence fusion for language goals; LLM claims never create sensor facts."""

import math
import re
import time

from .memory_records import task_tokens
from .spatial import angle_delta, goal_estimate
from .motion import registered_action


def stair_motion_direction(goal):
    """Classify only an explicit stair movement, independent of LLM wording.

    The same classification drives planning criteria, motion policy and progress
    checks. A 'descent' milestone must not silently become a LEVEL_ONLY route.
    """
    action = str(goal.get('action', ''))
    if re.search(r'\b(?:descend\w*|descent|downstairs)\b|\b(?:go|walk|move|head)\s+down\s+(?:the\s+)?stairs\b|下楼|下台阶', action, re.I):
        return 'DOWN'
    if re.search(r'\b(?:ascend\w*|ascent|upstairs)\b|\b(?:go|walk|move|head)\s+up\s+(?:the\s+)?stairs\b|上楼|上台阶', action, re.I):
        return 'UP'
    return None


def turn_requirement(goal):
    """Derive an orientation-only requirement from the existing language action."""
    text = goal.get('action', '').lower()
    text = re.sub(r'\brightward\b','right',text)
    text = re.sub(r'\bleftward\b','left',text)
    if not re.search(r'\b(turn|rotate|orient|orientation|reorient|face|facing|heading)\b|转身|掉头|左转|右转|调整朝向', text):
        return None
    if re.search(r'\b(walk|proceed|enter|exit|pass|continue|descend|ascend|climb|advance|go|move|follow)\b|然后.*走', text):
        return None
    around = bool(re.search(r'\baround\b|opposite|reverse|转身|掉头', text))
    explicit = re.search(r'(\d+(?:\.\d+)?)\s*(?:degrees?|°|度)', text)
    if not around and not explicit and not re.search(r'\bleft\b|\bright\b|左|右', text):
        return None
    angle = float(explicit.group(1)) if explicit else 180 if around else 90
    if not 0 < angle <= 180:
        return None
    direction = 'right' if re.search(r'\bright\b|右', text) else 'left' if re.search(r'\bleft\b|左', text) else 'either'
    return dict(angle_deg=angle, direction=direction)


def requires_landmark_crossing(goal):
    """A route *through space until a target* need not pass that target.

    Crossing is a semantic requirement only for a stated boundary transition;
    generic path phrasing must not invent a landmark_passed predicate.
    """
    text = str(goal.get('action', ''))
    explicit = bool(re.search(r'\b(?:enter|entry|pass|cross)\b|穿过|经过|进入', text, re.I))
    if explicit:
        return True
    if re.search(r'\bthrough\b', text, re.I):
        return not bool(re.search(r'\b(?:until|to reach|to arrive|arriv(?:e|al) at)\b', text, re.I))
    return False


def turn_evidence(memory):
    task = memory.task
    goal = task.subgoals[task.current_subgoal_index] if task.current_subgoal_index < len(task.subgoals) else {}
    requirement = turn_requirement(goal)
    actions = [a for a in memory.actions if a['step'] >= task.subgoal_started]
    signed = sum((1 if a['action'] == 'TURN_LEFT' else -1 if a['action'] == 'TURN_RIGHT' else 0)
                 * memory.camera.turn_angle for a in actions)
    degrees = (signed + 180) % 360 - 180
    complete = False
    remaining, directions = None, []
    if requirement:
        expected = requirement['angle_deg']
        desired = -expected if requirement['direction'] == 'right' else expected
        delta = (desired-degrees+180) % 360-180
        remaining = abs(delta)
        directions = ['left', 'right'] if math.isclose(remaining, 180) else ['left' if delta > 0 else 'right']
        error = min(abs(degrees-expected), abs(degrees+expected)) if requirement['direction'] == 'either' else \
                abs((degrees-(expected if requirement['direction'] == 'left' else -expected)+180) % 360-180)
        complete = error < memory.camera.turn_angle/2 and any(
            a['action'].startswith('TURN_') and a['result'].get('visual_change') for a in actions)
    return dict(requirement=requirement, net_turn_deg=degrees, completed=complete,
                remaining_deg=remaining, allowed_directions=directions,
                turn_steps=[a['step'] for a in actions if a['action'].startswith('TURN_')],
                source='accepted_turn_receipts_and_public_view_change', confidence='estimated_not_gt_yaw')


def current_visual_evidence(source, memory, now, max_age):
    if source.get('image_scope', 'current') != 'current':
        return False
    current = memory.current
    same = (source['pose'] == current['pose'] and source['rgb_hash'] == current['rgb_hash']
            and source.get('depth_hash') == current.get('depth_hash'))
    # Re-reading identical public RGB-D reconfirms static visual content, not
    # historical human locations. Retain the original source timestamp.
    return same and now - current['last_read_at'] <= max_age


def _passed_landmark(memory, matches, motions):
    """Require a tracked landmark to cross behind the registered route direction."""
    observations = {o['observation_id']: o for o in memory.observations}
    for current in matches:
        if current['position']['status'] != 'LOCALIZED':
            continue
        for origin in memory.groundings:
            if origin['status'] != 'LOCALIZED' or origin['position']['status'] != 'LOCALIZED' \
                    or origin['target'].lower() != current['target'].lower() \
                    or origin['source']['subgoal'] != memory.task.current_subgoal_index:
                continue
            start = observations[origin['source']['observation_id']]['step']
            heading = origin['source']['pose']['heading'] + math.radians(origin['position']['bearing_degrees'])
            travel, uncertainty = 0.0, 0.0
            for action in motions:
                if action['step'] < start:
                    continue
                delta = angle_delta(action['pose']['heading'], heading)
                measurement = action['result']['rgbd_motion']
                translation = measurement['translation_m']
                travel += translation['z_forward'] * math.cos(delta) - translation['x_right'] * math.sin(delta)
                uncertainty += measurement['uncertainty_m']
            distance = math.hypot(origin['position']['x_right'], origin['position']['z_forward'])
            bearing = current['source']['pose']['heading'] + math.radians(current['position']['bearing_degrees'])
            behind = math.cos(angle_delta(bearing, heading)) < -0.2
            if behind and travel > distance + uncertainty + memory.camera.robot_radius:
                return True
    return False


def _verified_stair_transition(directed, landing_steps, landings, displacement,
                                uncertainty, travel, travel_error, continuous,
                                contradictions, forward_step):
    """Independent measured alternative to a precisely localized entry point.

    A correct short descent can begin *on* a stair without ever obtaining an
    entry box. Never infer the transition from language or commanded actions:
    require multiple registered descending translations, multiple registered
    horizontal landing actions, current RGB-D-supported landing appearance,
    sufficient net displacement, continuity and zero contradictions.
    """
    return bool(
        len(directed) >= 2 and
        len({a['result']['observation_after'] for a in landing_steps}) >= 2 and
        landings and continuous and not contradictions and
        displacement - uncertainty >= max(0.20, forward_step) and
        travel - travel_error >= 2 * forward_step
    )


def _stair_evidence(memory, goal, ids, now, max_age, arrival_distance):
    down = stair_motion_direction(goal) == 'DOWN'
    direction, sign = ('down', -1) if down else ('up', 1)
    observed = {o['observation_id']: o for o in memory.observations}
    wanted = task_tokens(goal['landmark'] + ' ' + goal['target'])
    groundings = [g for g in memory.groundings if g['status'] in ('LOCALIZED','PARTIAL')
                  and g['source']['subgoal'] == memory.task.current_subgoal_index
                  and wanted & task_tokens(g['target']) and g['semantic']['confidence'] != 'low']
    origins = [g for g in groundings if g['semantic']['stair_direction'] in (direction, 'unknown')
               and g.get('semantic_presence', 'VISIBLE') == 'VISIBLE'
               and (g['position']['status'] == 'LOCALIZED' or g['entry']['status'] == 'LOCALIZED')
               and observed[g['source']['observation_id']]['step'] >= memory.task.subgoal_started
               and g['source']['pose'].get('motion_epoch',0)==memory.current['pose'].get('motion_epoch',0)]
    supported, missing, contradictions = [], [], []
    details = dict(source_observation_ids=[], motion_steps=[], direction=direction)
    details['confirmed_phases'] = []
    visible = [g for g in memory.groundings if g['source']['subgoal'] == memory.task.current_subgoal_index
               and wanted & task_tokens(g['target']) and g['semantic'].get('target_visible')
               and g.get('semantic_presence', 'VISIBLE') == 'VISIBLE']
    if visible:
        supported.append('staircase_observed')
        details['confirmed_phases'].append('SEE_STAIRS')
    details['registration_limitations'] = sorted({
        a['result']['rgbd_motion'].get('reason', 'registration_unknown')
        for a in memory.actions if a['step'] >= memory.task.subgoal_started
        and a['action'] == 'MOVE_FORWARD' and a['result'].get('rgbd_motion')
        and a['result']['rgbd_motion']['status'] == 'UNKNOWN'
    })
    if not origins:
        return 'NOT_STARTED', supported, ['localized_staircase_and_direction'], contradictions, details
    if 'staircase_observed' not in supported:
        supported.append('staircase_observed')
    near = [g for g in origins if g['entry']['status'] == 'LOCALIZED'
            and g['entry'].get('depth_spread_metres', 99) < 0.6
            and math.hypot(g['entry']['x_right'], g['entry']['z_forward']) <= arrival_distance]
    origin = near[0] if near else origins[-1]
    if 'visual_direction_conflicts_with_tread_depth' in origin.get('stair_evidence', {}).get('direction_rejection_reasons', []):
        contradictions.append('stair_direction_geometry_conflicts')
    details['source_observation_ids'].append(origin['source']['observation_id'])
    origin_step = observed[origin['source']['observation_id']]['step']
    entry = origin['entry']
    near_entry = (entry['status'] == 'LOCALIZED' and entry.get('depth_spread_metres', 99) < 0.6
                  and math.hypot(entry['x_right'], entry['z_forward']) <= arrival_distance)
    if near_entry:
        supported.append('localized_nearby_stair_entry')
        details['confirmed_phases'].append('FIND_STAIR_ENTRY')
    else:
        missing.append('unambiguous_nearby_stair_entry')
    heading = origin['source']['pose']['heading'] + math.radians(
        entry.get('bearing_degrees', origin['position'].get('bearing_degrees', 0)))
    motions = [a for a in memory.actions if a['step'] >= origin_step and a['action'] == 'MOVE_FORWARD'
               and registered_action(a, observed)
               and abs(angle_delta(a['pose']['heading'], heading)) <= math.radians(45)]
    forward_window=[a for a in memory.actions if motions and origin_step<=a['step']<=motions[-1]['step']
                    and a['action']=='MOVE_FORWARD']
    continuous=bool(motions) and [a['step'] for a in forward_window]==[a['step'] for a in motions]
    # TURN/LOOK and same-position RGB-D refreshes may lie between registered
    # steps. They change observations, never supply translation evidence.
    continuous=continuous and all(
        observed[left['result']['observation_after']]['pose']['x']==observed[right['result']['observation_before']]['pose']['x']
        and observed[left['result']['observation_after']]['pose']['z']==observed[right['result']['observation_before']]['pose']['z']
        for left,right in zip(motions,motions[1:]))
    if motions and any(a['action'] == 'MOVE_FORWARD' and a['step'] > motions[-1]['step'] for a in memory.actions):
        continuous = False
    details['continuous_rgbd_motion_chain'] = continuous
    if motions and not continuous:
        missing.append('continuous_registered_entry_to_landing_chain')
    directed = [a for a in motions if sign * a['result']['rgbd_motion']['translation_m']['y_up'] >
                a['result']['rgbd_motion']['uncertainty_m']]
    opposite = [a for a in motions if sign * a['result']['rgbd_motion']['translation_m']['y_up'] <
                -a['result']['rgbd_motion']['uncertainty_m']]
    if opposite:
        contradictions.append('registered_motion_opposes_stair_instruction')
    if motions and near_entry:
        supported.append('registered_movement_along_stair_direction')
        details['confirmed_phases'].append('APPROACHING_STAIRS')
    displacement = sum(sign * a['result']['rgbd_motion']['translation_m']['y_up'] for a in directed)
    uncertainty = sum(a['result']['rgbd_motion']['uncertainty_m'] for a in directed)
    details.update(relative_vertical_estimate_m=displacement, uncertainty_bound_m=uncertainty,
                   motion_steps=[a['step'] for a in motions],
                   vertical_motion_steps=[a['step'] for a in directed], absolute_height_available=False)
    descent = bool(visible) and len(directed) >= 2 and displacement >= max(0.12, 2 * uncertainty)
    if descent:
        supported.append('multi_frame_registered_' + direction + '_motion')
        if not contradictions:
            details['confirmed_phases'].extend((['DESCENDING_STAIR_DIRECTION'] if down else []) + ['TRAVERSING_STAIRS'])
    else:
        missing.append('multi_frame_registered_' + direction + '_motion')
    last_vertical = max((a['step'] for a in directed), default=origin_step)
    landing_steps = [a for a in motions if a['step'] > last_vertical
                     and a['result']['rgbd_motion']['vertical_motion'] == 'LEVEL'
                     and (a['result'].get('landing_support') or {}).get('status') == 'SUPPORTED']
    details['landing_motion_steps'] = [a['step'] for a in landing_steps]
    if len({a['result']['observation_after'] for a in landing_steps}) >= 2:
        supported.append('multi_frame_level_landing_transition')
    else:
        missing.append('multi_frame_level_landing_transition')
    landings = [g for g in groundings if current_visual_evidence(g['source'], memory, now, max_age)
                and g['source']['observation_id'] in ids
                and (g.get('landing_geometry') or {}).get('status') == 'SUPPORTED'
                and g['semantic'].get('landing_box') is not None
                and g['semantic'].get('landing_visible', down and g['semantic']['lower_landing_visible'])]
    if landings:
        supported.extend(['current_landing_visual_region', 'measured_near_landing_plane'])
        details['source_observation_ids'].append(landings[-1]['source']['observation_id'])
    else:
        missing.append('current_grounded_landing_with_depth_support')
    travel = sum(a['result']['rgbd_motion']['translation_m']['z_forward'] *
                 math.cos(angle_delta(a['pose']['heading'], heading)) for a in motions)
    travel_error = sum(a['result']['rgbd_motion']['uncertainty_m'] for a in motions)
    entry_distance = math.hypot(entry.get('x_right', 0), entry.get('z_forward', 0))
    left_entry = near_entry and travel - travel_error > max(memory.camera.forward_step, entry_distance)
    if left_entry and len(landing_steps) >= 2:
        supported.append('registered_exit_from_initial_stair_entry_region')
    else:
        missing.append('registered_exit_from_initial_stair_entry_region')
    details['alternative_visual_spatial_evidence'] = [item for item in supported if item in (
        'localized_nearby_stair_entry', 'current_landing_visual_region', 'measured_near_landing_plane',
        'registered_movement_along_stair_direction', 'registered_exit_from_initial_stair_entry_region')]
    independent_transition = _verified_stair_transition(
        directed, landing_steps, landings, displacement, uncertainty,
        travel, travel_error, continuous, contradictions, memory.camera.forward_step)
    details['independent_stair_to_landing_transition'] = independent_transition
    if independent_transition:
        # Evidence proves the crossing without requiring a fragile *entry*
        # pixel exactly at the original robot's floor elevation.
        supported.extend(('sensor_verified_stair_to_landing_transition',
                          'registered_movement_along_stair_direction'))
        details['confirmed_phases'].extend(('FIND_STAIR_ENTRY','ARRIVED_AT_LANDING'))
        missing = [item for item in missing if item not in (
            'unambiguous_nearby_stair_entry',
            'registered_exit_from_initial_stair_entry_region')]
    elif descent and landings and len(landing_steps) >= 2 and left_entry and continuous and not contradictions:
        details['confirmed_phases'].append('ARRIVED_AT_LANDING')
    if origin['source']['observation_id'] not in ids:
        missing.append('cite_initial_stair_observation')
    stage = 'IN_PROGRESS' if motions and near_entry else 'APPROACHING'
    if descent and landings:
        stage = 'POSSIBLY_COMPLETED'
    if not missing and not contradictions:
        stage = 'COMPLETED'
    return stage, supported, missing, contradictions, details


def verify_progress(memory, claim, evidence, observation_ids, max_age, arrival_distance,
                    static_max_age=300, human_max_age=2):
    task, now = memory.task, time.monotonic()
    result = dict(status='INSUFFICIENT_EVIDENCE', reason=None, subgoal=task.current_subgoal_index,
                  stage='NOT_STARTED', supported=[], missing=[], contradictions=[],
                  completion_confirmed=False, vertical_motion='UNKNOWN', evidence_details={},
                  information_options=[], confirmed_phases=[], confirmation_questions=[])
    if task.current_subgoal_index >= len(task.subgoals):
        result.update(status='REJECTED', reason='all_subgoals_already_completed')
    else:
        goal, ids = task.subgoals[task.current_subgoal_index], set(observation_ids)
        known = {o['observation_id'] for o in memory.observations}
        scene = memory.scene_memory
        fresh_scene = bool(scene and current_visual_evidence(scene['source'], memory, now, max_age))
        wanted = task_tokens(goal['landmark'] + ' ' + goal['target'])
        matches = [g for g in memory.groundings if g['source']['observation_id'] in ids
                   and g['status'] == 'LOCALIZED' and current_visual_evidence(g['source'], memory, now, max_age)
                   and not goal_estimate(g, memory, memory.current['pose'], now, max_age,
                                         static_max_age, human_max_age)['requires_reconfirmation']
                   and wanted & task_tokens(g['target'])]
        estimates = [goal_estimate(g, memory, memory.current['pose'], now, max_age,
                                  static_max_age, human_max_age) for g in matches]
        actions = [a for a in memory.actions if a['step'] >= task.subgoal_started
                   and a['result'].get('visual_change')]
        text = goal['action'].lower()
        stairs = stair_motion_direction(goal) is not None
        turn = turn_evidence(memory)
        turning = turn['requirement'] is not None
        if not ids <= known or not evidence.strip():
            result.update(reason='unknown_observation_or_empty_evidence', missing=['valid_observation_citations'])
        elif stairs:
            stage, supported, missing, contradictions, details = _stair_evidence(
                memory, goal, ids, now, max_age, arrival_distance)
            result.update(stage=stage, supported=supported, missing=missing, contradictions=contradictions,
                          evidence_details=details,
                          confirmed_phases=details.get('confirmed_phases', []),
                          reason='multi_evidence_checks_passed' if stage == 'COMPLETED' else 'stair_evidence_incomplete')
            if any(x.startswith('multi_frame_registered_') for x in supported):
                result['vertical_motion'] = 'DOWN_ESTIMATED' if details['direction'] == 'down' else 'UP_ESTIMATED'
            if claim == 'subgoal_completed' and stage == 'COMPLETED':
                result.update(status='VERIFIED', completion_confirmed=True)
            elif claim != 'subgoal_completed':
                satisfied = (claim == 'landmark_found' and 'staircase_observed' in supported or
                             claim == 'entered' and 'registered_movement_along_stair_direction' in supported or
                             claim == 'bottom_visible' and any(g['semantic']['lower_landing_visible'] for g in matches))
                if satisfied:
                    result.update(status='RECORDED', reason='stage_evidence_recorded')
        elif not fresh_scene:
            result.update(reason='current_visual_evidence_required', missing=['current_scene_evidence'])
        else:
            motion = ([a for a in actions if a['action'].startswith('TURN_')] if turning else
                      [a for a in actions if a['action'] == 'MOVE_FORWARD'
                       and (a['result'].get('rgbd_motion') or {}).get('status') == 'ESTIMATED'])
            result['evidence_details'] = dict(motion_steps=[a['step'] for a in motion],
                                               source_observation_ids=[g['source']['observation_id'] for g in matches])
            if turning:
                result['evidence_details']['orientation'] = turn
            stopping = bool(re.match(r'stop|wait|停', text))
            result['stage'] = 'IN_PROGRESS' if motion else 'APPROACHING' if matches else 'NOT_STARTED'
            result['supported'] = (['current_task_landmark'] if matches else []) + \
                (['relevant_registered_motion' if not turning else 'executed_turn'] if motion else [])
            if not turning and not matches:
                result.update(reason='localized_task_landmark_required', missing=['current_task_grounding'])
            elif claim == 'landmark_found' and matches:
                result.update(status='RECORDED', stage='APPROACHING', reason='localized_landmark_recorded')
            elif not motion and not stopping:
                result.update(reason='relevant_executed_motion_required', missing=['registered_task_motion'])
            elif turning and not turn['completed']:
                result.update(reason='turn_angle_not_yet_supported', missing=['required_orientation_change'])
            elif not turning and requires_landmark_crossing(goal) \
                    and not _passed_landmark(memory, matches, motion):
                result.update(reason='landmark_transition_not_verified', missing=['registered_landmark_crossing'])
            elif not turning and not any(math.hypot(e['x_right_m'], e['z_forward_m'])
                                         <= arrival_distance for e in estimates):
                result.update(reason='final_target_not_grounded_nearby' if task.current_subgoal_index ==
                              len(task.subgoals) - 1 else 'task_target_not_grounded_nearby',
                              missing=['nearby_task_target'])
            elif task.current_subgoal_index == len(task.subgoals) - 1 and not (
                scene['scene']['final_target_visible'] and scene['scene']['final_target_evidence'].strip()
            ):
                result.update(reason='final_arrival_visual_evidence_required', missing=['final_arrival_visual_evidence'])
            elif claim == 'subgoal_completed':
                result.update(status='VERIFIED', stage='COMPLETED', completion_confirmed=True,
                              reason='visual_and_execution_checks_passed')
            else:
                result.update(status='RECORDED', stage='IN_PROGRESS', reason='visual_stage_claim_not_task_completion')
        if claim=='subgoal_completed':
            phases=result.get('confirmed_phases',[])
            observed={o['observation_id']:o for o in memory.observations}
            registered=[a for a in memory.actions if a['step']>=task.subgoal_started
                        and a['action']=='MOVE_FORWARD' and registered_action(a,observed)]
            criteria=goal.get('success_criteria',[])
            arrived='ARRIVED_AT_LANDING' in phases
            facts=dict(
                landmark_visible=bool(matches) or 'staircase_observed' in result['supported'],
                entry_confirmed='FIND_STAIR_ENTRY' in phases,
                actual_descent_observed='multi_frame_registered_down_motion' in result['supported'],
                actual_ascent_observed='multi_frame_registered_up_motion' in result['supported'],
                registered_translation=bool(registered),orientation_completed=turn['completed'],
                landmark_passed=bool(matches and registered and _passed_landmark(memory,matches,registered)),
                target_nearby=arrived or any(math.hypot(e['x_right_m'],e['z_forward_m'])<=arrival_distance for e in estimates),
                final_arrival_visible=bool(fresh_scene and scene['scene']['final_target_visible']
                    and scene['scene']['final_target_evidence'].strip()))
            result['criteria_verdicts']={name:bool(facts.get(name,False)) for name in criteria}
            missing=[name for name in criteria if not facts.get(name,False)]
            if not criteria or missing:
                result.update(status='INSUFFICIENT_EVIDENCE',completion_confirmed=False,stage='IN_PROGRESS',
                    reason='semantic_success_conditions_unverified')
                result['missing']=list(dict.fromkeys(result['missing']+(missing or ['typed_success_criteria'])))
        if result['status'] == 'VERIFIED':
            task.completed_subgoals.append(task.current_subgoal_index)
            task.completion_evidence.append(dict(subgoal=task.current_subgoal_index, evidence=evidence,
                                                observation_ids=observation_ids,
                                                executed_steps=result['evidence_details'].get('motion_steps', []),
                                                verification=result.copy()))
            task.current_subgoal_index += 1
            task.subgoal_started = len(memory.actions)
        elif result['status'] == 'RECORDED':
            task.progress_claims.append(dict(claim=claim, evidence=evidence, observation_ids=observation_ids,
                                            subgoal=task.current_subgoal_index, step=len(memory.actions)))
    result['confirmation_questions'] = (['Confirm evidence for: ' + item for item in result['missing']] +
                                        ['Resolve contradiction: ' + item for item in result['contradictions']])
    if result['missing']:
        result['information_options'] = ['ground_landmark', 'analyze_scene', 'query_task_memory', 'query_spatial_memory']
        if any('motion' in item or 'transition' in item or 'crossing' in item for item in result['missing']):
            result['information_options'].extend(['get_waypoint_candidates', 'navigate_local'])
            result['evidence_details']['motion_evidence_precondition'] = (
                'Only model-requested navigation with a justified path and live Safety approval can '
                'produce motion evidence. Observation or a completion claim alone cannot supply it.')
    memory.progress_stage = ('NOT_STARTED' if result['completion_confirmed'] and
                             task.current_subgoal_index < len(task.subgoals) else result['stage'])
    memory.progress_feedback = result
    memory.event('progress_checked', claim=claim, evidence=evidence, observation_ids=observation_ids, result=result)
    return result
