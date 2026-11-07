"""Observation Agent: coherent RGB-D camera exploration for an entire mission.

All VLM turns share a single dialogue of real observed images and replies.
NavigationAgent alone decides which finite route to execute next. No semantic
milestone validator, retrospective reopen, or automatic waypoint authorization.
"""
import copy
import math
import time
from .api_client import APIError,BudgetExceeded
from .heading import (box_center_bearing, direction_from_bearing,
                      relative_bearing, same_local_stance)

def _angle_diff(a,b):
    return round((a-b+180)%360-180,2)

class ObservationController:
    def __init__(self,runtime):
        self.runtime=runtime

    def scan_mission(self,objective,sweep_degrees=180):
        r=self.runtime
        if not isinstance(objective,str) or not objective.strip():
            return dict(status='INVALID_ARGUMENTS',
                        reason='observation_needs_natural_language_objective')
        if sweep_degrees not in (30,60,90,120,180,270,360):
            return dict(status='INVALID_ARGUMENTS',reason='invalid_sweep_degrees')
        m=r.memory
        pose=r.motion_controller.pose()
        start_id=m.current['observation_id']
        record=dict(observation_task_id='observation_%03d'%len(m.observation_searches),
            mode='mission_vlm',objective=objective.strip(),
            mission_instruction=m.task.instruction,
            progress_at_start=m.task.progress_narrative,
            intent_at_start=m.task.navigation_intent,
            start_observation_id=start_id,start_pose=copy.deepcopy(pose),
            started_monotonic=time.monotonic(),outcome='IN_PROGRESS',
            checkpoints=[],actions=[],waypoints=[])
        m.observation_searches.append(record)
        session=[]  # Real RGB-D and ALL prior VLM responses within this call.
        seen=set()
        viewed=[]
        waypoints=[]
        turned=0
        last_visual={}
        last_failure=None
        # One model request per actual viewpoint. A bounded scan is a
        # resource safeguard, not a task/route success threshold.
        call_limit=min(26,2+math.ceil(sweep_degrees/max(r.camera.turn_angle,1)))
        for round_index in range(call_limit):
            if r.done or len(m.actions)>=r.config['max_steps']-1:
                last_failure='action_budget_or_environment_done'
                break
            source=r.source()
            here=r.motion_controller.pose()
            heading=math.degrees(here['heading'])%360
            key=(source['rgb_hash'],source['depth_hash'],
                 round(heading,1),round(r.pitch_deg,1),
                 here.get('motion_epoch',0))
            if key in seen:
                last_failure='previously_inspected_identical_RGBD_view'
                break
            seen.add(key)
            current=dict(observation_id=source['observation_id'],
                heading_deg=round(heading,2),pitch_deg=r.pitch_deg)
            try:
                visual=r.visual_perception.guide_scan_view(
                    r.observation,source,objective,['complete mission route'],
                    view_history=viewed,
                    remaining_turn_deg=max(0,sweep_degrees-turned),
                    session=session,mode='APPROACH',
                    observed_yaw_pitch=[dict(heading_deg=p['heading_deg'],
                                             pitch_deg=p['pitch_deg'])
                                        for p in viewed[-12:]],
                    # Both agents receive the same mission_evidence view
                    # inside guide_scan_view; no duplicate stale route digest.
                    route_history=None,geometry_evidence=None)
            except (APIError,BudgetExceeded) as error:
                last_failure=str(error)[:220]
                break
            last_visual=visual
            current['visual']=copy.deepcopy(visual)
            viewed.append(current)
            record['checkpoints'].append(current)
            evidence=(visual.get('observation_summary') or
                      visual.get('evidence') or '').strip()
            if evidence and source['observation_id'] in m.keyframes:
                m.note_semantics(source,[evidence[:350]])
            # Waypoint is a source-frame VISUAL BEARING, not the camera yaw.
            # Do not promote a cue without a localized box to a motion target.
            target_box=visual.get('box')
            if (visual.get('candidate_approach') is True
                    and visual.get('target_visible') is True
                    and target_box is not None):
                try:
                    image_offset=box_center_bearing(target_box,r.camera.hfov)
                except ValueError:
                    image_offset=None
                if image_offset is not None:
                    goal_heading=(heading+image_offset)%360
                    proposal=dict(
                        source_observation_id=source['observation_id'],
                        source_pose=copy.deepcopy(here),
                        heading_at_capture_deg=round(heading,2),
                        pitch_at_capture_deg=r.pitch_deg,
                        target_box_center_offset_deg=image_offset,
                        absolute_heading_deg=round(goal_heading,2),
                        bearing_source='source_RGB_target_box_center_and_calibrated_HFOV',
                        relative_bearing_deg_from_current=None,
                        distance_m_hint=visual.get('distance_m_hint'),
                        confidence=visual.get('confidence','low'),
                        target_type=visual.get('target_type'),
                        target_region=copy.deepcopy(target_box),
                        observation_summary=evidence[:500],
                        reason=(visual.get('navigation_suggestion') or
                                visual.get('reason') or '')[:500],
                        uncertainty=(visual.get('uncertainty') or '')[:300],
                        traversability='UNCONFIRMED',
                        motion_authorized=False)
                    waypoints.append(proposal)
            action=visual.get('next_action','finish')
            if action=='finish':
                record['outcome']='OBSERVATION_COMPLETED'
                break
            if action in ('look_up','look_down'):
                delta=r.camera.tilt_angle if action=='look_up' else -r.camera.tilt_angle
                if abs(r.pitch_deg+delta)>60:
                    last_failure='camera_pitch_limit'
                    break
                atomic=['LOOK_UP' if action=='look_up' else 'LOOK_DOWN']
            elif action in ('turn_left','turn_right'):
                degrees=visual.get('turn_deg',0)
                if (not isinstance(degrees,(int,float)) or degrees<=0
                        or degrees%r.camera.turn_angle!=0
                        or turned+degrees>sweep_degrees):
                    last_failure='bounded_scan_turn_exceeded'
                    break
                count=int(round(degrees/r.camera.turn_angle))
                atomic=[('TURN_LEFT' if action=='turn_left' else 'TURN_RIGHT')]*count
            else:
                last_failure='invalid_vlm_camera_action'
                break
            interrupted=False
            for a in atomic:
                response=r._step(a)
                if response.get('status')!='EXECUTED':
                    last_failure=response.get('reason') or response.get('status')
                    interrupted=True
                    break
                record['actions'].append(a)
                if a in ('TURN_LEFT','TURN_RIGHT'):
                    turned+=r.camera.turn_angle
            if interrupted:
                break
        else:
            last_failure='observation_checkpoint_limit'

        final_pose=r.motion_controller.pose()
        final_yaw=math.degrees(final_pose['heading'])%360
        for waypoint in waypoints:
            fresh=same_local_stance(waypoint['source_pose'],final_pose)
            waypoint['source_stance_matches_current']=fresh
            delta=(relative_bearing(waypoint['absolute_heading_deg'],final_yaw)
                   if fresh else None)
            waypoint['relative_bearing_deg_from_current']=delta
            if delta is not None:
                side,degrees=direction_from_bearing(delta)
                quantized=round(degrees/r.camera.turn_angle)*r.camera.turn_angle
                waypoint['recommended_turn_direction']=(side if quantized else 'NONE')
                waypoint['recommended_turn_degrees']=quantized
            waypoint['source_pose_epoch']=waypoint['source_pose'].get('motion_epoch')
        summaries=[(p['visual'].get('observation_summary') or
                    p['visual'].get('evidence') or '').strip()
                   for p in viewed]
        summary=(last_visual.get('observation_summary') or
                 last_visual.get('evidence') or
                 '; '.join(s for s in summaries[-3:] if s) or
                 'No reliable scene description obtained.')
        suggestion=(last_visual.get('navigation_suggestion') or
                    last_visual.get('reason') or
                    'NavigationAgent should decide using current RGB and recorded views.')
        if record['outcome']=='IN_PROGRESS':
            record['outcome']='PARTIAL_OBSERVATION' if viewed else 'VISION_UNAVAILABLE'
        record.update(
            end_observation_id=m.current['observation_id'],
            covered_turn_deg=turned,
            observation_summary=summary[:750],
            navigation_suggestion=suggestion[:600],
            waypoints=copy.deepcopy(waypoints),
            evidence_observation_ids=[v['observation_id'] for v in viewed],
            session_rounds=len(session)//2,
            failure_reason=last_failure)
        m.note_observation_evidence(
            record['evidence_observation_ids'],objective=objective)
        r.event('mission_observation_completed',source='observation_agent',
                outcome=record['outcome'],views=len(viewed),
                waypoints=len(waypoints),covered_turn_deg=turned)
        return dict(
            status=('OK' if viewed else 'PARTIAL'),
            outcome=record['outcome'],objective=objective,
            observation_summary=record['observation_summary'],
            navigation_suggestion=record['navigation_suggestion'],
            waypoints=copy.deepcopy(waypoints),
            visual_reports=copy.deepcopy(viewed[-8:]),
            evidence_observation_ids=record['evidence_observation_ids'],
            covered_turn_deg=turned,session_rounds=record['session_rounds'],
            reason=last_failure,
            next_decision_owner='navigation_agent',
            motion_authorized=False)
