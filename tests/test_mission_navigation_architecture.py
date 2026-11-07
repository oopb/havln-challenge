"""Whole-mission navigation architecture contract tests.

Run: python -m unittest discover -s tests -p 'test_mission_navigation_architecture.py' -v
No Habitat, API keys, or remote inference required.
"""
import math
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from jsonschema import Draft7Validator

from havln_agent.contracts import registry,navigation_schema
from havln_agent.tool_menu import navigation_tool_menu
from havln_agent.memory_records import MemoryRecords
from havln_agent.navigation_agent import NavigationAgent
from havln_agent.active_observation import ObservationController
from havln_agent.scheduler import run
from havln_agent.runtime import AgentRuntime


def camera():
    return SimpleNamespace(turn_angle=15,tilt_angle=30,forward_step=.25,hfov=90)


def pose():
    return dict(heading=math.pi/2,x=0.,y=0.,z=0.,
                pitch_deg=0.,motion_epoch=0,viewpoint_generation=0)


class MissionContracts(unittest.TestCase):
    def test_runtime_event_accepts_explicit_observation_id_without_collision(self):
        # Reproduces worker-0-b0 episode-1 crash on first NavigationAgent RGB.
        runtime=AgentRuntime.__new__(AgentRuntime)
        runtime.owner=threading.get_ident()
        runtime.memory=SimpleNamespace(episode_id='episode-1',actions=[],
            current={'observation_id':'obs_0000'})
        runtime.phase='NAVIGATING'
        runtime.event_seq=0
        runtime.runtime_event_counts={}
        receipts=[]
        runtime.event_callback=receipts.append
        runtime.event('navigation_current_rgb_attached',
            source='navigation_agent',observation_id='obs_0000',
            rgb_hash='rgb-hash')
        self.assertEqual(receipts[-1]['observation_id'],'obs_0000')
        self.assertNotIn('reported_observation_id',receipts[-1])
        self.assertEqual(receipts[-1]['rgb_hash'],'rgb-hash')
        runtime.event('historical_frame_consulted',
            source='observation_agent',observation_id='obs_older')
        self.assertEqual(receipts[-1]['observation_id'],'obs_0000')
        self.assertEqual(receipts[-1]['reported_observation_id'],'obs_older')
        self.assertEqual(runtime.event_seq,2)

    def test_worker_registry_initializes_without_required_argument_crash(self):
        commands=registry(camera(),3,pitch_deg=0)
        self.assertEqual(set(commands),{
            'observe_for_goal','navigate_relative','turn_and_observe',
            'navigate_local','cancel_local_navigation',
            'query_task_memory','read_memory_record','request_finish'})
        for name,spec in commands.items():
            Draft7Validator.check_schema(spec['parameters'])
        self.assertNotIn('review_completed_subgoal',commands)
        self.assertNotIn('verify_subgoal',commands)
        self.assertNotIn('select_subgoal',commands)
        self.assertNotIn('request_task_review',commands)
        obs=commands['observe_for_goal']['parameters']
        self.assertTrue(Draft7Validator(obs).is_valid({
            'objective':'The stairs are behind us; find the right-hand hall.'}))
        self.assertFalse(Draft7Validator(obs).is_valid({
            'objective':'Find the next room','mode':'VERIFY'}))

    def test_navigation_outputs_narrative_with_one_tool(self):
        tools=navigation_tool_menu(registry(camera(),3))
        schema=navigation_schema(tools)
        Draft7Validator.check_schema(schema)
        message=dict(tool='navigate_relative',
            args=dict(turn_direction='RIGHT',turn_degrees=45,distance_m=.5),
            progress_narrative='I have descended the stairs, but have not located the piano.',
            navigation_intent='Find the hall and cross it toward the piano.',
            uncertainties=['Hall entry not yet localized.'],
            evidence_observation_ids=['obs_0000'])
        self.assertTrue(Draft7Validator(schema).is_valid(message))
        self.assertFalse(Draft7Validator(schema).is_valid({
            'tool':'verify_subgoal','args':{'subgoal_index':0}}))

    def test_memory_progress_is_editable_and_evidence_ids_must_exist(self):
        memory=MemoryRecords(camera())
        memory.reset('episode','Descend the stairs then stop beside the piano.')
        memory.observations=[dict(observation_id='obs_0001')]
        first=memory.note_navigation_assessment(dict(
            progress_narrative='Stairs seem to be finished.',
            navigation_intent='Seek the right-hand hall.',
            evidence_observation_ids=['obs_0001','fabricated']))
        self.assertEqual(first['evidence_observation_ids'],['obs_0001'])
        memory.note_navigation_assessment(dict(
            progress_narrative='The last step was not yet reached.',
            navigation_intent='Finish descending before turning.',
            evidence_observation_ids=['obs_0001']))
        self.assertEqual(memory.task.progress_narrative,
                         'The last step was not yet reached.')
        self.assertEqual(len(memory.task.progress_history),2)
        self.assertEqual(memory.task.evidence_observation_ids,['obs_0001'])
        self.assertFalse(hasattr(memory.task,'current_subgoal_index'))
        self.assertFalse(hasattr(memory.task,'completed_subgoals'))

    def test_route_history_stays_available_across_narrative_revisions(self):
        memory=MemoryRecords(camera())
        memory.reset('episode','Descend and find the piano.')
        for i,dy in enumerate((-.4,.2)):
            before='obs_%04d'%i
            after='obs_%04d'%(i+1)
            memory.actions.append(dict(step=i,action='MOVE_FORWARD',
                pose=pose(),result=dict(
                    observation_before=before,observation_after=after,
                    rgbd_motion=dict(status='ESTIMATED',
                        observed_frame_change=True,
                        translation_m=dict(y_up=dy),
                        uncertainty_m=.03))))
        values=memory.route_progress_summary()['successful_segments']
        self.assertEqual(len(values),1)
        self.assertEqual(values[0]['forward_steps'],2)
        self.assertAlmostEqual(values[0]['observed_height_change_m_est'],-.2)

    def test_observation_returns_narrative_and_optional_local_waypoint(self):
        m=MemoryRecords(camera())
        m.reset('episode','Descend, find right-hand hall, stop at piano.')
        state=pose()
        m.current=dict(observation_id='obs_0000',pose=state)
        m.observations=[dict(observation_id='obs_0000',pose=state)]
        m.keyframes={}
        guidance=Mock(return_value=dict(
            next_action='finish',turn_deg=0,
            observation_summary='A hallway entrance is ahead to the right.',
            navigation_suggestion='Approach the entrance cautiously.',
            candidate_approach=True,target_visible=True,confidence='medium',
            target_type='hall',box=[.7,.3,.9,.7],
            evidence='Hallway visible',uncertainty='Distance uncertain',
            reason='Visible entrance'))
        r=SimpleNamespace(memory=m,camera=camera(),done=False,
            config={'max_steps':88},motion_controller=SimpleNamespace(pose=lambda:state),
            pitch_deg=0,observation=SimpleNamespace(),
            visual_perception=SimpleNamespace(guide_scan_view=guidance),
            source=lambda:dict(observation_id='obs_0000',
                rgb_hash='rgb',depth_hash='depth',pose=state),
            event=Mock())
        out=ObservationController(r).scan_mission(
            'We have descended. Find the right-hand hall to continue.')
        self.assertEqual(out['status'],'OK')
        self.assertLess(out['waypoints'][0]['relative_bearing_deg_from_current'],0)
        self.assertEqual(out['waypoints'][0]['recommended_turn_direction'],'RIGHT')
        self.assertEqual(out['waypoints'][0]['bearing_source'],
            'source_RGB_target_box_center_and_calibrated_HFOV')
        self.assertEqual(m.task.evidence_observation_ids,['obs_0000'])
        self.assertEqual(out['waypoints'][0]['traversability'],'UNCONFIRMED')
        self.assertFalse(out['motion_authorized'])
        self.assertEqual(m.observation_searches[-1]['mode'],'mission_vlm')
        self.assertEqual(m.observation_searches[-1]['session_rounds'],0)

    def test_scheduler_records_navigation_narrative_without_planner(self):
        m=MemoryRecords(camera())
        m.reset('episode','Find the piano.')
        m.current=dict(observation_id='obs_0000')
        m.observations=[dict(observation_id='obs_0000')]
        r=SimpleNamespace(memory=m,config=dict(max_steps=88),
            client=SimpleNamespace(availability=lambda:'available'),
            navigation_agent=SimpleNamespace(decide=lambda feedback:dict(
                tool='request_finish',args=dict(outcome='success',
                    evidence='Piano nearby in current view.'),
                progress_narrative='At the piano.',
                navigation_intent='Stop.')),
            decisions=0,done=False,current_decision_id='navigation_agent_0000',
            refresh=Mock(),decision_id=Mock(),
            event=Mock(),start_mission=Mock(),assess=Mock())
        r.execute=Mock(side_effect=lambda command,binding:(
            setattr(r,'done',True) or dict(tool_call_id='tool_0000',
                tool='request_finish',status='FINISHED')))
        logs=[]
        run(r,lambda name,value:logs.append((name,value)))
        self.assertEqual(r.decisions,1)
        self.assertEqual(r.execute.call_count,1)
        self.assertEqual(m.task.progress_narrative,'At the piano.')
        self.assertEqual({name for name,_ in logs},{'navigation_agent'})
        self.assertTrue(r.start_mission.called)


if __name__=='__main__':
    unittest.main()
