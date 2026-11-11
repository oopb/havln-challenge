"""Phase 2 test-6 regression: scene goal survives camera turns and absent people.

Focused offline tests. No Habitat simulator or external API is required.
The test scenario mirrors the observed failure: model had already entered
an instruction-matching bedroom but navigation_intent had expired after
camera turns; subsequent absent-reader views led it through an open exit.
"""
import copy
import json
import unittest

from jsonschema import Draft7Validator
from havln_agent.contexts import compact_navigation_context
from havln_agent.contracts import navigation_schema,registry
from havln_agent.memory_records import MemoryRecords
from havln_agent.mission_history import mission_evidence
from havln_agent.model_angles import project_model_angles
from havln_agent.navigation_agent import MISSION_PROMPT
from havln_agent.visual_perception import SCAN_GUIDANCE_PROMPT
from test_forward_preflight_v9 import camera
from test_v23_route_evidence import fixture


class Phase2SceneGoalContinuity(unittest.TestCase):
    SCENE='Inspect the bedroom entered via the instructed left stair-exit route.'

    def make_scene(self):
        m=fixture()
        m.task.instruction=(
            'Descend the stairs, turn left, enter the bedroom, observe '
            'the magazine reader, and stop out of their path.')
        m.current=next(o for o in m.observations
                       if o['observation_id']=='obs_0019')
        m.note_navigation_assessment(dict(
            progress_narrative='Entered the instruction-left bedroom.',
            navigation_intent='Turn toward seating on the current right.',
            scene_goal=self.SCENE))
        return m

    def test_new_goal_is_valid_optional_nonexecuting_decision_metadata(self):
        tools=registry(camera(),100)
        decision=dict(
            observed_evidence=['Current RGB shows a bedroom layout.'],
            scene_goal=self.SCENE,
            decision_analysis='Explore the room without assuming a person is stationary.',
            tool='turn_and_observe',
            args=dict(turn_direction='RIGHT',turn_degrees=45,look='none'))
        self.assertTrue(Draft7Validator(navigation_schema(tools)).is_valid(decision))
        decision['tool']='navigate_relative'
        decision['args']=dict(turn_direction='RIGHT',
                              turn_degrees=90,distance_m=.5)
        # The schema does not impose a ban on turning around/backtracking.
        self.assertTrue(Draft7Validator(navigation_schema(tools)).is_valid(decision))
        self.assertNotIn('scene_goal',tools['navigate_relative']['parameters']['properties'])

    def test_scene_goal_survives_stale_camera_intent_and_compaction(self):
        m=self.make_scene()
        source=m.task.scene_goal_source_observation_id
        self.assertEqual(source,'obs_0019')
        self.assertEqual(m.task.scene_goal,self.SCENE)
        # Robot turns and moves to a different viewpoint: immediate steering
        # intent becomes historical but the semantic destination remains.
        m.current=next(o for o in m.observations
                       if o['observation_id']=='obs_0052')
        full=mission_evidence(m,m.current['pose'])
        self.assertIsNone(full['progress']['navigation_intent'])
        self.assertTrue(full['progress']['prior_navigation_intent']['historical_only'])
        self.assertEqual(full['progress']['scene_goal'],self.SCENE)
        self.assertEqual(full['progress']['scene_goal_source_observation_id'],source)
        self.assertTrue(full['progress']['scene_goal_is_model_selected_not_verified'])
        untouched=copy.deepcopy(full)
        compact=compact_navigation_context(dict(memory=full))['memory']
        self.assertEqual(full,untouched)
        self.assertEqual(compact['progress']['scene_goal'],self.SCENE)
        rebased=project_model_angles(dict(memory=compact),45)
        self.assertEqual(rebased['memory']['progress']['scene_goal'],self.SCENE)
        self.assertNotIn('source_heading_deg',json.dumps(rebased['memory']['progress']))

    def test_missing_reader_does_not_automatically_replace_scene_goal(self):
        m=self.make_scene()
        m.current=next(o for o in m.observations
                       if o['observation_id']=='obs_0052')
        m.note_navigation_assessment(dict(
            progress_narrative='No reader visible at this moment; the bedroom remains the route candidate.'))
        self.assertEqual(m.task.scene_goal,self.SCENE)
        self.assertEqual(m.task.scene_goal_source_observation_id,'obs_0019')
        # Rewriting the same goal does not spam history or re-anchor it.
        old_length=len(m.task.progress_history)
        m.note_navigation_assessment(dict(scene_goal=self.SCENE))
        self.assertEqual(len(m.task.progress_history),old_length)
        self.assertEqual(m.task.scene_goal_source_observation_id,'obs_0019')

    def test_explicit_evidence_based_replanning_remains_possible(self):
        m=self.make_scene()
        m.current=next(o for o in m.observations
                       if o['observation_id']=='obs_0052')
        new='Inspect the other room because its fixed entrance matches the instructed left branch.'
        result=m.note_navigation_assessment(dict(scene_goal=new,
            progress_narrative='New spatial route evidence contradicts prior entrance.'))
        self.assertTrue(result['changed'])
        self.assertEqual(m.task.scene_goal,new)
        self.assertEqual(m.task.scene_goal_source_observation_id,'obs_0052')
        self.assertEqual(m.snapshot()['progress_state']['scene_goal'],new)

    def test_unobserved_source_cannot_create_grounded_scene_goal(self):
        m=MemoryRecords(camera())
        m.reset('empty','Find the room')
        m.note_navigation_assessment(dict(scene_goal='Imagined room target'))
        self.assertEqual(m.task.scene_goal,'')
        self.assertEqual(m.task.scene_goal_source_observation_id,'')

    def test_prompt_prioritizes_spatial_goal_without_disabling_backtracking(self):
        for phrase in ('scene_goal','Before leaving for a different entrance',
                       'positive spatial evidence','target NOT_VISIBLE is uncertainty',
                       'favor a short model-chosen forward exploration'):
            self.assertIn(phrase,MISSION_PROMPT)
        self.assertIn('People are dynamic and can move',SCAN_GUIDANCE_PROMPT)
        self.assertIn('Do not recommend leaving',SCAN_GUIDANCE_PROMPT)
        self.assertNotIn('must never turn around',MISSION_PROMPT)
        self.assertLess(len(MISSION_PROMPT),5000)
        self.assertLess(len(SCAN_GUIDANCE_PROMPT),7600)


if __name__=='__main__':
    unittest.main()
