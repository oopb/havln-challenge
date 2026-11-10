"""v24 live person buffer; no blanket UNKNOWN veto or stale RGB boxes.

Synthetic public RGB-D; does not require Habitat, external models or GT pose.
"""
import copy
import math
import unittest
from jsonschema import Draft7Validator
from havln_agent.perception import Geometry
from havln_agent.forward_preflight import sense_forward
from havln_agent.contracts import registry, navigation_schema
from test_forward_preflight_v9 import camera,frame,test_runtime,geometry


PERSON=[[.35,.2,.65,.82]]


def actual_depth_geometry(distance_m,boxes=PERSON):
    g=Geometry(frame(distance_m),camera(),.05)
    if boxes:
        g.set_humans(dict(humans=[dict(box=b) for b in boxes],
                          uncertain_humanoids=[]))
    return g


def source():
    return dict(observation_id='obs_live',rgb_hash='rgb_hash',
        depth_hash='depth_hash',pose=dict(heading=0.,
            pitch_deg=0.,motion_epoch=0,viewpoint_generation=0,
            position_coordinates_valid=False,stance_id='0:0'))


class LiveHumanBuffer(unittest.TestCase):
    def test_person_inside_buffer_blocks_even_if_raw_depth_uncertain(self):
        g=actual_depth_geometry(.65)
        result=sense_forward(g,source(),human_margin_m=.45)
        self.assertEqual(result['status'],'BLOCKED')
        self.assertTrue(result['human_buffer_confirmed'])
        self.assertEqual(result['reason'],
                         'measured_current_RGB_person_buffer_intrusion')
        self.assertEqual(result['blocker']['blocker_kind'],
                         'current_RGB_person_depth_buffer')
        self.assertGreater(result['human_forward_clearance'][
            'buffer_support_pixels'],6)
        self.assertFalse(result['center_confirmed'])
        self.assertTrue(result['human_forward_clearance'][
            'human_identity_is_model_visual_hypothesis'])

    def test_no_person_region_does_not_block_conclusive_safety_unknown(self):
        r,actions=test_runtime(changed_forwards=True)
        r.geometry=lambda:geometry('UNCONFIRMED')
        result=r._navigate_relative('NONE',0,.25)
        self.assertEqual(result['status'],'ACTIONS_COMPLETED')
        self.assertEqual(actions,['MOVE_FORWARD'])
        self.assertEqual(r._last_forward_depth_probe['status'],'UNKNOWN')
        self.assertFalse(r._last_forward_depth_probe[
            'human_buffer_confirmed'])

    def test_unmeasured_far_human_is_not_a_block(self):
        g=actual_depth_geometry(4.)
        result=sense_forward(g,source())
        self.assertFalse(result['human_buffer_confirmed'])
        self.assertNotEqual(result['status'],'BLOCKED')

    def test_runtime_only_uses_person_boxes_at_exact_rgb_and_pose(self):
        r,_=test_runtime()
        pose=copy.deepcopy(r.motion_controller.pose())
        r._active_person_hint=dict(boxes=PERSON,
            rgb_hash=r.memory.current['rgb_hash'],
            pose=pose,observation_id=r.memory.current['observation_id'],
            near_person_measured=False)
        self.assertEqual(r.geometry().human_status['confirmed'],1)
        r.motion_controller.heading+=math.radians(15)
        self.assertEqual(r.geometry().human_status['confirmed'],0)

    def test_emergency_person_buffer_veto_before_any_move(self):
        r,actions=test_runtime(changed_forwards=True)
        r.accept_observation(frame(.65))
        r._active_person_hint=dict(boxes=PERSON,
            rgb_hash=r.memory.current['rgb_hash'],
            pose=copy.deepcopy(r.motion_controller.pose()),
            observation_id=r.memory.current['observation_id'],
            near_person_measured=False)
        result=r._prepare_forward()
        self.assertEqual(result['status'],'BLOCKED')
        self.assertIn('person_buffer',result['reason'])
        self.assertEqual(actions,[])
        self.assertEqual(result['safety']['human_buffer_confirmed'],True)
        self.assertFalse(result['safety']['center_confirmed'])

    def test_person_range_prompts_fresh_recheck_without_global_step_cap(self):
        r,actions=test_runtime(changed_forwards=True)
        decision=dict(observation_id=r.memory.current['observation_id'],
            current_rgb_hash=r.memory.current['rgb_hash'],
            visible_people_boxes=PERSON,actor='navigation_agent')
        result=r.execute(dict(tool='navigate_relative',args=dict(
            turn_direction='NONE',turn_degrees=0,distance_m=.75)),decision)
        # One executed step changes RGB-D, so the old person region becomes
        # invalid for the remaining route. Scheduler regains agency.
        self.assertIn(result['status'],('PARTIAL','REQUIRES_ATTENTION'))
        self.assertEqual(actions,['MOVE_FORWARD'])
        self.assertEqual(result.get('interrupted_status','REQUIRES_ATTENTION'),
                         'REQUIRES_ATTENTION')
        self.assertIn('fresh_person_RGBD_required_after_forward',
                      result.get('reason',''))

    def test_turn_with_person_boxes_reobserves_before_forward(self):
        r,actions=test_runtime(changed_forwards=True)
        decision=dict(observation_id=r.memory.current['observation_id'],
            current_rgb_hash=r.memory.current['rgb_hash'],
            visible_people_boxes=PERSON,actor='navigation_agent')
        result=r.execute(dict(tool='navigate_relative',args=dict(
            turn_direction='LEFT',turn_degrees=15,distance_m=.25)),decision)
        self.assertEqual(actions,['TURN_LEFT'])
        self.assertEqual(result['status'],'PARTIAL')
        self.assertEqual(result['interrupted_status'],'REQUIRES_ATTENTION')
        self.assertEqual(result['reason'],
                         'person_regions_stale_after_camera_or_motion')
        self.assertTrue(result['human_safety']['no_forward_executed'])

    def test_no_people_hint_keeps_full_continuous_motion(self):
        r,actions=test_runtime(changed_forwards=True)
        result=r._navigate_relative('NONE',0,.75)
        self.assertEqual(result['status'],'ACTIONS_COMPLETED')
        self.assertEqual(actions,['MOVE_FORWARD']*3)

    def test_model_box_schema_has_no_global_yaw_actuator(self):
        reg=registry(camera(),10)
        schema=navigation_schema(reg)
        Draft7Validator.check_schema(schema)
        data=dict(observed_evidence=['A person is ahead'],
            decision_analysis='Move relative left.',
            visible_people_boxes=PERSON,
            tool='navigate_relative',args=dict(
                turn_direction='LEFT',turn_degrees=15,distance_m=.25))
        self.assertTrue(Draft7Validator(schema).is_valid(data))
        data['args']=dict(target_heading_deg=120,distance_m=.25)
        self.assertFalse(Draft7Validator(schema).is_valid(data))


if __name__=='__main__':
    unittest.main()
