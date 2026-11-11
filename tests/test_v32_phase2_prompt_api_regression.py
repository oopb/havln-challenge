"""Phase2-test-5 regression: verify all 12 raw VLM coordinate failures.

Tests operate offline without Habitat, model calls or API secrets. The
region arrays are copied from the failed wire replies in the diagnostic ZIP;
they are source values, not model outputs invented to fit the patch.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from jsonschema import Draft7Validator
from havln_agent.api_client import APIClient, APIError
from havln_agent.vision_protocol import coordinate_schema, normalize_coordinates
from havln_agent.visual_perception import SCAN_GUIDANCE_SCHEMA, SCAN_GUIDANCE_PROMPT
from havln_agent.navigation_agent import MISSION_PROMPT
from test_api_parameter_compat import settings, accepted


# phase2-test-5: test-2 (8 failures) + test-3 (4 failures).
FAILED_WIRE_BOXES = (
    (None, [510, 600, 785, 820]),
    (None, [505, 668, 723, 864]),
    ([924, 473, 995, 661], [500, 683, 931, 945]),
    ([922, 460, 995, 638], [473, 687, 915, 986]),
    ([817, 461, 892, 640], [705, 700, 895, 810]),
    (None, [700, 665, 935, 775]),
    (None, [145, 675, 960, 970]),
    (None, [160, 670, 930, 970]),
    ([268, 0, 460, 995], [475, 835, 770, 980]),
    ([0, 0, 450, 1000], [460, 817, 772, 995]),
    (None, [0, 696, 222, 928]),
    (None, [0, 700, 222, 950]),
)


class Phase2PromptApiRegression(unittest.TestCase):
    def test_early_navigation_and_visual_prompt_are_restored(self):
        self.assertIn('Complete the ORIGINAL full navigation instruction', MISSION_PROMPT)
        self.assertIn('favor a short model-chosen forward exploration', MISSION_PROMPT)
        self.assertIn('Match instruction landmarks to observed scene geometry', MISSION_PROMPT)
        self.assertIn('people can move, appear or leave', MISSION_PROMPT)
        self.assertIn('route_reference_update', MISSION_PROMPT)
        self.assertNotIn('First carefully interpret the COMPLETE instruction', MISSION_PROMPT)
        self.assertLess(len(MISSION_PROMPT), 5000)
        self.assertIn('People are dynamic and can move', SCAN_GUIDANCE_PROMPT)
        self.assertIn('room geometry and floor connections', SCAN_GUIDANCE_PROMPT)
        self.assertIn('the last step', SCAN_GUIDANCE_PROMPT)
        self.assertLess(len(SCAN_GUIDANCE_PROMPT), 7600)

    @staticmethod
    def payload(box, passage_box):
        return dict(observation_summary='Doorway and room floor are visible.',
            evidence='Furniture and a doorframe visible; no reader confirmed.',
            target_visible=False, target_type='person reading a magazine',
            target_kind='UNKNOWN', confidence='medium', box=box,
            passage_box=passage_box,
            uncertainty='A moving person may not be visible in one frame.',
            navigation_suggestion='Proceed based on the room and passage layout.',
            reason='The visible doorframe is the current scene evidence.',
            expected_progress='UNKNOWN', candidate_approach=False,
            next_action='finish', turn_deg=0,
            coordinate_space='image_1000_xyxy')

    @staticmethod
    def make_client(path, raw):
        opts=settings()
        opts['temperature']=None
        client=APIClient({'navigation_agent':opts,
                          'visual_perception':opts},path)
        client.begin_episode('phase2-test-5-regression',8,
                             Path(path)/'api.jsonl')
        client._request=Mock(return_value=accepted(json.dumps(raw)))
        return client

    @staticmethod
    def ask(client):
        return client.complete(
            'visual_perception', 'vlm_guided_observation',
            'Return the requested JSON', {},
            coordinate_schema(SCAN_GUIDANCE_SCHEMA,'image_1000_xyxy'),
            validation_schema=coordinate_schema(
                SCAN_GUIDANCE_SCHEMA, 'image_1000_xyxy',
                allow_missing_unit=True),
            normalize=lambda value:normalize_coordinates(
                value,SCAN_GUIDANCE_SCHEMA,
                default_unit='image_1000_xyxy'),
            result_schema=SCAN_GUIDANCE_SCHEMA)

    def test_all_12_real_failed_wire_shapes_are_accepted_once(self):
        self.assertEqual(len(FAILED_WIRE_BOXES),12)
        with patch.dict('os.environ',{'COMPAT_TEST_KEY':'offline-only'}):
            for box, passage in FAILED_WIRE_BOXES:
                with self.subTest(box=box,passage=passage):
                    raw=self.payload(box,passage)
                    original=copy.deepcopy(raw)
                    with tempfile.TemporaryDirectory() as tmp:
                        c=self.make_client(tmp,raw)
                        result=self.ask(c)
                        self.assertEqual(c.calls,1)
                        self.assertEqual(c._request.call_count,1)
                        self.assertEqual(raw,original)
                        self.assertNotIn('coordinate_space',result)
                        self.assertEqual(result['box'],
                             None if box is None else [v/1000 for v in box])
                        self.assertEqual(result['passage_box'],
                                         [v/1000 for v in passage])
                        self.assertTrue(Draft7Validator(
                            SCAN_GUIDANCE_SCHEMA).is_valid(result))

    def test_invalid_wire_units_still_fail_before_normalization(self):
        with patch.dict('os.environ',{'COMPAT_TEST_KEY':'offline-only'}):
            for passage in (
                [.51,.60,.78,.82],  # normalized floats in integer wire mode
                [1001,50,1100,400],  # illegal thousand-scale image bounds
                [700,500,200,600],  # inverted rectangle
            ):
                with self.subTest(passage=passage):
                    with tempfile.TemporaryDirectory() as tmp:
                        c=self.make_client(tmp,self.payload(None,passage))
                        with self.assertRaises(APIError):
                            self.ask(c)
                        self.assertEqual(c.calls,1)


if __name__=='__main__':
    unittest.main()
