"""Provider-prefaced model JSON must be unambiguous and independently validated."""
import json
import unittest

from jsonschema import Draft7Validator

from havln_agent.api_client import decode_response


class EmbeddedJsonTests(unittest.TestCase):
    def test_real_batch46_navigation_preface_is_recovered(self):
        raw=(
            "The prior route only turned; I will choose a short local segment. "
            '{"tool":"navigate_relative","args":'
            '{"turn_deg":30,"distance_m":0.25},'
            '"memory_delta":[{"finding":"live safety required",'
            '"source_observation_ids":["obs_0044"]}]}'
        )
        value, source=decode_response(raw)
        self.assertEqual(source,'unique_embedded_json_container')
        self.assertEqual(value['tool'],'navigate_relative')
        self.assertEqual(value['args']['turn_deg'],30)
        self.assertEqual(value['memory_delta'][0]['source_observation_ids'],
                         ['obs_0044'])

    def test_strict_json_does_not_normalize(self):
        self.assertEqual(decode_response('{"tool":"inspect_geometry","args":{}}')[1],None)

    def test_reject_multiple_objects(self):
        with self.assertRaises(json.JSONDecodeError):
            decode_response('First {"tool":"inspect_geometry","args":{}} '
                            'second {"tool":"request_finish","args":{}}')

    def test_reject_duplicate_keys(self):
        with self.assertRaises(json.JSONDecodeError):
            decode_response('Preface {"tool":"x","tool":"y","args":{}}')

    def test_reject_incomplete_extra_json(self):
        with self.assertRaises(json.JSONDecodeError):
            decode_response('First {"bad": incomplete; chosen '
                            '{"tool":"inspect_geometry","args":{}}')

    def test_reject_nonfinite_payload(self):
        with self.assertRaises(json.JSONDecodeError):
            decode_response('text {"tool":"navigate_relative",'
                            '"args":{"turn_deg":NaN}}')

    def test_preface_does_not_bypass_validation(self):
        # The parser ONLY extracts the JSON payload. Independent schema
        # validation must still reject undeclared tools/unsafe args.
        value, _=decode_response('Prose {"tool":"unregistered","args":{}}')
        schema=dict(type='object',properties=dict(
            tool=dict(type='string',enum=['inspect_geometry']),
            args=dict(type='object')),
            required=['tool','args'],additionalProperties=False)
        self.assertFalse(Draft7Validator(schema).is_valid(value))


if __name__=='__main__':
    unittest.main()
