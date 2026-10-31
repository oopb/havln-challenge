"""Independent multi-sensor transition proof; no model-only task completion."""
import unittest
from havln_agent.progress import _verified_stair_transition


def motion(after):
    return dict(result=dict(observation_after=after))


class IndependentProgressProofTests(unittest.TestCase):
    def candidate(self):
        return dict(directed=[motion('d1'),motion('d2')],
                    landing_steps=[motion('l1'),motion('l2')],
                    landings=[dict(status='PARTIAL',
                        landing_geometry=dict(status='SUPPORTED'))],
                    displacement=.42,uncertainty=.05,
                    travel=.9,travel_error=.08,continuous=True,
                    contradictions=[],forward_step=.25)

    def test_multiple_independent_measurements_can_confirm_without_entry_box(self):
        self.assertTrue(_verified_stair_transition(**self.candidate()))

    def test_visual_claim_alone_never_confirms(self):
        item=self.candidate()
        item['directed']=[]
        self.assertFalse(_verified_stair_transition(**item))

    def test_landing_requires_two_distinct_measured_action_receipts(self):
        item=self.candidate()
        item['landing_steps']=[motion('l1'),motion('l1')]
        self.assertFalse(_verified_stair_transition(**item))

    def test_uncertain_or_discontinuous_motion_cannot_confirm(self):
        item=self.candidate()
        item['continuous']=False
        self.assertFalse(_verified_stair_transition(**item))
        item=self.candidate()
        item['displacement']=.10
        self.assertFalse(_verified_stair_transition(**item))
        item=self.candidate()
        item['contradictions']=['registered_motion_opposes_stair_instruction']
        self.assertFalse(_verified_stair_transition(**item))

    def test_landing_visual_depth_evidence_remains_required(self):
        item=self.candidate()
        item['landings']=[]
        self.assertFalse(_verified_stair_transition(**item))


if __name__ == '__main__':
    unittest.main()
