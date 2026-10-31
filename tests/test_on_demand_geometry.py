"""Fast collision guard is independent from full terrain approval."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from havln_agent.perception import Geometry
from havln_agent.safety import check_action, forward_probe
from havln_agent.waypoint import generate_waypoints


class GeometryOnDemandTests(unittest.TestCase):
    def geometry(self, status='FREE', accepted=True):
        g=Geometry.__new__(Geometry)
        g.camera=SimpleNamespace(forward_step=0.25,turn_angle=15,hfov=90)
        g.human_status=dict(unlocalized=0)
        g.diagnose=Mock(return_value=dict(
            path_state=status, accepted=accepted, clearance=0.25 if accepted else 0.03,
            rejection_reasons=[] if accepted else ['clearance_below_forward_step']))
        # The previously blocking terrain mask must not be consulted during
        # routine candidate generation or immediate forward execution.
        g.assess_step=Mock(side_effect=AssertionError('full terrain called implicitly'))
        return g

    def test_plain_depth_free_is_actionable_even_if_terrain_would_reject(self):
        g=self.geometry()
        state=forward_probe(g)
        self.assertEqual(state['state'],'EXECUTABLE')
        self.assertEqual(state['support']['status'],'DEFERRED')
        self.assertEqual(check_action('MOVE_FORWARD',g),('MOVE_FORWARD',None))
        g.assess_step.assert_not_called()

    def test_measured_obstacle_blocks_short_move_without_full_inspection(self):
        g=self.geometry('OCCUPIED',False)
        check=forward_probe(g)
        self.assertEqual(check['state'],'BLOCKED')
        self.assertEqual(check['diagnostic_tool'],'inspect_geometry')
        self.assertEqual(check_action('MOVE_FORWARD',g),
                         (None,'occupied_next_step'))
        g.assess_step.assert_not_called()

    def test_unknown_depth_requires_new_view(self):
        g=self.geometry('UNKNOWN',False)
        self.assertEqual(forward_probe(g)['state'],'NEEDS_VIEW')
        self.assertEqual(check_action('MOVE_FORWARD',g),
                         (None,'unknown_depth_or_fov'))

    def test_unlocalized_human_requires_another_view(self):
        g=self.geometry()
        g.human_status['unlocalized']=1
        self.assertEqual(forward_probe(g)['reason'],'human_localization_unverified')

    def test_waypoint_preview_never_fits_terrain(self):
        g=self.geometry()
        paths=generate_waypoints(g, 0.5, probe=True)
        self.assertTrue(paths)
        self.assertTrue(all(p['first_step']=='EXECUTABLE' for p in paths))
        g.assess_step.assert_not_called()


if __name__=='__main__':
    unittest.main()
