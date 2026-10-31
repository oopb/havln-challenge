"""Target-aware shortlisting, one-direction passive sensing and duplicate feedback."""
import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from havln_agent.tools import ToolExecutor
from havln_agent.waypoint import rank_candidates
from havln_agent.perception import Observation


class ObservationEfficiencyTests(unittest.TestCase):
    def test_grounded_candidate_ranking_uses_target_progress(self):
        def path(identifier, degrees, gain):
            return dict(id=identifier,angle=math.radians(degrees),
                first_step='EXECUTABLE',goal_association=[
                dict(relevant=True,distance_before_m=1.0,distance_after_m=1.0-gain)])
        candidates=[path('straight',0,.01),path('left',15,.25),
                    path('right',-15,.09),path('far',30,.4)]
        ranked=rank_candidates(candidates,'approach_landmark')
        self.assertEqual([p['id'] for p in ranked],
                         ['far','left','right','straight'])
        explore=rank_candidates(candidates,'explore')
        self.assertEqual(explore[0]['id'],'straight')

    def test_unrelated_waypoints_are_not_promoted(self):
        relevant=dict(angle=math.radians(30),first_step='EXECUTABLE',
            goal_association=[dict(relevant=True,distance_before_m=1.0,
                                   distance_after_m=.6)])
        unrelated=dict(angle=0,first_step='EXECUTABLE',
            goal_association=[dict(relevant=False,distance_before_m=1.0,
                                   distance_after_m=.2)])
        self.assertIs(rank_candidates([unrelated,relevant],'approach_landmark')[0],
                      relevant)

    def test_observation_only_probes_front_and_reuses_identical_view(self):
        fake=ToolExecutor.__new__(ToolExecutor)
        frame=Observation(np.zeros((16,16,3),dtype=np.uint8),
            np.full((16,16),2.,dtype=np.float32),'descend',time.monotonic())
        fake.observation=None
        fake.motion_controller=SimpleNamespace(pose=lambda:dict(x=0,z=0,heading=0))
        record=dict(observation_id='obs_0000',pose=dict(x=0,z=0,heading=0),
                    rgb_hash='not_reused',depth_hash='not_reused',directions=[])
        fake.memory=SimpleNamespace(current=None,observe=Mock(return_value=record))
        geo=SimpleNamespace(movement_probe=Mock(return_value=dict(
            state='EXECUTABLE',reason=None,depth=dict(
                path_state='FREE',clearance=.25,accepted=True),
            support=dict(status='DEFERRED'))))
        fake.geometry=Mock(return_value=geo)
        fake.record_observation=None
        fake.load_observation=Mock()
        fake.frames={}
        fake.accept_observation(frame)
        diagnostic=fake.memory.observe.call_args.args[2]
        self.assertEqual(len(diagnostic),1)
        self.assertEqual(diagnostic[0]['angle_degrees'],0)
        geo.movement_probe.assert_called_once_with(0)
        # On unchanged view, no additional geometry pass is needed.
        from havln_agent.observation import array_hash
        record.update(rgb_hash=array_hash(frame.rgb),depth_hash=array_hash(frame.depth))
        fake.memory.current=record
        fake.accept_observation(frame)
        geo.movement_probe.assert_called_once_with(0)
        self.assertEqual(fake.memory.observe.call_count,2)


if __name__=='__main__':
    unittest.main()
