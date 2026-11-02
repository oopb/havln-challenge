"""Direct model-directed move failures must survive context compaction."""
import unittest
from types import SimpleNamespace
from havln_agent.memory_service import MemoryService


class DirectFailureMemoryTests(unittest.TestCase):
    def test_direct_blocked_and_pixel_needs_view_are_remembered(self):
        m=MemoryService.__new__(MemoryService)
        m.runtime=SimpleNamespace(semantic_motion_policy='LEVEL_ONLY')
        m.task=SimpleNamespace(current_subgoal_index=1,subgoal_started=0)
        m.actions=[]
        m.observations=[]
        m.tool_ledger=[
            dict(call_id='earlier',subgoal_id='sg_0',
                call=dict(tool='navigate_relative',args=dict(turn_deg=-45,distance_m=.5)),
                result=dict(status='BLOCKED',reason='occupied_next_step')),
            dict(call_id='direct',subgoal_id='sg_1',
                call=dict(tool='navigate_relative',args=dict(turn_deg=30,distance_m=.75)),
                result=dict(status='PARTIAL',interrupted_status='BLOCKED',
                    reason='occupied_next_step',waypoint_id='direct_123',
                    end_observation_id='obs_0030')),
            dict(call_id='pixel',subgoal_id='sg_1',
                call=dict(tool='navigate_to_pixel',args=dict(u=.8,v=.4,distance_m=.5)),
                result=dict(status='NEEDS_VIEW',reason='pixel_target_depth_unmeasured',
                    end_observation_id='obs_0031'))]
        summary=m.navigation_summary()
        failures=summary['recent_failed_waypoints']
        self.assertEqual(len(failures),2)
        self.assertEqual(failures[0]['tool'],'navigate_to_pixel')
        self.assertEqual(failures[1]['status'],'BLOCKED')
        self.assertEqual(failures[1]['waypoint_id'],'direct_123')
        self.assertEqual(failures[1]['requested_motion'],dict(turn_deg=30,distance_m=.75))
        self.assertNotIn('earlier',[r['tool_call_id'] for r in failures])


if __name__=='__main__':
    unittest.main()
