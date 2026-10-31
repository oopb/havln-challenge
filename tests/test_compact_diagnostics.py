"""Ensure compact traces keep model, tools and critical RGB-D without duplication."""
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from havln_agent.episode_runner import canonical_step_frame_refs
from scripts.pack_agent_diagnostics import pack


class CompactTraceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'sample-run'
        self.trace=self.root/'traces'/'val_seen-1'
        self.images=self.trace/'observations'
        self.images.mkdir(parents=True)
        samples=[]
        for i in range(8):
            ident=f'obs_{i:04d}'
            (self.images/(ident+'.npz')).write_bytes(b'fake-npz-'+bytes([i]))
            samples.append(dict(observation_id=ident,rgb_hash='rgb'+str(i),
                                depth_hash='depth'+str(i)))
        self.write_json('memory.json',dict(observations=samples))
        self.write_json('summary.json',dict(move_forward_steps=1))
        self.write_json('episode.json',dict(episode_id='1'))
        self.write_json('final_state.json',dict())
        for name in ('api.jsonl','navigation_agent.jsonl','events.jsonl'):
            (self.trace/name).write_text('{}\n')
        steps=[]
        for i in range(7):
            steps.append(dict(step=i,
                action='MOVE_FORWARD' if i==3 else 'TURN_LEFT',
                observation=dict(rgb_hash='rgb'+str(i),depth_hash='depth'+str(i)),
                post_observation=dict(rgb_hash='rgb'+str(i+1),
                                      depth_hash='depth'+str(i+1))))
        self.write_lines('steps.jsonl',steps)
        self.write_lines('tools.jsonl',[dict(
            call=dict(tool='navigate_local'),
            result=dict(status='BLOCKED',end_observation_id='obs_0005'))])
        (self.trace/'input-003.npz').write_bytes(b'duplicate-input')
        (self.trace/'post-003.npz').write_bytes(b'duplicate-post')
        (self.trace/'tool_ledger.json').write_text('[]')
        (self.root/'cache'/'0').mkdir(parents=True)
        (self.root/'cache'/'0'/'cache.json').write_text('{}')

    def write_json(self,name,data):
        (self.trace/name).write_text(json.dumps(data))

    def write_lines(self,name,items):
        (self.trace/name).write_text(''.join(json.dumps(x)+'\n' for x in items))

    def test_canonical_refs_exist_and_malformed_references_fail(self):
        refs=canonical_step_frame_refs(self.trace,dict(
            observation_before='obs_0002',observation_after='obs_0003'))
        self.assertEqual(refs['input_file'],'observations/obs_0002.npz')
        self.assertEqual(refs['post_file'],'observations/obs_0003.npz')
        with self.assertRaisesRegex(RuntimeError,'missing_canonical'):
            canonical_step_frame_refs(self.trace,dict(
                observation_before='obs_0002',observation_after='obs_missing'))

    def test_compact_archive_retains_forward_and_blocked_evidence(self):
        destination=Path(self.temp.name)/'critical.zip'
        manifest=pack(self.root,destination,frames='critical',interval=100)
        self.assertEqual(manifest['episodes']['val_seen-1']['total_frames'],8)
        with ZipFile(destination) as archive:
            names=set(archive.namelist())
        prefix='sample-run/traces/val_seen-1/'
        self.assertIn(prefix+'api.jsonl',names)
        self.assertIn(prefix+'tools.jsonl',names)
        self.assertIn(prefix+'steps.jsonl',names)
        self.assertIn(prefix+'observations/obs_0003.npz',names)
        self.assertIn(prefix+'observations/obs_0004.npz',names)
        self.assertIn(prefix+'observations/obs_0005.npz',names)
        self.assertNotIn(prefix+'input-003.npz',names)
        self.assertNotIn(prefix+'post-003.npz',names)
        self.assertNotIn(prefix+'tool_ledger.json',names)
        self.assertFalse(any('cache' in name for name in names))

    def test_api_log_stored_once_and_sensitive_key_redacted(self):
        from havln_agent.api_client import APIClient
        client=APIClient.__new__(APIClient)
        client.episode_id='1'
        client.log_path=self.trace/'api.jsonl'
        client.trace_context=dict(decision_id='d1')
        client.keys={('navigation_agent',0):'PRIVATE_API_TOKEN_X'}
        client._log(dict(role='navigation_agent',
            messages=[dict(role='system',content='private test '+
                           'PRIVATE_API_TOKEN_X')]))
        self.assertIn('navigation_agent',(self.trace/'api.jsonl').read_text())
        self.assertIn('[REDACTED]',(self.trace/'api.jsonl').read_text())
        self.assertNotIn('PRIVATE_API_TOKEN_X',(self.trace/'api.jsonl').read_text())
        self.assertFalse((self.trace/'api-navigation_agent.jsonl').exists())

    def test_archive_contains_diagnostic_event_logs_but_no_worker_task(self):
        (self.trace/'memory.jsonl').write_text('{}\n')
        (self.trace/'safety.jsonl').write_text('{}\n')
        (self.trace/'progress.jsonl').write_text('{}\n')
        (self.trace/'task_planner.jsonl').write_text('{}\n')
        (self.root/'worker-0.log').write_text('worker finished\n')
        (self.root/'worker-0.json').write_text('{\"config\":\"internal\"}')
        archive=Path(self.temp.name)/'logs.zip'
        pack(self.root,archive)
        with ZipFile(archive) as z:
            names=set(z.namelist())
        prefix='sample-run/traces/val_seen-1/'
        for name in ('memory.jsonl','safety.jsonl','progress.jsonl','task_planner.jsonl'):
            self.assertIn(prefix+name,names)
        self.assertIn('sample-run/worker-0.log',names)
        self.assertNotIn('sample-run/worker-0.json',names)


    def test_critical_pack_keeps_motion_frames_without_memory_snapshot(self):
        (self.trace/'memory.json').unlink()
        (self.trace/'steps.jsonl').write_text(json.dumps(dict(
            step=0,action='MOVE_FORWARD',
            input_file='observations/obs_0003.npz',
            post_file='observations/obs_0004.npz'))+'\n')
        destination=Path(self.temp.name)/'failed-run.zip'
        pack(self.root,destination,frames='critical',interval=100)
        with ZipFile(destination) as z:
            names=set(z.namelist())
        self.assertIn('sample-run/traces/val_seen-1/observations/obs_0003.npz',names)
        self.assertIn('sample-run/traces/val_seen-1/observations/obs_0004.npz',names)

    def test_incomplete_last_jsonl_line_does_not_break_failure_archive(self):
        original=self.trace/'steps.jsonl'
        original.write_text(original.read_text()+'{incomplete-record\n')
        destination=Path(self.temp.name)/'truncated-logs.zip'
        pack(self.root,destination,frames='critical')
        with ZipFile(destination) as z:
            self.assertTrue(any(n.endswith('.npz') for n in z.namelist()))

    def test_full_and_no_frames_modes(self):
        full=Path(self.temp.name)/'full.zip'
        none=Path(self.temp.name)/'none.zip'
        pack(self.root,full,frames='all')
        pack(self.root,none,frames='none')
        with ZipFile(full) as z:
            self.assertEqual(sum(n.endswith('.npz') for n in z.namelist()),8)
        with ZipFile(none) as z:
            self.assertFalse(any(n.endswith('.npz') for n in z.namelist()))


if __name__=='__main__':
    unittest.main()
