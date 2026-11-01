"""Independent Habitat worker distribution and parent terminal-log controls."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from havln_agent.export import distribute_jobs, worker_slots, main
from havln_agent.progress_monitor import ProgressMonitor


class BatchEvaluationTests(unittest.TestCase):
    def jobs(self):
        return [
            dict(key='val_seen-sceneA',split='val_seen',
                 ids=[str(i) for i in range(8)]),
            dict(key='val_seen-sceneB',split='val_seen',ids=['8','9']),
            dict(key='val_unseen-sceneC',split='val_unseen',
                 ids=['10','11','12','13']),
        ]

    def test_one_scene_can_fill_four_same_gpu_workers(self):
        job=[dict(key='val_seen-sceneA',split='val_seen',ids=[str(i) for i in range(8)])]
        workers=worker_slots(job,[0],4)
        self.assertEqual(len(workers),4)
        self.assertEqual([w['worker_id'] for w in workers],
                         ['0-b0','0-b1','0-b2','0-b3'])
        self.assertTrue(all(w['gpu']==0 for w in workers))
        keys=[j['key'] for w in workers for j in w['jobs']]
        self.assertEqual(len(keys),len(set(keys)))
        seen=[i for w in workers for j in w['jobs'] for i in j['ids']]
        self.assertCountEqual(seen,[str(i) for i in range(8)])
        self.assertEqual([sum(len(j['ids']) for j in w['jobs']) for w in workers],
                         [2,2,2,2])

    def test_balancing_multi_gpu_and_no_duplicate_episodes(self):
        jobs=self.jobs()
        workers=worker_slots(jobs,[0,1],3)
        self.assertEqual(len(workers),6)
        self.assertEqual(len({w['worker_id'] for w in workers}),6)
        self.assertEqual([w['gpu'] for w in workers],[0,0,0,1,1,1])
        slots=[sum(len(j['ids']) for j in w['jobs']) for w in workers]
        self.assertLessEqual(max(slots)-min(slots),2)
        flat=[(j['split'],item) for w in workers for j in w['jobs'] for item in j['ids']]
        original=[(j['split'],item) for j in jobs for item in j['ids']]
        self.assertCountEqual(flat,original)
        shard_keys=[j['key'] for w in workers for j in w['jobs']]
        self.assertEqual(len(shard_keys),len(set(shard_keys)))

    def test_one_episode_runs_only_one_process_even_when_batch_is_large(self):
        jobs=[dict(key='val_seen-one',split='val_seen',ids=['1'])]
        slots=worker_slots(jobs,[0],8)
        self.assertEqual(len(slots),1)
        self.assertEqual(slots[0]['jobs'][0]['ids'],['1'])

    def test_default_batch_keeps_original_gpu_and_shard_mapping(self):
        jobs=self.jobs()
        workers=worker_slots(jobs,[0,1],1)
        self.assertEqual([w['worker_id'] for w in workers],['0','1'])
        self.assertEqual(workers[0]['jobs'],jobs[::2])
        self.assertEqual(workers[1]['jobs'],jobs[1::2])

    def test_progress_monitor_follows_concurrent_worker_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            for name in ('0-b0','0-b1'):
                (output/f'worker-{name}.progress.json').write_text(
                    json.dumps(dict(phase='finished',split='all',episode_id='all')))
            values=[]
            monitor=ProgressMonitor(output,['0-b0','0-b1'],
                                    emit=lambda message,**kwargs:values.append(message))
            monitor.report()
            self.assertEqual(len(values),2)
            self.assertIn('GPU 0-b0',values[0])
            self.assertIn('GPU 0-b1',values[1])

    def test_terminal_log_off_suppresses_normal_output_not_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            args=['run_api_agent.py','--output-dir',directory,'--batch','2',
                  '--terminal-log','off','--no-auto-archive']
            stdout=io.StringIO()
            def fake_export(args,config):
                self.assertEqual(args.batch,2)
                self.assertEqual(args.terminal_log,'off')
                print('normal parent progress')
            with patch.object(sys,'argv',args),patch(
                    'havln_agent.export.load_config',return_value=dict(runtime={})),patch(
                    'havln_agent.export.export',side_effect=fake_export),contextlib.redirect_stdout(stdout):
                main()
            self.assertEqual(stdout.getvalue(),'')

    def test_batch_argument_rejects_zero_and_above_safety_ceiling(self):
        for batch in ('0','33'):
            with self.subTest(batch=batch),patch.object(sys,'argv',[
                    'run_api_agent.py','--output-dir','/tmp/demo',
                    '--batch',batch]),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code,2)


if __name__=='__main__':
    unittest.main()
