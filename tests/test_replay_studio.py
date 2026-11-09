"""Replay capture, byte-exact image references, legacy and ZIP playback."""
from dataclasses import asdict
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from havln_agent.perception import Camera, depth_image_url, image_url
from havln_agent.replay_capture import ReplayRecorder
from havln_agent.replay_server import ReplaySource, EpisodeReplay
from scripts.pack_agent_diagnostics import pack


class ReplayTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'run'
        self.trace = self.root / 'traces' / 'val_seen-1'
        images = self.trace / 'observations'
        images.mkdir(parents=True)
        self.camera = Camera(width=32, height=24, hfov=90,
                             min_depth=.1, max_depth=10.0, normalized=True,
                             camera_height=1.2, robot_height=1.7,
                             robot_radius=.2, forward_step=.25, turn_angle=15)
        self.frame = SimpleNamespace(
            rgb=np.full((24, 32, 3), [30, 80, 120], dtype=np.uint8),
            depth=np.full((24, 32), .4, dtype=np.float32))
        np.savez_compressed(images / 'obs_0000.npz',
                            rgb=self.frame.rgb, depth=self.frame.depth, pitch_deg=0)
        (self.trace / 'episode.json').write_text(json.dumps(
            dict(episode_id='1', instruction='Reach the piano.',
                 camera=asdict(self.camera))))
        (self.trace / 'steps.jsonl').write_text(json.dumps(
            dict(step=0, action='TURN_LEFT',
                 input_file='observations/obs_0000.npz',
                 post_file='observations/obs_0000.npz'))+'\n')
        (self.trace / 'tools.jsonl').write_text('')
        (self.trace / 'summary.json').write_text('{}')
        (self.trace / 'events.jsonl').write_text('')

    def test_source_images_are_reused_and_history_is_complete(self):
        recorder = ReplayRecorder(self.trace, self.camera)
        rgb = image_url(self.frame, max_edge=640)
        depth = depth_image_url(self.frame, self.camera, max_edge=640)
        history = [dict(role='user', content=[
            dict(type='text', text=json.dumps(dict(source_observation_id='obs_0000'))),
            dict(type='image_url', image_url=dict(url=rgb)),
            dict(type='image_url', image_url=dict(url=depth))])]
        current = dict(role='user', content=[
            dict(type='text', text=json.dumps(dict(source_observation_id='obs_0000'))),
            dict(type='image_url', image_url=dict(url=rgb))])
        request = recorder.request_snapshot(
            [dict(role='system', content='system'), *history, current],
            dict(source_observation_id='obs_0000'),
            dict(model='test', response_format=dict(type='json_object'),
                 temperature=0, max_tokens=256))
        self.assertEqual(len(request['prior_messages']), 1)
        self.assertEqual([p['image_url']['replay_media']['sensor']
                          for p in request['prior_messages'][0]['content'][1:]],
                         ['rgb', 'depth'])
        self.assertEqual(request['current_media'][0]['observation_id'], 'obs_0000')
        self.assertFalse((self.trace / 'replay' / 'media').exists(),
                         'Canonical frame media must not be duplicated on disk')
        self.assertNotIn('base64,', json.dumps(request))
        self.assertEqual(request['request_parameters']['max_tokens'], 256)

    def test_unmatched_media_content_addressed_exactly_once(self):
        recorder = ReplayRecorder(self.trace, self.camera)
        other = SimpleNamespace(rgb=np.zeros((8,8,3), dtype=np.uint8))
        url = image_url(other, max_edge=640)
        a = recorder.media(url, [])
        b = recorder.media(url, [])
        self.assertEqual(a,b)
        self.assertEqual(a['kind'],'stored')
        self.assertEqual(len(list((self.trace/'replay'/'media').glob('*.png'))),1)

    def test_episode_index_and_zip_reuse_same_canonical_files(self):
        recorder = ReplayRecorder(self.trace, self.camera)
        (self.trace / 'api.jsonl').write_text(json.dumps(
            dict(role='navigation_agent', success=True, model='test',
                 purpose='navigation_decision',
                 messages=[dict(role='system', content='a'),
                           dict(role='user', content=json.dumps(dict(
                               current_rgb=dict(observation_id='obs_0000'))))]))+'\n')
        recorder.observation('obs_0000')
        recorder.api_logged(dict(role='navigation_agent',
                                 purpose='navigation_decision',success=True))
        recorder.action(dict(step=0, action='TURN_LEFT',
                             input_file='observations/obs_0000.npz',
                             post_file='observations/obs_0000.npz'))
        src=ReplaySource(self.root)
        replay=EpisodeReplay(src,'val_seen-1')
        self.assertTrue(replay.precise_order)
        self.assertEqual(replay.overview()['counts']['api'],1)
        self.assertEqual(replay.events[1]['kind'],'api')
        self.assertEqual(replay.detail(1)['record']['model'],'test')
        self.assertTrue(replay.frame('obs_0000','rgb').startswith(b'\\x89PNG'))
        self.assertTrue(replay.frame('obs_0000','depth').startswith(b'\\x89PNG'))
        archive=Path(self.root.parent)/'full-replay.zip'
        pack(self.root,archive,frames='all')
        with ZipFile(archive) as z:
            self.assertEqual(len([n for n in z.namelist()
                                  if n.endswith('obs_0000.npz')]),1)
            self.assertEqual(len([n for n in z.namelist()
                                  if n.endswith('replay/events.jsonl')]),1)
        zip_src=ReplaySource(archive)
        self.addCleanup(zip_src.zip.close)
        self.assertEqual(EpisodeReplay(zip_src,'val_seen-1').frame('obs_0000','rgb'),
                         replay.frame('obs_0000','rgb'))

    def test_legacy_mode_never_claims_exact_cross_stream_order(self):
        source=ReplaySource(self.root)
        replay=EpisodeReplay(source,'val_seen-1')
        self.assertFalse(replay.precise_order)
        self.assertEqual(replay.overview()['counts']['actions'],1)
        with self.assertRaises(FileNotFoundError):
            replay.frame('obs_9999','rgb')


if __name__ == '__main__':
    unittest.main()
