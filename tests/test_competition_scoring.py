"""Regression tests for official RoboWorld HA-VLN 2.0 offline scoring."""
import math
import unittest

from havln_agent.competition_scoring import (
    EXPECTED_EPISODES, HUMAN_BETA, aggregate_episodes, official_formula,
    summarize_run, summarize_split,
)
from havln_agent.replay_server import _split_for_episode


class FormulaTests(unittest.TestCase):
    def test_exact_official_equation(self):
        sr, ne, cr, tcr = 0.180, 6.223, 0.623, 13.726
        result = official_formula(sr, ne, cr, tcr)
        nav = .80 * sr + .20 * 3/(3+ne)
        social = .75*(1-cr) + .25/(1+tcr)
        self.assertAlmostEqual(result['navigation'], nav, places=12)
        self.assertAlmostEqual(result['social'], social, places=12)
        self.assertAlmostEqual(result['score'], 100*nav*(.70+.30*social), places=12)
        # Published baseline metrics are rounded; a recomputation based on
        # these rounded values need not equal the official full-precision score.
        self.assertAlmostEqual(result['score'], 16.513612034321422, places=8)

    def test_perfect_episode_aggregate(self):
        result = official_formula(1, 0, 0, 0)
        self.assertEqual(result['score'], 100)
        self.assertEqual(result['navigation'], 1)
        self.assertEqual(result['social'], 1)

    def test_reject_non_finite_and_wrong_types(self):
        for value in (float('nan'), float('inf'), -1, True, '0.2', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                official_formula(value, 3, .5, 2)
        with self.assertRaises(ValueError):
            official_formula(1.01, 0, 0, 0)

    def test_aggregate_before_nonlinear_composite(self):
        rows = [
            dict(episode_id='1', navigation_error_m=0, success=True,
                 adjusted_collision_count=0, human_influenced=True),
            dict(episode_id='2', navigation_error_m=6, success=False,
                 adjusted_collision_count=0, human_influenced=True),
        ]
        aggregate, result, scope = aggregate_episodes(rows, 'val_seen')
        self.assertEqual(scope, 'measured_human_influenced_episodes')
        self.assertEqual(aggregate['NE'], 3)
        self.assertEqual(aggregate['SR'], .5)
        self.assertEqual(aggregate['CR'], 0)
        self.assertEqual(aggregate['TCR'], 0)
        self.assertAlmostEqual(aggregate['score'], official_formula(.5,3,0,0)['score'])

    def test_adjusted_human_collisions_and_beta(self):
        rows = [
            dict(episode_id='5', distance_to_goal_m=1.5, success=1,
                 collisions_count=5, reference_collision_count=3),
            dict(episode_id='6', distance_to_goal_m=6, success=0,
                 collisions_count=1, reference_collision_count=2),
        ]
        aggregate, details, mode = aggregate_episodes(rows, 'val_seen')
        self.assertEqual(mode,'official_split_beta_scaled_to_sample')
        self.assertEqual(details[0]['TCR'],2)
        self.assertEqual(details[0]['SR'],0)
        self.assertEqual(details[1]['TCR'],0)
        self.assertAlmostEqual(aggregate['CR'],1/(2*HUMAN_BETA['val_seen']))
        self.assertEqual(aggregate['TCR'],1)
        self.assertAlmostEqual(aggregate['NE'],3.75)

    def test_official_simulator_shaped_episode_record(self):
        row=dict(episode_id='11', distance_to_goal=2.5, success=1.0,
                 collisions=dict(count=3), expected_collisions=2)
        metrics, details, _=aggregate_episodes([row], 'val_seen')
        self.assertEqual(details[0]['TCR'], 1)
        self.assertEqual(details[0]['SR'], 0,
                         'A visually near goal is not collision-free success')
        self.assertEqual(metrics['NE'], 2.5)

    def test_duplicate_episodes_rejected(self):
        row=dict(episode_id='a',navigation_error_m=0,success=1,
                 adjusted_collision_count=0)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            aggregate_episodes([row,row],'val_seen')


class ReportTests(unittest.TestCase):
    def test_missing_truth_not_mistaken_for_score_zero(self):
        payload = summarize_run({'val_seen':['1','2']})
        one = payload['by_split']['val_seen']
        self.assertEqual(one['status'],'missing_metrics')
        self.assertIsNone(one['score'])
        self.assertIsNone(one['metrics'])
        self.assertFalse(one['authoritative'])
        self.assertEqual(one['expected_split_episodes'],778)

    def test_split_metrics_and_per_episode_details(self):
        report=dict(splits=dict(val_unseen=dict(
            metrics=dict(SR=.2, NE=4.5, CR=.3, TCR=1.5),
            episodes=[dict(episode_id='9', navigation_error_m=2.7,
                           success=True, adjusted_collision_count=0)]
        )))
        payload=summarize_run({'val_unseen':['9','10']},report,'score_metrics.json')
        one=payload['by_split']['val_unseen']
        self.assertEqual(one['status'],'scored')
        self.assertAlmostEqual(one['score'],official_formula(.2,4.5,.3,1.5)['score'])
        self.assertEqual(one['per_episode']['9']['SR'],1)
        self.assertNotIn('10',one['per_episode'])
        self.assertEqual(one['scope'],'imported_split_metrics')
        self.assertFalse(payload['official'])

    def test_episode_only_import_is_provisional(self):
        report=dict(episodes=[
            dict(episode_id='1',navigation_error_m=2,
                 success=True,adjusted_collision_count=0),
        ])
        one=summarize_split('val_seen',['1'],report,'local.json')
        self.assertEqual(one['status'],'scored')
        self.assertEqual(one['evaluated_episodes'],1)
        self.assertTrue(any('not leaderboard' in note for note in one['limitations']))

    def test_invalid_input_does_not_become_perfect_score(self):
        one=summarize_split('val_seen',['1'],dict(metrics=dict(
            SR=1,NE=-1,CR=0,TCR=0)), 'invalid.json')
        self.assertEqual(one['status'],'invalid_metrics')
        self.assertIsNone(one['score'])

    def test_reject_unrecognised_report_and_split(self):
        with self.assertRaises(ValueError):
            summarize_run({'val_seen':['1']},dict(unknown='garbage'))
        with self.assertRaises(ValueError):
            summarize_run({'val_seen':['1']},
                          dict(split='unreleased', metrics=dict(SR=0,NE=1,CR=0,TCR=0)))

    def test_split_identity(self):
        self.assertEqual(_split_for_episode('val_seen-123'),'val_seen')
        self.assertEqual(_split_for_episode('val_unseen-999'),'val_unseen')
        self.assertEqual(_split_for_episode('test-42'),'test')
        self.assertIsNone(_split_for_episode('unknown-1'))


if __name__ == '__main__':
    unittest.main()
