"""Pure, read-only RoboWorld 2026 Track 2 (HA-VLN 2.0) score calculation.

Official rule: https://roboworld2026.github.io/track2
Score is calculated from *aggregated* SR, NE, CR and TCR, never from
action counts, model self-assessments or RGB-D similarity. The competition
server remains the authority for leaderboard results.

The optional imported report contains either the four already-evaluated
split metrics or independently evaluated per-episode simulator metrics.
No privileged goal/collision data are requested by the inference agent.
"""
import math
import json
from functools import lru_cache
from pathlib import Path

FORMULA_ID = 'roboworld2026-track2-havln2-score-v1'
RULE_URL = 'https://roboworld2026.github.io/track2'
EXPECTED_EPISODES = {
    'train': 10819, 'val_seen': 778, 'val_unseen': 1839, 'test': 3408
}
# Reference adjusted-collision normalisation from HA-VLN HASimulator/metric.py.
# This is an official full-split constant, not a measured rate in a small run.
HUMAN_BETA = {
    'train': 1 - 1615 / 10819,
    'val_seen': 1 - 96 / 778,
    'val_unseen': 1 - 246 / 1839,
    'test': 1 - 549 / 3408,
}
METRIC_KEYS = ('SR', 'NE', 'CR', 'TCR')

# Public validation reference collision counts from:
# UWMILab/HA-VLN/Data/HA-R2R-tools/collision_num_val_{seen,unseen}.json
# The test split has no released collision annotations and is never inferred.
PUBLISHED_REFERENCE_SOURCE = (
    'UWMILab/HA-VLN/Data/HA-R2R-tools/collision_num_{split}.json')


@lru_cache(maxsize=2)
def _published_validation_references(split):
    if split not in ('val_seen', 'val_unseen'):
        return {}
    file = (Path(__file__).resolve().parent / 'scoring_data' /
            ('collision_num_' + split + '.json'))
    try:
        rows = json.loads(file.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    if (not isinstance(rows, dict)
            or len(rows) != EXPECTED_EPISODES[split]):
        return {}
    if any(type(v) is not int or v < 0 for v in rows.values()):
        return {}
    return rows


def published_reference_collision_count(split, episode_id):
    """Exact published validation annotation, or None; never invent test truth."""
    return _published_validation_references(split).get(str(episode_id))


def numeric(value, field, *, minimum=0.0, maximum=None):
    """Reject bool, NaN, infinity, malformed strings and invalid ranges."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('%s must be a finite number' % field)
    value = float(value)
    if not math.isfinite(value) or value < minimum or (
            maximum is not None and value > maximum):
        raise ValueError('%s is out of range' % field)
    return value


def official_formula(sr, ne, cr, tcr):
    """Score in points; input SR/CR are fractions, NE is metres, TCR a count."""
    sr = numeric(sr, 'SR', maximum=1)
    ne = numeric(ne, 'NE')
    cr = numeric(cr, 'CR')  # small beta-normalised samples may exceed 1
    tcr = numeric(tcr, 'TCR')
    navigation = 0.80 * sr + 0.20 * 3 / (3 + ne)
    social = 0.75 * (1 - cr) + 0.25 / (1 + tcr)
    return dict(
        SR=sr, NE=ne, CR=cr, TCR=tcr,
        navigation=navigation,
        social=social,
        social_factor=0.70 + 0.30 * social,
        score=100 * navigation * (0.70 + 0.30 * social),
    )


def _aggregate_metrics(item):
    """Only consume explicit scoring metrics; never infer from trace statistics."""
    if not isinstance(item, dict):
        return None
    # The default Habitat collector published calculated_metrics while the
    # original Replay importer expected metrics. Accept both schemas without
    # mistaking diagnostic component averages for a complete official Score.
    raw = (item.get('metrics') if isinstance(item.get('metrics'), dict)
           else item.get('calculated_metrics') if isinstance(
               item.get('calculated_metrics'), dict) else item)
    if not all(k in raw for k in METRIC_KEYS):
        return None
    return official_formula(*(raw[k] for k in METRIC_KEYS))


def _bool01(value, key):
    if type(value) is bool:
        return value
    if type(value) in (int, float) and value in (0, 1):
        return bool(value)
    raise ValueError('%s must be true/false or 0/1' % key)


def _episode_row(row):
    """Accepted *post-evaluation* episode evidence (not agent observations)."""
    if not isinstance(row, dict) or not str(row.get('episode_id', '')).strip():
        raise ValueError('episode_id required for scored episode')
    nav = row.get('navigation_error_m',
                  row.get('distance_to_goal_m', row.get('distance_to_goal')))
    success = row.get('success')
    if 'adjusted_collision_count' in row:
        adjusted = numeric(row['adjusted_collision_count'], 'adjusted_collision_count')
    else:
        collisions = row.get('collisions_count')
        if collisions is None and isinstance(row.get('collisions'), dict):
            collisions = row['collisions'].get('count')
        reference = row.get('reference_collision_count',
                            row.get('expected_collisions'))
        if collisions is None or reference is None:
            raise ValueError('adjusted_collision_count or both collision counts required')
        adjusted = max(0, numeric(collisions, 'collisions_count')
                          - numeric(reference, 'reference_collision_count'))
    error = numeric(nav, 'navigation_error_m')
    arrived = _bool01(success, 'success')
    return dict(
        episode_id=str(row['episode_id']),
        NE=error,
        TCR=adjusted,
        CR=1 if adjusted > 0 else 0,
        SR=1 if arrived and adjusted == 0 else 0,
        # True/False may be supplied by a scorer; never guess from instruction.
        human_influenced=(None if row.get('human_influenced') is None
                          else _bool01(row['human_influenced'], 'human_influenced')),
    )


def aggregate_episodes(rows, split):
    """Compute component means *first* and then apply the official formula.

    A partial batch is a diagnostic subset, NOT an official leaderboard score.
    CR uses the published split beta unless all episode-level influence labels
    are supplied, in which case the measured sample denominator is used.
    """
    if split not in HUMAN_BETA:
        raise ValueError('Unknown split: ' + str(split))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Nonempty evaluated episodes required')
    episodes = [_episode_row(r) for r in rows]
    ids = [r['episode_id'] for r in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate scored episode_id')
    n = len(episodes)
    flags = [r['human_influenced'] for r in episodes]
    if all(x is not None for x in flags):
        denominator = sum(flags)
        cr_scope = 'measured_human_influenced_episodes'
    else:
        denominator = HUMAN_BETA[split] * n
        cr_scope = 'official_split_beta_scaled_to_sample'
    if denominator <= 0:
        raise ValueError('CR denominator is zero')
    metrics = official_formula(
        sum(r['SR'] for r in episodes) / n,
        sum(r['NE'] for r in episodes) / n,
        sum(r['CR'] for r in episodes) / denominator,
        sum(r['TCR'] for r in episodes) / n,
    )
    return metrics, episodes, cr_scope


def _reports(data):
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError('Metrics report must be a JSON object')
    if isinstance(data.get('splits'), dict):
        reports = data['splits']
    elif isinstance(data.get('split'), str):
        reports = {data['split']: data}
    else:
        raise ValueError('Supply a split or splits map')
    if not reports:
        raise ValueError('No split in metrics report')
    for name, report in reports.items():
        if name not in EXPECTED_EPISODES or not isinstance(report, dict):
            raise ValueError('Unknown split or malformed report: ' + str(name))
    return reports


def summarize_split(split, run_ids, report=None, imported_from=None):
    """Return UI-friendly breakdown with explicit availability and provenance."""
    ids = set(map(str, run_ids))
    expected = EXPECTED_EPISODES.get(split)
    base = dict(split=split, status='missing_metrics',
                formula_id=FORMULA_ID, formula_url=RULE_URL,
                authoritative=False, run_episodes=len(ids),
                expected_split_episodes=expected,
                metrics=None, score=None, navigation=None, social=None,
                social_factor=None, evaluated_episodes=0,
                per_episode={}, imported_from=imported_from,
                limitations=['Missing independent goal-distance and human-collision evaluation.'])
    if not isinstance(report, dict):
        return base
    try:
        # Raw measured fields are useful even when hidden test goals or collision
        # reference allowances make the composite Score impossible.
        components = report.get('components') if isinstance(
            report.get('components'), dict) else {}
        observed_rows = report.get('observed_episodes')
        if not isinstance(observed_rows, list):
            observed_rows = []
        observed = {str(row['episode_id']): dict(row) for row in observed_rows
                    if isinstance(row, dict) and str(row.get('episode_id','')) in ids}
        for episode_id, row in observed.items():
            if row.get('reference_collision_count') is None:
                allowance = published_reference_collision_count(split, episode_id)
                if allowance is not None:
                    row['reference_collision_count'] = allowance
                    row['reference_collision_source'] = 'published_validation_annotations'
        base.update(
            components=components,
            observed_episodes=observed,
            captured_complete=report.get('captured_complete'),
            fully_scored_episodes=report.get('fully_scored_episodes'),
            requested_episodes=report.get('requested_episodes'),
            capture_source=report.get('source', 'simulator_metric_capture'),
            missing_or_partial=report.get('missing_or_partial', [])[:12])
        per_episode = {}
        episode_metrics = None
        cr_scope = None
        scored_rows = []
        if isinstance(report.get('episodes'), list):
            scored_rows = [r for r in report['episodes']
                           if isinstance(r, dict) and str(r.get('episode_id')) in ids]
        known_ids = {str(r.get('episode_id')) for r in scored_rows}
        # Old diagnostic archives contain complete Habitat observations but
        # were written before the public validation reference table was used.
        # Reconstruct ONLY complete, genuine observed episodes. Never use
        # Codex's final message, generated action summaries, or unknown test
        # collision allowances to manufacture success.
        reconstructed = 0
        for episode_id, obs in observed.items():
            if episode_id in known_ids or obs.get('complete') is not True:
                continue
            ref = obs.get('reference_collision_count')
            if ref is None:
                continue
            if (type(obs.get('navigation_error_m')) not in (int, float)
                    or type(obs.get('success')) is not bool
                    or type(obs.get('collisions_count')) not in (int, float)):
                continue
            scored_rows.append(dict(
                episode_id=episode_id,
                navigation_error_m=obs['navigation_error_m'],
                success=obs['success'],
                collisions_count=obs['collisions_count'],
                reference_collision_count=ref,
            ))
            reconstructed += 1
        if scored_rows:
            episode_metrics, rows, cr_scope = aggregate_episodes(scored_rows, split)
            per_episode = {r['episode_id']: r for r in rows}
            base['evaluated_episodes'] = len(rows)
        if reconstructed:
            base['reconstructed_from_observed_episodes'] = reconstructed
            base['collision_reference_source'] = (
                'published_validation_annotations')
        provided_metrics = _aggregate_metrics(report)
        # A live-Habitat collector may contain an older partial aggregate.
        # The complete, grounded reconstructed set takes precedence for its
        # selected episodes, but not over independent imported scorer data.
        if (report.get('source') == 'actual_live_habitat_measurements'
                and len(scored_rows) == len(ids) and ids):
            chosen = episode_metrics
            provided_metrics = None
        else:
            chosen = provided_metrics or episode_metrics
        if chosen is None:
            base['status'] = ('partial_metrics' if components or observed else 'missing_metrics')
            base['limitations'] = [
                'Real Habitat measurements are available, but a composite Score '
                'requires goal distance, navigation success, and reference-adjusted '
                'collisions for the evaluated episodes.' if base['status']=='partial_metrics'
                else 'No captured simulator evaluation exists for this run. '
                     'Old traces cannot recover true evaluation from images or actions.']
            return base
        result = dict(base, status='scored', metrics={k: chosen[k] for k in METRIC_KEYS},
                      score=chosen['score'], navigation=chosen['navigation'],
                      social=chosen['social'], social_factor=chosen['social_factor'],
                      per_episode=per_episode,
                      scope=('imported_split_metrics' if provided_metrics
                             else 'calculated_from_evaluated_episodes'),
                      collision_normalization=cr_scope,
                      limitations=[])
        if provided_metrics:
            if report.get('fully_scored_episodes') is not None:
                result['evaluated_episodes'] = int(report['fully_scored_episodes'])
            else:
                result['limitations'].append(
                    'Imported aggregate: sample size cannot be verified from four metrics alone.')
            if result['evaluated_episodes'] and result['evaluated_episodes'] != expected:
                result['limitations'].append(
                    'Episode-level evidence covers only %d/%d; it does not define the imported aggregate.'
                    % (result['evaluated_episodes'], expected))
        elif result['evaluated_episodes'] != expected:
            result['limitations'].append(
                'Only %d/%d episodes evaluated: exploratory subset, not leaderboard.'
                % (result['evaluated_episodes'], expected))
        if result['evaluated_episodes'] != len(ids):
            result['limitations'].append(
                'Only %d/%d selected run episodes have a complete independent Score; '
                'displayed Score is a diagnostic subset, not a run-wide estimate.'
                % (result['evaluated_episodes'], len(ids)))
        if report.get('scope') == 'diagnostic_subset_or_missing_truth':
            result['limitations'].append(
                'Some recorded episodes are missing required scoring truth.')
        result['limitations'].append(
            'Local reconstruction; only the CodaBench scorer determines official ranking.')
        return result
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        base['status'] = 'invalid_metrics'
        base['limitations'] = [str(error)]
        return base


def summarize_run(run_by_split, imported=None, source_name=None):
    reports = _reports(imported)
    result = {}
    for split, ids in run_by_split.items():
        result[split] = summarize_split(split, ids, reports.get(split), source_name)
    for split, report in reports.items():
        if split not in result:
            result[split] = summarize_split(split, [], report, source_name)
    return dict(formula_id=FORMULA_ID, formula_url=RULE_URL, by_split=result,
                scoring_basis='independent_evaluation_only',
                official=False)
