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
    raw = item.get('metrics') if isinstance(item.get('metrics'), dict) else item
    if not all(k in raw for k in METRIC_KEYS):
        return None
    return official_formula(*(raw[k] for k in METRIC_KEYS))


def _bool01(value, key):
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError('%s must be true/false or 0/1' % key)


def _episode_row(row):
    """Accepted *post-evaluation* episode evidence (not agent observations)."""
    if not isinstance(row, dict) or not str(row.get('episode_id', '')).strip():
        raise ValueError('episode_id required for scored episode')
    nav = row.get('navigation_error_m', row.get('distance_to_goal_m'))
    success = row.get('success')
    if 'adjusted_collision_count' in row:
        adjusted = numeric(row['adjusted_collision_count'], 'adjusted_collision_count')
    else:
        collisions = row.get('collisions_count')
        reference = row.get('reference_collision_count')
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
        per_episode = {}
        episode_metrics = None
        cr_scope = None
        if isinstance(report.get('episodes'), list) and report['episodes']:
            episode_metrics, rows, cr_scope = aggregate_episodes(report['episodes'], split)
            per_episode = {r['episode_id']: r for r in rows if r['episode_id'] in ids}
            base['evaluated_episodes'] = len(rows)
        provided_metrics = _aggregate_metrics(report)
        chosen = provided_metrics or episode_metrics
        if chosen is None:
            base['limitations'] = [
                'Import complete SR/NE/CR/TCR or evaluated episode records.']
            return base
        result = dict(base, status='scored', metrics={k: chosen[k] for k in METRIC_KEYS},
                      score=chosen['score'], navigation=chosen['navigation'],
                      social=chosen['social'], social_factor=chosen['social_factor'],
                      per_episode=per_episode,
                      scope=('imported_split_metrics' if provided_metrics
                             else 'calculated_from_evaluated_episodes'),
                      collision_normalization=cr_scope,
                      limitations=[])
        if result['evaluated_episodes'] and result['evaluated_episodes'] != expected:
            result['limitations'].append(
                'Only %d/%d episodes evaluated: exploratory subset, not leaderboard.'
                % (result['evaluated_episodes'], expected))
        elif not result['evaluated_episodes']:
            result['limitations'].append(
                'Imported split metrics; original episode sample size not verified.')
        if not provided_metrics and len(per_episode) != len(ids):
            result['limitations'].append(
                'Not all run episodes have independently evaluated metrics.')
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
