"""Separate, explicitly assumed recovery pricing policy for a synthetic demonstration.

The full Grit score includes financial capacity. It is displayed, never blended into
Genius or multiplied into a limit. Only completed, non-overlapping recovery evidence
can earn a small margin concession; neither PD nor financial eligibility is changed.
"""
import numpy as np


def evidence(grit, quality=0, now=None, firm_index=0):
    shape = np.asarray(grit.get('score', [[np.nan]])).shape
    t = shape[1] - 1 if now is None else now
    def last(key, default=0):
        return float(np.asarray(grit.get(key, np.full(shape, default)))[firm_index, t])
    n = last('recovery_n')
    parts = grit.get('recovery_features', {})
    vals = [float(np.asarray(parts.get(k, np.full(shape, np.nan)))[firm_index, t])
            for k in ('gap_regained', 'stayed_recovered', 'funded_by_operations', 'response')]
    # response is normalised to 0..1 by grit.compute, like the other parts.
    valid = all(np.isfinite(v) and 0 <= v <= 1 for v in vals)
    episodes = [e for e in (grit.get('episodes') or []) if e.get('kind') == 'setback' and e.get('firm', 0) == firm_index]
    judged = [e for e in episodes if e.get('judged') is not None and e['judged'] <= t]
    age = t - max((e['judged'] for e in judged), default=-10000)
    unfolding = any(e.get('start', t + 1) <= t and
                    (e.get('judged') is None or e['judged'] > t) for e in episodes)
    index = 100 * float(np.mean(vals)) if valid else None
    eligible = bool(n >= 1 and valid and quality >= 85 and age <= 24 and not unfolding)
    reason = ('Completed recovery evidence qualifies for the demo policy.' if eligible else
              'No adjustment: needs a completed recovery within 24 months, high-quality books and no unresolved setback.')
    return dict(eligible=eligible, recovery_index=index, completed_recoveries=n,
                evidence_weight=n / (n + 2) if n > 0 else 0.0,
                age_months=age if judged else None, reason=reason,
                source='Assumed synthetic-demo policy; not validated on real lending outcomes')


def concession(record, max_bps=50, margin=2):
    """Percentage points, capped by available margin. No penalty for absent evidence."""
    if not record or not record.get('eligible'):
        return 0.0
    strength = np.clip((record['recovery_index'] - 50) / 50, 0, 1)
    freshness = 2 ** (-max(0, record['age_months']) / 24)
    return float(min(max(0, max_bps) / 100 * strength * record['evidence_weight'] * freshness,
                     max(0, margin)))
