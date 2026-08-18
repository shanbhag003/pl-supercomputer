"""Manager transition layer.

Effects are estimated by departing-tenure bucket from 415 transitions since
1993, using only data available before the season being forecast.
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, sys, copy, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, f'{ROOT}/src')
warnings.filterwarnings('ignore')

from simulate import bootstrap_models, fixture_grids, simulate, positions, summarise
from preseason_bt import actual_table

CFG = dict(xi=0.0045, w_xg=0.7, ridge=2.0)
DRIFT, B, N = 0.16, 60, 20000
WEIGHTS = [0.0, 0.5, 1.0]
OUT = f'{ROOT}/data/processed/manager_bt.csv'
BUCKETS = [(0, 365), (365, 1095), (1095, 1825), (1825, 10 ** 9)]

mt = pd.read_parquet(f'{ROOT}/data/processed/matches.parquet')
MGR = pd.read_csv(f'{ROOT}/data/processed/managers.csv',
                  parse_dates=['start', 'end'])
TR = pd.read_csv(f'{ROOT}/data/processed/manager_transitions.csv',
                 parse_dates=['date'])


def bucket(days):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= days < hi:
            return i
    return len(BUCKETS) - 1


def effects(before_date):
    """Effect per tenure bucket, fitted only on transitions before this date."""
    t = TR[TR.date < before_date].copy()
    t['b'] = t.out_days.map(bucket)
    g = t.groupby('b').effect.agg(['mean', 'count'])
    # shrink small buckets toward the overall mean
    overall = t.effect.mean()
    k = 20.0
    return {b: (r['count'] * r['mean'] + k * overall) / (r['count'] + k)
            for b, r in g.iterrows()}, overall


def manager_delta(season, base_xg):
    """Log-scale shift per club from a summer manager change."""
    d = mt[mt.season == season]
    start = d.date.min()
    prev_start = mt[mt.season == season - 1].date.min()
    eff, overall = effects(start)
    out = {}
    for club in sorted(set(d.home)):
        sp = MGR[(MGR.club == club) & (MGR.start < start)].sort_values('start')
        if not len(sp):
            out[club] = 0.0
            continue
        cur = sp.iloc[-1]
        if cur.start <= prev_start:          # same manager as last season
            out[club] = 0.0
            continue
        prev = sp.iloc[-2] if len(sp) > 1 else None
        days = int(prev.days) if prev is not None else 365
        out[club] = eff.get(bucket(days), overall) / base_xg
    return out


def run(season):
    d = mt[mt.season == season].sort_values('date')
    ref = d.date.min() - pd.Timedelta(days=1)
    teams = sorted(set(d.home))
    prev = set(mt[mt.season == season - 1].home)
    base = float(mt[mt.season == season - 1].hnpxg.mean())
    delta = manager_delta(season, base)
    mods0 = bootstrap_models(mt, ref, teams, {t: 0. for t in prev},
                             {t: 0. for t in prev}, B=B, seed=7, **CFG)
    fx = list(zip(d.home, d.away))
    act = actual_table(season)
    out = []
    for w in WEIGHTS:
        mods = copy.deepcopy(mods0)
        r2 = np.random.default_rng(99)
        for m in mods:
            for t in m['teams']:
                s = delta.get(t, 0.0) * w
                m['att'][t] += s / 2 + r2.normal(0, DRIFT)
                m['dfn'][t] += s / 2 + r2.normal(0, DRIFT)
        cum = fixture_grids(mods, fx, teams)
        pts, gd, gf = simulate(cum, fx, teams, N=N, seed=11)
        pos = positions(pts, gd, gf, seed=12)
        s = summarise(pts, pos, teams).merge(act, on='team')
        ti = {t: i for i, t in enumerate(teams)}
        s['pct'] = [(pts[:, ti[t]] < a).mean() for t, a in zip(s.team, s.act_pts)]
        out.append(dict(season=season, weight=w,
                        mae=(s.act_pts - s.xPts).abs().mean(),
                        rankcorr=s[['rank', 'act_pos']].corr(method='spearman').iloc[0, 1],
                        cover80=((s.pct > .10) & (s.pct < .90)).mean(),
                        n_changed=sum(1 for v in delta.values() if v != 0)))
        print(f'  w={w:<4} MAE {out[-1]["mae"]:.2f}  rank {out[-1]["rankcorr"]:.3f}  '
              f'cover {out[-1]["cover80"]:.0%}', flush=True)
    return pd.DataFrame(out), delta


if __name__ == '__main__':
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    rows = done.to_dict('records')
    for s in [int(x) for x in sys.argv[1:]]:
        if len(done) and s in set(done.season):
            print(f'{s} done'); continue
        print(f'--- {s}', flush=True)
        r, dl = run(s)
        rows += r.to_dict('records')
        pd.DataFrame(rows).to_csv(OUT, index=False)
        ch = {k: round(v, 3) for k, v in dl.items() if v != 0}
        print(f'   {len(ch)} clubs changed manager: {ch}', flush=True)
