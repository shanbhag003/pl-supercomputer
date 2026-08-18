"""Pre-season backtest across many seasons + measure true season-to-season drift."""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, warnings, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, f'{ROOT}/src')
from ratings import fit_ratings
from simulate import bootstrap_models, fixture_grids, simulate, positions, summarise
warnings.filterwarnings('ignore')

df = pd.read_parquet(f'{ROOT}/data/processed/matches.parquet')
CFG = dict(xi=0.0045, w_xg=0.7, ridge=2.0)


def actual_table(season):
    d = df[df.season == season]
    st = {}
    for _, r in d.iterrows():
        for t in (r.home, r.away):
            st.setdefault(t, [0, 0, 0])
        st[r.home][1] += r.hg; st[r.home][2] += r.ag
        st[r.away][1] += r.ag; st[r.away][2] += r.hg
        if r.hg > r.ag: st[r.home][0] += 3
        elif r.ag > r.hg: st[r.away][0] += 3
        else: st[r.home][0] += 1; st[r.away][0] += 1
    a = pd.DataFrame([(k, v[0], v[1] - v[2]) for k, v in st.items()],
                     columns=['team', 'act_pts', 'act_gd'])
    a = a.sort_values(['act_pts', 'act_gd'], ascending=False).reset_index(drop=True)
    a['act_pos'] = np.arange(1, len(a) + 1)
    return a


def measure_drift(seasons):
    """How much does a team's true rating move between consecutive seasons,
    beyond what estimation error explains?"""
    rows = []
    for s in seasons:
        d0, d1 = df[df.season == s - 1], df[df.season == s]
        m0 = fit_ratings(d0, d0.date.max() + pd.Timedelta(days=1), xi=0.0, max_years=2)
        m1 = fit_ratings(d1, d1.date.max() + pd.Timedelta(days=1), xi=0.0, max_years=2)
        for t in set(m0['teams']) & set(m1['teams']):
            rows.append(dict(season=s, team=t,
                             d_att=m1['att'][t] - m0['att'][t],
                             d_dfn=m1['dfn'][t] - m0['dfn'][t]))
    return pd.DataFrame(rows)


def preseason_forecast(season, B=80, N=20000, drift=0.0, seed=0):
    d = df[df.season == season].sort_values('date')
    ref = d.date.min() - pd.Timedelta(days=1)
    teams = sorted(set(d.home))
    prev = set(df[df.season == season - 1].home)
    pa = {t: 0.0 for t in prev}; pdf = {t: 0.0 for t in prev}
    mods = bootstrap_models(df, ref, teams, pa, pdf, B=B, seed=seed, **CFG)
    if drift > 0:
        rng = np.random.default_rng(seed + 77)
        for m in mods:
            for t in m['teams']:
                m['att'][t] += rng.normal(0, drift)
                m['dfn'][t] += rng.normal(0, drift)
    fx = list(zip(d.home, d.away))
    cum = fixture_grids(mods, fx, teams)
    pts, gd, gf = simulate(cum, fx, teams, N=N, seed=seed + 1)
    pos = positions(pts, gd, gf, seed=seed + 2)
    s = summarise(pts, pos, teams)
    return s, pts, teams


def evaluate(season, s, pts, teams):
    act = actual_table(season)
    p = s.merge(act, on='team')
    ti = {t: i for i, t in enumerate(teams)}
    p['pct'] = [(pts[:, ti[t]] < a).mean() for t, a in zip(p.team, p.act_pts)]
    p['err'] = p.act_pts - p.xPts
    p['season'] = season
    return p


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'drift':
        dr = measure_drift(range(2016, 2026))
        print(f'n={len(dr)} team-seasons')
        print(f'SD of att change: {dr.d_att.std():.3f}   '
              f'SD of def change: {dr.d_dfn.std():.3f}')
        dr.to_csv(f'{ROOT}/data/processed/drift.csv', index=False)
    else:
        season, drift = int(sys.argv[2]), float(sys.argv[3])
        s, pts, teams = preseason_forecast(season, drift=drift)
        p = evaluate(season, s, pts, teams)
        cov = ((p.pct > .10) & (p.pct < .90)).mean()
        print(f'{season} drift={drift}: MAE={p.err.abs().mean():.2f}  '
              f'cover80={cov:.0%}  '
              f'rankcorr={p[["rank","act_pos"]].corr(method="spearman").iloc[0,1]:.3f}')
        p.to_csv(f'{ROOT}/data/processed/eval_{season}_{drift}.csv',
                 index=False)
