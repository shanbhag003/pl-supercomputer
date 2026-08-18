"""2026/27 pre-season forecast."""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, f'{ROOT}/src')
from simulate import bootstrap_models, fixture_grids, simulate, positions
warnings.filterwarnings('ignore')

CFG = dict(xi=0.0045, w_xg=0.7, ridge=2.0)
DRIFT, B, N = 0.16, 80, 20000
REF = '2026-08-17'

FIX2US = {'Man Utd': 'Manchester United', 'Man City': 'Manchester City',
          'Newcastle': 'Newcastle United', 'Spurs': 'Tottenham',
          "Nott'm Forest": 'Nottingham Forest'}

df = pd.read_parquet(f'{ROOT}/data/processed/matches.parquet')
fx = pd.read_csv(f'{ROOT}/data/raw/fixtures_2627.csv')
fx['home'] = fx['Home Team'].map(lambda x: FIX2US.get(x, x))
fx['away'] = fx['Away Team'].map(lambda x: FIX2US.get(x, x))
teams = sorted(set(fx.home))
fixtures = list(zip(fx.home, fx.away))
assert len(fixtures) == 380 and len(teams) == 20

prev = set(df[df.season == 2025].home)          # last season's PL teams
prior_att = {t: 0.0 for t in prev}
prior_dfn = {t: 0.0 for t in prev}
print('promoted / no recent PL prior:', sorted(set(teams) - prev))

mods = bootstrap_models(df, REF, teams, prior_att, prior_dfn, B=B, seed=7, **CFG)
rng = np.random.default_rng(99)
for m in mods:
    for t in m['teams']:
        m['att'][t] += rng.normal(0, DRIFT)
        m['dfn'][t] += rng.normal(0, DRIFT)

cum = fixture_grids(mods, fixtures, teams)
pts, gd, gf = simulate(cum, fixtures, teams, N=N, seed=11)
pos = positions(pts, gd, gf, seed=12)

rows = []
for i, t in enumerate(teams):
    p = pos[:, i]
    rows.append(dict(team=t, xPts=pts[:, i].mean(),
                     lo=np.percentile(pts[:, i], 10),
                     hi=np.percentile(pts[:, i], 90),
                     title=(p == 1).mean(), top4=(p <= 4).mean(),
                     top6=(p <= 6).mean(), releg=(p >= 18).mean()))
out = pd.DataFrame(rows).sort_values('xPts', ascending=False).reset_index(drop=True)
out.insert(0, 'pos', np.arange(1, 21))
out.to_csv(f'{ROOT}/data/processed/pred_2627.csv', index=False)

# full position matrix
pm = pd.DataFrame({t: np.bincount(pos[:, i], minlength=22)[1:21] / N
                   for i, t in enumerate(teams)}).T
pm.columns = range(1, 21)
pm.loc[out.team].to_csv(f'{ROOT}/data/processed/posmatrix_2627.csv')

pr = out.copy()
for c in ['title', 'top4', 'top6', 'releg']:
    pr[c] = (100 * pr[c]).round(1)
print(pr.round(1).to_string(index=False))
