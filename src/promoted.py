"""Empirical prior for newly promoted teams.

For each season, fit single-season ratings (no decay) and record what promoted
teams actually turned out to be. Then test whether Championship performance
adds signal on top of the flat average.
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, glob
import numpy as np
import pandas as pd
sys.path.insert(0, f'{ROOT}/src')
from ratings import fit_ratings

RAW = f'{ROOT}/data/raw'
df = pd.read_parquet(f'{ROOT}/data/processed/matches.parquet')

# ---- who was promoted each season ----
seasons = sorted(df.season.unique())
promoted = {}
prev = None
for s in seasons:
    ts = set(df[df.season == s].home)
    if prev is not None:
        promoted[s] = sorted(ts - prev)
    prev = ts

# ---- realised first-season ratings ----
rec = []
for s, teams in promoted.items():
    d = df[df.season == s]
    m = fit_ratings(d, d.date.max() + pd.Timedelta(days=1), xi=0.0, max_years=2)
    for t in teams:
        rec.append(dict(season=s, team=t, att=m['att'][t], dfn=m['dfn'][t],
                        net=m['att'][t] + m['dfn'][t]))
    # league spread for reference
r = pd.DataFrame(rec)

# ---- Championship table the year before ----
def champ_table(code):
    f = f'{RAW}/E1_{code}.csv'
    try:
        c = pd.read_csv(f, encoding='latin-1')
    except FileNotFoundError:
        return None
    c = c.dropna(subset=['HomeTeam', 'FTHG'])
    st = {}
    for _, x in c.iterrows():
        h, a, hg, ag = x['HomeTeam'], x['AwayTeam'], int(x['FTHG']), int(x['FTAG'])
        for t in (h, a):
            st.setdefault(t, dict(pts=0, gf=0, ga=0, pl=0))
        st[h]['gf'] += hg; st[h]['ga'] += ag; st[h]['pl'] += 1
        st[a]['gf'] += ag; st[a]['ga'] += hg; st[a]['pl'] += 1
        if hg > ag: st[h]['pts'] += 3
        elif ag > hg: st[a]['pts'] += 3
        else: st[h]['pts'] += 1; st[a]['pts'] += 1
    t = pd.DataFrame(st).T
    t['gd'] = t.gf - t.ga
    return t

FD2US = {'Man City': 'Manchester City', 'Man United': 'Manchester United',
         'Newcastle': 'Newcastle United', 'Wolves': 'Wolverhampton Wanderers',
         "Nott'm Forest": 'Nottingham Forest', 'West Brom': 'West Bromwich Albion'}

rows = []
for _, x in r.iterrows():
    s = x.season
    code = f'{str(s-1)[2:]}{str(s)[2:]}'
    ct = champ_table(code)
    if ct is None:
        continue
    ct.index = [FD2US.get(i, i) for i in ct.index]
    if x.team in ct.index:
        rows.append(dict(season=s, team=x.team, net=x.net, att=x.att, dfn=x.dfn,
                         c_pts=ct.loc[x.team, 'pts'], c_gd=ct.loc[x.team, 'gd']))
q = pd.DataFrame(rows)

print(f'promoted teams with matched Championship data: {len(q)}')
print(f'\nMean first-season PL rating (promoted):  att {q.att.mean():+.3f}  '
      f'def {q.dfn.mean():+.3f}  net {q.net.mean():+.3f}')
print(f'SD of net: {q.net.std():.3f}')

# league-wide spread for comparison
allr = []
for s in seasons:
    d = df[df.season == s]
    m = fit_ratings(d, d.date.max() + pd.Timedelta(days=1), xi=0.0, max_years=2)
    allr += [m['att'][t] + m['dfn'][t] for t in m['teams']]
print(f'SD of net across all PL teams: {np.std(allr):.3f}')

# does Championship form predict PL rating?
for v in ['c_pts', 'c_gd']:
    c = np.corrcoef(q[v], q.net)[0, 1]
    print(f'corr(Championship {v}, PL net rating) = {c:+.3f}')

b = np.polyfit(q.c_pts, q.net, 1)
q['pred'] = np.polyval(b, q.c_pts)
print(f'\nregression: net = {b[0]:+.5f} * champ_pts {b[1]:+.3f}')
print(f'MAE flat-average prior : {np.abs(q.net - q.net.mean()).mean():.3f}')
print(f'MAE regression prior   : {np.abs(q.net - q.pred).mean():.3f}')
q.to_csv(f'{ROOT}/data/processed/promoted_prior.csv', index=False)
print()
print(q.sort_values('net', ascending=False).head(5).round(3).to_string(index=False))
print(q.sort_values('net').head(3).round(3).to_string(index=False))
