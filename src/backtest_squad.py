"""Does squad composition improve pre-season forecasts?

Signal used is the CHANGE in squad quality from one season to the next, so we
don't double-count club strength the team model already knows.

This is the optimistic version: squad minute-shares come from the season being
forecast, i.e. perfect knowledge of who actually played. If that doesn't help,
a realistic lineup forecast certainly won't.
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json, glob, copy, warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
sys.path.insert(0, f'{ROOT}/src')
warnings.filterwarnings('ignore')

from simulate import bootstrap_models, fixture_grids, simulate, positions, summarise
from preseason_bt import actual_table

CFG = dict(xi=0.0045, w_xg=0.7, ridge=2.0)
DRIFT, B, N, LAM = 0.16, 60, 20000, 1.0
WEIGHTS = [0.0, 0.25, 0.5, 1.0]
OUT = f'{ROOT}/data/processed/squad_bt3.csv'
CACHE = f'{ROOT}/data/processed/pm_merged.parquet'

mt = pd.read_parquet(f'{ROOT}/data/processed/matches.parquet')
mt = mt[mt.season >= 2014].reset_index(drop=True)
mt['mid'] = np.arange(len(mt))


def merged():
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    pm = pd.read_parquet(f'{ROOT}/data/processed/player_matches.parquet')
    pm = pm.merge(mt[['date', 'home', 'away', 'mid', 'season']],
                  left_on=['date', 'h_team', 'a_team'],
                  right_on=['date', 'home', 'away'], how='inner', suffixes=('', '_m'))
    pcl = {}
    for f in sorted(glob.glob(f'{ROOT}/data/understat/EPL_*.json')):
        s = int(f.split('_')[-1].split('.')[0])
        for p in json.load(open(f))['players']:
            pcl[(p['id'], s)] = p['team_title']
    pm['club'] = [pcl.get((p, s)) for p, s in zip(pm.pid, pm.season)]
    pm = pm.dropna(subset=['club'])
    pm = pm[(pm.club == pm.home) | (pm.club == pm.away)]
    pm['on_home'] = pm.club == pm.home
    pm['share'] = pm.mins / pm.groupby(['mid', 'on_home']).mins.transform('sum').clip(lower=1)
    pm.to_parquet(CACHE)
    return pm


PM = merged()


def fit_rapm(cutoff):
    """RAPM using only matches strictly before cutoff."""
    m = mt[mt.date < cutoff]
    pm = PM[PM.date < cutoff]
    km = pm.groupby('pid').mins.sum()
    big = set(km[km >= 900].index)
    key = np.where(pm.pid.isin(big), pm.pid, 'POOL')
    pm = pm.assign(key=key)
    players = sorted(set(pm.key))
    pidx = {p: i for i, p in enumerate(players)}
    P = len(players)
    seasons = sorted(m.season.unique())
    sidx = {s: i for i, s in enumerate(seasons)}
    ncol = 2 * P + 1 + len(seasons)
    grp = {k: v for k, v in pm.groupby(['mid', 'on_home'])}
    rows, cols, vals, y = [], [], [], []
    for r in m.itertuples():
        for ah in (True, False):
            a, d = grp.get((r.mid, ah)), grp.get((r.mid, not ah))
            if a is None or d is None:
                continue
            i = len(y)
            for k, sh in zip(a.key, a.share):
                rows.append(i); cols.append(pidx[k]); vals.append(sh)
            for k, sh in zip(d.key, d.share):
                rows.append(i); cols.append(P + pidx[k]); vals.append(sh)
            if ah:
                rows.append(i); cols.append(2 * P); vals.append(1.0)
            rows.append(i); cols.append(2 * P + 1 + sidx[r.season]); vals.append(1.0)
            y.append(r.hnpxg if ah else r.anpxg)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(y), ncol))
    y = np.asarray(y)
    idx = np.arange(2 * P)
    Pen = sparse.csr_matrix((np.full(len(idx), LAM), (np.arange(len(idx)), idx)),
                            shape=(len(idx), ncol))
    b = lsqr(sparse.vstack([X, Pen]).tocsr(),
             np.concatenate([y, np.zeros(len(idx))]),
             atol=1e-10, btol=1e-10, iter_lim=4000)[0]
    att = dict(zip(players, b[:P]))
    dfn = dict(zip(players, -b[P:2 * P]))
    return att, dfn, big


def squad_rating(season, att, dfn, big):
    """Minutes-weighted squad quality per club for one season."""
    p = PM[PM.season == season].copy()
    # Players without enough prior PL minutes are UNKNOWN, not POOL. POOL stays
    # in the fit as a control, but its coefficient describes twelve seasons of
    # fringe players and is far too harsh a prior for an actual new signing.
    p['key'] = np.where(p.pid.isin(big), p.pid, 'UNKNOWN')
    # mirror the live minute cap: nobody exceeds 90 a match
    CAP = 38 * 90
    p['mins'] = p.mins.clip(upper=CAP)
    tot = p.groupby('club').mins.sum()
    p['w'] = p.mins / p.club.map(tot)
    p['a'] = p.key.map(lambda k: att.get(k, 0.0) if k != 'UNKNOWN' else 0.0)
    p['d'] = p.key.map(lambda k: dfn.get(k, 0.0) if k != 'UNKNOWN' else 0.0)
    p['unk'] = p.key == 'UNKNOWN'
    g = p.groupby('club').apply(lambda x: pd.Series(
        dict(A=(x.w * x.a).sum(), D=(x.w * x.d).sum(),
             unknown_share=x.loc[x.unk, 'w'].sum())))
    return g


def run(season):
    d = mt[mt.season == season].sort_values('date')
    ref = d.date.min() - pd.Timedelta(days=1)
    teams = sorted(set(d.home))
    prev = set(mt[mt.season == season - 1].home)
    pa = {t: 0.0 for t in prev}
    pdf = {t: 0.0 for t in prev}

    att, dfn, big = fit_rapm(ref)
    cur = squad_rating(season, att, dfn, big)
    old = squad_rating(season - 1, att, dfn, big)

    # signal = change in squad quality, in xG units -> log-scale shift
    base = float(mt[mt.season == season - 1].hnpxg.mean())
    delta = {}
    for t in teams:
        if t in cur.index and t in old.index:
            dnet = (cur.loc[t, 'A'] + cur.loc[t, 'D']) - (old.loc[t, 'A'] + old.loc[t, 'D'])
            delta[t] = dnet / base
        else:
            delta[t] = 0.0

    mods0 = bootstrap_models(mt, ref, teams, pa, pdf, B=B, seed=7, **CFG)
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
        err = s.act_pts - s.xPts
        out.append(dict(season=season, weight=w, mae=err.abs().mean(),
                        rankcorr=s[['rank', 'act_pos']].corr(method='spearman').iloc[0, 1],
                        cover80=((s.pct > .10) & (s.pct < .90)).mean(),
                        champ_rank=int(s.loc[s.act_pos == 1, 'rank'].iloc[0]),
                        delta_sd=float(np.std(list(delta.values())))))
        print(f'  w={w:<5} MAE {out[-1]["mae"]:.2f}  rank {out[-1]["rankcorr"]:.3f}  '
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
        v = np.array(list(dl.values()))
        print(f'   delta: mean {v.mean():+.3f} sd {v.std():.3f}  '
              f'({(v > 0).sum()} up / {(v < 0).sum()} down)', flush=True)
