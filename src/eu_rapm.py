"""Player values in La Liga, Serie A, Bundesliga and Ligue 1, converted to
Premier League units using players who have played meaningfully in both.

Each league gets its own RAPM fit. A league's coefficients are only comparable
to ours after conversion, so we regress PL value on foreign value across the
movers and use that mapping for players who have never played in England.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

sys.path.insert(0, '/home/claude/pl/src')
EU_JSON = '/home/claude/pl/data/understat_eu'
EU_PM = '/home/claude/pl/data/processed/eu_player_matches'
PL_PM = '/home/claude/pl/data/processed/player_matches'
LEAGUES = ['La_liga', 'Serie_A', 'Bundesliga', 'Ligue_1']


def load_player_matches(store):
    rows = []
    for f in glob.glob(f'{store}/*.jsonl'):
        for line in open(f):
            r = json.loads(line)
            for m in r['rows']:
                if m.get('date') and m.get('time') is not None:
                    rows.append((r['pid'], m['date'][:10], int(m['season']),
                                 int(m['time']), m['h_team'], m['a_team']))
    d = pd.DataFrame(rows, columns=['pid', 'date', 'season', 'mins',
                                    'h_team', 'a_team'])
    d['date'] = pd.to_datetime(d['date'])
    return d


def league_matches(lg):
    rows, pcl = [], {}
    for f in sorted(glob.glob(f'{EU_JSON}/{lg}_*.json')):
        s = int(f.split('_')[-1].split('.')[0])
        j = json.load(open(f))
        for p in j['players']:
            pcl[(p['id'], s)] = p['team_title']
        for m in j['dates']:
            if not m.get('isResult'):
                continue
            rows.append(dict(season=s, date=m['datetime'][:10],
                             home=m['h']['title'], away=m['a']['title'],
                             hxg=float(m['xG']['h']), axg=float(m['xG']['a'])))
    d = pd.DataFrame(rows)
    d['date'] = pd.to_datetime(d['date'])
    d = d.drop_duplicates(['date', 'home', 'away']).reset_index(drop=True)
    d['mid'] = np.arange(len(d))
    return d, pcl


def fit_league(lg, pm_all, lam=1.0, min_min=900):
    mt, pcl = league_matches(lg)
    pm = pm_all.merge(mt[['date', 'home', 'away', 'mid', 'season']],
                      left_on=['date', 'h_team', 'a_team'],
                      right_on=['date', 'home', 'away'], how='inner',
                      suffixes=('', '_m'))
    pm['club'] = [pcl.get((p, s)) for p, s in zip(pm.pid, pm.season)]
    pm = pm.dropna(subset=['club'])
    pm = pm[(pm.club == pm.home) | (pm.club == pm.away)]
    if not len(pm):
        return {}, {}, 0
    pm['on_home'] = pm.club == pm.home
    pm['share'] = pm.mins / pm.groupby(['mid', 'on_home']).mins.transform('sum').clip(lower=1)
    km = pm.groupby('pid').mins.sum()
    big = set(km[km >= min_min].index)
    pm['key'] = np.where(pm.pid.isin(big), pm.pid, 'POOL')

    players = sorted(set(pm.key)); pidx = {p: i for i, p in enumerate(players)}
    P = len(players)
    seasons = sorted(mt.season.unique()); sidx = {s: i for i, s in enumerate(seasons)}
    ncol = 2 * P + 1 + len(seasons)
    grp = {k: v for k, v in pm.groupby(['mid', 'on_home'])}
    rows, cols, vals, y = [], [], [], []
    for r in mt.itertuples():
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
            y.append(r.hxg if ah else r.axg)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(y), ncol))
    y = np.asarray(y)
    idx = np.arange(2 * P)
    Pen = sparse.csr_matrix((np.full(len(idx), lam), (np.arange(len(idx)), idx)),
                            shape=(len(idx), ncol))
    b = lsqr(sparse.vstack([X, Pen]).tocsr(),
             np.concatenate([y, np.zeros(len(idx))]),
             atol=1e-10, btol=1e-10, iter_lim=4000)[0]
    att = {p: b[i] for p, i in pidx.items() if p != 'POOL'}
    dfn = {p: -b[P + i] for p, i in pidx.items() if p != 'POOL'}
    return att, dfn, len(y)
