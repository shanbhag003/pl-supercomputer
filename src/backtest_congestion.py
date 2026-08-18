"""Does a European-knockout congestion adjustment improve match forecasts?

Knockout ties only run February to May and affect a handful of clubs, so the
effect is heavily diluted across a full season. We therefore report two things:
RPS over every match, and RPS over only the matches actually affected.

Walk-forward: ratings are refitted weekly and never see a result before
predicting it.
"""
import sys, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, '/home/claude/pl/src')
warnings.filterwarnings('ignore')

from ratings import fit_ratings, score_matrix

CFG = dict(xi=0.0045, w_xg=0.7, ridge=2.0)
WINDOW_DAYS = 4

df = pd.read_parquet('/home/claude/pl/data/processed/matches.parquet')
df['res'] = np.where(df.hg > df.ag, 0, np.where(df.hg == df.ag, 1, 2))
ue = pd.read_csv('/home/claude/pl/data/processed/uefa_fixtures.csv',
                 parse_dates=['date'])

KO = {}
for r in ue[ue.stage == 'KO'].itertuples():
    for eng, team in [(r.eng_home, r.home), (r.eng_away, r.away)]:
        if eng:
            KO.setdefault(team, []).append(r.date)
KO = {k: np.array(sorted(v)) for k, v in KO.items()}


def recent_ko(team, date):
    d = KO.get(team)
    if d is None:
        return False
    gap = (date - d).astype('timedelta64[D]').astype(int)
    return bool(((gap > 0) & (gap <= WINDOW_DAYS)).any())


def rps(p, outcome):
    o = np.zeros(3); o[outcome] = 1
    return float(np.sum((np.cumsum(p) - np.cumsum(o)) ** 2) / 2)


def devig(r):
    p = np.array([1 / r.AvgH, 1 / r.AvgD, 1 / r.AvgA])
    return p / p.sum()


def run(season, k, refit_days=7):
    d = df[df.season == season].sort_values('date')
    prev = set(df[df.season == season - 1].home)
    teams = sorted(set(d.home))
    pa = {t: 0.0 for t in prev}; pdn = {t: 0.0 for t in prev}
    out, model, last = [], None, None
    for date, chunk in d.groupby('date'):
        if last is None or (date - last).days >= refit_days:
            model = fit_ratings(df, date, prior_att=pa, prior_dfn=pdn,
                                extra_teams=teams, **CFG)
            last = date
        for _, m in chunk.iterrows():
            hk, ak = recent_ko(m.home, date), recent_ko(m.away, date)
            # a tired side defends worse: drop its defence rating
            dh = model['dfn'][m.home] - (k if hk else 0)
            da = model['dfn'][m.away] - (k if ak else 0)
            lh = np.clip(np.exp(model['mu'] + model['gamma']
                                + model['att'][m.home] - da), .05, 8)
            la = np.clip(np.exp(model['mu'] + model['att'][m.away] - dh), .05, 8)
            M = score_matrix(lh, la, model['rho'], 10)
            p = np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])
            rec = dict(season=season, date=date, affected=hk or ak,
                       rps=rps(p, m.res))
            if not np.isnan(m.AvgH):
                rec['b_rps'] = rps(devig(m), m.res)
            out.append(rec)
    return pd.DataFrame(out)


if __name__ == '__main__':
    seasons = [int(x) for x in sys.argv[1:]] or list(range(2019, 2026))
    ks = [0.0, 0.035, 0.07, 0.12]
    rows = []
    for s in seasons:
        for k in ks:
            r = run(s, k)
            a = r[r.affected]
            rows.append(dict(season=s, k=k, n=len(r), n_aff=len(a),
                             rps=r.rps.mean(),
                             rps_aff=a.rps.mean() if len(a) else np.nan,
                             b_rps=r.b_rps.mean(),
                             b_rps_aff=a.b_rps.mean() if len(a) else np.nan))
        print(f'{s}: {rows[-1]["n_aff"]} affected matches', flush=True)
    t = pd.DataFrame(rows)
    t.to_csv('/home/claude/pl/data/processed/congestion_bt.csv', index=False)
    g = t.groupby('k').agg(rps=('rps', 'mean'), rps_aff=('rps_aff', 'mean'),
                           n_aff=('n_aff', 'sum'))
    g['d_all'] = g.rps - g.rps.iloc[0]
    g['d_aff'] = g.rps_aff - g.rps_aff.iloc[0]
    print()
    print(g.round(5).to_string())
    print(f"\nbookmaker on affected matches: {t.b_rps_aff.mean():.5f}")
