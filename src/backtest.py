"""Walk-forward match-level backtest. Refits weekly, never sees the future."""
import sys, itertools, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, '/home/claude/pl/src')
from ratings import fit_ratings, outcome_probs, PROMOTED_ATT, PROMOTED_DEF
warnings.filterwarnings('ignore')

df = pd.read_parquet('/home/claude/pl/data/processed/matches.parquet')
df['res'] = np.where(df.hg > df.ag, 0, np.where(df.hg == df.ag, 1, 2))


def devig(row):
    p = np.array([1 / row.AvgH, 1 / row.AvgD, 1 / row.AvgA])
    return p / p.sum()


def rps(p, outcome):
    """Ranked probability score for ordered 3-outcome (H, D, A)."""
    o = np.zeros(3); o[outcome] = 1
    cp, co = np.cumsum(p), np.cumsum(o)
    return np.sum((cp - co) ** 2) / 2


def run_season(season, xi, w_xg, ridge, max_years=4.0, refit_days=7):
    d = df[df.season == season].sort_values('date')
    prev_teams = set(df[df.season == season - 1].home)
    prior_att = {t: 0.0 for t in prev_teams}
    prior_dfn = {t: 0.0 for t in prev_teams}
    teams = sorted(set(d.home))

    out = []
    model = None
    last_fit = None
    for date, chunk in d.groupby('date'):
        if last_fit is None or (date - last_fit).days >= refit_days:
            model = fit_ratings(df, date, xi=xi, w_xg=w_xg, max_years=max_years,
                                ridge=ridge, prior_att=prior_att,
                                prior_dfn=prior_dfn, extra_teams=teams)
            last_fit = date
        for _, m in chunk.iterrows():
            o = outcome_probs(model, m.home, m.away)
            p = np.array([o['H'], o['D'], o['A']])
            rec = dict(date=date, home=m.home, away=m.away, res=m.res,
                       pH=p[0], pD=p[1], pA=p[2],
                       rps=rps(p, m.res), ll=-np.log(max(p[m.res], 1e-9)))
            if not np.isnan(m.AvgH):
                b = devig(m)
                rec['b_rps'] = rps(b, m.res)
                rec['b_ll'] = -np.log(max(b[m.res], 1e-9))
            out.append(rec)
    return pd.DataFrame(out)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'tune'

    if mode == 'tune':
        TUNE = [2019, 2020, 2021, 2022, 2023, 2024]
        grid = list(itertools.product(
            [0.0015, 0.003, 0.0045, 0.007],      # xi (time decay)
            [0.0, 0.4, 0.7, 1.0],                # w_xg
            [0.0, 2.0, 6.0, 15.0],               # ridge
        ))
        res = []
        for xi, w, rg in grid:
            r = pd.concat([run_season(s, xi, w, rg) for s in TUNE])
            res.append(dict(xi=xi, w_xg=w, ridge=rg, rps=r.rps.mean(),
                            ll=r.ll.mean(), n=len(r)))
            print(f'xi={xi:<7} w_xg={w:<4} ridge={rg:<5} '
                  f'RPS={r.rps.mean():.5f} LL={r.ll.mean():.5f}', flush=True)
        pd.DataFrame(res).to_csv('/home/claude/pl/data/processed/tuning.csv',
                                 index=False)
        best = pd.DataFrame(res).sort_values('rps').iloc[0]
        print('\nBEST:', dict(best.round(5)))
