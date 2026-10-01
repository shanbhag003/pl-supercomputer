"""Dynamic (Kalman) ratings vs the live static model, match level.

Same data as live: game-state adjusted xG (beta 0.1) blended 70/30 with goals.
The filter runs once through 2014/15-2025/26; every prediction uses only
matches before its snapshot date, on the same 7-day snapshot schedule as
backtest.py's refits, so the comparison is like for like.

Settings are chosen on 2016/17-2018/19 only (2014-15 is burn-in); 2019/20-
2025/26 are held out and compared, match by match, with the live static model
(backtest_gamestate.py, beta 0.1).

    python src/backtest_dynamic.py run
    python src/backtest_dynamic.py report
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, itertools, warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.optimize import minimize_scalar
sys.path.insert(0, f'{ROOT}/src')
import dynamic as D
from ratings import score_matrix
warnings.filterwarnings('ignore')

OUT = f'{ROOT}/data/processed/dynamic_bt.parquet'
TUNE, TEST = [2016, 2017, 2018], list(range(2019, 2026))
GRID = [dict(sigma_w=w, sigma_s=s, kappa=k, phi=ph)
        for w, s, k, ph in itertools.product([0.005, 0.01, 0.02, 0.03],
                                             [0.1, 0.17, 0.25], [1.0, 0.85],
                                             [0.5, 0.75, 1.0])]


def data():
    import backtest_gamestate as bg
    return bg.matches_with(0.1)


def fit_rho(pred, mt):
    """Dixon-Coles low-score parameter on tuning seasons, given the lambdas."""
    j = pred.merge(mt[['date', 'home', 'away', 'hg', 'ag']], on=['date', 'home', 'away'])
    j = j[j.season.isin(TUNE)]
    lh, la, h, a = j.lh.values, j.la.values, j.hg.values, j.ag.values

    def nll(r):
        t = np.ones(len(j))
        m = (h == 0) & (a == 0); t[m] = 1 - lh[m] * la[m] * r
        m = (h == 0) & (a == 1); t[m] = 1 + lh[m] * r
        m = (h == 1) & (a == 0); t[m] = 1 + la[m] * r
        m = (h == 1) & (a == 1); t[m] = 1 - r
        return -np.log(np.clip(t, 1e-9, None)).sum()
    return minimize_scalar(nll, bounds=(-0.25, 0.25), method='bounded').x


def probs(lh, la, rho):
    M = score_matrix(lh, la, rho, 10)
    return np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()


def one(p):
    mt = data()
    pred, _ = D.run(mt, p)
    rho = fit_rho(pred, mt)
    pr = np.array([probs(lh, la, rho) for lh, la in zip(pred.lh, pred.la)])
    pred[['pH', 'pD', 'pA']] = pr
    j = pred.merge(mt[['date', 'home', 'away', 'hg', 'ag']], on=['date', 'home', 'away'])
    j['res'] = np.where(j.hg > j.ag, 0, np.where(j.hg == j.ag, 1, 2))
    for k, v in p.items():
        j[k] = v
    j['rho'] = rho
    return j[j.season >= TUNE[0]]


def run():
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        out = pool.map(one, GRID)
    pd.concat(out, ignore_index=True).to_parquet(OUT)


def score(df):
    P = df[['pH', 'pD', 'pA']].values
    y = df.res.values.astype(int)
    cp, co = np.cumsum(P, 1), np.cumsum(np.eye(3)[y], 1)
    return ((cp - co) ** 2).sum(1) / 2


def report():
    r = pd.read_parquet(OUT)
    r['rps'] = score(r)
    keys = ['sigma_w', 'sigma_s', 'kappa', 'phi']
    g = r.groupby(keys + [r.season.isin(TUNE).rename('tune')]).rps.mean().unstack('tune')
    g.columns = ['test', 'tune']
    g = g.sort_values('tune')
    pd.set_option('display.width', 200)
    print('\n=== best 10 settings by TUNING-season RPS (2016-18), with held-out RPS ===')
    print(g.head(10).round(5).to_string())
    best = g.index[0]
    b = r[(r[keys].values == np.array(best)).all(1) & r.season.isin(TEST)]
    print(f'\nchosen on tuning seasons: {dict(zip(keys, best))}, rho {b.rho.iloc[0]:.3f}')

    # live static model on the same matches
    s = pd.read_parquet(f'{ROOT}/data/processed/gamestate_bt.parquet')
    s = s[s.beta == '0.1'].copy()
    s['rps_static'] = score(s)
    j = b.merge(s[['date', 'home', 'away', 'rps_static']], on=['date', 'home', 'away'])
    d = (j.rps - j.rps_static).values
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    by = j.assign(d=d).groupby('season')[['rps', 'rps_static', 'd']].mean()
    print(f'\n=== held-out 2019-25, {len(j)} matches: dynamic vs live static ===')
    print(f'  dynamic {j.rps.mean():.5f}   static {j.rps_static.mean():.5f}   '
          f'diff {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  '
          f'better in {int((by.d < 0).sum())}/{len(by)} seasons')
    print(by.round(5).to_string())
    early = j.date.dt.month.isin([8, 9])
    print(f'  Aug-Sep only: dynamic {j.rps[early].mean():.5f}  static {j.rps_static[early].mean():.5f}')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'report'
    run() if what == 'run' else report()
