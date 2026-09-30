"""Does blending bookmaker odds into the MATCH predictions help?

Not the same question as backtest_market.py, which blended opening-day odds
into pre-season TABLE forecasts and failed. This is per fixture: the published
home/draw/away probabilities, which the scorecard grades.

Odds are football-data's market average PRE-CLOSING odds (AvgH/D/A), collected
Friday afternoon for weekend games and Tuesday afternoon for midweek ones. That
is what the live pipeline can actually fetch before kickoff, so it is what the
backtest uses. Closing odds would flatter the blend.

Model probabilities are the walk-forward ones from backtest.run_season: refit
weekly, never seeing a result before predicting it.

Blends, each fitted only on seasons BEFORE the one it is scored on:
  linear   p = w * market + (1 - w) * model
  loglin   softmax(a * log model + b * log market + c), c = draw/away bias
           (multinomial logistic regression on the two log-probabilities)

    python src/backtest_odds.py          # ~3 min, caches model probabilities
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.optimize import minimize
sys.path.insert(0, f'{ROOT}/src')
warnings.filterwarnings('ignore')

CACHE = f'{ROOT}/data/processed/odds_bt_probs.parquet'
OUT = f'{ROOT}/data/processed/odds_bt.csv'
SEASONS = list(range(2017, 2026))
TEST = list(range(2019, 2026))          # always at least two training seasons
CFG = dict(xi=0.0045, w_xg=0.7, ridge=2.0)


def _season(s):
    from backtest import run_season
    r = run_season(s, **CFG)
    r['season'] = s
    return r


def model_probs():
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    with Pool(min(len(SEASONS), max(1, (os.cpu_count() or 2) - 1))) as p:
        r = pd.concat(p.map(_season, SEASONS), ignore_index=True)
    mt = pd.read_parquet(f'{ROOT}/data/processed/matches.parquet')
    r = r.merge(mt[['date', 'home', 'away', 'AvgH', 'AvgD', 'AvgA']],
                on=['date', 'home', 'away'], how='left')
    r.to_parquet(CACHE)
    return r


def devig(odds):
    inv = 1 / odds
    return inv / inv.sum(1, keepdims=True)


def rps(P, y):
    cp, co = np.cumsum(P, 1), np.cumsum(np.eye(3)[y], 1)
    return ((cp - co) ** 2).sum(1) / 2


def logloss(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))


def loglin(theta, Lm, Lq):
    a, b, cd, ca = theta
    z = a * Lm + b * Lq + np.array([0.0, cd, ca])
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_loglin(Pm, Pq, y):
    Lm, Lq = np.log(Pm), np.log(Pq)
    f = lambda t: logloss(loglin(t, Lm, Lq), y).mean()
    return minimize(f, [0.5, 0.5, 0.0, 0.0], method='L-BFGS-B').x


def fit_linear(Pm, Pq, y):
    ws = np.linspace(0, 1, 101)
    return ws[int(np.argmin([rps(w * Pq + (1 - w) * Pm, y).mean() for w in ws]))]


def main():
    r = model_probs()
    r = r.dropna(subset=['AvgH', 'AvgD', 'AvgA']).reset_index(drop=True)
    y = r.res.values.astype(int)
    Pm = r[['pH', 'pD', 'pA']].values
    Pq = devig(r[['AvgH', 'AvgD', 'AvgA']].values)

    rows, per_match = [], []
    for s in TEST:
        tr, te = (r.season < s).values, (r.season == s).values
        th = fit_loglin(Pm[tr], Pq[tr], y[tr])
        w = fit_linear(Pm[tr], Pq[tr], y[tr])
        P = dict(model=Pm[te], market=Pq[te],
                 linear=w * Pq[te] + (1 - w) * Pm[te],
                 loglin=loglin(th, np.log(Pm[te]), np.log(Pq[te])))
        for k, p in P.items():
            rows.append(dict(season=s, method=k, n=int(te.sum()),
                             rps=rps(p, y[te]).mean(), ll=logloss(p, y[te]).mean(),
                             hit=(p.argmax(1) == y[te]).mean(),
                             draws_called=int((p.argmax(1) == 1).sum()),
                             w=w if k == 'linear' else np.nan,
                             a=th[0] if k == 'loglin' else np.nan,
                             b=th[1] if k == 'loglin' else np.nan))
            per_match.append(pd.DataFrame(dict(season=s, method=k, rps=rps(p, y[te]))))
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    pm = pd.concat(per_match)

    pd.set_option('display.width', 200)
    print('\n=== RPS by held-out season (lower is better) ===')
    print(res.pivot(index='season', columns='method', values='rps').round(5).to_string())
    print('\n=== fitted blend weights (on seasons before each) ===')
    print(res[res.method.isin(['linear', 'loglin'])]
          .pivot(index='season', columns='method', values=['w', 'a', 'b'])
          .dropna(axis=1, how='all').round(3).to_string())

    print(f'\n=== pooled, {len(pm) // 4} matches ===')
    agg = res.groupby('method').apply(lambda g: pd.Series(dict(
        rps=np.average(g.rps, weights=g.n), ll=np.average(g.ll, weights=g.n),
        hit=np.average(g.hit, weights=g.n), draws_called=g.draws_called.sum())))
    print(agg.round(5).to_string())

    print('\n=== paired per-match RPS differences ===')
    base = pm[pm.method == 'model'].rps.values
    mkt = pm[pm.method == 'market'].rps.values
    rng = np.random.default_rng(0)
    for k in ('market', 'linear', 'loglin'):
        v = pm[pm.method == k].rps.values
        for name, ref in (('model', base), ('market', mkt)):
            if k == name:
                continue
            d = v - ref
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            wins = int((res[res.method == k].rps.values
                        < res[res.method == name].rps.values).sum())
            print(f'  {k:<7} vs {name:<7} {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  '
                  f'better in {wins}/{len(TEST)} seasons')


if __name__ == '__main__':
    main()
