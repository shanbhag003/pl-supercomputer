"""Does the drift noise in the live match predictions cost accuracy?

Live fixture probabilities are the average scoreline grid over 40 bootstrap
models, each with the season-simulation drift added to every rating:
N(0, drift), drift = max(0.04, 0.16 * (1 - games/12)). Averaging over noisy
ratings flattens the probabilities, most of all early in the season. The
match-level validation (backtest.py, and the 1.1% market result) scored a
single point fit with no noise - not what is published.

Drift is right for the season simulation: one draw of ratings persists across
all 38 matches, which is what makes title odds honest. Whether it is right for
a single match is an empirical question, so this scores, walk-forward:

  point      one fit, no bootstrap, no noise       (what backtest.py validates)
  boot       bootstrap average, no noise
  noise_x    bootstrap average with drift * x,     x = 0.5, 1 (live), 2

Same seasons, same weekly refits, same bootstrap draws for every variant.

    python src/backtest_flatten.py run       # ~20-30 min, one season per core
    python src/backtest_flatten.py report
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, time, warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
sys.path.insert(0, f'{ROOT}/src')
from ratings import fit_ratings, score_matrix
from simulate import bootstrap_models
warnings.filterwarnings('ignore')

df = pd.read_parquet(f'{ROOT}/data/processed/matches.parquet')
df['res'] = np.where(df.hg > df.ag, 0, np.where(df.hg == df.ag, 1, 2))
OUT = f'{ROOT}/data/processed/flatten_bt.parquet'
SEASONS = list(range(2018, 2026))
CFG = dict(xi=0.0045, w_xg=0.7, ridge=2.0)
B = 40                                  # live uses mods[:40]
SCALES = [0.5, 1.0, 2.0]


def live_drift(games):
    return max(0.04, 0.16 * (1 - games / 12))


def probs(M):
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())


def grid(m, h, a, att=None, dfn=None):
    att = att or m['att']; dfn = dfn or m['dfn']
    lh = np.clip(np.exp(m['mu'] + m['gamma'] + att[h] - dfn[a]), .05, 8)
    la = np.clip(np.exp(m['mu'] + att[a] - dfn[h]), .05, 8)
    return score_matrix(lh, la, m['rho'], 10)


def season(s):
    t0 = time.time()
    d = df[df.season == s].sort_values('date')
    teams = sorted(set(d.home) | set(d.away))
    ti ={t: i for i, t in enumerate(teams)}
    prev = set(df[df.season == s - 1].home)
    pa = {t: 0.0 for t in prev}; pdf_ = {t: 0.0 for t in prev}
    rng = np.random.default_rng(s)
    rows, last = [], None
    for date, chunk in d.groupby('date'):
        if last is None or (date - last).days >= 7:
            point = fit_ratings(df, date, prior_att=pa, prior_dfn=pdf_,
                                extra_teams=teams, **CFG)
            mods = bootstrap_models(df, date, teams, pa, pdf_, B=B,
                                    seed=int(date.strftime('%Y%m%d')), **CFG)
            games = 2 * (d.date < date).sum() / len(teams)
            dr = live_drift(games)
            z = rng.normal(size=(len(mods), len(teams), 2))
            last = date
        for r in chunk.itertuples():
            rec = dict(season=s, date=date, home=r.home, away=r.away, res=r.res,
                       games=games, drift=dr)
            rec['point'] = probs(grid(point, r.home, r.away))
            rec['boot'] = probs(np.mean([grid(m, r.home, r.away) for m in mods], 0))
            for x in SCALES:
                Ms = []
                for b, m in enumerate(mods):
                    att, dfn = dict(m['att']), dict(m['dfn'])
                    for t in (r.home, r.away):
                        att[t] += x * dr * z[b, ti[t], 0]
                        dfn[t] += x * dr * z[b, ti[t], 1]
                    Ms.append(grid(m, r.home, r.away, att, dfn))
                rec[f'noise_{x:g}'] = probs(np.mean(Ms, 0))
            rows.append(rec)
    print(f'  {s}: {len(rows)} matches, {time.time() - t0:.0f}s', flush=True)
    return rows


def run():
    with Pool(min(len(SEASONS), os.cpu_count() or 2)) as p:
        rows = sum(p.map(season, SEASONS), [])
    out = pd.DataFrame(rows)
    for k in ['point', 'boot'] + [f'noise_{x:g}' for x in SCALES]:
        out[[f'{k}_H', f'{k}_D', f'{k}_A']] = pd.DataFrame(out.pop(k).tolist())
    out.to_parquet(OUT)


def rps(P, y):
    cp, co = np.cumsum(P, 1), np.cumsum(np.eye(3)[y], 1)
    return ((cp - co) ** 2).sum(1) / 2


def report():
    r = pd.read_parquet(OUT)
    y = r.res.values.astype(int)
    V = ['point', 'boot'] + [f'noise_{x:g}' for x in SCALES]
    P = {k: r[[f'{k}_H', f'{k}_D', f'{k}_A']].values for k in V}
    S = {k: rps(P[k], y) for k in V}
    L = {k: -np.log(np.clip(P[k][np.arange(len(y)), y], 1e-12, 1)) for k in V}
    stage = pd.cut(r.games, [-1, 6, 12, 99], labels=['GW1-6', 'GW7-12', 'GW13+'])

    pd.set_option('display.width', 200)
    print(f'\n=== {len(r)} matches, seasons {r.season.min()}-{r.season.max()} ===')
    t = pd.DataFrame({k: dict(rps=S[k].mean(), logloss=L[k].mean(),
                              hit=(P[k].argmax(1) == y).mean(),
                              fav_prob=P[k].max(1).mean()) for k in V}).T
    print(t.round(5).to_string())

    print('\n=== RPS by stage of season (drift is largest early) ===')
    print(pd.DataFrame({k: pd.Series(S[k]).groupby(stage.values).mean() for k in V})
          .round(5).to_string())

    print('\n=== RPS by season ===')
    print(pd.DataFrame({k: pd.Series(S[k]).groupby(r.season.values).mean() for k in V})
          .round(5).to_string())

    print('\n=== paired vs live (noise_1), per-match RPS ===')
    rng = np.random.default_rng(0)
    for k in V:
        if k == 'noise_1':
            continue
        dlt = S[k] - S['noise_1']
        bs = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ws = (pd.Series(dlt).groupby(r.season.values).mean() < 0).sum()
        early = dlt[(stage == 'GW1-6').values].mean()
        print(f'  {k:<9} {dlt.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  '
              f'better in {ws}/{r.season.nunique()} seasons   GW1-6: {early:+.5f}')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'report'
    run() if what == 'run' else report()
