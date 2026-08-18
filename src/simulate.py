"""Monte Carlo season simulator.

Two sources of randomness, both required:
  1. match randomness  - sample a scoreline from each fixture's DC probability grid
  2. parameter uncertainty - bootstrap the ratings fit, resample which ratings apply
Skipping (2) makes title probabilities far too confident.
"""
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
from ratings import fit_ratings, score_matrix
warnings.filterwarnings('ignore')

MAXG = 10


def bootstrap_models(df, ref_date, teams, prior_att, prior_dfn,
                     B=100, seed=0, **fitkw):
    """Refit ratings on B bootstrap resamples of the historical matches."""
    rng = np.random.default_rng(seed)
    hist = df[df.date < pd.Timestamp(ref_date)]
    models = []
    for b in range(B):
        s = hist.sample(len(hist), replace=True, random_state=int(rng.integers(1e9)))
        try:
            models.append(fit_ratings(s, ref_date, prior_att=prior_att,
                                      prior_dfn=prior_dfn, extra_teams=teams,
                                      **fitkw))
        except Exception:
            continue
    return models


def fixture_grids(models, fixtures, teams):
    """[B, M, (MAXG+1)^2] cumulative scoreline probabilities."""
    B, M, K = len(models), len(fixtures), (MAXG + 1) ** 2
    cum = np.zeros((B, M, K))
    for bi, m in enumerate(models):
        att, dfn = m['att'], m['dfn']
        for mi, (h, a) in enumerate(fixtures):
            lh = np.clip(np.exp(m['mu'] + m['gamma'] + att[h] - dfn[a]), .05, 8)
            la = np.clip(np.exp(m['mu'] + att[a] - dfn[h]), .05, 8)
            cum[bi, mi] = np.cumsum(score_matrix(lh, la, m['rho'], MAXG).ravel())
    return cum


def simulate(cum, fixtures, teams, N=20000, start=None, seed=1):
    """Returns points, goal difference, goals for  [N, T]."""
    rng = np.random.default_rng(seed)
    B, M, _ = cum.shape
    T = len(teams)
    ti = {t: i for i, t in enumerate(teams)}
    hidx = np.array([ti[h] for h, a in fixtures])
    aidx = np.array([ti[a] for h, a in fixtures])

    pts = np.zeros((N, T), dtype=np.int32)
    gf = np.zeros((N, T), dtype=np.int32)
    ga = np.zeros((N, T), dtype=np.int32)
    if start is not None:
        for t, (p, f, a) in start.items():
            pts[:, ti[t]] += p; gf[:, ti[t]] += f; ga[:, ti[t]] += a

    per = int(np.ceil(N / B))
    row = 0
    for b in range(B):
        n = min(per, N - row)
        if n <= 0:
            break
        u = rng.random((n, M))
        for mi in range(M):
            k = np.searchsorted(cum[b, mi], u[:, mi])
            hg = (k // (MAXG + 1)).astype(np.int32)
            ag = (k % (MAXG + 1)).astype(np.int32)
            sl = slice(row, row + n)
            gf[sl, hidx[mi]] += hg; ga[sl, hidx[mi]] += ag
            gf[sl, aidx[mi]] += ag; ga[sl, aidx[mi]] += hg
            pts[sl, hidx[mi]] += np.where(hg > ag, 3, (hg == ag).astype(np.int32))
            pts[sl, aidx[mi]] += np.where(ag > hg, 3, (hg == ag).astype(np.int32))
        row += n
    return pts, gf - ga, gf


def positions(pts, gd, gf, seed=2):
    """PL tiebreak: points, then GD, then goals scored, then random (playoff)."""
    rng = np.random.default_rng(seed)
    N, T = pts.shape
    key = (pts.astype(np.float64) * 1e9 + (gd + 200) * 1e5 + gf * 1e1
           + rng.random((N, T)))
    order = np.argsort(-key, axis=1)
    pos = np.empty_like(order)
    np.put_along_axis(pos, order, np.arange(1, T + 1)[None, :].repeat(N, 0), axis=1)
    return pos


def summarise(pts, pos, teams):
    T = len(teams)
    rows = []
    for i, t in enumerate(teams):
        p = pos[:, i]
        rows.append(dict(
            team=t, xPts=pts[:, i].mean(),
            pts_lo=np.percentile(pts[:, i], 10), pts_hi=np.percentile(pts[:, i], 90),
            avg_pos=p.mean(),
            title=(p == 1).mean(), top4=(p <= 4).mean(), top6=(p <= 6).mean(),
            releg=(p >= T - 2).mean(),
        ))
    d = pd.DataFrame(rows).sort_values('xPts', ascending=False).reset_index(drop=True)
    d.insert(0, 'rank', np.arange(1, len(d) + 1))
    return d
