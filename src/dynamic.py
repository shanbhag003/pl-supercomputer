"""Dynamic club ratings: a Kalman filter over attack and defence.

The static model (ratings.py) refits on every run with exponentially decaying
match weights: every club is assumed to change at the same rate, and the fit is
shrunk toward a fixed prior. Here each club's attack and defence are states
that drift over time with an uncertainty attached:

  between match days   variance grows by sigma_w^2 per week
  over the summer      variance grows by sigma_s^2; means shrink by kappa
  promoted clubs       restart at the promoted-club prior (ratings.py)
  league level, home advantage   also states, drifting slowly - so the 2020/21
                       empty stadiums can lower home advantage and recover

Each match's blended target (as ratings.blend_target) is a quasi-Poisson
observation of exp(mu [+ gamma] + att - def), with dispersion phi. Updates are
an iterated extended Kalman filter: a club the model is unsure about moves
more, a well-established one less.

One pass through history yields walk-forward predictions for every match, so
tuning needs no refits.
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from ratings import blend_target, PROMOTED_ATT, PROMOTED_DEF, PROMOTED_SD

DEFAULTS = dict(sigma_w=0.01, sigma_s=0.17, kappa=1.0, phi=0.75,
                sigma_mu=0.005, sigma_g=0.01, w_xg=0.7)


class Filter:
    def __init__(self, teams, p=None):
        self.p = {**DEFAULTS, **(p or {})}
        self.teams = sorted(teams)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        T = len(self.teams)
        self.n = 2 * T + 2
        self.MU, self.G = 2 * T, 2 * T + 1
        self.m = np.zeros(self.n)
        self.m[self.MU], self.m[self.G] = np.log(1.35), 0.25
        self.P = np.diag([PROMOTED_SD ** 2] * (2 * T) + [0.1 ** 2, 0.1 ** 2])
        self.date = None
        self.league = set()

    def a(self, t):
        return 2 * self.idx[t]

    def d(self, t):
        return 2 * self.idx[t] + 1

    # ------------------------------------------------------------ process
    def advance(self, date):
        """Grow uncertainty for the time since the last update."""
        if self.date is not None:
            wk = max((date - self.date).days, 0) / 7.0
            q = np.zeros(self.n)
            q[:2 * len(self.teams)] = self.p['sigma_w'] ** 2 * wk
            q[self.MU] = self.p['sigma_mu'] ** 2 * wk
            q[self.G] = self.p['sigma_g'] ** 2 * wk
            self.P[np.diag_indices(self.n)] += q
        self.date = date

    def new_season(self, teams, prior_shift=None):
        """Summer: continuing clubs shrink by kappa and gain sigma_s^2 of
        variance; promoted clubs restart at the promoted prior. prior_shift
        {team: (d_att, d_def)} lets the squad/manager layers move the start."""
        teams = set(teams)
        k, s2 = self.p['kappa'], self.p['sigma_s'] ** 2
        first = not self.league           # burn-in: nobody is "promoted" yet
        for t in teams:
            ia, idf = self.a(t), self.d(t)
            if first:
                for i in (ia, idf):
                    self.m[i] = 0.0
                    self.P[i, i] = 0.3 ** 2
            elif t in self.league:
                for i in (ia, idf):
                    self.m[i] *= k
                    self.P[i, :] *= k; self.P[:, i] *= k
                    self.P[i, i] += s2
            else:
                for i, mu0 in ((ia, PROMOTED_ATT), (idf, PROMOTED_DEF)):
                    self.P[i, :] = 0; self.P[:, i] = 0
                    self.m[i] = mu0
                    self.P[i, i] = PROMOTED_SD ** 2
            if prior_shift and t in prior_shift:
                self.m[ia] += prior_shift[t][0]
                self.m[idf] += prior_shift[t][1]
        self.league = teams

    # -------------------------------------------------------- observation
    def rows(self, home, away):
        X = np.zeros((2, self.n))
        X[0, [self.MU, self.G, self.a(home)]] = 1; X[0, self.d(away)] = -1
        X[1, [self.MU, self.a(away)]] = 1; X[1, self.d(home)] = -1
        return X

    def lambdas(self, home, away, m=None):
        eta = self.rows(home, away) @ (self.m if m is None else m)
        return np.exp(np.clip(eta, -3, 2.5))

    def update(self, home, away, yh, ya, iters=3):
        """Iterated EKF step for one match (two quasi-Poisson observations)."""
        X, y = self.rows(home, away), np.array([yh, ya])
        m0, P, mi = self.m, self.P, self.m.copy()
        for _ in range(iters):
            lam = np.exp(np.clip(X @ mi, -3, 2.5))
            H = lam[:, None] * X                       # d lambda / d state
            S = H @ P @ H.T + np.diag(self.p['phi'] * lam)
            K = P @ H.T @ np.linalg.inv(S)
            mi = m0 + K @ (y - lam - H @ (m0 - mi))
        self.m = mi
        self.P = P - K @ H @ P
        self.P = (self.P + self.P.T) / 2

    def snapshot(self):
        return self.m.copy(), self.P.copy()


def run(df, p=None, snapshot_days=7, season_starts=None):
    """Filter through df (sorted by date), predicting each match from the most
    recent snapshot - taken, like backtest.py's refits, at a season's first
    match day and then whenever 7+ days have passed. Returns (predictions
    frame with lambdas, final Filter)."""
    p = {**DEFAULTS, **(p or {})}
    df = df.sort_values('date', kind='stable')
    f = Filter(set(df.home) | set(df.away), p)
    yh_all, ya_all = blend_target(df, p['w_xg'])
    df = df.assign(tgt_h=yh_all, tgt_a=ya_all)
    out = []
    for season, sd in df.groupby('season', sort=True):
        f.new_season(set(sd.home) | set(sd.away))
        snap, last = None, None
        for date, day in sd.groupby('date', sort=True):
            f.advance(date)
            if last is None or (date - last).days >= snapshot_days:
                snap, last = f.snapshot(), date
            for r in day.itertuples():
                lh, la = f.lambdas(r.home, r.away, snap[0])
                out.append((season, date, r.home, r.away, lh, la))
            for r in day.itertuples():
                if np.isfinite(r.tgt_h) and np.isfinite(r.tgt_a):
                    f.update(r.home, r.away, r.tgt_h, r.tgt_a)
    pred = pd.DataFrame(out, columns=['season', 'date', 'home', 'away', 'lh', 'la'])
    return pred, f
