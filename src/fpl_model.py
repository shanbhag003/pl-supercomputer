"""Expected FPL points per player per fixture, from information available
before the gameweek.

Two passes, so the backtest stays walk-forward:
  components()  raw expectations for every player-fixture - minutes, goal and
                assist involvement, opponent danger - using only data before
                each gameweek's first kickoff.
  calibrate() / points()   turn components into points with scale factors and
                a bonus model learned from EARLIER seasons only.

Scoring is FPL's 2020/21-2024/25 rules (goals 6/6/5/4 by position, assists 3,
clean sheets 4/4/1/0, -1 per 2 conceded for GKP/DEF, +1 per 3 saves, bonus).
2025/26 added defensive-contribution points, which are not modelled yet.
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy.stats import poisson

GOAL_PTS = {'GKP': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}
CS_PTS = {'GKP': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}
PRIOR_G = {'GKP': 0.0, 'DEF': 0.05, 'MID': 0.10, 'FWD': 0.22}   # share of team xG per 90
PRIOR_A = {'GKP': 0.01, 'DEF': 0.06, 'MID': 0.11, 'FWD': 0.10}
K_SHRINK = 4.0          # team-xG units of prior evidence for a player's shares
HALF_DAYS = 250         # recency half-life for chance shares
N_RECENT = 8            # appearances (incl. unused) for the minutes model
HALF_APPS = 4


def _floor_half(lam):
    """E[floor(G/2)] for G ~ Poisson(lam)."""
    k = np.arange(0, 15)
    return float((poisson.pmf(k, lam) * (k // 2)).sum())


def _floor_third(lam):
    k = np.arange(0, 25)
    return float((poisson.pmf(k, lam) * (k // 3)).sum())


def person_keys(hist, maps):
    """A stable id across seasons: the Understat player id when matched,
    otherwise season+element."""
    key = [maps.get(s, {}).get(e) for s, e in zip(hist.season, hist.element)]
    return [k if k is not None else f'{s}_{e}' for k, s, e in zip(key, hist.season, hist.element)]


def roster_frame(rosters):
    """Understat player-matches with the side's total xG and xA."""
    r = rosters.copy()
    r['club'] = np.where(r.h_a == 'h', r.home, r.away)
    side = r.groupby(['match_id', 'h_a'])[['xG', 'xA']].transform('sum')
    r['team_xg'], r['team_xa'] = side.xG, side.xA
    return r[['player_id', 'date', 'club', 'time', 'xG', 'xA', 'team_xg', 'team_xa']]


def shares(ro, before, pos_of):
    """Each player's share of his side's xG and xA per 90, recency-weighted,
    shrunk toward his position's prior. {player_id: (share_g, share_a)}."""
    r = ro[(ro.date < before) & (ro.date >= before - pd.Timedelta(days=550)) & (ro.time > 0)]
    w = 0.5 ** ((before - r.date).dt.days / HALF_DAYS)
    on = r.time / 90.0
    g = pd.DataFrame({'pid': r.player_id, 'xg': w * r.xG, 'xa': w * r.xA,
                      'tg': w * r.team_xg * on, 'ta': w * r.team_xa * on}).groupby('pid').sum()
    out = {}
    for pid, x in g.iterrows():
        p = pos_of.get(pid, 'MID')
        sg = (x.xg + PRIOR_G[p] * K_SHRINK) / (x.tg + K_SHRINK)
        sa = (x.xa + PRIOR_A[p] * K_SHRINK) / (x.ta + K_SHRINK)
        out[pid] = (sg, sa)
    return out


def minutes_model(h_before):
    """From each person's last N_RECENT fixture rows (0-minute rows count):
    chance of 60+ minutes, of 1-59, expected minutes, saves per 90, yellow
    rate. Recency-weighted. Indexed by person."""
    t = h_before.groupby('person').tail(N_RECENT).copy()
    t['rk'] = t.groupby('person').cumcount(ascending=False)
    t['w'] = 0.5 ** (t.rk / HALF_APPS)
    t['p60'] = (t.minutes >= 60) * t.w
    t['psub'] = ((t.minutes > 0) & (t.minutes < 60)) * t.w
    t['mins'] = t.minutes * t.w
    t['sv'] = t.saves * t.w
    t['yl'] = t.yellow * t.w
    g = t.groupby('person')[['w', 'p60', 'psub', 'mins', 'sv', 'yl']].sum()
    out = pd.DataFrame({'p60': g.p60 / g.w, 'psub': g.psub / g.w, 'emins': g.mins / g.w})
    played = (g.mins / 90).clip(lower=0.25)
    out['saves90'] = g.sv / played
    out['yrate'] = g.yl / (g.p60 + g.psub).clip(lower=0.25)
    return out


def components(hist, rosters, lambdas, maps):
    """Raw expectations for every player-fixture in hist (all seasons
    concatenated), each from data before its gameweek's first kickoff."""
    hist = hist.copy()
    hist['person'] = person_keys(hist, maps)
    pid_of = {k: k for k in set(hist.person) if not str(k).count('_')}
    ro = roster_frame(rosters)
    lam = lambdas.copy()
    lam['date'] = pd.to_datetime(lam.date)
    # last season's position for the share priors
    pos_of = hist.drop_duplicates('person', keep='last').set_index('person').pos.to_dict()
    lam_by = {}
    rows = []
    for (season, gw), g in hist.groupby(['season', 'gw'], sort=True):
        start = g.kickoff.min()
        before = hist[hist.kickoff < start]
        mm = minutes_model(before)
        sh = shares(ro, start.tz_convert(None).normalize(), pos_of)
        if season not in lam_by:
            ls = lam[lam.season == season]
            lam_by[season] = {(h_, a_): (lh_, la_) for h_, a_, lh_, la_ in
                              zip(ls.home, ls.away, ls.lh, ls.la)}
        for r in g.itertuples():
            home, away = (r.club, r.opp) if r.home else (r.opp, r.club)
            if (home, away) not in lam_by[season]:
                continue
            lh, la = lam_by[season][(home, away)]
            lam_for, lam_ag = (lh, la) if r.home else (la, lh)
            m = mm.loc[r.person] if r.person in mm.index else None
            sg, sa = sh.get(r.person if r.person in pid_of else None, (PRIOR_G[r.pos], PRIOR_A[r.pos]))
            rows.append(dict(
                season=season, gw=gw, element=r.element, person=r.person, pos=r.pos, club=r.club,
                opp=r.opp, home=r.home, lam_for=lam_for, lam_ag=lam_ag,
                p60=m.p60 if m is not None else np.nan, psub=m.psub if m is not None else np.nan,
                emins=m.emins if m is not None else np.nan,
                saves90=m.saves90 if m is not None else np.nan, yrate=m.yrate if m is not None else 0.0,
                share_g=sg, share_a=sa, new=m is None,
                # outcomes, for calibration and scoring
                minutes=r.minutes, points=r.points, goals=r.goals, assists=r.assists, cs=r.cs,
                gc=r.gc, saves=r.saves, bonus=r.bonus, xP=r.xP))
    return pd.DataFrame(rows)       # newcomers' gaps are filled by fill_new()


NEWCOMER = dict(p60=0.25, psub=0.15, emins=28.0)


def fill_new(c):
    c = c.copy()
    for k, v in NEWCOMER.items():
        c[k] = c[k].fillna(v)
    c['saves90'] = c.saves90.fillna(c[c.pos == 'GKP'].saves90.median())
    return c


def raw(c):
    """Uncalibrated component expectations."""
    on = c.emins / 90.0
    eg = c.share_g * c.lam_for * on
    ea = c.share_a * c.lam_for * on
    p0 = np.exp(-c.lam_ag)
    pcs = c.p60 * p0
    egc = np.array([_floor_half(l) for l in c.lam_ag]) * on
    esv = c.saves90 * on * c.lam_ag / 1.4
    return eg, ea, pcs, egc, esv


def calibrate(train):
    """Scale factors and a bonus model from earlier seasons' player-fixtures."""
    t = fill_new(train)
    eg, ea, pcs, egc, esv = raw(t)
    played = t.minutes > 0
    cal = dict(g=t.goals.sum() / eg.sum(), a=t.assists.sum() / ea.sum(),
               cs=t.cs[t.pos.isin(['GKP', 'DEF'])].sum() / pcs[t.pos.isin(['GKP', 'DEF'])].sum(),
               sv=t.saves[t.pos == 'GKP'].sum() / max(esv[t.pos == 'GKP'].sum(), 1e-9))
    bonus = {}
    for p in GOAL_PTS:
        x = t[played & (t.pos == p)]
        X = np.column_stack([np.ones(len(x)), x.goals, x.assists, x.cs, (x.minutes >= 60).astype(float),
                             x.saves if p == 'GKP' else np.zeros(len(x))])
        bonus[p] = np.linalg.lstsq(X, x.bonus.values, rcond=None)[0]
    cal['bonus'] = bonus
    return cal


def points(c, cal):
    """Expected FPL points per player-fixture."""
    c = fill_new(c)
    eg, ea, pcs, egc, esv = raw(c)
    eg, ea, pcs, esv = eg * cal['g'], ea * cal['a'], np.clip(pcs * cal['cs'], 0, 1), esv * cal['sv']
    pos = c.pos.values
    gpts = np.array([GOAL_PTS[p] for p in pos])
    cspt = np.array([CS_PTS[p] for p in pos])
    gk_def = np.isin(pos, ['GKP', 'DEF'])
    app = c.p60 * 2 + c.psub * 1
    gc_pen = np.where(gk_def, egc, 0.0)
    sv_pts = np.where(pos == 'GKP', [_floor_third(s) for s in esv], 0.0)
    b = np.array([cal['bonus'][p] for p in pos])
    eb = (b[:, 0] * (c.p60 + c.psub) + b[:, 1] * eg + b[:, 2] * ea + b[:, 3] * pcs
          + b[:, 4] * c.p60 + b[:, 5] * esv).clip(lower=0)
    xpts = app + gpts * eg + 3 * ea + cspt * pcs - gc_pen + sv_pts + eb - c.yrate * (c.p60 + c.psub)
    return pd.DataFrame({'xpts': xpts, 'eg': eg, 'ea': ea, 'pcs': pcs, 'ebonus': eb, 'app': app},
                        index=c.index)
