"""Game-state adjusted xG ("score effects").

A team two goals up sits deep and creates less than it could; the team chasing
the game gets chances it would not get at 0-0. So raw xG from lopsided spells
misstates both sides' strength. This rescales each shot's xG to what it would
be worth at level, using factors measured from the data.

Understat stores each shot's minute and result but not the score when it was
taken, so the score is replayed in shot order. Own goals are listed under the
team whose player scored them and count for the other side - verified by
rebuilding every final score.

States, from the shooting team's side, before the shot:
  lvl  level    up1 / dn1  one goal up / down    up2 / dn2  two or more up / down

Penalties are excluded, matching the non-penalty xG the ratings use.
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import glob
import numpy as np
import pandas as pd

SHOTS = f'{ROOT}/data/understat/shots'
STATES = ['lvl', 'up1', 'dn1', 'up2', 'dn2']
SET_PIECES = {'FromCorner', 'SetPiece', 'DirectFreekick'}   # penalties excluded anyway
FLIP = dict(lvl='lvl', up1='dn1', dn1='up1', up2='dn2', dn2='up2')
T = 90                                   # match length; later goals count at 90


def state_of(diff):
    return ('lvl' if diff == 0 else 'up1' if diff == 1 else 'dn1' if diff == -1
            else 'up2' if diff >= 2 else 'dn2')


def load_shots():
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{SHOTS}/EPL_*.parquet'))],
                     ignore_index=True)


def match_table(shots):
    """One row per match: for each side, non-penalty xG and minutes in each
    state. attrs['unreconciled'] lists matches whose rebuilt score disagrees
    with the recorded final score (their rows are dropped)."""
    s = shots.sort_values(['match_id', 'minute', 'id'])
    rows, bad = [], []
    for mid, g in s.groupby('match_id', sort=False):
        score = {'h': 0, 'a': 0}
        xg = {(side, st): 0.0 for side in 'ha' for st in STATES}
        sp = {(side, st): 0.0 for side in 'ha' for st in STATES}   # set-piece part
        mins = dict.fromkeys(STATES, 0.0)          # from the home side's view
        t_prev = 0
        for r in g.itertuples():
            other = 'a' if r.h_a == 'h' else 'h'
            if r.situation != 'Penalty' and r.result != 'OwnGoal':
                k = (r.h_a, state_of(score[r.h_a] - score[other]))
                xg[k] += r.xG
                if r.situation in SET_PIECES:
                    sp[k] += r.xG
            if r.result in ('Goal', 'OwnGoal'):
                t = min(max(r.minute, t_prev), T)
                mins[state_of(score['h'] - score['a'])] += t - t_prev
                t_prev = t
                score[r.h_a if r.result == 'Goal' else other] += 1
        mins[state_of(score['h'] - score['a'])] += T - t_prev
        if (score['h'], score['a']) != (g.h_goals.iat[0], g.a_goals.iat[0]):
            bad.append(mid)
            continue
        row = dict(match_id=mid, date=pd.Timestamp(g.date.iat[0][:10]),
                   home=g.h_team.iat[0], away=g.a_team.iat[0])
        for st in STATES:
            row[f'h_xg_{st}'] = xg[('h', st)]
            row[f'a_xg_{st}'] = xg[('a', st)]
            row[f'h_sp_{st}'] = sp[('h', st)]
            row[f'a_sp_{st}'] = sp[('a', st)]
            row[f'h_min_{st}'] = mins[st]
            row[f'a_min_{st}'] = mins[FLIP[st]]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs['unreconciled'] = bad
    return out


STEP = dict(lvl=0, up1=1, dn1=-1, up2=2, dn2=-2)


def factors(beta):
    """Scale for each state: exp(beta * goals ahead, capped at 2).

    One parameter rather than measured factors, because measuring them is
    biased both ways. Against a team's season-long level rate, leading looks
    productive (a team that goes 2 up is usually having a good day). Against its
    level rate in the same match, the effect is exaggerated (the side that
    scored first was probably on top at level). The data put the truth between
    the two, so the walk-forward backtest chooses beta. beta=0 is no adjustment.
    """
    return {st: float(np.exp(beta * k)) for st, k in STEP.items()}


def adjusted_npxg(m, beta, w_sp=1.0):
    """Per-side non-penalty xG with each shot rescaled for game state, and
    set-piece xG (corners, free kicks) weighted by w_sp. w_sp=1 is no change."""
    f = factors(beta)
    h = sum(f[st] * (m[f'h_xg_{st}'] + (w_sp - 1) * m[f'h_sp_{st}']) for st in STATES)
    a = sum(f[st] * (m[f'a_xg_{st}'] + (w_sp - 1) * m[f'a_sp_{st}']) for st in STATES)
    return h.values, a.values


def apply(df, beta, w_sp=1.0):
    """Replace hnpxg/anpxg in a match frame with game-state adjusted xG.

    Matches with no shot data - a result football-data has posted before
    understat, filled with goals - keep what they had. Returns (frame, number
    of matches adjusted)."""
    if (beta == 0 and w_sp == 1) or not glob.glob(f'{SHOTS}/EPL_*.parquet'):
        return df, 0
    st = match_table(load_shots())
    h, a = adjusted_npxg(st, beta, w_sp)
    adj = st[['date', 'home', 'away']].assign(h_adj=h, a_adj=a)
    out = df.merge(adj, on=['date', 'home', 'away'], how='left')
    n = int(out.h_adj.notna().sum())
    out['hnpxg'] = out.h_adj.fillna(out.hnpxg)
    out['anpxg'] = out.a_adj.fillna(out.anpxg)
    return out.drop(columns=['h_adj', 'a_adj']), n


# ------------------------------------------------------------------ download
HDR = {'User-Agent': 'Mozilla/5.0', 'X-Requested-With': 'XMLHttpRequest'}
KEEP = ['id', 'match_id', 'minute', 'h_a', 'result', 'situation', 'xG', 'h_team',
        'a_team', 'h_goals', 'a_goals', 'date']


def _fetch(mid, session):
    import json, time
    for attempt in range(3):
        try:
            r = session.get(f'https://understat.com/getMatchData/{mid}',
                            headers=HDR, timeout=30)
            if r.status_code == 200:
                s = json.loads(r.text)['shots']
                return mid, [{k: x.get(k) for k in KEEP} for side in 'ha' for x in s[side]]
        except Exception:
            pass
        time.sleep(2 + 3 * attempt)
    return mid, None


def pull_season(season, session, match_ids=None, workers=4):
    """Fetch shots for any played match of a season not yet cached, one request
    per match, into data/understat/shots/EPL_<season>.parquet. match_ids, if
    given, saves re-reading the league page. Returns (added, failed)."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(SHOTS, exist_ok=True)
    f = f'{SHOTS}/EPL_{season}.parquet'
    have = pd.read_parquet(f) if os.path.exists(f) else pd.DataFrame(columns=KEEP)
    done = set(have.match_id.astype(str))
    if match_ids is None:
        r = session.get(f'https://understat.com/getLeagueData/EPL/{season}',
                        headers=HDR, timeout=40)
        r.raise_for_status()
        match_ids = [m['id'] for m in r.json()['dates'] if m.get('isResult')]
    todo = [str(m) for m in match_ids if str(m) not in done]
    if not todo:
        return 0, 0
    rows, failed = [], 0
    with ThreadPoolExecutor(workers) as ex:
        for mid, shots in ex.map(lambda m: _fetch(m, session), todo):
            if shots is None:
                failed += 1
            else:
                rows += shots
    out = pd.concat([have, pd.DataFrame(rows, columns=KEEP)], ignore_index=True)
    for c in ['id', 'match_id', 'minute', 'h_goals', 'a_goals']:
        out[c] = out[c].astype(int)
    out['xG'] = out.xG.astype(float)
    out.to_parquet(f, index=False)
    return len(todo) - failed, failed
