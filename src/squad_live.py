"""Live squad layer for 2026/27.

Understat has no player data for a season that hasn't started, so current
rosters come from the FPL API and player values come from RAPM fitted on
everything through 2025/26.

Minutes are allocated on a fixed club budget (38 x 11 x 90). Returning players
keep last season's minutes; whatever a club freed up through departures is
redistributed to its new signings in proportion to FPL price. So a club that
sells a key player and buys nobody loses that value, and one that reinvests
gets it partly back.
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, glob, os, re, sys, unicodedata
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, f'{ROOT}/src')

TOTAL_MIN = 38 * 11 * 90
FPL2US = {
    'Man City': 'Manchester City', 'Man Utd': 'Manchester United',
    'Spurs': 'Tottenham', 'Newcastle': 'Newcastle United',
    "Nott'm Forest": 'Nottingham Forest', 'Wolves': 'Wolverhampton Wanderers',
}
CHARS = str.maketrans({'ø': 'o', 'Ø': 'o', 'đ': 'd', 'Đ': 'd', 'ł': 'l',
                       'Ł': 'l', 'ı': 'i', 'æ': 'ae', 'Æ': 'ae', 'œ': 'oe',
                       'ß': 'ss', 'ð': 'd', 'þ': 'th'})


def norm(s):
    s = s.translate(CHARS)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z ]', '', s.lower()).strip()


def fpl_squads():
    b = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/',
                     headers={'User-Agent': 'Mozilla/5.0'}, timeout=40).json()
    tm = {t['id']: FPL2US.get(t['name'], t['name']) for t in b['teams']}
    rows = []
    for e in b['elements']:
        rows.append(dict(
            fpl_id=e['id'], club=tm[e['team']],
            full=norm(f"{e['first_name']} {e['second_name']}"),
            web=norm(e['web_name']),
            cost=e['now_cost'] / 10, status=e['status'],
            chance=e['chance_of_playing_next_round'], news=e['news']))
    return pd.DataFrame(rows)


def understat_last(season=2025):
    if os.path.exists(CACHE_LAST):
        return pd.DataFrame(json.load(open(CACHE_LAST)))
    f = f'{ROOT}/data/understat/EPL_{season}.json'
    out = []
    for p in json.load(open(f))['players']:
        out.append(dict(pid=p['id'], name=p['player_name'],
                        full=norm(p['player_name']),
                        sur=norm(p['player_name']).split()[-1],
                        club=p['team_title'], mins=int(p['time'])))
    return pd.DataFrame(out)


CACHE_EU = f'{ROOT}/data/processed/eu_name_index.json'
CACHE_LAST = f'{ROOT}/data/processed/last_season_players.json'


def european_index():
    """Prefer the precomputed index; fall back to reading the raw league files."""
    if os.path.exists(CACHE_EU):
        return json.load(open(CACHE_EU))
    return _european_index_from_raw()


def _european_index_from_raw():
    """name -> pid for players in the other four leagues, most recent first.

    Without this, a signing who has never played in England has no id at all,
    so a European value can never be attached to them.
    """
    import glob
    out = {}
    for f in sorted(glob.glob(f'{ROOT}/data/understat_eu/*_*.json')):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        for p in j.get('players', []):
            n = norm(p['player_name'])
            out[n] = p['id']
            sur = n.split()[-1] if n else ''
            out.setdefault('SUR:' + sur, p['id'])
    return out


def match(fpl, us, extra=None):
    by_full = dict(zip(us.full, us.pid))
    by_sur_club = {(r.sur, r.club): r.pid for r in us.itertuples()}
    by_sur = {}
    for r in us.itertuples():
        by_sur.setdefault(r.sur, []).append(r.pid)
    pids, how = [], []
    for r in fpl.itertuples():
        if r.full in by_full:
            pids.append(by_full[r.full]); how.append('full')
        elif (r.web, r.club) in by_sur_club:
            pids.append(by_sur_club[(r.web, r.club)]); how.append('surname+club')
        elif len(by_sur.get(r.web, [])) == 1:
            pids.append(by_sur[r.web][0]); how.append('surname')
        elif extra and r.full in extra:
            pids.append(extra[r.full]); how.append('europe')
        elif extra and ('SUR:' + r.web) in extra:
            pids.append(extra['SUR:' + r.web]); how.append('europe-surname')
        else:
            pids.append(None); how.append('none')
    fpl = fpl.copy(); fpl['pid'] = pids; fpl['how'] = how
    return fpl


# How many fixtures a flagged player is assumed to miss. FPL puts return dates
# in free text, not structured fields, so these are documented assumptions.
# 'u' is treated as permanent because it almost always means left the club.
MISSED = {'i': 6, 's': 2, 'd': 1}


def allocate_minutes(sq, remaining=38):
    """Allocate a club's minute budget, discounting flagged players over the
    REMAINING fixtures rather than the whole season."""
    out = []
    for club, g in sq.groupby('club'):
        g = g.copy()
        ret = g.last_mins.fillna(0)
        used = ret.sum()
        if used > TOTAL_MIN:                      # squad kept more than fits
            g['exp_min'] = ret * TOTAL_MIN / used
        else:
            spare = TOTAL_MIN - used
            new = g.last_mins.isna() | (g.last_mins == 0)
            wt = np.where(new, g.cost ** 2, 0.0)
            wt = wt / wt.sum() if wt.sum() > 0 else wt
            g['exp_min'] = ret + spare * wt
        # Availability, scaled to the horizon we are simulating. A two-match
        # suspension with 30 fixtures left should cost ~7% of a player, not 100%.
        rem = max(int(remaining), 1)
        avail = np.ones(len(g))
        for i, (st, ch) in enumerate(zip(g.status, g.chance)):
            if st == 'u':
                avail[i] = 0.0                       # left the club / gone
            elif st in MISSED:
                miss = MISSED[st]
                if st == 'd':
                    miss = miss * (1 - (ch if ch == ch else 50) / 100)
                avail[i] = 1.0 - min(1.0, miss / rem)
        g['exp_min_adj'] = g.exp_min * avail
        if g.exp_min_adj.sum() > 0:
            g['exp_min_adj'] *= TOTAL_MIN / g.exp_min_adj.sum()
        out.append(g)
    return pd.concat(out)


def build(att, dfn, known, base_xg, use_availability=True, remaining=38,
          eu_index=None):
    fpl = match(fpl_squads(), understat_last(), extra=eu_index)
    last = understat_last()
    mins = dict(zip(last.pid, last.mins))
    fpl['last_mins'] = fpl.pid.map(mins)
    sq = allocate_minutes(fpl, remaining=remaining)

    col = 'exp_min_adj' if use_availability else 'exp_min'
    sq['a'] = [att.get(p, 0.0) if p in known else 0.0 for p in sq.pid]
    sq['d'] = [dfn.get(p, 0.0) if p in known else 0.0 for p in sq.pid]
    sq['w'] = sq[col] / TOTAL_MIN
    new = sq.groupby('club').apply(
        lambda x: pd.Series(dict(A=(x.w * x.a).sum(), D=(x.w * x.d).sum(),
                                 unknown=x.loc[~x.pid.isin(known), 'w'].sum())))

    # last season's actual squad rating for the same clubs
    last['a'] = [att.get(p, 0.0) if p in known else 0.0 for p in last.pid]
    last['d'] = [dfn.get(p, 0.0) if p in known else 0.0 for p in last.pid]
    tot = last.groupby('club').mins.sum()
    last['w'] = last.mins / last.club.map(tot)
    old = last.groupby('club').apply(
        lambda x: pd.Series(dict(A=(x.w * x.a).sum(), D=(x.w * x.d).sum())))

    delta = {}
    for c in new.index:
        if c in old.index:
            delta[c] = ((new.loc[c, 'A'] + new.loc[c, 'D'])
                        - (old.loc[c, 'A'] + old.loc[c, 'D'])) / base_xg
        else:
            delta[c] = 0.0        # promoted club: handled by the promoted prior
    return delta, new, old, sq
