"""FPL expected points: data, player matching and the points model.

FPL history comes from the vaastav/Fantasy-Premier-League archive
(data/fpl_history/merged_gw_<season>.csv, one row per player per fixture):
real points, minutes and the components that score them, plus FPL's own
expected points (xP) in 2020/21-2024/25. Chance creation comes from the
Understat rosters (players.py), team goal expectations from the model.
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import re, unicodedata
import numpy as np
import pandas as pd

HIST = f'{ROOT}/data/fpl_history'
FPL2US = {
    'Man City': 'Manchester City', 'Man Utd': 'Manchester United', 'Spurs': 'Tottenham',
    'Newcastle': 'Newcastle United', "Nott'm Forest": 'Nottingham Forest',
    'Wolves': 'Wolverhampton Wanderers', 'Sheffield Utd': 'Sheffield United',
    'West Brom': 'West Bromwich Albion', 'Hull City': 'Hull', 'Ipswich Town': 'Ipswich',
    'Coventry City': 'Coventry', 'Leeds United': 'Leeds', 'Leicester City': 'Leicester',
    'Norwich City': 'Norwich', 'Luton Town': 'Luton',
}
CHARS = str.maketrans({'ø': 'o', 'Ø': 'o', 'đ': 'd', 'Đ': 'd', 'ł': 'l', 'Ł': 'l',
                       'ı': 'i', 'æ': 'ae', 'Æ': 'ae', 'œ': 'oe', 'ß': 'ss', 'ð': 'd', 'þ': 'th'})
POS = {'GK': 'GKP', 'GKP': 'GKP', 'DEF': 'DEF', 'MID': 'MID', 'FWD': 'FWD', 'AM': 'MID'}


def norm(s):
    s = str(s).translate(CHARS)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z ]', ' ', s.lower()).split()


def load_history(label):
    """One FPL season ('2023-24') as tidy rows: season (start year), gw,
    element, name, pos, club, opp, home, date, kickoff, minutes, points and
    components, xP. Opponents are recovered from the fixture id."""
    d = pd.read_csv(f'{HIST}/merged_gw_{label}.csv', encoding='utf-8', low_memory=False)
    d['club'] = d.team.map(lambda t: FPL2US.get(t, t))
    teams = d.groupby('fixture').club.agg(lambda s: sorted(set(s)))
    d['opp'] = [next((c for c in teams[f] if c != me), None) for f, me in zip(d.fixture, d.club)]
    d['kickoff'] = pd.to_datetime(d.kickoff_time, utc=True)
    out = pd.DataFrame(dict(
        season=int(label[:4]), gw=d.GW, element=d.element, name=d.name, pos=d.position.map(POS),
        club=d.club, opp=d.opp, home=d.was_home.astype(bool), kickoff=d.kickoff,
        date=d.kickoff.dt.tz_convert(None).dt.normalize(), minutes=d.minutes,
        points=d.total_points, goals=d.goals_scored, assists=d.assists, cs=d.clean_sheets,
        gc=d.goals_conceded, saves=d.saves, bonus=d.bonus, yellow=d.yellow_cards,
        pen_saved=d.penalties_saved, pen_missed=d.penalties_missed, own_goals=d.own_goals,
        red=d.red_cards, xP=d.get('xP')))
    return out.sort_values(['kickoff', 'element']).reset_index(drop=True)


def web_names(label):
    """FPL's short display names for a season ('Rodri', 'Raphinha')."""
    import os
    f = f'{HIST}/players_raw_{label}.csv'
    if not os.path.exists(f):
        return {}
    p = pd.read_csv(f, encoding='utf-8', low_memory=False)
    return dict(zip(p.id, p.web_name))


def match_players(fpl, rosters, web=None):
    """FPL element -> Understat player_id for one season, by name within the
    same club. FPL uses full legal names ('David Raya Martin', 'Rodrigo
    Hernandez'), Understat football names ('David Raya', 'Rodri'), so rules in
    order, each only if it picks out exactly one player at the club:
      exact name; Understat name contained in the FPL name; FPL's short web
      name equals the Understat name or its surname; same surname; any shared
      name part.
    A player who moved mid-season matches at either club. {element: id}."""
    season = int(fpl.season.iloc[0])
    web = web or {}
    ro = rosters[rosters.season == season].copy()
    ro['club'] = np.where(ro.h_a == 'h', ro.home, ro.away)
    us = ro.groupby(['player_id', 'club']).player.last().reset_index()
    us['toks'] = us.player.map(norm)
    by_club = {c: g for c, g in us.groupby('club')}
    out = {}
    for el, g in fpl.groupby('element'):
        toks = norm(g.name.iloc[0])
        wtok = norm(web.get(el, ''))
        rules = [
            lambda t: t == toks,
            lambda t: bool(t) and set(t) <= set(toks),
            lambda t: bool(wtok) and bool(t) and (t == wtok or t[-1:] == wtok[-1:] and len(wtok) == 1),
            lambda t: bool(t) and bool(toks) and t[-1] == toks[-1],
            lambda t: len(set(t) & set(toks)) > 0,
        ]
        hit = None
        for club in g.club.unique():
            cand = by_club.get(club)
            if cand is None:
                continue
            for rule in rules:
                x = cand[cand.toks.map(rule)]
                if len(x) == 1:
                    hit = x.player_id.iloc[0]
                    break
            if hit is not None:
                break
        if hit is not None:
            out[el] = hit
    return out
