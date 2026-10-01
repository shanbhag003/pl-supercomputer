"""Player-match data for every Premier League match Understat has, 2014/15 on.

From the same per-match endpoint as the shots (getMatchData): every player who
appeared, his side, minutes, position and per-match xG, xA, xGChain, xGBuildup.
One request per match, cached per season in data/understat/rosters/, resumable.

Cleaner than the per-player route in pull_players.py: the side a player played
for is recorded per match, so his club does not have to be inferred.

    python scripts/pull_understat_rosters.py 2014 2026
"""
import os as _os
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json, time
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

HDR = {'User-Agent': 'Mozilla/5.0', 'X-Requested-With': 'XMLHttpRequest'}
OUT = f'{ROOT}/data/understat/rosters'
KEEP = ['player_id', 'player', 'h_a', 'time', 'position', 'positionOrder', 'goals',
        'shots', 'xG', 'xA', 'key_passes', 'assists', 'xGChain', 'xGBuildup',
        'red_card', 'roster_in', 'roster_out']
NUM = ['time', 'positionOrder', 'goals', 'shots', 'xG', 'xA', 'key_passes',
       'assists', 'xGChain', 'xGBuildup', 'red_card']


def fetch(m, session):
    for attempt in range(3):
        try:
            r = session.get(f'https://understat.com/getMatchData/{m["id"]}',
                            headers=HDR, timeout=30)
            if r.status_code == 200:
                ro = json.loads(r.text)['rosters']
                rows = []
                for side in 'ha':
                    for p in ro[side].values():
                        row = {k: p.get(k) for k in KEEP}
                        row.update(match_id=m['id'], date=m['datetime'][:10],
                                   home=m['h']['title'], away=m['a']['title'])
                        rows.append(row)
                return m['id'], rows
        except Exception:
            pass
        time.sleep(2 + 3 * attempt)
    return m['id'], None


def pull(season, session, workers=4):
    os.makedirs(OUT, exist_ok=True)
    f = f'{OUT}/EPL_{season}.parquet'
    have = pd.read_parquet(f) if os.path.exists(f) else pd.DataFrame()
    done = set(have.match_id.astype(str)) if len(have) else set()
    r = session.get(f'https://understat.com/getLeagueData/EPL/{season}',
                    headers=HDR, timeout=40)
    r.raise_for_status()
    todo = [m for m in r.json()['dates'] if m.get('isResult') and m['id'] not in done]
    if not todo:
        return 0, 0
    rows, failed = [], 0
    with ThreadPoolExecutor(workers) as ex:
        for mid, rr in ex.map(lambda m: fetch(m, session), todo):
            if rr is None:
                failed += 1
            else:
                rows += rr
    new = pd.DataFrame(rows)
    for c in NUM:
        new[c] = pd.to_numeric(new[c])
    new['date'] = pd.to_datetime(new['date'])
    out = pd.concat([have, new], ignore_index=True) if len(have) else new
    out.to_parquet(f, index=False)
    return len(todo) - failed, failed


if __name__ == '__main__':
    a, b = (int(x) for x in sys.argv[1:3])
    with requests.Session() as s:
        for season in range(a, b + 1):
            t0 = time.time()
            added, failed = pull(season, s)
            print(f'{season}: +{added} matches, {failed} failed, {time.time() - t0:.0f}s',
                  flush=True)
