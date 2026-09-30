"""Shot-level data for every Premier League match Understat has, 2014/15 on.

One request per match, cached per season in data/understat/shots/, resumable.
The live pipeline tops up the current season itself on every run; this is for
the history.

    python scripts/pull_understat_shots.py 2014 2025
"""
import os as _os
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, time
import requests
sys.path.insert(0, f'{ROOT}/src')
from gamestate import pull_season

if __name__ == '__main__':
    a, b = (int(x) for x in sys.argv[1:3])
    with requests.Session() as s:
        for season in range(a, b + 1):
            t0 = time.time()
            added, failed = pull_season(season, s)
            print(f'{season}: +{added} matches, {failed} failed, {time.time() - t0:.0f}s',
                  flush=True)
