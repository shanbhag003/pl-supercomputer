"""Pull per-match minutes for every Premier League player, 2014/15-2025/26.

Resumable and threaded. Each record is one player-match: who played, for whom,
for how long, plus their attacking contribution in that match.
"""
import json, os, sys, time, threading
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

import os as _o
STORE = _o.environ.get('STORE', '/home/claude/pl/data/processed/player_matches')
os.makedirs(STORE, exist_ok=True)
IDS = '/home/claude/pl/data/processed/player_ids.json'
HDR = {'User-Agent': 'Mozilla/5.0', 'X-Requested-With': 'XMLHttpRequest'}
MIN_MINUTES = 270           # below this a player is folded into a residual bucket
KEEP = ['date', 'season', 'time', 'position', 'h_team', 'a_team', 'id',
        'npxG', 'xA', 'xGChain', 'xGBuildup', 'key_passes', 'shots', 'goals']

lock = threading.Lock()
done_count = [0]


def shard(pid):
    return f'{STORE}/{int(pid) % 40:02d}.jsonl'


def already_done():
    have = set()
    for f in os.listdir(STORE):
        for line in open(f'{STORE}/{f}'):
            try:
                have.add(json.loads(line)['pid'])
            except Exception:
                pass
    return have


def fetch(pid, session):
    try:
        r = session.get(f'https://understat.com/getPlayerData/{pid}',
                        headers=HDR, timeout=30)
        if r.status_code != 200:
            return pid, None
        d = r.json()
        rows = [{k: m.get(k) for k in KEEP} for m in d.get('matches', [])]
        return pid, rows
    except Exception:
        return pid, None


def main(budget=210, workers=8):
    import os as _os
    if _os.environ.get('EU'):
        todo = json.load(open('/home/claude/pl/data/processed/eu_todo.json'))
        ids = {p: {'name': ''} for p in todo}
    else:
        ids = json.load(open(IDS))
        todo = [p for p, v in ids.items() if v['mins'] > MIN_MINUTES]
    have = already_done()
    todo = [p for p in todo if p not in have]
    print(f'{len(have)} already stored, {len(todo)} to go', flush=True)
    if not todo:
        return

    t0 = time.time()
    s = requests.Session()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        it = iter(todo)
        for _ in range(min(workers * 3, len(todo))):
            p = next(it, None)
            if p:
                futs[ex.submit(fetch, p, s)] = p
        while futs:
            for f in as_completed(list(futs), timeout=120):
                pid = futs.pop(f)
                try:
                    pid, rows = f.result()
                except Exception:
                    rows = None
                if rows is not None:
                    with lock:
                        with open(shard(pid), 'a') as fh:
                            fh.write(json.dumps(
                                {'pid': pid, 'name': ids[pid]['name'],
                                 'rows': rows}) + '\n')
                        done_count[0] += 1
                if time.time() - t0 < budget:
                    p = next(it, None)
                    if p:
                        futs[ex.submit(fetch, p, s)] = p
                break
            if time.time() - t0 > budget:
                for f in futs:
                    f.cancel()
                break
    el = time.time() - t0
    print(f'pulled {done_count[0]} players in {el:.0f}s '
          f'({done_count[0] / max(el, 1):.1f}/s)', flush=True)
    print(f'total stored now: {len(already_done())}', flush=True)


if __name__ == '__main__':
    main(budget=int(sys.argv[1]) if len(sys.argv) > 1 else 210,
         workers=int(sys.argv[2]) if len(sys.argv) > 2 else 8)
