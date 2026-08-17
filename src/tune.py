"""Resumable grid search. Saves after every combo so it can run in chunks."""
import sys, os, json, time, itertools, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, '/home/claude/pl/src')
from backtest import run_season
warnings.filterwarnings('ignore')

OUT = '/home/claude/pl/data/processed/tuning.csv'
TUNE = [2022, 2023, 2024]          # tuning seasons
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 200   # seconds

GRID = [dict(xi=xi, w_xg=w, ridge=rg)
        for xi, w, rg in itertools.product(
            [0.0015, 0.003, 0.0045, 0.007],
            [0.0, 0.4, 0.7, 1.0],
            [0.0, 2.0, 6.0, 15.0])]

done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
seen = set()
if len(done):
    seen = {(r.xi, r.w_xg, r.ridge) for r in done.itertuples()}

t0 = time.time()
rows = done.to_dict('records')
for g in GRID:
    if (g['xi'], g['w_xg'], g['ridge']) in seen:
        continue
    if time.time() - t0 > BUDGET:
        break
    r = pd.concat([run_season(s, g['xi'], g['w_xg'], g['ridge']) for s in TUNE])
    rows.append({**g, 'rps': r.rps.mean(), 'll': r.ll.mean(),
                 'b_rps': r.b_rps.mean(), 'n': len(r)})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"xi={g['xi']:<7} w_xg={g['w_xg']:<4} ridge={g['ridge']:<5} "
          f"RPS={r.rps.mean():.5f}  (bookie {r.b_rps.mean():.5f})", flush=True)

d = pd.DataFrame(rows)
print(f'\n{len(d)}/{len(GRID)} combos done')
if len(d):
    print(d.sort_values('rps').head(5).round(5).to_string(index=False))
