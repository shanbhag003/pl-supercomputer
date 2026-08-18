"""Build a single clean match-level dataset: goals, xG, npxG, odds."""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, glob, os
import pandas as pd
import numpy as np

RAW = f'{ROOT}/data/raw'
US = f'{ROOT}/data/understat'
OUT = f'{ROOT}/data/processed'
os.makedirs(OUT, exist_ok=True)

# ---- Understat: match-level xG ----
rows = []
for f in sorted(glob.glob(f'{US}/EPL_*.json')):
    season = int(f.split('_')[-1].split('.')[0])
    j = json.load(open(f))
    # team history keyed by team -> list of per-game dicts (has npxG)
    hist = {}
    for tid, t in j['teams'].items():
        for g in t['history']:
            hist[(t['title'], g['date'][:10], g['h_a'])] = g
    for m in j['dates']:
        if not m['isResult']:
            continue
        d = m['datetime'][:10]
        gh = hist.get((m['h']['title'], d, 'h'))
        ga = hist.get((m['a']['title'], d, 'a'))
        rows.append(dict(
            season=season, date=m['datetime'][:10],
            home=m['h']['title'], away=m['a']['title'],
            hg=int(m['goals']['h']), ag=int(m['goals']['a']),
            hxg=float(m['xG']['h']), axg=float(m['xG']['a']),
            hnpxg=float(gh['npxG']) if gh else np.nan,
            anpxg=float(ga['npxG']) if ga else np.nan,
        ))
df = pd.DataFrame(rows)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# fall back to xG where npxG missing
df['hnpxg'] = df['hnpxg'].fillna(df['hxg'])
df['anpxg'] = df['anpxg'].fillna(df['axg'])

# ---- football-data.co.uk: closing odds (benchmark) ----
NAME_FD2US = {
    'Man City': 'Manchester City', 'Man United': 'Manchester United',
    'Tottenham': 'Tottenham', 'Newcastle': 'Newcastle United',
    'Wolves': 'Wolverhampton Wanderers', 'Nott\'m Forest': 'Nottingham Forest',
    'Sheffield United': 'Sheffield United', 'West Brom': 'West Bromwich Albion',
    'Leicester': 'Leicester', 'Brighton': 'Brighton', 'Leeds': 'Leeds',
    'West Ham': 'West Ham', 'Crystal Palace': 'Crystal Palace',
    'Bournemouth': 'Bournemouth', 'Aston Villa': 'Aston Villa',
    'Norwich': 'Norwich', 'Huddersfield': 'Huddersfield', 'Cardiff': 'Cardiff',
    'Swansea': 'Swansea', 'Stoke': 'Stoke', 'Middlesbrough': 'Middlesbrough',
    'Hull': 'Hull', 'Sunderland': 'Sunderland', 'Watford': 'Watford',
    'Burnley': 'Burnley', 'Everton': 'Everton', 'Chelsea': 'Chelsea',
    'Arsenal': 'Arsenal', 'Liverpool': 'Liverpool', 'Southampton': 'Southampton',
    'Fulham': 'Fulham', 'Brentford': 'Brentford', 'Luton': 'Luton',
    'Ipswich': 'Ipswich', 'QPR': 'QPR', 'Coventry': 'Coventry',
}
odds = []
for f in sorted(glob.glob(f'{RAW}/E0_*.csv')):
    o = pd.read_csv(f, encoding='latin-1')
    cols = [c for c in ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
                        'AvgH', 'AvgD', 'AvgA', 'B365H', 'B365D', 'B365A'] if c in o.columns]
    o = o[cols].dropna(subset=['HomeTeam'])
    o['date'] = pd.to_datetime(o['Date'], dayfirst=True, format='mixed')
    o['home'] = o['HomeTeam'].map(lambda x: NAME_FD2US.get(x, x))
    o['away'] = o['AwayTeam'].map(lambda x: NAME_FD2US.get(x, x))
    for c in ['AvgH', 'AvgD', 'AvgA']:
        if c not in o.columns:
            o[c] = o.get('B365' + c[-1], np.nan)
    odds.append(o[['date', 'home', 'away', 'AvgH', 'AvgD', 'AvgA']])
odds = pd.concat(odds, ignore_index=True)

df = df.merge(odds, on=['date', 'home', 'away'], how='left')
matched = df['AvgH'].notna().mean()

df.to_parquet(f'{OUT}/matches.parquet')
print(f'matches: {len(df)}  seasons: {df.season.min()}-{df.season.max()}')
print(f'odds matched: {matched:.1%}')
print(f'xG missing: {df.hxg.isna().sum()}  npxG missing: {df.hnpxg.isna().sum()}')
print(df.tail(3)[['date', 'home', 'away', 'hg', 'ag', 'hxg', 'axg', 'AvgH']].to_string(index=False))
