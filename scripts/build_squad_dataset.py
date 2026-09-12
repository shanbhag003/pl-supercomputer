"""Squad, age and contract dataset for the top four leagues.

Premier League, LaLiga, Serie A and Bundesliga, last five seasons, built from
the open Transfermarkt extract at github.com/dcaribou/transfermarkt-datasets.

    python scripts/build_squad_dataset.py
    python scripts/build_squad_dataset.py --seasons 2021 2022 2023 2024 2025
    python scripts/build_squad_dataset.py --local-zip ~/Downloads/tm.zip
    python scripts/build_squad_dataset.py --no-fill-dob

What you get, and what you do not
---------------------------------
Contract expiry on Transfermarkt is a SNAPSHOT of the current deal, not a time
series. Nobody publishes what a player's contract was in 2021. So contract
columns are meaningful for players still on the books and empty or stale for
anyone who has since left the covered leagues. Age is exact for every season,
because date of birth does not move.

The upstream dataset paused updates on 6 July 2026 and does not cover 2026/27
squads. Seasons through 2025/26 are complete.

Schema safety
-------------
The upstream column names could not be verified before writing this, so every
column is resolved by name with fallbacks, and the first thing the script prints
is the actual schema of each file it loaded. If something is missing it says
which and carries on rather than dying at the last step.
"""
import argparse
import io
import json
import os
import sys
import time
import zipfile
from datetime import date, datetime, timezone

import pandas as pd
import requests

DATA_URL = ('https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/'
            'transfermarkt-datasets.zip')

# Transfermarkt competition codes. Ligue 1 is selectable but not on by
# default, because the brief was the top four.
LEAGUES = {'GB1': 'Premier League', 'ES1': 'LaLiga',
           'IT1': 'Serie A', 'L1': 'Bundesliga', 'FR1': 'Ligue 1'}
DEFAULT_LEAGUES = ['GB1', 'ES1', 'IT1', 'L1']

HDR = {'User-Agent': 'Mozilla/5.0 (squad-dataset-builder)'}


# ── loading ──────────────────────────────────────────────────────────────────

# The archive is not guaranteed to be CSV - the upstream pipeline is dbt, which
# emits parquet - so try each format rather than assuming one.
READERS = {
    '.csv':        lambda f: pd.read_csv(f, low_memory=False),
    '.csv.gz':     lambda f: pd.read_csv(f, low_memory=False, compression='gzip'),
    '.parquet':    lambda f: pd.read_parquet(io.BytesIO(f.read())),
    '.pq':         lambda f: pd.read_parquet(io.BytesIO(f.read())),
    '.json':       lambda f: pd.read_json(f),
    '.jsonl':      lambda f: pd.read_json(f, lines=True),
}


def list_archive(names, limit=60):
    """Print what is actually inside, so a miss is diagnosable in one run."""
    real = [n for n in names if not n.endswith('/') and '__MACOSX' not in n]
    print(f'\n  archive holds {len(real)} files. Data-looking entries:')
    data = [n for n in real
            if any(n.lower().endswith(e) for e in READERS)]
    shown = data or real
    for n in sorted(shown)[:limit]:
        print(f'      {n}')
    if len(shown) > limit:
        print(f'      ... and {len(shown) - limit} more')
    if not data:
        print('      (nothing with a recognised data extension)')


def load_tables(zip_bytes, want):
    """Pull the named tables out of the archive, whatever format they are in."""
    out = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = [n for n in z.namelist()
                 if not n.endswith('/') and '__MACOSX' not in n]
        missing = []
        for key in want:
            # Exact stem first, then near-misses like tm_players_v2.parquet.
            # Scored rather than first-match, because a loose "contains" would
            # let `games` match `club_games` and silently load the wrong table.
            scored = []
            for n in names:
                low = n.lower()
                ext = next((e for e in sorted(READERS, key=len, reverse=True)
                            if low.endswith(e)), None)
                if not ext:
                    continue
                stem = low[:-len(ext)].split('/')[-1]
                if stem == key:
                    rank = 0
                elif stem.endswith(f'_{key}') or stem.startswith(f'{key}_'):
                    rank = 1
                elif f'_{key}_' in stem or stem.endswith(key):
                    rank = 2
                else:
                    continue
                scored.append((rank, len(n), n, ext))
            if not scored:
                missing.append(key)
                continue
            scored.sort()
            rank, _, path, ext = scored[0]
            if rank:
                print(f'  ~ {key}: no exact match, using {path}')
            try:
                with z.open(path) as f:
                    out[key] = READERS[ext](f)
                print(f'  {key:14} {len(out[key]):>9,} rows  from {path}')
            except Exception as e:                    # noqa: BLE001
                print(f'  ! {key}: found {path} but could not read it ({e})')
        if missing:
            print(f'\n  ! not found: {", ".join(missing)}')
            list_archive(names)
    return out


def col(df, *candidates, required=False, label=''):
    """First matching column name, case-insensitively. None when absent."""
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f'{label or candidates[0]}: none of {candidates} found. '
                       f'Available: {list(df.columns)}')
    return None


def report_schema(tables):
    print('\n' + '=' * 72)
    print('SCHEMA AS LOADED  (check this if anything downstream looks wrong)')
    print('=' * 72)
    for name, df in tables.items():
        print(f'\n{name}.csv  -  {len(df):,} rows, {len(df.columns)} columns')
        for c in df.columns:
            print(f'    {c}')
    print('=' * 72 + '\n')


# ── date of birth backfill ───────────────────────────────────────────────────

def wikidata_dob(name, club_hint=None, session=None, timeout=20):
    """Date of birth for one footballer, or None.

    Search Wikidata for the name, then take the first result that is a human
    (P31=Q5) with a date of birth. When a club is known and one candidate lists
    it under P54 (member of sports team), prefer that one - the name alone is
    ambiguous often enough to matter.
    """
    s = session or requests.Session()
    try:
        r = s.get('https://www.wikidata.org/w/api.php', headers=HDR, timeout=timeout,
                  params={'action': 'wbsearchentities', 'search': name,
                          'language': 'en', 'format': 'json', 'limit': 5,
                          'type': 'item'})
        hits = r.json().get('search', [])
        if not hits:
            return None
        ids = [h['id'] for h in hits]
        r2 = s.get('https://www.wikidata.org/w/api.php', headers=HDR, timeout=timeout,
                   params={'action': 'wbgetentities', 'ids': '|'.join(ids),
                           'props': 'claims', 'format': 'json'})
        ents = r2.json().get('entities', {})

        best = None
        for qid in ids:
            claims = (ents.get(qid) or {}).get('claims', {})
            if not any(c['mainsnak'].get('datavalue', {}).get('value', {}).get('id') == 'Q5'
                       for c in claims.get('P31', [])):
                continue                                  # not a human
            dob = None
            for c in claims.get('P569', []):
                t = c['mainsnak'].get('datavalue', {}).get('value', {}).get('time')
                if t:
                    dob = t.lstrip('+')[:10]
                    break
            if not dob:
                continue
            if best is None:
                best = dob
            if club_hint:
                labels = json.dumps(claims.get('P54', []))
                if club_hint.split()[0].lower() in labels.lower():
                    return dob
        return best
    except Exception:                                     # noqa: BLE001
        return None


def fill_missing_dob(players, name_c, dob_c, club_c, limit, pause=0.35):
    miss = players[players[dob_c].isna()]
    if miss.empty:
        print('  no missing dates of birth - nothing to fill')
        return players, 0
    print(f'  {len(miss)} players missing a date of birth')
    if limit and len(miss) > limit:
        miss = miss.head(limit)
        print(f'  filling the first {limit} (raise --dob-limit for more)')

    s, filled = requests.Session(), 0
    for i, (idx, row) in enumerate(miss.iterrows(), 1):
        dob = wikidata_dob(str(row[name_c]),
                           str(row[club_c]) if club_c and pd.notna(row.get(club_c)) else None,
                           session=s)
        if dob:
            players.loc[idx, dob_c] = dob
            filled += 1
        if i % 25 == 0:
            print(f'    {i}/{len(miss)} checked, {filled} filled', flush=True)
        time.sleep(pause)
    print(f'  filled {filled} of {len(miss)} from Wikidata')
    return players, filled


# ── build ────────────────────────────────────────────────────────────────────

def season_start(y):
    return date(y, 8, 1)


def age_at(dob, on):
    if pd.isna(dob):
        return None
    d = pd.to_datetime(dob, errors='coerce')
    if pd.isna(d):
        return None
    d = d.date()
    return round((on - d).days / 365.25, 1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seasons', type=int, nargs='+',
                    default=[2021, 2022, 2023, 2024, 2025],
                    help='starting years; 2025 means 2025/26')
    ap.add_argument('--leagues', nargs='+', default=DEFAULT_LEAGUES,
                    choices=list(LEAGUES),
                    help='default is the top four; add FR1 for Ligue 1')
    ap.add_argument('--local-zip', help='use a downloaded archive instead of fetching')
    ap.add_argument('--data-url', default=DATA_URL)
    ap.add_argument('--out', default='squads_contracts.xlsx')
    ap.add_argument('--no-fill-dob', action='store_true',
                    help='skip the Wikidata backfill')
    ap.add_argument('--with-derived', action='store_true',
                    help='also emit age and contract-countdown columns. Off by '
                         'default: anything measured against today is stale the '
                         'moment the file is saved, so the workbook ships raw '
                         'dates and you compute from them at read time')
    ap.add_argument('--dob-limit', type=int, default=400,
                    help='cap on Wikidata lookups')
    a = ap.parse_args()

    # ---- fetch
    if a.local_zip:
        print(f'reading {a.local_zip}')
        blob = open(a.local_zip, 'rb').read()
    else:
        print(f'downloading {a.data_url}')
        r = requests.get(a.data_url, headers=HDR, timeout=600, stream=True)
        r.raise_for_status()
        blob = r.content
    print(f'  {len(blob)/1e6:.0f} MB\n')

    tables = load_tables(blob, ['players', 'appearances', 'games', 'clubs'])
    report_schema(tables)
    for need in ('players', 'appearances', 'games'):
        if need not in tables:
            print(f'cannot continue without a "{need}" table. See the archive '
                  f'listing above and pass the right names, or open an issue.',
                  file=sys.stderr)
            return 1

    P, AP, G = tables['players'], tables['appearances'], tables['games']
    CL = tables.get('clubs')

    # ---- resolve columns
    p_id = col(P, 'player_id', required=True, label='players.player_id')
    p_nm = col(P, 'name', 'player_name', 'pretty_name', required=True)
    p_dob = col(P, 'date_of_birth', 'dateOfBirth', 'birth_date')
    p_con = col(P, 'contract_expiration_date', 'contract_expires', 'contract_until')
    p_pos = col(P, 'position')
    p_sub = col(P, 'sub_position')
    p_ft = col(P, 'foot')
    p_ht = col(P, 'height_in_cm', 'height')
    p_cit = col(P, 'country_of_citizenship', 'citizenship')
    p_mv = col(P, 'market_value_in_eur', 'market_value')
    p_hmv = col(P, 'highest_market_value_in_eur')
    p_cc = col(P, 'current_club_name')
    p_url = col(P, 'url')

    g_id = col(G, 'game_id', required=True)
    g_se = col(G, 'season', required=True)
    g_cp = col(G, 'competition_id', required=True)

    a_gid = col(AP, 'game_id', required=True)
    a_pid = col(AP, 'player_id', required=True)
    a_clb = col(AP, 'player_club_id', 'club_id')
    a_min = col(AP, 'minutes_played', 'minutes')
    a_gl = col(AP, 'goals')
    a_as = col(AP, 'assists')

    for label, c in (('date_of_birth', p_dob), ('contract_expiration_date', p_con)):
        print(f'  {label:28} {"FOUND: " + c if c else "NOT FOUND - column will be blank"}')
    print()

    # ---- which players appeared in which league-season
    g = G[[g_id, g_se, g_cp]].rename(
        columns={g_id: 'game_id', g_se: 'season', g_cp: 'competition_id'})
    g = g[g.competition_id.isin(a.leagues) & g.season.isin(a.seasons)]
    print(f'games in scope: {len(g):,}')

    keep = [c for c in (a_gid, a_pid, a_clb, a_min, a_gl, a_as) if c]
    ap_ = AP[keep].rename(columns={a_gid: 'game_id', a_pid: 'player_id'})
    ap_ = ap_.merge(g, on='game_id', how='inner')
    print(f'appearances in scope: {len(ap_):,}')
    if ap_.empty:
        print('\nNothing matched. Check --seasons against the season column, and '
              'that league codes match competition_id in games.csv.', file=sys.stderr)
        print('season values present:', sorted(G[g_se].dropna().unique())[:12],
              file=sys.stderr)
        print('competition_id values:', sorted(G[g_cp].dropna().unique())[:12],
              file=sys.stderr)
        return 1

    agg = {'apps': ('game_id', 'count')}
    if a_min:
        agg['minutes'] = (a_min, 'sum')
    if a_gl:
        agg['goals'] = (a_gl, 'sum')
    if a_as:
        agg['assists'] = (a_as, 'sum')
    if a_clb:
        agg['club_id'] = (a_clb, lambda s: s.mode().iat[0] if not s.mode().empty else None)
    squads = (ap_.groupby(['competition_id', 'season', 'player_id'], as_index=False)
                 .agg(**agg))

    # ---- bio, contract, DOB backfill
    pcols = [c for c in (p_id, p_nm, p_dob, p_con, p_pos, p_sub, p_ft, p_ht,
                         p_cit, p_mv, p_hmv, p_cc, p_url) if c]
    bio = P[pcols].copy()
    if p_dob and not a.no_fill_dob:
        print('\nbackfilling missing dates of birth from Wikidata')
        bio, _ = fill_missing_dob(bio, p_nm, p_dob, p_cc, a.dob_limit)

    df = squads.merge(bio, left_on='player_id', right_on=p_id, how='left')
    if CL is not None:
        c_id, c_nm = col(CL, 'club_id'), col(CL, 'name', 'club_name', 'pretty_name')
        if c_id and c_nm and 'club_id' in df.columns:
            df = df.merge(CL[[c_id, c_nm]].rename(columns={c_id: 'club_id',
                                                           c_nm: 'club'}),
                          on='club_id', how='left')

    df['league'] = df['competition_id'].map(LEAGUES)
    df['season_label'] = df['season'].astype(str) + '/' + \
        (df['season'] + 1).astype(str).str[-2:]

    today = datetime.now(timezone.utc).date()
    if p_dob:
        df['date_of_birth'] = pd.to_datetime(df[p_dob], errors='coerce')
        if a.with_derived:
            df['age_at_season_start'] = [
                age_at(d, season_start(int(s)))
                for d, s in zip(df['date_of_birth'], df['season'])]
            df['age_today'] = [age_at(d, today) for d in df['date_of_birth']]
    if p_con:
        df['contract_expires'] = pd.to_datetime(df[p_con], errors='coerce')
        df['contract_months_left'] = (
            (df['contract_expires'] - pd.Timestamp(today)).dt.days / 30.44).round(1)
        # The bands people actually act on. Under six months a player can talk
        # to foreign clubs; under twelve is the last window to sell for a fee.
        def band(m):
            if pd.isna(m):
                return 'unknown'
            if m < 0:
                return 'expired / stale'
            if m < 6:
                return '0-6 months'
            if m < 12:
                return '6-12 months'
            if m < 24:
                return '1-2 years'
            return '2+ years'
        df['contract_status'] = df['contract_months_left'].apply(band)
        if not a.with_derived:
            # Keep the raw expiry date; drop the countdown, which is only true
            # on the day the file was built.
            df = df.drop(columns=['contract_months_left', 'contract_status'])

    ren = {p_nm: 'player', p_pos: 'position', p_sub: 'sub_position', p_ft: 'foot',
           p_ht: 'height_cm', p_cit: 'citizenship', p_mv: 'market_value_eur',
           p_hmv: 'highest_market_value_eur', p_cc: 'current_club',
           p_url: 'transfermarkt_url'}
    df = df.rename(columns={k: v for k, v in ren.items() if k})

    order = ['league', 'season_label', 'club', 'player', 'position', 'sub_position',
             'date_of_birth', 'age_at_season_start', 'age_today', 'citizenship',
             'foot', 'height_cm', 'apps', 'minutes', 'goals', 'assists',
             'contract_expires', 'contract_months_left', 'contract_status',
             'market_value_eur', 'highest_market_value_eur', 'current_club',
             'player_id', 'transfermarkt_url']
    squad_sheet = df[[c for c in order if c in df.columns]].sort_values(
        ['league', 'season_label', 'club', 'minutes'],
        ascending=[True, False, True, False])

    # ---- derived views
    latest = max(a.seasons)
    current = squad_sheet[squad_sheet['season_label'].str.startswith(str(latest))]
    sellable = pd.DataFrame()
    if 'contract_expires' in current.columns:
        w = current.copy()
        # This one sheet is explicitly a snapshot view, so it computes the
        # countdown here rather than relying on a stored column.
        w['contract_months_left'] = (
            (w['contract_expires'] - pd.Timestamp(today)).dt.days / 30.44).round(1)
        w = w[w['contract_months_left'].notna() & (w['contract_months_left'] < 18)]
        # Imminent expiries first, already-expired last. A negative figure means
        # the deal has run out or Transfermarkt has not refreshed it, so it is
        # not actionable in the way "four months left" is - sorting it to the
        # top would bury the cases that matter.
        w['_order'] = w['contract_months_left'].where(
            w['contract_months_left'] >= 0, 9999 + w['contract_months_left'])
        sellable = w.sort_values('_order').drop(columns='_order')

    age_profile = pd.DataFrame()
    if a.with_derived and 'age_at_season_start' in squad_sheet.columns:
        wm = lambda g_: ((g_['age_at_season_start'] * g_['minutes']).sum()
                         / g_['minutes'].sum()) if g_['minutes'].sum() else None
        rows = []
        for (lg, se, cb), grp in squad_sheet.groupby(['league', 'season_label', 'club']):
            rows.append(dict(league=lg, season=se, club=cb, players=len(grp),
                             mean_age=round(grp['age_at_season_start'].mean(), 2)
                             if grp['age_at_season_start'].notna().any() else None,
                             median_age=round(grp['age_at_season_start'].median(), 2)
                             if grp['age_at_season_start'].notna().any() else None,
                             minutes_weighted_age=round(wm(grp), 2)
                             if 'minutes' in grp else None,
                             total_minutes=int(grp['minutes'].sum())
                             if 'minutes' in grp else None))
        age_profile = pd.DataFrame(rows).sort_values(['league', 'season', 'club'])

    # ---- write
    def sheet(w, name, d):
        if d is None or d.empty:
            return
        n = name[:31]
        d.to_excel(w, sheet_name=n, index=False)
        ws = w.sheets[n]
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(len(d), 1), max(len(d.columns) - 1, 0))
        for i, c in enumerate(d.columns):
            longest = d[c].astype(str).str.len().max() if len(d) else 0
            ws.set_column(i, i, min(max(len(str(c)) + 2, int(longest or 0) + 2), 34))

    with pd.ExcelWriter(a.out, engine='xlsxwriter', datetime_format='yyyy-mm-dd',
                        date_format='yyyy-mm-dd') as w:
        sheet(w, 'Squads', squad_sheet)
        sheet(w, f'Current {latest}', current)
        sheet(w, 'Contract watch', sellable)
        sheet(w, 'Age profile', age_profile)
        notes = pd.DataFrame({'field': [
            'source', 'built (UTC)', 'seasons', 'leagues', 'player-seasons',
            'unique players', 'clubs', 'date of birth present',
            'contract date present', '', 'CONTRACTS', '', 'COVERAGE'],
            'value': [
            'github.com/dcaribou/transfermarkt-datasets (Transfermarkt)',
            datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M'),
            ', '.join(f'{s}/{str(s+1)[-2:]}' for s in a.seasons),
            ', '.join(LEAGUES[l] for l in a.leagues),
            len(squad_sheet), squad_sheet['player'].nunique(),
            squad_sheet['club'].nunique() if 'club' in squad_sheet else 'n/a',
            f"{squad_sheet['date_of_birth'].notna().sum()} of {len(squad_sheet)}"
            if 'date_of_birth' in squad_sheet else 'column not found',
            f"{squad_sheet['contract_expires'].notna().sum()} of {len(squad_sheet)}"
            if 'contract_expires' in squad_sheet else 'column not found', '',
            'Contract expiry is a SNAPSHOT of the current deal, not history. '
            'It is meaningful for players still on the books and stale or blank '
            'for anyone who has left the covered leagues. Age is exact for '
            'every season.', '',
            'Upstream paused updates on 2026-07-06. 2026/27 squads are not '
            'covered; seasons through 2025/26 are complete.']})
        sheet(w, 'About', notes)

    print(f'\nwrote {a.out}')
    print(f'  {len(squad_sheet):,} player-seasons | '
          f'{squad_sheet["player"].nunique():,} players')
    if 'date_of_birth' in squad_sheet:
        print(f'  date of birth: {squad_sheet["date_of_birth"].notna().sum():,} present')
    if 'contract_expires' in squad_sheet:
        print(f'  contract date: {squad_sheet["contract_expires"].notna().sum():,} present')
        print(f'  contract watch (<18 months): {len(sellable):,} players')
    return 0


if __name__ == '__main__':
    sys.exit(main())
