"""Build data/processed/rapm_live.pkl, the player values the squad layer uses.

Informed-prior plus-minus (players.py): each player's attack and defence value,
shrunk toward what his own per-90 numbers suggest rather than toward league
average. Fitted on every complete Premier League season in the roster data,
damp 0.5 - the setting backtest_players.py chose. See VALIDATION.md section 16.

Output keeps the format update.py and squad_live.py read:
  att    {understat player id: value}   adds to his side's xG
  dfn    {understat player id: value}   higher = better (he lowers the
                                        opponent's xG; players.py's sign flipped)
  known  [ids with a fitted value]

Rebuilt every June by .github/workflows/player-values.yml, which tops up the
lineups and shots first. By hand:

    python scripts/pull_understat_rosters.py 2014 2026
    python scripts/pull_understat_shots.py 2014 2026
    python src/build_player_values.py

Before overwriting, the new values are checked against the current file (see
check()); if they look broken the file is left alone and the run exits with an
error. --force skips the check.
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, pickle
import datetime as dt
sys.path.insert(0, f'{ROOT}/src')
import players as PL

DAMP = 0.5
OUT = f'{ROOT}/data/processed/rapm_live.pkl'


def build(last_season=None):
    rows, lu = PL.team_matches(PL.load_rosters(), PL.training_targets())
    complete = rows.groupby('season').size()
    last = last_season or int(complete[complete >= 760].index.max())   # 380 x 2 sides
    tr = rows[rows.season <= last]
    m = PL.fit(tr, lu, DAMP, informed=True)
    att = {str(k): float(v) for k, v in m['att'].items()}
    dfn = {str(k): float(-v) for k, v in m['dfn'].items()}
    return dict(att=att, dfn=dfn, known=sorted(att),
                meta=dict(method='informed-prior RAPM', damp=DAMP, seasons_through=last,
                          built=dt.date.today().isoformat(), n_players=len(att)))


def check(new, old, min_corr=0.9, min_share=0.9):
    """Reasons the new values look broken against the current file; empty if
    they look fine. A real new season moves values a little; a data fault -
    missing seasons, a bad download - moves them a lot or drops players.
    Calibrated: the 2024 -> 2025 update correlates 0.975; values built with
    the last nine seasons missing correlate 0.746."""
    import numpy as np
    problems = []
    if len(new['att']) < min_share * len(old['att']):
        problems.append(f"only {len(new['att'])} players, against {len(old['att'])} now")
    om = old.get('meta', {}).get('seasons_through')
    if om is not None and new['meta']['seasons_through'] < om:
        problems.append(f"fitted through {new['meta']['seasons_through']}, "
                        f"older than the current file's {om}")
    both = sorted(set(new['att']) & set(old['att']))
    if len(both) < 200:
        problems.append(f'only {len(both)} players in common with the current file')
    else:
        a = np.array([old['att'][p] + old['dfn'][p] for p in both])
        b = np.array([new['att'][p] + new['dfn'][p] for p in both])
        r = float(np.corrcoef(a, b)[0, 1])
        print(f'net value correlation with current file: {r:.3f} over {len(both)} players')
        if r < min_corr:
            problems.append(f'values correlate only {r:.2f} with the current file')
    return problems


if __name__ == '__main__':
    import os
    d = build()
    print(f"built: {d['meta']}")
    if os.path.exists(OUT) and '--force' not in sys.argv:
        cur = pickle.load(open(OUT, 'rb'))
        # Within 1e-6 counts as unchanged: the same data fitted on another
        # machine's library versions differs by ~3e-8, which is not news.
        same = (set(cur.get('att', {})) == set(d['att']) and
                max((max(abs(cur['att'][p] - d['att'][p]), abs(cur['dfn'][p] - d['dfn'][p]))
                     for p in d['att']), default=0.0) < 1e-6)
        if same:
            print('values unchanged - file left as it is')
            sys.exit(0)
        problems = check(d, cur)
        if problems:
            print('NOT WRITTEN - the new values look wrong:')
            for p in problems:
                print(f'  - {p}')
            print('Current file left in place. Investigate, or rerun with --force.')
            sys.exit(1)
    with open(OUT, 'wb') as fh:
        pickle.dump(d, fh)
    print(f'wrote {OUT}')
