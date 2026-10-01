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

Run once a season, after the last round, with the rosters topped up:

    python scripts/pull_understat_rosters.py 2014 2026
    python src/build_player_values.py
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


if __name__ == '__main__':
    d = build()
    with open(OUT, 'wb') as fh:
        pickle.dump(d, fh)
    print(f"wrote {OUT}: {d['meta']}")
