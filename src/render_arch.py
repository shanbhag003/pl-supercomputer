"""Architecture graphic — vertical pipeline, same theme as the standings image."""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import backdrop, ANTON, BC_B, BC_M, PURPLE_D, PINK, CYAN, GREEN, WHITE

LAV = '#C4AECC'
W, H = 7.2, 9.0
L, R = .058, .942

STAGES = [
    ('01', 'RATING MODEL',   'Dixon-Coles  ·  70% non-penalty xG  ·  154-day half-life'),
    ('02', 'PROMOTED PRIOR', 'Championship points predict nothing  ·  r = +0.04'),
    ('03', 'DRIFT',          'Summer squad change  ·  0.16 SD, decays with matches'),
    ('04', 'BOOTSTRAP',      '80 refits  ·  what we do not know about the ratings'),
    ('05', 'MONTE CARLO',    '20,000 seasons  ·  real Premier League tiebreaks'),
    ('06', 'SELF-SCORING',   "Grades last week's calls against bookmaker odds"),
]

fig = plt.figure(figsize=(W, H), dpi=150)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
backdrop(ax, glow_x=.32, glow_y=.03)


def Y(p):
    return 1 - p / H


def panel(top, h, x0, x1, accent=None, fill=.075, lw=1.1, edge=None):
    ax.add_patch(FancyBboxPatch((x0, Y(top + h)), x1 - x0, h / H,
                                boxstyle='round,pad=0,rounding_size=.013',
                                fc=accent or WHITE, ec='none', alpha=fill, zorder=2))
    ax.add_patch(FancyBboxPatch((x0, Y(top + h)), x1 - x0, h / H,
                                boxstyle='round,pad=0,rounding_size=.013',
                                fc='none', ec=edge or accent or WHITE,
                                alpha=.5 if (accent or edge) else .16, lw=lw, zorder=3))


def arrow(top, col):
    ax.plot([.5, .5], [Y(top), Y(top + .17)], lw=1.6, color=col, alpha=.4, zorder=3)
    ax.add_patch(Polygon([[.5, Y(top + .28)], [.485, Y(top + .16)],
                          [.515, Y(top + .16)]], closed=True, fc=col, alpha=.55,
                         ec='none', zorder=3))


def kicker(top, text):
    ax.text(L, Y(top), text, fontproperties=BC_M, fontsize=10.5, color=LAV,
            alpha=.85, va='center_baseline', zorder=4)


# ---------------------------------------------------------------- header
ax.text(L, Y(.54), 'AUTONOMOUS  ·  SELF-VALIDATING  ·  FREE TO RUN',
        fontproperties=BC_M, fontsize=11, color=GREEN, va='center_baseline', zorder=4)
ax.text(L, Y(1.10), 'PL SUPERCOMPUTER', fontproperties=ANTON, fontsize=36,
        color=WHITE, va='center', zorder=4)
ax.text(L, Y(1.54), 'HOW IT WORKS', fontproperties=ANTON, fontsize=23,
        color=PINK, va='center', zorder=4)
ax.text(L, Y(1.94), 'Wakes every 2 hours. Publishes 12 hours after a gameweek',
        fontproperties=BC_M, fontsize=12.5, color=LAV, va='center_baseline', zorder=4)
ax.text(L, Y(2.14), 'settles — midweek rounds included.',
        fontproperties=BC_M, fontsize=12.5, color=LAV, va='center_baseline', zorder=4)

# ---------------------------------------------------------------- inputs
kicker(2.40, 'INPUTS')
for x0, x1, title, lines in [
        (L, .492, 'FROZEN HISTORY',
         ['4,560 matches  ·  12 seasons', 'xG and non-penalty xG']),
        (.508, R, 'PULLED EACH RUN',
         ['Understat  ·  football-data', 'Fixtures with kickoff times'])]:
    panel(2.52, .74, x0, x1)
    ax.text(x0 + .026, Y(2.75), title, fontproperties=BC_B, fontsize=13,
            color=WHITE, va='center_baseline', zorder=4)
    for i, ln in enumerate(lines):
        ax.text(x0 + .026, Y(2.96 + i * .17), ln, fontproperties=BC_M,
                fontsize=10.6, color=LAV, va='center_baseline', zorder=4)

panel(3.36, .42, L, R, accent=PINK, fill=.10)
ax.text(L + .026, Y(3.58), 'GUARD   the feed once served National League fixtures',
        fontproperties=BC_M, fontsize=11.2, color='#FFA9C8',
        va='center_baseline', zorder=4)

arrow(3.80, CYAN)

# ---------------------------------------------------------------- engine
ETOP, CH, PITCH = 4.10, .44, .50
EH = .56 + (len(STAGES) - 1) * PITCH + CH + .14
panel(ETOP, EH, L - .014, R + .014, accent=GREEN, fill=.05, lw=1.4)
ax.text(.5, Y(ETOP + .34), 'THE ENGINE', fontproperties=ANTON, fontsize=17,
        color=GREEN, ha='center', va='center', zorder=4)

for i, (n, title, desc) in enumerate(STAGES):
    t = ETOP + .56 + i * PITCH
    last = i == len(STAGES) - 1
    col = CYAN if last else GREEN
    panel(t, CH, L + .014, R - .014, accent=col if last else None,
          fill=.075 if last else .05, lw=1.0)
    ax.add_patch(Circle((L + .056, Y(t + CH / 2)), .0178, fc=PURPLE_D, ec=col,
                        lw=1.4, zorder=4))
    ax.text(L + .056, Y(t + CH / 2), n, fontproperties=BC_B, fontsize=8.4,
            color=col, ha='center', va='center', zorder=5)
    ax.text(L + .098, Y(t + .165), title, fontproperties=BC_B, fontsize=13.5,
            color=col if last else WHITE, va='center_baseline', zorder=4)
    ax.text(L + .098, Y(t + .335), desc, fontproperties=BC_M, fontsize=10.6,
            color=LAV, va='center_baseline', zorder=4)

arrow(ETOP + EH + .02, GREEN)

# ---------------------------------------------------------------- output
OTOP = ETOP + EH + .44
kicker(OTOP - .14, 'OUTPUT')
panel(OTOP, .60, L, R, accent=GREEN, fill=.09)
ax.text(L + .026, Y(OTOP + .22), 'EMAILED TO YOU, COMMITTED TO THE REPO',
        fontproperties=BC_B, fontsize=13.5, color=GREEN, va='center_baseline', zorder=4)
ax.text(L + .026, Y(OTOP + .44), 'Graphic  ·  full CSV  ·  what moved and why  ·  accuracy log',
        fontproperties=BC_M, fontsize=10.8, color=LAV, va='center_baseline', zorder=4)

ax.plot([L, R], [Y(8.70)] * 2, color=WHITE, alpha=.15, lw=1, zorder=3)
ax.text(.5, Y(8.86), 'PYTHON  ·  NUMPY  ·  SCIPY  ·  GITHUB ACTIONS  ·  ZERO COST',
        fontproperties=BC_M, fontsize=11, color=LAV, alpha=.75,
        ha='center', va='center', zorder=4)

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'outputs', 'architecture.png')
fig.savefig(out, facecolor=PURPLE_D)
print(f'{out}  (engine ends {ETOP + EH:.2f}, output ends {OTOP + .60:.2f} of {H})')
