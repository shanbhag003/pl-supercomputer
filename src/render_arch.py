"""Architecture graphic — plain English, vertical flow, same theme."""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import (backdrop, Centred, ANTON, BC_B, BC_M,
                    PURPLE_D, PINK, CYAN, GREEN, WHITE, LAV)

W, H = 7.2, 9.0
L, R = .058, .942

STEPS = [
    ('1', 'IT LEARNS FROM THE PAST',
     'Twelve seasons of results — and how good the chances were,',
     'not just who got lucky'),
    ('2', 'IT SCORES EVERY CLUB',
     'How well each one attacks, how well it defends,',
     'with recent form counting most'),
    ('3', 'IT ADMITS WHAT IT CANNOT KNOW',
     'New signings, new managers, a whole summer of change —',
     'so nothing is treated as certain'),
    ('4', 'IT PLAYS THE SEASON 20,000 TIMES',
     'Every remaining match, over and over,',
     'letting the luck fall differently each time'),
    ('5', 'IT COUNTS WHAT HAPPENED',
     'Won the league in 6,220 of them? That is a 31% title chance.',
     'Same for top four and relegation'),
    ('6', 'IT MARKS ITS OWN HOMEWORK',
     'Every week it checks last week\'s predictions against reality,',
     'and against the bookmakers'),
]

fig = plt.figure(figsize=(W, H), dpi=150)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
backdrop(ax, glow_x=.32, glow_y=.03)
C = Centred(fig, ax)


def Y(p):
    return 1 - p / H


def panel(top, h, x0, x1, accent=None, fill=.075, lw=1.1):
    ax.add_patch(FancyBboxPatch((x0, Y(top + h)), x1 - x0, h / H,
                                boxstyle='round,pad=0,rounding_size=.013',
                                fc=accent or WHITE, ec='none', alpha=fill, zorder=2))
    ax.add_patch(FancyBboxPatch((x0, Y(top + h)), x1 - x0, h / H,
                                boxstyle='round,pad=0,rounding_size=.013',
                                fc='none', ec=accent or WHITE,
                                alpha=.5 if accent else .16, lw=lw, zorder=3))


# ------------------------------------------------------------------ header
C.text(L, Y(.56), 'IT RUNS ITSELF  ·  IT CHECKS ITSELF  ·  IT COSTS NOTHING',
       fontproperties=BC_M, fontsize=11.5, color=GREEN, zorder=4)
ax.text(L, Y(1.14), 'PREDICTING THE', fontproperties=ANTON, fontsize=33,
        color=WHITE, va='center', zorder=4)
ax.text(L, Y(1.62), 'PREMIER LEAGUE', fontproperties=ANTON, fontsize=33,
        color=PINK, va='center', zorder=4)
C.text(L, Y(2.12), 'A computer plays out the whole season, thousands of times over,',
       fontproperties=BC_M, fontsize=13, color=LAV, zorder=4)
C.text(L, Y(2.34), 'then counts how often each club came first — and how often it went down.',
       fontproperties=BC_M, fontsize=13, color=LAV, zorder=4)

# ------------------------------------------------------------------- steps
TOP, CH, PITCH = 2.70, .78, .845
for i, (n, title, l1, l2) in enumerate(STEPS):
    t = TOP + i * PITCH
    last = i == len(STEPS) - 1
    col = CYAN if last else GREEN
    panel(t, CH, L, R, accent=col if last else None, fill=.085 if last else .055)
    ax.add_patch(Circle((L + .055, Y(t + CH / 2)), .0235, fc=PURPLE_D, ec=col,
                        lw=1.6, zorder=4))
    C.text(L + .055, Y(t + CH / 2), n, fontproperties=ANTON, fontsize=14,
           color=col, ha='center', zorder=5)
    C.text(L + .105, Y(t + .215), title, fontproperties=BC_B, fontsize=15.5,
           color=col if last else WHITE, zorder=4)
    C.text(L + .105, Y(t + .455), l1, fontproperties=BC_M, fontsize=11.4,
           color=LAV, zorder=4)
    C.text(L + .105, Y(t + .625), l2, fontproperties=BC_M, fontsize=11.4,
           color=LAV, zorder=4)
    if not last:
        ax.add_patch(Polygon([[.5, Y(t + CH + .075)], [.489, Y(t + CH + .012)],
                              [.511, Y(t + CH + .012)]], closed=True,
                             fc=GREEN, alpha=.4, ec='none', zorder=3))

# ------------------------------------------------------------------ footer
BOT = TOP + (len(STEPS) - 1) * PITCH + CH
panel(BOT + .24, .56, L, R, accent=PINK, fill=.09)
C.text(L + .028, Y(BOT + .44), 'THEN IT EMAILS YOU THE ANSWER',
       fontproperties=BC_B, fontsize=14.5, color=PINK, zorder=4)
C.text(L + .028, Y(BOT + .66), 'A fresh table twelve hours after every gameweek ends. No buttons pressed.',
       fontproperties=BC_M, fontsize=11.2, color=LAV, zorder=4)

C.text(.5, Y(H - .26), 'Dixon-Coles expected-goals model  ·  Monte Carlo simulation  ·  Python + GitHub Actions',
       fontproperties=BC_M, fontsize=10.5, color=LAV, alpha=.6, ha='center', zorder=4)

C.apply()
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'outputs', 'architecture.png')
fig.savefig(out, facecolor=PURPLE_D)
print(f'{out}   last panel ends {BOT + .78:.2f} of {H}')
