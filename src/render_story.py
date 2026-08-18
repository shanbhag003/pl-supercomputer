"""9:16 story version of the standings, for Instagram and WhatsApp Status."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import (backdrop, Centred, ANTON, BC_B, BC_M, PURPLE_D,
                    PINK, CYAN, GREEN, WHITE, LAV, CLUB, SHORT, _pct, _tier)

W, H = 6.0, 10.667          # 1080 x 1920
L, R = .075, .925


def render_story(pred, out_path, gw_label, n_sims=20000, top=10):
    d = pred.head(top).reset_index(drop=True)
    fig = plt.figure(figsize=(W, H), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    backdrop(ax, glow_x=.5, glow_y=.06)
    C = Centred(fig, ax)

    def Y(p):
        return 1 - p / H

    C.text(.5, Y(1.45), gw_label.upper(), fontproperties=BC_B, fontsize=15,
           color=CYAN, ha='center', zorder=4)
    ax.text(.5, Y(2.15), '2026-27', fontproperties=ANTON, fontsize=54,
            color=WHITE, ha='center', va='center', zorder=4)
    ax.text(.5, Y(2.78), 'PREMIER LEAGUE', fontproperties=ANTON, fontsize=30,
            color=WHITE, ha='center', va='center', zorder=4)
    ax.text(.5, Y(3.32), 'MY PREDICTIONS', fontproperties=ANTON, fontsize=30,
            color=PINK, ha='center', va='center', zorder=4)

    ax.add_patch(Rectangle((L, Y(3.95)), R - L, .045 / H, color=PINK, zorder=3))
    for lbl, x in [('xPTS', .70), ('TITLE', .87)]:
        C.text(x, Y(4.28), lbl, fontproperties=BC_B, fontsize=13, color=WHITE,
               ha='center', alpha=.7, zorder=4)

    TOP, ROW = 4.50, .49
    for i, r in d.iterrows():
        yc = Y(TOP + i * ROW + ROW / 2)
        h = (ROW - .085) / H
        zone = GREEN if i < 4 else (CYAN if i < 6 else None)
        ax.add_patch(FancyBboxPatch((L, yc - h / 2), R - L, h,
                                    boxstyle='round,pad=0,rounding_size=.014',
                                    fc=WHITE, ec='none', alpha=.075, zorder=2))
        if zone:
            ax.add_patch(Rectangle((L, yc - h / 2), .009, h, color=zone, zorder=3))
        C.text(L + .05, yc, str(i + 1), fontproperties=ANTON, fontsize=20,
               color=WHITE, alpha=.5, ha='center', zorder=4)
        ax.add_patch(Rectangle((L + .085, yc - h / 2), .012, h,
                               color=CLUB.get(r.team, '#888'), zorder=3))
        C.text(L + .115, yc, SHORT.get(r.team, r.team.upper()),
               fontproperties=BC_B, fontsize=22, color=WHITE, zorder=4)
        C.text(.70, yc, f'{r.xPts:.1f}', fontproperties=BC_B, fontsize=22,
               color=WHITE, ha='center', zorder=4)
        tv = 100 * r.title
        C.text(.87, yc, _pct(tv), fontproperties=BC_B, fontsize=22,
               color={'g': GREEN, 'c': CYAN, 'd': '#9C7FA6'}.get(
                   _tier(tv), WHITE), ha='center', zorder=4)

    yb = TOP + top * ROW + .42
    C.text(.5, Y(yb), f'{n_sims:,} SIMULATIONS OF THE SEASON',
           fontproperties=BC_B, fontsize=15, color=PINK, ha='center', zorder=4)
    C.text(.5, Y(yb + .34), 'Full table, all 20 clubs, updated every gameweek',
           fontproperties=BC_M, fontsize=14.5, color=LAV, ha='center', zorder=4)
    C.text(.5, Y(yb + .68), 'shanbhag003.github.io/pl-supercomputer',
           fontproperties=BC_B, fontsize=14, color=CYAN, ha='center', zorder=4)

    C.apply()
    fig.savefig(out_path, facecolor=PURPLE_D)
    plt.close(fig)
    return out_path


if __name__ == '__main__':
    import pandas as pd
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pred = pd.read_csv(f'{root}/outputs/prediction_latest.csv')
    lbl = sys.argv[1] if len(sys.argv) > 1 else 'PRE-SEASON'
    print(render_story(pred, f'{root}/outputs/story.png', lbl))
