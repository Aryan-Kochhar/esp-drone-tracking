# -*- coding: utf-8 -*-
"""Accuracy map sized for the short deck: drawn at its on-slide size so labels stay >= 13 pt."""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import sim_core as S


def smooth(a, sigma=1.6):
    """Separable Gaussian, edge-padded - display only (same as fig_accuracy.py)."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    p = np.pad(a, r, mode="edge")
    p = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, p)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, p)


MED, OUT = sys.argv[1], sys.argv[2]
med = np.load(MED)
STEP = 0.5
xs = np.arange(S.XMIN + STEP / 2, S.XMAX, STEP)
ys = np.arange(S.YMIN + STEP / 2, S.YMAX, STEP)
sm = smooth(med)

NAVY, INK, SLATE = "#16416E", "#101823", "#5C6875"
cmap = LinearSegmentedColormap.from_list("err", ["#FFFFFF", "#F6E6DE", "#EDC6AF", "#DE9C77", "#C9693E"])
VMAX = 8.0

FW, FH = 6.4, 4.55                       # inches = size on the slide
plt.rcParams.update({"font.family": "Calibri", "font.size": 14, "xtick.color": SLATE, "ytick.color": SLATE})
fig = plt.figure(figsize=(FW, FH), dpi=300)
PW = 5.05                                # plot width; height follows the 4:3 world
PH = PW * (S.YMAX - S.YMIN) / (S.XMAX - S.XMIN)
ax = fig.add_axes([0.52 / FW, 0.52 / FH, PW / FW, PH / FH])
cax = fig.add_axes([(0.52 + PW + 0.18) / FW, 0.52 / FH, 0.13 / FW, PH / FH])

im = ax.imshow(np.clip(sm, 0, VMAX), origin="lower", cmap=cmap, vmin=0, vmax=VMAX,
               extent=(S.XMIN, S.XMAX, S.YMIN, S.YMAX), interpolation="bilinear", aspect="equal")
X, Y = np.meshgrid(xs, ys)
cs = ax.contour(X, Y, sm, levels=[3, 5], colors=INK, linewidths=1.0, alpha=0.85)
ax.clabel(cs, fmt=lambda v: f"{v:.0f} m", fontsize=14, colors=INK, inline_spacing=6)

tri = np.vstack([S.NODES, S.NODES[:1]])
ax.plot(tri[:, 0], tri[:, 1], ls=(0, (4, 3)), color=NAVY, lw=1.6)
for (x, y), lab, (dx, dy, ha) in zip(S.NODES, ["N1", "N2", "N3"],
                                     [(-0.6, -1.1, "right"), (0.6, -1.1, "left"), (0.8, 0.55, "left")]):
    ax.plot(x, y, marker="s", ms=10, color=NAVY, mec="white", mew=1.6, zorder=5)
    ax.text(x + dx, y + dy, lab, fontfamily="Arial", fontweight="bold", fontsize=15,
            color=NAVY, ha=ha, va="center", zorder=6)

ax.set_xlim(S.XMIN, S.XMAX); ax.set_ylim(S.YMIN, S.YMAX)
ax.set_xticks([0, 8, 16, 24]); ax.set_yticks([0, 6, 12, 18])
ax.tick_params(length=0, pad=5, labelsize=14)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xlabel("metres", color=SLATE, fontsize=14, labelpad=4)

cb = fig.colorbar(im, cax=cax)
cb.outline.set_visible(False)
cb.set_ticks([0, 4, 8])
cb.set_ticklabels(["0 m", "4 m", "8 m+"])
cb.ax.tick_params(length=0, colors=SLATE, labelsize=14)
fig.savefig(OUT, facecolor="white")
print("wrote", OUT, f"({FW}x{FH} in)")
