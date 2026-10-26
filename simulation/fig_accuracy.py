# -*- coding: utf-8 -*-
"""Accuracy-map figure for the results slide, from the saved Monte Carlo field."""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import sim_core as S

MED, OUT = sys.argv[1], sys.argv[2]
med = np.load(MED)
STEP = 0.5
xs = np.arange(S.XMIN + STEP / 2, S.XMAX, STEP)
ys = np.arange(S.YMIN + STEP / 2, S.YMAX, STEP)


def smooth(a, sigma=1.6):
    """Separable Gaussian, edge-padded - display only; the stats come from the raw field."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    p = np.pad(a, r, mode="edge")
    p = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, p)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, p)


sm = smooth(med)

NAVY, INK, SLATE = "#16416E", "#101823", "#5C6875"
# one hue, light -> dark; tops out at a mid rust so the field never dominates the slide
cmap = LinearSegmentedColormap.from_list(
    "err", ["#FFFFFF", "#F6E6DE", "#EDC6AF", "#DE9C77", "#C9693E"])
VMAX = 8.0

plt.rcParams.update({"font.family": "Calibri", "font.size": 10.5,
                     "xtick.color": SLATE, "ytick.color": SLATE})
fig = plt.figure(figsize=(7.2, 4.95), dpi=220)
ax = fig.add_axes([0.40 / 7.2, 0.38 / 4.95, 6.0 / 7.2, 4.5 / 4.95])
cax = fig.add_axes([6.55 / 7.2, 0.38 / 4.95, 0.13 / 7.2, 4.5 / 4.95])

im = ax.imshow(np.clip(sm, 0, VMAX), origin="lower", cmap=cmap, vmin=0, vmax=VMAX,
               extent=(S.XMIN, S.XMAX, S.YMIN, S.YMAX), interpolation="bilinear", aspect="equal")
X, Y = np.meshgrid(xs, ys)
cs = ax.contour(X, Y, sm, levels=[3, 5], colors=INK, linewidths=0.8, alpha=0.8)
ax.clabel(cs, fmt=lambda v: f"{v:.0f} m", fontsize=9.5, colors=INK, inline_spacing=5)

tri = np.vstack([S.NODES, S.NODES[:1]])
ax.plot(tri[:, 0], tri[:, 1], ls=(0, (4, 3)), color=NAVY, lw=1.2)
for (x, y), lab, (dx, dy, ha) in zip(S.NODES, ["N1", "N2", "N3"],
                                     [(-0.55, -0.95, "right"), (0.55, -0.95, "left"), (0.65, 0.5, "left")]):
    ax.plot(x, y, marker="s", ms=7.5, color=NAVY, mec="white", mew=1.4, zorder=5)
    ax.text(x + dx, y + dy, lab, fontfamily="Arial", fontweight="bold", fontsize=10,
            color=NAVY, ha=ha, va="center", zorder=6)

ax.set_xlim(S.XMIN, S.XMAX); ax.set_ylim(S.YMIN, S.YMAX)
ax.set_xticks(range(-4, 25, 4)); ax.set_yticks(range(-2, 19, 4))
ax.tick_params(length=0, pad=4)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xlabel("x (m)", color=SLATE, labelpad=3)
ax.set_ylabel("y (m)", color=SLATE, labelpad=2)

cb = fig.colorbar(im, cax=cax)
cb.outline.set_visible(False)
cb.set_ticks([0, 2, 4, 6, 8])
cb.set_ticklabels(["0", "2", "4", "6", "8+"])
cb.ax.tick_params(length=0, colors=SLATE, labelsize=9.5)
cb.set_label("median position error (m)", color=SLATE, fontsize=10)
fig.savefig(OUT, facecolor="white")
print("wrote", OUT)
