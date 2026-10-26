# -*- coding: utf-8 -*-
"""Monte Carlo accuracy of the project's solver over the deployment area."""
import json
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import sim_core as S

OUT_PNG, OUT_JSON = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(20261025)

STEP = 0.5
TRIALS = 120
xs = np.arange(S.XMIN + STEP / 2, S.XMAX, STEP)
ys = np.arange(S.YMIN + STEP / 2, S.YMAX, STEP)


def run(n_true, trials):
    med = np.full((len(ys), len(xs)), np.nan)
    all_err, all_in = [], []
    fails = 0
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            p = (x, y)
            errs = []
            for _ in range(trials):
                frames = np.array([S.frame_rssi(p, rng, n_true=n_true) for _ in range(S.WINDOW)])
                est = S.solve_from_avg(frames.mean(axis=0))
                if est is None:
                    fails += 1
                    continue
                errs.append(np.hypot(est[0] - x, est[1] - y))
            if errs:
                med[iy, ix] = np.median(errs)
                all_err.extend(errs)
                all_in.extend([bool(S.inside_triangle(x, y))] * len(errs))
    return med, np.array(all_err), np.array(all_in), fails


t0 = time.time()
med, err, isin, fails = run(S.N_EXP, TRIALS)
t1 = time.time()
med_mm, err_mm, isin_mm, fails_mm = run(2.7, 40)     # real decay n=2.7, backend still assumes 2.0
t2 = time.time()

stats = {
    "sigma_db": S.SIGMA_DB, "window": S.WINDOW, "tx": S.TX_POWER, "n": S.N_EXP,
    "grid_step_m": STEP, "trials_per_point": TRIALS, "points": int(med.size),
    "solves": int(err.size + fails), "fails": int(fails),
    "median_in": float(np.median(err[isin])),
    "median_out": float(np.median(err[~isin])),
    "p90_in": float(np.percentile(err[isin], 90)),
    "within2_in": float(np.mean(err[isin] < 2.0)),
    "within2_out": float(np.mean(err[~isin] < 2.0)),
    "mm_n_true": 2.7, "mm_trials_per_point": 40,
    "mm_median_in": float(np.median(err_mm[isin_mm])),
    "mm_median_out": float(np.median(err_mm[~isin_mm])),
    "secs": [round(t1 - t0, 1), round(t2 - t1, 1)],
}
json.dump(stats, open(OUT_JSON, "w"), indent=1)
np.save(OUT_JSON.replace(".json", "_med.npy"), med)
print(json.dumps(stats, indent=1))

# ------------------------------------------------------------------ figure
NAVY, INK, SLATE, RULE = "#16416E", "#101823", "#5C6875", "#DFE5EB"
# single-hue sequential ramp (rust), light -> dark
cmap = LinearSegmentedColormap.from_list(
    "rust", ["#FFFFFF", "#F6E6DE", "#E9B79C", "#CF7A4E", "#B14A20", "#6E2A10"])
VMAX = 6.0

plt.rcParams.update({"font.family": "Calibri", "font.size": 10.5,
                     "axes.edgecolor": RULE, "xtick.color": SLATE, "ytick.color": SLATE})
fig = plt.figure(figsize=(7.2, 4.95), dpi=220)
ax = fig.add_axes([0.40 / 7.2, 0.38 / 4.95, 6.0 / 7.2, 4.5 / 4.95])
cax = fig.add_axes([6.55 / 7.2, 0.38 / 4.95, 0.13 / 7.2, 4.5 / 4.95])

im = ax.imshow(np.clip(med, 0, VMAX), origin="lower", cmap=cmap, vmin=0, vmax=VMAX,
               extent=(S.XMIN, S.XMAX, S.YMIN, S.YMAX), interpolation="bilinear", aspect="equal")
X, Y = np.meshgrid(xs, ys)
cs = ax.contour(X, Y, med, levels=[1, 2, 4], colors=INK, linewidths=0.7, alpha=0.75)
ax.clabel(cs, fmt=lambda v: f"{v:.0f} m", fontsize=9, colors=INK, inline_spacing=4)

tri = np.vstack([S.NODES, S.NODES[:1]])
ax.plot(tri[:, 0], tri[:, 1], ls=(0, (4, 3)), color=NAVY, lw=1.1)
for (x, y), lab, (dx, dy, ha) in zip(S.NODES, ["N1", "N2", "N3"],
                                     [(-0.5, -1.0, "right"), (0.5, -1.0, "left"), (0.6, 0.45, "left")]):
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
cb.set_ticks([0, 1, 2, 3, 4, 5, 6])
cb.set_ticklabels(["0", "1", "2", "3", "4", "5", "6+"])
cb.ax.tick_params(length=0, colors=SLATE, labelsize=9.5)
cb.set_label("median position error (m)", color=SLATE, fontsize=10)
fig.savefig(OUT_PNG, facecolor="white")
print("wrote", OUT_PNG)
