# -*- coding: utf-8 -*-
"""
Three-UAV crowd-monitoring simulation, rendered in the deck's design system.

    python render_sim.py frames OUTDIR t1 t2 ...   -> PNG stills at those times (QA)
    python render_sim.py video  OUT.mp4            -> full 1920x1080 / 30 fps H.264

Every position estimate shown is computed by the project's backend solver
(backend/trilateration.py) from simulated RSSI, via the same 5-frame average utils.py uses.
"""
import os
import subprocess
import sys
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Rectangle

import sim_core as S

MODE = sys.argv[1]
W, H, FPS, DUR = 1920, 1080, 30, 42.0
F = int(DUR * FPS)
DT = 1.0 / FPS
SEED = int(os.environ.get("SIM_SEED", "23"))

NAVY, INK, SLATE, RUST = "#16416E", "#101823", "#5C6875", "#B14A20"
CARD, WARM, RULE, WHITE, STAGE_C = "#F1F4F7", "#F6E6DE", "#DFE5EB", "#FFFFFF", "#CDD5DF"
LED_OFF = "#9FB0C2"

PHASES = [  # start, end, chip, title, lede
    (0, 5, "DEPLOY", "Three UAVs take formation.",
     "Each UAV carries one ESP32 node, hovering at the positions in config/node_positions.json."),
    (5, 12, "LISTEN", "Every phone is broadcasting.",
     "Phones send probe requests on their own. Each node logs the sender’s MAC and signal strength."),
    (12, 19, "RANGE", "Signal strength becomes distance.",
     "For one phone: five frames averaged per node, converted to metres by the log-distance model."),
    (19, 26, "SOLVE", "Three circles, one position.",
     "The backend’s least-squares solver finds where the circles meet — then does it for every phone."),
    (26, 42, "MAP", "The crowd moves. The map follows.",
     "Positions refresh every second. When Zone B passes twice the venue average, an alert fires."),
]
T_LISTEN = 5.0
T_FOCUS = 12.3          # focus device's averaged RSSI is frozen here for the walkthrough
T_SOLVE_ALL = 21.5
T_MAP = 26.0
THRESH = 2.0
ZONE = (5.0, 4.4, 10.0, 5.6)         # x, y, w, h  (stage front)
STAGE = (5.0, 10.4, 10.0, 1.1)
AUDIENCE = (-2.0, -2.0, 24.0, 12.0)  # baseline area for 'venue average'
N_PEOPLE = 84
PROBE_RATE = 0.9        # probes / s / phone (accelerated; real phones probe far less often)
FOCUS_MAC = "DA:2A:9E:5D:B1:7F"


def pt(px):
    return px * 0.72


def clamp01(x):
    return max(0.0, min(1.0, x))


def ease(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def win(t, a, b, fin=0.4, fout=0.4):
    return clamp01((t - a) / fin) * clamp01((b - t) / fout)


def set_a(txt, a):
    """Text alpha that also governs its bbox, and hides it entirely at zero."""
    txt.set_alpha(a)
    bp = txt.get_bbox_patch()
    if bp is not None:
        bp.set_alpha(a)
    txt.set_visible(a > 0.003)


# Calibri ships embedded bitmap strikes at these pixel sizes; this matplotlib build draws
# them blank at 100 dpi. Any Calibri text at one of these sizes is a bug - see check_fonts().
CALIBRI_BROKEN_PX = {12, 13, 15, 16, 17, 19}


def check_fonts(figure):
    import matplotlib.text as mtext
    bad = []
    for txt in figure.findobj(mtext.Text):
        if not txt.get_text():
            continue
        fam = txt.get_fontfamily()
        fam = fam[0] if isinstance(fam, (list, tuple)) else fam
        if fam == "Calibri" and round(txt.get_fontsize() / 0.72) in CALIBRI_BROKEN_PX:
            bad.append((txt.get_text()[:40], round(txt.get_fontsize() / 0.72)))
    if bad:
        raise SystemExit(f"Calibri at a broken pixel size: {bad}")


# =============================================================== simulation
def precompute(seed):
    rng = np.random.default_rng(seed)
    n = N_PEOPLE
    pos = np.empty((n, 2))
    pos[0] = (8.6, 5.4)                                   # focus device
    k = n - 1
    n_grp = int(k * 0.4)
    centres = np.c_[rng.uniform(1, 19, 6), rng.uniform(0.0, 8.5, 6)]
    g = centres[rng.integers(0, 6, n_grp)] + rng.normal(0, 1.0, (n_grp, 2))
    u = np.c_[rng.uniform(-1.5, 21.5, k - n_grp), rng.uniform(-1.6, 9.6, k - n_grp)]
    pos[1:] = np.vstack([g, u])
    vel = rng.normal(0, 0.2, (n, 2))
    drift = rng.random(n) < 0.72
    drift[0] = False
    target = np.c_[rng.uniform(5.4, 14.6, n), rng.uniform(4.8, 9.9, n)]

    P = np.zeros((F, n, 2))
    emits = []
    bufs = [[deque(maxlen=S.WINDOW) for _ in range(3)] for _ in range(n)]
    frames_cnt = np.zeros(F, int)
    snaps = {}
    focus_avg = None
    forced = {int(round(12.35 * FPS)), int(round(15.8 * FPS))}
    cnt = 0
    for fi in range(F):
        t = fi * DT
        sf = 0.25 if 12.0 <= t < T_MAP else 1.0
        vel += rng.normal(0, 0.8, (n, 2)) * DT
        vel *= 0.985
        if t >= T_MAP:
            d = target - pos
            dist = np.linalg.norm(d, axis=1, keepdims=True)
            desired = d / np.maximum(dist, 1e-6) * np.minimum(1.3, dist)
            steer = drift[:, None]
            vel = np.where(steer, vel + (desired - vel) * 3.0 * DT, vel)
        sp = np.linalg.norm(vel, axis=1, keepdims=True)
        cap = np.where(drift[:, None] & (t >= T_MAP), 1.35, 0.6)
        vel = vel / np.maximum(sp, 1e-9) * np.minimum(sp, cap)
        vel[0] *= 0.15
        pos = pos + vel * DT * sf
        for ax_i, lo, hi in ((0, -3.5, 23.5), (1, -1.9, 10.1)):
            low, high = pos[:, ax_i] < lo, pos[:, ax_i] > hi
            pos[low, ax_i] = lo
            pos[high, ax_i] = hi
            vel[low | high, ax_i] *= -1
        P[fi] = pos

        e = np.nonzero(rng.random(n) < PROBE_RATE * DT)[0].tolist()
        if fi in forced and 0 not in e:
            e.append(0)
        emits.append(e)
        if t >= T_LISTEN:
            for i in e:
                r = S.frame_rssi(pos[i], rng)
                for j in range(3):
                    if r[j] >= S.RSSI_FLOOR:
                        bufs[i][j].append(r[j])
                        cnt += 1
        frames_cnt[fi] = cnt

        if fi == int(round(T_FOCUS * FPS)):
            focus_avg = [float(np.mean(bufs[0][j])) for j in range(3)]
        if (fi == int(round(T_SOLVE_ALL * FPS))) or (t >= 22.0 and fi % FPS == 0):
            est = np.full((n, 2), np.nan)
            for i in range(n):
                if all(len(bufs[i][j]) for j in range(3)):
                    r = S.solve_from_avg([np.mean(bufs[i][j]) for j in range(3)])
                    if r is not None:
                        est[i] = r
            snaps[fi] = est

    focus_est = np.array(S.solve_from_avg(focus_avg))
    focus_dist = [S.rssi_to_distance(r) for r in focus_avg]
    reveal = T_SOLVE_ALL + 0.1 + 2.8 * rng.random(n)
    reveal[0] = T_SOLVE_ALL + 0.05
    return dict(P=P, emits=emits, frames_cnt=frames_cnt, snaps=snaps,
                focus_avg=focus_avg, focus_dist=focus_dist, focus_est=focus_est,
                focus_true=P[int(round(T_FOCUS * FPS)), 0].copy(), reveal=reveal)


SIM = precompute(SEED)
P, EMITS, SNAPS = SIM["P"], SIM["emits"], SIM["snaps"]
SNAP_FR = sorted(SNAPS)
FOCUS_ERR = float(np.hypot(*(SIM["focus_est"] - SIM["focus_true"])))

# map extent (equal aspect, wider than the world so it fills the column)
MAP_Y0, MAP_Y1 = S.YMIN, S.YMAX
MAP_HALF = (786 / (470 / (MAP_Y1 - MAP_Y0))) / 2
MAP_X0, MAP_X1 = 10 - MAP_HALF, 10 + MAP_HALF
GX = np.arange(MAP_X0, MAP_X1, 0.25)
GY = np.arange(MAP_Y0, MAP_Y1, 0.25)
GXX, GYY = np.meshgrid(GX, GY)


def kde(points, h=1.5):
    pts = points[~np.isnan(points).any(axis=1)]
    if len(pts) == 0:
        return np.zeros_like(GXX)
    d2 = (GXX[..., None] - pts[:, 0]) ** 2 + (GYY[..., None] - pts[:, 1]) ** 2
    return np.exp(-d2 / (2 * h * h)).sum(axis=-1)


def zone_ratio(points):
    pts = points[~np.isnan(points).any(axis=1)]
    if len(pts) == 0:
        return 0.0
    x, y, w, h = ZONE
    inz = ((pts[:, 0] >= x) & (pts[:, 0] <= x + w) & (pts[:, 1] >= y) & (pts[:, 1] <= y + h)).sum()
    ax_, ay_, aw_, ah_ = AUDIENCE
    ina = ((pts[:, 0] >= ax_) & (pts[:, 0] <= ax_ + aw_) & (pts[:, 1] >= ay_) & (pts[:, 1] <= ay_ + ah_)).sum()
    return (inz / (w * h)) / (max(ina, 1) / (aw_ * ah_))


HEAT = {fr: kde(SNAPS[fr]) for fr in SNAP_FR if fr >= int((T_MAP - 1) * FPS)}
HEAT_MAX = max(h.max() for h in HEAT.values()) * 0.85
RATIO = {fr: zone_ratio(SNAPS[fr]) for fr in SNAP_FR}
MEDERR = {}
for fr in SNAP_FR:
    e = SNAPS[fr]
    ok = ~np.isnan(e).any(axis=1)
    MEDERR[fr] = float(np.median(np.hypot(*(e[ok] - P[fr][ok]).T))) if ok.any() else np.nan

# ================================================================ figure
plt.rcParams["font.family"] = "Calibri"
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
fig.patch.set_facecolor(WHITE)


def axes_px(x, y, w, h):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


ui = axes_px(0, 0, W, H)
ui.set_xlim(0, W); ui.set_ylim(H, 0); ui.axis("off"); ui.set_zorder(0)


def utext(x, y, s, size, color, weight="normal", family="Calibri", ha="left", va="top", **kw):
    return ui.text(x, y, s, fontsize=pt(size), color=color, fontweight=weight,
                   fontfamily=family, ha=ha, va=va, **kw)


M_L, M_R = 89, 1831
eyebrow = utext(M_L, 54, "", 26, NAVY, "bold", "Arial")
title = utext(M_L, 104, "", 76, INK, "bold", "Arial")
lede = utext(M_L, 207, "", 30, SLATE)

chips = []
CHIP_W, CHIP_H, CHIP_G = 150, 42, 10
cx0 = M_R - (5 * CHIP_W + 4 * CHIP_G)
for i, ph in enumerate(PHASES):
    x = cx0 + i * (CHIP_W + CHIP_G)
    r = Rectangle((x, 46), CHIP_W, CHIP_H, fc=CARD, ec="none")
    ui.add_patch(r)
    tx = utext(x + CHIP_W / 2, 46 + CHIP_H / 2 + 1, f"{i + 1:02d} {ph[2]}", 20, SLATE,
               family="Consolas", ha="center", va="center")
    chips.append((r, tx))

utext(M_L, 266, "WHAT IS HAPPENING", 24, SLATE, "bold", "Arial")
utext(1045, 266, "WHAT THE SYSTEM SEES", 24, SLATE, "bold", "Arial")
utext(M_L, 1008, "Simulated data. Every estimate is computed by the project’s own solver, "
      "backend/trilateration.py. Time accelerated.", 18, SLATE)

tiles = []
for i, lab in enumerate(["FRAMES\nCAPTURED", "DEVICES\nLOCATED", "MEDIAN\nERROR", "ZONE B\nDENSITY"]):
    x = 1045 + i * (187.5 + 12)
    ui.add_patch(Rectangle((x, 794), 187.5, 196, fc=CARD, ec="none"))
    num = utext(x + 16, 812, "—", 54, NAVY, "bold", "Arial")
    utext(x + 16, 900, lab, 18, INK, "bold", "Arial", linespacing=1.2)
    tiles.append(num)

# ---------------------------------------------------------------- scene
sc = axes_px(89, 306, 912, 684)
sc.set_zorder(1)
sc.set_xlim(S.XMIN, S.XMAX); sc.set_ylim(S.YMIN, S.YMAX); sc.set_aspect("equal")
sc.set_facecolor(CARD)
sc.set_xticks([]); sc.set_yticks([])
for s_ in sc.spines.values():
    s_.set_visible(False)
for gx in range(-4, 25, 2):
    sc.axvline(gx, color=RULE, lw=0.7, zorder=0)
for gy in range(-2, 19, 2):
    sc.axhline(gy, color=RULE, lw=0.7, zorder=0)
sc.add_patch(Rectangle((S.XMIN, 11.5), S.XMAX - S.XMIN, S.YMAX - 11.5, fc=to_rgba("#D9DFE7", 0.5),
                       ec="none", zorder=0.5))
sc.text(S.XMIN + 0.5, 11.85, "BACKSTAGE", fontfamily="Arial", fontweight="bold", fontsize=pt(18),
        color=SLATE, ha="left", va="bottom", zorder=1, alpha=0.85)
sc.add_patch(Rectangle(STAGE[:2], STAGE[2], STAGE[3], fc=STAGE_C, ec="none", zorder=1))
sc.text(10, 10.95, "STAGE", fontfamily="Arial", fontweight="bold", fontsize=pt(22), color=SLATE,
        ha="center", va="center", zorder=2)
sc.plot([-3.2, 1.8], [-2.45, -2.45], color=SLATE, lw=1.6, solid_capstyle="butt", zorder=2)
sc.text(-0.7, -2.2, "5 m", fontsize=pt(18), color=SLATE, ha="center", va="bottom", zorder=2)

sc_zone = Rectangle(ZONE[:2], ZONE[2], ZONE[3], fill=False, ec=SLATE, lw=1.4, ls=(0, (5, 4)), zorder=2, alpha=0)
sc.add_patch(sc_zone)
sc_zone_lab = sc.text(ZONE[0] - 0.35, ZONE[1] + ZONE[3] - 0.05, "ZONE B  ·  STAGE FRONT", fontfamily="Arial",
                      fontweight="bold", fontsize=pt(18), color=SLATE, ha="right", va="top", zorder=6, alpha=0,
                      bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.2", fc=to_rgba(CARD, 0.92), ec="none"))

tri = np.vstack([S.NODES, S.NODES[:1]])
formation, = sc.plot(tri[:, 0], tri[:, 1], ls=(0, (6, 5)), color=NAVY, lw=1.4, alpha=0, zorder=2)
base_labs = []
for (x, y), s_ in (((10, -0.75), "20 m"), ((3.7, 8.0), "18 m"), ((16.3, 8.0), "18 m")):
    base_labs.append(sc.text(x, y, s_, fontsize=pt(20), color=NAVY, ha="center", va="center",
                             zorder=2, alpha=0))

links = LineCollection([], linewidths=0.9, zorder=2)
sc.add_collection(links)
pulses = sc.scatter([], [], s=[], facecolors="none", edgecolors=[], linewidths=1.2, zorder=3)
people = sc.scatter(P[0, :, 0], P[0, :, 1], s=46, c=SLATE, edgecolors=WHITE, linewidths=0.9, zorder=4)
whisk = LineCollection([], colors=RUST, linewidths=0.9, zorder=4)
sc.add_collection(whisk)
crosses = sc.scatter([], [], s=30, marker="x", c=RUST, linewidths=1.4, zorder=5)

range_circ = [Circle(tuple(S.NODES[j]), 0.01, fc=to_rgba(NAVY, 0), ec=to_rgba(NAVY, 0), lw=1.8, zorder=3)
              for j in range(3)]
for c in range_circ:
    sc.add_patch(c)
focus_links = LineCollection([], colors=NAVY, linewidths=2.2, zorder=3)
sc.add_collection(focus_links)
halo = sc.scatter([], [], s=[], facecolors="none", edgecolors=RUST, linewidths=1.6, zorder=6)
focus_dot = sc.scatter([], [], s=125, c=INK, edgecolors=WHITE, linewidths=1.4, zorder=7)
focus_lab = sc.text(0, 0, f"MAC {FOCUS_MAC}", fontfamily="Consolas", fontsize=pt(20), color=INK,
                    ha="left", va="bottom", zorder=8, alpha=0,
                    bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.25", fc=WHITE, ec=RULE, lw=1))
READ_POS = [((1.4, -1.6), "left"), ((18.6, -1.6), "right"), ((8.6, 14.95), "right")]
readouts = [sc.text(x, y, "", fontfamily="Consolas", fontsize=pt(22), color=INK, ha=ha, va="center",
                    zorder=9, alpha=0,
                    bbox=dict(boxstyle="round,pad=0.4,rounding_size=0.25", fc=WHITE, ec=RULE, lw=1))
            for (x, y), ha in READ_POS]
lock = sc.scatter([], [], s=[], facecolors="none", edgecolors=RUST, linewidths=2.0, zorder=8)
est_dot = sc.scatter([], [], s=145, c=RUST, edgecolors=WHITE, linewidths=1.6, zorder=8)
err_line, = sc.plot([], [], ls=(0, (3, 2)), color=RUST, lw=1.6, zorder=7)
est_lab = sc.text(0, 0, "", fontfamily="Consolas", fontsize=pt(20), color=RUST, ha="left", va="top",
                  zorder=9, alpha=0,
                  bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.25", fc=WHITE, ec=RULE, lw=1))

DRONE_START = [(-9.0, -8.0), (29.0, -8.0), (10.0, 25.0)]
ARM = 0.78
drones = []
for j in range(3):
    d = {}
    d["shadow"] = Ellipse((0, 0), 2.0, 2.0, fc=to_rgba(INK, 0.09), ec="none", zorder=6)
    sc.add_patch(d["shadow"])
    d["arms"] = [sc.plot([], [], color=INK, lw=2.6, solid_capstyle="round", zorder=7)[0] for _ in range(2)]
    d["blur"] = [Circle((0, 0), 0.34, fc=to_rgba(NAVY, 0.16), ec="none", zorder=7) for _ in range(4)]
    d["rot"] = [Circle((0, 0), 0.38, fc="none", ec=INK, lw=1.3, zorder=8) for _ in range(4)]
    d["blades"] = [sc.plot([], [], color=INK, lw=1.2, alpha=0.65, zorder=8)[0] for _ in range(4)]
    for p_ in d["blur"] + d["rot"]:
        sc.add_patch(p_)
    d["body"] = Rectangle((0, 0), 0.6, 0.6, fc=NAVY, ec="none", zorder=9)
    sc.add_patch(d["body"])
    d["led"] = Circle((0, 0), 0.12, fc=LED_OFF, ec="none", zorder=10)
    sc.add_patch(d["led"])
    ha = ["right", "left", "left"][j]
    d["lab"] = sc.text(0, 0, f"N{j + 1}", fontfamily="Arial", fontweight="bold", fontsize=pt(26),
                       color=NAVY, ha=ha, va="bottom", zorder=10, alpha=0)
    d["sub"] = sc.text(0, 0, f"UAV-{j + 1}", fontsize=pt(18), color=SLATE, ha=ha, va="top",
                       zorder=10, alpha=0)
    drones.append(d)

# ---------------------------------------------------------------- system map
mp = axes_px(1045, 306, 786, 470)
mp.set_zorder(1)
mp.set_xlim(MAP_X0, MAP_X1); mp.set_ylim(MAP_Y0, MAP_Y1); mp.set_aspect("equal")
mp.set_facecolor(WHITE)
mp.set_xticks([]); mp.set_yticks([])
for s_ in mp.spines.values():
    s_.set_color(RULE); s_.set_linewidth(1.4)
for gx in range(-6, 27, 2):
    mp.axvline(gx, color=RULE, lw=0.5, alpha=0.7, zorder=0)
for gy in range(-2, 19, 2):
    mp.axhline(gy, color=RULE, lw=0.5, alpha=0.7, zorder=0)
mp.add_patch(Rectangle(STAGE[:2], STAGE[2], STAGE[3], fc=CARD, ec="none", zorder=1))
mp.text(10, 10.95, "STAGE", fontfamily="Arial", fontweight="bold", fontsize=pt(16), color=SLATE,
        ha="center", va="center", zorder=1)
heat_cmap = LinearSegmentedColormap.from_list(
    "heat", [(1, 1, 1, 0), to_rgba(WARM, 0.75), to_rgba("#DE9C77", 0.85), to_rgba(RUST, 0.9)])
heat = mp.imshow(np.zeros((len(GY), len(GX), 4)), origin="lower",
                 extent=(MAP_X0, MAP_X1, MAP_Y0, MAP_Y1), aspect="auto", zorder=1, interpolation="bilinear")
mp_zone = Rectangle(ZONE[:2], ZONE[2], ZONE[3], fill=False, ec=SLATE, lw=1.3, ls=(0, (5, 4)), zorder=3, alpha=0)
mp.add_patch(mp_zone)
mp_zone_lab = mp.text(ZONE[0] - 0.35, ZONE[1] + ZONE[3] - 0.05, "ZONE B", fontfamily="Arial", fontweight="bold",
                      fontsize=pt(18), color=SLATE, ha="right", va="top", zorder=6, alpha=0,
                      bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.2", fc=to_rgba(WHITE, 0.88), ec="none"))
for (x, y), lab, (dx, dy, ha) in zip(S.NODES, ["N1", "N2", "N3"],
                                     [(-0.5, -0.9, "right"), (0.5, -0.9, "left"), (0.6, 0.3, "left")]):
    mp.plot(x, y, marker="s", ms=8, color=NAVY, mec=WHITE, mew=1.3, zorder=6)
    mp.text(x + dx, y + dy, lab, fontfamily="Arial", fontweight="bold", fontsize=pt(18), color=NAVY,
            ha=ha, va="center", zorder=6)
trails = LineCollection([], colors=to_rgba(RUST, 0.14), linewidths=1.0, zorder=4)
mp.add_collection(trails)
est_pts = mp.scatter([], [], s=42, c=RUST, edgecolors=WHITE, linewidths=0.9, zorder=5)
status = mp.text(10, 5.0, "", fontsize=pt(24), color=SLATE, ha="center", va="center", zorder=7)
live_lab = mp.text(MAP_X1 - 0.6, MAP_Y1 - 0.7, "LIVE · 1 Hz", fontfamily="Consolas", fontsize=pt(18),
                   color=SLATE, ha="right", va="top", zorder=7, alpha=0)
live_dot = mp.scatter([MAP_X1 - 6.3], [MAP_Y1 - 1.35], s=60, c=RUST, zorder=7, alpha=0)
alert = mp.text(MAP_X0 + 0.7, MAP_Y1 - 0.7, "", fontfamily="Arial", fontweight="bold", fontsize=pt(22),
                color=RUST, ha="left", va="top", zorder=8, alpha=0,
                bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.3", fc=WARM, ec="none"))


# ================================================================ per-frame
def phase_of(t):
    for i, ph in enumerate(PHASES):
        if ph[0] <= t < ph[1]:
            return i
    return len(PHASES) - 1


def drone_pos(j, t):
    a = 0.3 + 0.35 * j
    b = a + 3.1
    u = ease((t - a) / (b - a))
    s0 = np.array(DRONE_START[j])
    p = s0 + (S.NODES[j] - s0) * u
    bob = 0.07 * np.sin(2 * np.pi * (0.6 * t + 0.3 * j))
    return p + np.array([0.0, bob * u]), clamp01((t - b + 0.4) / 0.6)


def snap_at(fi):
    """Index into SNAP_FR of the latest refresh at or before frame fi, else -1."""
    k = -1
    for i, fr in enumerate(SNAP_FR):
        if fr <= fi:
            k = i
        else:
            break
    return k


def displayed_estimates(fi):
    k = snap_at(fi)
    if k < 0:
        return None, k, 0.0
    cur = SNAPS[SNAP_FR[k]]
    prev = SNAPS[SNAP_FR[k - 1]] if k > 0 else cur
    u = ease((fi - SNAP_FR[k]) / (0.35 * FPS))
    out = np.where(np.isnan(prev), cur, prev + (cur - prev) * u)
    return out, k, u


def draw(fi):
    t = fi * DT
    ph = phase_of(t)
    p0, p1, chip, ttl, ld = PHASES[ph]

    # header crossfade
    a_in = 1.0 if ph == 0 else clamp01((t - p0) / 0.35)
    a_out = 1.0 if ph == len(PHASES) - 1 else clamp01((p1 - t) / 0.25)
    ha_ = a_in * a_out
    eyebrow.set_text(f"SIMULATION  /  {ph + 1:02d}  {chip}")
    title.set_text(ttl); lede.set_text(ld)
    for tx in (title, lede):
        set_a(tx, ha_)
    for i, (r, tx) in enumerate(chips):
        if i == ph:
            r.set_facecolor(WARM); tx.set_color(RUST); tx.set_fontweight("bold")
        else:
            r.set_facecolor(CARD); tx.set_color(INK if i < ph else SLATE); tx.set_fontweight("normal")

    # ---------------- people
    pos = P[fi]
    people.set_offsets(pos)
    dim = 1.0 - 0.62 * win(t, 12.0, 25.6, 0.6, 0.8)
    pc = np.tile(to_rgba(SLATE), (N_PEOPLE, 1))
    pc[:, 3] = dim
    people.set_facecolors(pc)

    # ---------------- probe pulses + links
    off, sz, ec, segs, lc = [], [], [], [], []
    for back in range(0, 27):
        f2 = fi - back
        if f2 < 0:
            break
        age = back * DT
        for i in EMITS[f2]:
            src = P[f2, i]
            a = 0.42 * (1 - age / 0.9) * (1.0 if i == 0 else dim)
            r = 0.25 + 1.1 * (age / 0.9) ** 0.7
            off.append(src); sz.append((46.5 * r) ** 2); ec.append(to_rgba(NAVY, max(a, 0)))
            if f2 * DT >= T_LISTEN and age < 0.32 and i != 0:
                la = 0.28 * (1 - age / 0.32) * dim
                for j in range(3):
                    segs.append([src, drone_pos(j, t)[0]]); lc.append(to_rgba(NAVY, la))
    pulses.set_offsets(np.array(off) if off else np.empty((0, 2)))
    pulses.set_sizes(np.array(sz) if sz else np.empty(0))
    pulses.set_edgecolors(ec if ec else "none")
    links.set_segments(segs); links.set_colors(lc if lc else [(0, 0, 0, 0)])

    # ---------------- drones
    rx_recent = any(f2 * DT >= T_LISTEN and EMITS[f2] for f2 in range(max(0, fi - 3), fi + 1))
    for j, d in enumerate(drones):
        (x, y), lab_a = drone_pos(j, t)
        d["shadow"].set_center((x + 0.55, y - 0.55))
        for k_, sgn in enumerate((1, -1)):
            d["arms"][k_].set_data([x - ARM, x + ARM], [y - ARM * sgn, y + ARM * sgn])
        tips = [(x + ARM, y + ARM), (x - ARM, y + ARM), (x - ARM, y - ARM), (x + ARM, y - ARM)]
        for k_, (rx, ry) in enumerate(tips):
            d["blur"][k_].set_center((rx, ry)); d["rot"][k_].set_center((rx, ry))
            th = t * 31.0 + k_ * 0.8
            dx, dy = 0.31 * np.cos(th), 0.31 * np.sin(th)
            d["blades"][k_].set_data([rx - dx, rx + dx], [ry - dy, ry + dy])
        d["body"].set_xy((x - 0.3, y - 0.3))
        d["led"].set_center((x, y))
        d["led"].set_facecolor(RUST if (rx_recent and (fi + j * 2) % 4 < 2) else LED_OFF)
        lab_off = [(-1.25, 0.12), (1.25, 0.12), (1.25, 0.12)][j]
        d["lab"].set_position((x + lab_off[0], y + lab_off[1]))
        d["sub"].set_position((x + lab_off[0], y + lab_off[1] - 0.15))
        set_a(d["lab"], lab_a); set_a(d["sub"], lab_a)

    fa = clamp01((t - 3.7) / 0.9)
    formation.set_alpha(0.75 * fa)
    bl = fa * (1 - clamp01((t - 11.6) / 0.6))
    for b_ in base_labs:
        set_a(b_, bl)

    # ---------------- focus walkthrough (phases 3-4)
    fw = win(t, 11.8, 25.8, 0.5, 0.8)
    ftrue = P[fi, 0]
    if fw > 0:
        focus_dot.set_offsets([ftrue])
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 1.2 * t)
        halo.set_offsets([ftrue]); halo.set_sizes([(46.5 * (0.55 + 0.25 * pulse)) ** 2])
        halo.set_alpha(fw * 0.9)
        focus_dot.set_alpha(fw)
        focus_lab.set_position((ftrue[0] + 0.55, ftrue[1] + 0.55)); set_a(focus_lab, fw)
    else:
        focus_dot.set_offsets(np.empty((0, 2))); halo.set_offsets(np.empty((0, 2)))
        set_a(focus_lab, 0)

    fsegs, fcol = [], []
    for back in range(0, 25):
        f2 = fi - back
        if f2 < 0:
            break
        if 0 in EMITS[f2] and f2 * DT >= T_LISTEN and 11.8 <= f2 * DT < 26:
            a = 0.9 * (1 - back * DT / 0.8)
            for j in range(3):
                fsegs.append([P[f2, 0], drone_pos(j, t)[0]]); fcol.append(to_rgba(NAVY, max(a, 0)))
    focus_links.set_segments(fsegs); focus_links.set_colors(fcol if fcol else [(0, 0, 0, 0)])

    out_a = 1 - clamp01((t - 21.8) / 0.7)
    for j in range(3):
        t_read = 12.7 + 0.5 * j
        ra = clamp01((t - t_read) / 0.35) * out_a
        readouts[j].set_text(f"{SIM['focus_avg'][j]:.1f} dBm → {SIM['focus_dist'][j]:.1f} m"
                             .replace("-", "−"))
        set_a(readouts[j], ra)
        t_c = 13.2 + 0.7 * j
        g = ease((t - t_c) / 1.1)
        range_circ[j].set_radius(max(0.01, SIM["focus_dist"][j] * g))
        ca = (1.0 if g > 0 else 0.0) * out_a      # colour alphas, not Patch.set_alpha (that floods the fill)
        range_circ[j].set_facecolor(to_rgba(NAVY, 0.06 * ca))
        range_circ[j].set_edgecolor(to_rgba(NAVY, 0.85 * ca))

    est = SIM["focus_est"]
    la = clamp01((t - 19.4) / 0.25) * (1 - clamp01((t - 25.4) / 0.6))
    if la > 0:
        lk = ease((t - 19.4) / 1.0)
        lock.set_offsets([est]); lock.set_sizes([(46.5 * (2.4 - 2.0 * lk)) ** 2])
        lock.set_alpha(la * (1 - lk) * 0.9 + la * 0.0)
        ea = clamp01((t - 20.1) / 0.3) * la
        est_dot.set_offsets([est]); est_dot.set_alpha(ea)
        err_line.set_data([ftrue[0], est[0]], [ftrue[1], est[1]]); err_line.set_alpha(ea)
        est_lab.set_text(f"estimate ({est[0]:.1f}, {est[1]:.1f})\nerror {FOCUS_ERR:.1f} m")
        est_lab.set_position((est[0] + 0.6, est[1] - 0.55)); set_a(est_lab, clamp01((t - 20.5) / 0.4) * la)
    else:
        lock.set_offsets(np.empty((0, 2))); est_dot.set_offsets(np.empty((0, 2)))
        err_line.set_data([], []); set_a(est_lab, 0)

    # ---------------- estimates (scene whiskers in phase 4, map from 21.5 s)
    disp, k, u = displayed_estimates(fi)
    revealed = SIM["reveal"] <= t
    if disp is not None:
        ok = revealed & ~np.isnan(disp).any(axis=1)
    else:
        ok = np.zeros(N_PEOPLE, bool)
    wa = win(t, T_SOLVE_ALL + 0.2, 25.4, 0.4, 0.8)
    if wa > 0 and ok.any():
        idx = np.nonzero(ok)[0]
        idx = idx[idx != 0]
        whisk.set_segments([[pos[i], disp[i]] for i in idx])
        whisk.set_alpha(0.32 * wa)
        crosses.set_offsets(disp[idx]); crosses.set_alpha(0.75 * wa)
    else:
        whisk.set_segments([]); crosses.set_offsets(np.empty((0, 2)))

    if ok.any():
        est_pts.set_offsets(disp[ok])
        if t >= T_MAP + 1 and k >= 1:
            tr = []
            for i in np.nonzero(ok)[0]:
                hist = [SNAPS[SNAP_FR[q]][i] for q in range(max(0, k - 2), k)] + [disp[i]]
                hist = [h for h in hist if not np.isnan(h).any()]
                if len(hist) > 1:
                    tr.append(hist)
            trails.set_segments(tr)
        else:
            trails.set_segments([])
    else:
        est_pts.set_offsets(np.empty((0, 2))); trails.set_segments([])

    if t < 4.6:
        status.set_text("Waiting for nodes to take position…"); set_a(status, 1)
    elif t < T_SOLVE_ALL:
        status.set_text("Listening — frames arriving from N1, N2, N3")
        set_a(status, 1 - clamp01((t - 21.0) / 0.4))
    else:
        set_a(status, 0)

    lv = clamp01((t - T_SOLVE_ALL) / 0.5)
    set_a(live_lab, lv)
    if k >= 0:
        since = (fi - SNAP_FR[k]) * DT
        live_dot.set_alpha(lv * (0.25 + 0.75 * clamp01(1 - since / 0.35)))
    else:
        live_dot.set_alpha(0)

    # heat + zone
    hz = clamp01((t - T_MAP) / 1.2)
    if hz > 0 and k >= 0 and SNAP_FR[k] in HEAT:
        hc = HEAT[SNAP_FR[k]]
        hp = HEAT.get(SNAP_FR[k - 1], hc) if k > 0 else hc
        hm = hp + (hc - hp) * u
        rgba = heat_cmap(np.clip(hm / HEAT_MAX, 0, 1))
        rgba[..., 3] *= hz
        heat.set_data(rgba)
    else:
        heat.set_data(np.zeros((len(GY), len(GX), 4)))

    za = clamp01((t - (T_MAP - 0.3)) / 0.8)
    ratio = None
    if k >= 0 and t >= T_MAP:
        rc = RATIO[SNAP_FR[k]]
        rp = RATIO[SNAP_FR[k - 1]] if k > 0 else rc
        ratio = rp + (rc - rp) * u
    hot = ratio is not None and ratio >= THRESH
    zc = RUST if hot else SLATE
    for z_, zl in ((sc_zone, sc_zone_lab), (mp_zone, mp_zone_lab)):
        z_.set_alpha(za); set_a(zl, za)
        z_.set_edgecolor(zc); zl.set_color(zc)
        z_.set_linewidth(2.2 if hot else 1.3)
    if hot:
        alert.set_text(f"ALERT  ·  ZONE B AT {ratio:.1f}× AVERAGE")
        set_a(alert, 1.0)
    else:
        set_a(alert, 0)

    # ---------------- tiles
    tiles[0].set_text(f"{SIM['frames_cnt'][fi]:,}")
    tiles[1].set_text(f"{int(ok.sum())}" if t >= T_SOLVE_ALL else "0")
    me = MEDERR.get(SNAP_FR[k]) if k >= 0 else None
    tiles[2].set_text(f"{me:.1f} m" if (me is not None and ok.any()) else "—")
    tiles[3].set_text(f"{ratio:.1f}×" if ratio is not None else "—")
    tiles[3].set_color(RUST if hot else NAVY)


# ================================================================ output
if MODE == "info":
    print(f"seed={SEED} focus_true={SIM['focus_true'].round(2)} focus_est={SIM['focus_est'].round(2)} "
          f"err={FOCUS_ERR:.2f} m  dists={np.round(SIM['focus_dist'], 1)} rssi={np.round(SIM['focus_avg'], 1)}")
    print("ratio by snapshot:", {round(fr / FPS): round(v, 2) for fr, v in RATIO.items()})
    print("median err by snapshot:", {round(fr / FPS): round(v, 2) for fr, v in MEDERR.items()})
    print("frames total:", SIM["frames_cnt"][-1])
elif MODE == "frames":
    outdir = sys.argv[2]
    for ts in sys.argv[3:]:
        fi = min(F - 1, int(round(float(ts) * FPS)))
        draw(fi)
        check_fonts(fig)
        path = os.path.join(outdir, f"frame_{float(ts):05.1f}.png")
        fig.savefig(path, dpi=100, facecolor=WHITE)
        print(path)
elif MODE == "video":
    import imageio_ffmpeg
    import time
    out = sys.argv[2]
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-profile:v", "high",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for fi in range(F):
        draw(fi)
        fig.canvas.draw()
        proc.stdin.write(fig.canvas.buffer_rgba())
        if fi % 150 == 0:
            check_fonts(fig)
            print(f"  frame {fi}/{F}  {time.time() - t0:.0f}s", flush=True)
    proc.stdin.close()
    proc.wait()
    print("wrote", out, f"in {time.time() - t0:.0f}s, exit", proc.returncode)
