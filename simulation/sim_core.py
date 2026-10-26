# -*- coding: utf-8 -*-
"""Shared simulation model. Uses the project's own backend solver and config."""
import json
import os
import sys

import numpy as np

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT, "backend"))
from trilateration import rssi_to_distance, trilaterate_ls  # noqa: E402  (the real solver)

with open(os.path.join(PROJECT, "config", "node_positions.json")) as f:
    _cfg = json.load(f)
NODE_IDS = list(_cfg.keys())                      # same ordering utils.py uses
NODES = np.array([_cfg[k] for k in NODE_IDS], dtype=float)

TX_POWER = -40.0      # rssi_to_distance default in backend/trilateration.py
N_EXP = 2.0           # rssi_to_distance default
SIGMA_DB = 4.0        # per-frame log-normal shadowing, simulated
WINDOW = 5            # RSSI_WINDOW in backend/utils.py
RSSI_FLOOR = -85      # firmware drops frames below this

# world shown in the simulation (metres)
XMIN, XMAX, YMIN, YMAX = -4.0, 24.0, -3.0, 18.0


def true_rssi(p, n_true=N_EXP, tx_true=TX_POWER):
    """Mean RSSI at each node for a device at p (log-distance model)."""
    d = np.maximum(np.hypot(NODES[:, 0] - p[0], NODES[:, 1] - p[1]), 0.1)
    return tx_true - 10.0 * n_true * np.log10(d)


def frame_rssi(p, rng, n_true=N_EXP, tx_true=TX_POWER):
    """One captured frame per node: mean + shadowing noise, rounded like the radio reports."""
    return np.round(true_rssi(p, n_true, tx_true) + rng.normal(0.0, SIGMA_DB, len(NODES)))


def solve_from_avg(avg_rssi):
    """Exactly the backend path: averaged RSSI -> rssi_to_distance -> trilaterate_ls."""
    dists = [rssi_to_distance(r) for r in avg_rssi]
    return trilaterate_ls(NODES.tolist(), dists)


def inside_triangle(x, y):
    (x1, y1), (x2, y2), (x3, y3) = NODES[:3]
    def s(ax, ay, bx, by, cx, cy):
        return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1 = s(x, y, x1, y1, x2, y2)
    d2 = s(x, y, x2, y2, x3, y3)
    d3 = s(x, y, x3, y3, x1, y1)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)
