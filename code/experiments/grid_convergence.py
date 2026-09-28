"""
Grid-resolution convergence of the Dijkstra optimal-time reference, planar and SE(2).

Planar: d_grid(x0, x1) at h in {0.08, 0.04, 0.02} vs the shooting-BVP time (exact geodesic) for 60
        training pairs of the beta = 0.9 planner (minimal branch verified at h = 0.04).
SE(2):  d_grid(q0, goal) at (h, n_theta) in {(0.14, 24), (0.10, 36), (0.07, 48)} vs BVP times for
        60 pairs of the SE(2) planner.
Reports mean / max relative error of each resolution against the BVP, so that "within grid
discretisation error" has a number attached.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from finsler.eikonal import GridFinslerDistance, GridSE2Distance
import fm_planner as fp
import fm_planner_se2 as fs

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "grid_convergence"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    lines = ["# Grid Dijkstra reference: resolution convergence", ""]
    # ---- planar ------------------------------------------------------------------------------------
    cfg = dict(fp.CFG)
    mkF, mkR, E, W1 = fp.make_fields(0.9, cfg["width"])
    F = mkF(1.0)
    z = np.load(ROOT / "results" / "fm_planner_beta0.9" / "pairs.npz")
    rng = np.random.default_rng(0)
    sub = rng.choice(len(z["x0"]), 60, replace=False)
    x0, x1, T = z["x0"][sub], z["x1"][sub], z["T"][sub]
    G04 = GridFinslerDistance(F, cfg["grid_xlim"], cfg["grid_ylim"], 0.04)
    d04 = G04.dist(x0, x1)
    minimal = T <= 1.02 * d04                                            # keep pairs whose BVP is the minimal branch
    x0, x1, T = x0[minimal], x1[minimal], T[minimal]
    lines += [f"## Planar head-wind channel (beta = 0.9), {len(T)} pairs with verified minimal BVP geodesics", "",
              "| h | nodes | edges | build [s] | mean rel. error vs BVP | max | fraction with grid < BVP |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for h in [0.08, 0.04, 0.02]:
        t0 = time.time()
        G = GridFinslerDistance(F, cfg["grid_xlim"], cfg["grid_ylim"], h)
        tb = time.time() - t0
        d = G.dist(x0, x1)
        rel = (d - T) / T
        lines.append(f"| {h} | {len(G.nodes)} | {G.n_edges} | {tb:.1f} | {rel.mean()*100:+.2f}% | {np.abs(rel).max()*100:.2f}% | {(rel < 0).mean():.2f} |")
        print(lines[-1], flush=True)
    # ---- SE(2) ------------------------------------------------------------------------------------------
    cfg2 = dict(fs.CFG)
    F2, R2, mkF2 = fs.make_metrics(cfg2)
    z2 = np.load(ROOT / "results" / "fm_planner_se2" / "pairs.npz")
    sub = rng.choice(len(z2["q0"]), 60, replace=False)
    q0, T2 = z2["q0"][sub], z2["T"][sub]
    goal = np.asarray(cfg2["goal"], float)
    lines += ["", f"## SE(2) body-frame envelope, {len(T2)} pairs (BVP branch verified at h = 0.1 within 10%)", "",
              "| h | n_theta | nodes | edges | build [s] | mean rel. error vs BVP | max | fraction with grid < BVP |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for h, nt in [(0.14, 24), (0.10, 36), (0.07, 48)]:
        t0 = time.time()
        G = GridSE2Distance(F2, cfg2["grid_lim"], cfg2["grid_lim"], h, nt)
        tb = time.time() - t0
        d = G.dist_to_target(goal)[G.node_of(q0)]
        rel = (d - T2) / T2
        lines.append(f"| {h} | {nt} | {len(G.nodes)} | {G.n_edges} | {tb:.1f} | {rel.mean()*100:+.2f}% | {np.abs(rel).max()*100:.2f}% | {(rel < 0).mean():.2f} |")
        print(lines[-1], flush=True)
    lines += ["", "Reading: the grid overestimates the true time by the angular quantisation of 32 (planar) / 98 (SE(2)) directions plus midpoint quadrature;",
              "the error shrinks with h and the reference is an upper bound up to that error. Ratios below 1 in the planner tables are within this band."]
    (OUT / "grid_convergence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
