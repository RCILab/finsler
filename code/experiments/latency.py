"""
Query latency of the amortised planner vs classical solvers (CPU, this machine).

  FM planner      : one RK4 integration of the learned field (n_steps x 4 MLP evaluations), batch 1 and 100.
  FM feedback     : one MLP evaluation (t = 0 field) per control step.
  Grid Dijkstra   : one reverse Dijkstra per new goal (all starts at once), plus graph build once per map.
  Shooting BVP    : one 3-branch continuation solve per query (what produced the training data).
Planar (beta 0.9 planner) and SE(2) (pose planner).  Numbers are wall-clock medians of repeated runs.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from finsler.eikonal import GridFinslerDistance, GridSE2Distance
from fm_path_benefit import MLP
import fm_planner as fp
import fm_planner_se2 as fs

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "latency"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)


def timeit(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    lines = ["# Query latency (CPU, wall-clock medians)", "", "| space | method | batch | per-query latency | notes |", "|---|---|---:|---:|---|"]
    # ---- planar ----------------------------------------------------------------------------------
    cfg = dict(fp.CFG)
    mkF, mkR, E, W1 = fp.make_fields(0.9, cfg["width"])
    F = mkF(1.0)
    m = MLP(d_in=5, d_out=2, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
    m.load_state_dict(torch.load(ROOT / "results" / "fm_planner_beta0.9" / "model_finsler.pt"))
    m.eval()
    goal = np.asarray(cfg["goals"][0], float)
    rng = np.random.default_rng(0)
    starts1 = rng.uniform([-3.5, -1.8], [-1.2, 1.8], (1, 2))
    starts100 = rng.uniform([-3.5, -1.8], [-1.2, 1.8], (100, 2))
    for steps in [200, 50]:
        t1 = timeit(lambda: fp.integrate(m, starts1, steps, goal=goal))
        t100 = timeit(lambda: fp.integrate(m, starts100, steps, goal=goal))
        lines.append(f"| plane | FM plan (RK4 {steps} steps) | 1 | {t1*1e3:.1f} ms | full path, {4*steps} MLP evals |")
        lines.append(f"| plane | FM plan (RK4 {steps} steps) | 100 | {t100/100*1e3:.2f} ms | batched, {t100*1e3:.0f} ms total |")
    with torch.no_grad():
        inp = torch.tensor(np.concatenate([starts100, np.zeros((100, 1)), np.repeat(goal[None], 100, 0)], 1), dtype=torch.float32)
        tfb = timeit(lambda: m(inp))
    lines.append(f"| plane | FM feedback (t = 0 field) | 100 | {tfb/100*1e6:.0f} µs | one MLP evaluation per control step |")
    tb = timeit(lambda: GridFinslerDistance(F, cfg["grid_xlim"], cfg["grid_ylim"], cfg["grid_h"]), reps=2)
    G = GridFinslerDistance(F, cfg["grid_xlim"], cfg["grid_ylim"], cfg["grid_h"])
    tg = timeit(lambda: G.dist_to_target(goal), reps=5)
    lines.append(f"| plane | grid Dijkstra (h = {cfg['grid_h']}, 32 dirs) | all starts | {tg*1e3:.0f} ms per new goal | + {tb:.2f} s graph build per map, {G.n_edges/1e6:.1f} M edges |")
    x0 = starts1
    x1 = goal[None] + 0.0
    tbvp = timeit(lambda: fp.solve_bvps(mkF, x0, x1, cfg["log_steps"]), reps=2)
    lines.append(f"| plane | shooting BVP (3 branches, continuation) | 1 | {tbvp:.1f} s | exact geodesic; may land on a non-minimal branch |")
    tbvp100 = timeit(lambda: fp.solve_bvps(mkF, starts100, np.repeat(goal[None], 100, 0), cfg["log_steps"]), reps=1)
    lines.append(f"| plane | shooting BVP (3 branches, continuation) | 100 | {tbvp100/100:.2f} s | batched; {tbvp100:.0f} s total |")
    # ---- SE(2) ----------------------------------------------------------------------------------------
    cfg2 = dict(fs.CFG)
    F2, R2, mkF2 = fs.make_metrics(cfg2)
    m2 = MLP(d_in=5, d_out=3, hidden=cfg2["hidden"], n_layers=cfg2["n_layers"])
    m2.load_state_dict(torch.load(ROOT / "results" / "fm_planner_se2" / "model_finsler.pt"))
    m2.eval()
    q1 = rng.uniform([-2, -2, -np.pi], [2, 2, np.pi], (1, 3))
    q100 = rng.uniform([-2, -2, -np.pi], [2, 2, np.pi], (100, 3))
    for steps in [200, 50]:
        t1 = timeit(lambda: fs.integrate(m2, q1, steps))
        t100 = timeit(lambda: fs.integrate(m2, q100, steps))
        lines.append(f"| SE(2) | FM plan (RK4 {steps} steps) | 1 | {t1*1e3:.1f} ms | full pose path |")
        lines.append(f"| SE(2) | FM plan (RK4 {steps} steps) | 100 | {t100/100*1e3:.2f} ms | batched |")
    tb2 = timeit(lambda: GridSE2Distance(F2, cfg2["grid_lim"], cfg2["grid_lim"], cfg2["grid_h"], cfg2["grid_ntheta"]), reps=1)
    G2 = GridSE2Distance(F2, cfg2["grid_lim"], cfg2["grid_lim"], cfg2["grid_h"], cfg2["grid_ntheta"])
    tg2 = timeit(lambda: G2.dist_to_target(np.zeros(3)), reps=3)
    lines.append(f"| SE(2) | grid Dijkstra (h = 0.1, 10°, 98 dirs) | all starts | {tg2*1e3:.0f} ms per new goal | + {tb2:.1f} s graph build, {G2.n_edges/1e6:.1f} M edges, 3% overestimate |")
    goal3 = np.zeros((1, 3))
    tbvp2 = timeit(lambda: fs.solve_bvps(mkF2, q1.copy(), goal3.copy(), cfg2["log_steps"], use_continuation=True), reps=2)
    lines.append(f"| SE(2) | shooting BVP (3 wraps, continuation) | 1 | {tbvp2:.1f} s | exact geodesic; ~1/3 of solves land on non-minimal branches |")
    lines += ["", "Amortisation: the flow model answers a new start-goal query in milliseconds and batches trivially; the solvers that produced its training data cost seconds per query (BVP) or a per-map graph build plus a per-goal sweep (Dijkstra) whose memory grows as h^-d."]
    (OUT / "latency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
