"""
H1-2 (MuJoCo) killer demo: goal pose behind / beside the robot, planned with the IDENTIFIED asymmetric
envelope (Finsler) vs its symmetric part (Riemannian-style), executed closed-loop on the simulated humanoid.

Pipeline
  1. envelope: fit a Randers norm F_body(xi) = |L xi| + b.xi to the achieved-velocity sweep of
     identify_envelope.py (boundary = directional maxima of the achieved set among non-fallen rollouts).
  2. planner: SE(2) grid Dijkstra value function for the fitted metric (and for its symmetric part);
     feedback = greedy descent of the value, executed at 90% of the envelope speed, re-evaluated at 20 Hz.
     (The FM planner trained on this envelope replaces the grid in the next step; the geometry effect is
      identical and this version needs no training.)
  3. execution: the Unitree RL locomotion policy tracks the commanded body velocity in MuJoCo.
Tasks: goal 2.5 m behind with the same heading, behind with reversed heading, 2 m to the left, 2 m ahead.
Metrics: arrival time, success within (0.25 m, 15 deg) by 20 s, falls, and the trajectories (top view).

Usage: python sim/demo_turn_or_reverse.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from finsler.eikonal import GridSE2Distance
from finsler.se2_randers import SE2Randers
from fit_envelope import BodyFit
from h1_2_env import H12Env

torch.set_default_dtype(torch.float64)
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
C = {"finsler": "#1baf7a", "symmetric": "#eb6834"}
LABEL = {"finsler": "identified asymmetric envelope (Finsler)", "symmetric": "symmetric part only (Riemannian-style)"}
C_INK, C_INK2, C_SURF, C_GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e1e0d9"


def wrap(a):
    return np.remainder(a + np.pi, 2 * np.pi) - np.pi


# ---------------------------------------------------------------------------------------------------------
def fit_envelope_from_sweep(path, n_bins=160, seed=0):
    z = np.load(path)
    ach, fallen, scale = z["achieved"], z["fallen"], z["scale"]
    ok = ~fallen & np.isfinite(ach).all(1)
    V = ach[ok]
    # directional maxima in the scaled space -> boundary samples; everything else interior
    Vs = V / scale
    r = np.linalg.norm(Vs, axis=1)
    u = Vs / np.maximum(r, 1e-9)[:, None]
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_bins, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    bin_id = np.argmax(u @ centers.T, 1)
    boundary = np.zeros(len(V), bool)
    for k in range(n_bins):
        idx = np.where(bin_id == k)[0]
        if len(idx):
            boundary[idx[np.argmax(r[idx])]] = True
    Vb, Vi = V[boundary], V[~boundary]
    torch.manual_seed(seed)
    model = BodyFit(1, init_scale=1.5)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    xb, xi = torch.as_tensor(Vb), torch.as_tensor(Vi)
    for it in range(3000):
        loss = torch.mean((model.F(xb) - 1.0) ** 2) + torch.mean(torch.relu(model.F(xi) - 1.0) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    L, b = model.export()
    A = L[0].T @ L[0]
    return A, b, dict(n_ok=int(ok.sum()), n_boundary=int(boundary.sum()), loss=float(loss), speeds=SE2Randers(A, b).speeds())


# ---------------------------------------------------------------------------------------------------------
def make_feedback(F, goal, lim=(-3.5, 3.5), h=0.1, nth=36):
    G = GridSE2Distance(F, lim, lim, h, nth)
    D = G.dist_to_target(goal)
    offs = [(i, j, k) for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3) if (i, j, k) != (0, 0, 0) and np.gcd.reduce([abs(i), abs(j), abs(k)]) == 1]
    nx, ny, nt = G.shape

    def policy(q):
        q = np.asarray(q, float)[None]
        n = G.node_of(q)
        ix, iy, it = n // (ny * nt), (n // nt) % ny, n % nt
        best, bestv = None, np.inf
        for (i, j, k) in offs:
            jx, jy, jt = ix + i, iy + j, (it + k) % nt
            if not (0 <= jx[0] < nx and 0 <= jy[0] < ny):
                continue
            nb = (jx * ny + jy) * nt + jt
            dq = np.array([i * h, j * h, k * G.hth])
            c = float(F.F(q + 0.5 * dq, dq[None])[0]) + D[nb[0]]
            if c < bestv:
                bestv, best = c, dq
        xi = SE2Randers.body_velocity(q[0], best)
        xi = xi / max(float(F.F_body(xi)), 1e-9)                          # full envelope speed along the chosen direction
        return xi, D[n[0]]
    return policy, D, G


def run_task(env, policy, start, goal, speed_frac=0.9, T_max=20.0, pos_tol=0.25, head_tol=np.radians(15), rate=20.0):
    env.reset(start)
    dt_cmd = 1.0 / rate
    hold = {"cmd": np.zeros(3), "next": 0.0}

    def cmd_fn(t, p, v):
        if t >= hold["next"]:
            xi, _ = policy(p)
            hold["cmd"] = speed_frac * xi
            hold["next"] = t + dt_cmd
        return hold["cmd"]

    n = int(round(T_max / env.ctrl_dt))
    T, P, Cm = [], [], []
    t_arr = np.nan
    for k in range(n):
        t = k * env.ctrl_dt
        p = env.pose()
        c = cmd_fn(t, p, None)
        T.append(t)
        P.append(p)
        Cm.append(c.copy())
        if np.linalg.norm(p[:2] - goal[:2]) < pos_tol and abs(wrap(p[2] - goal[2])) < head_tol:
            t_arr = t
            break
        if env.step(c):
            break
    return dict(t=np.array(T), pose=np.array(P), cmd=np.array(Cm), fallen=env.fallen, t_arrive=t_arr)


def draw_pose(ax, q, color, size=0.16, alpha=1.0):
    x, y, th = q
    tri = np.array([[size, 0], [-0.6 * size, 0.45 * size], [-0.6 * size, -0.45 * size]])
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    ax.add_patch(plt.Polygon(tri @ Rm.T + np.array([x, y]), closed=True, facecolor=color, edgecolor=C_INK, lw=0.6, alpha=alpha))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix of the envelope sweep file (e.g. _limited)")
    args = ap.parse_args()
    sweep = OUT / f"h1_2_envelope{args.tag}.npz"
    A, b, info = fit_envelope_from_sweep(sweep)
    lim = np.load(sweep)["limits"]
    cmd_limits = None if np.all(lim == 0) else lim
    print("interface limits applied to the executing robot:", cmd_limits.tolist() if cmd_limits is not None else None)
    print("fitted envelope from sweep:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items() if k != "speeds"}), "speeds", {k: round(v, 2) for k, v in info["speeds"].items()})
    metrics = {"finsler": SE2Randers(A, b), "symmetric": SE2Randers(A, np.zeros(3))}
    print("symmetric part speeds:", {k: round(v, 2) for k, v in metrics["symmetric"].speeds().items()})
    tasks = {
        "behind, same heading": (np.zeros(3), np.array([-2.5, 0.0, 0.0])),
        "behind, reversed heading": (np.zeros(3), np.array([-2.5, 0.0, np.pi])),
        "left, same heading": (np.zeros(3), np.array([0.0, 2.0, 0.0])),
        "ahead, same heading": (np.zeros(3), np.array([2.5, 0.0, 0.0])),
    }
    env = H12Env(cmd_limits=cmd_limits)
    rows, trajs = [], {}
    for tname, (start, goal) in tasks.items():
        for mname, F in metrics.items():
            policy, D, G = make_feedback(F, goal)
            t0 = time.time()
            r = run_task(env, policy, start, goal)
            trajs[(tname, mname)] = r
            opt_time = float(D[G.node_of(start[None])][0])
            rows.append(dict(task=tname, metric=mname, arrived=not np.isnan(r["t_arrive"]), t_arrive=r["t_arrive"], fallen=r["fallen"], plan_time=opt_time))
            print(f"  {tname:26s} {mname:9s}: arrived {not np.isnan(r['t_arrive'])}, t = {r['t_arrive']:.2f} s, planned {opt_time:.2f} s, fallen {r['fallen']}  ({time.time()-t0:.0f}s)", flush=True)
    lines = ["# H1-2 MuJoCo demo: goal behind / beside, identified envelope vs symmetric part", "",
             f"identified envelope speeds (m/s, rad/s): {{{', '.join(f'{k}: {v:.2f}' for k, v in info['speeds'].items())}}}; symmetric part: {{{', '.join(f'{k}: {v:.2f}' for k, v in metrics['symmetric'].speeds().items())}}}", "",
             "| task | planner envelope | arrived | arrival time [s] | planner's own estimate [s] | fell |", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['metric']} | {r['arrived']} | {r['t_arrive']:.2f} | {r['plan_time']:.2f} | {r['fallen']} |")
    (OUT / f"demo_turn_or_reverse{args.tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    # ---- figure: top views ------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.4 * len(tasks), 4.6), facecolor=C_SURF)
    for ax, (tname, (start, goal)) in zip(axes, tasks.items()):
        ax.set_facecolor(C_SURF)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(colors=C_INK2, labelsize=8)
        ax.grid(True, color=C_GRID, lw=0.6)
        ax.set_axisbelow(True)
        for mname in metrics:
            r = trajs[(tname, mname)]
            P = r["pose"]
            ax.plot(P[:, 0], P[:, 1], color=C[mname], lw=1.8, label=f"{LABEL[mname]}: {r['t_arrive']:.1f} s" if not np.isnan(r["t_arrive"]) else f"{LABEL[mname]}: fail")
            for k in np.linspace(0, len(P) - 1, 8).astype(int):
                draw_pose(ax, P[k], C[mname], alpha=0.35)
        draw_pose(ax, start, "#ffffff", size=0.2)
        draw_pose(ax, goal, C_INK, size=0.22)
        ax.set_aspect("equal")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.0, 3.0)
        ax.set_title(tname, fontsize=9.5, color=C_INK, loc="left")
        ax.legend(frameon=False, fontsize=7, loc="lower left", labelcolor=C_INK2)
    fig.suptitle("Simulated H1-2 (Unitree RL policy): closed-loop execution of the identified-envelope planner vs its symmetric part", fontsize=10, color=C_INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / f"demo_turn_or_reverse{args.tag}.png", dpi=160)
    plt.close(fig)
    np.savez(OUT / f"demo_turn_or_reverse{args.tag}_trajs.npz", **{f"{t}__{m}": v["pose"] for (t, m), v in trajs.items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
