"""
Flow matching as a time-optimal pose planner on SE(2) with a body-frame velocity envelope.

Robot: holonomic base with asymmetric achievable body velocities (quadruped-like): forward 1.2 m/s,
backward 0.5, sideways 0.4, yaw 1.2 rad/s -> a left-invariant Randers metric on SE(2).
Task: reach the pose g = (0, 0, 0) from a start pose q0 (left-invariance makes one goal general).
Conditional paths from q0 to g:
    Euclidean   straight interpolation in (x, y, theta), heading difference wrapped    (OT-CFM)
    Riemannian  geodesics of the symmetric ellipsoid (b = 0: forward = backward = 0.71 m/s)   (RFMP-style)
    Finsler     minimal Finsler geodesics (multi-guess BVP + continuation in the asymmetry b)  (ours)
The learned field v_theta(t, q) is integrated from held-out poses; the path is executed at full speed,
so its Finsler length is its travel time.  Optimal times from SE(2) grid Dijkstra (h = 0.1, 10 deg).

Stages cached in results/fm_planner_se2/ : pairs -> riemann -> train -> eval -> fig
Usage: python experiments/fm_planner_se2.py [--stage all|pairs,riemann,train,eval,fig]
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from finsler.eikonal import GridSE2Distance
from finsler.se2_randers import SE2Randers
from fm_path_benefit import MLP

C = {"euclid": "#2a78d6", "riemann": "#eb6834", "finsler": "#1baf7a"}
LABEL = {"euclid": "Euclidean interpolation (OT-CFM)", "riemann": "symmetric-ellipsoid geodesics (RFMP-style)", "finsler": "Finsler geodesics (ours)"}
C_MUTED, C_GRID, C_INK, C_INK2, C_SURF = "#898781", "#e1e0d9", "#0b0b0b", "#52514e", "#fcfcfb"
METHODS = ["euclid", "riemann", "finsler"]

CFG = dict(
    speeds=dict(v_fwd=1.2, v_back=0.5, v_side=0.4, w_max=1.2), goal=[0.0, 0.0, 0.0],
    start_box=[[-2.5, 2.5], [-2.5, 2.5]], n_pairs=3000, K=48, T_max=12.0, log_steps=120, branch_tol=1.10,
    hidden=256, n_layers=4, train_steps=8000, batch=2048, lr=1e-3,
    eval_pos=5, eval_head=8, eval_extent=2.0, eval_steps=200, pos_tol=0.2, head_tol_deg=15.0,
    grid_lim=[-3.2, 3.2], grid_h=0.1, grid_ntheta=36,
)


def wrap(a):
    return np.remainder(a + np.pi, 2 * np.pi) - np.pi


def make_metrics(cfg):
    F = SE2Randers.from_speeds(**cfg["speeds"])
    R = SE2Randers(F.A, np.zeros(3))
    mkF = lambda s=1.0: SE2Randers(F.A, s * F.bvec)          # continuation in the asymmetry
    return F, R, mkF


# --------------------------------------------------------------------------------------------------
def solve_bvps(mk, q0, q1, n_steps, use_continuation):
    """Minimal geodesic q0 -> q1: several initial guesses (heading wraps), optional continuation in b."""
    F = mk(1.0)
    n = len(q0)
    dq = q1 - q0
    dq[:, 2] = wrap(dq[:, 2])
    cands = []
    for k in (-1, 0, 1):
        w0 = dq.copy()
        w0[:, 2] += 2 * np.pi * k
        if use_continuation:
            w, err = w0, None
            for s_ in np.linspace(0.25, 1.0, 4):
                w, err = mk(s_).log_batched(q0, q1, n_steps=n_steps, iters=10, w_init=w)
        else:
            w, err = F.log_batched(q0, q1, n_steps=n_steps, iters=15, w_init=w0)
        cands.append((w, err))
    times = np.stack([np.where(e < 1e-6, F.F(q0, w), np.inf) for w, e in cands])
    best = np.argmin(times, 0)
    w = np.stack([cands[b][0][i] for i, b in enumerate(best)])
    T = times[best, np.arange(n)]
    return w, T, np.isfinite(T), best


def gen_pairs(F, mkF, G, cfg, rng):
    n, K = cfg["n_pairs"], cfg["K"]
    (xlo, xhi), (ylo, yhi) = cfg["start_box"]
    q0 = np.stack([rng.uniform(xlo, xhi, n), rng.uniform(ylo, yhi, n), rng.uniform(-np.pi, np.pi, n)], 1)
    q1 = np.repeat(np.asarray(cfg["goal"], float)[None], n, 0)
    t0 = time.time()
    w, T, ok, branch = solve_bvps(mkF, q0, q1, cfg["log_steps"], use_continuation=True)
    d_grid = G.dist_to_target(np.asarray(cfg["goal"], float))[G.node_of(q0)]
    minimal = T <= cfg["branch_tol"] * d_grid
    keep = ok & minimal & (T <= cfg["T_max"])
    print(f"finsler BVPs: converged {ok.sum()}/{n}, minimal (T <= {cfg['branch_tol']} grid) {(ok & minimal).sum()}, kept {keep.sum()}; "
          f"wraps -1/0/+1 = {(branch[keep]==0).sum()}/{(branch[keep]==1).sum()}/{(branch[keep]==2).sum()}; T in [{T[keep].min():.2f}, {T[keep].max():.2f}] s; "
          f"grid/BVP median {np.median(d_grid[keep]/T[keep]):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    q0, w, T = q0[keep], w[keep], T[keep]
    _, qs, ws = F.geodesic(q0, w, 1.0, K - 1)
    XF, UF = qs.transpose(1, 0, 2), ws.transpose(1, 0, 2)
    speed = F.F(XF.reshape(-1, 3), UF.reshape(-1, 3)).reshape(len(q0), K)
    print(f"pairs: {len(q0)}; F(q_t, u_t)/T in [{(speed/T[:,None]).min():.4f}, {(speed/T[:,None]).max():.4f}] (should be 1)")
    return dict(q0=q0, q1=np.repeat(np.asarray(cfg["goal"], float)[None], len(q0), 0), T=T, t=np.linspace(0, 1, K), XF=XF, UF=UF)


def riemann_paths(R, GR, pairs, cfg):
    q0, q1 = pairs["q0"], pairs["q1"]
    t0 = time.time()
    w, T, ok, branch = solve_bvps(lambda s=1.0: R, q0, q1, cfg["log_steps"], use_continuation=False)
    d_grid = GR.dist_to_target(np.asarray(cfg["goal"], float))[GR.node_of(q0)]
    minimal = T <= cfg["branch_tol"] * d_grid
    keep = ok & minimal
    print(f"riemannian BVPs: converged {ok.sum()}/{len(ok)}, minimal {keep.sum()}  ({time.time()-t0:.0f}s)", flush=True)
    _, qs, ws = R.geodesic(q0[keep], w[keep], 1.0, cfg["K"] - 1)
    return dict(ok=keep, XR=qs.transpose(1, 0, 2), UR=ws.transpose(1, 0, 2))


def features(q, t):
    """network input: (x, y, cos th, sin th, t)"""
    return np.concatenate([q[:, :2], np.cos(q[:, 2:3]), np.sin(q[:, 2:3]), t[:, None]], 1)


def build_dataset(pairs, rie, method, cfg):
    ok = rie["ok"]
    t = pairs["t"]
    K = len(t)
    if method == "finsler":
        X, U = pairs["XF"][ok], pairs["UF"][ok]
    elif method == "riemann":
        X, U = rie["XR"], rie["UR"]
    else:
        q0, q1 = pairs["q0"][ok], pairs["q1"][ok]
        dq = q1 - q0
        dq[:, 2] = wrap(dq[:, 2])
        X = q0[:, None, :] + t[None, :, None] * dq[:, None, :]
        U = np.repeat(dq[:, None, :], K, 1)
    n = len(X)
    return features(X.reshape(-1, 3), np.tile(t, n)), U.reshape(-1, 3)


def train_model(X, Y, cfg, seed):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    model = MLP(d_in=X.shape[1], d_out=3, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["train_steps"], eta_min=cfg["lr"] * 1e-2)
    g = torch.Generator().manual_seed(seed)
    for _ in range(cfg["train_steps"]):
        b = torch.randint(0, len(Xt), (cfg["batch"],), generator=g)
        loss = torch.mean((model(Xt[b]) - Yt[b]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        final = torch.mean((model(Xt) - Yt) ** 2).item()
    return model, final


def integrate(model, starts, n_steps):
    q = starts.copy()
    path = [q.copy()]
    dt = 1.0 / n_steps

    def f(t, q):
        with torch.no_grad():
            return model(torch.tensor(features(q, np.full(len(q), t)), dtype=torch.float32)).numpy().astype(float)

    for s in range(n_steps):
        t = s * dt
        k1 = f(t, q)
        k2 = f(t + dt / 2, q + dt / 2 * k1)
        k3 = f(t + dt / 2, q + dt / 2 * k2)
        k4 = f(t + dt, q + dt * k3)
        q = q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        path.append(q.copy())
    return np.array(path)


def travel_time(F, path):
    d = path[1:] - path[:-1]
    return F.F(path[:-1].reshape(-1, 3), d.reshape(-1, 3)).reshape(d.shape[0], d.shape[1]).sum(0)


def eval_starts(cfg):
    e = cfg["eval_extent"]
    xs = np.linspace(-e, e, cfg["eval_pos"])
    ths = np.linspace(-np.pi, np.pi, cfg["eval_head"], endpoint=False) + np.pi / cfg["eval_head"]
    X, Y, TH = np.meshgrid(xs, xs, ths, indexing="ij")
    starts = np.stack([X.ravel(), Y.ravel(), TH.ravel()], -1)
    return starts[np.linalg.norm(starts[:, :2], axis=1) > 0.3]            # drop starts sitting on the goal


def evaluate(models, F, G, cfg, out):
    goal = np.asarray(cfg["goal"], float)
    starts = eval_starts(cfg)
    D = G.dist_from(starts)
    d_goal = G.dist_to_target(goal)[G.node_of(starts)]
    rows, paths = [], {}
    for name in METHODS:
        path = integrate(models[name], starts, cfg["eval_steps"])
        paths[name] = path
        end = path[-1]
        pos_err = np.linalg.norm(end[:, :2] - goal[:2], axis=1)
        head_err = np.abs(wrap(end[:, 2] - goal[2]))
        succ = (pos_err < cfg["pos_tol"]) & (head_err < np.radians(cfg["head_tol_deg"]))
        L = travel_time(F, path)
        d_end = D[np.arange(len(starts)), G.node_of(end)]
        # heading of the start relative to the direction to the goal: 0 = goal ahead, pi = goal behind
        bearing = np.arctan2(goal[1] - starts[:, 1], goal[0] - starts[:, 0])
        rel = np.abs(wrap(bearing - starts[:, 2]))
        for j in range(len(starts)):
            rows.append(dict(method=name, sx=starts[j, 0], sy=starts[j, 1], sth=starts[j, 2], rel_bearing=rel[j], pos_err=pos_err[j], head_err_deg=np.degrees(head_err[j]),
                             success=int(succ[j]), travel=L[j], optimal_end=d_end[j], optimal_goal=d_goal[j], ratio=L[j] / d_end[j]))
        print(f"  eval {name}: success {succ.mean():.2f} (pos<{cfg['pos_tol']} {np.mean(pos_err < cfg['pos_tol']):.2f}), travel {L.mean():.2f} s, optimal to goal {d_goal.mean():.2f} s, "
              f"ratio {np.mean(L / d_end):.3f} (median {np.median(L / d_end):.3f})", flush=True)
    with open(out / "fm_planner_se2_eval.csv", "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)
    np.savez(out / "fm_planner_se2_paths.npz", starts=starts, **{f"path_{k}": v for k, v in paths.items()})
    return rows, starts, paths


def summarise(rows, cfg, train_loss, out):
    bins = [(0, 45, "goal ahead (|rel| < 45°)"), (45, 135, "goal to the side (45–135°)"), (135, 181, "goal behind (> 135°)")]
    lines = ["# FM as a time-optimal pose planner on SE(2)", "", f"config: `{json.dumps(cfg)}`", "", f"final train MSE: {train_loss}", "",
             "Envelope: forward 1.2, backward 0.5, sideways 0.4 m/s, yaw 1.2 rad/s.  Optimal times from SE(2) grid Dijkstra (h = 0.1 m, 10°; a few % tolerance).",
             "Success = final position error < %.2f m and heading error < %.0f°." % (cfg["pos_tol"], cfg["head_tol_deg"]), "",
             "| method | start class | n | success | mean travel [s] | mean optimal to goal [s] | mean ratio ± s.e. | median ratio | worst ratio |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    summ = {}
    for name in METHODS:
        for lo, hi, lab in bins + [(0, 181, "all")]:
            sel = [r for r in rows if r["method"] == name and lo <= np.degrees(r["rel_bearing"]) < hi]
            ratio = np.array([r["ratio"] for r in sel])
            v = dict(n=len(sel), success=np.mean([r["success"] for r in sel]), travel=np.mean([r["travel"] for r in sel]), opt=np.mean([r["optimal_goal"] for r in sel]),
                     ratio=ratio.mean(), se=ratio.std(ddof=1) / np.sqrt(len(ratio)), med=np.median(ratio), mx=ratio.max())
            summ[(name, lab)] = v
            lines.append(f"| {name} | {lab} | {v['n']} | {v['success']:.2f} | {v['travel']:.2f} | {v['opt']:.2f} | {v['ratio']:.3f} ± {v['se']:.3f} | {v['med']:.3f} | {v['mx']:.2f} |")
    (out / "fm_planner_se2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return summ


# --------------------------------------------------------------------------------------------------
def style(ax):
    ax.set_facecolor(C_SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=C_INK2, labelsize=9)


def draw_pose(ax, q, color, size=0.18, alpha=1.0, lw=1.0):
    x, y, th = q
    tri = np.array([[size, 0], [-0.6 * size, 0.45 * size], [-0.6 * size, -0.45 * size]])
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = tri @ Rm.T + np.array([x, y])
    ax.add_patch(plt.Polygon(pts, closed=True, facecolor=color, edgecolor=C_INK, lw=lw, alpha=alpha))


def fig_paths(paths, starts, rows, cfg, out):
    goal = np.asarray(cfg["goal"], float)
    rng = np.random.default_rng(1)
    # pick 6 starts with varied relative bearings
    rel = np.array([r["rel_bearing"] for r in rows if r["method"] == "finsler"])
    order = np.argsort(rel)
    picks = order[np.linspace(0, len(order) - 1, 6).astype(int)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), facecolor=C_SURF)
    for ax, name in zip(axes, METHODS):
        style(ax)
        ax.set_aspect("equal")
        p = paths[name]
        for j in picks:
            ax.plot(p[:, j, 0], p[:, j, 1], color=C[name], lw=1.6)
            for k in np.linspace(0, len(p) - 1, 7).astype(int):
                draw_pose(ax, p[k, j], C[name], alpha=0.35 if 0 < k < len(p) - 1 else 0.9, lw=0.6)
            draw_pose(ax, starts[j], "#ffffff", size=0.2)
        draw_pose(ax, goal, C_INK, size=0.24)
        sel = [r for r in rows if r["method"] == name]
        ax.set_title(f"{LABEL[name]}\nsuccess {np.mean([r['success'] for r in sel]):.2f}, mean travel/optimal {np.mean([r['ratio'] for r in sel]):.2f}", fontsize=9.5, color=C_INK, loc="left")
        ax.set_xlim(-2.8, 2.8)
        ax.set_ylim(-2.8, 2.8)
    fig.suptitle("Generated pose plans to the goal pose (black) for six held-out starts (white). Triangles show heading; forward 1.2 m/s, backward 0.5, sideways 0.4.", fontsize=10, color=C_INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fm_planner_se2_paths.png", dpi=160)
    plt.close(fig)


def fig_bearing(rows, cfg, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), facecolor=C_SURF)
    edges = np.linspace(0, 180, 7)
    for ax, key, title in [(axes[0], "ratio", "travel time / optimal time"), (axes[1], "success", "success rate")]:
        style(ax)
        ax.grid(True, axis="y", color=C_GRID, lw=0.8)
        ax.set_axisbelow(True)
        for name in METHODS:
            sel = [r for r in rows if r["method"] == name]
            rb = np.degrees([r["rel_bearing"] for r in sel])
            val = np.array([r[key] for r in sel])
            mu, se, xc = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (rb >= lo) & (rb < hi)
                if m.sum() > 1:
                    mu.append(val[m].mean())
                    se.append(val[m].std(ddof=1) / np.sqrt(m.sum()))
                    xc.append(0.5 * (lo + hi))
            mu, se, xc = np.array(mu), np.array(se), np.array(xc)
            ax.plot(xc, mu, color=C[name], lw=2, marker="o", ms=6, label=LABEL[name])
            ax.fill_between(xc, mu - se, mu + se, color=C[name], alpha=0.12, lw=0)
        ax.set_xlabel("start heading relative to the goal direction [deg]  (0 = goal ahead, 180 = goal behind)", color=C_INK2, fontsize=8.5)
        ax.set_title(title, fontsize=10, color=C_INK, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=C_INK2)
    fig.suptitle("SE(2) pose planner: optimality and success by relative bearing of the goal", fontsize=10, color=C_INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fm_planner_se2_bearing.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    cfg = dict(CFG)
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "fm_planner_se2"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    F, R, mkF = make_metrics(cfg)
    print("envelope speeds:", {k: round(v, 3) for k, v in F.speeds().items()}, " symmetric baseline:", {k: round(v, 3) for k, v in R.speeds().items()})
    t0 = time.time()
    G = GridSE2Distance(F, cfg["grid_lim"], cfg["grid_lim"], cfg["grid_h"], cfg["grid_ntheta"])
    print(f"SE(2) grid {G.shape}, {G.n_edges} edges ({time.time()-t0:.0f}s)", flush=True)
    ALL = ["pairs", "riemann", "train", "eval", "fig"]
    req = ALL if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    last = max(ALL.index(s) for s in req)

    if "pairs" in req:
        pairs = gen_pairs(F, mkF, G, cfg, rng)
        np.savez(out / "pairs.npz", **pairs)
    else:
        pairs = dict(np.load(out / "pairs.npz"))
    if last == 0:
        return
    if "riemann" in req:
        GR = GridSE2Distance(R, cfg["grid_lim"], cfg["grid_lim"], cfg["grid_h"], cfg["grid_ntheta"])
        rie = riemann_paths(R, GR, pairs, cfg)
        np.savez(out / "riemann.npz", **rie)
    else:
        rie = dict(np.load(out / "riemann.npz"))
    if last == 1:
        return
    models, train_loss = {}, {}
    if "train" in req:
        for name in METHODS:
            X, Y = build_dataset(pairs, rie, name, cfg)
            t0 = time.time()
            models[name], train_loss[name] = train_model(X, Y, cfg, seed=1)
            torch.save(models[name].state_dict(), out / f"model_{name}.pt")
            print(f"trained {name} on {len(X)} points: final MSE {train_loss[name]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        (out / "train_loss.json").write_text(json.dumps(train_loss), encoding="utf-8")
    else:
        for name in METHODS:
            models[name] = MLP(d_in=5, d_out=3, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
            models[name].load_state_dict(torch.load(out / f"model_{name}.pt"))
            models[name].eval()
        train_loss = json.loads((out / "train_loss.json").read_text(encoding="utf-8"))
    if last == 2:
        return
    if "eval" in req:
        rows, starts, paths = evaluate(models, F, G, cfg, out)
    else:
        with open(out / "fm_planner_se2_eval.csv", newline="") as f:
            rows = [{k: (v if k == "method" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
        z = np.load(out / "fm_planner_se2_paths.npz")
        starts = z["starts"]
        paths = {k: z[f"path_{k}"] for k in METHODS}
    summarise(rows, cfg, train_loss, out)
    fig_paths(paths, starts, rows, cfg, out)
    fig_bearing(rows, cfg, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
