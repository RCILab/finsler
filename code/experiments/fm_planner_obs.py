"""
Time-optimal planner with OBSTACLES: obstacles enter the Finsler metric as a conformal factor
    F_obs(x, v) = s(x) F(x, v),   s = 1 + 30 * sigmoid((r + margin - dist)/0.08)  (30x slower inside),
so geodesics avoid them and every baseline is obstacle-aware; the three methods differ only in the
wind-awareness of the underlying metric:
    euclid   s(x) |v|                    obstacle-aware, wind-blind
    riemann  s(x) sqrt(a(x)(v,v))        obstacle-aware, anisotropic but direction-blind
    finsler  s(x) (sqrt(a(v,v)) + b(v))  obstacle-aware, asymmetric (ours)
Conditional paths: the grid Dijkstra path selects the homotopy class, then the discrete Finsler length
of the polyline is minimised by gradient descent (torch, fixed endpoints, small smoothness term) and
the result is resampled at constant Finsler speed.  Shooting is not used here: the conformal factor
makes the Hamiltonian stiff near obstacles and Newton converged for only 30-60% of pairs.

Stages: pairs (per metric, parallelisable), train, eval, fig.
Usage:  python experiments/fm_planner_obs.py --stage pairs --metric finsler
        python experiments/fm_planner_obs.py --stage train,eval,fig
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
from scipy.sparse.csgraph import dijkstra

from finsler import channel_wind, zermelo_to_randers
from finsler.eikonal import GridFinslerDistance
from finsler.randers import RandersField
from fm_path_benefit import MLP
import fm_planner as fp

C, LABEL, METHODS = fp.C, {"euclid": "obstacle-aware Euclidean paths", "riemann": "obstacle-aware Riemannian a(x) paths", "finsler": "obstacle-aware Finsler paths (ours)"}, fp.METHODS
C_INK, C_INK2, C_SURF = fp.C_INK, fp.C_INK2, fp.C_SURF
ROOT = pathlib.Path(__file__).resolve().parents[1]

CFG = dict(fp.CFG)
CFG.update(dict(n_pairs=1600, obstacles=[[0.2, 1.3, 0.45], [-0.6, -1.2, 0.4], [1.4, 0.55, 0.35]], obs_gain=30.0, obs_margin=0.05, obs_width=0.08,
                log_steps=100, bvp_iters=12, branch_tol=1.05, polish_iters=400, polish_lr=0.01, polish_smooth=0.02))


def obstacle_scale(cfg):
    obs = np.asarray(cfg["obstacles"], float)

    def s(x):
        x = np.asarray(x, float)
        out = np.ones(x.shape[:-1])
        for cx, cy, r in obs:
            d = np.linalg.norm(x[..., :2] - np.array([cx, cy]), axis=-1)
            out = out + cfg["obs_gain"] / (1.0 + np.exp((d - r - cfg["obs_margin"]) / cfg["obs_width"]))
        return out

    return s


def make_metrics(beta, cfg):
    W1 = channel_wind(1.0, cfg["width"])
    s = obstacle_scale(cfg)
    Wb = lambda x: beta * W1(x)
    F = RandersField(Wb, scale=s)

    def h_a(x):
        x = np.asarray(x, float)
        A, _, _ = zermelo_to_randers(np.broadcast_to(np.eye(2), x.shape + (2,)), Wb(x))
        return A

    R = RandersField(lambda x: np.zeros_like(np.asarray(x, float)), h_a, scale=s)
    E = RandersField(lambda x: np.zeros_like(np.asarray(x, float)), scale=s)
    return {"euclid": E, "riemann": R, "finsler": F}, W1, s


class TorchObsMetric:
    """torch version of the obstacle-aware metrics for differentiable path polishing."""

    def __init__(self, name, beta, cfg):
        self.name, self.beta, self.cfg = name, beta, cfg
        self.obs = torch.tensor(cfg["obstacles"], dtype=torch.float64)

    def s(self, x):
        out = torch.ones(x.shape[:-1], dtype=x.dtype)
        for k in range(len(self.obs)):
            d = torch.linalg.norm(x[..., :2] - self.obs[k, :2], dim=-1)
            out = out + self.cfg["obs_gain"] * torch.sigmoid(-(d - self.obs[k, 2] - self.cfg["obs_margin"]) / self.cfg["obs_width"])
        return out

    def F(self, x, v):
        W = torch.stack([-self.beta * torch.exp(-(x[..., 1] ** 2) / self.cfg["width"] ** 2), torch.zeros_like(x[..., 0])], -1)
        if self.name == "euclid":
            F0 = torch.sqrt((v * v).sum(-1) + 1e-18)
        else:
            lam = 1.0 - (W * W).sum(-1)
            Wv = (W * v).sum(-1)
            vv = (v * v).sum(-1)
            alpha = torch.sqrt((lam * vv + Wv**2) / lam**2 + 1e-18)            # sqrt(v^T A v), A = (lam I + W W^T)/lam^2
            F0 = alpha if self.name == "riemann" else alpha - Wv / lam          # + b.v with b = -W/lam
        return self.s(x) * F0


def polish_paths(metric, P0, iters=400, lr=0.01, smooth=0.02):
    """Minimise the discrete Finsler length of polylines (n, m, 2) with fixed endpoints; returns polished paths."""
    P0 = torch.as_tensor(P0, dtype=torch.float64)
    inner = P0[:, 1:-1].clone().requires_grad_(True)
    ends = (P0[:, :1], P0[:, -1:])
    opt = torch.optim.Adam([inner], lr=lr)
    for it in range(iters):
        P = torch.cat([ends[0], inner, ends[1]], 1)
        d = P[:, 1:] - P[:, :-1]
        mid = 0.5 * (P[:, 1:] + P[:, :-1])
        length = metric.F(mid, d).sum(1)
        reg = smooth * ((d[:, 1:] - d[:, :-1]) ** 2).sum((1, 2))
        loss = (length + reg).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        P = torch.cat([ends[0], inner, ends[1]], 1)
        d = P[:, 1:] - P[:, :-1]
        L = metric.F(0.5 * (P[:, 1:] + P[:, :-1]), d).sum(1)
    return P.detach().numpy(), L.numpy()


def in_collision(path, cfg):
    """path: (steps, n, 2) -> (n,) bool: any point strictly inside an obstacle disc."""
    hit = np.zeros(path.shape[1], bool)
    for cx, cy, r in cfg["obstacles"]:
        hit |= (np.linalg.norm(path - np.array([cx, cy]), axis=-1) < r).any(0)
    return hit


# --------------------------------------------------------------------------------------------------
def resample_grid_path(M, P, K, smooth=False):
    """(Optionally smooth) a polyline P (m, 2) and resample it at K points with constant Finsler speed.
    Returns X (K, 2), U (K, 2) with F(X_t, U_t) = T (the path's Finsler length) and T."""
    if smooth and len(P) >= 5:
        Q = P.copy()
        for _ in range(2):
            Q[1:-1] = 0.2 * (Q[:-2] + Q[1:-1] + Q[2:]) + 0.2 * (np.roll(Q, 2, 0)[1:-1] + np.roll(Q, -2, 0)[1:-1])
            Q[0], Q[-1] = P[0], P[-1]
        P = Q
    d = P[1:] - P[:-1]
    mid = 0.5 * (P[1:] + P[:-1])
    c = M.F(mid, d)
    cum = np.concatenate([[0.0], np.cumsum(c)])
    T = cum[-1]
    tk = np.linspace(0, 1, K) * T
    seg = np.clip(np.searchsorted(cum, tk, side="right") - 1, 0, len(c) - 1)
    lam = (tk - cum[seg]) / np.maximum(c[seg], 1e-12)
    X = P[seg] + lam[:, None] * d[seg]
    U = T * d[seg] / M.F(X, d[seg])[:, None]
    return X, U, T


def resample_uniform(P, m):
    """Resample a polyline at m points uniform in Euclidean arc length (initialisation for polishing)."""
    d = np.linalg.norm(P[1:] - P[:-1], axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    tk = np.linspace(0, cum[-1], m)
    seg = np.clip(np.searchsorted(cum, tk, side="right") - 1, 0, len(d) - 1)
    lam = (tk - cum[seg]) / np.maximum(d[seg], 1e-12)
    return P[seg] + lam[:, None] * (P[seg + 1] - P[seg])


def grid_warm_start(M, G, x0, x1, K, look=4):
    """Grid Dijkstra paths: initial velocities for shooting (w0 = T_grid * dir / M.F(x0, dir)) and the
    resampled grid paths themselves (fallback conditional paths where shooting fails)."""
    n = len(x0)
    w0 = np.zeros((n, 2))
    Tg = np.zeros(n)
    Xg = np.zeros((n, K, 2))
    Ug = np.zeros((n, K, 2))
    Praw = np.zeros((n, 64, 2))
    chunk = 150
    for a in range(0, n, chunk):
        idx = np.arange(a, min(a + chunk, n))
        D, pred = dijkstra(G.graph, directed=True, indices=G.node_of(x0[idx]), return_predecessors=True)
        ends = G.node_of(x1[idx])
        for j, i in enumerate(idx):
            Tg[i] = D[j, ends[j]]
            path = [ends[j]]
            start_node = G.node_of(x0[i:i + 1])[0]
            while path[-1] != start_node and pred[j, path[-1]] >= 0:
                path.append(pred[j, path[-1]])
            path = path[::-1]
            P = G.nodes[path].copy()
            P[0], P[-1] = x0[i], x1[i]
            Xg[i], Ug[i], _ = resample_grid_path(M, P, K, smooth=True)
            Praw[i] = resample_uniform(P, 64)
            k = min(look, len(path) - 1)
            d = G.nodes[path[k]] - x0[i]
            if np.linalg.norm(d) < 1e-9:
                d = x1[i] - x0[i]
            w0[i] = Tg[i] * d / M.F(x0[i:i + 1], d[None])[0]
    return w0, Tg, Xg, Ug, Praw


def gen_pairs_metric(name, M, cfg, beta, rng, out):
    goals = np.asarray(cfg["goals"], float)
    (xlo, xhi), (ylo, yhi) = cfg["start_box"]
    n, K = cfg["n_pairs"], cfg["K"]
    rng = np.random.default_rng(0)                                        # identical pairs for every metric
    x0 = np.stack([rng.uniform(xlo, xhi, n), rng.uniform(ylo, yhi, n)], 1)
    which = np.arange(n) % len(goals)
    x1 = goals[which] + cfg["goal_std"] * rng.normal(size=(n, 2))
    t0 = time.time()
    G = GridFinslerDistance(M, cfg["grid_xlim"], cfg["grid_ylim"], cfg["grid_h"])
    w0, Tg, Xg, Ug, Praw = grid_warm_start(M, G, x0, x1, K)
    print(f"[{name}] grid paths ({time.time()-t0:.0f}s); T_grid in [{Tg.min():.2f}, {Tg.max():.2f}]", flush=True)
    t0 = time.time()
    metric = TorchObsMetric(name, beta, cfg)
    P, Lp = polish_paths(metric, Praw, iters=cfg["polish_iters"], lr=cfg["polish_lr"], smooth=cfg["polish_smooth"])
    X = np.zeros((n, K, 2))
    U = np.zeros((n, K, 2))
    Tp = np.zeros(n)
    for i in range(n):
        X[i], U[i], Tp[i] = resample_grid_path(M, P[i], K)
    ratio = Tp / Tg
    print(f"[{name}] polished {n} paths: length/grid median {np.median(ratio):.4f}, max {ratio.max():.3f}, collisions after polishing {in_collision(X.transpose(1, 0, 2), cfg).mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)
    ok = (Tg <= cfg["T_max"]) & (ratio <= 1.05)
    np.savez(out / f"pairs_{name}.npz", x0=x0, x1=x1, which=which, ok=ok, geo=np.zeros(n, bool), T=Tp, Tg=Tg, X=X, U=U, t=np.linspace(0, 1, K))


def load_pairs(out):
    P = {name: dict(np.load(out / f"pairs_{name}.npz")) for name in METHODS}
    keep = np.logical_and.reduce([P[name]["ok"] for name in METHODS])
    print(f"pairs kept by all three metrics: {keep.sum()} / {len(keep)}; exact geodesics per metric: " + ", ".join(f"{k} {P[k]['geo'].sum()}" for k in METHODS))
    return P, keep


def build_dataset(P, keep, name, cfg):
    goals = np.asarray(cfg["goals"], float)
    p = P[name]
    idx = np.cumsum(p["ok"]) - 1
    X, U = p["X"][idx[keep]], p["U"][idx[keep]]
    t = p["t"]
    n, K = len(X), len(t)
    cols = [X.reshape(-1, 2), np.tile(t, n)[:, None], np.repeat(goals[p["which"][keep]], K, 0)]
    return np.concatenate(cols, 1), U.reshape(-1, 2)


# --------------------------------------------------------------------------------------------------
def evaluate(models, metrics, cfg, out):
    F = metrics["finsler"]
    G = GridFinslerDistance(F, cfg["grid_xlim"], cfg["grid_ylim"], cfg["grid_h"])
    goals = np.asarray(cfg["goals"], float)
    (xlo, xhi), (ylo, yhi) = cfg["start_box"]
    gx, gy = cfg["eval_grid"]
    starts = np.stack(np.meshgrid(np.linspace(xlo + 0.2, xhi - 0.2, gx), np.linspace(ylo + 0.2, yhi - 0.2, gy), indexing="ij"), -1).reshape(-1, 2)
    D_start = G.dist_from(starts)
    d_goal = D_start[:, G.node_of(goals)]
    Dg = [G.dist_to_target(g) for g in goals]
    rows, paths = [], {}
    for name in METHODS:
        for gi in range(len(goals)):
            path = fp.integrate(models[name], starts, cfg["eval_steps"], goal=goals[gi])
            paths[(name, gi)] = path
            end = path[-1]
            resid = np.linalg.norm(end - goals[gi], axis=1)
            path = np.clip(path, -6.0, 6.0)
            L = fp.finsler_length(F, path)                                  # true (obstacle-aware, wind-aware) travel time
            d_end = D_start[np.arange(len(starts)), G.node_of(end)]
            d_final = Dg[gi][G.node_of(end)]
            hit = in_collision(path, cfg)
            for j in range(len(starts)):
                rows.append(dict(method=name, goal_cond=gi, sx=starts[j, 0], sy=starts[j, 1], ex=end[j, 0], ey=end[j, 1], goal=gi, resid=resid[j], travel=L[j],
                                 optimal_end=d_end[j], optimal_goal=d_goal[j, gi], final_leg=d_final[j], ratio=L[j] / d_end[j],
                                 ratio_goal=(L[j] + d_final[j]) / d_goal[j, gi], collision=int(hit[j]), straight_ratio=np.nan))
            print(f"  eval {name} goal={gi}: reach<{cfg['goal_r'][0]} {np.mean(resid < cfg['goal_r'][0]):.2f}, collisions {hit.mean():.2f}, travel {L.mean():.2f}, ratio {np.mean(L / d_end):.3f}, time-to-goal ratio {np.mean((L + d_final) / d_goal[:, gi]):.3f}", flush=True)
    with open(out / "fm_planner_obs_eval.csv", "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)
    np.savez(out / "fm_planner_obs_paths.npz", starts=starts, **{f"path_{k[0]}_{k[1]}": v for k, v in paths.items()})
    return rows, starts, paths


def summarise(rows, cfg, beta, train_loss, out):
    r0 = cfg["goal_r"][0]
    lines = [f"# Planner with obstacles (beta = {beta})", "", f"config: `{json.dumps(cfg)}`", "", f"final train MSE: {train_loss}", "",
             "All three conditional-path families are obstacle-aware (conformal factor s(x), 30x slower inside obstacles); they differ in wind-awareness only.", "",
             "| method | goal | n | reach < %.2f | collision rate | median travel [s] | mean optimal to goal [s] | ratio median (mean) | worst | time-to-goal ratio median (mean) |" % r0,
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in METHODS:
        for gsel in [0, 1, "all"]:
            sel = [r for r in rows if r["method"] == name and (gsel == "all" or r["goal_cond"] == gsel)]
            ratio = np.array([r["ratio"] for r in sel])
            rg = np.array([r["ratio_goal"] for r in sel])
            lines.append(f"| {name} | {gsel} | {len(sel)} | {np.mean([r['resid'] < r0 for r in sel]):.2f} | {np.mean([r['collision'] for r in sel]):.2f} | {np.median([r['travel'] for r in sel]):.2f} | "
                         f"{np.mean([r['optimal_goal'] for r in sel]):.2f} | {np.median(ratio):.3f} (mean {ratio.mean():.3f}) | {ratio.max():.2f} | {np.median(rg):.3f} (mean {rg.mean():.3f}) |")
    (out / "fm_planner_obs_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def fig_paths(paths, starts, rows, cfg, beta, W1, out):
    goals = np.asarray(cfg["goals"], float)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), facecolor=C_SURF)
    xx, yy = np.meshgrid(np.linspace(-4.0, 3.5, 151), np.linspace(-2.4, 2.6, 101))
    grid = np.stack([xx, yy], -1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5"])
    show = np.arange(0, len(starts), max(1, len(starts) // 8))
    for ax, name in zip(axes, METHODS):
        fp.style(ax)
        ax.contourf(xx, yy, beta * np.linalg.norm(W1(grid), axis=-1), levels=np.linspace(0, beta, 8), cmap=cmap)
        for cx, cy, r in cfg["obstacles"]:
            ax.add_patch(plt.Circle((cx, cy), r, color="#c3c2b7"))
        for g in goals:
            ax.add_patch(plt.Circle(g, cfg["goal_r"][0], fill=False, color=C_INK, lw=1.2, ls=":"))
            ax.plot(*g, "*", color=C_INK, ms=12)
        for gi in range(len(goals)):
            p = paths[(name, gi)]
            for j in show:
                ax.plot(p[:, j, 0], p[:, j, 1], color=C[name], lw=1.5, alpha=0.9 if gi == 0 else 0.55)
        for j in show:
            ax.plot(*starts[j], "o", color=C_INK, ms=4)
        sel = [r for r in rows if r["method"] == name]
        ax.set_title(f"{LABEL[name]}\ntravel/optimal {np.mean([r['ratio'] for r in sel]):.2f}, collisions {np.mean([r['collision'] for r in sel]):.2f}, reach {np.mean([r['resid'] < cfg['goal_r'][0] for r in sel]):.2f}", fontsize=9.5, color=C_INK, loc="left")
        ax.set_aspect("equal")
        ax.set_xlim(-4.0, 3.5)
        ax.set_ylim(-2.4, 2.6)
    fig.suptitle(f"Goal-conditioned plans with obstacles (grey), head-wind channel β = {beta}. Obstacles are a conformal factor of each metric; only wind-awareness differs.", fontsize=10, color=C_INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fm_planner_obs_paths.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--metric", default="all", help="for the pairs stage: euclid|riemann|finsler|all")
    args = ap.parse_args()
    cfg = dict(CFG)
    beta = args.beta
    out = ROOT / "results" / f"fm_planner_obs_beta{beta}"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    metrics, W1, s = make_metrics(beta, cfg)
    req = ["pairs", "train", "eval", "fig"] if args.stage == "all" else [x.strip() for x in args.stage.split(",")]

    if "pairs" in req:
        for name in (METHODS if args.metric == "all" else [args.metric]):
            gen_pairs_metric(name, metrics[name], cfg, beta, rng, out)
        if not any(x in req for x in ["train", "eval", "fig"]):
            return
    P, keep = load_pairs(out)
    models, train_loss = {}, {}
    if "train" in req:
        for name in METHODS:
            X, Y = build_dataset(P, keep, name, cfg)
            t0 = time.time()
            models[name], train_loss[name] = fp.train_model(X, Y, cfg, seed=1)
            torch.save(models[name].state_dict(), out / f"model_{name}.pt")
            print(f"trained {name} on {len(X)} points: final MSE {train_loss[name]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        (out / "train_loss.json").write_text(json.dumps(train_loss), encoding="utf-8")
    else:
        for name in METHODS:
            models[name] = MLP(d_in=5, d_out=2, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
            models[name].load_state_dict(torch.load(out / f"model_{name}.pt"))
            models[name].eval()
        train_loss = json.loads((out / "train_loss.json").read_text(encoding="utf-8"))
    if "eval" in req:
        rows, starts, paths = evaluate(models, metrics, cfg, out)
    else:
        with open(out / "fm_planner_obs_eval.csv", newline="") as f:
            rows = [{k: (v if k == "method" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
        for r in rows:
            r["goal_cond"] = int(r["goal_cond"])
        z = np.load(out / "fm_planner_obs_paths.npz")
        starts = z["starts"]
        paths = {(k.split("_")[1], int(k.split("_")[2])): z[k] for k in z.files if k.startswith("path_")}
    summarise(rows, cfg, beta, train_loss, out)
    fig_paths(paths, starts, rows, cfg, beta, W1, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
