"""
Flow matching as a generative time-optimal planner: the flow trajectory IS the plan.

Source p0 = distribution of start states (west of a head-wind channel), target q = goal mixture (east).
Conditional paths x_t from x0 to x1 are geodesics of one of three metrics, all with the schedule
d(x_t, x1) = (1 - t) d(x0, x1):
    Euclidean   straight lines                    (standard OT-CFM)
    Riemannian  geodesics of the symmetric part a(x)   (RFM-style)
    Finsler     forward Randers geodesics         (ours; minimal branch of a 3-branch BVP, checked against grid Dijkstra)
The learned unconditional field v_theta(t, x) is integrated from held-out starts; the resulting path
is executed at full speed, so its Finsler length is its travel time.  We report the optimality ratio
    travel time / d_F(start, reached point)
with d_F from a three-branch BVP solve, plus goal-reaching success.

Stages cached in results/fm_planner_beta<beta>/ : pairs -> riemann -> train -> eval -> fig
Usage: python experiments/fm_planner.py [--beta 0.9] [--stage all|pairs,riemann,train,eval,fig]
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

from finsler import channel_wind, euclidean_field, finsler_field, riemannian_part
from finsler.eikonal import GridFinslerDistance
from fm_path_benefit import MLP, log_continuation

C = {"euclid": "#2a78d6", "riemann": "#eb6834", "finsler": "#1baf7a"}
LABEL = {"euclid": "Euclidean paths (OT-CFM)", "riemann": "Riemannian paths a(x) (RFM)", "finsler": "Finsler geodesic paths (ours)"}
C_MUTED, C_GRID, C_INK, C_INK2, C_SURF = "#898781", "#e1e0d9", "#0b0b0b", "#52514e", "#fcfcfb"
METHODS = ["euclid", "riemann", "finsler"]

CFG = dict(
    width=0.8, goals=[[2.5, 0.0], [2.5, 1.6]], goal_std=0.15, start_box=[[-3.5, -1.2], [-1.8, 1.8]],
    T_max=14.0, n_pairs=2000, K=48,
    hidden=256, n_layers=4, train_steps=6000, batch=2048, lr=1e-3,
    eval_grid=[7, 7], eval_steps=200, goal_r=[0.35, 0.5], log_steps=120,
    grid_xlim=[-4.2, 3.6], grid_ylim=[-2.6, 2.8], grid_h=0.04, branch_tol=1.05, goal_cond=True,
)


def make_fields(beta, width):
    W1 = channel_wind(1.0, width)
    mkF = lambda s=1.0: finsler_field(lambda x, s=s: (s * beta) * W1(x))
    mkR = lambda s=1.0: riemannian_part(lambda x, s=s: (s * beta) * W1(x))
    return mkF, mkR, euclidean_field(), W1


def solve_bvps(mk, x0, x1, n_steps=150, waypoint_y=(1.4, -1.4)):
    """log_{x0}(x1) for the field mk(1.0): min-time over three branches (direct / via upper / via lower waypoint)."""
    F = mk(1.0)
    n = len(x0)
    cands = []
    w, err = log_continuation(mk, x0, x1, list(np.linspace(0.15, 1.0, 7)), n_steps, iters=10)
    cands.append((w, err))
    for wy in waypoint_y:
        wp = np.stack([0.5 * (x0[:, 0] + x1[:, 0]), np.full(n, wy)], 1)
        w, err = log_continuation(mk, x0, wp, list(np.linspace(0.34, 1.0, 3)), n_steps, iters=10)
        for s_ in np.linspace(0.25, 1.0, 4):
            w, err = F.log_batched(x0, wp + s_ * (x1 - wp), n_steps=n_steps, iters=10, w_init=w)
        cands.append((w, err))
    times = np.stack([np.where(e < 1e-6, F.F(x0, w), np.inf) for w, e in cands])
    best = np.argmin(times, 0)
    w = np.stack([cands[b][0][i] for i, b in enumerate(best)])
    T = times[best, np.arange(n)]
    return w, T, np.isfinite(T), best


# --------------------------------------------------------------------------------------------------
def gen_pairs(mkF, cfg, rng):
    """(x0, x1) ~ Uniform(start box) x goal mixture; Finsler conditional path = minimal geodesic (3-branch BVP)."""
    F = mkF(1.0)
    goals = np.asarray(cfg["goals"], float)
    (xlo, xhi), (ylo, yhi) = cfg["start_box"]
    K, n = cfg["K"], cfg["n_pairs"]
    x0 = np.stack([rng.uniform(xlo, xhi, n), rng.uniform(ylo, yhi, n)], 1)
    if cfg.get("global_starts"):
        far = np.linalg.norm(x0[:, None, :] - goals[None], axis=-1).min(1) > 0.6
        x0 = x0[far]
        n = len(x0)
    which = np.arange(n) % len(goals)
    x1 = goals[which] + cfg["goal_std"] * rng.normal(size=(n, 2))
    t0 = time.time()
    w, T, ok, branch = solve_bvps(mkF, x0, x1, cfg["log_steps"])
    ok &= T <= cfg["T_max"]
    print(f"finsler BVPs: {ok.sum()}/{n} kept (converged and T <= {cfg['T_max']}), branches direct/up/down = "
          f"{(branch[ok]==0).sum()}/{(branch[ok]==1).sum()}/{(branch[ok]==2).sum()}, T in [{T[ok].min():.1f}, {T[ok].max():.1f}]  ({time.time()-t0:.0f}s)", flush=True)
    x0, x1, w, T, which = x0[ok], x1[ok], w[ok], T[ok], which[ok]
    _, xs, vs = F.geodesic(x0, w, T=1.0, n_steps=K - 1)                 # parameter time 1 <-> flow time; F-speed = T
    XF, UF = xs.transpose(1, 0, 2), vs.transpose(1, 0, 2)
    speed = F.F(XF.reshape(-1, 2), UF.reshape(-1, 2)).reshape(len(x0), K)
    print(f"pairs: {len(x0)}; F(x_t, u_t)/T in [{(speed/T[:,None]).min():.4f}, {(speed/T[:,None]).max():.4f}] (should be 1); goal split {np.bincount(which, minlength=len(goals))}")
    return dict(x0=x0, x1=x1, T=T, which=which, t=np.linspace(0, 1, K), XF=XF, UF=UF)


def riemann_paths(mkR, pairs, cfg):
    x0, x1 = pairs["x0"], pairs["x1"]
    t0 = time.time()
    w, T, ok, branch = solve_bvps(mkR, x0, x1, cfg["log_steps"])
    print(f"riemannian BVPs: {ok.sum()}/{len(ok)} converged, branches direct/up/down = {(branch[ok]==0).sum()}/{(branch[ok]==1).sum()}/{(branch[ok]==2).sum()}  ({time.time()-t0:.0f}s)")
    R = mkR(1.0)
    _, xs, vs = R.geodesic(x0[ok], w[ok], T=1.0, n_steps=cfg["K"] - 1)
    return dict(ok=ok, XR=xs.transpose(1, 0, 2), UR=vs.transpose(1, 0, 2))


def branch_filter(pairs, rie, mkF, mkR, cfg):
    """Drop pairs whose BVP geodesic is not the minimal one (time > branch_tol * grid distance), for F and R."""
    GF = GridFinslerDistance(mkF(1.0), cfg["grid_xlim"], cfg["grid_ylim"], cfg["grid_h"])
    GR = GridFinslerDistance(mkR(1.0), cfg["grid_xlim"], cfg["grid_ylim"], cfg["grid_h"])
    ok = rie["ok"].copy()
    dF = GF.dist(pairs["x0"], pairs["x1"])
    okF = pairs["T"] <= cfg["branch_tol"] * dF
    R = mkR(1.0)
    TR = np.full(len(ok), np.inf)
    TR[ok] = R.F(rie["XR"][:, 0], rie["UR"][:, 0])                     # R-time of the R-geodesic (constant R-speed)
    dR = GR.dist(pairs["x0"], pairs["x1"])
    okR = TR <= cfg["branch_tol"] * dR
    keep = ok & okF & okR
    print(f"branch filter: F non-minimal {(~okF).sum()}, R non-minimal {(ok & ~okR).sum()}, kept {keep.sum()} / {len(keep)} pairs", flush=True)
    return keep


def build_dataset(pairs, rie, method, keep, cfg):
    goals = np.asarray(cfg["goals"], float)
    t = pairs["t"]
    K = len(t)
    ok_r = rie["ok"]
    if method == "finsler":
        X, U = pairs["XF"][keep], pairs["UF"][keep]
    elif method == "riemann":
        idx_r = np.cumsum(ok_r) - 1                                       # position of each pair inside the R arrays
        X, U = rie["XR"][idx_r[keep]], rie["UR"][idx_r[keep]]
    else:
        x0, x1 = pairs["x0"][keep], pairs["x1"][keep]
        X = x0[:, None, :] + t[None, :, None] * (x1 - x0)[:, None, :]
        U = np.repeat((x1 - x0)[:, None, :], K, 1)
    n = len(X)
    cols = [X.reshape(-1, 2), np.tile(t, n)[:, None]]
    if cfg["goal_cond"]:
        g = goals[pairs["which"][keep]]
        cols.append(np.repeat(g, K, 0))
    return np.concatenate(cols, 1), U.reshape(-1, 2)


def train_model(X, Y, cfg, seed):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    model = MLP(d_in=X.shape[1], d_out=2, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
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


def integrate(model, starts, n_steps, goal=None):
    """RK4 integration of dx/dt = v_theta(t, x [, g]) from t=0 to 1; returns path (n_steps+1, n, 2)."""
    x = starts.copy()
    path = [x.copy()]
    dt = 1.0 / n_steps
    gcol = None if goal is None else np.repeat(np.asarray(goal, float)[None], len(x), 0)

    def f(t, x):
        cols = [x, np.full((len(x), 1), t)] + ([] if gcol is None else [gcol])
        with torch.no_grad():
            return model(torch.tensor(np.concatenate(cols, 1), dtype=torch.float32)).numpy().astype(float)

    for s in range(n_steps):
        t = s * dt
        k1 = f(t, x)
        k2 = f(t + dt / 2, x + dt / 2 * k1)
        k3 = f(t + dt / 2, x + dt / 2 * k2)
        k4 = f(t + dt, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        path.append(x.copy())
    return np.array(path)


def finsler_length(F, path):
    """Discrete Finsler length sum_k F(x_k, x_{k+1} - x_k): travel time when executed at full speed."""
    d = path[1:] - path[:-1]
    return F.F(path[:-1].reshape(-1, 2), d.reshape(-1, 2)).reshape(d.shape[0], d.shape[1]).sum(0)


def evaluate(models, mkF, cfg, out):
    F = mkF(1.0)
    G = GridFinslerDistance(F, cfg["grid_xlim"], cfg["grid_ylim"], cfg["grid_h"])
    goals = np.asarray(cfg["goals"], float)
    (xlo, xhi), (ylo, yhi) = cfg["start_box"]
    gx, gy = cfg["eval_grid"]
    sx = np.linspace(xlo + 0.2, xhi - 0.2, gx)
    sy = np.linspace(ylo + 0.2, yhi - 0.2, gy)
    starts = np.stack(np.meshgrid(sx, sy, indexing="ij"), -1).reshape(-1, 2)
    D_start = G.dist_from(starts)                                          # optimal time from each start to every grid node
    d_goal = D_start[:, G.node_of(goals)]                                  # optimal time start -> goal centres
    goal_list = list(range(len(goals))) if cfg["goal_cond"] else [None]
    rows, paths = [], {}
    for name in METHODS:
        for gi_cond in goal_list:
            path = integrate(models[name], starts, cfg["eval_steps"], goal=None if gi_cond is None else goals[gi_cond])
            paths[(name, gi_cond)] = path
            end = path[-1]
            dist_goal = np.linalg.norm(end[:, None, :] - goals[None], axis=-1)
            gi = dist_goal.argmin(1) if gi_cond is None else np.full(len(starts), gi_cond)
            resid = dist_goal[np.arange(len(starts)), gi]
            L = finsler_length(F, path)
            d_end = D_start[np.arange(len(starts)), G.node_of(end)]         # optimal time start -> reached point
            d_final = np.stack([G.dist_to_target(goals[k])[G.node_of(end)] for k in range(len(goals))], 1)[np.arange(len(starts)), gi]   # reached point -> goal centre
            L_straight = finsler_length(F, np.stack([starts + s * (end - starts) for s in np.linspace(0, 1, 201)]))
            for j in range(len(starts)):
                rows.append(dict(method=name, goal_cond=-1 if gi_cond is None else gi_cond, sx=starts[j, 0], sy=starts[j, 1], ex=end[j, 0], ey=end[j, 1],
                                 goal=int(gi[j]), resid=resid[j], travel=L[j], optimal_end=d_end[j], optimal_goal=d_goal[j, gi[j]], final_leg=d_final[j],
                                 ratio=L[j] / d_end[j], ratio_goal=(L[j] + d_final[j]) / d_goal[j, gi[j]], straight_ratio=L_straight[j] / d_end[j]))
            print(f"  eval {name} goal={gi_cond}: reach<{cfg['goal_r'][0]} {np.mean(resid < cfg['goal_r'][0]):.2f}, reach<{cfg['goal_r'][1]} {np.mean(resid < cfg['goal_r'][1]):.2f}, "
                  f"travel {L.mean():.2f} (optimal to reached point {d_end.mean():.2f}, to goal centre {d_goal[np.arange(len(starts)), gi].mean():.2f}), ratio {np.mean(L / d_end):.3f}", flush=True)
    with open(out / "fm_planner_eval.csv", "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)
    np.savez(out / "fm_planner_paths.npz", starts=starts, **{f"path_{k[0]}_{k[1]}": v for k, v in paths.items()})
    return rows, starts, paths


def summarise(rows, cfg, beta, train_loss, out):
    r0, r1 = cfg["goal_r"]
    lines = [f"# FM as a time-optimal planner (beta = {beta})", "", f"config: `{json.dumps(cfg)}`", "", f"final train MSE: {train_loss}", "",
             "Optimal times from grid Dijkstra with asymmetric Finsler edge costs (h = %.2f, 32 directions); ratio = travel time of the generated plan / optimal time to the point it reached." % cfg["grid_h"], "",
             "time-to-goal ratio = (travel time of the plan + optimal time from its end point to the goal centre) / optimal time from the start to the goal centre, i.e. the plan followed by an optimal final leg.", "",
             f"| method | goal | n | reach < {r0} | reach < {r1} | mean travel [s] | mean optimal to goal [s] | mean ratio ± s.e. | median ratio | worst ratio | time-to-goal ratio ± s.e. | straight-line ratio (ref.) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summ = {}
    goal_vals = sorted({r["goal_cond"] for r in rows})
    for name in METHODS:
        for gsel in goal_vals + ["all"]:
            sel = [r for r in rows if r["method"] == name and (gsel == "all" or r["goal_cond"] == gsel)]
            ratio = np.array([r["ratio"] for r in sel])
            rg = np.array([r["ratio_goal"] for r in sel])
            v = dict(n=len(sel), reach0=np.mean([r["resid"] < r0 for r in sel]), reach1=np.mean([r["resid"] < r1 for r in sel]), travel=np.mean([r["travel"] for r in sel]),
                     opt_goal=np.mean([r["optimal_goal"] for r in sel]), ratio=np.mean(ratio), ratio_se=np.std(ratio, ddof=1) / np.sqrt(len(ratio)), ratio_med=np.median(ratio),
                     ratio_max=np.max(ratio), ratio_goal=rg.mean(), ratio_goal_se=rg.std(ddof=1) / np.sqrt(len(rg)), straight=np.mean([r["straight_ratio"] for r in sel]))
            summ[(name, gsel)] = v
            lines.append(f"| {name} | {gsel} | {v['n']} | {v['reach0']:.2f} | {v['reach1']:.2f} | {v['travel']:.2f} | {v['opt_goal']:.2f} | {v['ratio']:.3f} ± {v['ratio_se']:.3f} | {v['ratio_med']:.3f} | {v['ratio_max']:.2f} | {v['ratio_goal']:.3f} ± {v['ratio_goal_se']:.3f} | {v['straight']:.3f} |")
    (out / "fm_planner_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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


def fig_paths(paths, starts, mkF, W1, cfg, beta, rows, out):
    goals = np.asarray(cfg["goals"], float)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), facecolor=C_SURF)
    xx, yy = np.meshgrid(np.linspace(-4.0, 3.5, 151), np.linspace(-2.4, 2.6, 101))
    grid = np.stack([xx, yy], -1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5"])
    show = np.arange(0, len(starts), max(1, len(starts) // 8))
    keys = sorted({k[1] for k in paths}, key=lambda v: -1 if v is None else v)
    for ax, name in zip(axes, METHODS):
        style(ax)
        ax.contourf(xx, yy, beta * np.linalg.norm(W1(grid), axis=-1), levels=np.linspace(0, beta, 8), cmap=cmap)
        for g in goals:
            ax.add_patch(plt.Circle(g, cfg["goal_r"][0], fill=False, color=C_INK, lw=1.2, ls=":"))
            ax.plot(*g, "*", color=C_INK, ms=12)
        for gk in keys:
            p = paths[(name, gk)]
            for j in show:
                ax.plot(p[:, j, 0], p[:, j, 1], color=C[name], lw=1.5, alpha=0.9 if gk in (0, None) else 0.55)
        for j in show:
            ax.plot(*starts[j], "o", color=C_INK, ms=4)
        sel = [r for r in rows if r["method"] == name]
        ax.set_title(f"{LABEL[name]}\nmean travel/optimal = {np.mean([r['ratio'] for r in sel]):.2f}, reach<{cfg['goal_r'][0]}: {np.mean([r['resid'] < cfg['goal_r'][0] for r in sel]):.2f}", fontsize=9.5, color=C_INK, loc="left")
        ax.set_aspect("equal")
        ax.set_xlim(-4.0, 3.5)
        ax.set_ylim(-2.4, 2.6)
    fig.suptitle(f"Goal-conditioned plans from held-out starts, head-wind channel β = {beta}. Same (start, goal) pairs, only the conditional path geometry differs.", fontsize=10, color=C_INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fm_planner_paths.png", dpi=160)
    plt.close(fig)


def fig_ratio_map(rows, cfg, beta, out):
    gx, gy = cfg["eval_grid"]
    goal_vals = sorted({r["goal_cond"] for r in rows})
    fig, axes = plt.subplots(len(goal_vals), 3, figsize=(13, 3.9 * len(goal_vals)), facecolor=C_SURF, squeeze=False)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("r", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    vmax = max(2.0, np.nanpercentile([r["ratio"] for r in rows], 97))
    for gi, gk in enumerate(goal_vals):
        for ax, name in zip(axes[gi], METHODS):
            style(ax)
            sel = sorted([r for r in rows if r["method"] == name and r["goal_cond"] == gk], key=lambda r: (r["sx"], r["sy"]))
            Z = np.array([r["ratio"] for r in sel]).reshape(gx, gy).T
            sx = sorted({r["sx"] for r in sel})
            sy = sorted({r["sy"] for r in sel})
            im = ax.imshow(Z, origin="lower", extent=(sx[0], sx[-1], sy[0], sy[-1]), cmap=cmap, vmin=1.0, vmax=vmax, aspect="auto")
            for r in sel:
                ax.text(r["sx"], r["sy"], f"{r['ratio']:.1f}", ha="center", va="center", fontsize=6.5, color=C_INK if r["ratio"] < 0.5 * (1 + vmax) else "#ffffff")
            ax.set_title(f"{LABEL[name]}" + (f"  to goal {gk} {tuple(cfg['goals'][gk])}" if gk >= 0 else ""), fontsize=9, color=C_INK, loc="left")
            ax.set_xlabel("start x1", color=C_INK2, fontsize=9)
            ax.set_ylabel("start x2", color=C_INK2, fontsize=9)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("travel time / optimal time", color=C_INK2, fontsize=9)
    cb.ax.tick_params(colors=C_INK2, labelsize=8)
    fig.suptitle(f"Optimality ratio of the generated plan per start, β = {beta}  (1.0 = time-optimal; channel core at x2 = 0)", fontsize=10, color=C_INK, x=0.01, ha="left")
    fig.savefig(out / "fm_planner_ratio_map.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--uncond", action="store_true", help="unconditional planner (no goal input)")
    ap.add_argument("--global-starts", action="store_true", help="starts uniform over the whole domain (t=0 field becomes a global feedback law)")
    args = ap.parse_args()
    cfg = dict(CFG)
    cfg["goal_cond"] = not args.uncond
    if args.global_starts:
        cfg["start_box"] = [[-3.5, 3.3], [-1.8, 1.8]]
        cfg["global_starts"] = True
    beta = args.beta
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / (f"fm_planner_beta{beta}" + ("_global" if args.global_starts else ""))
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    mkF, mkR, E, W1 = make_fields(beta, cfg["width"])
    ALL = ["pairs", "riemann", "train", "eval", "fig"]
    req = ALL if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    last = max(ALL.index(s) for s in req)

    if "pairs" in req:
        pairs = gen_pairs(mkF, cfg, rng)
        np.savez(out / "pairs.npz", **pairs)
    else:
        pairs = dict(np.load(out / "pairs.npz"))
    if last == 0:
        return
    if "riemann" in req:
        rie = riemann_paths(mkR, pairs, cfg)
        np.savez(out / "riemann.npz", **rie)
    else:
        rie = dict(np.load(out / "riemann.npz"))
    if last == 1:
        return
    models, train_loss = {}, {}
    if "train" in req:
        keep = branch_filter(pairs, rie, mkF, mkR, cfg)
        for name in METHODS:
            X, Y = build_dataset(pairs, rie, name, keep, cfg)
            t0 = time.time()
            models[name], train_loss[name] = train_model(X, Y, cfg, seed=1)
            torch.save(models[name].state_dict(), out / f"model_{name}.pt")
            print(f"trained {name} on {len(X)} points: final MSE {train_loss[name]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        (out / "train_loss.json").write_text(json.dumps(train_loss), encoding="utf-8")
    else:
        for name in METHODS:
            models[name] = MLP(d_in=5 if cfg["goal_cond"] else 3, d_out=2, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
            models[name].load_state_dict(torch.load(out / f"model_{name}.pt"))
            models[name].eval()
        train_loss = json.loads((out / "train_loss.json").read_text(encoding="utf-8"))
    if last == 2:
        return
    if "eval" in req:
        rows, starts, paths = evaluate(models, mkF, cfg, out)
    else:
        with open(out / "fm_planner_eval.csv", newline="") as f:
            rows = [{k: (v if k == "method" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
        for r in rows:
            r["goal_cond"] = int(r["goal_cond"])
            r["goal"] = int(r["goal"])
        z = np.load(out / "fm_planner_paths.npz")
        starts = z["starts"]
        paths = {}
        for k in z.files:
            if k.startswith("path_"):
                _, name, gk = k.split("_")
                paths[(name, None if gk == "None" else int(gk))] = z[k]
    summarise(rows, cfg, beta, train_loss, out)
    fig_paths(paths, starts, mkF, W1, cfg, beta, rows, out)
    fig_ratio_map(rows, cfg, beta, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
