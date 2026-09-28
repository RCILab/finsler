"""
Generalisation to unseen fields: a wind-strength-conditioned planner  v_theta(t, x | g, beta).

Training pairs draw beta ~ U(0.30, 0.85) per pair (each pair has its own Randers field beta * W1);
the model is then evaluated at beta in {0.5 (inside the training range), 0.9 and 0.95 (outside it)}.
Conditional paths: Euclidean straight lines / Riemannian a(x; beta) geodesics / Finsler geodesics, all
solved per pair with the 3-branch continuation BVP on batch-aligned parameter fields, then filtered by
grid Dijkstra distances computed on a coarse set of beta bins.

Usage: python experiments/fm_planner_betacond.py --stage all|pairs,riemann,train,eval
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

from finsler import channel_wind, zermelo_to_randers
from finsler.eikonal import GridFinslerDistance
from finsler.randers import RandersField
from fm_path_benefit import MLP
import fm_planner as fp

ROOT = pathlib.Path(__file__).resolve().parents[1]
METHODS = fp.METHODS
CFG = dict(fp.CFG)
CFG.update(dict(n_pairs=2400, beta_train=[0.30, 0.85], beta_test=[0.5, 0.9, 0.95], n_bins=12, branch_tol=1.08, log_steps=120))


class ParamRanders(RandersField):
    """Randers field W(x) = beta_i * W1(x) with a per-sample beta aligned with the leading batch axis."""

    def __init__(self, W1, beta, riemannian_part=False):
        self.W1 = W1
        self.beta = np.asarray(beta, float)
        self.riem = riemannian_part
        if riemannian_part:
            super().__init__(lambda x: np.zeros_like(np.asarray(x, float)), self._h_a)
        else:
            super().__init__(self._W)

    def _bcast(self, x):
        x = np.asarray(x, float)
        b = self.beta
        while b.ndim < x.ndim - 1:
            b = b[None]
        return b

    def _W(self, x):
        return self._bcast(x)[..., None] * self.W1(x)

    def _h_a(self, x):
        x = np.asarray(x, float)
        Wb = self._bcast(x)[..., None] * self.W1(x)
        A, _, _ = zermelo_to_randers(np.broadcast_to(np.eye(2), x.shape + (2,)), Wb)
        return A


def make_param(W1, beta, riem=False):
    return ParamRanders(W1, beta, riem)


def solve_bvps_param(W1, beta, x0, x1, n_steps, riem, waypoint_y=(1.4, -1.4)):
    """Same 3-branch scheme as fm_planner.solve_bvps but with per-pair beta (continuation scales beta)."""
    n = len(x0)
    cands = []
    F = make_param(W1, beta, riem)
    w, err = x1 - x0, None
    for s_ in np.linspace(0.15, 1.0, 7):
        w, err = make_param(W1, s_ * beta, riem).log_batched(x0, x1, n_steps=n_steps, iters=10, w_init=w)
    cands.append((w, err))
    for wy in waypoint_y:
        wp = np.stack([0.5 * (x0[:, 0] + x1[:, 0]), np.full(n, wy)], 1)
        w, err = x1 - x0, None
        w = wp - x0
        for s_ in np.linspace(0.34, 1.0, 3):
            w, err = make_param(W1, s_ * beta, riem).log_batched(x0, wp, n_steps=n_steps, iters=10, w_init=w)
        for s_ in np.linspace(0.25, 1.0, 4):
            w, err = F.log_batched(x0, wp + s_ * (x1 - wp), n_steps=n_steps, iters=10, w_init=w)
        cands.append((w, err))
    times = np.stack([np.where(e < 1e-6, F.F(x0, w), np.inf) for w, e in cands])
    best = np.argmin(times, 0)
    w = np.stack([cands[b][0][i] for i, b in enumerate(best)])
    T = times[best, np.arange(n)]
    return w, T, np.isfinite(T), best


def binned_grid_filter(W1, beta, x0, x1, T, cfg, riem):
    """Drop non-minimal branches using grid distances on beta bins."""
    edges = np.linspace(cfg["beta_train"][0], cfg["beta_train"][1], cfg["n_bins"] + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_id = np.clip(np.searchsorted(edges, beta) - 1, 0, cfg["n_bins"] - 1)
    ok = np.zeros(len(beta), bool)
    for k, bc in enumerate(centers):
        sel = bin_id == k
        if not sel.any():
            continue
        Fk = make_param(W1, np.full(sel.sum(), bc), riem)
        # the grid needs a scalar-beta field: wrap with a constant-beta RandersField
        Fs = RandersField(lambda x, bc=bc: bc * W1(x)) if not riem else fp.make_fields(bc, cfg["width"])[1](1.0)
        G = GridFinslerDistance(Fs, cfg["grid_xlim"], cfg["grid_ylim"], 0.05)
        d = G.dist(x0[sel], x1[sel])
        ok[sel] = T[sel] <= cfg["branch_tol"] * d
    return ok


def gen_pairs(W1, cfg, rng, out):
    goals = np.asarray(cfg["goals"], float)
    (xlo, xhi), (ylo, yhi) = cfg["start_box"]
    n, K = cfg["n_pairs"], cfg["K"]
    x0 = np.stack([rng.uniform(xlo, xhi, n), rng.uniform(ylo, yhi, n)], 1)
    which = np.arange(n) % len(goals)
    x1 = goals[which] + cfg["goal_std"] * rng.normal(size=(n, 2))
    beta = rng.uniform(cfg["beta_train"][0], cfg["beta_train"][1], n)
    t0 = time.time()
    w, T, ok, branch = solve_bvps_param(W1, beta, x0, x1, cfg["log_steps"], riem=False)
    okg = binned_grid_filter(W1, beta, x0, x1, T, cfg, riem=False)
    keep = ok & okg & (T <= cfg["T_max"])
    print(f"finsler BVPs (per-pair beta): converged {ok.sum()}/{n}, minimal {keep.sum()}, branches {np.bincount(branch[keep], minlength=3)}  ({time.time()-t0:.0f}s)", flush=True)
    F = make_param(W1, beta[keep])
    _, xs, vs = F.geodesic(x0[keep], w[keep], T=1.0, n_steps=K - 1)
    return dict(x0=x0[keep], x1=x1[keep], T=T[keep], which=which[keep], beta=beta[keep], t=np.linspace(0, 1, K), XF=xs.transpose(1, 0, 2), UF=vs.transpose(1, 0, 2))


def riemann_paths(W1, pairs, cfg):
    x0, x1, beta = pairs["x0"], pairs["x1"], pairs["beta"]
    t0 = time.time()
    w, T, ok, branch = solve_bvps_param(W1, beta, x0, x1, cfg["log_steps"], riem=True)
    okg = binned_grid_filter(W1, beta, x0, x1, T, cfg, riem=True)
    keep = ok & okg
    print(f"riemannian BVPs (per-pair beta): converged {ok.sum()}/{len(ok)}, minimal {keep.sum()}  ({time.time()-t0:.0f}s)", flush=True)
    R = make_param(W1, beta[keep], riem=True)
    _, xs, vs = R.geodesic(x0[keep], w[keep], T=1.0, n_steps=cfg["K"] - 1)
    return dict(ok=keep, XR=xs.transpose(1, 0, 2), UR=vs.transpose(1, 0, 2))


def build_dataset(pairs, rie, method, cfg):
    goals = np.asarray(cfg["goals"], float)
    ok = rie["ok"]
    t = pairs["t"]
    K = len(t)
    if method == "finsler":
        X, U = pairs["XF"][ok], pairs["UF"][ok]
    elif method == "riemann":
        X, U = rie["XR"], rie["UR"]
    else:
        x0, x1 = pairs["x0"][ok], pairs["x1"][ok]
        X = x0[:, None, :] + t[None, :, None] * (x1 - x0)[:, None, :]
        U = np.repeat((x1 - x0)[:, None, :], K, 1)
    n = len(X)
    cols = [X.reshape(-1, 2), np.tile(t, n)[:, None], np.repeat(goals[pairs["which"][ok]], K, 0), np.repeat(pairs["beta"][ok][:, None], K, 0)]
    return np.concatenate(cols, 1), U.reshape(-1, 2)


def integrate(model, starts, n_steps, goal, beta):
    x = starts.copy()
    path = [x.copy()]
    dt = 1.0 / n_steps
    extra = np.concatenate([np.repeat(np.asarray(goal, float)[None], len(x), 0), np.full((len(x), 1), beta)], 1)

    def f(t, x):
        with torch.no_grad():
            return model(torch.tensor(np.concatenate([x, np.full((len(x), 1), t), extra], 1), dtype=torch.float32)).numpy().astype(float)

    for s in range(n_steps):
        t = s * dt
        k1 = f(t, x)
        k2 = f(t + dt / 2, x + dt / 2 * k1)
        k3 = f(t + dt / 2, x + dt / 2 * k2)
        k4 = f(t + dt, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        path.append(x.copy())
    return np.array(path)


def evaluate(models, W1, cfg, out):
    goals = np.asarray(cfg["goals"], float)
    (xlo, xhi), (ylo, yhi) = cfg["start_box"]
    gx, gy = cfg["eval_grid"]
    starts = np.stack(np.meshgrid(np.linspace(xlo + 0.2, xhi - 0.2, gx), np.linspace(ylo + 0.2, yhi - 0.2, gy), indexing="ij"), -1).reshape(-1, 2)
    rows = []
    for beta in cfg["beta_test"]:
        F = RandersField(lambda x, b=beta: b * W1(x))
        G = GridFinslerDistance(F, cfg["grid_xlim"], cfg["grid_ylim"], cfg["grid_h"])
        D_start = G.dist_from(starts)
        Dg = [G.dist_to_target(g) for g in goals]
        for name in METHODS:
            for gi, g in enumerate(goals):
                path = integrate(models[name], starts, cfg["eval_steps"], g, beta)
                end = path[-1]
                resid = np.linalg.norm(end - g, axis=1)
                L = fp.finsler_length(F, path)
                d_end = D_start[np.arange(len(starts)), G.node_of(end)]
                d_goal = D_start[:, G.node_of(g[None])][:, 0]
                final = Dg[gi][G.node_of(end)]
                for j in range(len(starts)):
                    rows.append(dict(beta=beta, method=name, goal=gi, resid=resid[j], travel=L[j], ratio=L[j] / d_end[j], ratio_goal=(L[j] + final[j]) / d_goal[j], reach=int(resid[j] < cfg["goal_r"][0])))
            sel = [r for r in rows if r["beta"] == beta and r["method"] == name]
            print(f"  beta={beta} {name}: reach {np.mean([r['reach'] for r in sel]):.2f}, ratio {np.mean([r['ratio'] for r in sel]):.3f}, time-to-goal ratio {np.mean([r['ratio_goal'] for r in sel]):.3f}", flush=True)
    with open(out / "fm_planner_betacond_eval.csv", "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)
    lines = ["# beta-conditioned planner: unseen wind strengths", "", f"train beta ~ U{cfg['beta_train']}, test beta {cfg['beta_test']} (0.9 and 0.95 are outside the training range)", "",
             "| test beta | method | reach < %.2f | travel/optimal ± s.e. | worst | time-to-goal ratio ± s.e. |" % cfg["goal_r"][0], "|---:|---|---:|---:|---:|---:|"]
    for beta in cfg["beta_test"]:
        for name in METHODS:
            sel = [r for r in rows if r["beta"] == beta and r["method"] == name]
            ratio = np.array([r["ratio"] for r in sel])
            rg = np.array([r["ratio_goal"] for r in sel])
            lines.append(f"| {beta} | {name} | {np.mean([r['reach'] for r in sel]):.2f} | {ratio.mean():.3f} ± {ratio.std(ddof=1)/np.sqrt(len(ratio)):.3f} | {ratio.max():.2f} | {rg.mean():.3f} ± {rg.std(ddof=1)/np.sqrt(len(rg)):.3f} |")
    (out / "fm_planner_betacond_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    cfg = dict(CFG)
    out = ROOT / "results" / "fm_planner_betacond"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    W1 = channel_wind(1.0, cfg["width"])
    ALL = ["pairs", "riemann", "train", "eval"]
    req = ALL if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    last = max(ALL.index(s) for s in req)
    if "pairs" in req:
        pairs = gen_pairs(W1, cfg, rng, out)
        np.savez(out / "pairs.npz", **pairs)
    else:
        pairs = dict(np.load(out / "pairs.npz"))
    if last == 0:
        return
    if "riemann" in req:
        rie = riemann_paths(W1, pairs, cfg)
        np.savez(out / "riemann.npz", **rie)
    else:
        rie = dict(np.load(out / "riemann.npz"))
    if last == 1:
        return
    models = {}
    if "train" in req:
        losses = {}
        for name in METHODS:
            X, Y = build_dataset(pairs, rie, name, cfg)
            t0 = time.time()
            models[name], losses[name] = fp.train_model(X, Y, cfg, seed=1)
            torch.save(models[name].state_dict(), out / f"model_{name}.pt")
            print(f"trained {name} on {len(X)} points: MSE {losses[name]:.4f} ({time.time()-t0:.0f}s)", flush=True)
        (out / "train_loss.json").write_text(json.dumps(losses), encoding="utf-8")
    else:
        for name in METHODS:
            models[name] = MLP(d_in=6, d_out=2, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
            models[name].load_state_dict(torch.load(out / f"model_{name}.pt"))
            models[name].eval()
    if last == 2:
        return
    evaluate(models, W1, cfg, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
