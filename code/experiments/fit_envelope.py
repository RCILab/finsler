"""
Learning the velocity envelope (= the Finsler metric) from rollouts, and planning with it.

Ground truth: an asymmetric p-norm box (p = 4) on body velocities, outside the SumRanders family:
    F_true(xi) = ( sum_i ( max(xi_i,0)/s_i^+ + max(-xi_i,0)/s_i^- )^p )^(1/p),
with max speeds forward 1.2, backward 0.5, sideways 0.4, yaw 1.2 (quadruped-like, boxy with rounded corners).
Data: commanded body velocities u in a box; the robot achieves  xi = u / max(1, F_true(u)) + noise,
      i.e. commands outside the envelope saturate onto its boundary.  Saturated samples lie on the
      indicatrix {F = 1}; unsaturated ones lie inside.
Fit:  (a) Randers  F = |L xi| + b.xi  with |b|_A < 1 enforced by b = L^T c, |c| < 1;
      (b) two-term SumRanders.
      Loss: (F(xi) - 1)^2 on saturated samples + relu(F(xi) - 1)^2 on interior samples.
Eval: max speeds in 6 directions, indicatrix Hausdorff distance, and planning quality: time-optimal
      SE(2) paths planned with the fitted metric (grid Dijkstra), executed under the TRUE envelope,
      travel time / true optimal time.

Usage: python experiments/fit_envelope.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra

from finsler.eikonal import GridSE2Distance
from finsler.se2 import SE2Finsler, quadruped_envelope
from finsler.se2_randers import SE2Randers

torch.set_default_dtype(torch.float64)
C_INK, C_INK2, C_GRID, C_SURF, C_MUTED = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb", "#898781"
COL = {"true": "#0b0b0b", "randers": "#2a78d6", "sum2": "#1baf7a", "sum3": "#1baf7a", "psum": "#1baf7a", "symmetric": "#eb6834"}
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "fit_envelope"
OUT.mkdir(parents=True, exist_ok=True)


class PNormEnvelope:
    """Asymmetric p-norm Minkowski norm on body velocities (numpy)."""

    def __init__(self, s_plus, s_minus, p=4.0):
        self.sp, self.sm, self.p = np.asarray(s_plus, float), np.asarray(s_minus, float), p

    def F_body(self, xi):
        xi = np.asarray(xi, float)
        z = np.maximum(xi, 0) / self.sp + np.maximum(-xi, 0) / self.sm
        return (z**self.p).sum(-1) ** (1.0 / self.p)


class PNormSE2:
    """numpy-facing left-invariant SE(2) metric from a body-frame p-norm envelope (for grid edge costs)."""

    def __init__(self, env: PNormEnvelope):
        self.env = env
        self.dim = 3

    def F(self, q, qdot):
        q = np.asarray(q, float)
        qdot = np.asarray(qdot, float)
        c, s_ = np.cos(q[..., 2]), np.sin(q[..., 2])
        xi = np.stack([c * qdot[..., 0] + s_ * qdot[..., 1], -s_ * qdot[..., 0] + c * qdot[..., 1], qdot[..., 2]], -1)
        return self.env.F_body(xi)


class TorchSE2Wrapper:
    """numpy-facing F(q, qdot) for a torch SE2Finsler (used for grid edge costs)."""

    def __init__(self, fld: SE2Finsler):
        self.fld = fld
        self.dim = 3

    def F(self, q, qdot):
        with torch.no_grad():
            return self.fld.F(torch.as_tensor(np.asarray(q, float)), torch.as_tensor(np.asarray(qdot, float))).numpy()


class BodyFit(torch.nn.Module):
    """Body-frame SumRanders with K terms; b = rho * L_1^T c with |c| < 1 keeps F > 0."""

    def __init__(self, K, init_scale=1.0):
        super().__init__()
        self.K = K
        self.Lraw = torch.nn.Parameter(torch.stack([torch.eye(3) * init_scale / K + 0.01 * torch.randn(3, 3) for _ in range(K)]))
        self.craw = torch.nn.Parameter(torch.zeros(3))

    def Ls(self):
        return torch.tril(self.Lraw)

    def b(self):
        L1 = self.Ls()[0]
        c = self.craw / (1.0 + torch.linalg.norm(self.craw))          # |c| < 1
        return L1.T @ c

    def F(self, xi):
        L = self.Ls()
        alpha = torch.linalg.norm(torch.einsum("kij,nj->nki", L, xi), dim=-1).sum(-1)
        return alpha + xi @ self.b()

    def export(self):
        return self.Ls().detach().numpy(), self.b().detach().numpy()


class BodyFitP(torch.nn.Module):
    """p-sum of ellipsoid norms (box-like for large p) + small regularising ellipsoid + linear term:
         F(xi) = (sum_k ||L_k xi||^p)^(1/p) + eps ||L_0 xi|| + b . xi,   b = L_0^T c, |c| < 1  (times eps to keep F > 0 safe)
    p = 1 + softplus(praw) is learned.  The eps-ellipsoid keeps the norm strongly convex."""

    def __init__(self, K=3, eps=0.15, init_scale=1.0):
        super().__init__()
        self.K, self.eps = K, eps
        base = torch.stack([torch.zeros(3, 3) for _ in range(K)])
        for k in range(K):
            base[k, k % 3, k % 3] = init_scale
        self.Lraw = torch.nn.Parameter(base + 0.02 * torch.randn(K, 3, 3))
        self.L0raw = torch.nn.Parameter(torch.eye(3) * init_scale)
        self.craw = torch.nn.Parameter(torch.zeros(3))
        self.praw = torch.nn.Parameter(torch.tensor(2.0))

    def p(self):
        return 1.0 + torch.nn.functional.softplus(self.praw)

    def b(self):
        c = self.craw / (1.0 + torch.linalg.norm(self.craw))
        return self.eps * torch.tril(self.L0raw).T @ c + (torch.tril(self.Lraw)[0].T @ c) * 0.0

    def F(self, xi):
        L = torch.tril(self.Lraw)
        norms = torch.linalg.norm(torch.einsum("kij,nj->nki", L, xi), dim=-1) + 1e-12
        pp = self.p()
        psum = (norms**pp).sum(-1) ** (1.0 / pp)
        reg = self.eps * torch.linalg.norm(xi @ torch.tril(self.L0raw).T, dim=-1)
        return psum + reg + xi @ self.b()


class TorchBodyWrapper:
    """numpy-facing left-invariant SE(2) metric from any torch body-frame F(xi) (for grid edge costs)."""

    def __init__(self, Fbody):
        self.Fbody = Fbody
        self.dim = 3

    def F(self, q, qdot):
        q = np.asarray(q, float)
        qdot = np.asarray(qdot, float)
        c, s_ = np.cos(q[..., 2]), np.sin(q[..., 2])
        xi = np.stack([c * qdot[..., 0] + s_ * qdot[..., 1], -s_ * qdot[..., 0] + c * qdot[..., 1], qdot[..., 2]], -1)
        with torch.no_grad():
            return self.Fbody(torch.as_tensor(xi.reshape(-1, 3))).numpy().reshape(xi.shape[:-1])


def speeds_of(Ffun):
    dirs = {"forward": [1, 0, 0], "backward": [-1, 0, 0], "left": [0, 1, 0], "right": [0, -1, 0], "yaw+": [0, 0, 1], "yaw-": [0, 0, -1]}
    return {k: float(1.0 / Ffun(np.array([v], float))[0]) for k, v in dirs.items()}


def indicatrix(Ffun, n=4000, rng=None):
    rng = rng or np.random.default_rng(0)
    e = rng.normal(size=(n, 3))
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e / Ffun(e)[:, None]


def hausdorff(P, Q):
    d = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)
    return max(d.min(1).max(), d.min(0).max())


def path_from_predecessors(pred, src, dst):
    path = [dst]
    while path[-1] != src:
        p = pred[path[-1]]
        if p < 0:
            return None
        path.append(p)
    return path[::-1]


def main():
    rng = np.random.default_rng(0)
    # ---- ground truth: asymmetric p-norm box --------------------------------------------------------------
    env = PNormEnvelope(s_plus=[1.2, 0.4, 1.2], s_minus=[0.5, 0.4, 1.2], p=4.0)
    F_true = env.F_body
    sp_true = speeds_of(F_true)
    print("true envelope (p-norm box, p=4) speeds:", {k: round(v, 3) for k, v in sp_true.items()})

    # ---- rollouts: commanded -> achieved body velocities -----------------------------------------------------
    n_cmd = 6000
    u = rng.uniform(-1, 1, (n_cmd, 3)) * np.array([1.8, 0.9, 2.0])
    Fu = F_true(u)
    sat = Fu > 1.0
    xi = u / np.maximum(Fu, 1.0)[:, None] + 0.02 * rng.normal(size=u.shape)
    print(f"rollouts: {n_cmd} commands, {sat.sum()} saturated (on the envelope boundary), noise std 0.02")

    # ---- fits ----------------------------------------------------------------------------------------
    fits = {}
    for name, K in [("randers", 1), ("sum2", 2), ("sum3", 3)]:
        torch.manual_seed(0)
        model = BodyFit(K, init_scale=1.5)
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        xi_t = torch.as_tensor(xi)
        sat_t = torch.as_tensor(sat)
        t0 = time.time()
        for it in range(3000):
            Fx = model.F(xi_t)
            loss = torch.mean((Fx[sat_t] - 1.0) ** 2) + torch.mean(torch.relu(Fx[~sat_t] - 1.0) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        Ls_fit, b_fit = model.export()
        fits[name] = (Ls_fit, b_fit, float(loss.detach()))
        print(f"fit {name} (K={K}): loss {float(loss):.2e}  ({time.time()-t0:.0f}s)")

    torch.manual_seed(0)
    pmodel = BodyFitP(K=3, eps=0.15, init_scale=1.5)
    opt = torch.optim.Adam(pmodel.parameters(), lr=0.02)
    t0 = time.time()
    for it in range(4000):
        Fx = pmodel.F(torch.as_tensor(xi))
        loss = torch.mean((Fx[torch.as_tensor(sat)] - 1.0) ** 2) + torch.mean(torch.relu(Fx[~torch.as_tensor(sat)] - 1.0) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"fit psum (K=3, learned p = {float(pmodel.p()):.2f}): loss {float(loss):.2e}  ({time.time()-t0:.0f}s)")
    Fpsum = lambda xi_: pmodel.F(torch.as_tensor(np.asarray(xi_, float).reshape(-1, 3))).detach().numpy().reshape(np.asarray(xi_).shape[:-1])

    # ---- metric-level comparison (radial error of the indicatrix on shared directions) --------------------
    models = {"randers": SE2Finsler(fits["randers"][0], fits["randers"][1]), "sum2": SE2Finsler(fits["sum2"][0], fits["sum2"][1]),
              "sum3": SE2Finsler(fits["sum3"][0], fits["sum3"][1]), "symmetric": SE2Finsler(fits["randers"][0], np.zeros(3))}
    Ffun = {"true": F_true, "psum": Fpsum}
    Ffun.update({k: (lambda xi, m=m: m.F_body(torch.as_tensor(np.asarray(xi, float))).numpy()) for k, m in models.items()})
    e = rng.normal(size=(4000, 3))
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    r_true = 1.0 / Ffun["true"](e)
    lines = ["# Learning the velocity envelope from rollouts", "", f"true envelope: asymmetric p-norm box (p = 4), speeds {sp_true}; {n_cmd} commands, {sat.sum()} saturated; noise 0.02", "",
             "| model | forward | backward | left | yaw | radial error of indicatrix: mean / max (rel.) |", "|---|---:|---:|---:|---:|---:|"]
    for k in ["true", "randers", "sum2", "sum3", "psum", "symmetric"]:
        sp = speeds_of(Ffun[k])
        r = 1.0 / Ffun[k](e)
        rel = np.abs(r - r_true) / r_true
        lines.append(f"| {k} | {sp['forward']:.3f} | {sp['backward']:.3f} | {sp['left']:.3f} | {sp['yaw+']:.3f} | {rel.mean()*100:.2f}% / {rel.max()*100:.1f}% |")
    print(chr(10).join(lines[4:]))

    # ---- planning with the fitted metric, executed under the true envelope ------------------------------------
    lim, h, nth = (-3.2, 3.2), 0.1, 36
    goal = np.zeros(3)
    grids = {}
    t0 = time.time()
    grids["true"] = GridSE2Distance(PNormSE2(env), lim, lim, h, nth)
    print(f"grid for true: {time.time()-t0:.0f}s", flush=True)
    for k in ["randers", "sum2", "sum3", "symmetric"]:
        t0 = time.time()
        grids[k] = GridSE2Distance(TorchSE2Wrapper(models[k]), lim, lim, h, nth)
        print(f"grid for {k}: {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    grids["psum"] = GridSE2Distance(TorchBodyWrapper(pmodel.F), lim, lim, h, nth)
    print(f"grid for psum: {time.time()-t0:.0f}s", flush=True)
    Gt = grids["true"]
    starts = np.array([[x, y, th] for x in np.linspace(-2, 2, 4) for y in np.linspace(-2, 2, 4) for th in np.linspace(-np.pi, np.pi, 6, endpoint=False)])
    starts = starts[np.linalg.norm(starts[:, :2], axis=1) > 0.3]
    gnode = Gt.node_of(goal[None])[0]
    Dtrue_to_goal = Gt.dist_to_target(goal)
    res_lines = ["", "Planning with the fitted metric, executed under the true envelope (SE(2) grid Dijkstra, %d starts):" % len(starts), "",
                 "| planning metric | mean true travel / true optimal | median | worst |", "|---|---:|---:|---:|"]
    ratios = {}
    for k in ["true", "randers", "sum2", "sum3", "psum", "symmetric"]:
        G = grids[k]
        snodes = G.node_of(starts)
        D, pred = dijkstra(G.graph, directed=True, indices=snodes, return_predecessors=True)
        r = []
        for i, sn in enumerate(snodes):
            path = path_from_predecessors(pred[i], sn, gnode)
            if path is None:
                continue
            # true cost of this node path
            cost = 0.0
            for a, b_ in zip(path[:-1], path[1:]):
                cost += Gt.graph[a, b_] if Gt.graph[a, b_] != 0 else np.inf
            r.append(cost / Dtrue_to_goal[sn])
        r = np.array(r)
        ratios[k] = r
        res_lines.append(f"| {k} | {r.mean():.3f} | {np.median(r):.3f} | {r.max():.3f} |")
    print("\n".join(res_lines))
    (OUT / "fit_envelope_summary.md").write_text("\n".join(lines + res_lines) + "\n", encoding="utf-8")
    json.dump({k: dict(Ls=v[0].tolist(), b=v[1].tolist(), loss=v[2]) for k, v in fits.items()}, open(OUT / "fits.json", "w"), indent=1)

    # ---- figure: indicatrix slices (v_x, v_y) at omega = 0 and (v_x, omega) at v_y = 0 --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), facecolor=C_SURF)
    th = np.linspace(0, 2 * np.pi, 400)
    for ax, (i, j, lab_i, lab_j) in zip(axes, [(0, 1, "v_x [m/s]", "v_y [m/s]"), (0, 2, "v_x [m/s]", "ω [rad/s]")]):
        ax.set_facecolor(C_SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=C_INK2, labelsize=9)
        ax.grid(True, color=C_GRID, lw=0.8)
        ax.set_axisbelow(True)
        e2 = np.zeros((len(th), 3))
        e2[:, i], e2[:, j] = np.cos(th), np.sin(th)
        m = np.zeros(3, bool)
        m[[i, j]] = True
        pts = xi[sat][:, [i, j]][np.abs(xi[sat][:, [k for k in range(3) if not m[k]][0]]) < 0.08]
        ax.scatter(pts[:, 0], pts[:, 1], s=6, color=C_MUTED, alpha=0.5, lw=0, label="saturated rollout samples (slice)")
        for k, ls in [("true", "-"), ("randers", "--"), ("psum", "-"), ("symmetric", ":")]:
            r = 1.0 / Ffun[k](e2)
            ax.plot(r * e2[:, i], r * e2[:, j], ls, color=COL[k], lw=2 if k != "true" else 2.6, label={"true": "true envelope (p-norm box)", "randers": "fitted Randers (1 term)", "psum": "fitted p-sum (3 terms + reg.)", "symmetric": "symmetric part (RFMP-style)"}[k])
        ax.set_xlabel(lab_i, color=C_INK2, fontsize=9)
        ax.set_ylabel(lab_j, color=C_INK2, fontsize=9)
        ax.set_aspect("equal")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=C_INK2, loc="upper left")
    fig.suptitle("Body-frame velocity envelope learned from saturated rollouts: slices of the indicatrix {F = 1}", fontsize=10, color=C_INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fit_envelope_indicatrix.png", dpi=160)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
