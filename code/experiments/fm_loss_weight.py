"""
Oracle demonstration (no neural network): the loss weight in conditional flow matching must not
depend on the conditioning variable x1.

Setting.  Constant Randers metric on R^2, F(v) = |v| + beta * v_1.  Geodesics of a constant
(Minkowski) metric are straight lines, so the Finsler conditional path x_t = (1-t) x0 + t x1 is
IDENTICAL to Euclidean OT-CFM.  The *only* thing that differs between the two rows below is the
weight matrix in the regression loss

    L(v) = E_{t, x0, x1} [ (v(t, x_t) - u_t(x_t | x1))^T  M  (v(t, x_t) - u_t(x_t | x1)) ].

* M = I (or any M(t, x)):   argmin = E[u_t(x_t|x1) | x_t = x]  = the marginal field -> pushes p0 to p1.
* M = g_{u_t(x|x1)}       :   argmin = (E[g])^{-1} E[g u]      != marginal field  -> wrong terminal law.

Both minimisers are computed exactly (Monte Carlo over the data for the posterior x1 | x_t = x),
then integrated from the prior.  Energy distance to fresh data and per-mode mass quantify the damage.

Usage:  python experiments/fm_loss_weight.py [--out results]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from finsler import RandersField, randers_to_zermelo

C_GEN, C_DATA, C_INK, C_INK2, C_GRID, C_SURF = "#2a78d6", "#898781", "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
MEANS = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.2]])
STD = 0.35


def const_randers(beta, direction=(1.0, 0.0)):
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    h, W, _ = randers_to_zermelo(np.eye(2), beta * d)
    return RandersField(lambda x: np.broadcast_to(W, np.shape(x)), lambda x: np.broadcast_to(h, np.shape(x) + (2,)))


def sample_data(n, rng):
    lab = rng.integers(0, 3, n)
    return MEANS[lab] + STD * rng.normal(size=(n, 2))


def posterior_weights(t, x, data):
    """w_j(t, x) ∝ N(x; t x1_j, (1-t)^2 I)  for x0 ~ N(0, I), x_t = (1-t) x0 + t x1."""
    diff = x[:, None, :] - t * data[None, :, :]
    logw = -0.5 * np.sum(diff**2, -1) / (1.0 - t) ** 2
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw)
    return w / w.sum(1, keepdims=True)


def oracle_field(t, x, data, field, weighting):
    w = posterior_weights(t, x, data)                       # (N, M)
    u = (data[None, :, :] - x[:, None, :]) / (1.0 - t)       # (N, M, 2) conditional targets
    if weighting == "marginal":
        return np.einsum("nm,nmd->nd", w, u)
    # Finsler-weighted regression minimiser: (sum_j w_j g_{u_j})^{-1} sum_j w_j g_{u_j} u_j, g_u u = L(u)
    xz = np.zeros_like(u)
    g = field.fundamental_tensor(xz, u)                       # (N, M, 2, 2)
    Lu = field.legendre(xz, u)                                # (N, M, 2)
    G = np.einsum("nm,nmij->nij", w, g)
    rhs = np.einsum("nm,nmi->ni", w, Lu)
    return np.linalg.solve(G, rhs[..., None])[..., 0]


def integrate(x0, data, field, weighting, n_steps=100, t_end=0.99):
    x = x0.copy()
    ts = np.linspace(0.0, t_end, n_steps + 1)
    for k in range(n_steps):
        t, dt = ts[k], ts[k + 1] - ts[k]
        f = lambda tt, xx: oracle_field(tt, xx, data, field, weighting)
        k1 = f(t, x)
        k2 = f(t + dt / 2, x + dt / 2 * k1)
        k3 = f(t + dt / 2, x + dt / 2 * k2)
        k4 = f(t + dt, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def energy_distance(X, Y):
    def md(P, Q):
        return np.mean(np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1))

    return 2 * md(X, Y) - md(X, X) - md(Y, Y)


def mode_mass(X):
    lab = np.argmin(np.linalg.norm(X[:, None, :] - MEANS[None], axis=-1), 1)
    return np.bincount(lab, minlength=3) / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--n", type=int, default=1200)
    args = ap.parse_args()
    out = pathlib.Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    data = sample_data(args.n, rng)          # for the oracle posterior
    fresh = sample_data(args.n, rng)         # held-out reference
    x0 = rng.normal(size=(args.n, 2))
    betas = [0.0, 0.3, 0.6, 0.9]
    weightings = ["marginal", "finsler"]
    titles = {"marginal": "weight M = I  (any M(t,x) gives the same minimiser)", "finsler": "weight M = g_{u_t(x|x1)}  (depends on x1)"}

    res = {}
    for beta in betas:
        field = const_randers(beta)
        for wgt in weightings:
            X = integrate(x0, data, field, wgt)
            res[(beta, wgt)] = dict(X=X, ed=energy_distance(X, fresh), mass=mode_mass(X))
            print(f"beta={beta:.1f} {wgt:9s}  energy distance={res[(beta, wgt)]['ed']:.4f}  mode mass={np.round(res[(beta, wgt)]['mass'], 3)}", flush=True)
    ed_ref = energy_distance(sample_data(args.n, rng), fresh)
    print(f"reference: energy distance between two fresh data samples = {ed_ref:.4f}, true mode mass = [0.333 0.333 0.333]")

    # ---- figure ---------------------------------------------------------------------------------
    fig, axes = plt.subplots(len(weightings), len(betas), figsize=(3.1 * len(betas), 3.3 * len(weightings)), facecolor=C_SURF, squeeze=False)
    for i, wgt in enumerate(weightings):
        for j, beta in enumerate(betas):
            ax = axes[i, j]
            ax.set_facecolor(C_SURF)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            r = res[(beta, wgt)]
            ax.scatter(fresh[:, 0], fresh[:, 1], s=6, color=C_DATA, alpha=0.35, linewidths=0, label="data")
            ax.scatter(r["X"][:, 0], r["X"][:, 1], s=6, color=C_GEN, alpha=0.6, linewidths=0, label="generated")
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-1.8, 3.8)
            ax.set_aspect("equal")
            m = r["mass"]
            ax.set_title(f"β = {beta}   ED = {r['ed']:.3f}\nmass L/R/T = {m[0]:.2f} / {m[1]:.2f} / {m[2]:.2f}", fontsize=8.5, color=C_INK2, loc="left")
            if j == 0:
                ax.text(-0.02, 0.5, titles[wgt], transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=8.5, color=C_INK)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=C_INK2, markerscale=2)
    fig.suptitle("Same conditional paths, different loss weight: only the x1-dependent weight g_{u_t} distorts p1  (F = |v| + β v₁, slow direction = +x)", fontsize=9.5, color=C_INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0.03, 0, 1, 0.95))
    fig.savefig(out / "fm_loss_weight.png", dpi=160)
    plt.close(fig)

    lines = ["| β | weight | energy distance to data | mode mass L / R / T |", "|---:|---|---:|---|"]
    for beta in betas:
        for wgt in weightings:
            r = res[(beta, wgt)]
            lines.append(f"| {beta} | {wgt} | {r['ed']:.4f} | {r['mass'][0]:.3f} / {r['mass'][1]:.3f} / {r['mass'][2]:.3f} |")
    lines.append(f"| – | fresh data vs fresh data | {ed_ref:.4f} | 0.333 / 0.333 / 0.333 |")
    (out / "fm_loss_weight.md").write_text("# FM loss-weight oracle experiment\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
