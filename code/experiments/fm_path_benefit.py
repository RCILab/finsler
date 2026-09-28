"""
Does the Finsler conditional path buy anything for a streaming-flow-type policy?

Setting (Streaming-Flow-Policy style: flow time = physical time of an action chunk)
-----------------------------------------------------------------------------------
* Randers head-wind channel, envelope F(x, v_phys) <= 1.
* Demos = time-optimal trajectories to a goal (unit-speed Finsler geodesics), cut to chunks of T_c s.
* Policy v_theta(a, t | c) with c = demo start; conditional target
        v(a, t | xi) = xi_dot(t) + k * log_a(xi(t))
  with three log maps:  Euclidean  y - a  |  Riemannian part a(x)  |  Finsler (forward geodesic).
  In a constant metric all three coincide; differences come only from geodesic curvature.
* Test: start off the demo (|delta| = m), integrate the learned field, measure
     (free)      envelope violation of the requested velocity,
     (projected) tracking error when the executed velocity is projected onto the envelope.
  Oracle rows integrate the analytic target fields (no network) to separate learning from geometry.

Stages are cached under results/fm_path_beta<beta>/ :  demos -> data -> train -> eval -> fig
Usage:  python experiments/fm_path_benefit.py [--beta 0.7] [--stage all|demos|data|train|eval|fig] [--no-oracle]
"""
from __future__ import annotations

import argparse
import csv
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

from finsler import channel_wind, euclidean_field, finsler_field, riemannian_part

C = {"euclid": "#2a78d6", "riemann": "#eb6834", "finsler": "#1baf7a"}
LABEL = {"euclid": "Euclidean log  (y − a)", "riemann": "Riemannian log  a(x)", "finsler": "Finsler log  (forward geodesic)"}
C_MUTED, C_GRID, C_INK, C_INK2, C_SURF = "#898781", "#e1e0d9", "#0b0b0b", "#52514e", "#fcfcfb"
METHODS = ["euclid", "riemann", "finsler"]

CFG = dict(
    width=0.8, goal=(2.5, 0.0), T_max=12.0, T_c=2.0, n_demo=48, n_pts=101,
    n_t=32, n_p=10, sigma0=0.35, sigma_min=0.05, k=4.0,
    hidden=256, n_layers=3, train_steps=4000, batch=1024, lr=1e-3,
    eval_m=[0.1, 0.25, 0.5, 0.8], eval_dirs=8, eval_steps=100, oracle_dirs=4, oracle_demos=24, oracle_steps=40,
    log_steps=50, cont_stages=[0.33, 0.66, 1.0],
)


# --------------------------------------------------------------------------------------------------
def make_fields(beta, width):
    W1 = channel_wind(1.0, width)
    mkF = lambda s=1.0: finsler_field(lambda x, s=s: (s * beta) * W1(x))
    mkR = lambda s=1.0: riemannian_part(lambda x, s=s: (s * beta) * W1(x))
    return mkF, mkR, euclidean_field(), W1


def log_continuation(mk, a, y, stages, n_steps, w_init=None, iters=8):
    """log_a(y) for the field mk(1.0) by continuation in the wind strength (mk(s), s in stages)."""
    w = (y - a) if w_init is None else w_init
    if w_init is not None:
        return mk(1.0).log_batched(a, y, n_steps=n_steps, iters=iters, w_init=w)
    err = None
    for s in stages:
        w, err = mk(s).log_batched(a, y, n_steps=n_steps, iters=iters, w_init=w)
    return w, err


# --------------------------------------------------------------------------------------------------
def solve_demo_bvps(mkF, starts, goal, waypoint_y=(1.4, -1.4), n_steps=150):
    """Time-optimal initial velocities w (exp_{start}(w) = goal, time = F(start, w)) by comparing three
    candidate branches: beta-continuation from the straight line, and detours via a waypoint above / below
    the channel (target continuation, warm-started).  Returns (w, time, ok)."""
    F = mkF(1.0)
    n = len(starts)
    g = np.repeat(goal[None], n, 0)
    cands = []
    w, err = log_continuation(mkF, starts, g, list(np.linspace(0.1, 1.0, 10)), n_steps, iters=10)
    cands.append((w, err))
    for wy in waypoint_y:
        wp = np.stack([0.5 * (starts[:, 0] + goal[0]), np.full(n, wy)], 1)
        w, err = log_continuation(mkF, starts, wp, list(np.linspace(0.25, 1.0, 4)), n_steps, iters=10)
        for s_ in np.linspace(1.0 / 6, 1.0, 6):
            w, err = F.log_batched(starts, wp + s_ * (g - wp), n_steps=n_steps, iters=10, w_init=w)
        cands.append((w, err))
    times = np.stack([np.where(e < 1e-6, F.F(starts, w), np.inf) for w, e in cands])   # (3, n)
    best = np.argmin(times, 0)
    w = np.stack([cands[b][0][i] for i, b in enumerate(best)])
    T = times[best, np.arange(n)]
    return w, T, np.isfinite(T), best


def gen_demos(mkF, cfg, rng):
    """Time-optimal demos from random starts west of the goal, solved as BVPs (min over three branches)."""
    F = mkF(1.0)
    goal = np.asarray(cfg["goal"], float)
    n_try = int(cfg["n_demo"] * 2)
    starts = np.stack([rng.uniform(-3.5, -1.2, n_try), rng.uniform(-1.5, 1.5, n_try)], 1)
    t0 = time.time()
    w, T, ok, branch = solve_demo_bvps(mkF, starts, goal)
    keep = ok & (T >= cfg["T_c"] + 0.5) & (T <= cfg["T_max"])       # T > T_max = the detour branches failed and the slow crawl was picked
    print(f"demo BVPs: {ok.sum()}/{n_try} converged, {keep.sum()} kept ({cfg['T_c']+0.5} <= T <= {cfg['T_max']}); branches direct/up/down = "
          f"{(branch[keep]==0).sum()}/{(branch[keep]==1).sum()}/{(branch[keep]==2).sum()}; T in [{T[keep].min():.2f}, {T[keep].max():.2f}] s  ({time.time()-t0:.0f}s)")
    idx = np.where(keep)[0][: cfg["n_demo"]]
    n = len(idx)
    starts, w, T = starts[idx], w[idx], T[idx]
    # unit-speed geodesic for the chunk duration T_c (F(x, xdot) = 1 by construction)
    v0 = w / T[:, None]
    ts, xs, vs = F.geodesic(starts, v0, T=cfg["T_c"], n_steps=cfg["n_pts"] - 1)
    xi = np.transpose(xs, (1, 0, 2))
    xid = np.transpose(vs, (1, 0, 2))
    t_grid = ts / cfg["T_c"]
    speed = F.F(xi.reshape(-1, 2), xid.reshape(-1, 2))
    print(f"demos: {n}, chunk {cfg['T_c']} s, F(xi, xi_dot_phys) in [{speed.min():.4f}, {speed.max():.4f}] (should be 1)")
    return dict(t=t_grid, xi=xi, xid_phys=xid, xid_flow=xid * cfg["T_c"], start=xi[:, 0].copy(), T_total=T)


def demo_at(demos, idx, t):
    """xi_i(t), xi_dot_flow_i(t) for arrays idx (N,), t (N,)."""
    tg = demos["t"]
    j = np.clip(np.searchsorted(tg, t) - 1, 0, len(tg) - 2)
    lam = ((t - tg[j]) / (tg[j + 1] - tg[j]))[:, None]
    xi = demos["xi"][idx, j] * (1 - lam) + demos["xi"][idx, j + 1] * lam
    xid = demos["xid_flow"][idx, j] * (1 - lam) + demos["xid_flow"][idx, j + 1] * lam
    return xi, xid


# --------------------------------------------------------------------------------------------------
def gen_data(mkF, mkR, demos, cfg, rng):
    F = mkF(1.0)
    n, nt, npert = cfg["n_demo"], cfg["n_t"], cfg["n_p"]
    idx = np.repeat(np.arange(n), nt * npert)
    t = np.tile(np.repeat(np.linspace(0.0, 0.97, nt), npert), n)
    y, yd = demo_at(demos, idx, t)
    sig = (cfg["sigma0"] * (1 - t) + cfg["sigma_min"])[:, None]
    w = rng.normal(size=(len(t), 2)) * sig
    t0 = time.time()
    a, vend = F.reverse().exp(y, w, n_steps=80)                            # perturbed points; exact Finsler log = -vend
    logF = -vend
    logE = y - a
    print(f"data: {len(t)} points, |a - y| mean {np.linalg.norm(a-y,axis=1).mean():.3f} max {np.linalg.norm(a-y,axis=1).max():.3f}  ({time.time()-t0:.1f}s)")
    t0 = time.time()
    logR, err = log_continuation(mkR, a, y, cfg["cont_stages"], cfg["log_steps"])
    ok = err < 1e-6
    print(f"riemannian log by continuation: unconverged {(~ok).sum()} / {len(ok)}  ({time.time()-t0:.1f}s)")
    # sanity: Finsler log by continuation vs exact (fraction on the same branch)
    sub = rng.choice(len(t), 400, replace=False)
    wF, errF = log_continuation(mkF, a[sub], y[sub], cfg["cont_stages"], cfg["log_steps"])
    same = np.linalg.norm(wF - logF[sub], axis=1) < 1e-3
    print(f"finsler log: BVP-by-continuation agrees with exact reverse-exp log on {same.mean()*100:.1f}% of 400 samples (rest = other geodesic branch or unconverged)")
    k = cfg["k"]
    X = np.concatenate([a, t[:, None], demos["start"][idx]], 1)[ok]
    Y = {"euclid": (yd + k * logE)[ok], "riemann": (yd + k * logR)[ok], "finsler": (yd + k * logF)[ok]}
    return dict(X=X, Y=Y, a=a[ok], y=y[ok], t=t[ok], idx=idx[ok], logE=logE[ok], logR=logR[ok], logF=logF[ok])


# --------------------------------------------------------------------------------------------------
class MLP(torch.nn.Module):
    def __init__(self, d_in=5, d_out=2, hidden=256, n_layers=3):
        super().__init__()
        layers, d = [], d_in
        for _ in range(n_layers):
            layers += [torch.nn.Linear(d, hidden), torch.nn.SiLU()]
            d = hidden
        layers += [torch.nn.Linear(d, d_out)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(X, Y, cfg, seed):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    model = MLP(hidden=cfg["hidden"], n_layers=cfg["n_layers"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["train_steps"], eta_min=cfg["lr"] * 1e-2)
    n = len(Xt)
    g = torch.Generator().manual_seed(seed)
    for step in range(cfg["train_steps"]):
        b = torch.randint(0, n, (cfg["batch"],), generator=g)
        loss = torch.mean((model(Xt[b]) - Yt[b]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        final = torch.mean((model(Xt) - Yt) ** 2).item()
    return model, final


def model_field(model):
    def v(a, t, c):
        with torch.no_grad():
            inp = torch.tensor(np.concatenate([a, t[:, None], c], 1), dtype=torch.float32)
            return model(inp).numpy().astype(float)
    return v


# --------------------------------------------------------------------------------------------------
def rollout(vfield, F, demos, idx, a0, cfg, n_steps, project):
    """Integrate da/dt = v(a, t | c_i) over t in [0, 1] with RK4; returns metrics per rollout."""
    T_c = cfg["T_c"]
    c = demos["start"][idx]
    a = a0.copy()
    dt = 1.0 / n_steps
    viol = np.zeros(len(a))
    frac = np.zeros(len(a))
    err_mean = np.zeros(len(a))

    def f(t, a):
        v = vfield(a, np.full(len(a), t), c)
        Fv = F.F(a, v / T_c)                                 # physical speed in the envelope metric
        return v, Fv

    for s in range(n_steps):
        t = s * dt
        ks = []
        for stage_t, stage_a in [(t, a), (t + dt / 2, None), (t + dt / 2, None), (t + dt, None)]:
            pass
        v1, F1 = f(t, a)
        viol += np.maximum(0.0, F1 - 1.0) / n_steps
        frac += (F1 > 1.0) / n_steps
        xi_t, _ = demo_at(demos, idx, np.full(len(a), t))
        err_mean += np.linalg.norm(a - xi_t, axis=1) / n_steps
        if project:
            def g(t_, a_):
                v_, Fv_ = f(t_, a_)
                return v_ / np.maximum(1.0, Fv_)[:, None]
        else:
            def g(t_, a_):
                return f(t_, a_)[0]
        k1 = g(t, a)
        k2 = g(t + dt / 2, a + dt / 2 * k1)
        k3 = g(t + dt / 2, a + dt / 2 * k2)
        k4 = g(t + dt, a + dt * k3)
        a = a + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    xi_1 = demos["xi"][idx, -1]
    final_err = np.linalg.norm(a - xi_1, axis=1)
    return dict(viol=viol, frac=frac, err_mean=err_mean, final_err=final_err, a_end=a)


class OracleField:
    """v = xi_dot(t) + k log_a(xi(t)) with the log map solved by shooting (warm-started along the flow)."""

    def __init__(self, kind, mkF, mkR, demos, cfg):
        self.kind, self.mkF, self.mkR, self.demos, self.cfg = kind, mkF, mkR, demos, cfg
        self.w_prev = None
        self.n_calls = 0
        self.unconv = 0

    def __call__(self, a, t, c, idx):
        y, yd = demo_at(self.demos, idx, t)
        if self.kind == "euclid":
            lg = y - a
        else:
            mk = self.mkF if self.kind == "finsler" else self.mkR
            if self.w_prev is None:
                lg, err = log_continuation(mk, a, y, self.cfg["cont_stages"], self.cfg["log_steps"])
            else:
                lg, err = log_continuation(mk, a, y, None, self.cfg["log_steps"], w_init=self.w_prev, iters=5)
            bad = err > 1e-5
            self.unconv += bad.sum()
            lg = np.where(bad[:, None], y - a, lg)          # fall back to the Euclidean direction where the BVP failed
            self.w_prev = lg
        self.n_calls += 1
        return yd + self.cfg["k"] * lg


def evaluate(models, mkF, mkR, demos, cfg, rng, out, do_oracle=True):
    F = mkF(1.0)
    rows = []
    # ---- learned policies: all demos x dirs x magnitudes ----------------------------------------
    n = cfg["n_demo"]
    dirs = np.linspace(0, 2 * np.pi, cfg["eval_dirs"], endpoint=False)
    for m in cfg["eval_m"]:
        idx = np.repeat(np.arange(n), len(dirs))
        th = np.tile(dirs, n)
        a0 = demos["start"][idx] + m * np.stack([np.cos(th), np.sin(th)], -1)
        for name in METHODS:
            vf = model_field(models[name])
            r_free = rollout(vf, F, demos, idx, a0, cfg, cfg["eval_steps"], project=False)
            r_proj = rollout(vf, F, demos, idx, a0, cfg, cfg["eval_steps"], project=True)
            for j in range(len(idx)):
                rows.append(dict(kind="learned", method=name, m=m, demo=int(idx[j]), dir=float(th[j]), viol=r_free["viol"][j], frac=r_free["frac"][j],
                                 final_err_free=r_free["final_err"][j], final_err_proj=r_proj["final_err"][j], err_mean_proj=r_proj["err_mean"][j]))
        print(f"  learned  m={m}: " + "  ".join(f"{nm}: viol {np.mean([r['viol'] for r in rows if r['kind']=='learned' and r['method']==nm and r['m']==m]):.3f} err_proj {np.mean([r['final_err_proj'] for r in rows if r['kind']=='learned' and r['method']==nm and r['m']==m]):.3f}" for nm in METHODS), flush=True)
    # ---- oracle target fields: subset --------------------------------------------------------------
    if do_oracle:
        sub_demos = rng.choice(n, cfg["oracle_demos"], replace=False)
        dirs_o = np.linspace(0, 2 * np.pi, cfg["oracle_dirs"], endpoint=False) + np.pi / 8
        for m in cfg["eval_m"]:
            idx = np.repeat(sub_demos, len(dirs_o))
            th = np.tile(dirs_o, len(sub_demos))
            a0 = demos["start"][idx] + m * np.stack([np.cos(th), np.sin(th)], -1)
            for name in METHODS:
                t0 = time.time()
                res = {}
                for project in (False, True):
                    orc = OracleField(name, mkF, mkR, demos, cfg)
                    vf = lambda a, t, c, orc=orc: orc(a, t, c, idx)
                    res[project] = rollout(vf, F, demos, idx, a0, cfg, cfg["oracle_steps"], project=project)
                    if name != "euclid":
                        print(f"    oracle {name} m={m} project={project}: BVP fallbacks {orc.unconv} of {orc.n_calls*len(idx)}  ({time.time()-t0:.0f}s)", flush=True)
                for j in range(len(idx)):
                    rows.append(dict(kind="oracle", method=name, m=m, demo=int(idx[j]), dir=float(th[j]), viol=res[False]["viol"][j], frac=res[False]["frac"][j],
                                     final_err_free=res[False]["final_err"][j], final_err_proj=res[True]["final_err"][j], err_mean_proj=res[True]["err_mean"][j]))
    with open(out / "fm_path_rollouts.csv", "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)
    return rows


def summarise(rows, out, cfg, beta, train_loss):
    keys = ["viol", "frac", "final_err_free", "final_err_proj", "err_mean_proj"]
    lines = [f"# FM path-benefit experiment (beta = {beta})", "", f"config: `{json.dumps(cfg)}`", "", f"final train MSE: {train_loss}", "",
             "| kind | m | method | n | envelope excess (free) | frac F>1 (free) | final err (free) | final err (projected) | mean err (projected) |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    summ = {}
    for kind in ["learned", "oracle"]:
        for m in cfg["eval_m"]:
            for name in METHODS:
                sel = [r for r in rows if r["kind"] == kind and r["m"] == m and r["method"] == name]
                if not sel:
                    continue
                v = {k: (np.mean([r[k] for r in sel]), np.std([r[k] for r in sel], ddof=1) / np.sqrt(len(sel))) for k in keys}
                summ[(kind, m, name)] = v
                lines.append(f"| {kind} | {m} | {name} | {len(sel)} | " + " | ".join(f"{v[k][0]:.3f} ± {v[k][1]:.3f}" for k in keys) + " |")
    (out / "fm_path_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_env(mkF, mkR, W1, demos, data, cfg, beta, out):
    F = mkF(1.0)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=C_SURF)
    style(ax)
    ax.grid(False)
    xx, yy = np.meshgrid(np.linspace(-4.5, 3.5, 161), np.linspace(-3.2, 3.2, 129))
    grid = np.stack([xx, yy], -1)
    cs = ax.contourf(xx, yy, beta * np.linalg.norm(W1(grid), axis=-1), levels=np.linspace(0, beta, 8),
                     cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5"]))
    for i in range(cfg["n_demo"]):
        ax.plot(demos["xi"][i, :, 0], demos["xi"][i, :, 1], color=C_MUTED, linewidth=1.2, alpha=0.8)
    ax.plot(*cfg["goal"], "*", color=C_INK, markersize=13)
    # three log directions from one perturbed training point
    j = np.argsort(-np.linalg.norm(data["a"] - data["y"], axis=1))[40]
    a, y = data["a"][j], data["y"][j]
    ax.plot(*a, "o", color=C_INK, markersize=7)
    ax.plot(*y, "s", color=C_INK, markersize=7)
    for name, lg in [("euclid", data["logE"][j]), ("riemann", data["logR"][j]), ("finsler", data["logF"][j])]:
        d = lg / np.linalg.norm(lg) * 0.9
        ax.annotate("", xy=a + d, xytext=a, arrowprops=dict(arrowstyle="->", color=C[name], lw=2.2))
        ax.plot([], [], color=C[name], lw=2.2, label=LABEL[name])
    # the actual geodesics a -> y for the two curved metrics
    for name, mk, lg in [("riemann", mkR, data["logR"][j]), ("finsler", mkF, data["logF"][j])]:
        _, xs, _ = mk(1.0).geodesic(a, lg, 1.0, 60)
        ax.plot(xs[:, 0], xs[:, 1], color=C[name], lw=1.2, ls="--")
    ax.plot([a[0], y[0]], [a[1], y[1]], color=C["euclid"], lw=1.2, ls="--")
    ax.set_aspect("equal")
    ax.set_xlim(-4.5, 3.5)
    ax.set_ylim(-3.2, 3.2)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=C_INK2)
    ax.set_title(f"Head-wind channel β = {beta}: time-optimal demos (grey), goal (★), and the three log maps from a perturbed point (●) back to the demo (■)", fontsize=9, color=C_INK, loc="left")
    cb = fig.colorbar(cs, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("|W(x)|", color=C_INK2, fontsize=9)
    cb.ax.tick_params(colors=C_INK2, labelsize=8)
    fig.savefig(out / "fm_path_env.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_metrics(summ, cfg, beta, out):
    panels = [("viol", "envelope excess  mean_t max(0, F−1)  (free request)"), ("final_err_proj", "final tracking error at t = 1  (projected execution)"), ("err_mean_proj", "mean tracking error  (projected execution)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), facecolor=C_SURF)
    ms = cfg["eval_m"]
    for ax, (key, title) in zip(axes, panels):
        style(ax)
        for name in METHODS:
            for kind, ls in [("learned", "-"), ("oracle", "--")]:
                if (kind, ms[0], name) not in summ:
                    continue
                mu = np.array([summ[(kind, m, name)][key][0] for m in ms])
                se = np.array([summ[(kind, m, name)][key][1] for m in ms])
                ax.plot(ms, mu, ls, color=C[name], lw=2, marker="o" if kind == "learned" else "^", markersize=6, label=f"{LABEL[name]} — {kind}")
                ax.fill_between(ms, mu - se, mu + se, color=C[name], alpha=0.12, lw=0)
        ax.set_xlabel("start perturbation |δ|", color=C_INK2, fontsize=9)
        ax.set_title(title, fontsize=9.5, color=C_INK, loc="left")
        ax.set_xticks(ms)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=C_INK2)
    fig.suptitle(f"SFP-type stabilisation with three log maps, head-wind channel β = {beta}   (solid = learned MLP, dashed = oracle target field)", fontsize=10, color=C_INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fm_path_metrics.png", dpi=160)
    plt.close(fig)


def fig_rollouts(models, mkF, W1, demos, cfg, beta, out, m=0.8, n_show=3):
    F = mkF(1.0)
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, n_show, figsize=(4.3 * n_show, 4.0), facecolor=C_SURF)
    xx, yy = np.meshgrid(np.linspace(-4.5, 3.5, 161), np.linspace(-3.2, 3.2, 129))
    grid = np.stack([xx, yy], -1)
    picks = rng.choice(cfg["n_demo"], n_show, replace=False)
    for ax, i in zip(axes, picks):
        style(ax)
        ax.grid(False)
        ax.contourf(xx, yy, beta * np.linalg.norm(W1(grid), axis=-1), levels=np.linspace(0, beta, 8),
                    cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5"]))
        xi = demos["xi"][i]
        ax.plot(xi[:, 0], xi[:, 1], color=C_INK, lw=2.5, alpha=0.5, label="demo chunk")
        th = np.array([np.pi / 2, -np.pi / 2, np.pi])
        idx = np.full(len(th), i)
        a0 = demos["start"][idx] + m * np.stack([np.cos(th), np.sin(th)], -1)
        for name in METHODS:
            vf = model_field(models[name])
            # record the projected trajectory
            T_c, n_steps = cfg["T_c"], 60
            a = a0.copy()
            tr = [a.copy()]
            dt = 1 / n_steps
            for s in range(n_steps):
                def g(t_, a_):
                    v = vf(a_, np.full(len(a_), t_), demos["start"][idx])
                    return v / np.maximum(1.0, F.F(a_, v / T_c))[:, None]
                t = s * dt
                k1 = g(t, a); k2 = g(t + dt / 2, a + dt / 2 * k1); k3 = g(t + dt / 2, a + dt / 2 * k2); k4 = g(t + dt, a + dt * k3)
                a = a + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                tr.append(a.copy())
            tr = np.array(tr)
            for q in range(len(th)):
                ax.plot(tr[:, q, 0], tr[:, q, 1], color=C[name], lw=1.6, label=LABEL[name] if q == 0 else None)
        ax.plot(a0[:, 0], a0[:, 1], "o", color=C_INK, markersize=5)
        lo = xi.min(0) - 1.3
        hi = xi.max(0) + 1.3
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_title(f"demo {i}, |δ| = {m}, projected execution", fontsize=9, color=C_INK, loc="left")
    axes[0].legend(frameon=False, fontsize=7.5, loc="best", labelcolor=C_INK2)
    fig.tight_layout()
    fig.savefig(out / "fm_path_rollouts.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--no-oracle", action="store_true")
    args = ap.parse_args()
    cfg = dict(CFG)
    beta = args.beta
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / f"fm_path_beta{beta}"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    mkF, mkR, E, W1 = make_fields(beta, cfg["width"])
    ALL = ["demos", "data", "train", "eval", "fig"]
    requested = ALL if args.stage == "all" else [st.strip() for st in args.stage.split(",")]
    last = max(ALL.index(st) for st in requested)

    if "demos" in requested:
        demos = gen_demos(mkF, cfg, rng)
        np.savez(out / "demos.npz", **demos)
    else:
        demos = dict(np.load(out / "demos.npz"))
    if last == 0:
        print("wrote", out)
        return

    if "data" in requested:
        data = gen_data(mkF, mkR, demos, cfg, rng)
        np.savez(out / "data.npz", X=data["X"], a=data["a"], y=data["y"], t=data["t"], idx=data["idx"], logE=data["logE"], logR=data["logR"], logF=data["logF"],
                 **{f"Y_{k}": v for k, v in data["Y"].items()})
    else:
        z = np.load(out / "data.npz")
        data = dict(X=z["X"], a=z["a"], y=z["y"], t=z["t"], idx=z["idx"], logE=z["logE"], logR=z["logR"], logF=z["logF"], Y={k: z[f"Y_{k}"] for k in METHODS})
    if last == 1:
        print("wrote", out)
        return

    models, train_loss = {}, {}
    if "train" in requested:
        for name in METHODS:
            t0 = time.time()
            models[name], train_loss[name] = train_model(data["X"], data["Y"][name], cfg, seed=1)
            torch.save(models[name].state_dict(), out / f"model_{name}.pt")
            print(f"trained {name}: final MSE {train_loss[name]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        (out / "train_loss.json").write_text(json.dumps(train_loss), encoding="utf-8")
    else:
        for name in METHODS:
            models[name] = MLP(hidden=cfg["hidden"], n_layers=cfg["n_layers"])
            models[name].load_state_dict(torch.load(out / f"model_{name}.pt"))
            models[name].eval()
        train_loss = json.loads((out / "train_loss.json").read_text(encoding="utf-8"))
    if last == 2:
        print("wrote", out)
        return

    if "eval" in requested:
        rows = evaluate(models, mkF, mkR, demos, cfg, rng, out, do_oracle=not args.no_oracle)
    else:
        with open(out / "fm_path_rollouts.csv", newline="") as f:
            rows = [{k: (v if k in ("kind", "method") else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
            for r in rows:
                r["demo"] = int(r["demo"])

    summ = summarise(rows, out, cfg, beta, train_loss)
    if "fig" in requested or "eval" in requested:
        fig_env(mkF, mkR, W1, demos, data, cfg, beta, out)
        fig_metrics(summ, cfg, beta, out)
        fig_rollouts(models, mkF, W1, demos, cfg, beta, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
