"""External learned-planner baselines on the SE(2) benchmark.

The paper currently compares conditional-path geometries inside one flow-matching framework.
A reviewer will ask whether the *generative mechanism* matters, i.e. whether a published-style
trajectory diffusion planner or plain trajectory regression would do just as well.  This script
trains both on exactly the same Finsler geodesics and scores them on exactly the same 192 held-out
poses, so the only thing that differs is how the plan is produced.

  A. Trajectory diffusion (Diffuser / Motion-Planning-Diffusion style): DDPM over the whole
     48-waypoint trajectory, conditioned on the start pose, with start/goal inpainting.
  B. Trajectory regression: one forward pass from the start pose to the whole trajectory.
  C. Finsler flow matching (ours), re-scored at the same waypoint count for a fair length comparison.

Not wired into the paper; writes results/external_baselines/.

Run: python experiments/external_baselines.py [--steps 6000] [--diff-steps 100]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from finsler.eikonal import GridSE2Distance
from finsler.se2_randers import SE2Randers

ROOT = pathlib.Path(__file__).resolve().parents[1]
SE2 = ROOT / "results" / "fm_planner_se2"
OUT = ROOT / "results" / "external_baselines"
OUT.mkdir(parents=True, exist_ok=True)
SPEEDS = dict(v_fwd=1.2, v_back=0.5, v_side=0.4, w_max=1.2)
GOAL = np.zeros(3)


def wrap(a):
    return np.remainder(a + np.pi, 2 * np.pi) - np.pi


def enc(q):
    """(x, y, theta) -> (x, y, cos, sin)"""
    return np.concatenate([q[..., :2], np.cos(q[..., 2:3]), np.sin(q[..., 2:3])], -1)


def dec(z):
    return np.concatenate([z[..., :2], np.arctan2(z[..., 3:4], z[..., 2:3])], -1)


def travel_time(F, path):
    """path (N, B, 3) -> Finsler length per trajectory = execution time."""
    d = path[1:] - path[:-1]
    d[..., 2] = wrap(d[..., 2])
    return F.F(path[:-1].reshape(-1, 3), d.reshape(-1, 3)).reshape(d.shape[0], d.shape[1]).sum(0)


class MLP(torch.nn.Module):
    def __init__(self, d_in, d_out, hidden=512, n_layers=4):
        super().__init__()
        layers, d = [], d_in
        for _ in range(n_layers):
            layers += [torch.nn.Linear(d, hidden), torch.nn.SiLU()]
            d = hidden
        layers += [torch.nn.Linear(d, d_out)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TemporalUNet(torch.nn.Module):
    """Diffuser-style backbone: 1-D convolutions along the waypoint axis, with the diffusion step and
    the start pose injected per-channel (FiLM).  Much better suited to trajectories than a flat MLP."""

    def __init__(self, K, d=4, d_cond=4, ch=64, n_blocks=3, t_dim=64):
        super().__init__()
        self.K, self.d = K, d
        self.inp = torch.nn.Conv1d(d, ch, 5, padding=2)
        self.film = torch.nn.ModuleList([torch.nn.Linear(t_dim + d_cond, 2 * ch) for _ in range(n_blocks)])
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv1d(ch, ch, 5, padding=2), torch.nn.SiLU(),
                                torch.nn.Conv1d(ch, ch, 5, padding=2))
            for _ in range(n_blocks)])
        self.norms = torch.nn.ModuleList([torch.nn.GroupNorm(8, ch) for _ in range(n_blocks)])
        self.out = torch.nn.Conv1d(ch, d, 5, padding=2)
        self.act = torch.nn.SiLU()

    def forward(self, z, cond):
        B = z.shape[0]
        h = self.inp(z.view(B, self.K, self.d).transpose(1, 2))
        for film, blk, nrm in zip(self.film, self.blocks, self.norms):
            g, b = film(cond).chunk(2, -1)
            h = h + blk(self.act(nrm(h)) * (1 + g[..., None]) + b[..., None])
        return self.out(self.act(h)).transpose(1, 2).reshape(B, -1)


def timestep_embed(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


# ---------------------------------------------------------------------------------------------
def train_regression(Z, C, steps, batch, lr, seed=0):
    """C (start features) -> Z (flattened trajectory)."""
    torch.manual_seed(seed)
    model = MLP(C.shape[1], Z.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 1e-2)
    Zt, Ct = torch.tensor(Z, dtype=torch.float32), torch.tensor(C, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for s in range(steps):
        b = torch.randint(0, len(Zt), (batch,), generator=g)
        loss = torch.mean((model(Ct[b]) - Zt[b]) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model, float(loss.item())


def train_diffusion(Z, C, steps, batch, lr, n_diff, K, seed=0):
    """DDPM over the whole trajectory, eps-prediction, cosine schedule, temporal-conv backbone."""
    torch.manual_seed(seed)
    model = TemporalUNet(K, d=4, d_cond=C.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 1e-2)
    Zt, Ct = torch.tensor(Z, dtype=torch.float32), torch.tensor(C, dtype=torch.float32)
    s_ = torch.linspace(0, 1, n_diff + 1)
    ab = torch.cos((s_ + 0.008) / 1.008 * np.pi / 2) ** 2
    ab = (ab / ab[0]).clamp(1e-5, 1.0)                       # alpha_bar
    g = torch.Generator().manual_seed(seed)
    for s in range(steps):
        b = torch.randint(0, len(Zt), (batch,), generator=g)
        z0 = Zt[b]
        k = torch.randint(1, n_diff + 1, (batch,), generator=g)
        a = ab[k][:, None]
        eps = torch.randn(z0.shape, generator=g)
        zk = a.sqrt() * z0 + (1 - a).sqrt() * eps
        pred = model(zk, torch.cat([timestep_embed(k), Ct[b]], -1))
        loss = torch.mean((pred - eps) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model, ab, float(loss.item())


@torch.no_grad()
def sample_diffusion(model, ab, C, K, n_diff, seed=0, anchor=None):
    """Ancestral sampling with start/goal inpainting (Diffuser-style hard conditioning)."""
    g = torch.Generator().manual_seed(seed)
    Ct = torch.tensor(C, dtype=torch.float32)
    z = torch.randn((len(C), (K) * 4), generator=g)
    for k in range(n_diff, 0, -1):
        kk = torch.full((len(C),), k)
        eps = model(torch.cat([z, Ct, timestep_embed(kk)], -1))
        a, ap = ab[k], ab[k - 1]
        z0 = ((z - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-4, 4)
        if anchor is not None:                                # pin first and last waypoint
            zz = z0.view(len(C), K, 4)
            zz[:, 0, :] = torch.tensor(anchor[0], dtype=torch.float32)
            zz[:, -1, :] = torch.tensor(anchor[1], dtype=torch.float32)
            z0 = zz.view(len(C), -1)
        if k > 1:
            beta = 1 - a / ap
            mean = (ap.sqrt() * beta / (1 - a)) * z0 + (((a / ap).sqrt() * (1 - ap)) / (1 - a)) * z
            z = mean + (beta * (1 - ap) / (1 - a)).sqrt() * torch.randn(z.shape, generator=g)
        else:
            z = z0
    return z.numpy()


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--diff-steps", type=int, default=100)
    ap.add_argument("--grid-h", type=float, default=0.1)
    args = ap.parse_args()

    F = SE2Randers.from_speeds(**SPEEDS)
    pairs = np.load(SE2 / "pairs.npz")
    XF, q0 = pairs["XF"], pairs["q0"]                          # (N, K, 3), (N, 3)
    N, K, _ = XF.shape
    print(f"training data: {N} Finsler geodesics x {K} waypoints")

    Z = enc(XF).reshape(N, K * 4).astype(np.float32)
    C = enc(q0).astype(np.float32)
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    Zn = (Z - mu) / sd
    anchor = None                                              # set per-sample below instead

    # ---- train
    t0 = time.time()
    reg, l_reg = train_regression(Zn, C, args.steps, args.batch, args.lr)
    t_reg = time.time() - t0
    t0 = time.time()
    dif, ab, l_dif = train_diffusion(Zn, C, args.steps, args.batch, args.lr, args.diff_steps, K)
    t_dif = time.time() - t0
    n_reg = sum(p.numel() for p in reg.parameters())
    n_dif = sum(p.numel() for p in dif.parameters())
    print(f"  regression: loss {l_reg:.5f}, {n_reg/1e3:.0f}k params, {t_reg:.0f}s")
    print(f"  diffusion : loss {l_dif:.5f}, {n_dif/1e3:.0f}k params, {t_dif:.0f}s", flush=True)

    # ---- evaluation poses: identical to fm_planner_se2
    e = 2.0
    xs = np.linspace(-e, e, 5)
    ths = np.linspace(-np.pi, np.pi, 8, endpoint=False) + np.pi / 8
    X, Y, TH = np.meshgrid(xs, xs, ths, indexing="ij")
    starts = np.stack([X.ravel(), Y.ravel(), TH.ravel()], -1)
    starts = starts[np.linalg.norm(starts[:, :2], axis=1) > 0.3]
    Ce = enc(starts).astype(np.float32)
    print(f"evaluation: {len(starts)} held-out poses")

    G = GridSE2Distance(F, (-3.2, 3.2), (-3.2, 3.2), args.grid_h, 36)
    d_goal = G.dist_to_target(GOAL)[G.node_of(starts)]

    results = {}

    # (A) regression
    t0 = time.time()
    with torch.no_grad():
        Zp = reg(torch.tensor(Ce)).numpy() * sd + mu
    lat_reg = (time.time() - t0) / len(starts)
    P_reg = dec(Zp.reshape(-1, K, 4))
    P_reg[:, 0] = starts                                       # plans start where the robot is
    results["regression"] = (P_reg.transpose(1, 0, 2), lat_reg)

    # (B) diffusion, with start/goal inpainting in normalised space
    a_start = ((enc(starts) - mu.reshape(K, 4)[0]) / sd.reshape(K, 4)[0])
    a_goal = ((enc(np.repeat(GOAL[None], len(starts), 0)) - mu.reshape(K, 4)[-1]) / sd.reshape(K, 4)[-1])
    t0 = time.time()
    g = torch.Generator().manual_seed(0)
    Ct = torch.tensor(Ce, dtype=torch.float32)
    z = torch.randn((len(starts), K * 4), generator=g)
    with torch.no_grad():
        for k in range(args.diff_steps, 0, -1):
            kk = torch.full((len(starts),), k)
            eps = dif(z, torch.cat([timestep_embed(kk), Ct], -1))
            a, apr = ab[k], ab[k - 1]
            z0 = ((z - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-5, 5)
            zz = z0.view(len(starts), K, 4)
            zz[:, 0, :] = torch.tensor(a_start, dtype=torch.float32)
            zz[:, -1, :] = torch.tensor(a_goal, dtype=torch.float32)
            z0 = zz.reshape(len(starts), -1)
            if k > 1:
                beta = 1 - a / apr
                mean = (apr.sqrt() * beta / (1 - a)) * z0 + (((a / apr).sqrt() * (1 - apr)) / (1 - a)) * z
                z = mean + (beta * (1 - apr) / (1 - a)).sqrt() * torch.randn(z.shape, generator=g)
            else:
                z = z0
    lat_dif = (time.time() - t0) / len(starts)
    P_dif = dec((z.numpy() * sd + mu).reshape(-1, K, 4))
    P_dif[:, 0] = starts
    results["diffusion"] = (P_dif.transpose(1, 0, 2), lat_dif)

    # (C) ours, re-integrated at the same waypoint count
    sys.path.insert(0, str(ROOT / "experiments"))
    from fm_planner_se2 import MLP as FMLP, features, integrate
    fm = FMLP(d_in=5, d_out=3, hidden=256, n_layers=4)
    fm.load_state_dict(torch.load(SE2 / "model_finsler.pt", map_location="cpu"))
    fm.eval()
    t0 = time.time()
    P_fm = integrate(fm, starts.copy(), K - 1)
    lat_fm = (time.time() - t0) / len(starts)
    results["finsler_fm"] = (P_fm, lat_fm)

    # ---- score
    rows = []
    for name, (P, lat) in results.items():
        L = travel_time(F, P)
        end = P[-1]
        pos = np.linalg.norm(end[:, :2] - GOAL[:2], axis=1)
        head = np.abs(wrap(end[:, 2] - GOAL[2]))
        succ = (pos < 0.2) & (head < np.radians(15))
        r = L / d_goal
        rows.append(dict(method=name, success=float(succ.mean()), mean_travel=float(L.mean()),
                         ratio_mean=float(r.mean()), ratio_median=float(np.median(r)),
                         ratio_worst=float(r.max()), pos_err_mean=float(pos.mean()),
                         latency_ms=float(lat * 1e3)))
        print(f"  {name:12s}: success {succ.mean():.2f}, travel {L.mean():.3f} s, "
              f"ratio {r.mean():.4f} (med {np.median(r):.4f}, worst {r.max():.3f}), "
              f"pos err {pos.mean():.3f} m, {lat*1e3:.2f} ms/query", flush=True)

    nice = {"regression": "Trajectory regression (start -> waypoints)",
            "diffusion": f"Trajectory diffusion, {args.diff_steps} steps (Diffuser/MPD style)",
            "finsler_fm": "Finsler flow matching (ours)"}
    lines = ["# External learned-planner baselines on the SE(2) benchmark", "",
             f"All three are trained on the same {N} Finsler geodesics and produce a {K}-waypoint plan for the "
             f"same {len(starts)} held-out poses; travel time is the Finsler length of the produced polyline, "
             f"and the reference is the SE(2) grid (h = {args.grid_h}).  Only the way the plan is produced differs.", "",
             f"Parameters: regression {n_reg/1e3:.0f}k, diffusion {n_dif/1e3:.0f}k, flow matching "
             f"{sum(p.numel() for p in fm.parameters())/1e3:.0f}k.", "",
             "| method | success | mean travel [s] | travel / optimal | median | worst | final pos. err [m] | latency [ms] |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {nice[r['method']]} | {r['success']:.2f} | {r['mean_travel']:.3f} | "
                     f"{r['ratio_mean']:.4f} | {r['ratio_median']:.4f} | {r['ratio_worst']:.3f} | "
                     f"{r['pos_err_mean']:.3f} | {r['latency_ms']:.2f} |")
    (OUT / "external_baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump(rows, open(OUT / "external_baselines.json", "w"), indent=1)
    np.savez(OUT / "paths.npz", starts=starts, **{f"path_{k}": v[0] for k, v in results.items()})
    print("wrote", OUT / "external_baselines.md")


if __name__ == "__main__":
    main()
