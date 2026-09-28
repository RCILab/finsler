"""External learned-planner baselines on the PLANAR drift-field benchmark.

The SE(2) benchmark has a single goal (the metric is left-invariant, so every query is reduced to the
origin) and one dominant homotopy class, which makes it structurally easy for plain regression: the last
waypoint is a constant it can memorise.  The planar head-wind channel is the harder test of the
*generative mechanism*: two goals, and for the channel-core goal two symmetric approach branches.

Same three methods, same training geodesics, same 98 held-out queries as the paper.

Run: python experiments/external_baselines_planar.py [--steps 30000]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from finsler import channel_wind, finsler_field
from finsler.eikonal import GridFinslerDistance
from external_baselines import MLP, TemporalUNet, timestep_embed

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "fm_planner_beta0.9"
OUT = ROOT / "results" / "external_baselines"
OUT.mkdir(parents=True, exist_ok=True)
GOALS = np.array([[2.5, 0.0], [2.5, 1.6]])
BETA, WIDTH = 0.9, 0.8


def travel_time(F, path):
    d = path[1:] - path[:-1]
    return F.F(path[:-1].reshape(-1, 2), d.reshape(-1, 2)).reshape(d.shape[0], d.shape[1]).sum(0)


def train_regression(Z, C, steps, batch, lr, seed=0):
    torch.manual_seed(seed)
    model = MLP(C.shape[1], Z.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 1e-2)
    Zt, Ct = torch.tensor(Z, dtype=torch.float32), torch.tensor(C, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        b = torch.randint(0, len(Zt), (batch,), generator=g)
        loss = torch.mean((model(Ct[b]) - Zt[b]) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model, float(loss.item())


def train_diffusion(Z, C, steps, batch, lr, n_diff, K, seed=0):
    torch.manual_seed(seed)
    model = TemporalUNet(K, d=2, d_cond=C.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 1e-2)
    Zt, Ct = torch.tensor(Z, dtype=torch.float32), torch.tensor(C, dtype=torch.float32)
    s_ = torch.linspace(0, 1, n_diff + 1)
    ab = torch.cos((s_ + 0.008) / 1.008 * np.pi / 2) ** 2
    ab = (ab / ab[0]).clamp(1e-5, 1.0)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        b = torch.randint(0, len(Zt), (batch,), generator=g)
        z0 = Zt[b]
        k = torch.randint(1, n_diff + 1, (batch,), generator=g)
        a = ab[k][:, None]
        eps = torch.randn(z0.shape, generator=g)
        pred = model(a.sqrt() * z0 + (1 - a).sqrt() * eps, torch.cat([timestep_embed(k), Ct[b]], -1))
        loss = torch.mean((pred - eps) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model, ab, float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--diff-steps", type=int, default=100)
    args = ap.parse_args()

    F = finsler_field(channel_wind(BETA, WIDTH))
    pairs = np.load(SRC / "pairs.npz")
    XF, x0, which = pairs["XF"], pairs["x0"], pairs["which"].astype(int)
    N, K, _ = XF.shape
    print(f"training data: {N} Finsler geodesics x {K} waypoints, goals {np.bincount(which)}")

    Z = XF.reshape(N, K * 2).astype(np.float32)
    onehot = np.eye(2, dtype=np.float32)[which]
    C = np.concatenate([x0.astype(np.float32), onehot], 1)          # start + goal code
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    Zn = (Z - mu) / sd

    t0 = time.time(); reg, l_reg = train_regression(Zn, C, args.steps, args.batch, args.lr)
    print(f"  regression: loss {l_reg:.5f}, {sum(p.numel() for p in reg.parameters())/1e3:.0f}k params, {time.time()-t0:.0f}s", flush=True)
    t0 = time.time(); dif, ab, l_dif = train_diffusion(Zn, C, args.steps, args.batch, args.lr, args.diff_steps, K)
    print(f"  diffusion : loss {l_dif:.5f}, {sum(p.numel() for p in dif.parameters())/1e3:.0f}k params, {time.time()-t0:.0f}s", flush=True)

    # evaluation grid identical to the paper: 7x7 starts x 2 goals
    gx = np.linspace(-3.5, -1.2, 7)
    gy = np.linspace(-1.8, 1.8, 7)
    S = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    starts = np.concatenate([S, S], 0)
    gid = np.concatenate([np.zeros(len(S), int), np.ones(len(S), int)])
    Ce = np.concatenate([starts.astype(np.float32), np.eye(2, dtype=np.float32)[gid]], 1)
    print(f"evaluation: {len(starts)} queries")

    G = GridFinslerDistance(F, (-4.2, 3.6), (-2.6, 2.8), 0.04)
    d_goal = np.array([G.dist_to_target(GOALS[g])[G.node_of(starts[i:i + 1])][0]
                       for i, g in enumerate(gid)])

    results = {}
    t0 = time.time()
    with torch.no_grad():
        Zp = reg(torch.tensor(Ce)).numpy() * sd + mu
    lat = (time.time() - t0) / len(starts)
    P = Zp.reshape(-1, K, 2); P[:, 0] = starts
    results["regression"] = (P.transpose(1, 0, 2), lat)

    a_start = (starts.astype(np.float32) - mu.reshape(K, 2)[0]) / sd.reshape(K, 2)[0]
    a_goal = (GOALS[gid].astype(np.float32) - mu.reshape(K, 2)[-1]) / sd.reshape(K, 2)[-1]
    t0 = time.time()
    g_ = torch.Generator().manual_seed(0)
    Ct = torch.tensor(Ce, dtype=torch.float32)
    z = torch.randn((len(starts), K * 2), generator=g_)
    with torch.no_grad():
        for k in range(args.diff_steps, 0, -1):
            kk = torch.full((len(starts),), k)
            eps = dif(z, torch.cat([timestep_embed(kk), Ct], -1))
            a, apr = ab[k], ab[k - 1]
            z0 = ((z - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-5, 5)
            zz = z0.view(len(starts), K, 2)
            zz[:, 0, :] = torch.tensor(a_start); zz[:, -1, :] = torch.tensor(a_goal)
            z0 = zz.reshape(len(starts), -1)
            if k > 1:
                beta = 1 - a / apr
                mean = (apr.sqrt() * beta / (1 - a)) * z0 + (((a / apr).sqrt() * (1 - apr)) / (1 - a)) * z
                z = mean + (beta * (1 - apr) / (1 - a)).sqrt() * torch.randn(z.shape, generator=g_)
            else:
                z = z0
    lat = (time.time() - t0) / len(starts)
    P = (z.numpy() * sd + mu).reshape(-1, K, 2); P[:, 0] = starts
    results["diffusion"] = (P.transpose(1, 0, 2), lat)

    # ours, at the same waypoint count
    sys.path.insert(0, str(ROOT / "experiments"))
    from fm_planner import MLP as FMLP, integrate
    fm = FMLP(d_in=5, d_out=2, hidden=256, n_layers=4)
    fm.load_state_dict(torch.load(SRC / "model_finsler.pt", map_location="cpu"))
    fm.eval()
    Pf = []
    t0 = time.time()
    for g in (0, 1):
        m = gid == g
        Pf.append((integrate(fm, starts[m].copy(), K - 1, GOALS[g]), np.where(m)[0]))
    lat = (time.time() - t0) / len(starts)
    P = np.zeros((K, len(starts), 2))
    for arr, idx in Pf:
        P[:, idx] = arr
    results["finsler_fm"] = (P, lat)

    rows = []
    for name, (P, lat) in results.items():
        L = travel_time(F, P)
        end = P[-1]
        reach = np.linalg.norm(end - GOALS[gid], axis=1) < 0.35
        r = L / d_goal
        rows.append(dict(method=name, reach=float(reach.mean()), mean_travel=float(L.mean()),
                         ratio_mean=float(r.mean()), ratio_median=float(np.median(r)),
                         ratio_worst=float(r.max()), latency_ms=float(lat * 1e3)))
        print(f"  {name:12s}: reach {reach.mean():.2f}, travel {L.mean():.2f} s, ratio {r.mean():.4f} "
              f"(med {np.median(r):.4f}, worst {r.max():.2f}), {lat*1e3:.2f} ms/query", flush=True)

    nice = {"regression": "Trajectory regression", "diffusion": f"Trajectory diffusion ({args.diff_steps} steps)",
            "finsler_fm": "Finsler flow matching (ours)"}
    lines = ["# External learned-planner baselines, planar head-wind channel (beta = 0.9)", "",
             f"All trained on the same {N} Finsler geodesics, goal-conditioned, evaluated on the same "
             f"{len(starts)} queries as the paper. Unlike SE(2), this benchmark has two goals and, for the "
             "channel-core goal, two symmetric approach branches, so it tests the generative mechanism rather "
             "than memorisation of a single endpoint.", "",
             "| method | reach < 0.35 | mean travel [s] | travel / optimal | median | worst | latency [ms] |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {nice[r['method']]} | {r['reach']:.2f} | {r['mean_travel']:.2f} | {r['ratio_mean']:.4f} "
                     f"| {r['ratio_median']:.4f} | {r['ratio_worst']:.2f} | {r['latency_ms']:.2f} |")
    (OUT / "external_baselines_planar.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump(rows, open(OUT / "external_baselines_planar.json", "w"), indent=1)
    print("wrote", OUT / "external_baselines_planar.md")


if __name__ == "__main__":
    main()
