"""
Closed-loop execution of the learned planners under disturbances (receding horizon / streaming).

The goal-conditioned flow evaluated at t = 0 is a feedback law: v(q) = v_theta(0, q [, g]) points along
the (learned) time-optimal path from q.  Executing it at full envelope speed and re-evaluating every
control step turns the generative planner into a time-optimal feedback controller that re-plans for free.

Scenarios
  SE(2) pose regulation (models from fm_planner_se2.py): slip noise, unknown lateral drift, periodic pushes.
  Planar wind channel (models from fm_planner.py, beta 0.9): slip noise, wind gusts (+-30% beta), pushes.
Reference: the grid Dijkstra feedback policy (moves along the min of edge cost + value), same disturbances.
Metrics: success within tolerance by the time limit, mean time-to-goal, ratio to the undisturbed optimum.

Usage: python experiments/closed_loop.py
"""
from __future__ import annotations

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

from finsler import channel_wind, finsler_field
from finsler.eikonal import GridFinslerDistance, GridSE2Distance
from finsler.se2_randers import SE2Randers
from fm_path_benefit import MLP
import fm_planner as fp
import fm_planner_se2 as fs

C = {"euclid": "#2a78d6", "riemann": "#eb6834", "finsler": "#1baf7a", "stream": "#008300", "bc": "#eda100", "dijkstra": "#0b0b0b"}
LABEL = {"euclid": "Euclidean-path flow (t=0 field)", "riemann": "Riemannian/symmetric-path flow (t=0 field)", "finsler": "Finsler-path flow (t=0 field)", "stream": "Finsler-path flow, streaming (flow clock)", "bc": "direct regression of optimal field (BC)", "dijkstra": "grid Dijkstra feedback (reference)"}
C_INK, C_INK2, C_GRID, C_SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "closed_loop"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ["euclid", "riemann", "finsler", "stream", "bc", "dijkstra"]


def wrap(a):
    return np.remainder(a + np.pi, 2 * np.pi) - np.pi


# ==================================================================================================
# SE(2)
# ==================================================================================================
def se2_models(cfg):
    models = {}
    for name in ["euclid", "riemann", "finsler"]:
        m = MLP(d_in=5, d_out=3, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
        m.load_state_dict(torch.load(ROOT / "results" / "fm_planner_se2" / f"model_{name}.pt"))
        m.eval()
        models[name] = m
    return models


def se2_flow_policy(model):
    def pol(q):
        with torch.no_grad():
            return model(torch.tensor(fs.features(q, np.zeros(len(q))), dtype=torch.float32)).numpy().astype(float)
    return pol


def se2_dijkstra_policy(G, Dgoal, offsets_cost):
    """Greedy descent of the grid value: pick the neighbour offset minimising edge cost + value."""
    offs, = offsets_cost
    nt = G.shape[2]

    def pol(q):
        n = G.node_of(q)
        best = np.zeros((len(q), 3))
        bestv = np.full(len(q), np.inf)
        ix, iy, it = n // (G.shape[1] * nt), (n // nt) % G.shape[1], n % nt
        for (i, j, k) in offs:
            jx, jy, jt = ix + i, iy + j, (it + k) % nt
            ok = (jx >= 0) & (jx < G.shape[0]) & (jy >= 0) & (jy < G.shape[1])
            nb = (jx * G.shape[1] + jy) * nt + jt
            dq = np.array([i * G.h, j * G.h, k * G.hth])
            mid = q + 0.5 * dq
            c = G.F.F(mid, np.broadcast_to(dq, mid.shape)) + Dgoal[np.where(ok, nb, 0)]
            c = np.where(ok, c, np.inf)
            upd = c < bestv
            bestv = np.where(upd, c, bestv)
            best[upd] = dq
        return best
    return pol


def run_se2(F, policy, starts, goal, dist, rng, dt=0.05, T_max=15.0, pos_tol=0.15, head_tol=np.radians(10)):
    """Vectorised closed-loop episodes. dist = dict(slip=(sig_xy, sig_th), drift=(vx, vy) world, push=(period, mag))."""
    q = starts.copy()
    n = len(q)
    done = np.zeros(n, bool)
    t_arr = np.full(n, T_max)
    n_steps = int(round(T_max / dt))
    traj = [q.copy()]
    for s in range(n_steps):
        v = policy(q)
        xi = F.body_velocity(q, v)
        speed = F.F_body(xi)
        xi = xi / np.maximum(speed, 1e-9)[:, None]                      # full envelope speed along the commanded direction
        dq = F.world_velocity(q, xi) * dt
        if "drift" in dist:
            dq[:, :2] += np.asarray(dist["drift"]) * dt
        if "slip" in dist:
            sxy, sth = dist["slip"]
            dq[:, :2] += rng.normal(scale=sxy * np.sqrt(dt), size=(n, 2))
            dq[:, 2] += rng.normal(scale=sth * np.sqrt(dt), size=n)
        if "push" in dist and s > 0 and s % int(round(dist["push"][0] / dt)) == 0:
            ang = rng.uniform(0, 2 * np.pi, n)
            dq[:, :2] += dist["push"][1] * np.stack([np.cos(ang), np.sin(ang)], 1)
        q = np.where(done[:, None], q, q + dq)
        pos_err = np.linalg.norm(q[:, :2] - goal[:2], axis=1)
        head_err = np.abs(wrap(q[:, 2] - goal[2]))
        arrived = (pos_err < pos_tol) & (head_err < head_tol) & ~done
        t_arr[arrived] = (s + 1) * dt
        done |= arrived
        traj.append(q.copy())
        if done.all():
            break
    return dict(success=done, time=t_arr, traj=np.array(traj))


def scenario_se2(seed=0):
    cfg = dict(fs.CFG)
    F, R, mkF = fs.make_metrics(cfg)
    goal = np.asarray(cfg["goal"], float)
    G = GridSE2Distance(F, cfg["grid_lim"], cfg["grid_lim"], cfg["grid_h"], cfg["grid_ntheta"])
    Dgoal = G.dist_to_target(goal)
    offs = [(i, j, k) for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3) if (i, j, k) != (0, 0, 0) and np.gcd.reduce([abs(i), abs(j), abs(k)]) == 1]
    starts = fs.eval_starts(cfg)
    d_opt = Dgoal[G.node_of(starts)]
    models = se2_models(cfg)
    policies = {k: se2_flow_policy(m) for k, m in models.items()}
    policies["dijkstra"] = se2_dijkstra_policy(G, Dgoal, (offs,))
    disturbances = {
        "none": {},
        "slip": dict(slip=(0.08, 0.15)),
        "slip+drift": dict(slip=(0.08, 0.15), drift=(0.0, 0.15)),
        "slip+pushes": dict(slip=(0.08, 0.15), push=(2.0, 0.3)),
    }
    rows, trajs = [], {}
    for dname, dist in disturbances.items():
        for name, pol in policies.items():
            rng = np.random.default_rng(seed)
            r = run_se2(F, pol, starts, goal, dist, rng)
            ratio = np.where(r["success"], r["time"] / d_opt, np.nan)
            rows.append(dict(space="SE(2)", disturbance=dname, method=name, success=r["success"].mean(), time=r["time"].mean(),
                             ratio=np.nanmean(ratio), ratio_se=np.nanstd(ratio, ddof=1) / np.sqrt(np.isfinite(ratio).sum()), worst=np.nanmax(ratio), n=len(starts)))
            trajs[(dname, name)] = r["traj"]
            print(f"  SE(2) {dname:12s} {name:9s}: success {r['success'].mean():.2f}, time {r['time'].mean():.2f} s, ratio {np.nanmean(ratio):.3f} (worst {np.nanmax(ratio):.2f})", flush=True)
    return rows, trajs, starts, goal, F


# ==================================================================================================
# planar wind channel
# ==================================================================================================
PLANAR_DIR = ROOT / "results" / ("fm_planner_beta0.9_global" if (ROOT / "results" / "fm_planner_beta0.9_global" / "model_finsler.pt").exists() else "fm_planner_beta0.9")


def planar_models(cfg):
    models = {}
    for name in ["euclid", "riemann", "finsler"]:
        m = MLP(d_in=5, d_out=2, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
        m.load_state_dict(torch.load(PLANAR_DIR / f"model_{name}.pt"))
        m.eval()
        models[name] = m
    return models


def train_bc(cfg, seed=1):
    """Direct regression baseline: u*(x, g) = optimal velocity direction (unit Finsler speed) at every point of the
    same Finsler-optimal training paths, without the flow-time input.  Executed as a feedback law."""
    z = np.load(PLANAR_DIR / "pairs.npz")
    goals = np.asarray(cfg["goals"], float)
    X, U, T, which = z["XF"], z["UF"], z["T"], z["which"]
    n, K = X.shape[0], X.shape[1]
    inp = np.concatenate([X.reshape(-1, 2), np.repeat(goals[which], K, 0)], 1)
    tgt = (U / T[:, None, None]).reshape(-1, 2)                              # unit-Finsler-speed optimal direction
    torch.manual_seed(seed)
    Xt, Yt = torch.tensor(inp, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)
    m = MLP(d_in=4, d_out=2, hidden=cfg["hidden"], n_layers=cfg["n_layers"])
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["train_steps"], eta_min=cfg["lr"] * 1e-2)
    g = torch.Generator().manual_seed(seed)
    for _ in range(cfg["train_steps"]):
        b = torch.randint(0, len(Xt), (cfg["batch"],), generator=g)
        loss = torch.mean((m(Xt[b]) - Yt[b]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    m.eval()
    print(f"  BC baseline trained on {len(inp)} points, MSE {loss.item():.4f}", flush=True)
    return m


def planar_bc_policy(model, goal):
    def pol(x):
        with torch.no_grad():
            inp = np.concatenate([x, np.repeat(goal[None], len(x), 0)], 1)
            return model(torch.tensor(inp, dtype=torch.float32)).numpy().astype(float)
    return pol


def planar_flow_policy(model, goal):
    def pol(x):
        with torch.no_grad():
            inp = np.concatenate([x, np.zeros((len(x), 1)), np.repeat(goal[None], len(x), 0)], 1)
            return model(torch.tensor(inp, dtype=torch.float32)).numpy().astype(float)
    return pol


def planar_stream_policy(model, goal, F, dt=0.05, T_min=0.5):
    """Streaming execution of the flow: keep a flow clock t_k = elapsed / T_hat (T_hat = F(x0, v(0, x0)) is the
    plan duration under the (1-t) schedule) and command v(t_k, x).  The field at t > 0 is trained on every
    point of the optimal paths, unlike the t = 0 field.  Re-anchors the clock when the flow speed estimate
    F(x, v(t, x)) / (1 - t) disagrees with the elapsed clock by more than 30%."""
    state = {}

    def pol(x):
        n = len(x)
        if "t" not in state or len(state["t"]) != n:
            with torch.no_grad():
                v0 = model(torch.tensor(np.concatenate([x, np.zeros((n, 1)), np.repeat(goal[None], n, 0)], 1), dtype=torch.float32)).numpy().astype(float)
            state["T"] = np.maximum(F.F(x, v0), T_min)
            state["t"] = np.zeros(n)
        t = np.minimum(state["t"], 0.95)
        with torch.no_grad():
            v = model(torch.tensor(np.concatenate([x, t[:, None], np.repeat(goal[None], n, 0)], 1), dtype=torch.float32)).numpy().astype(float)
        # remaining time implied by the field: F(x, v) = T (1 - t)/(1 - t)... under the schedule F(x_t, u_t) = T, so T_implied = F(x, v)
        T_impl = np.maximum(F.F(x, v), T_min)
        bad = np.abs(T_impl - state["T"]) > 0.3 * state["T"]
        state["T"] = np.where(bad, T_impl, state["T"])
        state["t"] = np.where(bad, 0.0, state["t"]) + dt / state["T"]
        return v
    return pol


def planar_dijkstra_policy(G, Dgoal):
    offs = [(i, j) for i in range(-3, 4) for j in range(-3, 4) if (i, j) != (0, 0) and np.gcd(abs(i), abs(j)) == 1]

    def pol(x):
        n = G.node_of(x)
        ix, iy = n // G.shape[1], n % G.shape[1]
        best = np.zeros((len(x), 2))
        bestv = np.full(len(x), np.inf)
        for (i, j) in offs:
            jx, jy = ix + i, iy + j
            ok = (jx >= 0) & (jx < G.shape[0]) & (jy >= 0) & (jy < G.shape[1])
            nb = jx * G.shape[1] + jy
            d = np.array([i * G.h, j * G.h])
            c = G.F.F(x + 0.5 * d, np.broadcast_to(d, x.shape)) + Dgoal[np.where(ok, nb, 0)]
            c = np.where(ok, c, np.inf)
            upd = c < bestv
            bestv = np.where(upd, c, bestv)
            best[upd] = d
        return best
    return pol


def run_planar(F_true, policy, starts, goal, dist, rng, dt=0.05, T_max=25.0, tol=0.3):
    """F_true: metric of the TRUE wind (may differ from the model's beta); executes at full true speed."""
    x = starts.copy()
    n = len(x)
    done = np.zeros(n, bool)
    t_arr = np.full(n, T_max)
    traj = [x.copy()]
    for s in range(int(round(T_max / dt))):
        v = policy(x)
        Ftrue = F_true.F(x, v)
        dx = v / np.maximum(Ftrue, 1e-9)[:, None] * dt
        if "slip" in dist:
            dx += rng.normal(scale=dist["slip"] * np.sqrt(dt), size=(n, 2))
        if "push" in dist and s > 0 and s % int(round(dist["push"][0] / dt)) == 0:
            ang = rng.uniform(0, 2 * np.pi, n)
            dx += dist["push"][1] * np.stack([np.cos(ang), np.sin(ang)], 1)
        x = np.where(done[:, None], x, x + dx)
        arrived = (np.linalg.norm(x - goal, axis=1) < tol) & ~done
        t_arr[arrived] = (s + 1) * dt
        done |= arrived
        traj.append(x.copy())
        if done.all():
            break
    return dict(success=done, time=t_arr, traj=np.array(traj))


def scenario_planar(seed=0, beta=0.9):
    cfg = dict(fp.CFG)
    mkF, mkR, E, W1 = fp.make_fields(beta, cfg["width"])
    F = mkF(1.0)
    goals = np.asarray(cfg["goals"], float)
    goal = goals[0]                                                      # the core goal (hard one)
    (xlo, xhi), (ylo, yhi) = cfg["start_box"]
    sx = np.linspace(xlo + 0.2, xhi - 0.2, 7)
    sy = np.linspace(ylo + 0.2, yhi - 0.2, 7)
    starts = np.stack(np.meshgrid(sx, sy, indexing="ij"), -1).reshape(-1, 2)
    G = GridFinslerDistance(F, cfg["grid_xlim"], cfg["grid_ylim"], cfg["grid_h"])
    Dgoal = G.dist_to_target(goal)
    d_opt = Dgoal[G.node_of(starts)]
    models = planar_models(cfg)
    policies = {k: planar_flow_policy(m, goal) for k, m in models.items()}
    policies["bc"] = planar_bc_policy(train_bc(cfg), goal)
    policies["stream"] = None                                              # built per scenario (stateful)
    policies["dijkstra"] = planar_dijkstra_policy(G, Dgoal)
    print(f"  planar models from {PLANAR_DIR.name}", flush=True)
    scenarios = {
        "none": (F, {}),
        "slip": (F, dict(slip=0.1)),
        "gust +30%": (finsler_field(lambda x: 1.3 * beta * W1(x)) if 1.3 * beta < 1 else finsler_field(lambda x: 0.99 * W1(x)), dict(slip=0.1)),
        "lull -30%": (finsler_field(lambda x: 0.7 * beta * W1(x)), dict(slip=0.1)),
        "slip+pushes": (F, dict(slip=0.1, push=(2.0, 0.4))),
    }
    rows, trajs = [], {}
    for sname, (F_true, dist) in scenarios.items():
        Gt = GridFinslerDistance(F_true, cfg["grid_xlim"], cfg["grid_ylim"], cfg["grid_h"]) if F_true is not F else G
        d_opt_true = Gt.dist_to_target(goal)[Gt.node_of(starts)]
        for name, pol in policies.items():
            if name == "stream":
                pol = planar_stream_policy(models["finsler"], goal, F)
            rng = np.random.default_rng(seed)
            r = run_planar(F_true, pol, starts, goal, dist, rng)
            ratio = np.where(r["success"], r["time"] / d_opt_true, np.nan)
            rows.append(dict(space="plane", disturbance=sname, method=name, success=r["success"].mean(), time=r["time"].mean(),
                             ratio=np.nanmean(ratio), ratio_se=np.nanstd(ratio, ddof=1) / np.sqrt(max(np.isfinite(ratio).sum(), 2)), worst=np.nanmax(ratio), n=len(starts)))
            trajs[(sname, name)] = r["traj"]
            print(f"  plane {sname:12s} {name:9s}: success {r['success'].mean():.2f}, time {r['time'].mean():.2f} s, ratio {np.nanmean(ratio):.3f} (worst {np.nanmax(ratio):.2f})", flush=True)
    return rows, trajs, starts, goal, W1, beta


# ==================================================================================================
def style(ax):
    ax.set_facecolor(C_SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=C_INK2, labelsize=9)


def fig_bars(rows, out):
    spaces = ["SE(2)", "plane"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2), facecolor=C_SURF)
    for r_i, space in enumerate(spaces):
        sel = [r for r in rows if r["space"] == space]
        dists = list(dict.fromkeys(r["disturbance"] for r in sel))
        xs = np.arange(len(dists))
        wbar = 0.2
        for c_i, (key, title) in enumerate([("ratio", "time-to-goal / undisturbed optimum (successful runs)"), ("success", "success rate")]):
            ax = axes[r_i, c_i]
            style(ax)
            ax.grid(True, axis="y", color=C_GRID, lw=0.8)
            ax.set_axisbelow(True)
            present = [m for m in METHODS if any(r["method"] == m for r in sel)]
            wbar = 0.8 / len(present)
            for m_i, m in enumerate(present):
                vals = [next(r[key] for r in sel if r["disturbance"] == d and r["method"] == m) for d in dists]
                errs = [next(r["ratio_se"] for r in sel if r["disturbance"] == d and r["method"] == m) for d in dists] if key == "ratio" else None
                ax.bar(xs + (m_i - (len(present) - 1) / 2) * wbar, vals, wbar - 0.03, color=C[m], label=LABEL[m], yerr=errs, error_kw=dict(ecolor=C_INK2, lw=0.8, capsize=2))
            ax.set_xticks(xs)
            ax.set_xticklabels(dists, fontsize=8.5)
            ax.set_title(f"{space}: {title}", fontsize=9.5, color=C_INK, loc="left")
            if key == "ratio":
                ax.axhline(1.0, color=C_INK2, lw=0.8, ls=":")
                ax.set_ylim(0.9, None)
            else:
                ax.set_ylim(0, 1.05)
    axes[1, 0].legend(frameon=False, fontsize=7.5, labelcolor=C_INK2, loc="upper left")
    fig.suptitle("Closed-loop execution: the t = 0 flow field as a time-optimal feedback law under disturbances", fontsize=10.5, color=C_INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "closed_loop_bars.png", dpi=160)
    plt.close(fig)


def fig_traj(trajs_se2, starts_se2, goal_se2, trajs_pl, starts_pl, goal_pl, W1, beta, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=C_SURF)
    ax = axes[0]
    style(ax)
    ax.set_aspect("equal")
    pick = np.linspace(0, len(starts_se2) - 1, 5).astype(int)
    for name in ["euclid", "riemann", "finsler"]:
        tr = trajs_se2[("slip+pushes", name)]
        for j in pick:
            ax.plot(tr[:, j, 0], tr[:, j, 1], color=C[name], lw=1.4, alpha=0.9, label=LABEL[name] if j == pick[0] else None)
    for j in pick:
        fs.draw_pose(ax, starts_se2[j], "#ffffff", size=0.2)
    fs.draw_pose(ax, goal_se2, C_INK, size=0.24)
    ax.set_title("SE(2), slip + pushes every 2 s (0.3 m): closed-loop paths for five starts", fontsize=9.5, color=C_INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=C_INK2, loc="lower left")
    ax = axes[1]
    style(ax)
    ax.set_aspect("equal")
    xx, yy = np.meshgrid(np.linspace(-4.0, 3.5, 151), np.linspace(-2.4, 2.6, 101))
    grid = np.stack([xx, yy], -1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5"])
    ax.contourf(xx, yy, beta * np.linalg.norm(W1(grid), axis=-1), levels=np.linspace(0, beta, 8), cmap=cmap)
    pick = np.linspace(0, len(starts_pl) - 1, 6).astype(int)
    for name in ["euclid", "riemann", "finsler"]:
        tr = trajs_pl[("slip+pushes", name)]
        for j in pick:
            ax.plot(tr[:, j, 0], tr[:, j, 1], color=C[name], lw=1.4, alpha=0.9)
    for j in pick:
        ax.plot(*starts_pl[j], "o", color=C_INK, ms=4)
    ax.plot(*goal_pl, "*", color=C_INK, ms=12)
    ax.set_xlim(-4.0, 3.5)
    ax.set_ylim(-2.4, 2.6)
    ax.set_title(f"wind channel β = {beta}, slip + pushes (0.4): closed-loop paths to the core goal", fontsize=9.5, color=C_INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "closed_loop_paths.png", dpi=160)
    plt.close(fig)


def main():
    t0 = time.time()
    print("SE(2) closed loop", flush=True)
    rows_se2, trajs_se2, starts_se2, goal_se2, F_se2 = scenario_se2()
    print("planar closed loop", flush=True)
    rows_pl, trajs_pl, starts_pl, goal_pl, W1, beta = scenario_planar()
    rows = rows_se2 + rows_pl
    lines = ["# Closed-loop execution under disturbances", "",
             "Policy = learned flow field at t = 0, executed at full envelope speed, re-evaluated every 0.05 s.  Reference = grid Dijkstra feedback.",
             "SE(2): 192 start poses, tolerance 0.15 m / 10°, limit 15 s.  Plane: 49 starts to the core goal, tolerance 0.3, limit 25 s; the model was trained at β = 0.9, gusts/lulls change the TRUE wind.",
             "ratio = time-to-goal / undisturbed optimal time (successful runs).", "",
             "| space | disturbance | method | success | mean time [s] | ratio ± s.e. | worst |", "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['space']} | {r['disturbance']} | {r['method']} | {r['success']:.2f} | {r['time']:.2f} | {r['ratio']:.3f} ± {r['ratio_se']:.3f} | {r['worst']:.2f} |")
    (OUT / "closed_loop_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    fig_bars(rows, OUT)
    fig_traj(trajs_se2, starts_se2, goal_se2, trajs_pl, starts_pl, goal_pl, W1, beta, OUT)
    print(f"wrote {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
