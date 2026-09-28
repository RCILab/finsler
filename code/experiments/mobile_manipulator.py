"""Mobile manipulator: the end-effector velocity envelope is a Minkowski sum, and it is non-reversible.

A skid-steer base carries a two-link planar arm held at a nominal posture.  Writing p0 for the
end-effector offset in the body frame, the achievable end-effector twist is

    (xi_ee, omega) = (v_b + omega * z_hat x p0 + J_a qdot_a ,  omega),
    (v_b, omega) in B_base ,  qdot_a in Q_arm ,

so the achievable set is the Minkowski sum of the base's contribution and the arm's.  Two consequences:

  * support functions add, F*(xi) = F*_base(xi) + F*_arm(xi), which is why a sum-of-norms family is the
    natural model class on the dual side;
  * the base is non-holonomic (no lateral velocity) yet the sum is full dimensional, because the arm
    supplies exactly the direction the base cannot.  The arm repairs the base's nonholonomy at the
    end effector, and what survives is asymmetry, not degeneracy.

The set is constant in the body frame, so the metric is left invariant on SE(2) and the same grid
reference as the quadruped experiment applies.  Plans are computed under each geometry and executed
under the true envelope.

Run: python experiments/mobile_manipulator.py [--h 0.1] [--ntheta 36]
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
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import ConvexHull

from finsler.eikonal import GridSE2Distance

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "mobile_manipulator"
OUT.mkdir(parents=True, exist_ok=True)


def wrap(a):
    return np.remainder(a + np.pi, 2 * np.pi) - np.pi


def base_vertices(vx_back, vx_fwd, w_max):
    """Skid-steer twist set: no lateral velocity, asymmetric forward/backward."""
    return np.array([[vx, 0.0, w] for vx in (-vx_back, vx_fwd) for w in (-w_max, w_max)])


def arm_vertices(l1, l2, q_nom, qd_max):
    """End-effector velocity contributions of the arm at a fixed posture, joint speeds in a box."""
    q1, q2 = q_nom
    s1, c1, s12, c12 = np.sin(q1), np.cos(q1), np.sin(q1 + q2), np.cos(q1 + q2)
    J = np.array([[-l1 * s1 - l2 * s12, -l2 * s12],
                  [l1 * c1 + l2 * c12, l2 * c12]])
    p0 = np.array([l1 * c1 + l2 * c12, l1 * s1 + l2 * s12])
    V = np.array([[a, b] for a in (-qd_max[0], qd_max[0]) for b in (-qd_max[1], qd_max[1])])
    return (J @ V.T).T, p0


class EEPolytope:
    """Left-invariant metric on SE(2) whose body-frame unit ball is the Minkowski-sum polytope."""

    def __init__(self, base_v, arm_v, p0):
        # base twist (vx, vy, w) -> end-effector twist (vx - w*p0y, vy + w*p0x, w)
        B = np.stack([base_v[:, 0] - base_v[:, 2] * p0[1],
                      base_v[:, 1] + base_v[:, 2] * p0[0],
                      base_v[:, 2]], 1)
        A = np.concatenate([arm_v, np.zeros((len(arm_v), 1))], 1)
        pts = (B[:, None, :] + A[None, :, :]).reshape(-1, 3)
        hull = ConvexHull(pts)
        a, b = hull.equations[:, :3], hull.equations[:, 3]        # a.x + b <= 0
        keep = b < -1e-9                                          # origin strictly inside
        self.A, self.c = a[keep], -b[keep]
        self.vertices = pts[hull.vertices]

    def gauge_body(self, xi):
        xi = np.atleast_2d(np.asarray(xi, float))
        return np.max((xi @ self.A.T) / self.c[None], axis=1)

    def to_body(self, q, dq):
        c, s = np.cos(q[:, 2]), np.sin(q[:, 2])
        return np.stack([c * dq[:, 0] + s * dq[:, 1], -s * dq[:, 0] + c * dq[:, 1], dq[:, 2]], 1)

    def F(self, q, dq):
        q = np.atleast_2d(np.asarray(q, float))
        dq = np.atleast_2d(np.asarray(dq, float))
        return np.maximum(self.gauge_body(self.to_body(q, dq)), 0.0)

    def speeds(self):
        d = {"forward": [1, 0, 0], "backward": [-1, 0, 0], "left": [0, 1, 0], "right": [0, -1, 0],
             "yaw+": [0, 0, 1], "yaw-": [0, 0, -1]}
        return {k: float(1.0 / max(self.gauge_body(np.array([v]))[0], 1e-12)) for k, v in d.items()}


class Symmetrised:
    def __init__(self, f):
        self.f = f

    def F(self, q, dq):
        dq = np.atleast_2d(np.asarray(dq, float))
        return 0.5 * (self.f.F(q, dq) + self.f.F(q, -dq))


class Isotropic:
    def __init__(self, f, w_scale):
        dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)
        self.v = float(np.mean(1.0 / np.maximum(f.gauge_body(dirs), 1e-12)))
        self.w = w_scale

    def F(self, q, dq):
        dq = np.atleast_2d(np.asarray(dq, float))
        return np.sqrt((np.linalg.norm(dq[:, :2], axis=1) / self.v) ** 2 + (dq[:, 2] / self.w) ** 2)


def path_nodes(graph, src, dst):
    d, pred = dijkstra(graph, directed=True, indices=src, return_predecessors=True)
    if not np.isfinite(d[dst]):
        return None
    out = [dst]
    while out[-1] != src:
        p = pred[out[-1]]
        if p < 0:
            return None
        out.append(p)
    return np.array(out[::-1])


def exec_time(field, nodes):
    dq = nodes[1:] - nodes[:-1]
    dq[:, 2] = wrap(dq[:, 2])
    mid = nodes[:-1].copy()
    mid[:, :2] += 0.5 * dq[:, :2]
    mid[:, 2] += 0.5 * dq[:, 2]
    return float(field.F(mid, dq).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.1)
    ap.add_argument("--ntheta", type=int, default=36)
    ap.add_argument("--vx-fwd", type=float, default=1.0)
    ap.add_argument("--vx-back", type=float, default=0.3)
    ap.add_argument("--w-max", type=float, default=1.0)
    ap.add_argument("--qd", type=float, default=1.2)
    args = ap.parse_args()

    bv = base_vertices(args.vx_back, args.vx_fwd, args.w_max)
    av, p0 = arm_vertices(0.35, 0.30, (0.5, 0.9), (args.qd, args.qd))
    ee = EEPolytope(bv, av, p0)
    sym, iso = Symmetrised(ee), Isotropic(ee, args.w_max)
    sp = ee.speeds()
    print(f"skid-steer base: forward {args.vx_fwd}, backward {args.vx_back} m/s, yaw {args.w_max} rad/s, "
          f"no lateral velocity")
    print(f"arm at nominal posture, end-effector offset ({p0[0]:.2f}, {p0[1]:.2f}) m in the body frame")
    print("end-effector envelope (Minkowski sum), max speeds:")
    print("   " + ", ".join(f"{k} {v:.2f}" for k, v in sp.items()))
    print(f"   forward/backward ratio {sp['forward']/sp['backward']:.2f}; lateral {sp['left']:.2f} m/s "
          f"comes entirely from the arm, since the base has none")
    print(f"   polytope: {len(ee.vertices)} vertices, {len(ee.A)} facets")

    t0 = time.time()
    grids = {name: GridSE2Distance(f, (-2.6, 2.6), (-2.6, 2.6), args.h, args.ntheta)
             for name, f in [("true", ee), ("symmetric", sym), ("euclid", iso)]}
    print(f"grids built ({time.time()-t0:.0f}s)", flush=True)

    goal = np.zeros(3)
    tasks = [(np.array([2.0, 0.0, 0.0]), "goal 2 m behind, same heading"),
             (np.array([2.0, 0.0, np.pi]), "goal 2 m behind, heading reversed"),
             (np.array([0.0, 2.0, 0.0]), "goal 2 m to the side"),
             (np.array([-2.0, 0.0, 0.0]), "goal 2 m ahead"),
             (np.array([1.5, 1.5, np.pi / 2]), "goal behind and to the side")]

    rows = []
    for s, label in tasks:
        src = int(grids["true"].node_of(s[None])[0])
        dst = int(grids["true"].node_of(goal[None])[0])
        t_opt = float(dijkstra(grids["true"].graph, directed=True, indices=src)[dst])
        rec = dict(task=label, optimal=t_opt)
        for name in ("true", "symmetric", "euclid"):
            idx = path_nodes(grids[name].graph, src, dst)
            rec[name] = float("nan") if idx is None else exec_time(ee, grids[name].nodes[idx]) / t_opt
            if idx is not None:
                rec[name + "_path"] = grids[name].nodes[idx]
        rows.append(rec)
        print(f"  {label:34s} optimal {t_opt:5.2f} s | Finsler {rec['true']:.3f} "
              f"| symmetric {rec['symmetric']:.3f} | isotropic {rec['euclid']:.3f}", flush=True)

    fin = np.array([r["true"] for r in rows])
    sy = np.array([r["symmetric"] for r in rows])
    eu = np.array([r["euclid"] for r in rows])
    lines = ["# Mobile manipulator: end-effector planning under a Minkowski-sum envelope", "",
             f"Skid-steer base (forward {args.vx_fwd}, backward {args.vx_back} m/s, yaw {args.w_max} rad/s, no "
             f"lateral velocity) carrying a two-link arm at a nominal posture with the end effector "
             f"{np.linalg.norm(p0):.2f} m from the base origin.", "",
             "The achievable end-effector twist set is the Minkowski sum of the base and arm contributions, a "
             f"polytope with {len(ee.vertices)} vertices. It is full dimensional even though the base has no "
             "lateral velocity, because the arm supplies that direction; what remains is asymmetry rather than "
             "degeneracy.", "",
             "| direction | max end-effector speed |", "|---|---:|"]
    for k, v in sp.items():
        lines.append(f"| {k} | {v:.2f} |")
    lines += ["", f"Forward/backward ratio {sp['forward']/sp['backward']:.2f}.", "",
              "Plans computed under each geometry, executed under the true envelope; the symmetric baseline is "
              "the exact symmetrisation (F(v) + F(-v))/2, which keeps the anisotropy and removes only the "
              "asymmetry.", "",
              "| task | optimal [s] | Finsler | symmetric part | isotropic |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['optimal']:.2f} | {r['true']:.3f} | {r['symmetric']:.3f} "
                     f"| {r['euclid']:.3f} |")
    lines += ["", f"Mean over the {len(rows)} tasks: Finsler {fin.mean():.3f}, symmetric {sy.mean():.3f}, "
              f"isotropic {eu.mean():.3f}."]
    (OUT / "mobile_manipulator.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.savez(OUT / "mobile_paths.npz", vertices=ee.vertices, p0=p0,
             **{f"{r['task'].replace(' ', '_').replace(',', '')}__{n}": r[n + "_path"]
                for r in rows for n in ("true", "symmetric", "euclid") if n + "_path" in r})
    json.dump(dict(speeds=sp, finsler=float(fin.mean()), symmetric=float(sy.mean()),
                   isotropic=float(eu.mean())), open(OUT / "mobile_manipulator.json", "w"), indent=1)
    print(f"\nmean: Finsler {fin.mean():.3f}, symmetric {sy.mean():.3f}, isotropic {eu.mean():.3f}")
    print("wrote", OUT / "mobile_manipulator.md")


if __name__ == "__main__":
    main()
