"""Manipulator task-space planning under a non-reversible velocity envelope.

The shoulder-elbow plane of a UR5: joints 2 and 3 move the wrist in a vertical plane, joint 1 rotates
that plane, so the planar reduction carries the arm's real link lengths (0.425 m upper arm, 0.392 m
forearm), link masses and 120 deg/s joint speed limit, with the rated 5 kg payload at the tip.  The
joints are additionally given a budget on the gravitational power they may consume.  Writing u = J(q)^{-1} v for the joint velocity
that realises an end-effector velocity v, the achievable set is

    B(x) = { v : |u_i| <= w_i ,  g(q)^T u <= P_max }

whose gauge is closed form,

    F(x,v) = max( |u_1|/w_1 , |u_2|/w_2 , max(0, g(q)^T u)/P_max ).

The third term switches on only when the motion consumes gravitational power, so raising the payload is
capped and lowering it is not: F(x,v) != F(x,-v) by construction, and the asymmetry is physics, not a
modelling choice.  Grasping a heavier object increases g(q) and tightens only the upward half.

The experiment mirrors the envelope-identification study of the paper: plan under each geometry, execute
under the TRUE envelope, and report the time ratio against the true optimum.

Run: python experiments/arm_envelope.py [--payload 3.0] [--h 0.02]
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

from finsler.eikonal import GridFinslerDistance

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "arm_envelope"
OUT.mkdir(parents=True, exist_ok=True)
G_ACC = 9.81


class Arm:
    """Two-link planar arm in the vertical plane, elbow-up branch."""

    def __init__(self, l1=0.425, l2=0.392, m1=8.39, m2=3.50, payload=5.0,
                 w=(2.094, 2.094), p_max=60.0,
                 workpiece=None, e_max=2.0, zone=0.35, r_min=0.26, r_max=0.76):
        self.l1, self.l2, self.m1, self.m2, self.mp = l1, l2, m1, m2, payload
        self.w = np.asarray(w, float)
        self.p_max = p_max
        self.workpiece = None if workpiece is None else np.asarray(workpiece, float)
        self.e_max, self.zone = e_max, zone
        # the Jacobian degenerates at the folded (r -> 0) and fully extended (r -> l1 + l2)
        # configurations, where the gauge of some directions collapses; keep planning inside an annulus
        self.r_min, self.r_max = r_min, r_max

    def inertia(self, q):
        """Joint-space inertia of the two-link arm with a point payload at the tip."""
        lc1, lc2 = self.l1 / 2, self.l2 / 2
        I1, I2 = self.m1 * self.l1 ** 2 / 12, self.m2 * self.l2 ** 2 / 12
        c2 = np.cos(q[:, 1])
        m11 = (self.m1 * lc1 ** 2 + I1 + self.m2 * (self.l1 ** 2 + lc2 ** 2 + 2 * self.l1 * lc2 * c2) + I2
               + self.mp * (self.l1 ** 2 + self.l2 ** 2 + 2 * self.l1 * self.l2 * c2))
        m12 = (self.m2 * (lc2 ** 2 + self.l1 * lc2 * c2) + I2
               + self.mp * (self.l2 ** 2 + self.l1 * self.l2 * c2))
        m22 = np.full_like(c2, self.m2 * lc2 ** 2 + I2 + self.mp * self.l2 ** 2)
        M = np.empty((len(q), 2, 2))
        M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1] = m11, m12, m12, m22
        return M

    def m_eff(self, x, d):
        """Effective mass seen at the end effector along unit directions d: 1 / (d^T J M^-1 J^T d)."""
        x = np.atleast_2d(np.asarray(x, float))
        d = np.atleast_2d(np.asarray(d, float))
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        q = self.ik(x)
        J, M = self.jac(q), self.inertia(q)
        Minv = np.linalg.inv(M)
        Lam_inv = np.einsum("nij,njk,nlk->nil", J, Minv, J)
        return 1.0 / np.maximum(np.einsum("ni,nij,nj->n", d, Lam_inv, d), 1e-12)

    # ---- kinematics ------------------------------------------------------------------------------
    def ik(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        r2 = np.sum(x ** 2, 1)
        c2 = np.clip((r2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2), -1.0, 1.0)
        q2 = np.arccos(c2)                                   # elbow-up
        k1 = self.l1 + self.l2 * np.cos(q2)
        k2 = self.l2 * np.sin(q2)
        q1 = np.arctan2(x[:, 1], x[:, 0]) - np.arctan2(k2, k1)
        return np.stack([q1, q2], 1)

    def jac(self, q):
        q1, q12 = q[:, 0], q[:, 0] + q[:, 1]
        s1, c1, s12, c12 = np.sin(q1), np.cos(q1), np.sin(q12), np.cos(q12)
        J = np.empty((len(q), 2, 2))
        J[:, 0, 0] = -self.l1 * s1 - self.l2 * s12
        J[:, 0, 1] = -self.l2 * s12
        J[:, 1, 0] = self.l1 * c1 + self.l2 * c12
        J[:, 1, 1] = self.l2 * c12
        return J

    def gravity(self, q):
        """Joint torques that hold the arm against gravity."""
        q1, q12 = q[:, 0], q[:, 0] + q[:, 1]
        a = (self.m1 * self.l1 / 2 + (self.m2 + self.mp) * self.l1)
        b = (self.m2 * self.l2 / 2 + self.mp * self.l2)
        g1 = G_ACC * (a * np.cos(q1) + b * np.cos(q12))
        g2 = G_ACC * b * np.cos(q12)
        return np.stack([g1, g2], 1)

    # ---- envelope --------------------------------------------------------------------------------
    def F(self, x, v):
        """Exact gauge of the achievable end-effector velocity set."""
        x = np.atleast_2d(np.asarray(x, float))
        v = np.atleast_2d(np.asarray(v, float))
        q = self.ik(x)
        J = self.jac(q)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        det = np.where(np.abs(det) < 1e-9, np.sign(det) * 1e-9 + 1e-12, det)
        Jinv = np.empty_like(J)
        Jinv[:, 0, 0], Jinv[:, 0, 1] = J[:, 1, 1] / det, -J[:, 0, 1] / det
        Jinv[:, 1, 0], Jinv[:, 1, 1] = -J[:, 1, 0] / det, J[:, 0, 0] / det
        u = np.einsum("nij,nj->ni", Jinv, v)
        g = self.gravity(q)
        power = np.sum(g * u, 1)
        F = np.maximum(np.maximum(np.abs(u[:, 0]) / self.w[0], np.abs(u[:, 1]) / self.w[1]),
                       np.maximum(power, 0.0) / self.p_max)
        if self.workpiece is not None:
            # power-and-force limiting: closing on the workpiece is capped by the energy that would be
            # transferred on contact, 1/2 m_eff v^2 <= e_max, and retreating is not capped at all.
            rel = self.workpiece[None] - x
            dist = np.linalg.norm(rel, axis=1)
            nhat = rel / np.maximum(dist, 1e-12)[:, None]
            v_safe = np.sqrt(2 * self.e_max / self.m_eff(x, nhat))
            closing = np.maximum(np.sum(v * nhat, 1), 0.0)
            active = dist < self.zone
            F = np.maximum(F, np.where(active, closing / np.maximum(v_safe, 1e-9), 0.0))
        r = np.linalg.norm(x, axis=1)
        return np.where((r < self.r_min) | (r > self.r_max), 1e6, F)

    def max_speed(self, x, dirs):
        """Speed available in each unit direction (1 / gauge)."""
        n = len(dirs)
        xs = np.repeat(np.atleast_2d(x), n, 0)
        return 1.0 / np.maximum(self.F(xs, dirs), 1e-12)


# --------------------------------------------------------------------------------------------------
class SymmetrisedField:
    """The reversible metric closest to the truth: F_bar(v) = (F(v) + F(-v)) / 2.

    Convex and one-homogeneous because F is, even by construction, and it keeps every bit of the
    anisotropy while discarding exactly the asymmetry.  This is a stronger symmetric baseline than any
    fitted ellipsoid, and it needs no fit, so the comparison isolates non-reversibility alone."""

    def __init__(self, arm):
        self.arm = arm

    def F(self, x, v):
        v = np.atleast_2d(np.asarray(v, float))
        return 0.5 * (self.arm.F(x, v) + self.arm.F(x, -v))


class EuclideanField:
    """Isotropic metric scaled to the mean achievable speed: knows neither anisotropy nor asymmetry.

    It does respect the reachable annulus, which is kinematics rather than a modelling choice, so that
    the comparison is about the shape of the envelope and not about who knows where the arm can go."""

    def __init__(self, scale, arm=None):
        self.scale = scale
        self.arm = arm

    def F(self, x, v):
        x = np.atleast_2d(np.asarray(x, float))
        v = np.atleast_2d(np.asarray(v, float))
        f = np.linalg.norm(v, axis=1) / self.scale
        if self.arm is not None:
            r = np.linalg.norm(x, axis=1)
            f = np.where((r < self.arm.r_min) | (r > self.arm.r_max), 1e6, f)
        return f


def path_from(graph, src, dst):
    d, pred = dijkstra(graph, directed=True, indices=src, return_predecessors=True)
    if not np.isfinite(d[dst]):
        return None
    path = [dst]
    while path[-1] != src:
        p = pred[path[-1]]
        if p < 0:
            return None
        path.append(p)
    return np.array(path[::-1])


def exec_time(arm, pts):
    d = pts[1:] - pts[:-1]
    mid = 0.5 * (pts[1:] + pts[:-1])
    return float(arm.F(mid, d).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", type=float, default=5.0)
    ap.add_argument("--h", type=float, default=0.02)
    ap.add_argument("--workpiece", default="", help="'x,y' to enable the safe-approach constraint")
    ap.add_argument("--e-max", type=float, default=2.0)
    ap.add_argument("--zone", type=float, default=0.35)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    wp = None if not args.workpiece else [float(t) for t in args.workpiece.split(",")]
    arm = Arm(payload=args.payload, workpiece=wp, e_max=args.e_max, zone=args.zone)
    if wp is not None:
        print(f"safe-approach constraint active: workpiece at {wp}, E_max {args.e_max} J, zone {args.zone} m")
    # kept away from the singular boundary (r -> l1 + l2) and from the folded configuration (r -> 0)
    xlim, ylim = (-0.60, 0.60), (-0.10, 0.70)
    print(f"UR5 shoulder-elbow plane: links {arm.l1}/{arm.l2} m, payload {args.payload} kg, "
          f"joint speed {arm.w[0]:.2f} rad/s (120 deg/s), gravitational power budget {arm.p_max} W")

    # ---- how asymmetric is the envelope, really? -------------------------------------------------
    probes = np.array([[0.50, 0.25], [0.0, 0.60], [-0.40, 0.30], [0.30, 0.48]])
    up, dn, lf, rt = [], [], [], []
    for p in probes:
        s = arm.max_speed(p, np.array([[0, 1.0], [0, -1.0], [-1.0, 0], [1.0, 0]]))
        up.append(s[0]); dn.append(s[1]); lf.append(s[2]); rt.append(s[3])
    print("  max speed [m/s] at four probes:")
    for i, p in enumerate(probes):
        print(f"    ({p[0]:+.2f},{p[1]:+.2f})  up {up[i]:.3f}  down {dn[i]:.3f}  ratio {dn[i]/up[i]:.2f}"
              f"   left {lf[i]:.3f}  right {rt[i]:.3f}")

    fin = arm                                     # plan with the exact envelope
    sym = SymmetrisedField(arm)                   # plan with its reversible symmetrisation
    ref_dirs = np.stack([np.cos(np.linspace(0, 2 * np.pi, 64, endpoint=False)),
                         np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False))], 1)
    mean_speed = float(np.mean([arm.max_speed(p, ref_dirs).mean() for p in probes]))
    euc = EuclideanField(mean_speed, arm)

    # ---- grids -----------------------------------------------------------------------------------
    grids, t0 = {}, time.time()
    for name, field in [("true", arm), ("finsler", fin), ("symmetric", sym), ("euclid", euc)]:
        grids[name] = GridFinslerDistance(field, xlim, ylim, args.h, radius=3)
    print(f"  four grids built ({grids['true'].n_edges/1e6:.1f}M edges each, {time.time()-t0:.0f}s)", flush=True)

    # ---- queries: lift the payload and put it down ------------------------------------------------
    queries = [(np.array([0.50, 0.00]), np.array([0.05, 0.62]), "lift, out to up"),
               (np.array([0.05, 0.62]), np.array([0.50, 0.00]), "lower, up to out"),
               (np.array([-0.45, 0.10]), np.array([0.45, 0.10]), "traverse, left to right"),
               (np.array([0.40, 0.50]), np.array([-0.40, 0.15]), "across and down")]
    if arm.workpiece is not None:
        wp = arm.workpiece
        queries = [(np.array([-0.40, 0.45]), wp.copy(), "reach the workpiece from upper left"),
                   (np.array([0.48, 0.42]), wp.copy(), "reach the workpiece from upper right"),
                   (np.array([-0.48, 0.05]), wp.copy(), "reach the workpiece from lower left"),
                   (np.array([0.05, 0.62]), wp.copy(), "reach the workpiece from above")]

    rows = []
    for s, g, label in queries:
        src, dst = int(grids["true"].node_of(s)[0]), int(grids["true"].node_of(g)[0])
        t_opt = float(dijkstra(grids["true"].graph, directed=True, indices=src)[dst])
        rec = dict(query=label, optimal=t_opt)
        for name in ("finsler", "symmetric", "euclid"):
            idx = path_from(grids[name].graph, src, dst)
            if idx is None:
                rec[name] = float("nan"); continue
            pts = grids[name].nodes[idx]
            rec[name] = exec_time(arm, pts) / t_opt
            rec[name + "_path"] = pts
        rows.append(rec)
        print(f"  {label:26s} optimal {t_opt:5.2f} s | finsler {rec['finsler']:.3f} "
              f"| symmetric {rec['symmetric']:.3f} | euclid {rec['euclid']:.3f}", flush=True)

    fin_r = np.array([r["finsler"] for r in rows])
    sym_r = np.array([r["symmetric"] for r in rows])
    euc_r = np.array([r["euclid"] for r in rows])
    lines = ["# UR5 (shoulder-elbow plane): planning under a non-reversible task-space velocity envelope", "",
             f"UR5 link lengths {arm.l1}/{arm.l2} m, payload {args.payload} kg, joint speed "
             f"{arm.w[0]:.2f} rad/s (120 deg/s), gravitational power "
             f"budget {arm.p_max} W. The envelope is the exact achievable end-effector velocity set; its gauge "
             "is non-reversible because the power budget binds only when the motion consumes gravitational "
             "power.", "",
             "The symmetric baseline is the exact symmetrisation (F(v) + F(-v))/2, which keeps all of the "
             "anisotropy and discards exactly the non-reversibility, so the comparison isolates asymmetry "
             "with no fitting error in between.", "",
             "Max end-effector speed, showing the asymmetry the symmetric metric must average away:", "",
             "| workspace point | up [m/s] | down [m/s] | down/up |", "|---|---:|---:|---:|"]
    for i, p in enumerate(probes):
        lines.append(f"| ({p[0]:+.2f}, {p[1]:+.2f}) | {up[i]:.3f} | {dn[i]:.3f} | {dn[i]/up[i]:.2f} |")
    lines += ["", "Plans are computed under each geometry and executed under the true envelope; the reference "
              "is the grid optimum under the true envelope.", "",
              "| query | optimal [s] | Finsler | symmetric part | Euclidean |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['query']} | {r['optimal']:.2f} | {r['finsler']:.3f} | {r['symmetric']:.3f} "
                     f"| {r['euclid']:.3f} |")
    lines += ["", f"Mean over the four queries: Finsler {fin_r.mean():.3f}, symmetric part {sym_r.mean():.3f}, "
              f"Euclidean {euc_r.mean():.3f}."]
    (OUT / f"arm_envelope{args.tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.savez(OUT / f"arm_paths{args.tag}.npz", probes=probes,
             **{f"{r['query'].replace(' ', '_').replace(',', '')}__{n}": r[n + "_path"]
                for r in rows for n in ("finsler", "symmetric", "euclid") if n + "_path" in r})
    json.dump(dict(payload=args.payload,
                   finsler=float(fin_r.mean()), symmetric=float(sym_r.mean()), euclid=float(euc_r.mean())),
              open(OUT / f"arm_envelope{args.tag}.json", "w"), indent=1)
    print(f"\nmean ratio: Finsler {fin_r.mean():.3f}, symmetric {sym_r.mean():.3f}, Euclidean {euc_r.mean():.3f}")
    print("wrote", OUT / f"arm_envelope{args.tag}.md")


if __name__ == "__main__":
    main()
