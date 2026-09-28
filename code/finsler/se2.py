"""
Left-invariant Finsler metrics on SE(2) from a body-frame velocity envelope.

State q = (x, y, theta); body velocity xi = (v_x, v_y, omega) = (R(theta)^T (xdot, ydot), thetadot).
Given a Minkowski norm F_body on body velocities (a SumRanders with constant coefficients), the
left-invariant metric is  F(q, qdot) = F_body(Lambda(theta)^{-1} qdot),  Lambda = blockdiag(R(theta), 1).

Two equivalent implementations of the geodesics:
  * `se2_metric` : coordinates view, an x-dependent SumRanders (autograd Euler-Lagrange).  Slow, general.
  * `SE2Finsler` : Lie-group view, Euler-Poincare in the body frame
                       qdot = Lambda(theta) xi,   pdot = ad*_xi p,   xi = L_body^{-1}(p),
    with ad*_xi p = (omega p_y, -omega p_x, v_y p_x - v_x p_y) for se(2).  Only the body-frame
    Legendre inverse is needed (a 3x3 Newton solve), no autograd, ~50x faster.  This is the object
    used in the SE(2) experiments.

`quadruped_envelope` is a smooth, asymmetric body-frame envelope shaped like a legged robot's
achievable velocities (fast forward, slow backward, slow sideways, symmetric yaw).
"""
from __future__ import annotations

import numpy as np
import torch

from .general import SumRanders, constant_sum_randers

__all__ = ["se2_metric", "SE2Finsler", "quadruped_envelope", "body_velocity", "world_velocity"]


def _Lambda(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


def body_velocity(q, qdot):
    """xi = Lambda(theta)^{-1} qdot."""
    return torch.einsum("...ji,...j->...i", _Lambda(q[..., 2]), qdot)          # Lambda^T = Lambda^{-1}


def world_velocity(q, xi):
    return torch.einsum("...ij,...j->...i", _Lambda(q[..., 2]), xi)


def se2_metric(Ls_body, b_body) -> SumRanders:
    """Coordinates view: left-invariant SumRanders on SE(2) (autograd geodesics; reference implementation)."""
    Lb = torch.as_tensor(np.asarray(Ls_body), dtype=torch.float64)
    bb = torch.as_tensor(np.asarray(b_body), dtype=torch.float64)

    def Ls(q):
        LamT = _Lambda(q[..., 2]).transpose(-1, -2)
        return torch.einsum("kij,...jl->...kil", Lb, LamT)                # L_k Lambda^{-1}

    def b(q):
        return torch.einsum("...ij,j->...i", _Lambda(q[..., 2]), bb)

    return SumRanders(Ls, b, dim=3)


class SE2Finsler:
    """Lie-group view of the same metric.  All tensors are torch float64, batched over leading dims."""

    def __init__(self, Ls_body, b_body):
        self.body = constant_sum_randers(np.asarray(Ls_body), np.asarray(b_body), dim=3)
        self.dim = 3

    # ---- metric quantities (pulled back to the body frame) --------------------------------------
    def F(self, q, qdot):
        return self.body.F(torch.zeros_like(qdot), body_velocity(q, qdot))

    def F_body(self, xi):
        return self.body.F(torch.zeros_like(xi), xi)

    def fundamental_tensor(self, q, qdot):
        """g in world coordinates: Lambda^{-T} g_body Lambda^{-1}."""
        Lam = _Lambda(q[..., 2])
        gb = self.body.fundamental_tensor(torch.zeros_like(qdot), body_velocity(q, qdot))
        return torch.einsum("...ij,...jk,...lk->...il", Lam, gb, Lam)       # Lambda g_b Lambda^T (Lambda^{-T} = Lambda)

    def legendre_body(self, xi):
        return self.body.legendre(torch.zeros_like(xi), xi)

    def legendre_inv_body(self, p, xi0=None):
        return self.body.legendre_inv(torch.zeros_like(p), p, v0=xi0)

    def project(self, q, u):
        return u / torch.clamp(self.F(q, u), min=1.0)[..., None]

    # ---- Euler-Poincare geodesics ---------------------------------------------------------------------
    @staticmethod
    def _ad_star(xi, p):
        vx, vy, om = xi[..., 0], xi[..., 1], xi[..., 2]
        px, py = p[..., 0], p[..., 1]
        return torch.stack([om * py, -om * px, vy * px - vx * py], -1)

    def _rhs(self, q, p, xi0):
        xi = self.legendre_inv_body(p, xi0)              # warm-started Newton (1-2 iterations along a smooth geodesic)
        return world_velocity(q, xi), self._ad_star(xi, p), xi

    def geodesic(self, q0, w0, T=1.0, n_steps=100):
        """Geodesic with initial (world) velocity w0 for parameter time T.  Returns (qs, ws): (n+1, ..., 3)."""
        q = q0.clone().double()
        xi = body_velocity(q, w0.double())
        p = self.legendre_body(xi)
        dt = T / n_steps
        qs, ws = [q], [w0.double()]
        for _ in range(n_steps):
            k1q, k1p, xi = self._rhs(q, p, xi)
            k2q, k2p, xi = self._rhs(q + 0.5 * dt * k1q, p + 0.5 * dt * k1p, xi)
            k3q, k3p, xi = self._rhs(q + 0.5 * dt * k2q, p + 0.5 * dt * k2p, xi)
            k4q, k4p, xi = self._rhs(q + dt * k3q, p + dt * k3p, xi)
            q = q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
            p = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
            xi = self.legendre_inv_body(p, xi)
            qs.append(q)
            ws.append(world_velocity(q, xi))
        return torch.stack(qs), torch.stack(ws)

    def exp(self, q0, w, n_steps=100):
        qs, ws = self.geodesic(q0, w, 1.0, n_steps)
        return qs[-1], ws[-1]

    def reverse(self) -> "SE2Finsler":
        Lb = self.body.Ls(torch.zeros(1, 3))[0].numpy()
        bb = self.body.b(torch.zeros(1, 3))[0].numpy()
        return SE2Finsler(Lb, -bb)

    def log_batched(self, q0, q1, n_steps=100, iters=20, tol=1e-9, fd_eps=1e-6, w_init=None, wrap_angle=True):
        """Shooting solve of exp_{q0}(w) = q1.  The heading residual is wrapped to (-pi, pi] if wrap_angle."""
        d = 3
        w = (q1 - q0).clone() if w_init is None else w_init.clone()

        def resid(w_):
            r = self.exp(q0, w_, n_steps)[0] - q1
            if wrap_angle:
                r[..., 2] = torch.remainder(r[..., 2] + np.pi, 2 * np.pi) - np.pi
            return r

        r = resid(w)
        err = torch.linalg.norm(r, dim=-1)
        for _ in range(iters):
            active = err > tol
            if not bool(active.any()):
                break
            J = torch.zeros(q0.shape + (d,), dtype=q0.dtype)
            for j in range(d):
                e = torch.zeros(d, dtype=q0.dtype)
                e[j] = fd_eps
                J[..., :, j] = (self.exp(q0, w + e, n_steps)[0] - self.exp(q0, w - e, n_steps)[0]) / (2 * fd_eps)
            dw = -torch.linalg.solve(J, r[..., None])[..., 0]
            step = torch.ones_like(err)
            for _ in range(6):
                w_try = w + step[..., None] * dw
                r_try = resid(w_try)
                err_try = torch.linalg.norm(r_try, dim=-1)
                worse = active & (err_try > err)
                if not bool(worse.any()):
                    break
                step = torch.where(worse, 0.5 * step, step)
            accept = active & (err_try <= err)
            w = torch.where(accept[..., None], w_try, w)
            r = torch.where(accept[..., None], r_try, r)
            err = torch.where(accept, err_try, err)
        return w, err


def quadruped_envelope(v_fwd=1.2, v_back=0.5, v_side=0.4, w_max=1.2, roundness=0.35):
    """Body-frame envelope with the given max speeds along +x, -x, +/-y, +/-yaw.

    Randers term (ellipsoid translated forward) plus a second term that breaks the exact-ellipsoid
    shape (`roundness` = weight of the second term; 0 gives pure Randers).
    Returns (Ls_body, b_body, speeds) with the realised max speeds.
    """
    a_x = 0.5 * (1.0 / v_fwd + 1.0 / v_back)
    b_x = 0.5 * (1.0 / v_back - 1.0 / v_fwd)
    scale = 1.0 / (1.0 + roundness)
    L1 = scale * np.diag([a_x, 1.0 / v_side, 1.0 / w_max])
    L2 = scale * roundness * np.array([[a_x, 0.0, 0.2 * a_x], [0.0, 1.0 / v_side, 0.0], [0.0, 0.0, 1.0 / w_max]])
    b = np.array([-b_x, 0.0, 0.0])                          # unscaled: F(+x) = a_x - b_x = 1/v_fwd, F(-x) = a_x + b_x = 1/v_back
    Ls = np.stack([L1, L2])
    fld = SE2Finsler(Ls, b)
    speeds = {}
    for name, e in [("forward", [1, 0, 0]), ("backward", [-1, 0, 0]), ("left", [0, 1, 0]), ("right", [0, -1, 0]), ("yaw+", [0, 0, 1]), ("yaw-", [0, 0, -1])]:
        speeds[name] = float(1.0 / fld.F_body(torch.tensor([e], dtype=torch.float64)))
    return Ls, b, speeds
