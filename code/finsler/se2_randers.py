"""
Fast numpy implementation of a left-invariant *Randers* metric on SE(2) (body-frame velocity envelope
= translated ellipsoid), with Euler-Poincare geodesics in closed form.

Body metric: RandersField with constant (h, W) on body velocities xi = (v_x, v_y, omega), so the
Legendre inverse is closed form:  xi = L^{-1}(p) = F*(p) (W + h^{-1} p / |p|_{h^{-1}}).
Euler-Poincare:  qdot = Lambda(theta) xi,   pdot = ad*_xi p,  ad*_xi p = (omega p_y, -omega p_x, v_y p_x - v_x p_y).

Roughly 20x faster than the torch SumRanders version (no Newton solve), which is what the SE(2)
experiments need (thousands of BVPs).  `SE2Randers.from_speeds` builds the envelope from the four
max speeds (forward, backward, sideways, yaw).
"""
from __future__ import annotations

import numpy as np

from .randers import RandersField, randers_to_zermelo

__all__ = ["SE2Randers"]


def _Lambda(theta):
    c, s = np.cos(theta), np.sin(theta)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


class SE2Randers:
    def __init__(self, A_body, b_body):
        """F_body(xi) = sqrt(xi^T A xi) + b . xi with |b|_A < 1."""
        self.A = np.asarray(A_body, float)
        self.bvec = np.asarray(b_body, float)
        h, W, lam = randers_to_zermelo(self.A, self.bvec)
        assert lam > 0, "need |b|_A < 1"
        self.h, self.W = h, W
        self.body = RandersField(lambda x: np.broadcast_to(W, np.shape(x)), lambda x: np.broadcast_to(h, np.shape(x) + (3,)), dim=3)
        self.dim = 3

    @classmethod
    def from_speeds(cls, v_fwd=1.2, v_back=0.5, v_side=0.4, w_max=1.2):
        a_x = 0.5 * (1.0 / v_fwd + 1.0 / v_back)
        b_x = 0.5 * (1.0 / v_back - 1.0 / v_fwd)
        A = np.diag([a_x**2, 1.0 / v_side**2, 1.0 / w_max**2])
        return cls(A, np.array([-b_x, 0.0, 0.0]))

    # ---- frames -----------------------------------------------------------------------------------
    @staticmethod
    def body_velocity(q, qdot):
        return np.einsum("...ji,...j->...i", _Lambda(np.asarray(q, float)[..., 2]), np.asarray(qdot, float))

    @staticmethod
    def world_velocity(q, xi):
        return np.einsum("...ij,...j->...i", _Lambda(np.asarray(q, float)[..., 2]), np.asarray(xi, float))

    # ---- metric quantities -------------------------------------------------------------------------
    def F_body(self, xi):
        xi = np.asarray(xi, float)
        return self.body.F(np.zeros_like(xi), xi)

    def F(self, q, qdot):
        return self.F_body(self.body_velocity(q, qdot))

    def speeds(self):
        return {n: float(1.0 / self.F_body(np.array(d, float))) for n, d in [("forward", [1, 0, 0]), ("backward", [-1, 0, 0]), ("left", [0, 1, 0]), ("right", [0, -1, 0]), ("yaw+", [0, 0, 1]), ("yaw-", [0, 0, -1])]}

    def fundamental_tensor(self, q, qdot):
        Lam = _Lambda(np.asarray(q, float)[..., 2])
        xi = self.body_velocity(q, qdot)
        gb = self.body.fundamental_tensor(np.zeros_like(xi), xi)
        return np.einsum("...ij,...jk,...lk->...il", Lam, gb, Lam)

    def legendre_body(self, xi):
        xi = np.asarray(xi, float)
        return self.body.legendre(np.zeros_like(xi), xi)

    def legendre_inv_body(self, p):
        p = np.asarray(p, float)
        return self.body.legendre_inv(np.zeros_like(p), p)

    def project(self, q, u):
        u = np.asarray(u, float)
        return u / np.maximum(self.F(q, u), 1.0)[..., None]

    def reverse(self) -> "SE2Randers":
        return SE2Randers(self.A, -self.bvec)

    # ---- Euler-Poincare geodesics (closed-form RHS) -----------------------------------------------------
    @staticmethod
    def _ad_star(xi, p):
        vx, vy, om = xi[..., 0], xi[..., 1], xi[..., 2]
        px, py = p[..., 0], p[..., 1]
        return np.stack([om * py, -om * px, vy * px - vx * py], -1)

    def _rhs(self, q, p):
        xi = self.legendre_inv_body(p)
        return self.world_velocity(q, xi), self._ad_star(xi, p)

    def geodesic(self, q0, w0, T=1.0, n_steps=100):
        q = np.asarray(q0, float).copy()
        p = self.legendre_body(self.body_velocity(q, w0))
        dt = T / n_steps
        qs, ws = [q], [np.asarray(w0, float)]
        for _ in range(n_steps):
            k1q, k1p = self._rhs(q, p)
            k2q, k2p = self._rhs(q + 0.5 * dt * k1q, p + 0.5 * dt * k1p)
            k3q, k3p = self._rhs(q + 0.5 * dt * k2q, p + 0.5 * dt * k2p)
            k4q, k4p = self._rhs(q + dt * k3q, p + dt * k3p)
            q = q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
            p = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
            qs.append(q)
            ws.append(self.world_velocity(q, self.legendre_inv_body(p)))
        return np.linspace(0, T, n_steps + 1), np.stack(qs), np.stack(ws)

    def exp(self, q0, w, n_steps=100):
        _, qs, ws = self.geodesic(q0, w, 1.0, n_steps)
        return qs[-1], ws[-1]

    def log_batched(self, q0, q1, n_steps=100, iters=20, tol=1e-9, fd_eps=1e-6, w_init=None, wrap_angle=True, max_step=2.0):
        """Shooting solve of exp_{q0}(w) = q1 (Gauss-Newton, FD Jacobian, backtracking, heading wrapped)."""
        q0 = np.asarray(q0, float)
        q1 = np.asarray(q1, float)
        d = 3
        w = (q1 - q0).copy() if w_init is None else np.asarray(w_init, float).copy()

        def resid(w_):
            r = self.exp(q0, w_, n_steps)[0] - q1
            if wrap_angle:
                r[..., 2] = np.remainder(r[..., 2] + np.pi, 2 * np.pi) - np.pi
            return r

        r = resid(w)
        err = np.linalg.norm(r, axis=-1)
        for _ in range(iters):
            active = err > tol
            if not active.any():
                break
            J = np.zeros(q0.shape + (d,))
            for j in range(d):
                e = np.zeros(d)
                e[j] = fd_eps
                J[..., :, j] = (self.exp(q0, w + e, n_steps)[0] - self.exp(q0, w - e, n_steps)[0]) / (2 * fd_eps)
            dw = -np.linalg.solve(J, r[..., None])[..., 0]
            nrm = np.linalg.norm(dw, axis=-1, keepdims=True)
            dw = dw * np.minimum(1.0, max_step / np.maximum(nrm, 1e-12))
            step = np.ones(q0.shape[:-1])
            for _ in range(6):
                w_try = w + step[..., None] * dw
                r_try = resid(w_try)
                err_try = np.linalg.norm(r_try, axis=-1)
                worse = active & (err_try > err)
                if not worse.any():
                    break
                step = np.where(worse, 0.5 * step, step)
            accept = active & (err_try <= err)
            w = np.where(accept[..., None], w_try, w)
            r = np.where(accept[..., None], r_try, r)
            err = np.where(accept, err_try, err)
        return w, err
