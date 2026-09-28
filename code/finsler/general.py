"""
General (non-Randers) Finsler metrics as sums of Randers-type terms, in torch.

    F(x, v) = sum_k || L_k(x) v ||  +  b(x) . v,        b(x) = rho(x) L_1(x)^T u(x),  rho in [0, 1), |u| <= 1

* positive 1-homogeneous by construction,
* strongly convex: Hess_v F = sum_k Hess ||L_k v|| >= Hess ||L_1 v|| > 0 on the hyperplane transverse to v,
* positive: b.v >= -rho |u| ||L_1 v|| > -||L_1 v||  so  F > 0 for v != 0.

One term (K = 1) is exactly a Randers metric; K >= 2 gives indicatrices that are not translated
ellipsoids (see `egg_indicatrix`).  Everything is closed form except L^{-1} (Newton, quadratic
convergence, Jacobian = g_v) and geodesics (Euler-Lagrange with autograd in x).

Fields L_k(x), b(x) are torch callables, so they can be small networks: that is the "learned
velocity envelope" route (`fit_envelope` gives a minimal example).
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["SumRanders", "constant_sum_randers", "egg_indicatrix"]


class SumRanders:
    """F(x, v) = sum_k ||L_k(x) v|| + b(x).v.   Ls(x) -> (..., K, d, d),  b(x) -> (..., d)."""

    def __init__(self, Ls, b, dim=2):
        self.Ls = Ls
        self.b = b
        self.dim = dim

    # ---- basic quantities (closed form) ----------------------------------------------------------
    def _terms(self, x, v):
        L = self.Ls(x)                                            # (..., K, d, d)
        Lv = torch.einsum("...kij,...j->...ki", L, v)             # (..., K, d)
        alpha = torch.linalg.norm(Lv, dim=-1)                     # (..., K)
        return L, Lv, alpha

    def F(self, x, v):
        _, _, alpha = self._terms(x, v)
        return alpha.sum(-1) + (self.b(x) * v).sum(-1)

    def dF(self, x, v):
        """dF/dv (0-homogeneous covector)."""
        L, Lv, alpha = self._terms(x, v)
        LtLv = torch.einsum("...kji,...kj->...ki", L, Lv)          # L_k^T L_k v
        return (LtLv / alpha[..., None]).sum(-2) + self.b(x)

    def legendre(self, x, v):
        return self.F(x, v)[..., None] * self.dF(x, v)

    def fundamental_tensor(self, x, v):
        """g_v = dF dF^T + F * sum_k ( A_k/alpha_k - (A_k v)(A_k v)^T / alpha_k^3 ),  A_k = L_k^T L_k."""
        L, Lv, alpha = self._terms(x, v)
        A = torch.einsum("...kji,...kjl->...kil", L, L)             # (..., K, d, d)
        Av = torch.einsum("...kij,...j->...ki", A, v)
        dF = (Av / alpha[..., None]).sum(-2) + self.b(x)
        F = alpha.sum(-1) + (self.b(x) * v).sum(-1)
        hessF = (A / alpha[..., None, None] - Av[..., :, None] * Av[..., None, :] / alpha[..., None, None] ** 3).sum(-3)
        return dF[..., :, None] * dF[..., None, :] + F[..., None, None] * hessF

    # ---- Legendre inverse by Newton ----------------------------------------------------------------
    def legendre_inv(self, x, p, iters=30, tol=1e-12, v0=None):
        """Solve L(v) = p.  Newton with Jacobian g_v; initial guess v0 or from the first (Riemannian) term."""
        if v0 is None:
            L = self.Ls(x)
            A1 = torch.einsum("...ji,...jl->...il", L[..., 0, :, :], L[..., 0, :, :])
            v = torch.linalg.solve(A1, p[..., None])[..., 0]
        else:
            v = v0.clone()
        for _ in range(iters):
            r = self.legendre(x, v) - p
            if torch.linalg.norm(r, dim=-1).max() < tol:
                break
            g = self.fundamental_tensor(x, v)
            dv = -torch.linalg.solve(g, r[..., None])[..., 0]
            # damped step: keep v away from 0 (F has a kink there)
            step = torch.ones_like(dv[..., 0])
            for _ in range(8):
                v_try = v + step[..., None] * dv
                r_try = self.legendre(x, v_try) - p
                worse = torch.linalg.norm(r_try, dim=-1) > torch.linalg.norm(r, dim=-1)
                if not worse.any():
                    break
                step = torch.where(worse, 0.5 * step, step)
            v = v + step[..., None] * dv
        return v

    def F_dual(self, x, p):
        return self.F(x, self.legendre_inv(x, p))

    def grad(self, x, df):
        return self.legendre_inv(x, df)

    # ---- geodesics: Euler-Lagrange for E = F^2/2,  g_v vdot = d_x E - (d_x p) v -------------------
    def acceleration(self, x, v):
        """vdot from the Euler-Lagrange equation of E = F^2/2:  g_v vdot = d_x E - (d_x p) v,  p = d_v E."""
        x = x.detach()
        v = v.detach()
        xg = x.clone().requires_grad_(True)
        E = 0.5 * self.F(xg, v) ** 2
        if E.requires_grad:
            dE_dx = torch.autograd.grad(E.sum(), xg, allow_unused=True)[0]
            if dE_dx is None:
                dE_dx = torch.zeros_like(x)
        else:                                                                        # F independent of x
            dE_dx = torch.zeros_like(x)
        _, dp_dx_v = torch.func.jvp(lambda xx: self.legendre(xx, v), (x,), (v,))    # (d_x p) v  via forward-mode
        g = self.fundamental_tensor(x, v)
        return torch.linalg.solve(g, (dE_dx - dp_dx_v)[..., None])[..., 0]

    def geodesic(self, x0, v0, T=1.0, n_steps=100):
        x, v = x0.clone().double(), v0.clone().double()
        dt = T / n_steps
        xs, vs = [x], [v]
        for _ in range(n_steps):
            k1x, k1v = v, self.acceleration(x, v)
            k2x, k2v = v + 0.5 * dt * k1v, self.acceleration(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
            k3x, k3v = v + 0.5 * dt * k2v, self.acceleration(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
            k4x, k4v = v + dt * k3v, self.acceleration(x + dt * k3x, v + dt * k3v)
            x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            xs.append(x)
            vs.append(v)
        return torch.stack(xs), torch.stack(vs)

    def exp(self, x0, w, n_steps=100):
        xs, vs = self.geodesic(x0, w, 1.0, n_steps)
        return xs[-1], vs[-1]

    def reverse(self) -> "SumRanders":
        return SumRanders(self.Ls, lambda x: -self.b(x), self.dim)

    def log_batched(self, x0, x1, n_steps=100, iters=20, tol=1e-9, fd_eps=1e-6, w_init=None):
        """Shooting solve of exp_{x0}(w) = x1 (batched Gauss-Newton, finite-difference Jacobian, backtracking)."""
        d = self.dim
        w = (x1 - x0).clone() if w_init is None else w_init.clone()
        r = self.exp(x0, w, n_steps)[0] - x1
        err = torch.linalg.norm(r, dim=-1)
        for _ in range(iters):
            active = err > tol
            if not bool(active.any()):
                break
            J = torch.zeros(x0.shape + (d,), dtype=x0.dtype)
            for j in range(d):
                e = torch.zeros(d, dtype=x0.dtype)
                e[j] = fd_eps
                J[..., :, j] = (self.exp(x0, w + e, n_steps)[0] - self.exp(x0, w - e, n_steps)[0]) / (2 * fd_eps)
            dw = -torch.linalg.solve(J, r[..., None])[..., 0]
            step = torch.ones_like(err)
            for _ in range(6):
                w_try = w + step[..., None] * dw
                r_try = self.exp(x0, w_try, n_steps)[0] - x1
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

    # ---- control utilities -----------------------------------------------------------------------------
    def project(self, x, u):
        return u / torch.clamp(self.F(x, u), min=1.0)[..., None]

    def cov_finsler(self, x, u):
        g = self.fundamental_tensor(x, u)
        C = torch.linalg.inv(g)
        return C / torch.linalg.det(C)[..., None, None] ** (1.0 / self.dim)


# --------------------------------------------------------------------------------------------------
def constant_sum_randers(Ls, b, dim=2):
    """Constant-coefficient SumRanders from arrays Ls (K, d, d) and b (d,)."""
    Ls_t = torch.as_tensor(np.asarray(Ls), dtype=torch.float64)
    b_t = torch.as_tensor(np.asarray(b), dtype=torch.float64)
    return SumRanders(lambda x: Ls_t.expand(x.shape[:-1] + Ls_t.shape), lambda x: b_t.expand(x.shape[:-1] + b_t.shape), dim)


def egg_indicatrix(n=720, Ls=None, b=None):
    """Points of the indicatrix {F = 1} of a two-term sum (an 'egg', not a translated ellipse)."""
    if Ls is None:
        Ls = np.array([[[1.0, 0.0], [0.0, 1.6]], [[0.6, 0.5], [0.0, 0.4]]])
    if b is None:
        b = np.array([0.35, 0.0])
    fld = constant_sum_randers(Ls, b)
    th = torch.linspace(0, 2 * np.pi, n, dtype=torch.float64)
    e = torch.stack([torch.cos(th), torch.sin(th)], -1)
    x = torch.zeros_like(e)
    return (e / fld.F(x, e)[..., None]).numpy(), fld
