"""
Randers metrics on an open subset of R^d, specified through Zermelo navigation data.

    F(x, v) = sqrt(v^T A(x) v) + b(x)^T v,          |b|_A < 1
    indicatrix {F(x, .) = 1} = { W(x) + e : |e|_{h(x)} = 1 }

(h, W) <-> (A, b) is the Bao-Robles-Shen correspondence.  The dual (co-)norm of a Randers
metric is again of Randers type and closed form,

    F*(x, p) = p . W(x) + |p|_{h(x)^{-1}},

so the Legendre transform and its inverse, the Finsler gradient, and Hamilton's equations for
geodesics (H = F*^2 / 2) are all explicit.  x-derivatives use central finite differences so any
callable field (analytic or learned) plugs in without autodiff.

Conventions
-----------
* All functions are vectorised over leading batch dimensions: x, v, p have shape (..., d).
* v is a tangent vector, p = L(v) = g_v(v, .) = F dF/dv a covector.
* The reverse metric F̄(x, v) = F(x, -v) is `field.reverse()` (W -> -W).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "RandersField",
    "zermelo_to_randers",
    "randers_to_zermelo",
    "cov_isotropic",
    "cov_riemannian",
    "cov_finsler",
]


# --------------------------------------------------------------------------------------
# Zermelo <-> Randers
# --------------------------------------------------------------------------------------
def zermelo_to_randers(h, W):
    """(h, W) -> (A, b, lam).  h: (..., d, d) SPD, W: (..., d) with |W|_h < 1."""
    h = np.asarray(h, float)
    W = np.asarray(W, float)
    hW = np.einsum("...ij,...j->...i", h, W)
    lam = 1.0 - np.einsum("...i,...i->...", W, hW)
    A = (lam[..., None, None] * h + hW[..., :, None] * hW[..., None, :]) / lam[..., None, None] ** 2
    b = -hW / lam[..., None]
    return A, b, lam


def randers_to_zermelo(A, b):
    """(A, b) -> (h, W, lam).  Inverse of `zermelo_to_randers`."""
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    Ainv_b = np.linalg.solve(A, b[..., None])[..., 0]
    lam = 1.0 - np.einsum("...i,...i->...", b, Ainv_b)
    h = lam[..., None, None] * (A - b[..., :, None] * b[..., None, :])
    W = -Ainv_b / lam[..., None]
    return h, W, lam


def _outer(a, b):
    return a[..., :, None] * b[..., None, :]


# --------------------------------------------------------------------------------------
# The metric
# --------------------------------------------------------------------------------------
class RandersField:
    """Randers metric given by wind W(x) and (optional) Riemannian h(x); h defaults to identity."""

    def __init__(self, W, h=None, dim=2, fd_eps=1e-6, scale=None):
        """scale(x) -> (...,) optional conformal factor s(x) > 0:  F = s(x) * F_randers  (obstacles = slow regions).
        The dual norm is F* = F_randers*/s, so everything stays closed form."""
        self._W = W
        self._h = h
        self.dim = dim
        self.eps = fd_eps
        self.h_is_identity = h is None
        self._scale = scale

    # ---- fields ----------------------------------------------------------------------
    def W(self, x):
        x = np.asarray(x, float)
        return np.broadcast_to(np.asarray(self._W(x), float), x.shape).copy()

    def h(self, x):
        x = np.asarray(x, float)
        if self._h is None:
            return np.broadcast_to(np.eye(self.dim), x.shape + (self.dim,)).copy()
        return np.broadcast_to(np.asarray(self._h(x), float), x.shape + (self.dim,)).copy()

    def hinv(self, x):
        if self.h_is_identity:
            x = np.asarray(x, float)
            return np.broadcast_to(np.eye(self.dim), x.shape + (self.dim,)).copy()
        h = self.h(x)
        if self.dim == 2:                                   # analytic 2x2 inverse (np.linalg.inv is slow on many small matrices)
            a, b, c, d = h[..., 0, 0], h[..., 0, 1], h[..., 1, 0], h[..., 1, 1]
            det = a * d - b * c
            out = np.empty_like(h)
            out[..., 0, 0] = d / det
            out[..., 0, 1] = -b / det
            out[..., 1, 0] = -c / det
            out[..., 1, 1] = a / det
            return out
        return np.linalg.inv(h)

    def randers(self, x):
        """(A, b, lam) at x (of the unscaled Randers part)."""
        return zermelo_to_randers(self.h(x), self.W(x))

    def s(self, x):
        x = np.asarray(x, float)
        if self._scale is None:
            return np.ones(x.shape[:-1])
        return np.broadcast_to(np.asarray(self._scale(x), float), x.shape[:-1]).copy()

    def reverse(self) -> "RandersField":
        """F̄(x, v) = F(x, -v): Zermelo data (h, -W), same scale."""
        return RandersField(lambda x: -self.W(x), self._h, self.dim, self.eps, self._scale)

    # ---- primal side -------------------------------------------------------------------
    def F(self, x, v):
        A, b, _ = self.randers(x)
        v = np.asarray(v, float)
        alpha = np.sqrt(np.einsum("...i,...ij,...j->...", v, A, v))
        return self.s(x) * (alpha + np.einsum("...i,...i->...", b, v))

    def F_zermelo(self, x, v):
        """Same as F, computed from the navigation picture: F(v) = t with |v/t - W|_h = 1."""
        h, W = self.h(x), self.W(x)
        v = np.asarray(v, float)
        hW = np.einsum("...ij,...j->...i", h, W)
        lam = 1.0 - np.einsum("...i,...i->...", W, hW)
        vhW = np.einsum("...i,...i->...", v, hW)
        vhv = np.einsum("...i,...ij,...j->...", v, h, v)
        return self.s(x) * (-vhW + np.sqrt(vhW**2 + lam * vhv)) / lam

    def ell(self, x, v):
        """dF/dv (0-homogeneous covector)."""
        A, b, _ = self.randers(x)
        v = np.asarray(v, float)
        Av = np.einsum("...ij,...j->...i", A, v)
        alpha = np.sqrt(np.einsum("...i,...i->...", v, Av))
        return self.s(x)[..., None] * (Av / alpha[..., None] + b)

    def legendre(self, x, v):
        """L(v) = g_v(v, .) = F(v) dF/dv  (tangent -> cotangent, 1-homogeneous, nonlinear)."""
        return self.F(x, v)[..., None] * self.ell(x, v)

    def fundamental_tensor(self, x, v):
        """g_v = Hess_v(F^2 / 2) = ell ell^T + F (A/alpha - Av Av^T / alpha^3)."""
        A, b, _ = self.randers(x)
        v = np.asarray(v, float)
        Av = np.einsum("...ij,...j->...i", A, v)
        alpha = np.sqrt(np.einsum("...i,...i->...", v, Av))
        F = alpha + np.einsum("...i,...i->...", b, v)
        ell = Av / alpha[..., None] + b
        g = _outer(ell, ell) + F[..., None, None] * (
            A / alpha[..., None, None] - _outer(Av, Av) / alpha[..., None, None] ** 3
        )
        return self.s(x)[..., None, None] ** 2 * g

    # ---- dual side ---------------------------------------------------------------------
    def F_dual(self, x, p):
        """F*(p) = sup_{F(v)<=1} p.v = p.W + |p|_{h^{-1}}."""
        W, hinv = self.W(x), self.hinv(x)
        p = np.asarray(p, float)
        return (np.einsum("...i,...i->...", p, W) + np.sqrt(np.einsum("...i,...ij,...j->...", p, hinv, p))) / self.s(x)

    def ell_dual(self, x, p):
        """dF*/dp = W + h^{-1} p / |p|_{h^{-1}}  (a point on the indicatrix)."""
        W, hinv = self.W(x), self.hinv(x)
        p = np.asarray(p, float)
        hp = np.einsum("...ij,...j->...i", hinv, p)
        nrm = np.sqrt(np.einsum("...i,...i->...", p, hp))
        return (W + hp / nrm[..., None]) / self.s(x)[..., None]

    def legendre_inv(self, x, p):
        """L^{-1}(p) = F*(p) dF*/dp  (cotangent -> tangent)."""
        return self.F_dual(x, p)[..., None] * self.ell_dual(x, p)

    def grad(self, x, df):
        """Finsler gradient of f from its differential: grad f = L^{-1}(df).  NB grad(-f) != -grad f."""
        return self.legendre_inv(x, df)

    # ---- geodesics (Hamiltonian form) ---------------------------------------------------
    def hamiltonian(self, x, p):
        return 0.5 * self.F_dual(x, p) ** 2

    def _dual_parts(self, x, p):
        W, hinv = self.W(x), self.hinv(x)
        hp = np.einsum("...ij,...j->...i", hinv, p)
        nrm = np.sqrt(np.einsum("...i,...i->...", p, hp))
        F0 = np.einsum("...i,...i->...", p, W) + nrm
        return W, hinv, hp, nrm, F0

    def _dF0_dx(self, x, p, nrm):
        """x-derivative of the unscaled dual norm F0*(x, p) by central differences on the fields."""
        out = np.zeros(np.broadcast(x, p).shape)
        for i in range(self.dim):
            e = np.zeros(self.dim)
            e[i] = self.eps
            dW = (self.W(x + e) - self.W(x - e)) / (2 * self.eps)
            dF = np.einsum("...i,...i->...", p, dW)
            if not self.h_is_identity:
                dhinv = (self.hinv(x + e) - self.hinv(x - e)) / (2 * self.eps)
                dF = dF + 0.5 * np.einsum("...i,...ij,...j->...", p, dhinv, p) / nrm
            out[..., i] = dF
        return out

    def _ds_dx(self, x):
        out = np.zeros(np.asarray(x, float).shape)
        if self._scale is None:
            return out
        for i in range(self.dim):
            e = np.zeros(self.dim)
            e[i] = self.eps
            out[..., i] = (self.s(x + e) - self.s(x - e)) / (2 * self.eps)
        return out

    def dH_dx(self, x, p):
        """dH/dx for H = (F0*/s)^2 / 2."""
        x = np.asarray(x, float)
        p = np.asarray(p, float)
        W, hinv, hp, nrm, F0 = self._dual_parts(x, p)
        sx = self.s(x)
        Fs = F0 / sx
        return Fs[..., None] * (self._dF0_dx(x, p, nrm) / sx[..., None] - (F0 / sx**2)[..., None] * self._ds_dx(x))

    def _rhs(self, x, p):
        """(dx/dt, dp/dt) = (dH/dp, -dH/dx) sharing the field evaluations at x."""
        W, hinv, hp, nrm, F0 = self._dual_parts(x, p)
        sx = self.s(x)
        Fs = F0 / sx
        xdot = (Fs / sx)[..., None] * (W + hp / nrm[..., None])
        pdot = -Fs[..., None] * (self._dF0_dx(x, p, nrm) / sx[..., None] - (F0 / sx**2)[..., None] * self._ds_dx(x))
        return xdot, pdot

    def geodesic(self, x0, v0, T=1.0, n_steps=200):
        """Integrate the geodesic with initial velocity v0 for parameter time T (RK4 on (x, p)).

        Returns (ts, xs, vs) with xs[k], vs[k] the position and velocity at ts[k].
        F(x, v) is constant along the curve (= F(x0, v0)); Finsler length = T * F(x0, v0).
        """
        x = np.asarray(x0, float)
        p = self.legendre(x, np.asarray(v0, float))
        dt = T / n_steps
        xs, vs = [x], [self.legendre_inv(x, p)]

        rhs = self._rhs

        for _ in range(n_steps):
            k1x, k1p = rhs(x, p)
            k2x, k2p = rhs(x + 0.5 * dt * k1x, p + 0.5 * dt * k1p)
            k3x, k3p = rhs(x + 0.5 * dt * k2x, p + 0.5 * dt * k2p)
            k4x, k4p = rhs(x + dt * k3x, p + dt * k3p)
            x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            p = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
            xs.append(x)
            vs.append(self.legendre_inv(x, p))
        return np.linspace(0.0, T, n_steps + 1), np.stack(xs), np.stack(vs)

    def exp(self, x0, w, n_steps=200):
        """exp_{x0}(w): endpoint (and end velocity) of the geodesic with initial velocity w, time 1."""
        _, xs, vs = self.geodesic(x0, w, 1.0, n_steps)
        return xs[-1], vs[-1]

    def log(self, x0, x1, w0=None, n_steps=200):
        """Shooting solve of exp_{x0}(w) = x1 (boundary-value problem; test/reference utility)."""
        from scipy.optimize import least_squares

        x0 = np.asarray(x0, float)
        x1 = np.asarray(x1, float)
        if w0 is None:
            w0 = x1 - x0
        res = least_squares(lambda w: self.exp(x0, w, n_steps)[0] - x1, w0, xtol=1e-13, ftol=1e-13, gtol=1e-13)
        return res.x

    def dist(self, x0, x1, **kw):
        """Forward distance d(x0, x1) = F(x0, log_{x0}(x1))."""
        return float(self.F(x0, self.log(x0, x1, **kw)))

    def log_batched(self, x0, x1, n_steps=100, iters=30, tol=1e-9, fd_eps=1e-6, w_init=None, n_cont=1, verbose=False):
        """Batched shooting solve of exp_{x0}(w) = x1: log map log_{x0}(x1) for x0, x1 of shape (N, d).

        Gauss-Newton with finite-difference Jacobians, per-sample backtracking, optional warm start
        `w_init`, and optional continuation in the target (n_cont > 1 moves the target from x0 to x1
        along the straight line, warm-starting each stage).  Returns (w, residual_norm).
        """
        x0 = np.asarray(x0, float)
        x1 = np.asarray(x1, float)
        d = self.dim
        w = (x1 - x0).copy() if w_init is None else np.asarray(w_init, float).copy()
        stages = [1.0] if (n_cont <= 1 or w_init is not None) else list(np.linspace(1.0 / n_cont, 1.0, n_cont))
        err = None
        for si, s in enumerate(stages):
            y = x0 + s * (x1 - x0)
            if w_init is None and si > 0:
                w = w * (s / stages[si - 1])
            r = self.exp(x0, w, n_steps)[0] - y
            err = np.linalg.norm(r, axis=-1)
            for it in range(iters):
                active = err > tol
                if not active.any():
                    break
                J = np.zeros(x0.shape + (d,))
                for j in range(d):
                    e = np.zeros(d)
                    e[j] = fd_eps
                    J[..., :, j] = (self.exp(x0, w + e, n_steps)[0] - self.exp(x0, w - e, n_steps)[0]) / (2 * fd_eps)
                dw = -np.linalg.solve(J, r[..., None])[..., 0]
                step = np.ones(x0.shape[:-1])
                for _ in range(6):                       # backtracking where the residual grew
                    w_try = w + step[..., None] * dw
                    r_try = self.exp(x0, w_try, n_steps)[0] - y
                    err_try = np.linalg.norm(r_try, axis=-1)
                    worse = active & (err_try > err)
                    if not worse.any():
                        break
                    step = np.where(worse, 0.5 * step, step)
                accept = active & (err_try <= err)
                w = np.where(accept[..., None], w_try, w)
                r = np.where(accept[..., None], r_try, r)
                err = np.where(accept, err_try, err)
                if verbose:
                    print(f"  stage {s:.2f} iter {it}: max err {err.max():.2e}, unconverged {(err > tol).sum()}")
        return w, err

    # ---- envelope utilities (control) -------------------------------------------------------
    def project(self, x, u):
        """Radial projection onto the velocity envelope {F(x, .) <= 1}."""
        u = np.asarray(u, float)
        s = np.maximum(self.F(x, u), 1.0)
        return u / s[..., None]


# --------------------------------------------------------------------------------------
# Sampling covariances for MPPI (all normalised to unit determinant)
# --------------------------------------------------------------------------------------
def _unit_det(C):
    d = C.shape[-1]
    det = np.linalg.det(C)
    return C / det[..., None, None] ** (1.0 / d)


def cov_isotropic(field: RandersField, x, u):
    x = np.asarray(x, float)
    return np.broadcast_to(np.eye(field.dim), x.shape + (field.dim,)).copy()


def cov_riemannian(field: RandersField, x, u):
    """A(x)^{-1}: the symmetric (Riemannian) part of the Randers metric."""
    A, _, _ = field.randers(x)
    return _unit_det(np.linalg.inv(A))


def cov_finsler(field: RandersField, x, u, tol=1e-8):
    """g_u(x)^{-1}: inverse fundamental tensor in the nominal direction u; falls back to A^{-1} at u = 0."""
    u = np.asarray(u, float)
    small = np.linalg.norm(u, axis=-1) < tol
    if np.all(small):
        return cov_riemannian(field, x, u)
    u_safe = np.where(small[..., None], 1.0, u)
    C = _unit_det(np.linalg.inv(field.fundamental_tensor(x, u_safe)))
    if np.any(small):
        C = np.where(small[..., None, None], cov_riemannian(field, x, u), C)
    return C
