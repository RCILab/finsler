"""
Checks for finsler/general.py (SumRanders, torch).  Run:  python tests/run_tests_general.py
"""
import sys
import pathlib

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from finsler import RandersField, channel_wind, finsler_field, randers_to_zermelo, zermelo_to_randers
from finsler.general import SumRanders, constant_sum_randers, egg_indicatrix

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(1)
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


T = lambda a: torch.as_tensor(np.asarray(a, float))

# ----------------------------------------------------------------------------------------------
# 1. One constant term == Randers metric (compare against the numpy RandersField)
# ----------------------------------------------------------------------------------------------
A = np.array([[2.0, 0.3], [0.3, 1.0]])
b = np.array([0.4, -0.2])
h, W, _ = randers_to_zermelo(A, b)
rf = RandersField(lambda x: W, lambda x: h)
L1 = np.linalg.cholesky(A).T                      # L^T L = A
sr = constant_sum_randers(L1[None], b)
x = np.zeros((40, 2))
v = rng.normal(size=(40, 2))
p = rng.normal(size=(40, 2))
check("K=1: F matches RandersField", np.allclose(sr.F(T(x), T(v)).numpy(), rf.F(x, v)))
check("K=1: g_v matches RandersField", np.allclose(sr.fundamental_tensor(T(x), T(v)).numpy(), rf.fundamental_tensor(x, v)))
check("K=1: L(v) matches RandersField", np.allclose(sr.legendre(T(x), T(v)).numpy(), rf.legendre(x, v)))
check("K=1: L^{-1}(p) (Newton) matches closed form", np.allclose(sr.legendre_inv(T(x), T(p)).numpy(), rf.legendre_inv(x, p), atol=1e-9))
check("K=1: F*(p) matches closed-form dual", np.allclose(sr.F_dual(T(x), T(p)).numpy(), rf.F_dual(x, p), atol=1e-9))
xs, vs = sr.geodesic(T(x[:5]), T(v[:5]), 1.0, 20)
check("constant coefficients: geodesics are straight (v constant)", torch.allclose(vs[-1], vs[0], atol=1e-10))

# ----------------------------------------------------------------------------------------------
# 2. x-dependent Randers (channel wind) as SumRanders vs the Hamiltonian integrator of RandersField
# ----------------------------------------------------------------------------------------------
beta, width = 0.6, 0.8
Wnp = channel_wind(beta, width)
rf2 = finsler_field(Wnp)


def Ls_t(x):
    Wt = torch.stack([-beta * torch.exp(-(x[..., 1] ** 2) / width**2), torch.zeros_like(x[..., 0])], -1)
    lam = 1.0 - (Wt * Wt).sum(-1)
    eye = torch.eye(2, dtype=x.dtype).expand(x.shape[:-1] + (2, 2))
    A = (lam[..., None, None] * eye + Wt[..., :, None] * Wt[..., None, :]) / lam[..., None, None] ** 2
    L = torch.linalg.cholesky(A).transpose(-1, -2)
    return L[..., None, :, :]


def b_t(x):
    Wt = torch.stack([-beta * torch.exp(-(x[..., 1] ** 2) / width**2), torch.zeros_like(x[..., 0])], -1)
    lam = 1.0 - (Wt * Wt).sum(-1)
    return -Wt / lam[..., None]


sr2 = SumRanders(Ls_t, b_t)
x0 = np.array([[-1.0, 0.2], [0.5, -0.4], [-0.3, 0.9]])
w0 = np.array([[1.2, 0.4], [-0.8, 0.9], [1.5, -0.3]])
check("channel: F matches", np.allclose(sr2.F(T(x0), T(w0)).numpy(), rf2.F(x0, w0)))
xe_np, ve_np = rf2.exp(x0, w0, n_steps=200)
xe_t, ve_t = sr2.exp(T(x0), T(w0), n_steps=200)
check("channel: Euler-Lagrange (torch, autograd) == Hamiltonian (numpy) geodesic endpoint", np.allclose(xe_t.numpy(), xe_np, atol=1e-6), f"max diff {np.abs(xe_t.numpy()-xe_np).max():.1e}")
check("channel: end velocities agree", np.allclose(ve_t.numpy(), ve_np, atol=1e-6))

# ----------------------------------------------------------------------------------------------
# 3. Two-term 'egg': a genuine non-Randers Finsler metric
# ----------------------------------------------------------------------------------------------
pts, egg = egg_indicatrix(n=2000)
x = np.zeros((2000, 2))
check("egg: F(indicatrix) = 1", np.allclose(egg.F(T(x), T(pts)).numpy(), 1.0))
v = rng.normal(size=(60, 2))
x = np.zeros((60, 2))
check("egg: F > 0", bool((egg.F(T(x), T(v)) > 0).all()))
check("egg: 1-homogeneous", np.allclose(egg.F(T(x), T(3.1 * v)).numpy(), 3.1 * egg.F(T(x), T(v)).numpy()))
g = egg.fundamental_tensor(T(x), T(v))
check("egg: g_v positive definite", bool((torch.linalg.eigvalsh(g) > 0).all()))
check("egg: g_v(v,v) = F^2", np.allclose(torch.einsum("ni,nij,nj->n", T(v), g, T(v)).numpy(), egg.F(T(x), T(v)).numpy() ** 2))


def fd_hess(f, v, eps=1e-5):
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei = np.zeros(2); ei[i] = eps
            ej = np.zeros(2); ej[j] = eps
            H[i, j] = (f(v + ei + ej) - f(v + ei - ej) - f(v - ei + ej) + f(v - ei - ej)) / (4 * eps**2)
    return H


ok = True
for k in range(5):
    fE = lambda u: 0.5 * egg.F(T(np.zeros(2)), T(u)).item() ** 2
    ok &= np.allclose(g[k].numpy(), fd_hess(fE, v[k]), atol=1e-5)
check("egg: g_v = Hess(F^2/2) (finite differences)", ok)
p = rng.normal(size=(60, 2))
vv = egg.legendre_inv(T(x), T(p))
check("egg: L(L^{-1}(p)) = p (Newton)", np.allclose(egg.legendre(T(x), vv).numpy(), p, atol=1e-9))
check("egg: L^{-1}(L(v)) = v", np.allclose(egg.legendre_inv(T(x), egg.legendre(T(x), T(v))).numpy(), v, atol=1e-9))
Fd = egg.F_dual(T(x), T(p)).numpy()
sup = (pts @ p.T).max(0)
check("egg: F*(p) = sup over indicatrix of p.v", np.allclose(Fd, sup, rtol=1e-5))
# not a Randers metric: the indicatrix has no centre of symmetry
c = 0.5 * (pts.max(0) + pts.min(0))                 # bounding-box centre = symmetry centre if one exists
refl = 2 * c - pts
d_refl = np.min(np.linalg.norm(refl[:, None, :] - pts[None, :, :], axis=-1), axis=1).max()
r_pts, rand_egg = egg_indicatrix(n=2000, Ls=np.array([[[1.0, 0.0], [0.0, 1.6]]]), b=np.array([0.35, 0.0]))
c1 = 0.5 * (r_pts.max(0) + r_pts.min(0))
d_refl_r = np.min(np.linalg.norm((2 * c1 - r_pts)[:, None, :] - r_pts[None, :, :], axis=-1), axis=1).max()
check("egg: indicatrix is NOT centrally symmetric (Randers one is)", d_refl > 0.01 and d_refl_r < 5e-3, f"egg asym {d_refl:.3f}, Randers asym {d_refl_r:.1e}")

# x-dependent egg: energy conservation along geodesics
def Ls_egg(x):
    th = x[..., 0]
    R = torch.stack([torch.stack([torch.cos(th), -torch.sin(th)], -1), torch.stack([torch.sin(th), torch.cos(th)], -1)], -2)
    L1 = torch.tensor([[1.0, 0.0], [0.0, 1.6]], dtype=x.dtype).expand(x.shape[:-1] + (2, 2))
    L2 = 0.5 * R
    return torch.stack([L1, L2], -3)


def b_egg(x):
    u = torch.stack([torch.cos(x[..., 1]), torch.sin(x[..., 1])], -1)
    rho = 0.3
    return rho * torch.stack([u[..., 0], 1.6 * u[..., 1]], -1)          # rho L_1^T u


egg_x = SumRanders(Ls_egg, b_egg)
x0 = T(np.array([[0.2, -0.3], [-0.5, 0.4]]))
v0 = T(np.array([[0.7, 0.5], [-0.4, 0.9]]))
xs, vs = egg_x.geodesic(x0, v0, 1.5, 300)
Fs = torch.stack([egg_x.F(xs[i], vs[i]) for i in range(len(xs))])
check("x-dependent egg: F(x, xdot) conserved along geodesics", bool(((Fs - Fs[0]).abs() / Fs[0] < 1e-6).all()), f"rel spread {((Fs.max(0).values-Fs.min(0).values)/Fs[0]).max():.1e}")
check("x-dependent egg: geodesic is curved", not torch.allclose(vs[-1] / torch.linalg.norm(vs[-1], dim=-1, keepdim=True), vs[0] / torch.linalg.norm(vs[0], dim=-1, keepdim=True), atol=1e-3))
C = egg.cov_finsler(T(np.zeros((3, 2))), T(np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])))
check("egg: cov_finsler unit determinant, direction dependent", np.allclose(torch.linalg.det(C).numpy(), 1.0) and not torch.allclose(C[0], C[1]))

print()
if FAILS:
    print(f"{len(FAILS)} check(s) FAILED: {FAILS}")
    sys.exit(1)
print("all checks passed")
