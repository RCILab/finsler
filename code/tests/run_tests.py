"""
Numerical checks for finsler/randers.py.  Run:  python tests/run_tests.py
No pytest dependency; each check prints PASS/FAIL and the script exits non-zero on any failure.
"""
import sys
import pathlib

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from finsler import RandersField, zermelo_to_randers, randers_to_zermelo

rng = np.random.default_rng(0)
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def cross2(a, b):
    """z-component of the 2-D cross product (np.cross no longer accepts 2-vectors)."""
    a, b = np.asarray(a), np.asarray(b)
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


# ----------------------------------------------------------------------------------------------
# Test fields
# ----------------------------------------------------------------------------------------------
def const_field(beta, direction=(1.0, 0.0), h=None):
    """Constant Randers with A = h (default I), b = beta * direction  (built via Zermelo data)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    A = np.eye(2) if h is None else np.asarray(h, float)
    b = beta * d
    hZ, WZ, _ = randers_to_zermelo(A, b)
    return RandersField(lambda x: np.broadcast_to(WZ, np.shape(x)), lambda x: np.broadcast_to(hZ, np.shape(x) + (2,)))


def shear_field(beta=0.6, width=0.7):
    """Head-wind channel W(x) = (-beta * exp(-x2^2/width^2), 0): curl != 0, geodesics bend."""
    return RandersField(lambda x: np.stack([-beta * np.exp(-(x[..., 1] ** 2) / width**2), np.zeros_like(x[..., 0])], -1))


# ----------------------------------------------------------------------------------------------
# 1. Zermelo <-> Randers round trip and indicatrix
# ----------------------------------------------------------------------------------------------
h = np.array([[2.0, 0.3], [0.3, 1.0]])
W = np.array([0.4, -0.2])
A, b, lam = zermelo_to_randers(h, W)
h2, W2, lam2 = randers_to_zermelo(A, b)
check("zermelo->randers->zermelo round trip", np.allclose(h, h2) and np.allclose(W, W2) and np.isclose(lam, lam2))
check("|b|_A < 1 iff |W|_h < 1", (b @ np.linalg.solve(A, b) < 1) == (W @ h @ W < 1))

fld = RandersField(lambda x: W, lambda x: h)
x0 = np.zeros(2)
thetas = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
e = np.stack([np.cos(thetas), np.sin(thetas)], -1)
e = e / np.sqrt(np.einsum("ni,ij,nj->n", e, h, e))[:, None]          # h-unit vectors
indicatrix = W + e
check("indicatrix {F=1} = W + h-unit sphere", np.allclose(fld.F(np.zeros((len(e), 2)), indicatrix), 1.0, atol=1e-12))
v = rng.normal(size=(50, 2))
check("F (Randers form) == F (Zermelo quadratic form)", np.allclose(fld.F(np.zeros((50, 2)), v), fld.F_zermelo(np.zeros((50, 2)), v)))
check("F positively 1-homogeneous", np.allclose(fld.F(x0, 2.7 * v), 2.7 * fld.F(x0, v)))
check("F is NOT reversible (F(-v) != F(v))", not np.allclose(fld.F(x0, -v), fld.F(x0, v)))
check("reverse(): F_bar(v) = F(-v)", np.allclose(fld.reverse().F(x0, v), fld.F(x0, -v)))

# ----------------------------------------------------------------------------------------------
# 2. Fundamental tensor
# ----------------------------------------------------------------------------------------------
def fd_hessian(f, v, eps=1e-5):
    d = v.size
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            ei = np.zeros(d); ei[i] = eps
            ej = np.zeros(d); ej[j] = eps
            H[i, j] = (f(v + ei + ej) - f(v + ei - ej) - f(v - ei + ej) + f(v - ei - ej)) / (4 * eps**2)
    return H


ok_hess = ok_euler = ok_L = ok_pd = ok_hom = True
for k in range(10):
    vk = rng.normal(size=2)
    g = fld.fundamental_tensor(x0, vk)
    ok_hess &= np.allclose(g, fd_hessian(lambda u: 0.5 * fld.F(x0, u) ** 2, vk), atol=1e-5)
    ok_euler &= np.isclose(g @ vk @ vk, fld.F(x0, vk) ** 2)
    ok_L &= np.allclose(g @ vk, fld.legendre(x0, vk))
    ok_pd &= np.all(np.linalg.eigvalsh(g) > 0)
    ok_hom &= np.allclose(fld.fundamental_tensor(x0, 3.0 * vk), g)
check("g_v = Hess_v(F^2/2)  (finite differences)", ok_hess)
check("g_v(v,v) = F(v)^2  (Euler)", ok_euler)
check("g_v v = L(v)", ok_L)
check("g_v positive definite (strong convexity)", ok_pd)
check("g_v is 0-homogeneous", ok_hom)
check("g_v depends on v (Finsler, not Riemannian)", not np.allclose(fld.fundamental_tensor(x0, np.array([1.0, 0])), fld.fundamental_tensor(x0, np.array([0, 1.0]))))

# ----------------------------------------------------------------------------------------------
# 3. Legendre transform, dual norm, gradient asymmetry
# ----------------------------------------------------------------------------------------------
ok_inv1 = ok_inv2 = ok_dual = ok_supp = True
for k in range(20):
    vk = rng.normal(size=2)
    pk = rng.normal(size=2)
    ok_inv1 &= np.allclose(fld.legendre_inv(x0, fld.legendre(x0, vk)), vk)
    ok_inv2 &= np.allclose(fld.legendre(x0, fld.legendre_inv(x0, pk)), pk)
    ok_dual &= np.isclose(fld.F(x0, fld.legendre_inv(x0, pk)), fld.F_dual(x0, pk))
    ok_supp &= np.isclose(fld.F_dual(x0, pk), np.max(indicatrix @ pk), rtol=1e-6)   # sup over a dense indicatrix
check("L^{-1}(L(v)) = v", ok_inv1)
check("L(L^{-1}(p)) = p", ok_inv2)
check("F(L^{-1}(p)) = F*(p)", ok_dual)
check("F*(p) = sup_{F(v)=1} p.v  (dense brute force)", ok_supp)

df = np.array([1.0, 0.3])
g_plus, g_minus = fld.grad(x0, df), fld.grad(x0, -df)
check("grad(-f) != -grad f  (non-reversible)", not np.allclose(g_minus, -g_plus),
      f"grad f={g_plus.round(3)}, grad(-f)={g_minus.round(3)}")
f0 = const_field(0.0)
check("grad(-f) == -grad f when b = 0 (Riemannian)", np.allclose(f0.grad(x0, -df), -f0.grad(x0, df)))

# 1-D sanity of the closed forms, beta = 0.6:  forward max speed 1/(1+beta), backward 1/(1-beta)
beta = 0.6
f1 = const_field(beta, (1.0, 0.0))
check("1-D speeds: F(e1) = 1+beta, F(-e1) = 1-beta",
      np.isclose(f1.F(x0, np.array([1.0, 0])), 1 + beta) and np.isclose(f1.F(x0, np.array([-1.0, 0])), 1 - beta))
check("1-D Legendre: L^{-1}(xi e1) = xi/(1+beta)^2 e1,  L^{-1}(-xi e1) = -xi/(1-beta)^2 e1",
      np.allclose(f1.legendre_inv(x0, np.array([2.0, 0])), [2 / (1 + beta) ** 2, 0])
      and np.allclose(f1.legendre_inv(x0, np.array([-2.0, 0])), [-2 / (1 - beta) ** 2, 0]))

# ----------------------------------------------------------------------------------------------
# 4. Geodesics in a constant metric: straight lines, d(x0,x1) = F(x1-x0), and the target field
# ----------------------------------------------------------------------------------------------
x0 = np.array([-1.0, 0.5])
x1 = np.array([1.5, -0.3])
rho = lambda x: f1.F(x, x1 - x)                     # forward distance to x1 in a Minkowski space
drho = -f1.ell(x0, x1 - x0)                          # d rho at x0  (d/dx F(x1 - x) = -dF/dv)
v_fwd = f1.legendre_inv(x0, -drho)                   # L^{-1}(-d rho): claimed unit initial velocity toward x1
check("F(L^{-1}(-d rho)) = 1 and d rho(v) = -1", np.isclose(f1.F(x0, v_fwd), 1.0) and np.isclose(drho @ v_fwd, -1.0))
T = rho(x0)
_, xs, vs = f1.geodesic(x0, v_fwd, T=T, n_steps=100)
check("shooting along L^{-1}(-d rho) for time d(x0,x1) arrives at x1", np.allclose(xs[-1], x1, atol=1e-8),
      f"end={xs[-1].round(6)}")
check("constant metric: geodesic is a straight line", np.allclose(cross2(xs - x0, x1 - x0), 0, atol=1e-8))
v_wrong = -f1.grad(x0, drho)                          # -grad rho: the tempting but wrong direction
_, xs_w, _ = f1.geodesic(x0, v_wrong / f1.F(x0, v_wrong), T=T, n_steps=100)
check("shooting along -grad rho MISSES x1", np.linalg.norm(xs_w[-1] - x1) > 0.1,
      f"miss distance={np.linalg.norm(xs_w[-1] - x1):.3f}")
check("d(x0,x1) != d(x1,x0)", not np.isclose(rho(x0), f1.F(x1, x0 - x1)), f"{rho(x0):.3f} vs {f1.F(x1, x0 - x1):.3f}")

# ----------------------------------------------------------------------------------------------
# 5. Variable wind: energy conservation, exp/log consistency, reverse-metric shooting trick
# ----------------------------------------------------------------------------------------------
fs = shear_field(0.6)
x0 = np.array([-1.0, 0.2])
w = np.array([1.2, 0.4])
ts, xs, vs = fs.geodesic(x0, w, T=1.0, n_steps=400)
Fs = fs.F(xs, vs)
check("F(x, xdot) constant along geodesic (H conserved)", np.allclose(Fs, Fs[0], rtol=1e-7), f"spread={Fs.max()-Fs.min():.2e}")
check("shear wind (curl != 0): geodesic is curved", np.max(np.abs(cross2(xs - x0, w))) > 1e-3)

x1, _ = fs.exp(x0, w, n_steps=400)
w_back = fs.log(x0, x1, n_steps=400)
check("log(x0, exp(x0, w)) = w  (shooting BVP)", np.allclose(w_back, w, atol=1e-6), f"err={np.linalg.norm(w_back-w):.2e}")

# Reverse-metric trick: sample x1, w; x0 := exp^{F_bar}_{x1}(w). Then the F-geodesic from x0 with
# initial velocity -(end velocity of the F_bar geodesic) reaches x1 in time 1 with F-length F_bar(x1, w).
fbar = fs.reverse()
x1 = np.array([0.8, -0.1])
w = np.array([-1.5, 0.9])
_, xs_bar, vs_bar = fbar.geodesic(x1, w, T=1.0, n_steps=400)
x0_new, v_end = xs_bar[-1], vs_bar[-1]
_, xs_f, vs_f = fs.geodesic(x0_new, -v_end, T=1.0, n_steps=400)
check("reverse-metric shooting: F-geodesic from x0 with -v_end returns to x1", np.allclose(xs_f[-1], x1, atol=1e-7),
      f"err={np.linalg.norm(xs_f[-1]-x1):.2e}")
check("... and retraces the same curve (reverse of an F_bar-geodesic is an F-geodesic)", np.allclose(xs_f, xs_bar[::-1], atol=1e-7))
check("... with F-length = F_bar(x1, w) = F(x1, -w)", np.isclose(fs.F(x0_new, -v_end), fbar.F(x1, w)))
t_idx = 250
u_t = -vs_bar[400 - t_idx]                 # conditional path x_t = exp^{F_bar}_{x1}((1-t) w): u_t = -(F_bar velocity)
check("u_t = -(F_bar-velocity) equals the F-geodesic velocity at the same point", np.allclose(u_t, vs_f[t_idx], atol=1e-7))

# ----------------------------------------------------------------------------------------------
# 6. Closed b => same geodesics as the Riemannian part (slope model changes time, not the path)
# ----------------------------------------------------------------------------------------------
fg = const_field(0.5, (1.0, 0.0))          # F = |v| + d(phi)(v), phi = 0.5 x_1
x0 = np.array([0.0, 0.0])
w = np.array([0.3, 1.0])
_, xs_g, _ = fg.geodesic(x0, w, T=1.0, n_steps=100)
check("closed b: geodesic of F is a straight line (same as the Riemannian part)", np.allclose(cross2(xs_g - x0, w), 0, atol=1e-9))

# ----------------------------------------------------------------------------------------------
print()
if FAILS:
    print(f"{len(FAILS)} check(s) FAILED: {FAILS}")
    sys.exit(1)
print("all checks passed")
