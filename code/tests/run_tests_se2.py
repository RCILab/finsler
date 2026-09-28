"""
Checks for finsler/se2.py (left-invariant Finsler metrics on SE(2)).  Run: python tests/run_tests_se2.py
"""
import sys, pathlib, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from finsler.se2 import se2_metric, SE2Finsler, quadruped_envelope, body_velocity, world_velocity

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)
FAILS = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok: FAILS.append(name)
T = lambda a: torch.as_tensor(np.asarray(a, float))

Ls, b, speeds = quadruped_envelope()
print("envelope max speeds:", {k: round(v, 3) for k, v in speeds.items()})
ref = se2_metric(Ls, b)          # autograd reference
fld = SE2Finsler(Ls, b)          # Euler-Poincare
check("envelope: forward 1.2, backward 0.5, side 0.4, yaw ~1.2",
      abs(speeds["forward"] - 1.2) < 0.03 and abs(speeds["backward"] - 0.5) < 0.012 and abs(speeds["left"] - 0.4) < 0.01 and abs(speeds["yaw+"] - 1.2) < 0.03)
check("envelope: asymmetric fore/aft, symmetric left/right and yaw",
      speeds["forward"] > 2 * speeds["backward"] and np.isclose(speeds["left"], speeds["right"]) and np.isclose(speeds["yaw+"], speeds["yaw-"]))

q = T(rng.uniform(-3, 3, (50, 3))); qd = T(rng.normal(size=(50, 3)))
check("SE2Finsler.F == coordinates SumRanders F", torch.allclose(fld.F(q, qd), ref.F(q, qd)))
check("SE2Finsler.g == coordinates g", torch.allclose(fld.fundamental_tensor(q, qd), ref.fundamental_tensor(q, qd)))
xi = body_velocity(q, qd)
check("F(q, qdot) = F_body(xi)", torch.allclose(fld.F(q, qd), fld.F_body(xi)))
phi = 0.7; c, s_ = np.cos(phi), np.sin(phi)
Rg = T([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
q2 = torch.stack([c * q[:, 0] - s_ * q[:, 1] + 0.3, s_ * q[:, 0] + c * q[:, 1] - 1.1, q[:, 2] + phi], -1)
qd2 = torch.einsum("ij,nj->ni", Rg, qd)
check("left-invariance under a group translation", torch.allclose(fld.F(q2, qd2), fld.F(q, qd)))
check("non-reversible: F(q,-qdot) != F(q,qdot)", not torch.allclose(fld.F(q, -qd), fld.F(q, qd)))
check("g positive definite (3x3)", bool((torch.linalg.eigvalsh(fld.fundamental_tensor(q, qd)) > 0).all()))
check("body Legendre inverse round trip", torch.allclose(fld.legendre_inv_body(fld.legendre_body(xi)), xi, atol=1e-8))

# geodesics: Euler-Poincare vs autograd Euler-Lagrange
q0 = T([[0.0, 0.0, 0.0], [1.0, -0.5, 0.8], [-0.3, 0.2, -2.0]]); w0 = T([[1.0, 0.3, 0.9], [-0.4, 0.6, -1.2], [0.5, -0.7, 0.4]])
t0 = time.time(); qs_ref, ws_ref = ref.geodesic(q0, w0, 1.0, 200); t_ref = time.time() - t0
t0 = time.time(); qs, ws = fld.geodesic(q0, w0, 1.0, 200); t_ep = time.time() - t0
check("Euler-Poincare geodesic == autograd geodesic (positions)", torch.allclose(qs, qs_ref, atol=1e-7), f"max diff {(qs - qs_ref).abs().max():.1e}; EP {t_ep:.2f}s vs autograd {t_ref:.2f}s")
check("... and velocities", torch.allclose(ws, ws_ref, atol=1e-7))
Fs = torch.stack([fld.F(qs[i], ws[i]) for i in range(len(qs))])
check("F(q, qdot) conserved along geodesics", bool((((Fs - Fs[0]).abs() / Fs[0]) < 1e-8).all()), f"rel spread {((Fs.max(0).values - Fs.min(0).values) / Fs[0]).max():.1e}")
q1 = qs[-1]
t0 = time.time(); wb, err = fld.log_batched(q0, q1, n_steps=200, iters=8); t_log = time.time() - t0
check("log(exp(w)) = w (3-D shooting, Euler-Poincare)", torch.allclose(wb, w0, atol=1e-6), f"err {err.max():.1e}, {t_log:.1f}s")
q0r = T([[0.0, 0.0, phi]]); w0r = torch.einsum("ij,nj->ni", Rg, w0[:1])
qsr, _ = fld.geodesic(q0r, w0r, 1.0, 200)
qs_rot = torch.einsum("ij,tnj->tni", Rg, qs[:, :1]); qs_rot[..., 2] = qs[:, :1, 2] + phi
check("geodesics equivariant under left translation", torch.allclose(qsr, qs_rot, atol=1e-8))
# batched speed check
qb = T(rng.uniform(-1, 1, (2000, 3))); wb0 = T(rng.normal(size=(2000, 3)))
t0 = time.time(); fld.exp(qb, wb0, n_steps=100); print(f"   batched exp: 2000 points x 100 steps in {time.time()-t0:.2f}s")

# physics with the fast numpy SE2Randers (pure Randers envelope, same speeds): goal 1.5 m directly behind.
from finsler.se2_randers import SE2Randers
fr = SE2Randers.from_speeds()
qs0 = np.zeros((2, 3)); qg = np.array([[-1.5, 0, 0], [-1.5, 0, np.pi]])
best = np.full(2, np.inf)
for guess in [[[-1.5, 0, 0], [-1.5, 0, np.pi]], [[-1.5, 0, 2.5], [-1.0, 0.8, 3.0]], [[-1.5, 0, -2.5], [-1.0, -0.8, -3.0]], [[-1.0, 0.8, 1.5], [-1.2, 0.5, 2.0]]]:
    w, e = fr.log_batched(qs0, qg, n_steps=150, w_init=np.array(guess, float))
    best = np.minimum(best, np.where(e < 1e-6, fr.F(qs0, w), np.inf))
print(f"   time-optimal (numpy Randers): 1.5 m behind, same heading {best[0]:.2f} s (pure reversing 3.00 s); reversed heading {best[1]:.2f} s")
check("goal behind, same heading: optimum == reversing time (3.00 s)", abs(best[0] - 3.0) < 1e-3)
check("goal behind, reversed heading: optimum < reverse-then-turn (5.6 s)", best[1] < 5.6)
# numpy Randers vs torch SumRanders(K=1) agree
L1 = np.linalg.cholesky(fr.A).T
ref1 = SE2Finsler(L1[None], fr.bvec)
qq = T(rng.uniform(-1, 1, (4, 3))); ww = T(rng.normal(size=(4, 3)))
qs_np = fr.geodesic(qq.numpy(), ww.numpy(), 1.0, 100)[1]
qs_t, _ = ref1.geodesic(qq, ww, 1.0, 100)
check("numpy SE2Randers geodesic == torch SE2Finsler (K=1) geodesic", np.allclose(qs_np, qs_t.numpy(), atol=1e-8))

print()
if FAILS:
    print(f"{len(FAILS)} check(s) FAILED: {FAILS}"); sys.exit(1)
print("all checks passed")
