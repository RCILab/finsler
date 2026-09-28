"""
Envelope identification for the simulated H1-2: sweep body-velocity commands, record achieved velocities.

Protocol per command: reset standing, 1 s zero command, then the command for 5 s; achieved velocity =
mean body-frame (vx, vy, yaw rate) over the last 3 s; a rollout counts as failed if the robot falls.
Commands: directions on the unit sphere (in a scaled space vx/1.5, vy/1.0, w/2.0) x magnitudes, so that
every direction is pushed until the policy saturates or falls.  Output: sim/results/h1_2_envelope.npz and
a summary of the achieved set (max speeds per axis, saturation onset, fall onset).

Usage: python sim/identify_envelope.py [--n-dirs 120] [--mags 0.3,0.6,0.9,1.2,1.5,1.9]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from h1_2_env import H12Env

OUT = pathlib.Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
SCALE = np.array([1.6, 1.0, 2.0])         # command-space scaling (vx, vy, yaw rate) so that unit directions are comparable


def fibonacci_sphere(n, rng):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5**0.5) * i
    pts = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    return pts @ R.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dirs", type=int, default=120)
    ap.add_argument("--mags", default="0.3,0.6,0.9,1.2,1.5,1.9")
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--hold", type=float, default=5.0)
    ap.add_argument("--limits", default="", help="emulated interface limits 'vxmin,vxmax,vymin,vymax,wmin,wmax'")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    limits = None
    if args.limits:
        v = [float(x) for x in args.limits.split(",")]
        limits = [(v[0], v[1]), (v[2], v[3]), (v[4], v[5])]
    mags = [float(m) for m in args.mags.split(",")]
    rng = np.random.default_rng(0)
    dirs = fibonacci_sphere(args.n_dirs, rng)
    env = H12Env(cmd_limits=limits)
    rows = []
    t0 = time.time()
    for di, d in enumerate(dirs):
        for m in mags:
            cmd = m * d * SCALE
            env.reset()
            out = env.run(lambda t, p, v, c=cmd: [0.0, 0.0, 0.0] if t < args.settle else c, args.settle + args.hold, record_every=2)
            sel = out["t"] > args.settle + args.hold - 3.0
            ach = out["vel"][sel].mean(0) if sel.any() and not out["fallen"] else np.full(3, np.nan)
            rows.append(dict(dir=di, mag=m, cmd=cmd, achieved=ach, fallen=out["fallen"], t_fall=out["t"][-1] if out["fallen"] else np.nan))
        if (di + 1) % 20 == 0:
            print(f"  {di+1}/{len(dirs)} directions ({time.time()-t0:.0f}s)", flush=True)
    cmd = np.array([r["cmd"] for r in rows])
    ach = np.array([r["achieved"] for r in rows])
    fallen = np.array([r["fallen"] for r in rows])
    mag = np.array([r["mag"] for r in rows])
    dir_id = np.array([r["dir"] for r in rows])
    np.savez(OUT / f"h1_2_envelope{args.tag}.npz", cmd=cmd, achieved=ach, fallen=fallen, mag=mag, dir_id=dir_id, dirs=dirs, scale=SCALE, limits=np.array(limits if limits else np.zeros((3, 2))))
    ok = ~fallen
    lines = ["# H1-2 (MuJoCo, Unitree pretrained policy) velocity envelope identification",
             f"{len(rows)} rollouts: {len(dirs)} directions x {len(mags)} magnitudes; hold {args.hold} s, achieved = mean of last 3 s; falls {fallen.sum()}", "",
             "| axis | max achieved (no fall) | command at which it saturates (achieved/cmd < 0.85) |", "|---|---:|---|"]
    names = ["+vx (forward)", "-vx (backward)", "+vy (left)", "-vy (right)", "+yaw", "-yaw"]
    axes = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
    for name, (ax, sg) in zip(names, axes):
        # rollouts whose command is mostly along this axis direction
        u = cmd / np.maximum(np.linalg.norm(cmd / SCALE, axis=1, keepdims=True), 1e-9) / SCALE
        along = (sg * u[:, ax] > 0.9) & ok
        if along.any():
            best = np.max(sg * ach[along, ax])
            ratio = (sg * ach[along, ax]) / np.maximum(sg * cmd[along, ax], 1e-9)
            sat = cmd[along][ratio < 0.85]
            lines.append(f"| {name} | {best:.2f} | {('%.2f' % np.min(np.abs(sat[:, ax]))) if len(sat) else 'none up to sweep max'} |")
        else:
            lines.append(f"| {name} | n/a | n/a |")
    lines += ["", f"fall rate by magnitude: " + ", ".join(f"{m}: {fallen[mag == m].mean():.2f}" for m in mags)]
    (OUT / f"h1_2_envelope{args.tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
