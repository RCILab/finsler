"""UR5e video: what a reversible metric gets wrong about a manipulator.

For an arm the asymmetry barely changes the route -- the symmetrised and the true metric pick paths
within 1% of each other -- but it changes the clock a great deal.  This video plays the symmetrised
metric's own plan twice: once at the speed that metric believes it can hold, and once at the speed the
arm actually has.  The second takes 45% longer.

The planner works in the shoulder-elbow plane.  Rather than guess the model's joint conventions, the
shoulder-lift and elbow angles are solved numerically against MuJoCo itself so that the attachment site
lands on each planned waypoint; the pan and wrist joints are held, so the motion stays in the plane.

Playback is in real time: the frame times are the executed times under the true velocity envelope.

Run: python web/make/make_ur5_video.py   (from finsler/)
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parents[1] / "code"
OUT = HERE.parent / "assets"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "experiments"))

import mujoco  # noqa: E402

from arm_envelope import Arm, SymmetrisedField  # noqa: E402

SCENE = CODE / "sim" / "third_party" / "menagerie" / "universal_robots_ur5e" / "scene.xml"
BLUE, ORANGE, INK2 = (42, 120, 214), (235, 104, 52), (82, 81, 78)


def font(sz):
    for n in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


class UR5:
    def __init__(self, width, height, pedestal=0.45):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, width)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, height)
        # a UR5 is bolted to a table, not to the floor; without the pedestal the elbow sweeps below
        # ground level on the low part of the reach
        base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.pedestal = pedestal
        self.m.body_pos[base] = self.m.body_pos[base] + np.array([0.0, 0.0, pedestal])
        # the stock scene gives the floor reflectance 0.2, and the mirror image of the arm below the
        # floor plane reads as the arm passing through it; the motion itself never comes within 0.4 m
        self.m.mat_reflectance[:] = 0.0
        self.d = mujoco.MjData(self.m)
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.elbow = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "forearm_link")
        self.wrist = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "wrist_1_link")
        self.rend = mujoco.Renderer(self.m, height, width)
        self.q0 = np.array([0.0, -1.2, 1.6, -1.95, -1.57, 0.0])
        self.d.qpos[:] = self.q0
        mujoco.mj_forward(self.m, self.d)
        sh = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "shoulder_link")
        self.z0 = float(self.d.xpos[sh][2])
        self.y0 = float(self.site_pos()[1])
        self.prev = None      # previous joint solution, for branch continuity

    def site_pos(self):
        return self.d.site_xpos[self.site].copy()

    def set_planar(self, xy, guess, elbow_y=None):
        """Solve shoulder-lift and elbow so the site sits at planar (x, y) -> world (x, ., y + z0).

        The two-link problem has two solutions.  Position is solved exactly from several starts; the
        first frame picks the branch whose elbow sits where the planner put it, and every later frame
        picks the solution nearest the previous one in joint space, so the arm never swaps branches
        mid-motion.  Selecting by elbow height throughout does not work: the UR5e's forearm frame sits
        on the joint axis, not at the idealised two-link elbow, so the two criteria disagree wherever
        the branches come close."""
        target = np.array([xy[0], xy[1] + self.z0])
        elbow_target = None if elbow_y is None else elbow_y + self.z0

        def resid(a):
            self.d.qpos[:] = self.q0
            self.d.qpos[1], self.d.qpos[2] = a
            mujoco.mj_forward(self.m, self.d)
            p = self.site_pos()
            return np.array([p[0] - target[0], p[2] - target[1]])

        sols = []
        for g in (guess, np.array([-2.0, 2.3]), np.array([-0.6, 1.0]), np.array([-1.2, -1.6])):
            r = least_squares(resid, g, max_nfev=80)
            err = float(np.linalg.norm(resid(r.x)))
            sols.append((r.x, err, float(self.d.xpos[self.elbow][2])))
        ok = [s for s in sols if s[1] < 1e-4] or sols
        if self.prev is not None:
            best = min(ok, key=lambda s: float(np.linalg.norm(s[0] - self.prev)))[:2]
        elif elbow_target is None:
            best = min(ok, key=lambda s: s[1])[:2]
        else:
            best = min(ok, key=lambda s: abs(s[2] - elbow_target))[:2]
        resid(best[0])
        self.prev = np.asarray(best[0], float).copy()
        return best


def box(scene, pos, half, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1], mujoco.mjtGeom.mjGEOM_BOX,
                        np.asarray(half, float), np.asarray(pos, float), np.eye(3).ravel(),
                        np.asarray(rgba, np.float32))


def marker(scene, pos, rgba, size=0.035):
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size] * 3), np.asarray(pos, float), np.eye(3).ravel(),
                        np.asarray(rgba, np.float32))


def label(img, title, rgb, t, T, done):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    dr.rectangle([0, 0, im.width, 78], fill=(252, 252, 251, 228))
    dr.text((16, 8), title, font=font(23), fill=rgb)
    dr.text((16, 42), f"{min(t, T):4.2f} s" + ("   arrived" if done else ""), font=font(26),
            fill=rgb if done else INK2)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    arm = Arm(workpiece=[0.30, 0.25], e_max=0.35, zone=2.0)
    d = np.load(CODE / "results/arm_envelope/arm_paths_safe2.npz")
    key = "reach_the_workpiece_from_upper_left"
    ur = UR5(args.width, args.height)

    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90.0, -6.0, 2.05
    cam.lookat[:] = [0.10, ur.y0, ur.z0 + 0.22]

    # Both panels play the SAME plan, the one the symmetrised metric produced.  The left panel runs it
    # on the clock that metric believes; the right panel runs it on the clock the arm actually has.
    # The route is the same to within 1%; the clocks differ by 45%, and that is the point.
    sym_field = SymmetrisedField(arm)
    runs = {}
    for name, rgb, lab, clock in [("predicted", ORANGE, "what the symmetrised metric predicts", sym_field),
                                  ("actual", BLUE, "what the arm can actually do", arm)]:
        pts = d[f"{key}__symmetric"]
        seg = pts[1:] - pts[:-1]
        mid = pts[:-1] + 0.5 * seg
        ts = np.concatenate([[0.0], np.cumsum(clock.F(mid, seg))])
        T = ts[-1]
        stamps = np.arange(0.0, T + 0.9, 1.0 / args.fps)
        xy = np.stack([np.interp(np.clip(stamps, 0, T), ts, pts[:, k]) for k in (0, 1)], 1)
        ur.prev = None
        frames, guess, worst = [], np.array([-1.2, 1.6]), 0.0
        trail = []
        for i, p in enumerate(xy):
            guess, err = ur.set_planar(p, guess, float(arm.elbow_y(p[None])[0]))
            worst = max(worst, err)
            trail.append(ur.site_pos())
            ur.rend.update_scene(ur.d, cam)
            box(ur.rend.scene, [0.0, ur.y0, ur.pedestal / 2], [0.10, 0.10, ur.pedestal / 2],
                (0.38, 0.38, 0.40, 1.0))
            wp = np.array([arm.workpiece[0], ur.y0, arm.workpiece[1] + ur.z0])
            marker(ur.rend.scene, wp, (0.05, 0.05, 0.05, 0.95), 0.030)
            for q in trail[::3]:
                marker(ur.rend.scene, q, tuple(c / 255 for c in rgb) + (0.85,), 0.011)
            frames.append(ur.rend.render())
        runs[name] = dict(frames=frames, T=T, rgb=rgb, lab=lab)
        print(f"  {name:10s}: {T:.2f} s, {len(frames)} frames, worst IK residual {worst*1e3:.1f} mm")

    n = max(len(r["frames"]) for r in runs.values())
    out = []
    for k in range(n):
        row = []
        for name in ("predicted", "actual"):
            r = runs[name]
            j = min(k, len(r["frames"]) - 1)
            row.append(label(r["frames"][j], r["lab"], r["rgb"], k / args.fps, r["T"],
                             k / args.fps >= r["T"]))
        out.append(np.concatenate(row, axis=1))
    path = OUT / "v5_ur5_effective_mass.mp4"
    imageio.mimwrite(path, out, fps=args.fps, codec="libx264", quality=8, macro_block_size=1,
                     ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"wrote {path}  ({len(out)} frames, {len(out)/args.fps:.1f} s)")


if __name__ == "__main__":
    main()
