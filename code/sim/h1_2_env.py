"""
Headless MuJoCo environment for the Unitree H1-2 humanoid driven by Unitree's pretrained velocity-command
locomotion policy (unitree_rl_gym, deploy/pre_train/h1_2/motion.pt).

Conventions (from deploy/deploy_mujoco/configs/h1_2.yaml): sim dt 0.002 s, policy at 50 Hz (decimation 10),
PD torque control with per-joint kp/kd, action -> target joint position = 0.25 * action + default angles,
observation (47) = [omega*0.25, gravity orientation, cmd*[2,2,0.25], (q - q_default), dq*0.05, last action,
sin/cos gait phase (period 0.8 s)].  Command = body-frame (vx [m/s], vy [m/s], yaw rate [rad/s]).

The environment exposes what the planner needs: the base pose (x, y, yaw), the achieved body-frame planar
velocity (vx, vy, yaw rate), and a fall flag.  It is used for (i) envelope identification (command sweep,
measure achieved velocities), (ii) closed-loop execution of planned velocity commands.
"""
from __future__ import annotations

import pathlib

import mujoco
import numpy as np
import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parent
GYM = ROOT / "third_party" / "unitree_rl_gym"


def quat_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def gravity_orientation(q):
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)])


def quat_rotate_inverse(q, v):
    """Rotate world vector v into the frame of quaternion q (w, x, y, z)."""
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R.T @ v


class H12Env:
    def __init__(self, config="h1_2.yaml", seed=0, cmd_limits=None):
        """cmd_limits: optional ((vx_min, vx_max), (vy_min, vy_max), (w_min, w_max)) applied to every command,
        emulating the (asymmetric) velocity limits of a deployed locomotion interface."""
        self.cmd_limits = None if cmd_limits is None else np.asarray(cmd_limits, float)
        cfg = yaml.safe_load(open(GYM / "deploy" / "deploy_mujoco" / "configs" / config))
        sub = lambda p: p.replace("{LEGGED_GYM_ROOT_DIR}", str(GYM))
        self.model = mujoco.MjModel.from_xml_path(sub(cfg["xml_path"]))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = cfg["simulation_dt"]
        self.dt = cfg["simulation_dt"]
        self.decim = cfg["control_decimation"]
        self.ctrl_dt = self.dt * self.decim
        self.kps = np.array(cfg["kps"], float)
        self.kds = np.array(cfg["kds"], float)
        self.default = np.array(cfg["default_angles"], float)
        self.scales = dict(ang=cfg["ang_vel_scale"], pos=cfg["dof_pos_scale"], vel=cfg["dof_vel_scale"], act=cfg["action_scale"], cmd=np.array(cfg["cmd_scale"], float))
        self.na, self.nobs = cfg["num_actions"], cfg["num_obs"]
        self.policy = torch.jit.load(sub(cfg["policy_path"]))
        self.policy.eval()
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ---- state -------------------------------------------------------------------------------------------
    def reset(self, pose=(0.0, 0.0, 0.0)):
        mujoco.mj_resetData(self.model, self.data)
        x, y, yaw = pose
        self.data.qpos[0:2] = [x, y]
        self.data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.data.qpos[7:] = self.default
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.action = np.zeros(self.na)
        self.target = self.default.copy()
        self.counter = 0
        self.fallen = False

    def pose(self):
        q = self.data.qpos
        return np.array([q[0], q[1], quat_to_yaw(q[3:7])])

    def body_velocity(self):
        """(vx, vy, yaw rate) of the base in its own frame (planar components)."""
        v = quat_rotate_inverse(self.data.qpos[3:7], self.data.qvel[0:3])
        return np.array([v[0], v[1], self.data.qvel[5]])

    def height(self):
        return float(self.data.qpos[2])

    # ---- one policy step (self.decim physics steps) ----------------------------------------------------------
    def step(self, cmd):
        cmd = np.asarray(cmd, float)
        if self.cmd_limits is not None:
            cmd = np.clip(cmd, self.cmd_limits[:, 0], self.cmd_limits[:, 1])
        for _ in range(self.decim):
            tau = (self.target - self.data.qpos[7:]) * self.kps - self.data.qvel[6:] * self.kds
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
            self.counter += 1
        obs = np.zeros(self.nobs, dtype=np.float32)
        obs[0:3] = self.data.qvel[3:6] * self.scales["ang"]
        obs[3:6] = gravity_orientation(self.data.qpos[3:7])
        obs[6:9] = cmd * self.scales["cmd"]
        obs[9:9 + self.na] = (self.data.qpos[7:] - self.default) * self.scales["pos"]
        obs[9 + self.na:9 + 2 * self.na] = self.data.qvel[6:] * self.scales["vel"]
        obs[9 + 2 * self.na:9 + 3 * self.na] = self.action
        phase = (self.counter * self.dt) % 0.8 / 0.8
        obs[9 + 3 * self.na:9 + 3 * self.na + 2] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
        with torch.no_grad():
            self.action = self.policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze().astype(float)
        self.target = self.action * self.scales["act"] + self.default
        g = gravity_orientation(self.data.qpos[3:7])
        if self.height() < 0.55 or g[2] > -0.5:
            self.fallen = True
        return self.fallen

    def run(self, cmd_fn, duration, record_every=1):
        """Run for `duration` seconds with cmd = cmd_fn(t, pose, body_vel). Returns dict of arrays."""
        n = int(round(duration / self.ctrl_dt))
        T, P, V, Cm = [], [], [], []
        for k in range(n):
            t = k * self.ctrl_dt
            p, v = self.pose(), self.body_velocity()
            c = np.asarray(cmd_fn(t, p, v), float)
            if k % record_every == 0:
                T.append(t)
                P.append(p)
                V.append(v)
                Cm.append(c)
            if self.step(c):
                break
        return dict(t=np.array(T), pose=np.array(P), vel=np.array(V), cmd=np.array(Cm), fallen=self.fallen)
