#!/usr/bin/env python3
"""G1 stand probe: hold ZERO action (default pose) for N steps, dump the
physics trajectory. Separates 'spawn config is broken' (falls with zero
action) from 'policy just needs training'.

Prints per step: root z, roll/pitch, feet contact, waist contact, |q_err|_inf.
Exit code 0 if the robot is still standing at the end (root z > 0.6 and no
base_contact), 1 otherwise.
"""
import os
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here.parent

os.environ.setdefault("G1_PRG", "1")
os.environ.setdefault("X1_GAIT_CLOCK", "1")
os.environ.setdefault("X1_STYLE_SCALE", "0")
os.environ.setdefault("G1_SMOKE_ENVS", "4")

_parts = [str(_root / "rsl_rl"), str(_root / "robolab")]
_old = os.environ.get("PYTHONPATH", "")
if _old:
    _parts.append(_old)
os.environ["PYTHONPATH"] = ":".join(_parts)
print(f"[STAND] PYTHONPATH={os.environ['PYTHONPATH']}", flush=True)

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import robolab.tasks  # noqa: F401,E402  (registers G1-AMP)

env = gym.make("G1-AMP", cfg_entry_point_type=None) if False else None
# use the plain isaclab make path
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from robolab.tasks.manager_based.amp.g1_amp_env_cfg import G1AmpEnvCfg  # noqa: E402

cfg = G1AmpEnvCfg()
cfg.scene.num_envs = 4
cfg.sim.device = os.environ.get("G1_PROBE_DEVICE", "cuda:0")
env = ManagerBasedRLEnv(cfg=cfg)

N = int(os.environ.get("G1_STAND_STEPS", "100"))
obs, _ = env.reset()
robot = env.scene["robot"]
q0 = robot.data.default_joint_pos.clone()
print(f"[STAND] q_default[:6]={q0[0, :6].tolist()}", flush=True)
print(f"[STAND] root z0={robot.data.root_pos_w[0, 2].item():.4f} "
      f"joint names[:6]={robot.data.joint_names[:6]}", flush=True)

fell = False
for i in range(N):
    obs, rew, term, trunc, info = env.step(torch.zeros(
        env.num_envs, robot.data.joint_names.__len__(), device=env.device))
    z = robot.data.root_pos_w[0, 2].item()
    qw = robot.data.root_quat_w[0, 0].item()
    q_err = (robot.data.joint_pos[0] - q0[0]).abs().max().item()
    if i % 5 == 0 or term.any():
        print(f"[STAND] step {i:3d} z={z:.4f} qw={qw:.3f} "
              f"|qerr|inf={q_err:.3f} term={term.tolist()} "
              f"rew={rew[0].item():.3f}", flush=True)
    if term[0]:
        fell = True
        break

z_end = robot.data.root_pos_w[0, 2].item()
standing = (not fell) and z_end > 0.60
print(f"[STAND] RESULT standing={standing} fell_at={'n/a' if not fell else i} "
      f"z_end={z_end:.4f}", flush=True)
env.close()
simulation_app.close()
sys.exit(0 if standing else 1)
