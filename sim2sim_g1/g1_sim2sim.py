#!/usr/bin/env python3
"""g1_sim2sim.py — Unitree G1 (29-DoF) sim2sim rollout: Isaac-trained policy
-> MuJoCo (g1_mocap_29dof.xml, GMR vendor model), PRG twin included.

Derived from the X1-verified sim2sim/mujoco_rollout.py (X1_29_AMP); every
block is item-by-item aligned with it.  G1-specific changes are marked
"[G1]" in comments.  Known deltas vs Isaac are listed in smoke_report.md
(gravity/integrator/contact params/latency/passive dynamics).

Control loop (aligned with X1 chain):
  50 Hz policy, PD torques recomputed every --pd-every physics substeps with a
  held position target (X1 v25+ semantics).  [G1] MuJoCo timestep forced to
  0.001 s => 20 substeps per 0.02 s control step (vendor XML default 0.002).

Observations (exactly the G1 training obs group, 300 dims total):
  frame(100) = [base_ang_vel(3), projected_gravity(3), cmd(3),
                joint_pos_rel(29), joint_vel_rel(29), last_action(29),
                gait_clock(4)]
  history 3, TERM-MAJOR flattening (each term's 3 frames concatenated,
  oldest-first within the term) — the IsaacLab obs-group layout proven by the
  X1 checkpoint normalizer statistics (mujoco_rollout.py v26 note).
  [G1] gait-clock column order = [sin(2pi phiL), sin(2pi phiR),
  cos(2pi phiL), cos(2pi phiR)] — the v71-authoritative order (lead-verified
  against mdp/gait_clock.py clock_obs = cat([sin, cos]); see prg_clock.py
  clock_obs docstring).

Gait clock / PRG twin (--prg-mode):
  none  v71 pure time formula  phi = (t / T*(|v_cmd|)) % 1  (B0 evaluation;
        reads the SAME X1_CLOCK_T_* env overrides as training gait_clock.py)
  hard  prg_clock.PRGState(reset_mode="hard")  — touchdown re-anchors phi=0
  soft  prg_clock.PRGState(reset_mode="soft")  — rate-limited pull toward 0
  hard/soft rebuild the training-identical PRGState (same default constants;
  PRG_* envvars of prg_clock.py apply), backend="numpy", and are stepped each
  control tick with (dt, v_cmd, contact(N,2), delta_f).  delta_f comes from
  the policy action tail when the action dim is 30 (29 joints + 1 delta_f);
  for 29-dim actions delta_f=None (v71 blending).

Policy channels (--ckpt):
  *.npz    l{i}_w / l{i}_b layers + mean/std normalizer (extract_ckpt_npz.py)
  *.onnx   exported graph with the fused normalizer (export_onnx.py)
  zero     action ≡ 0 (smoke; PD holds DEFAULT_Q)
  random   random-weight MLP (smoke; small init, action clipped to [-3,3])

Metrics JSON (--json): survival + tracking + gait metrics with gait_metrics.py
semantics (Schmitt trigger segmentation on lowest-sole-sphere height,
ON 4 mm / OFF 12 mm, 80 ms debounce): step cadence, duty factor, swing
duration, stance slip speed, and the PRG phase-contact sync error (circular
phase error of phi at Schmitt touchdowns; ideal 0 with PRG resets).
--log writes a gait_metrics-compatible npz (meta carries G1 hinge names; NOTE
upstream gait_metrics.py hard-codes X1 joint names in its G3/KH sections —
G1 reuse needs the X1->G1 renames from g1_joint_order.py).

Usage:
  python3 g1_sim2sim.py --ckpt zero --duration 4 --json out.json \
      [--video out.mp4] [--prg-mode none|hard|soft] [--cmd 1.0 0 0] ...
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

# make prg_clock / g1_joint_order importable when run from another cwd
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from g1_joint_order import LAB_DOF, MJ_ACTUATOR, build_lab2mj, verify_against_model  # noqa: E402

CONTROL_DT = 0.02          # [s] 50 Hz policy (X1/G1 AMP training lineage)
HIST = 3
ACTION_SCALE = 0.25        # JointPositionActionCfg scale (inherited from X1 cfg)
FRAME_DIM = 96 + 4         # [G1] clock policy: 96 legacy + 4 gait-clock
OBS_DIM = HIST * FRAME_DIM  # 300
SOLE_R = 0.005             # [G1] sole collision-sphere radius (x1 uses 0.002)
FALL_Z = 0.40              # [G1] standing height 0.79 -> fall below 0.40
FALL_TILT = 60.0           # deg
TWIN_DT = 0.001            # [G1] force MuJoCo timestep (XML default is 0.002)

# ---------------------------------------------------------------------------
# [G1] PD gains / effort limits / armature — first version ("v1"), derived
# verbatim from the training-side G1_CFG (g1_robot_cfg_ref.py, GMR rev_1_0
# limits; see g1_pd_calibration.md for provenance & pending identification).
# (kp, kd, effort_limit) per joint; armature = unitree_rl_lab motor-catalog
# reflected inertia (7520-14 0.0102 / 7520-22 0.0251 / 5020-16 0.0036 /
# W4010-25 0.00425).
# ---------------------------------------------------------------------------
G1_PD = {
    # legs: unitree_rl_gym official [HIGH]
    "left_hip_pitch_joint": (100.0, 8.0, 88.0),
    "right_hip_pitch_joint": (100.0, 8.0, 88.0),
    "left_hip_roll_joint": (100.0, 8.0, 139.0),
    "right_hip_roll_joint": (100.0, 8.0, 139.0),
    "left_hip_yaw_joint": (100.0, 8.0, 88.0),
    "right_hip_yaw_joint": (100.0, 8.0, 88.0),
    "left_knee_joint": (150.0, 8.0, 139.0),
    "right_knee_joint": (150.0, 8.0, 139.0),
    "left_ankle_pitch_joint": (40.0, 4.0, 50.0),
    "right_ankle_pitch_joint": (40.0, 4.0, 50.0),
    "left_ankle_roll_joint": (40.0, 2.5, 50.0),
    "right_ankle_roll_joint": (40.0, 2.5, 50.0),
    # waist: by motor class [MED] (yaw = 7520-14, roll/pitch = 5020-16)
    "waist_yaw_joint": (100.0, 8.0, 88.0),
    "waist_roll_joint": (40.0, 8.0, 50.0),
    "waist_pitch_joint": (40.0, 8.0, 50.0),
    # arms: IsaacLab 29dof-locomotion pattern + v2 stand-probe fix [MED-HIGH]
    "left_shoulder_pitch_joint": (100.0, 8.0, 25.0),
    "right_shoulder_pitch_joint": (100.0, 8.0, 25.0),
    "left_shoulder_roll_joint": (100.0, 8.0, 25.0),
    "right_shoulder_roll_joint": (100.0, 8.0, 25.0),
    "left_shoulder_yaw_joint": (60.0, 8.0, 25.0),
    "right_shoulder_yaw_joint": (60.0, 8.0, 25.0),
    "left_elbow_joint": (80.0, 8.0, 25.0),
    "right_elbow_joint": (80.0, 8.0, 25.0),
    "left_wrist_roll_joint": (20.0, 4.0, 25.0),
    "right_wrist_roll_joint": (20.0, 4.0, 25.0),
    "left_wrist_pitch_joint": (20.0, 4.0, 5.0),
    "right_wrist_pitch_joint": (20.0, 4.0, 5.0),
    "left_wrist_yaw_joint": (20.0, 4.0, 5.0),
    "right_wrist_yaw_joint": (20.0, 4.0, 5.0),
}
G1_ARMATURE = {
    "left_hip_pitch_joint": 0.0102, "right_hip_pitch_joint": 0.0102,
    "left_hip_roll_joint": 0.0251, "right_hip_roll_joint": 0.0251,
    "left_hip_yaw_joint": 0.0102, "right_hip_yaw_joint": 0.0102,
    "left_knee_joint": 0.0251, "right_knee_joint": 0.0251,
    "left_ankle_pitch_joint": 0.0036, "right_ankle_pitch_joint": 0.0036,
    "left_ankle_roll_joint": 0.0036, "right_ankle_roll_joint": 0.0036,
    "waist_yaw_joint": 0.0102, "waist_roll_joint": 0.0036,
    "waist_pitch_joint": 0.0036,
    "left_shoulder_pitch_joint": 0.0036, "right_shoulder_pitch_joint": 0.0036,
    "left_shoulder_roll_joint": 0.0036, "right_shoulder_roll_joint": 0.0036,
    "left_shoulder_yaw_joint": 0.0036, "right_shoulder_yaw_joint": 0.0036,
    "left_elbow_joint": 0.0036, "right_elbow_joint": 0.0036,
    "left_wrist_roll_joint": 0.0036, "right_wrist_roll_joint": 0.0036,
    "left_wrist_pitch_joint": 0.00425, "right_wrist_pitch_joint": 0.00425,
    "left_wrist_yaw_joint": 0.00425, "right_wrist_yaw_joint": 0.00425,
}

# [G1] G1_CFG.init_state joint_pos (FK-verified standing; pelvis 0.79)
G1_DEFAULT_Q = {
    "left_hip_pitch_joint": -0.15, "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0, "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.15, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.15, "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0, "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.15, "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.3,
    "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.3,
    "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


# ---------------------------------------------------------------------------
# Model build
# ---------------------------------------------------------------------------
def strip_mesh_xml(xml_text: str, drop_motors: bool = False) -> str:
    """[G1] Drop mesh assets/geoms from the vendor XML.

    The input tar ships g1_mocap_29dof.xml WITHOUT meshes/*.STL.  All mesh
    geoms are visual-only or coarse collision hulls; the REAL foot contacts
    are the 4 sole spheres per foot (r=0.005, contype default 1) which we
    keep — semantically aligned with the Isaac side's custom-collision URDF
    (sole cylinders).  Bodies keep their explicit <inertial>, so dynamics are
    unchanged.  Delta vs Isaac documented in smoke_report.md.

    drop_motors=True additionally removes the vendor <motor> actuators (used
    by the native-servo path so the injected <position> servos are the ONLY
    29 actuators and ctrl indices stay 1:1 with the joint table).
    """
    import re
    out = []
    for line in xml_text.splitlines():
        if re.search(r"<mesh\s+name=", line):
            continue
        if re.search(r'<geom[^>]*type="mesh"', line):
            continue
        if drop_motors and re.search(r"<motor\s+name=", line):
            continue
        out.append(line)
    return "\n".join(out)


def build_model(xml_path: Path, actuator_mode: str = "explicit",
                timestep: float = TWIN_DT, pd_table=None, armature_table=None):
    """Compile the G1 model with the sim2sim passive-dynamics alignment.

    Aligned with X1 build_model but [G1]-specific:
      * strip meshes (not shipped); floor ships IN the xml (friction 0.6)
      * dof_damping = 0 (G1 URDF carries NO joint damping — X1 set 1.0 because
        the X1 URDF had damping=1.0; setting 1.0 here would be WRONG)
      * dof_frictionloss = 0 (no dry friction on either side)
      * per-joint armature from G1_ARMATURE (motor catalog, pending ID)
      * native mode injects <position> servos (kp/kv/forcerange from the PD
        table) + implicitfast integrator, mirroring Isaac's implicit PD
    """
    import mujoco
    txt = strip_mesh_xml(Path(xml_path).read_text(),
                         drop_motors=(actuator_mode == "native"))
    if actuator_mode == "native":
        tbl = pd_table or G1_PD
        srv = "\n".join(
            f'<position name="servo_{n}" joint="{n}" kp="{kp}" kv="{kd}" '
            f'forcerange="-{eff} {eff}" ctrlrange="-6.28 6.28"/>'
            for n, (kp, kd, eff) in tbl.items())
        txt = txt.replace("</actuator>", srv + "\n  </actuator>")
        spec = mujoco.MjSpec.from_string(txt)
        model = spec.compile()
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    else:
        spec = mujoco.MjSpec.from_string(txt)
        model = spec.compile()
    model.opt.timestep = timestep

    arm = armature_table or G1_ARMATURE
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        adr = model.jnt_dofadr[j]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        model.dof_damping[adr] = 0.0        # [G1] URDF has no damping
        model.dof_frictionloss[adr] = 0.0   # no dry friction on either side
        model.dof_armature[adr] = arm.get(jname, 0.01)
    return model


def find_sole_geoms(model):
    """Per-foot 4 sole spheres + floor id (X1 find_sole_geoms semantics).

    [G1] g1_mocap_29dof.xml: spheres r=0.005 at local z=-0.03 — 2 heel
    (x=-0.05, y=±0.025) + 2 fore (x=+0.12, y=±0.03).  Sorted like X1 by
    (pos_x < 0, pos_y < 0): [heel_l, heel_r, fore_l, fore_r] — NOTE (X1
    gait_metrics LABELING CAVEAT): with this sort the [0,1] columns are the
    heel end and [2,3] the toe end.
    """
    import mujoco
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor >= 0, "floor geom missing from xml"
    out, colliders = {}, {}
    for b in range(model.nbody):
        name = model.body(b).name
        if not name.endswith("ankle_roll_link"):
            continue
        g0, gnum = int(model.body_geomadr[b]), int(model.body_geomnum[b])
        spheres = [g for g in range(g0, g0 + gnum)
                   if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                   and model.geom_size[g, 0] < 0.01]
        assert len(spheres) == 4, f"{name}: expected 4 sole spheres, got {len(spheres)}"
        spheres = sorted(spheres, key=lambda g: (model.geom_pos[g, 0] < 0,
                                                 model.geom_pos[g, 1] < 0))
        out[name] = spheres
        colliders[name] = [g for g in range(g0, g0 + gnum)
                           if model.geom_contype[g] or model.geom_conaffinity[g]]
    return out, floor, colliders


# ---------------------------------------------------------------------------
# Policy loading (npz / onnx / zero / random)
# ---------------------------------------------------------------------------
def load_policy_npz(path: Path):
    """l{i}_w/l{i}_b layers + mean/std (extract_ckpt_npz.py format)."""
    z = np.load(path)
    layers, i = [], 0
    while f"l{i}_w" in z.files:
        w = z[f"l{i}_w"].astype(np.float32)
        b = z[f"l{i}_b"].astype(np.float32) if f"l{i}_b" in z.files else None
        layers.append((w, b))
        i += 1
    if not layers:
        raise RuntimeError(f"no l*_w arrays in {path}")
    mean = z["mean"].astype(np.float32).reshape(-1)
    std = z["std"].astype(np.float32).reshape(-1)
    return layers, mean, std, {"src": "npz", "path": str(path)}


def run_policy_mlp(layers, mean, std, obs):
    """rsl_rl MLP: Linear -> ELU between layers, raw mean action (X1 exact)."""
    x = (np.asarray(obs, dtype=np.float32) - mean) / std
    for w, b in layers[:-1]:
        x = x @ w.T + (b if b is not None else 0)
        x = np.where(x < 0, np.expm1(np.minimum(x, 20)), x)  # ELU(alpha=1)
    w, b = layers[-1]
    return x @ w.T + (b if b is not None else 0)


def load_policy_random(in_dim, out_dim, seed=0):
    rng = np.random.default_rng(seed)
    dims = [in_dim, 128, 128, out_dim]
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers.append((rng.normal(0, 0.05, (b, a)).astype(np.float32),
                       np.zeros(b, dtype=np.float32)))
    mean = np.zeros(in_dim, dtype=np.float32)
    std = np.ones(in_dim, dtype=np.float32)
    return layers, mean, std, {"src": "random-mlp", "seed": seed}


def make_policy(kind: str, path=None, act_dim=30, seed=0):
    """Return (policy_fn, act_dim, meta). policy_fn(obs300) -> action."""
    if kind == "zero":
        def fn(ob):
            return np.zeros(act_dim, dtype=np.float64)
        return fn, act_dim, {"src": "zero"}
    if kind == "random":
        layers, mean, std, meta = load_policy_random(OBS_DIM, act_dim, seed)

        def fn(ob):
            return np.clip(run_policy_mlp(layers, mean, std, ob), -3.0, 3.0)
        return fn, act_dim, meta
    if kind == "npz":
        layers, mean, std, meta = load_policy_npz(Path(path))
        in_dim = layers[0][0].shape[1]
        out_dim = layers[-1][0].shape[0]

        def fn(ob):
            return run_policy_mlp(layers, mean, std, ob)
        return fn, out_dim, meta
    if kind == "onnx":
        import onnxruntime as _ort
        sess = _ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        in_dim = sess.get_inputs()[0].shape[-1]
        probe = sess.run([sess.get_outputs()[0].name],
                         {name: np.zeros((1, in_dim), dtype=np.float32)})[0]
        out_dim = int(probe.reshape(-1).shape[0])

        def fn(ob):
            ob2 = np.asarray(ob, dtype=np.float32).reshape(1, -1)
            return sess.run([sess.get_outputs()[0].name], {name: ob2})[0].reshape(-1)
        return fn, out_dim, {"src": "onnx", "path": str(path)}
    raise ValueError(f"unknown policy kind {kind}")


# ---------------------------------------------------------------------------
# Gait-clock twin: v71 pure formula (none) or PRGState (hard/soft)
# ---------------------------------------------------------------------------
def v71_clock(step, dt, v_cmd):
    """v71 pure-function clock, EXACT mujoco_rollout.py GAIT_CLOCK_OBS twin.

    Reads the SAME X1_CLOCK_T_* env overrides as the training-side
    gait_clock.py (defaults 1.46/0.36/0.70/1.75 — identical to prg_clock).
    Returns (obs4 [sinL,sinR,cosL,cosR], phi(2,)).
    """
    tA = float(os.environ.get("X1_CLOCK_T_A", "1.46"))
    tB = float(os.environ.get("X1_CLOCK_T_B", "0.36"))
    tLo = float(os.environ.get("X1_CLOCK_T_LO", "0.70"))
    tHi = float(os.environ.get("X1_CLOCK_T_HI", "1.75"))
    t = step * dt
    v = float(np.linalg.norm(v_cmd[:2]))
    T = min(max(tA - tB * v, tLo), tHi)
    pl = (t / T) % 1.0
    pr = (pl + 0.5) % 1.0
    obs = np.array([np.sin(2 * np.pi * pl), np.sin(2 * np.pi * pr),
                    np.cos(2 * np.pi * pl), np.cos(2 * np.pi * pr)],
                   dtype=np.float32)
    return obs, np.array([pl, pr])


class GaitClockTwin:
    """Deploy-side twin of the training gait clock.

    none: v71 pure formula (loop counter + fixed cmd) — byte-aligned with the
          X1 mujoco_rollout twin (B0 evaluation mode).
    hard/soft: prg_clock.PRGState rebuilt with the SAME parameters as training
          (module defaults; PRG_* envvars apply), backend="numpy", stepped
          every control tick with (dt, v_cmd, contact(1,2), delta_f).
    """

    def __init__(self, mode: str, dt: float, freeze_alpha: bool = False):
        assert mode in ("none", "hard", "soft")
        self.mode = mode
        self.dt = dt
        self.step_i = 0
        self.prg = None
        self.reset_events = 0
        if mode in ("hard", "soft"):
            from prg_clock import PRGState  # training-identical module
            self.prg = PRGState(num_envs=1, backend="numpy",
                                reset_mode=mode, freeze_alpha=freeze_alpha)
            self.prg.reset()

    def step(self, v_cmd, contact_lr, delta_f=None):
        """Advance one control tick. contact_lr = (2,) bool [left, right]."""
        self.step_i += 1
        if self.prg is None:
            obs, phi = v71_clock(self.step_i - 1, self.dt, v_cmd)
            return obs, phi
        phi_before = self.prg.phi.copy()
        self.prg.step(self.dt, np.array([float(np.linalg.norm(v_cmd[:2]))]),
                      np.asarray(contact_lr, dtype=bool).reshape(1, 2),
                      None if delta_f is None else np.array([float(delta_f)]))
        obs = np.asarray(self.prg.clock_obs(), dtype=np.float32).reshape(-1)
        # count resets fired this tick (phase discontinuity at touchdown)
        if self.mode == "hard":
            jumped = np.abs(self.prg.phi - phi_before)
            self.reset_events += int(np.sum((jumped > 0.2)
                                            & np.asarray(contact_lr, dtype=bool)))
        return obs, np.asarray(self.prg.phi).reshape(-1)

    @property
    def f_eff_hz(self):
        return float(self.prg.f_eff[0]) if self.prg is not None else float("nan")

    @property
    def alpha(self):
        return float(self.prg.alpha[0]) if self.prg is not None else 0.0


# ---------------------------------------------------------------------------
# Gait metrics (gait_metrics.py semantics, G1-native joint names)
# ---------------------------------------------------------------------------
ON_M, OFF_M = 0.004, 0.012  # Schmitt trigger on lowest sole-sphere BOTTOM height


def schmitt_contact(min_z, dt):
    """gait_metrics.schmitt_contact exact twin (debounce windows rescaled)."""
    min_on = max(2, int(round(0.08 / dt)))
    min_off = max(2, int(round(0.08 / dt)))
    n = len(min_z)
    on = np.zeros(n, dtype=bool)
    state = bool(min_z[0] < ON_M)
    i = 0
    while i < n:
        if state:
            j = i
            while j < n and min_z[j] < OFF_M:
                j += 1
            on[i:j] = True
            state = False
            i = j
        else:
            j = i
            while j < n and min_z[j] >= ON_M:
                j += 1
            if j - i < min_off and j < n and i > 0:
                on[i:j] = True
            state = True
            i = j
    return on


def stance_windows(contact, start_i, dt):
    """gait_metrics.stance_windows twin: [(td_i, liftoff_i), ...]."""
    min_on = max(2, int(round(0.08 / dt)))
    on = contact.astype(int)
    edges = np.diff(on, prepend=0)
    tds = list(np.where(edges == 1)[0])
    los = list(np.where(edges == -1)[0] - 1)
    if tds and los and los[0] < tds[0]:
        los = los[1:]
    if tds and los and tds[-1] > los[-1]:
        tds = tds[:-1]
    return [(t, l) for t, l in zip(tds, los) if t >= start_i and l - t >= min_on]


def wrap_circ(x):
    """Wrap phase error to [-0.5, 0.5)."""
    return (np.asarray(x) + 0.5) % 1.0 - 0.5


def gait_metrics_summary(t, sole_xyz, phi_hist, start_i, dt,
                         foot_names=("left", "right")):
    """Per-foot cadence/duty/swing/slip + phase-contact sync error.

    sole_xyz: (n, 2, 4, 3) sole sphere centres; bottoms = z - SOLE_R.
    phi_hist: (n, 2) gait-clock twin phases at each control tick.
    """
    bottoms = sole_xyz[:, :, :, 2] - SOLE_R           # (n, 2, 4)
    mid_xy = sole_xyz[:, :, :, :2].mean(axis=2)       # (n, 2, 2)
    out = {}
    sync_errs = {f: [] for f in foot_names}
    for f_i, fname in enumerate(foot_names):
        contact = schmitt_contact(bottoms[:, f_i].min(axis=1), dt)
        strides = stance_windows(contact, start_i, dt)
        tds = [td for td, _ in strides]
        cyc = list(np.diff(tds) * dt) if len(tds) >= 2 else []
        duty, swing, slip, steplen = [], [], [], []
        for (td, lo) in strides:
            dur = (lo - td) * dt
            duty.append(dur / cyc[0] if cyc else np.nan)  # refined below
            slip.append(float(np.linalg.norm(mid_xy[lo, f_i] - mid_xy[td, f_i])
                              / dur) if dur > 0 else np.nan)
        if len(tds) >= 2:
            # duty vs the OWN median cycle (robust to drift)
            med_cyc = float(np.median(cyc)) if cyc else np.nan
            duty = [((lo - td) * dt) / med_cyc for td, lo in strides]
            swing, steplen = [], []
            for k, (td, lo) in enumerate(strides):
                nxt_td = strides[k + 1][0] if k + 1 < len(strides) else None
                if nxt_td is not None and nxt_td > lo:
                    swing.append((nxt_td - lo) * dt)
                sync_errs[fname].append(float(wrap_circ(phi_hist[td, f_i])))
            for a, b in zip(tds[:-1], tds[1:]):
                steplen.append(float(np.linalg.norm(mid_xy[b, f_i] - mid_xy[a, f_i])))
        else:
            duty, swing, steplen, slip = [], [], [], []
        errs = np.array(sync_errs[fname]) if sync_errs[fname] else np.array([np.nan])
        out[fname] = {
            "n_events": len(strides),
            "cycle_s_med": float(np.median(cyc)) if cyc else None,
            "cadence_hz": float(1.0 / np.median(cyc)) if cyc else None,
            "cadence_spm": float(120.0 / np.median(cyc)) if cyc else None,
            "duty_med": float(np.median(duty)) if duty else None,
            "swing_s_med": float(np.median(swing)) if swing else None,
            "step_len_m_med": float(np.median(steplen)) if steplen else None,
            "slip_mps_med": float(np.median(slip)) if slip else None,
            "phase_sync_err_mean_abs": float(np.nanmean(np.abs(errs))),
            "phase_sync_err_std": float(np.nanstd(errs)),
            "phase_sync_err_max_abs": float(np.nanmax(np.abs(errs))),
        }
    valid = [v["phase_sync_err_mean_abs"] for v in out.values()
             if v["n_events"] > 0]
    out["overall"] = {
        "phase_sync_err_mean_abs": float(np.mean(valid)) if valid else None,
    }
    return out


# ---------------------------------------------------------------------------
# Soft (no-GL) stick renderer — X1 SoftRenderer twin
# ---------------------------------------------------------------------------
class SoftRenderer:
    def __init__(self, model, height=720, width=1280):
        os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "mplcache"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self._plt = plt
        self.model = model
        self.parent = [int(model.body_parentid[i]) for i in range(model.nbody)]
        self.foot_ids = [i for i in range(model.nbody)
                         if model.body(i).name.endswith("ankle_roll_link")]
        self.trail = []
        dpi = 100
        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection="3d")

    def capture(self, data) -> np.ndarray:
        ax, plt = self.ax, self._plt
        P = data.xpos.copy()
        root = P[1]
        self.trail.append(root.copy())
        if len(self.trail) > 100:
            self.trail.pop(0)
        ax.clear()
        for i in range(1, self.model.nbody):
            a, b = P[self.parent[i]], P[i]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="0.25", lw=1.8)
        for fi in self.foot_ids:
            f = P[fi]
            ax.scatter([f[0]], [f[1]], [f[2]], color="crimson", s=42)
        tr = np.array(self.trail)
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="steelblue", lw=1.2, alpha=0.8)
        ax.scatter([root[0]], [root[1]], [root[2]], color="navy", s=60)
        ax.set_xlim(root[0] - 1.5, root[0] + 1.5)
        ax.set_ylim(root[1] - 1.5, root[1] + 1.5)
        ax.set_zlim(0.0, 1.6)
        gx = np.arange(root[0] - 1.5, root[0] + 1.51, 0.5)
        gy = np.arange(root[1] - 1.5, root[1] + 1.51, 0.5)
        for x in gx:
            ax.plot([x, x], [gy[0], gy[-1]], [0, 0], color="0.75", lw=0.5)
        for y in gy:
            ax.plot([gx[0], gx[-1]], [y, y], [0, 0], color="0.75", lw=0.5)
        ax.view_init(elev=-18, azim=-75)
        ax.set_box_aspect((1, 1, 0.53))
        ax.set_axis_off()
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()

    def close(self):
        try:
            self._plt.close(self.fig)
        except Exception:
            pass


def quat2mat(q):
    import mujoco
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.ascontiguousarray(q, dtype=np.float64))
    return out.reshape(3, 3)


# ---------------------------------------------------------------------------
# Main rollout
# ---------------------------------------------------------------------------
def main():
    import mujoco
    global CONTROL_DT

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help=".policy.npz | .onnx | zero | random")
    ap.add_argument("--xml", default=str(_HERE / "g1_mocap_29dof.xml"))
    ap.add_argument("--cmd", nargs=3, type=float, default=[0.8, 0.0, 0.0],
                    help="vx vy wz (G1 training band: x 0.2-1.5, y +-0.2, wz +-0.5)")
    ap.add_argument("--control-dt", type=float, default=CONTROL_DT)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--prg-mode", choices=["none", "hard", "soft"], default="none")
    ap.add_argument("--freeze-alpha", action="store_true",
                    help="pin the PRG alpha curriculum at 0 (v71 blending)")
    ap.add_argument("--actuator", choices=["explicit", "native"], default="explicit")
    ap.add_argument("--pd-every", type=int, default=5,
                    help="recompute explicit PD every N physics substeps (5=200 Hz)")
    ap.add_argument("--pd-json", default=None,
                    help="JSON file {joint: [kp, kd, effort]} overriding G1_PD at "
                         "runtime (see g1_pd_calibration.md: MuJoCo explicit-PD "
                         "stability needs kd*dt_eff/armature < 0.8 — the G1_CFG "
                         "wrist gains violate it at pd-every=5)")
    ap.add_argument("--video", default=None)
    ap.add_argument("--render", choices=["gl", "soft"], default="soft")
    ap.add_argument("--json", default=None)
    ap.add_argument("--log", default=None, help="gait_metrics-compatible npz")
    ap.add_argument("--seed", type=int, default=0)
    # robustness injection (X1-parity)
    ap.add_argument("--obs-noise", type=float, default=0.0)
    ap.add_argument("--latency-steps", type=int, default=0)
    ap.add_argument("--action-lag", type=int, default=0)
    ap.add_argument("--action-lpf", type=float, default=0.0)
    ap.add_argument("--action-lpf-order", type=int, default=2)
    ap.add_argument("--push-mag", type=float, default=0.0)
    ap.add_argument("--push-every", type=float, default=2.0)
    ap.add_argument("--torso-mass-scale", type=float, default=1.0)
    args = ap.parse_args()
    CONTROL_DT = float(args.control_dt)

    # ---- model ----------------------------------------------------------
    pd_table = G1_PD
    if args.pd_json:
        ov = json.loads(Path(args.pd_json).read_text())
        pd_table = dict(G1_PD)
        for n, v in ov.items():
            assert n in pd_table, f"--pd-json: unknown joint {n}"
            assert len(v) == 3, f"--pd-json: {n} needs [kp, kd, effort]"
            pd_table[n] = (float(v[0]), float(v[1]), float(v[2]))
        print(f"[INFO] PD table overridden by {args.pd_json} "
              f"({len(ov)} joints)")
    model = build_model(Path(args.xml), args.actuator, pd_table=pd_table)
    info = verify_against_model(model, check_actuators=(args.actuator != "native"))
    print(f"[INFO] model OK: nu={model.nu} hinges=29, lab2mj identity="
          f"{info['lab2mj_is_identity']}, timestep={model.opt.timestep}")
    data = mujoco.MjData(model)
    soles, floor_id, foot_colliders = find_sole_geoms(model)
    foot_names = sorted(soles)   # [left_..., right_...]
    sole_gids = np.array([soles[f] for f in foot_names])   # (2,4)
    geom2foot = {}
    for f_i, fname in enumerate(foot_names):
        for g in foot_colliders.get(fname, []):
            geom2foot[g] = f_i
    lab2mj = np.array(build_lab2mj())

    # hinge addressing (MuJoCo order == MJ_ACTUATOR, asserted by verify)
    hinge = list(MJ_ACTUATOR)
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hinge}
    qadr = np.array([model.jnt_qposadr[jid[n]] for n in hinge])
    vadr = np.array([model.jnt_dofadr[jid[n]] for n in hinge])
    kp = np.array([pd_table[n][0] for n in hinge])
    kd = np.array([pd_table[n][1] for n in hinge])
    eff = np.array([pd_table[n][2] for n in hinge])
    q_default = np.array([G1_DEFAULT_Q[n] for n in hinge])
    servo_idx = None
    if args.actuator == "native":
        servo_idx = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"servo_{n}")
                     for n in hinge]
        assert all(i >= 0 for i in servo_idx), "servo actuators missing"

    # ---- policy ---------------------------------------------------------
    kind = args.ckpt if args.ckpt in ("zero", "random") else \
        Path(args.ckpt).suffix.lstrip(".")
    policy_fn, act_dim, pmeta = make_policy(kind, args.ckpt, act_dim=30,
                                            seed=args.seed)
    print(f"[INFO] policy src={pmeta['src']} act_dim={act_dim} "
          f"(delta_f={'tail' if act_dim == 30 else 'None (29-dim action)'})")
    assert act_dim in (29, 30), f"unexpected action dim {act_dim}"

    # ---- twin + initial state -------------------------------------------
    twin = GaitClockTwin(args.prg_mode, CONTROL_DT, freeze_alpha=args.freeze_alpha)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0, 0, 0.75]           # low start -> contact-calibrate (X1 v25)
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[qadr] = q_default
    mujoco.mj_forward(model, data)
    if data.ncon:
        worst = min(float(data.contact[i].dist) for i in range(data.ncon))
        data.qpos[2] -= worst + 0.002
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    print(f"[INFO] initial base z = {data.qpos[2]:.4f} "
          f"({data.ncon} sole contacts)")

    cmd = np.array(args.cmd, dtype=np.float64)
    substeps = max(1, round(CONTROL_DT / model.opt.timestep))
    n_steps = int(args.duration / CONTROL_DT)
    settle_steps = int(args.settle / CONTROL_DT)

    hist = np.zeros((HIST, FRAME_DIM), dtype=np.float32)
    last_act = np.zeros(29, dtype=np.float64)
    last_delta_f = None          # policy delta_f tail from the PREVIOUS tick
    frames = []
    vxy_err, yaw_err, alive_steps = [], [], 0
    fell, fall_reason = False, None
    LOG = args.log is not None
    L = {k: [] for k in ("t", "q", "dq", "base_pos", "base_quat", "v_b",
                         "sole_xyz", "contact_dist", "phi")}
    video = soft = None
    if args.video:
        if args.render == "soft":
            soft = SoftRenderer(model)
        else:
            video = mujoco.Renderer(model, height=720, width=1280)
            cam = mujoco.MjvCamera()

    rng = np.random.default_rng(args.seed)
    if args.torso_mass_scale != 1.0:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        model.body_mass[bid] *= args.torso_mass_scale
        print(f"[ROB] torso mass scaled by {args.torso_mass_scale}")
    ob_buf = collections.deque(maxlen=max(1, args.latency_steps) + 1)
    act_buf = collections.deque(maxlen=max(1, args.action_lag) + 1)
    lpf_alpha = (1.0 - math.exp(-2.0 * math.pi * args.action_lpf * CONTROL_DT)
                 if args.action_lpf > 0 else None)
    lpf_state = [np.zeros(29) for _ in range(max(1, args.action_lpf_order))]
    push_every_steps = int(args.push_every / CONTROL_DT)
    NOISE = dict(jpos=0.01, jvel=0.15, ang=0.05, grav=0.03)

    # term-major flatten slices (X1 v26 ROOT-CAUSE layout), clock appended
    slices = [(0, 3), (3, 6), (6, 9), (9, 38), (38, 67), (67, 96), (96, 100)]

    for step in range(n_steps):
        # ---------- obs frame (100) --------------------------------------
        q = data.qpos[qadr]
        dq = data.qvel[vadr]
        R = quat2mat(data.qpos[3:7])
        ang_b = data.qvel[3:6].copy()      # body frame (X1-verified identity)
        grav_b = R.T @ np.array([0, 0, -1.0])
        v_b = R.T @ data.qvel[0:3]
        if args.obs_noise > 0:
            n = args.obs_noise
            ang_b = ang_b + rng.normal(0, NOISE["ang"] * n, 3)
            grav_b = grav_b + rng.normal(0, NOISE["grav"] * n, 3)
            qn = q + rng.normal(0, NOISE["jpos"] * n, len(q))
            dqn = dq + rng.normal(0, NOISE["jvel"] * n, len(dq))
        else:
            qn, dqn = q, dq

        # foot contact (geometric: any sole sphere touching the floor)
        contact = np.zeros(2, dtype=bool)
        cd = np.full((2, 4), np.inf)
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if g1 != floor_id and g2 != floor_id:
                continue
            other = g2 if g1 == floor_id else g1
            fi = geom2foot.get(int(other))
            if fi is None:
                continue
            cd[fi, :] = min(cd[fi, 0], c.dist)
        cd[cd == np.inf] = 0.0
        contact = (cd.min(axis=1) <= 0.0)

        # gait-clock twin step (uses PREVIOUS-tick contact for touchdown edge
        # inside PRGState; obs of THIS tick reflects the advanced phase)
        delta_f = None
        if act_dim == 30 and last_delta_f is not None:
            delta_f = float(np.clip(last_delta_f, -1.0, 1.0))
        clock_obs, phi_now = twin.step(cmd, contact, delta_f)

        obs = np.concatenate([ang_b, grav_b, cmd,
                              (qn - q_default)[lab2mj], dqn[lab2mj], last_act,
                              clock_obs]).astype(np.float32)
        hist[:-1] = hist[1:]
        hist[-1] = obs
        ob = np.concatenate([hist[:, a:b].reshape(-1) for a, b in slices])
        assert ob.shape == (OBS_DIM,), f"obs assembled {ob.shape}, want {OBS_DIM}"
        ob_buf.appendleft(ob.copy())
        ob_eff = ob_buf[min(args.latency_steps, len(ob_buf) - 1)]

        # ---------- action ------------------------------------------------
        if step < settle_steps:
            act = np.zeros(act_dim)
        else:
            act = np.asarray(policy_fn(ob_eff), dtype=np.float64)
        if lpf_alpha is not None:
            for _i in range(len(lpf_state)):
                lpf_state[_i] = lpf_alpha * act[:29] + (1.0 - lpf_alpha) * lpf_state[_i]
                act = np.concatenate([lpf_state[_i], act[29:]]) if act_dim == 30 \
                    else lpf_state[_i].copy()
        act_buf.appendleft(np.asarray(act, dtype=np.float64).copy())
        act = act_buf[min(args.action_lag, len(act_buf) - 1)]
        last_act = act[:29].copy()
        last_delta_f = act[29] if act_dim == 30 else None
        if args.push_mag > 0 and step >= settle_steps and step % push_every_steps == 0:
            theta = rng.uniform(0, 2 * np.pi)
            data.qvel[0] += args.push_mag * np.cos(theta)
            data.qvel[1] += args.push_mag * np.sin(theta)

        q_tgt_full = q_default.copy()
        q_tgt_full[lab2mj] = q_default[lab2mj] + ACTION_SCALE * act[:29]

        # ---------- physics ------------------------------------------------
        if servo_idx is not None:
            data.ctrl[:] = 0.0
            data.ctrl[servo_idx] = q_tgt_full
            for _ in range(substeps):
                mujoco.mj_step(model, data)
        else:
            for s_i in range(substeps):
                if s_i % args.pd_every == 0:
                    tau = kp * (q_tgt_full - data.qpos[qadr]) - kd * data.qvel[vadr]
                    tau = np.clip(tau, -eff, eff)
                    data.qfrc_applied[vadr] = tau
                mujoco.mj_step(model, data)

        # ---------- metrics / logging ---------------------------------------
        base_z = data.qpos[2]
        tilt = np.degrees(np.arccos(np.clip(quat2mat(data.qpos[3:7])[2, 2], -1, 1)))
        R = quat2mat(data.qpos[3:7])
        v_b = R.T @ data.qvel[0:3]
        if step >= settle_steps:
            alive_steps += 1
            vxy_err.append(np.linalg.norm(v_b[:2] - cmd[:2]))
            yaw_err.append(abs(data.qvel[5] - cmd[2]))
            if base_z < FALL_Z or tilt > FALL_TILT:
                fell, fall_reason = True, f"z={base_z:.3f}" if base_z < FALL_Z \
                    else f"tilt={tilt:.1f}"
                print(f"[FALL] step {step} t={step*CONTROL_DT:.2f}s "
                      f"base_z={base_z:.3f} tilt={tilt:.1f}")
                break
        if LOG:
            L["t"].append(step * CONTROL_DT)
            L["q"].append(data.qpos[qadr].copy())
            L["dq"].append(data.qvel[vadr].copy())
            L["base_pos"].append(data.qpos[0:3].copy())
            L["base_quat"].append(data.qpos[3:7].copy())
            L["v_b"].append(v_b.copy())
            L["sole_xyz"].append(
                np.array([[data.geom_xpos[g].copy() for g in row]
                          for row in sole_gids]))
            L["contact_dist"].append(cd)
            L["phi"].append(phi_now.copy())
        if args.video:
            if soft is not None:
                frames.append(soft.capture(data))
            else:
                cam.lookat[:] = data.qpos[0:3].copy() + [0, 0, 0.2]
                cam.distance, cam.azimuth, cam.elevation = 2.8, 90.0, -12.0
                video.update_scene(data, camera=cam)
                frames.append(video.render())

    print(f"[INFO] survived {alive_steps} steps = {alive_steps*CONTROL_DT:.2f}s "
          f"of {n_steps - settle_steps} commanded steps "
          f"(prg={args.prg_mode}, resets={twin.reset_events}, "
          f"f_eff={twin.f_eff_hz:.3f} Hz, alpha={twin.alpha:.3f})")
    if vxy_err:
        print(f"[INFO] mean |v_xy err| = {np.mean(vxy_err):.3f} m/s, "
              f"mean |yaw err| = {np.mean(yaw_err):.3f} rad/s, "
              f"distance = {np.linalg.norm(data.qpos[:2]):.2f} m")

    # ---------- gait metrics + JSON ---------------------------------------
    result = {
        "meta": {
            "robot": "g1_29dof (GMR g1_mocap_29dof.xml, meshes stripped)",
            "ckpt": str(args.ckpt), "policy": pmeta, "act_dim": act_dim,
            "delta_f_channel": act_dim == 30,
            "cmd": list(args.cmd), "control_dt": CONTROL_DT,
            "duration_s": args.duration, "settle_s": args.settle,
            "prg_mode": args.prg_mode, "freeze_alpha": args.freeze_alpha,
            "prg_reset_events": twin.reset_events,
            "prg_f_eff_hz_final": None if math.isnan(twin.f_eff_hz) else twin.f_eff_hz,
            "prg_alpha_final": twin.alpha,
            "actuator_mode": args.actuator, "pd_every": args.pd_every,
            "mujoco_timestep": model.opt.timestep,
            "substeps": substeps,
            "pd_json": args.pd_json,
            "pd_table": "G1 v1 (G1_CFG-derived)" if not args.pd_json
                        else f"G1 v1 + override {args.pd_json}",
            "obs_dim": OBS_DIM, "frame_dim": FRAME_DIM, "hist": HIST,
            "clock_layout": "[sinL, sinR, cosL, cosR]",
            "obs_layout": "term-major x3: ang_b(3) grav_b(3) cmd(3) "
                          "qpos_rel(29) qvel_rel(29) last_act(29) clock(4)",
            "robustness": {"obs_noise": args.obs_noise,
                           "latency_steps": args.latency_steps,
                           "action_lag": args.action_lag,
                           "action_lpf": args.action_lpf,
                           "push_mag": args.push_mag,
                           "torso_mass_scale": args.torso_mass_scale,
                           "seed": args.seed},
        },
        "survival": {
            "survived_steps": alive_steps,
            "survived_s": alive_steps * CONTROL_DT,
            "fell": bool(fell or alive_steps < n_steps - settle_steps),
            "fall_reason": fall_reason,
            "final_base_z": float(data.qpos[2]),
            "distance_m": float(np.linalg.norm(data.qpos[:2])),
            "mean_vxy_err": float(np.mean(vxy_err)) if vxy_err else None,
            "mean_yaw_err": float(np.mean(yaw_err)) if yaw_err else None,
        },
    }
    if LOG and len(L["t"]) > 5:
        gm = gait_metrics_summary(
            np.array(L["t"]), np.array(L["sole_xyz"]),
            np.array(L["phi"]), settle_steps + 2, CONTROL_DT,
            foot_names=["left_ankle_roll_link", "right_ankle_roll_link"])
        result["gait"] = gm
        print("[GAIT] " + json.dumps(
            {k: {kk: vv for kk, vv in v.items()
                 if kk in ("n_events", "cycle_s_med", "duty_med", "swing_s_med",
                           "slip_mps_med", "phase_sync_err_mean_abs")}
             for k, v in gm.items()}, default=str))

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, default=str))
        print(f"[JSON] {args.json}")
    if LOG:
        npz = {k: np.asarray(v) for k, v in L.items()}
        npz["meta"] = json.dumps({
            "robot": "g1", "cmd": list(args.cmd), "ckpt": str(args.ckpt),
            "control_dt": CONTROL_DT, "settle_steps": settle_steps,
            "prg_mode": args.prg_mode,
            "fell": bool(fell or alive_steps < n_steps - settle_steps),
            "hinge_names": hinge, "foot_names": foot_names,
            "sole_order": "row=foot_names, cols=[toe_l, toe_r, heel_l, heel_r] "
                          "(x=+0.12 pair first: sort key (x<0,y<0) puts the "
                          "FORE pair in cols [0,1], heel x=-0.05 in [2,3]); "
                          "sole_xyz are sphere CENTRES of radius 0.005",
            "robustness": {"obs_noise": args.obs_noise,
                           "latency_steps": args.latency_steps,
                           "action_lag": args.action_lag,
                           "action_lpf": args.action_lpf,
                           "torso_mass_scale": args.torso_mass_scale,
                           "push_mag": args.push_mag, "seed": args.seed},
        })
        np.savez_compressed(args.log, **npz)
        print(f"[LOG] {args.log} ({len(L['t'])} steps)")
    if args.video and frames:
        import imageio.v2 as imageio
        imageio.mimwrite(args.video, frames, fps=int(1 / CONTROL_DT), quality=8)
        print(f"[VIDEO] {args.video} ({len(frames)} frames, "
              f"{Path(args.video).stat().st_size // 1024}KB)")
    if soft:
        soft.close()
    if video:
        video.close()


if __name__ == "__main__":
    main()
