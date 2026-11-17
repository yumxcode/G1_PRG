# G1 (Unitree, 29-DoF, rev_1_0) — two URDF variants, joint-by-joint comparison

Inputs compared (from `g1cfg-input.tar`):

| label | file | robot name | links | fixed joints | revolute | total mass |
|---|---|---|---|---|---|---|
| **GMR** | `g1_custom_collision_29dof.urdf` | `g1_29dof_rev_1_0` | 38 | 8 | 29 | **33.340 kg** |
| **UMR** | `g1_29dof.urdf` | `g1_29dof` | 55 | 25 | 29 | 35.149 kg |

GMR = the vendored Unitree G1 asset inside the GMR (mocap/retarget) stack; UMR = the
copy inside `UMR_weilai/assets/g1`. Both are derived from
`unitreerobotics/unitree_ros`; GMR tracks the **`g1_29dof_rev_1_0`** hardware
revision, UMR tracks the **pre-rev-1.0** description.

## 1. Joint name set / order

**Identical.** All 29 revolute joints, same names, same order:

```
left_hip_pitch_joint  left_hip_roll_joint  left_hip_yaw_joint  left_knee_joint
left_ankle_pitch_joint  left_ankle_roll_joint
right_hip_pitch_joint  right_hip_roll_joint  right_hip_yaw_joint  right_knee_joint
right_ankle_pitch_joint  right_ankle_roll_joint
waist_yaw_joint  waist_roll_joint  waist_pitch_joint
left_shoulder_pitch_joint  left_shoulder_roll_joint  left_shoulder_yaw_joint
left_elbow_joint  left_wrist_roll_joint  left_wrist_pitch_joint  left_wrist_yaw_joint
right_shoulder_pitch_joint  right_shoulder_roll_joint  right_shoulder_yaw_joint
right_elbow_joint  right_wrist_roll_joint  right_wrist_pitch_joint  right_wrist_yaw_joint
```

Note vs X1 convention: G1 knee is `*_knee_joint` (not `knee_pitch`), waist is
`waist_*` (not `lumbar_*`), arms are shoulder×3 + ONE elbow + wrist_roll/pitch/yaw
(X1: elbow_pitch + elbow_yaw + wrist_pitch + wrist_roll). Same 29-DoF count.

## 2. Position / velocity / effort limits — per joint

Position limits (`lower`/`upper`): **identical in both files for all 29 joints**
(including the asymmetric `hip_roll` −0.5236/2.9671 (left) and −2.9671/0.5236
(right), and the −0.087267 knee lower bound).

Effort/velocity — **exactly one difference**:

| joint | GMR effort / vel | UMR effort / vel | verdict |
|---|---|---|---|
| `left/right_hip_roll_joint` | **139 Nm / 20 rad/s** | 88 Nm / 32 rad/s | **GMR correct (rev 1.0)** |
| all other 27 joints | identical | identical | — |

Cross-check (2026-11-17, fetched this session): official
`unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf` moved hip_roll to the
**7520-22 motor** — `effort="139" velocity="20"` — confirmed verbatim by
IsaacLab issue #8530 ("[Bug Report] G1_29DOF_CFG hip roll uses pre rev 1.0 limits
(88 Nm, 32 rad/s) instead of 139 Nm, 20 rad/s"), which also cites mujoco-menagerie
`g1.xml` and mjlab `g1_constants.py` agreeing on 139/20. The UMR copy keeps the
pre-rev-1.0 value. Official `motor_info.md` (new-generation table) likewise puts
hip pitch/roll + knee on the strongest motor class (7519/7520-22.5: 121 Nm
continuous / 139 peak-class).

Full limit table (GMR, = official rev_1_0):

| joint group | joints | effort (Nm) | velocity (rad/s) |
|---|---|---|---|
| hip_pitch | L/R | 88 | 32 |
| hip_roll | L/R | **139** | **20** |
| hip_yaw | L/R | 88 | 32 |
| knee | L/R | 139 | 20 |
| ankle_pitch / ankle_roll | L/R | 50 | 37 |
| waist_yaw | 1 | 88 | 32 |
| waist_roll / waist_pitch | 2 | 50 | 37 |
| shoulder_pitch/roll/yaw, elbow, wrist_roll | L/R (5×2) | 25 | 37 |
| wrist_pitch / wrist_yaw | L/R (2×2) | 5 | 22 |

## 3. Kinematics (joint origins / axes)

Same parent→child tree and axes. Four origin differences, all in the torso chain
(GMR = rev_1_0 geometry, torso ~1 cm taller):

| joint | GMR origin z | UMR origin z |
|---|---|---|
| waist_roll_joint | 0.044 | 0.035 |
| waist_pitch_joint | 0 | 0.019 |
| left/right_shoulder_pitch_joint | 0.24778 | 0.23778 |

(x/y/rpy identical; all other 25 joint origins byte-identical.)

## 4. Inertias

All common links have **identical** mass/COM/inertia tensors **except** the three
waist-chain links (rev_1_0 re-measured):

| link | GMR mass | UMR mass | GMR COM z | UMR COM z |
|---|---|---|---|---|
| waist_yaw_link | 0.214 | 0.244 | 0.018034 | 0.018769 |
| waist_roll_link | 0.086 | 0.047 | 0 | 0.010111 |
| torso_link | 6.780 | 8.562 | 0.15082 | 0.153719 |

GMR pelvis mass 3.813 kg matches the official rev_1_0 blob verbatim. GMR meshes
`waist_*_rev_1_0.STL` / `torso_link_rev_1_0.STL` confirm the revision lineage.
The 1.8 kg difference vs UMR is mostly the UMR torso (8.562 vs 6.78).

## 5. Collision geometry (the decisive difference)

**GMR — primitives only** (the "custom collision" rework), 13 shapes:

| link | shapes |
|---|---|
| torso_link | 1 × cylinder r=0.08 |
| L/R hip_yaw_link, knee_link | cylinder r=0.03 each |
| L/R shoulder_roll_link, shoulder_yaw_link, elbow_link | cylinder r=0.03 each |
| L/R ankle_roll_link (foot) | 2 × cylinder r=0.02, len 0.15, at x=+0.05, z=−0.02 (toe+heel sole rolls) |

No collision on pelvis / hip_pitch / hip_roll / ankle_pitch / shoulder_pitch /
wrists / hands.

**UMR — mesh collisions everywhere** (33 mesh `.obj` collision shapes incl. the
non-convex visual shells of pelvis/torso/head/hands) **plus sphere packs**
(12 × ankle spheres r=0.005, 6 × finger spheres r=0.001, 2 × r=0.01 ankle
intermediate spheres).

Consequences for IsaacLab `UrdfFileCfg` conversion: UMR mesh collisions convert to
per-link convex hulls — slower to cook/simulate, and the hollow visual shells give
over-large, overly "sticky" contact surfaces (hands, head, torso); the 24
decorative micro-spheres add contact pairs without physics value. GMR's capsules
(after `replace_cylinders_with_capsules=True`) and flat sole rolls are exactly the
geometry an RL locomotion stack wants: cheap, convex, and a well-defined support
polygon — same philosophy as the X1 asset in this repo.

## 6. Extra frames

- GMR fixed frames: `pelvis_contour_link`, `left/right_toe_link`, `head_link`,
  `head_mocap`, `imu_in_torso`, `left/right_rubber_hand`.
- UMR fixed frames: all of the above except toe/mocap, plus `logo_link`,
  `waist_support_link`, `imu_link`, `d435_link`, `mid360_link`, 12 ankle-roll
  spheres, `left/right_{thumb,pinky}_link`, hands named `*_rubber_hand_link`.

For us: IMU body = `imu_in_torso` (GMR) — a clean fixed frame in the torso; UMR's
`imu_link` is equally usable but the sensor naming (`d435`/`mid360`) implies the
older developer kit. Feet/toe frames (`left/right_toe_link`) exist in GMR only —
useful later for foot-end reporting.

## 7. Meshes

- GMR: `meshes/*.STL`, **visual-only** (34 refs) — collisions are primitives, so
  meshes never affect physics; asset ships as `urdf + meshes/` (upstream
  `g1_description` layout).
- UMR: `assets/*.obj` used for **both** visual and collision.

## 8. Decision: GMR as base

**Chosen base: GMR `g1_custom_collision_29dof.urdf`.**

1. **Correct hardware revision** — hip_roll 139 Nm/20 rad/s matches official
   rev_1_0 (4 independent confirmations: official URDF commit, IsaacLab #8530,
   mujoco-menagerie, mjlab). UMR under-rates hip_roll torque by 37% and over-rates
   its speed — wrong for both training limits and sim2sim export.
2. **Collision geometry built for RL** — convex primitives/capsule feet vs UMR's
   mesh hulls + 24 micro-spheres; matches how the X1 asset is used in this repo
   and keeps the PhysX cook + contact-pair count small at 4096 envs.
3. **Cleaner mass model** — rev_1_0 inertias, rubber hands already merged (0.17 kg
   each), no sensor-mast mass ambiguity.
4. **Same joint names/order as UMR** — nothing is lost for motion retargeting
   pipelines that were written against the UMR names.

Risks accepted with GMR: (a) pelvis and hip_pitch/roll links have **no collision**
(self-collision and fall-on-hip contacts involving those links are invisible —
acceptable for flat-terrain velocity tracking, revisit for fall-recovery work);
(b) 1 cm taller torso vs UMR — irrelevant since all our data is generated against
the GMR URDF itself.

Repo placement convention (mirrors `robots/x1/urdf/f1.urdf`): URDF at
`data/robots/g1/urdf/g1_custom_collision_29dof.urdf`, meshes at
`data/robots/g1/urdf/meshes/*.STL` (URDF references `meshes/...` relative to its
own directory — upstream layout, zero URDF edits).
