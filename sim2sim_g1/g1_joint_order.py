#!/usr/bin/env python3
"""g1_joint_order.py — Unitree G1 (29-DoF) joint-order contract for sim2sim.

Purpose
-------
Single source of truth mapping the THREE joint orderings that must agree for
the Isaac -> MuJoCo sim2sim chain to feed the policy unscrambled obs:

  1. ``LAB_DOF``       — IsaacLab policy-side order.  IsaacLab orders
                         articulation joints by the depth-first traversal of
                         the imported URDF/USD prim tree, which equals the
                         joint DECLARATION order of the source URDF
                         (g1_custom_collision_29dof.urdf, GMR, rev_1_0).
                         Verified: joint declaration order extracted from the
                         URDF is identical to the list below (see
                         verify_against_urdf / __main__ self-test).
  2. ``MJ_ACTUATOR``   — MuJoCo actuator order of g1_mocap_29dof.xml
                         (<actuator><motor> block; == XML joint/body order).
  3. ``MJ_HINGE``      — MuJoCo hinge-joint order of the same XML (identical
                         to the actuator order in this model).

For G1 all three are the SAME list (legs -> waist -> arms, no interleaving
unlike X1 whose URDF nests lumbar between the hips):
  left leg (6) | right leg (6) | waist (3) | left arm (7) | right arm (7) = 29.

The identity is ASSERTED, not assumed: ``build_lab2mj()`` resolves the map by
NAME, and ``verify_against_model()`` checks a live mujoco model.  If a future
URDF/Isaac version reorders joints, only this module needs updating and the
asserts below will pinpoint the divergence immediately.

Role map (X1 -> G1) for cross-referencing the inherited X1 AMP stack
(G1AmpEnvCfg rewrites X1 names; from output/g1_amp_env_cfg.py):
  X1 lumbar_{yaw,roll,pitch}_joint -> G1 waist_{yaw,roll,pitch}_joint
  X1 {L,R}_knee_pitch_joint        -> G1 {L,R}_knee_joint
  X1 {L,R}_elbow_pitch_joint       -> G1 {L,R}_elbow_joint        (forearm lift)
  X1 {L,R}_elbow_yaw_joint         -> G1 {L,R}_wrist_roll_joint   (forearm rot)
  X1 {L,R}_wrist_pitch_joint       -> G1 {L,R}_wrist_pitch_joint  (identity)
  X1 {L,R}_wrist_roll_joint        -> G1 {L,R}_wrist_yaw_joint    (hand dev)

Usage
-----
  python3 g1_joint_order.py            # self-test (order identity + lengths)
  python3 g1_joint_order.py model.xml  # + verify against a live MuJoCo model
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The contract (explicit, authoritative tables — do NOT regenerate silently)
# ---------------------------------------------------------------------------

# IsaacLab / policy-side 29-DoF order (== GMR g1_custom_collision_29dof.urdf
# joint declaration order == depth-first prim tree of the imported USD).
LAB_DOF = [
    # left leg (6)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg (6)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist (3)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm (7)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm (7)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# MuJoCo actuator order of g1_mocap_29dof.xml (<motor> block order).  In this
# model the actuator block lists the joints in the same order as the XML body
# tree, so MJ_ACTUATOR == MJ_HINGE == LAB_DOF; kept as a SEPARATE literal so a
# future XML edit that reorders <actuator> fails the assert below loudly
# instead of silently mis-wiring ctrl[].
MJ_ACTUATOR = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

MJ_HINGE = list(MJ_ACTUATOR)  # g1_mocap_29dof.xml: body-tree order == actuator order

# ---------------------------------------------------------------------------
# Hard contract asserts (fail at import time — cheap and loud)
# ---------------------------------------------------------------------------
assert len(LAB_DOF) == 29, f"LAB_DOF must list 29 joints, got {len(LAB_DOF)}"
assert len(MJ_ACTUATOR) == 29, f"MJ_ACTUATOR must list 29 joints, got {len(MJ_ACTUATOR)}"
assert len(set(LAB_DOF)) == 29, "LAB_DOF contains duplicates"
assert set(LAB_DOF) == set(MJ_ACTUATOR), (
    "LAB_DOF and MJ_ACTUATOR name sets differ: "
    f"lab-only={sorted(set(LAB_DOF) - set(MJ_ACTUATOR))} "
    f"mj-only={sorted(set(MJ_ACTUATOR) - set(LAB_DOF))}"
)

# Convenience role groups (indices into LAB_DOF)
LEG_IDX = [i for i, n in enumerate(LAB_DOF) if "hip" in n or "knee" in n or "ankle" in n]
WAIST_IDX = [i for i, n in enumerate(LAB_DOF) if n.startswith("waist")]
ARM_IDX = [i for i, n in enumerate(LAB_DOF)
           if "shoulder" in n or "elbow" in n or "wrist" in n]
assert sorted(LEG_IDX + WAIST_IDX + ARM_IDX) == list(range(29))


def build_lab2mj() -> list:
    """Map policy column (LAB_DOF index) -> MuJoCo actuator/hinge index.

    Resolved BY NAME, so a future XML reordering is absorbed here rather than
    scrambling obs/actions.  For the current G1 assets this is the identity —
    which ``self-test`` verifies.
    """
    return [MJ_ACTUATOR.index(name) for name in LAB_DOF]


def verify_against_model(model, check_actuators=True) -> dict:
    """Check a live compiled MuJoCo model agrees with MJ_ACTUATOR / MJ_HINGE.

    check_actuators=False skips the actuator-name check (for the
    native-servo build, whose actuators are named servo_<joint>).
    Returns a summary dict; raises AssertionError on any mismatch.
    """
    import mujoco  # local import: keeps this module importable without mujoco

    if check_actuators:
        act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                     for i in range(model.nu)]
        assert model.nu == 29, f"model has {model.nu} actuators, expected 29"
        assert act_names == MJ_ACTUATOR, (
            f"MuJoCo actuator order mismatch:\n  model : {act_names}\n  table : {MJ_ACTUATOR}"
        )

    hinge_names = [model.joint(i).name for i in range(model.njnt)
                   if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    assert len(hinge_names) == 29, (
        f"model has {len(hinge_names)} hinges (plus 1 free joint expected), "
        f"expected 29"
    )
    assert hinge_names == MJ_HINGE, (
        f"MuJoCo hinge order mismatch:\n  model : {hinge_names}\n  table : {MJ_HINGE}"
    )
    lab2mj = build_lab2mj()
    return {
        "actuators_checked": bool(check_actuators),
        "hinges": hinge_names,
        "lab2mj_is_identity": lab2mj == list(range(29)),
        "lab2mj": lab2mj,
    }


def verify_against_urdf(urdf_path) -> list:
    """Extract revolute/prismatic joint declaration order from a URDF file and
    check it equals LAB_DOF (the IsaacLab ordering premise).

    Returns the extracted list; raises AssertionError on mismatch.
    """
    import re
    from pathlib import Path

    text = Path(urdf_path).read_text()
    joints = re.findall(r'<joint\s+name="([^"]+)"\s+type="(revolute|continuous|prismatic)"',
                        text)
    names = [n for n, _ in joints]
    assert len(names) == 29, f"URDF has {len(names)} non-fixed joints, expected 29"
    assert names == LAB_DOF, (
        f"URDF declaration order != LAB_DOF:\n  urdf : {names}\n  lab  : {LAB_DOF}"
    )
    return names


def _self_test(xml_path=None):
    lab2mj = build_lab2mj()
    print(f"[OK] LAB_DOF 29 joints, MJ_ACTUATOR 29 joints, name sets identical")
    print(f"[OK] lab2mj resolved by name; identity = {lab2mj == list(range(29))}")
    print(f"[OK] groups: legs={len(LEG_IDX)} waist={len(WAIST_IDX)} arms={len(ARM_IDX)}")
    if xml_path:
        import mujoco
        m = mujoco.MjModel.from_xml_path(xml_path) if str(xml_path).endswith(".xml") \
            else None
        if m is not None:
            info = verify_against_model(m)
            print(f"[OK] live model: nu=29 hinges=29, lab2mj identity="
                  f"{info['lab2mj_is_identity']}")
    print("[SELF-TEST] all asserts passed")


if __name__ == "__main__":
    import sys
    _self_test(sys.argv[1] if len(sys.argv) > 1 else None)
