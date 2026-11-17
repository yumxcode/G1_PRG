import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robolab.assets import ISAAC_DATA_DIR

# Unitree G1 humanoid: 29 DoF, hardware revision 1.0
# - Legs: hip_pitch/roll/yaw + knee + ankle_pitch/roll (6 x 2 = 12)
# - Waist: waist_yaw/roll/pitch (3)
# - Arms: shoulder_pitch/roll/yaw + elbow + wrist_roll/pitch/yaw (7 x 2 = 14)
# Total: 12 + 3 + 14 = 29
#
# URDF: GMR "g1_custom_collision_29dof.urdf" (robot name g1_29dof_rev_1_0,
# primitive collisions, sole cylinders as feet). See output/urdf_diff_report.md
# for the GMR-vs-UMR comparison and the rev_1_0 evidence (hip_roll 139 Nm/20 rad/s
# confirmed against official unitree_ros g1_29dof_rev_1_0.urdf, IsaacLab issue
# #8530, mujoco-menagerie g1.xml, mjlab g1_constants.py; all fetched 2026-11-17).
# Repo convention (mirrors robots/x1/urdf/f1.urdf):
#   URDF   data/robots/g1/urdf/g1_custom_collision_29dof.urdf
#   meshes data/robots/g1/urdf/meshes/*.STL   (URDF refs "meshes/..." relative
#          to the URDF dir — upstream g1_description layout, no URDF edits)
#
# ---------------------------------------------------------------------------
# kp/kd PROVENANCE (all web sources fetched & cross-checked 2026-11-17)
# ---------------------------------------------------------------------------
# [HIGH] Legs — unitreerobotics/unitree_rl_gym
#        legged_gym/envs/g1/g1_config.py (official, verbatim):
#        stiffness {hip_yaw 100, hip_roll 100, hip_pitch 100, knee 150,
#                   ankle 40}  [Nm/rad]
#        damping   {hip_yaw 2, hip_roll 2, hip_pitch 2, knee 4, ankle 2}
#        [Nms/rad]. Same numbers appear in IsaacLab's G1 leg group
#        (ankle 40 / 2 decoded verbatim from isaaclab_assets/robots/unitree.py
#        main @ a8544937).
# [MED-HIGH] Arms — IsaacLab G1 29-DoF locomotion pattern
#        (isaaclab_assets/robots/unitree.py): shoulder/elbow 40 / 8,
#        wrist 20 / 4. Cross-check verbatim from the same file (manipulation
#        variant): shoulder_yaw 40/8, elbow 40/8, wrist_* 20/4 identical;
#        shoulder_pitch/roll 100/15 there are manipulation-stiff and are NOT
#        used here (locomotion needs free-swinging arms; X1 in this repo uses
#        40/4 on the same 20-27 Nm torque class).
# [MED] Waist — no public G1 locomotion gains found (unitree_rl_gym ships only
#        the 12-DoF leg config). Assigned by MOTOR CLASS identity from the URDF
#        effort/velocity limits: waist_yaw shares the 88 Nm/32 rad/s motor of
#        hip_yaw -> 100 / 2; waist_roll/pitch share the 50 Nm/37 rad/s motor of
#        the ankles -> 40 / 2. Flagged for first-run re-identification.
# [MED] armature — unitreerobotics/unitree_rl_lab
#        assets/robots/unitree_actuators.py motor constants (rotor + 2-stage
#        gear reflected inertia): 7520-22 (hip_roll, knee) 0.0251,
#        7520-14 (hip_pitch/yaw, waist_yaw) 0.0102, 5020-16 (ankles, waist
#        roll/pitch, shoulders, elbows, wrist_roll) 0.0036,
#        W4010-25 (wrist_pitch/yaw) 0.00425. Catalog estimate, not bench
#        identified — X1 lesson (x1_armature_pd.md): re-fit from walk logs
#        before any sim2real claim; sim2sim is unaffected.
# [HIGH] effort/velocity limits — GMR URDF == official g1_29dof_rev_1_0.urdf
#        (see header): hip_pitch 88/32, hip_roll 139/20, hip_yaw 88/32,
#        knee 139/20, ankles 50/37, waist_yaw 88/32, waist_roll/pitch 50/37,
#        shoulder*/elbow/wrist_roll 25/37, wrist_pitch/yaw 5/22.
# ---------------------------------------------------------------------------
# initial_state DERIVATION (standing, legs near-straight slightly bent, arms
# hanging with slightly bent elbows — G1 official demo posture family):
#   pose: hip_pitch -0.15, knee +0.30, ankle_pitch -0.15 (foot exactly flat:
#   -0.15 + 0.30 - 0.15 = 0), elbow +0.30, everything else 0.
#   URDF FK (GMR file, computed in tests/test_g1_static.py): ankle_roll origin
#   at pelvis-frame z = -0.7497; sole cylinders sit at local z = -0.02 with
#   radius 0.02 -> lowest sole point z = -0.7897 => pelvis height 0.790 m.
#   Cross-references: IsaacLab G1_29DOF init z = 0.794 (same pose family),
#   unitree_rl_gym init z = 0.80 (pose -0.1/0.3/-0.2, drop-in margin),
#   mjlab HOME keyframe z = 0.783675. We set 0.79 (FK exact contact + 0.3 mm).
# ---------------------------------------------------------------------------

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_DATA_DIR}/robots/g1/urdf/g1_custom_collision_29dof.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.79),  # pelvis height; FK-derived standing height 0.7897 (see header)
        joint_pos={
            # waist
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            # left arm (hanging, elbow slightly bent)
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.3,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            # right arm
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.3,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            # left leg (near-straight, slight bend; foot flat)
            "left_hip_pitch_joint": -0.15,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,
            # right leg
            "right_hip_pitch_joint": -0.15,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        # Waist + hips + knees. Motor classes per official rev_1_0 limits:
        #   7520-14 (88 Nm / 32 rad/s): waist_yaw, hip_pitch, hip_yaw
        #   7520-22 (139 Nm / 20 rad/s): hip_roll, knee   <- rev_1_0 fix
        "waist_legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_hip_roll_joint": 139.0,   # rev_1_0 (7520-22); pre-rev = 88/32
                ".*_hip_yaw_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_hip_roll_joint": 20.0,   # rev_1_0
                ".*_hip_yaw_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            # kp/kd: unitree_rl_gym legs [HIGH] + waist-by-motor-class [MED]
            # v2 damping fix (2026-11-17 smoke postmortem): unitree_rl_gym kd
            # (2/4) targets IsaacGym direct-torque sim; on this IsaacSim
            # DelayedPD stack the X1 v68 recipe needs ζ≈0.7 → kd 8 on all
            # leg/waist joints (X1 120→65 kp / kd→8 was the same lesson).
            stiffness={
                "waist_yaw_joint": 100.0,    # motor class of hip_yaw
                "waist_roll_joint": 40.0,    # motor class of ankles
                "waist_pitch_joint": 40.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_yaw_joint": 100.0,
                ".*_knee_joint": 150.0,
            },
            damping={
                "waist_yaw_joint": 8.0,
                "waist_roll_joint": 8.0,
                "waist_pitch_joint": 8.0,
                ".*_hip_pitch_joint": 8.0,
                ".*_hip_roll_joint": 8.0,
                ".*_hip_yaw_joint": 8.0,
                ".*_knee_joint": 8.0,
            },
            # armature: unitree_rl_lab motor-catalog reflected inertia [MED]
            armature={
                "waist_yaw_joint": 0.0102,   # 7520-14
                "waist_roll_joint": 0.0036,  # 5020-16
                "waist_pitch_joint": 0.0036,
                ".*_hip_pitch_joint": 0.0102,  # 7520-14
                ".*_hip_roll_joint": 0.0251,   # 7520-22
                ".*_hip_yaw_joint": 0.0102,
                ".*_knee_joint": 0.0251,
            },
            min_delay=0,
            max_delay=2,
        ),
        # Ankles: 5020-16 motor (50 Nm / 37 rad/s)
        "ankles": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=40.0,
            damping={  # X1 ankle split: pitch ζ0.73 kd4, roll ζ0.93 kd2.5
                ".*_ankle_pitch_joint": 4.0,
                ".*_ankle_roll_joint": 2.5,
            },
            armature=0.0036,  # 5020-16
            min_delay=0,
            max_delay=2,
        ),
        # Shoulders + elbows + wrist_roll: 5020-16 motor (25 Nm / 37 rad/s)
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            # kp/kd: IsaacLab G1 29dof locomotion pattern [MED-HIGH]
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_joint": 40.0,
                ".*_wrist_roll_joint": 20.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 8.0,
                ".*_shoulder_roll_joint": 8.0,
                ".*_shoulder_yaw_joint": 8.0,
                ".*_elbow_joint": 8.0,
                ".*_wrist_roll_joint": 4.0,
            },
            armature=0.0036,  # 5020-16
            min_delay=0,
            max_delay=2,
        ),
        # Wrist pitch/yaw: W4010-25 motor (5 Nm / 22 rad/s) — weakest pair
        "wrists": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
            effort_limit_sim=5.0,
            velocity_limit_sim=22.0,
            stiffness=20.0,
            damping=4.0,
            armature=0.00425,  # W4010-25
            min_delay=0,
            max_delay=2,
        ),
    },
)


G1_LINKS = [
    "pelvis",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
]
