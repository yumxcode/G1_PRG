import os

from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.g1 import G1_CFG

from .x1_amp_env_cfg import X1AmpEnvCfg

# G1-AMP: Unitree G1 29-DoF (rev_1_0) velocity task. Thin inheritance of the
# ENTIRE X1 AMP stack (rewards/events/terminations, 50 Hz control, clock
# conditioning, disc obs) — same pattern as F1WAmpEnvCfg. Differences from X1:
#   * robot asset: G1_CFG (GMR g1_custom_collision_29dof.urdf; see
#     urdf_diff_report.md — official rev_1_0 limits, primitive collisions)
#   * link/joint NAMES: G1 naming differs from X1 (pelvis vs base_link,
#     waist_* vs lumbar_*, knee vs knee_pitch, one elbow + wrist_roll/pitch/yaw
#     vs elbow_pitch/elbow_yaw + wrist_pitch/wrist_roll). Every body_names /
#     joint_names reference anywhere in the inherited stack is retargeted to
#     the G1 name with the SAME ROLE (feet stay feet, forearm-rotation stays
#     forearm-rotation, ...). The retarget pass runs AFTER super().__post_init__
#     so X1AmpEnvCfg's env-var branches are NOT copied and keep their meaning.
#   * commands: G1 starting band lin_vel_x 0.2-1.5 m/s (G1_CMD_LO/G1_CMD_HI).
#   * demo dataset: G1 motions pending (C1 is task-first, zero-AMASS) — dir is
#     a placeholder under G1_MOTION_DIR; empty weights disable AMP sampling
#     until a G1 library exists.
#
# Joint-name map rationale (chain-position consistent):
#   X1 elbow_pitch (forearm lift)      -> G1 elbow
#   X1 elbow_yaw   (forearm rotation)  -> G1 wrist_roll  (G1's forearm rotation)
#   X1 wrist_pitch                     -> G1 wrist_pitch (identity)
#   X1 wrist_roll  (hand deviation)    -> G1 wrist_yaw   (G1's hand deviation)

# X1 body name -> G1 body name (exact string match; unlisted names identical).
# NOTE: left/right_wrist_roll_link exist in BOTH robots with DIFFERENT roles —
# X1's wrist_roll is the distal hand-deviation joint, G1's wrist_roll is the
# forearm-rotation joint. The map is ROLE-based (chain-position consistent),
# so X1 wrist_roll_link -> G1 wrist_yaw_link (G1's distal joint).
X1_TO_G1_BODY_NAMES = {
    "base_link": "pelvis",
    "lumbar_yaw_link": "waist_yaw_link",
    "lumbar_roll_link": "waist_roll_link",
    "lumbar_pitch_link": "torso_link",
    "left_knee_pitch_link": "left_knee_link",
    "right_knee_pitch_link": "right_knee_link",
    "left_elbow_pitch_link": "left_elbow_link",
    "right_elbow_pitch_link": "right_elbow_link",
    "left_elbow_yaw_link": "left_wrist_roll_link",
    "right_elbow_yaw_link": "right_wrist_roll_link",
    "left_wrist_roll_link": "left_wrist_yaw_link",
    "right_wrist_roll_link": "right_wrist_yaw_link",
}

# X1 joint-name pattern -> G1 joint-name pattern (substitution rules; FIRST
# matching rule wins so elbow_yaw->wrist_roll is not re-rewritten to wrist_yaw)
X1_TO_G1_JOINT_RULES = [
    ("lumbar_yaw_joint", "waist_yaw_joint"),
    ("lumbar_roll_joint", "waist_roll_joint"),
    ("lumbar_pitch_joint", "waist_pitch_joint"),
    ("knee_pitch_joint", "knee_joint"),
    ("elbow_pitch_joint", "elbow_joint"),
    ("elbow_yaw_joint", "wrist_roll_joint"),
    ("wrist_roll_joint", "wrist_yaw_joint"),
]

# 10 discriminator key bodies for G1 (role-identical to X1 KEY_BODY_NAMES:
# knees, ankles, elbow-distal, upper-arm proximal, wrist-distal). Must match the
# future G1 retarget config's lab_key_body_names ordering.
G1_KEY_BODY_NAMES = [
    "left_knee_link",
    "right_knee_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
]


def _retarget_joint_pattern(pattern, rules=X1_TO_G1_JOINT_RULES):
    """Rewrite one X1 joint-name pattern (exact or '.*_'-prefixed regex) to G1."""
    out = pattern
    for old, new in rules:
        if old in out:
            out = out.replace(old, new)
            break  # first matching rule wins (see comment above)
    return out


def _retarget_scene_entity_names(cfg, body_map=X1_TO_G1_BODY_NAMES, _seen=None):
    """Walk an env cfg after __post_init__ and rewrite every body/joint name.

    Covers self.observations.* (policy/critic/disc groups), self.rewards,
    self.events, self.terminations — every SceneEntityCfg's ``body_names`` /
    ``joint_names`` and every ``joint_names_expr`` list encountered anywhere in
    nested attributes, lists and RewardTerm/ObsTerm ``params`` dicts. Purely
    mechanical: no X1 env-var logic is read or copied. Idempotent (G1 names
    contain no X1-only tokens, so a second pass changes nothing).
    """
    if _seen is None:
        _seen = set()
    if id(cfg) in _seen:
        return
    _seen.add(id(cfg))

    if isinstance(cfg, (list, tuple)):
        for item in cfg:
            _retarget_scene_entity_names(item, body_map, _seen)
        return
    if isinstance(cfg, dict):
        for item in cfg.values():
            _retarget_scene_entity_names(item, body_map, _seen)
        return
    if not hasattr(cfg, "__dict__"):
        return

    changed_bodies, changed_joints = [], []
    for attr, val in vars(cfg).items():
        if attr == "body_names":
            if isinstance(val, str):
                if val in body_map:
                    setattr(cfg, attr, body_map[val])
                    changed_bodies.append(f"{val}->{body_map[val]}")
            elif isinstance(val, list):
                for i, name in enumerate(val):
                    if isinstance(name, str) and name in body_map:
                        val[i] = body_map[name]
                        changed_bodies.append(f"{name}->{val[i]}")
        elif attr in ("joint_names", "joint_names_expr"):
            if isinstance(val, str):
                new = _retarget_joint_pattern(val)
                if new != val:
                    setattr(cfg, attr, new)
                    changed_joints.append(f"{val}->{new}")
            elif isinstance(val, list):
                for i, name in enumerate(val):
                    if isinstance(name, str):
                        new = _retarget_joint_pattern(name)
                        if new != name:
                            val[i] = new
                            changed_joints.append(f"{name}->{new}")
        else:
            _retarget_scene_entity_names(val, body_map, _seen)
    if changed_bodies or changed_joints:
        print(f"[G1-RETARGET] {type(cfg).__name__}: bodies={changed_bodies} joints={changed_joints}")


@configclass
class G1AmpEnvCfg(X1AmpEnvCfg):
    # C1 PRG adaptive gait clock (G1AmpEnv reads these; set from env vars
    # below). B0 baseline = both False (pure v71 clock via X1_GAIT_CLOCK=1).
    prg_enabled: bool = False
    prg_freq_mod: bool = False

    def __post_init__(self):
        super().__post_init__()

        # --- C1 PRG wiring (single source: G1_PRG / G1_PRG_FREQ_MOD) ------
        import os as _os
        from .mdp import observations as _obs_mdp
        from .mdp import rewards as _rew_mdp
        _prg = _os.environ.get("G1_PRG", "0") == "1"
        _fmod = _os.environ.get("G1_PRG_FREQ_MOD", "0") == "1" and _prg
        self.prg_enabled = _prg
        self.prg_freq_mod = _fmod
        if _prg:
            # 1) clock obs: PRG state clock (falls back to v71 inside the
            #    func when the env has no PRG state, so the layout is stable)
            _clock_params = dict(getattr(
                getattr(self.observations.policy, "gait_clock", None), "params",
                {}) or {})
            if not _clock_params:
                # X1_GAIT_CLOCK was off — create the obs terms ourselves
                from isaaclab.managers import ObservationTermCfg as ObsTermV71
                self.observations.policy.gait_clock = ObsTermV71(
                    func=_obs_mdp.prg_clock_obs)
                self.observations.critic.gait_clock = ObsTermV71(
                    func=_obs_mdp.prg_clock_obs)
            else:
                self.observations.policy.gait_clock.func = _obs_mdp.prg_clock_obs
                self.observations.critic.gait_clock.func = _obs_mdp.prg_clock_obs
            # 2) clock rewards: PRG twins (identical kernels, PRG phase);
            #    ensure nonzero weights even when X1_GAIT_CLOCK=0 (PRG is
            #    its own switch — single-variable chain vs B0)
            if getattr(self.rewards, "foot_clock", None) is not None:
                self.rewards.foot_clock.func = _rew_mdp.prg_foot_clock_height
                if self.rewards.foot_clock.weight == 0:
                    self.rewards.foot_clock.weight = float(
                        _os.environ.get("X1_FOOT_CLOCK_W", "1.0"))
            if getattr(self.rewards, "contact_clock", None) is not None:
                self.rewards.contact_clock.func = _rew_mdp.prg_contact_clock_timing
                if self.rewards.contact_clock.weight == 0:
                    self.rewards.contact_clock.weight = -float(
                        _os.environ.get("X1_CONTACT_CLOCK_W", "0.5"))
        # unconditional runtime-effectiveness audit (RECORD_BOOK lesson)
        print(
            f"[G1-PRG-CFG] prg_enabled={int(self.prg_enabled)} "
            f"prg_freq_mod={int(self.prg_freq_mod)} "
            f"PRG_RESET_MODE={_os.environ.get('PRG_RESET_MODE','hard')} "
            f"obs_func={getattr(getattr(self.observations.policy, 'gait_clock', None), 'func', None)} "
            f"foot_clock_func={getattr(getattr(self.rewards, 'foot_clock', None), 'func', None)}"
        )

        # G1 robot (link/joint names are native G1; body-name differences in
        # the inherited stack are retargeted below)
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Body/joint name retarget across the whole inherited stack (obs,
        # rewards, events, terminations). Runs after X1AmpEnvCfg.__post_init__
        # so it also fixes whatever the X1 module consumed (KEY_BODY_NAMES etc.)
        for manager in (
            getattr(self, "observations", None),
            getattr(self, "rewards", None),
            getattr(self, "events", None),
            getattr(self, "terminations", None),
        ):
            if manager is not None:
                _retarget_scene_entity_names(manager)

        # Discriminator key bodies: 10 role-identical G1 bodies (G1 demo
        # library pending; kept consistent for the future retarget schema).
        _mds = getattr(self.motion_data, "motion_dataset", None)
        for _attr in ("key_body_names",):
            if _mds is not None and hasattr(_mds, _attr):
                setattr(_mds, _attr, list(G1_KEY_BODY_NAMES))
                print(f"[G1-RETARGET] motion_data.motion_dataset.{_attr} = {G1_KEY_BODY_NAMES}")

        # G1 demo dataset: a placeholder standing clip satisfies the
        # MotionDataManager (it asserts dir+files and reads dof_pos[0] — an
        # empty weights dict crashes at num_dofs). The clip is INERT for the
        # policy because G1 runs task-first with X1_STYLE_SCALE=0 (style
        # reward fully off, beyondmimic-phase-A precedent); swap in a real
        # retargeted G1 library only if a future variant wants AMP.
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions",
            os.environ.get("G1_MOTION_DIR", "g1_lab_v1")
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "standing_placeholder": 1.0,
        }

        # Commands: G1 starting band (walk spectrum first). G1_CMD_LO/G1_CMD_HI
        # widen later; X1 command env-vars are untouched.
        _lo = float(os.environ.get("G1_CMD_LO", "0.2"))
        _hi = float(os.environ.get("G1_CMD_HI", "1.5"))
        self.commands.base_velocity.ranges.lin_vel_x = (_lo, _hi)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        if self.__class__.__name__ == "G1AmpEnvCfg":
            self.disable_zero_weight_rewards()
            print(
                f"[G1-AMP] G1AmpEnvCfg ready: cmd_x=({_lo},{_hi}) "
                f"motion_dir={os.environ.get('G1_MOTION_DIR', 'g1_lab_v1')}"
            )


@configclass
class G1AmpEnvCfg_PLAY(G1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # fixed-command rollout: G1_PLAY_VEL (default 0.8 m/s mid-band walk)
        _v = float(os.environ.get("G1_PLAY_VEL", "0.8"))
        self.commands.base_velocity.ranges.lin_vel_x = (_v, _v)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        if self.__class__.__name__ == "G1AmpEnvCfg_PLAY":
            self.disable_zero_weight_rewards()
