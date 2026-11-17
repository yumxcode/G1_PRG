"""G1AmpEnv — AmpEnv with the C1 PRG adaptive gait clock wired in.

The PRG clock state (mdp/prg_clock.py::PRGState) is updated ONCE per
control step, AFTER the physics loop and BEFORE termination/reward
computation, so reward terms and observations see the post-contact phase.

step() is a verbatim copy of AmpEnv.step (repo commit 8fcc4da) with TWO
insertions, both marked `# [PRG]`:
  1. delta_f action-head stripping (C1b, G1_PRG_FREQ_MOD=1): the policy
     action is (N, 29+1); the last column is the frequency modulation
     delta_f in [-1, 1] and is peeled off BEFORE the LPF/delay pipeline
     so the joint-action path stays byte-identical to B0.
  2. self._prg_update() right after animation_manager.update(dt).

_reset_idx() additionally re-anchors the PRG state for the reset envs.

Toggle (read once from env cfg fields, set by G1AmpEnvCfg.__post_init__
from G1_PRG / G1_PRG_FREQ_MOD env vars):
  cfg.prg_enabled  — PRG clock on/off (B0 baseline: off, pure v71 clock)
  cfg.prg_freq_mod — delta_f action head on/off (C1a reset-only: off)

Effectiveness audit (RECORD_BOOK lesson): the [PRG] print below fires
UNCONDITIONALLY at first update — enabled flag, reset mode, alpha, foot
bodies, first touchdown stats — so "the change never took effect" is
directly readable from the task log.
"""
from __future__ import annotations

import os

import torch
from collections.abc import Sequence

from isaaclab.envs import VecEnvStepReturn

from .amp_env import AmpEnv
from .g1_amp_env_cfg import G1AmpEnvCfg


class G1AmpEnv(AmpEnv):

    cfg: G1AmpEnvCfg

    def __init__(self, cfg: G1AmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self.prg_state = None          # lazy: built on first _prg_update
        self._prg_foot_ids = None      # lazy: contact sensor body ids
        self._pending_delta_f = None
        self._prg_audited = False

    # -- PRG plumbing --------------------------------------------------------
    def _prg_active(self) -> bool:
        return bool(getattr(self.cfg, "prg_enabled", False))

    def _prg_freq_mod(self) -> bool:
        return bool(getattr(self.cfg, "prg_freq_mod", False))

    def _prg_foot_contact(self) -> torch.Tensor:
        """(N, 2) bool foot contact — same detector as contact_clock_timing:
        max over the force history window of |net force| > 1.0 N."""
        from isaaclab.managers import SceneEntityCfg
        sensor = self.scene["contact_forces"]
        if self._prg_foot_ids is None:
            cfg = SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            )
            cfg.resolve(self.scene)
            self._prg_foot_ids = cfg.body_ids
        forces = sensor.data.net_forces_w_history[:, :, self._prg_foot_ids, :]
        return forces.norm(dim=-1).max(dim=1)[0] > 1.0

    def _prg_update(self) -> None:
        if not self._prg_active():
            return
        if self.prg_state is None:
            from .mdp.prg_clock import PRGState
            reset_mode = os.environ.get("PRG_RESET_MODE", "hard")
            self.prg_state = PRGState(
                self.num_envs, device=self.device,
                dtype=torch.float32, reset_mode=reset_mode,
            )
            print(f"[PRG] state built: num_envs={self.num_envs} "
                  f"reset_mode={self.prg_state.reset_mode} "
                  f"freq_mod={self._prg_freq_mod()} device={self.device}")
        cmd = self.command_manager.get_command("base_velocity")
        v_cmd = torch.norm(cmd[:, :2], dim=1)
        contact = self._prg_foot_contact()
        delta_f = self._pending_delta_f
        self.prg_state.step(self.step_dt, v_cmd, contact, delta_f)
        if not self._prg_audited:
            self._prg_audited = True
            phi = self.prg_state.phi
            print(
                f"[PRG-AUDIT] enabled=1 reset={self.prg_state.reset_mode} "
                f"freq_mod={int(self._prg_freq_mod())} "
                f"alpha_max_env={os.environ.get('PRG_ALPHA_MAX','0.5')} "
                f"foot_ids={self._prg_foot_ids} "
                f"contact_any={bool(contact.any())} "
                f"phi[:2]={phi[:2].tolist()} v_cmd[:4]={v_cmd[:4].tolist()}"
            )

    # -- env lifecycle -------------------------------------------------------
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # [PRG] C1b: peel the delta_f action head BEFORE the LPF/delay
        # pipeline so the joint path stays byte-identical to B0.
        self._pending_delta_f = None
        if self._prg_active() and self._prg_freq_mod():
            self._pending_delta_f = torch.clamp(action[:, -1], -1.0, 1.0)
            action = action[:, :-1]

        # ---- verbatim AmpEnv.step (8fcc4da) + [PRG] insertion --------------
        self.action_manager.process_action(
            self._random_action_delay(self._action_lpf(action.to(self.device))))

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # post-step:
        self.animation_manager.update(dt=self.step_dt)
        self._prg_update()  # [PRG] adaptive clock steps AFTER physics, BEFORE rewards
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # [PRG] one-shot termination diagnostics (env 0..3) — which
            # termination, root z, and top-contact bodies at the instant of
            # death. PRINT_COUNT caps the spam; remove after diagnosis.
            if getattr(self, "_term_diag_count", 0) < 40:
                self._term_diag_count = getattr(self, "_term_diag_count", 0) + 1
                term_now = self.termination_manager.get_term("base_contact")
                z_now = self.scene["robot"].data.root_pos_w[:4, 2]
                print(f"[TERM-DIAG] n={len(reset_env_ids)} base_contact={term_now[:4].tolist()} "
                      f"z[:4]={z_now.tolist()}", flush=True)
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        if int(getattr(self.cfg, "obs_delay_steps", 0)) > 0:
            self.obs_buf["policy"] = self._obs_delay(
                self.obs_buf["policy"], self.reset_buf)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        # [PRG] re-anchor the adaptive clock for reset envs (left foot
        # touchdown at t=0, right antiphase — same anchor as v71).
        if self._prg_active() and self.prg_state is not None:
            self.prg_state.reset(env_ids)
