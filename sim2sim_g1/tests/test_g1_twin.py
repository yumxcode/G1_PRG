#!/usr/bin/env python3
"""test_g1_twin.py — pure-logic tests for the g1_sim2sim gait-clock twin.

No MuJoCo needed (numpy only).  Run:  python3 tests/test_g1_twin.py
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from g1_sim2sim import GaitClockTwin, v71_clock, wrap_circ  # noqa: E402
from prg_clock import PRGState, t_star  # noqa: E402


def _v71_reference(step, dt, v):
    T = float(t_star(v))
    pl = (step * dt / T) % 1.0
    pr = (pl + 0.5) % 1.0
    return np.array([np.sin(2 * np.pi * pl), np.sin(2 * np.pi * pr),
                     np.cos(2 * np.pi * pl), np.cos(2 * np.pi * pr)]), np.array([pl, pr])


def test_none_matches_v71_analytic():
    """--prg-mode none must equal the v71 pure formula bit-for-bit."""
    dt, v = 0.02, 1.0
    twin = GaitClockTwin("none", dt)
    cmd = np.array([v, 0.0, 0.0])
    for step in range(1500):
        obs, phi = twin.step(cmd, np.zeros(2, dtype=bool), None)
        ref_obs, ref_phi = _v71_reference(step, dt, v)
        assert np.allclose(obs, ref_obs, atol=1e-12), f"step {step}: {obs} vs {ref_obs}"
        # circular phase comparison (analytic vs integrated can differ by 1.0
        # near integer boundaries — prg_clock test convention)
        err = np.abs(wrap_circ(phi - ref_phi))
        assert np.all(err < 1e-9), f"step {step}: phi {phi} vs {ref_phi}"
    print("[PASS] none == v71 analytic (1500 steps, 1e-12 obs / 1e-9 circ phi)")


def test_none_constant_v_phase_advance():
    """v71: at v=1.0 (T*=1.10 s), 55 steps of 20 ms advance phi by 1.0 cycle."""
    dt = 0.02
    twin = GaitClockTwin("none", dt)
    T = float(t_star(1.0))
    n = int(round(T / dt))
    init_phi = np.array([0.0, 0.5])  # v71: phiL(0)=0, phiR(0)=0.5
    phi = init_phi
    # twin returns the t=(k-1)*dt clock at step k; one full cycle closes at
    # t = T* -> after n+1 steps the returned phase equals the initial one
    for _ in range(n + 1):
        _, phi = twin.step(np.array([1.0, 0, 0]), np.zeros(2, bool), None)
    err = np.abs(wrap_circ(phi - init_phi))
    assert np.all(err < 1e-9), f"{n} steps did not close the cycle: {phi} vs {init_phi}"
    print(f"[PASS] none: {n} steps @ v=1.0 close one cycle (T*={T:.3f}s)")


def test_hard_reset_anchors_phase():
    """hard: a touchdown inside the sync window re-anchors phi to ~0."""
    dt = 0.02
    twin = GaitClockTwin("hard", dt)
    prg = twin.prg
    assert prg.reset_mode == "hard"
    # free-run 0.3 s (phi drifts), then touchdown on left foot only
    contact = np.array([False, False])
    for _ in range(15):
        twin.step(np.array([1.0, 0, 0]), contact, None)
    phi_before = prg.phi[0].copy()
    assert 0.15 < phi_before[0] < 0.95, f"expected drifted phi, got {phi_before}"
    # drive phi near cycle end (inside [0.75,1) sync window), then touchdown
    while prg.phi[0, 0] < 0.80:
        twin.step(np.array([1.0, 0, 0]), contact, None)
    contact = np.array([True, False])
    twin.step(np.array([1.0, 0, 0]), contact, None)
    assert prg.phi[0, 0] < 0.1 or prg.phi[0, 0] > 0.9, \
        f"hard reset should anchor near 0, got {prg.phi[0]}"
    assert twin.reset_events >= 1
    print(f"[PASS] hard: touchdown re-anchored phiL {float('nan') if False else prg.phi[0,0]:.3f} "
          f"(resets={twin.reset_events})")


def test_soft_is_rate_limited():
    """soft: correction is clamped to SOFT_CLAMP*dt*beta per step (no jump)."""
    dt = 0.02
    twin = GaitClockTwin("soft", dt)
    prg = twin.prg
    contact = np.array([False, False])
    while prg.phi[0, 0] < 0.80:
        twin.step(np.array([1.0, 0, 0]), contact, None)
    phi_pre = prg.phi[0].copy()
    step_phi = twin.step(np.array([1.0, 0, 0]), np.array([True, False]), None)[1]
    jump = np.abs(wrap_circ(step_phi - phi_pre))
    # free-run advance ~ f*dt = 0.9*0.02 = 0.018; clamp = 0.15*0.02*2.0 = 0.006
    # total per-step change must stay small (no hard snap to 0)
    assert jump[0] < 0.1, f"soft reset jumped {jump[0]:.3f} in one step (not rate-limited)"
    # sustained contact pulls phi toward 0 over many steps
    for _ in range(50):
        twin.step(np.array([1.0, 0, 0]), np.array([True, True]), None)
    print(f"[PASS] soft: per-step jump {jump[0]:.4f} (rate-limited), "
          f"phi after 50 sustained-contact steps: {prg.phi[0]}")


def test_delta_f_modulates_frequency():
    """PRGState: delta_f=+1 raises f_eff by alpha*FMOD_SCALE*f_cmd."""
    prg = PRGState(num_envs=1, backend="numpy", reset_mode="none")
    f_cmd0 = 1.0 / float(t_star(1.0))
    # ramp alpha to max (t_ep >= ALPHA_RAMP_S)
    for _ in range(600):
        prg.step(0.02, np.array([1.0]), np.zeros((1, 2)), np.array([0.0]))
    assert prg.alpha[0] > 0.4, f"alpha should be ~ALPHA_MAX, got {prg.alpha[0]}"
    prg.step(0.02, np.array([1.0]), np.zeros((1, 2)), np.array([1.0]))
    expect = (1 - prg.alpha[0]) * f_cmd0 + prg.alpha[0] * f_cmd0 * (1 + 0.5)
    assert abs(prg.f_eff[0] - expect) < 1e-9, f"{prg.f_eff[0]} vs {expect}"
    print(f"[PASS] delta_f=+1 -> f_eff {prg.f_eff[0]:.3f} Hz (== blend {expect:.3f})")


def test_freeze_alpha_is_v71():
    """freeze_alpha + reset none + delta_f 0 == v71 integration."""
    prg = PRGState(num_envs=1, backend="numpy", reset_mode="none",
                   freeze_alpha=True)
    dt, v = 0.02, 1.0
    for _ in range(500):
        prg.step(dt, np.array([v]), np.zeros((1, 2)), np.array([0.0]))
    T = float(t_star(v))
    ref = (500 * dt / T) % 1.0
    err = abs(wrap_circ(prg.phi[0, 0] - ref))
    assert err < 1e-9, f"phi {prg.phi[0,0]} vs v71 {ref}"
    print(f"[PASS] freeze_alpha+none+df0 == v71 (err {err:.2e})")


if __name__ == "__main__":
    test_none_matches_v71_analytic()
    test_none_constant_v_phase_advance()
    test_hard_reset_anchors_phase()
    test_soft_is_rate_limited()
    test_delta_f_modulates_frequency()
    test_freeze_alpha_is_v71()
    print("\nALL TWIN TESTS PASSED")
