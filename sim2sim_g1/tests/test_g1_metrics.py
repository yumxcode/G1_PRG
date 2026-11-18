#!/usr/bin/env python3
"""test_g1_metrics.py — validate gait_metrics_summary on a synthetic gait.

No MuJoCo needed.  Builds a synthetic 12 s biped walk log (period 1.10 s,
duty 0.65, stance slip 0.05 m/s, perfect PRG phase sync: phi=0 at each
touchdown) and checks the summary recovers cadence / duty / swing / slip /
phase-sync error.  A second pass injects a constant phase offset (+0.1
cycle) and checks the sync error reads ~0.1.

Run:  python3 tests/test_g1_metrics.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from g1_sim2sim import gait_metrics_summary, SOLE_R  # noqa: E402

DT = 0.02
T_CYCLE = 1.10          # v71 T*(1.0)
DUTY = 0.65
SLIP = 0.05             # m/s during stance
PHASE_OFF = 0.0         # (case 1)
N_SEC = 12.0


def synth(phase_offset=0.0):
    n = int(N_SEC / DT)
    t = np.arange(n) * DT
    sole = np.zeros((n, 2, 4, 3))
    phi = np.zeros((n, 2))
    for f in (0, 1):  # 0 = left, 1 = right
        ph = (t / T_CYCLE + (0.0 if f == 0 else 0.5)) % 1.0   # ideal anti-phase
        phi[:, f] = ph
        stance = ph < DUTY
        # foot mid-point xy: advances SLIP during stance, swings forward fast
        x = np.zeros(n)
        for i in range(1, n):
            if stance[i] and stance[i - 1]:
                x[i] = x[i - 1] + SLIP * DT
            else:
                # swing: recover the slip + step forward (net forward speed 0)
                x[i] = x[i - 1] - (SLIP * DUTY * T_CYCLE) / ((1 - DUTY) * T_CYCLE) * DT
        z_foot = np.where(stance, 0.003, 0.05)   # sole centre height (r=0.005)
        # 4 spheres: heel pair x-0.085, toe pair x+0.085, y +-0.03
        for k, (dx, dy) in enumerate([(-0.085, -0.03), (-0.085, 0.03),
                                      (0.085, -0.03), (0.085, 0.03)]):
            sole[:, f, k, 0] = x + dx
            sole[:, f, k, 1] = dy
            sole[:, f, k, 2] = z_foot
    # phi AT each sample = clock phase (offset applied for case 2).  For the
    # sync-error metric the phase at a touchdown frame should read ~0 (+off).
    phi_off = (phi + phase_offset) % 1.0
    return t, sole, phi_off


def check_case(offset, tol_sync):
    t, sole, phi = synth(offset)
    out = gait_metrics_summary(t, sole, phi, start_i=5, dt=DT,
                               foot_names=("left", "right"))
    ok = True
    for foot in ("left", "right"):
        g = out[foot]
        n_exp = int((N_SEC - 1) / T_CYCLE)
        cyc_err = abs(g["cycle_s_med"] - T_CYCLE)
        duty_err = abs(g["duty_med"] - DUTY)
        swing_exp = (1 - DUTY) * T_CYCLE
        swing_err = abs(g["swing_s_med"] - swing_exp)
        slip_err = abs(g["slip_mps_med"] - SLIP)
        sync = g["phase_sync_err_mean_abs"]
        print(f"  {foot}: n={g['n_events']}/{n_exp} cycle={g['cycle_s_med']:.3f} "
              f"(err {cyc_err:.3f}) duty={g['duty_med']:.3f} (err {duty_err:.3f}) "
              f"swing={g['swing_s_med']:.3f} (err {swing_err:.3f}) "
              f"slip={g['slip_mps_med']:.3f} (err {slip_err:.3f}) "
              f"sync={sync:.4f}")
        ok &= (g["n_events"] >= n_exp - 2 and cyc_err < 0.02
               and duty_err < 0.03 and swing_err < 0.05
               and slip_err < 0.01 and abs(sync - abs(offset)) < tol_sync)
    return ok


if __name__ == "__main__":
    print("[case 1] perfect sync (phi=0 at TD):")
    ok1 = check_case(0.0, tol_sync=0.05)
    print("[case 2] constant phase offset +0.1 cycle:")
    ok2 = check_case(0.1, tol_sync=0.05)
    assert ok1 and ok2
    print("\nALL METRICS TESTS PASSED")
