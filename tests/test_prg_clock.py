"""Tests for prg_clock.py — C1 PRG adaptive gait clock.

Most important gate: the v71 degradation gate (B0 baseline regression).
With reset_mode="none", freeze_alpha=True (alpha pinned to 0) and
delta_f=None (or zeros), the integrated phase and clock_obs must match the
v71 analytic formula phi_L = (t / T*(v)) % 1 bitwise-close
(allclose, atol=1e-6) on identical (t, v) sequences.

Runs standalone (python3 test_prg_clock.py) and under pytest.
On torch-less hosts the numpy backend is exercised (an isomorphic layer is
provided by prg_clock itself); when torch is importable the same suite runs
on both backends.
"""

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "robolab", "robolab", "tasks", "manager_based", "amp", "mdp",
    ),
)

import prg_clock  # noqa: E402
from prg_clock import PRGState  # noqa: E402

HAS_TORCH = prg_clock._torch is not None
BACKENDS = ["numpy"] + (["torch"] if HAS_TORCH else [])

DT = 0.02  # 50 Hz control


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_np(x):
    if prg_clock._is_torch(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _allclose(a, b, atol=1e-6, rtol=1e-5):
    return np.allclose(_to_np(a), _to_np(b), atol=atol, rtol=rtol)


def _assert_close(a, b, atol=1e-6, rtol=1e-5, msg=""):
    an, bn = _to_np(a), _to_np(b)
    if not np.allclose(an, bn, atol=atol, rtol=rtol):
        raise AssertionError(
            "%s not close: max|a-b|=%.3e (atol=%g)\na=%s\nb=%s"
            % (msg, float(np.max(np.abs(an - bn))) if an.size else 0.0,
               atol, an, bn)
        )


def _assert_phase_close(a, b, atol, msg=""):
    """Compare phases on the unit circle: the linear allclose fails when two
    mathematically equal phases land on opposite sides of the 1.0 wrap
    (e.g. integral 11.000000000000002 vs analytic 10.999999999999998); the
    wrapped difference is the faithful bit-level agreement test."""
    an, bn = _to_np(a), _to_np(b)
    err = ((an - bn + 0.5) % 1.0) - 0.5
    if not np.all(np.abs(err) <= atol):
        raise AssertionError(
            "%s phases differ on circle: max err=%.3e (atol=%g)\na=%s\nb=%s"
            % (msg, float(np.max(np.abs(err))), atol, an, bn)
        )


# Independent v71 reference implementation (deliberately NOT imported from
# prg_clock, so curve constants are cross-checked too).
def v71_T(v):
    return min(max(1.46 - 0.36 * v, 0.70), 1.75)


def v71_duty(v):
    return min(max(0.68 - 0.07 * v, 0.55), 0.70)


def v71_h(v):
    return min(max(0.025 + 0.025 * v, 0.025), 0.075)


def v71_obs(phiL, phiR):
    return np.stack(
        [
            np.sin(2 * np.pi * phiL),
            np.sin(2 * np.pi * phiR),
            np.cos(2 * np.pi * phiL),
            np.cos(2 * np.pi * phiR),
        ],
        axis=-1,
    )


def _deg_env(backend, **kw):
    """v71-equivalent configuration for the degradation gate."""
    kw.setdefault("reset_mode", "none")
    kw.setdefault("freeze_alpha", True)
    return PRGState(2, backend=backend, **kw)


# ---------------------------------------------------------------------------
# 1. constants & envvar plumbing
# ---------------------------------------------------------------------------
def test_constants_match_spec():
    assert prg_clock.PRG_RESET_MODE == "hard"
    assert prg_clock.PRG_RESET_GAIN == 2.0
    assert prg_clock.PRG_SYNC_WINDOW == 0.25
    assert prg_clock.PRG_ALPHA_MAX == 0.5
    assert prg_clock.PRG_ALPHA_RAMP_S == 10.0
    assert prg_clock.PRG_FMOD_SCALE == 0.5
    assert prg_clock.PRG_SOFT_CLAMP == 0.15
    # v71 curve constants
    assert (prg_clock.PRG_T_INTERCEPT, prg_clock.PRG_T_VSLOPE) == (1.46, 0.36)
    assert (prg_clock.PRG_T_LO, prg_clock.PRG_T_HI) == (0.70, 1.75)
    assert (prg_clock.PRG_D_INTERCEPT, prg_clock.PRG_D_VSLOPE) == (0.68, 0.07)
    assert (prg_clock.PRG_D_LO, prg_clock.PRG_D_HI) == (0.55, 0.70)
    assert (prg_clock.PRG_H_INTERCEPT, prg_clock.PRG_H_VSLOPE) == (0.025, 0.025)
    assert (prg_clock.PRG_H_LO, prg_clock.PRG_H_HI) == (0.025, 0.075)
    # curve functions vs independent reference (scalars covering clamp bands)
    for v in [0.0, 0.5, 1.0, 2.0, 2.5, 3.5]:
        assert abs(float(_to_np(prg_clock.t_star(v))) - v71_T(v)) < 1e-12
        assert abs(float(_to_np(prg_clock.duty_star(v))) - v71_duty(v)) < 1e-12
        assert abs(float(_to_np(prg_clock.h_star(v))) - v71_h(v)) < 1e-12


def test_envvar_overrides():
    """PRG_* envvars must override module constants (checked in a subprocess
    so the already-imported module state of this process stays clean)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mdp_dir = os.path.join(
        root, "robolab", "robolab", "tasks", "manager_based", "amp", "mdp")
    base = os.environ.copy()

    def run(code, env_extra):
        env = dict(base)
        env["PYTHONPATH"] = mdp_dir + os.pathsep + env.get("PYTHONPATH", "")
        env.pop("PRG_RESET_MODE", None)
        env.pop("PRG_FREEZE_ALPHA", None)
        env.update(env_extra)
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env, cwd=root,
        )
        assert r.returncode == 0, "stderr:\n" + r.stderr
        return r.stdout

    run(
        "import prg_clock as p; assert p.PRG_RESET_MODE=='hard', p.PRG_RESET_MODE; "
        "s=p.PRGState(1); assert s.reset_mode=='hard'",
        {},
    )
    run(
        "import prg_clock as p; assert p.PRG_RESET_MODE=='soft'; "
        "s=p.PRGState(1); assert s.reset_mode=='soft'",
        {"PRG_RESET_MODE": "soft"},
    )
    run(
        "import prg_clock as p; assert p.PRG_FREEZE_ALPHA is True; "
        "s=p.PRGState(1); assert s.freeze_alpha is True; "
        "import numpy as np; "
        "s.step(0.02, np.array([1.0]), np.zeros((1,2),bool)); "
        "assert float(s.alpha[0])==0.0",
        {"PRG_FREEZE_ALPHA": "1"},
    )
    run(
        "import prg_clock as p; assert p.PRG_RESET_GAIN==3.5",
        {"PRG_RESET_GAIN": "3.5"},
    )
    # invalid mode must fail loudly at import
    env = dict(base)
    env["PYTHONPATH"] = mdp_dir + os.pathsep + env.get("PYTHONPATH", "")
    env["PRG_RESET_MODE"] = "bogus"
    r = subprocess.run(
        [sys.executable, "-c", "import prg_clock"],
        capture_output=True, text=True, env=env, cwd=root,
    )
    assert r.returncode != 0 and "PRG_RESET_MODE" in r.stderr


# ---------------------------------------------------------------------------
# 2. v71 degradation gate (THE B0 baseline regression gate)
# ---------------------------------------------------------------------------
def test_v71_gate_constant_speed():
    """Constant v: phi and clock_obs must equal phiL=(t/T*)%1 analytically,
    bitwise-close (atol=1e-6), for speeds covering all three T* clamp bands."""
    speeds = [0.0, 1.0, 2.5, 3.5]  # T* = 1.46 / 1.10 / 0.70 / 0.70(clamped)
    n_steps = 1500  # 30 s at 50 Hz
    for backend in BACKENDS:
        for delta_f_mode in ("none", "zeros"):  # both are v71-equivalent
            for v0 in speeds:
                st = _deg_env(backend)
                v = np.full(2, v0)
                df = None if delta_f_mode == "none" else np.zeros(2)
                for k in range(n_steps):
                    st.step(DT, v, np.zeros((2, 2), bool), delta_f=df)
                    t = (k + 1) * DT
                    refL = (t / v71_T(v0)) % 1.0
                    refR = (refL + 0.5) % 1.0
                    _assert_phase_close(
                        st.phi[:, 0], refL, atol=1e-6,
                        msg="phiL v=%g step=%d (%s)" % (v0, k, backend))
                    _assert_phase_close(
                        st.phi[:, 1], refR, atol=1e-6,
                        msg="phiR v=%g step=%d (%s)" % (v0, k, backend))
                    if k % 37 == 0:  # spot-check obs (cheap subset)
                        _assert_close(
                            st.clock_obs(), v71_obs(refL, refR), atol=1e-6,
                            msg="obs v=%g step=%d (%s)" % (v0, k, backend))
                # final full obs check (sin/cos are continuous across the
                # wrap, so linear allclose is the right comparison here)
                t = n_steps * DT
                refL = (t / v71_T(v0)) % 1.0
                refR = (refL + 0.5) % 1.0
                _assert_phase_close(st.phi[:, 0], refL, atol=1e-6,
                                    msg="final phiL v=%g (%s)" % (v0, backend))
                _assert_close(st.clock_obs(), v71_obs(refL, refR), atol=1e-6,
                              msg="final obs v=%g (%s)" % (v0, backend))
                # antiphase preserved exactly throughout
                off = (st.phi[:, 1] - st.phi[:, 0]) % 1.0
                _assert_phase_close(off, np.full(2, 0.5), atol=1e-9,
                                    msg="antiphase v=%g (%s)" % (v0, backend))
                # f_eff equals the analytic command frequency
                _assert_close(st.f_eff, np.full(2, 1.0 / v71_T(v0)), atol=1e-12,
                              msg="f_eff v=%g (%s)" % (v0, backend))


def test_v71_gate_piecewise_speed():
    """Time-varying v: matches the v71 per-step integration form
    phiL += dt / T*(v_k)  (the discrete-clock semantics of the same formula);
    a fresh PRGState and the reference accumulate identical increments."""
    seg_len, n_seg = 100, 12
    v_vals = [0.5, 1.2, 2.0, 1.0, 0.0, 2.5, 3.5, 0.8, 1.6, 2.2, 0.3, 1.4]
    for backend in BACKENDS:
        st = _deg_env(backend)
        refL = np.zeros(2)
        for seg in range(n_seg):
            v0 = v_vals[seg]
            v = np.full(2, v0)
            for _ in range(seg_len):
                st.step(DT, v, np.zeros((2, 2), bool))
                refL += DT / v71_T(v0)
                refL %= 1.0
                refR = (refL + 0.5) % 1.0
                _assert_close(st.phi[:, 0], refL, atol=1e-9,
                              msg="phiL seg=%d (%s)" % (seg, backend))
                _assert_close(st.phi[:, 1], refR, atol=1e-9,
                              msg="phiR seg=%d (%s)" % (seg, backend))


# ---------------------------------------------------------------------------
# 3. hard reset anchoring
# ---------------------------------------------------------------------------
def _set_phi(st, value_lr):
    st.phi[:, 0] = value_lr[0]
    st.phi[:, 1] = value_lr[1]


def _set_phi_full(st, arr):
    """Assign a full (N,2) numpy phase array regardless of backend."""
    arr = np.asarray(arr)
    if st.backend == "torch":
        st.phi[:] = prg_clock._torch.as_tensor(
            arr, dtype=st.phi.dtype, device=st.phi.device)
    else:
        st.phi[:] = arr


def test_hard_reset_anchor_in_window():
    """Touchdown while phi is inside the swing-tail sync window re-anchors
    phi to 0; the OTHER foot (out of window) keeps integrating untouched."""
    for backend in BACKENDS:
        st = PRGState(1, backend=backend, reset_mode="hard", freeze_alpha=True)
        v = np.array([1.0])
        st.step(DT, v, np.zeros((1, 2), bool))  # t_ep=dt, contact_prev=False
        f_dt = DT / v71_T(1.0)
        _set_phi(st, (0.97, 0.47))  # L in [1-0.25, 1): window ; R mid-swing
        st.step(DT, v, np.ones((1, 2), bool))  # touchdown both feet
        assert float(st.phi[0, 0]) == 0.0, "L must hard-reset to stance start"
        _assert_close(st.phi[0, 1], 0.47 + f_dt, atol=1e-12,
                      msg="R must keep integrating (out of window)")
        # contact stays True -> no new edge -> no further reset next steps
        _set_phi(st, (0.90, 0.40))
        st.step(DT, v, np.ones((1, 2), bool))
        _assert_close(st.phi[0, 0], 0.90 + f_dt, atol=1e-9,
                      msg="held contact must NOT re-trigger reset")


def test_hard_reset_ignores_out_of_window_touchdown():
    for backend in BACKENDS:
        st = PRGState(1, backend=backend, reset_mode="hard", freeze_alpha=True)
        v = np.array([1.0])
        st.step(DT, v, np.zeros((1, 2), bool))
        f_dt = DT / v71_T(1.0)
        _set_phi(st, (0.50, 0.10))  # both outside [0.75,1) U [0,0.05)
        st.step(DT, v, np.ones((1, 2), bool))  # touchdown, but ignored
        _assert_close(st.phi[0, 0], 0.50 + f_dt, atol=1e-12,
                      msg="mid-stance touchdown must not reset L")
        _assert_close(st.phi[0, 1], 0.10 + f_dt, atol=1e-12,
                      msg="mid-stance touchdown must not reset R")


def test_hard_reset_stance_head_window_and_per_foot():
    """phi just above 0 (stance head [0, 0.05)) also triggers; resets are
    strictly per-foot — a RIGHT-foot touchdown re-anchors only the right
    phase and the left foot keeps its antiphase relation to time."""
    for backend in BACKENDS:
        st = PRGState(1, backend=backend, reset_mode="hard", freeze_alpha=True)
        v = np.array([1.0])
        st.step(DT, v, np.zeros((1, 2), bool))
        f_dt = DT / v71_T(1.0)
        # right foot only, in swing-tail window
        _set_phi(st, (0.30, 0.76))
        st.step(DT, v, np.array([[False, True]], dtype=bool))
        _assert_close(st.phi[0, 1], 0.0, atol=1e-15,
                      msg="R touchdown in window must reset R to 0")
        _assert_close(st.phi[0, 0], 0.30 + f_dt, atol=1e-12,
                      msg="L phase must be untouched by R reset")
        # right foot in stance-head window (L was re-pinned to 0.30 above)
        st.contact_prev[:] = False
        _set_phi(st, (0.30, 0.03))
        st.step(DT, v, np.array([[False, True]], dtype=bool))
        _assert_close(st.phi[0, 1], 0.0, atol=1e-15,
                      msg="R in [0,0.05) head window must reset")
        _assert_close(st.phi[0, 0], 0.30 + f_dt, atol=1e-12,
                      msg="L must still integrate")


def test_no_touchdown_hard_equals_none():
    """Without contact edges, hard/soft/none produce identical open-loop
    trajectories (resets are purely event-driven)."""
    for backend in BACKENDS:
        states = {
            m: PRGState(2, backend=backend, reset_mode=m, freeze_alpha=False)
            for m in ("hard", "soft", "none")
        }
        rng = np.random.default_rng(0)
        for k in range(600):
            v = rng.uniform(0.0, 2.5, size=2)
            df = rng.uniform(-1.0, 1.0, size=2)
            for st in states.values():
                st.step(DT, v, np.zeros((2, 2), bool), delta_f=df)
        for m in ("hard", "soft"):
            _assert_close(states[m].phi, states["none"].phi, atol=1e-12,
                          msg="%s vs none w/o contact (%s)" % (m, backend))
            _assert_close(states[m].clock_obs(), states["none"].clock_obs(),
                          atol=1e-12, msg="obs %s vs none (%s)" % (m, backend))


# ---------------------------------------------------------------------------
# 4. soft reset convergence
# ---------------------------------------------------------------------------
def _soft_apply(st, phiL, v):
    """Force phi and fire one touchdown edge on the left foot; return phiL."""
    st.phi[:, 0] = phiL
    st.contact_prev[:] = False
    st.step(DT, v, np.array([[True, False], [True, False]], dtype=bool))
    return float(st.phi[0, 0])


def test_soft_reset_pulls_toward_zero():
    """Single soft reset: correction direction + magnitude follow
    corr = clip(-beta*dt*wrap_err(phi), +/- PRG_SOFT_CLAMP*dt*beta)."""
    beta, clamp_, v = prg_clock.PRG_RESET_GAIN, prg_clock.PRG_SOFT_CLAMP, 1.0
    f_dt = DT / v71_T(v)
    for backend in BACKENDS:
        for phi0 in [0.76, 0.80, 0.85, 0.90, 0.95, 0.99, 0.02, 0.04]:
            st = PRGState(2, backend=backend, reset_mode="soft",
                          freeze_alpha=True)
            st.step(DT, np.full(2, v), np.zeros((2, 2), bool))  # t_ep=dt
            phi_end = _soft_apply(st, phi0, v)
            q = (phi0 + f_dt) % 1.0
            in_sync = (q >= 1.0 - prg_clock.PRG_SYNC_WINDOW) or (q < 0.05)
            if in_sync:
                err = q if q < 0.5 else q - 1.0
                corr = max(-beta * DT * err, -clamp_ * DT * beta)
                corr = min(corr, clamp_ * DT * beta)
            else:
                corr = 0.0  # outside sync window: touchdown is NOT a reset
            expected = (phi0 + f_dt + corr) % 1.0
            _assert_close(phi_end, expected, atol=1e-12,
                          msg="soft corr phi0=%g (%s)" % (phi0, backend))
            # |wrap error| must strictly shrink when the correction bites
            err_after = expected if expected < 0.5 else expected - 1.0
            err_before = q if q < 0.5 else q - 1.0
            if in_sync and abs(err_before) > 1e-9:
                assert abs(err_after) < abs(err_before), (phi0, err_before,
                                                         err_after)
            elif not in_sync:
                _assert_close(err_after, err_before, atol=0.0,
                              msg="out-of-window touchdown must not correct")


def test_soft_reset_monotone_convergence():
    """Full gait cycles, one touchdown per cycle (v=1 -> T*=1.10 s = exactly
    55 steps at dt=0.02, so free-running adds no net phase error between
    touchdowns): the phase error at each touchdown event shrinks
    monotonically, factor (1 - beta*dt) per event while unclamped."""
    v0 = 1.0
    steps_per_cycle = round(v71_T(v0) / DT)  # 55
    assert abs(steps_per_cycle * DT - v71_T(v0)) < 1e-12
    for backend in BACKENDS:
        st = PRGState(1, backend=backend, reset_mode="soft", freeze_alpha=True)
        v = np.array([v0])
        st.step(DT, v, np.zeros((1, 2), bool))  # t_ep=dt, contact_prev=False
        # seed a phase lead error: physical foot lands early at phi=0.95
        st.phi[:, 0] = 0.95
        st.step(DT, v, np.array([[True, False]], dtype=bool))  # touchdown
        errs = []
        q = float(st.phi[0, 0]) % 1.0
        errs.append(q if q < 0.5 else q - 1.0)
        for _ in range(12):
            for _k in range(steps_per_cycle - 1):
                st.step(DT, v, np.zeros((1, 2), bool))
            st.step(DT, v, np.array([[True, False]], dtype=bool))
            q = float(st.phi[0, 0]) % 1.0
            errs.append(q if q < 0.5 else q - 1.0)
        mags = [abs(e) for e in errs]
        for a, b in zip(mags, mags[1:]):
            assert b <= a + 1e-9, "td error must not grow: %s" % mags
        assert mags[-1] < 0.75 * mags[0], (
            "td error must contract across events: %s" % mags)


# ---------------------------------------------------------------------------
# 5. frequency modulation limits & alpha curriculum
# ---------------------------------------------------------------------------
def test_fmod_clamp_and_bounds():
    """delta_f=+/-1 lands f_eff exactly on [f_cmd*(1-0.5a), f_cmd*(1+0.5a)];
    intermediate delta_f interpolates linearly; alpha=0 pins f_eff=f_cmd."""
    s, scale = prg_clock.PRG_ALPHA_MAX, prg_clock.PRG_FMOD_SCALE
    for backend in BACKENDS:
        st = PRGState(2, backend=backend, reset_mode="none", freeze_alpha=False)
        v = np.array([1.0, 2.0])
        for t_ep, alpha_exp in [(0.0, 0.0), (2.5, 0.125), (5.0, 0.25),
                                (10.0, 0.5), (30.0, 0.5)]:
            f_cmd = 1.0 / v71_T(1.0)
            f_cmd2 = 1.0 / v71_T(2.0)
            for df in (-1.0, -0.37, 0.0, 0.6, 1.0):
                st.t_ep[:] = t_ep  # pin episode time for this reading
                st.contact_prev[:] = False
                st.step(DT, v, np.zeros((2, 2), bool),
                        delta_f=np.full(2, df))
                _assert_close(st.alpha, np.full(2, alpha_exp), atol=1e-12,
                              msg="alpha at t_ep=%g" % t_ep)
                # exact endpoints at |df|=1
                if df == -1.0:
                    exp = [f_cmd * (1 - 0.5 * alpha_exp),
                           f_cmd2 * (1 - 0.5 * alpha_exp)]
                    _assert_close(st.f_eff, exp, atol=1e-12,
                                  msg="f_eff df=-1 t=%g" % t_ep)
                elif df == 1.0:
                    exp = [f_cmd * (1 + 0.5 * alpha_exp),
                           f_cmd2 * (1 + 0.5 * alpha_exp)]
                    _assert_close(st.f_eff, exp, atol=1e-12,
                                  msg="f_eff df=+1 t=%g" % t_ep)
                else:  # interior points within the bounds
                    lo = [f_cmd * (1 - 0.5 * alpha_exp),
                          f_cmd2 * (1 - 0.5 * alpha_exp)]
                    hi = [f_cmd * (1 + 0.5 * alpha_exp),
                          f_cmd2 * (1 + 0.5 * alpha_exp)]
                    fe = _to_np(st.f_eff)
                    assert np.all(fe >= np.array(lo) - 1e-12) and \
                        np.all(fe <= np.array(hi) + 1e-12), (t_ep, df, fe)
                if alpha_exp == 0.0:  # v71 mode: policy cannot bend f_eff
                    _assert_close(st.f_eff, [f_cmd, f_cmd2], atol=1e-12,
                                  msg="f_eff alpha=0 df=%g" % df)
        assert scale == 0.5 and s == 0.5  # spec constants


def test_alpha_curriculum_monotone():
    for backend in BACKENDS:
        st = PRGState(1, backend=backend, reset_mode="none", freeze_alpha=False)
        v = np.array([1.0])
        prev = -1.0
        for k in range(1001):  # 20 s > ramp
            st.step(DT, v, np.zeros((1, 2), bool))
            a = float(st.alpha[0])
            assert a >= prev - 1e-15, "alpha must be non-decreasing"
            prev = a
            # alpha is computed from t_ep BEFORE this step's increment,
            # i.e. from k*DT at the k-th call (0-indexed)
            t = k * DT
            assert abs(a - min(t / 10.0, 1.0) * 0.5) < 1e-12, (k, a, t)
        assert abs(prev - 0.5) < 1e-12  # saturates at PRG_ALPHA_MAX

        stf = PRGState(1, backend=backend, reset_mode="none", freeze_alpha=True)
        for _ in range(300):
            stf.step(DT, v, np.zeros((1, 2), bool))
            assert float(stf.alpha[0]) == 0.0, "frozen alpha must stay 0"


# ---------------------------------------------------------------------------
# 6. episode reset scoping & initial anchoring
# ---------------------------------------------------------------------------
def test_episode_reset_scoping():
    for backend in BACKENDS:
        st = PRGState(3, backend=backend, reset_mode="hard", freeze_alpha=False)
        v = np.array([1.0, 1.5, 2.0])
        contacts = np.zeros((3, 2), bool)
        for k in range(50):  # vary contacts to move contact_prev too
            contacts[:, k % 2] = (k % 4 < 2)
            st.step(DT, v, contacts, delta_f=np.array([0.3, -0.5, 0.0]))
        phi_before = _to_np(st.phi).copy()
        t_before = _to_np(st.t_ep).copy()
        cp_before = _to_np(st.contact_prev).copy()
        st.reset(env_ids=[1])
        phi_after = _to_np(st.phi)
        # env 1 re-anchored
        assert phi_after[1, 0] == 0.0 and phi_after[1, 1] == 0.5
        assert float(_to_np(st.t_ep)[1]) == 0.0
        assert not bool(_to_np(st.contact_prev)[1].any())
        assert float(_to_np(st.f_eff)[1]) == 0.0
        # envs 0 and 2 untouched (bit-for-bit)
        np.testing.assert_array_equal(phi_after[0], phi_before[0])
        np.testing.assert_array_equal(phi_after[2], phi_before[2])
        np.testing.assert_array_equal(_to_np(st.t_ep)[[0, 2]], t_before[[0, 2]])
        np.testing.assert_array_equal(
            _to_np(st.contact_prev)[[0, 2]], cp_before[[0, 2]])


def test_reset_all_and_initial_anchor():
    for backend in BACKENDS:
        st = PRGState(2, backend=backend)
        st.step(DT, np.full(2, 1.0), np.zeros((2, 2), bool))
        st.reset()  # all envs
        np.testing.assert_allclose(_to_np(st.phi), [[0.0, 0.5], [0.0, 0.5]])
        assert float(_to_np(st.t_ep).sum()) == 0.0
        assert not _to_np(st.contact_prev).any()


# ---------------------------------------------------------------------------
# 7. obs layout & targets
# ---------------------------------------------------------------------------
def test_clock_obs_layout():
    for backend in BACKENDS:
        st = PRGState(2, backend=backend)
        st.phi[:, 0] = 0.25
        st.phi[:, 1] = 0.0
        obs = _to_np(st.clock_obs())
        assert obs.shape == (2, 4)
        # [sinL, sinR, cosL, cosR] = [1, 0, 0, 1] for phiL=0.25, phiR=0
        np.testing.assert_allclose(obs[0], [1.0, 0.0, 0.0, 1.0], atol=1e-12)
        st.phi[:, 0] = 0.0
        st.phi[:, 1] = 0.25
        # new order [sinL, sinR, cosL, cosR]: phiR=0.25 -> [0, 1, 1, 0]
        # (also numerically equal to the old order by coincidence, so the
        # distinguishing case follows below)
        np.testing.assert_allclose(
            _to_np(st.clock_obs())[0], [0.0, 1.0, 1.0, 0.0], atol=1e-12)
        st.phi[:, 0] = 0.25
        st.phi[:, 1] = 0.25
        # distinguishing case: new order [1, 1, 0, 0] vs old order [1, 0, 1, 0]
        np.testing.assert_allclose(
            _to_np(st.clock_obs())[0], [1.0, 1.0, 0.0, 0.0], atol=1e-12)
        rng = np.random.default_rng(3)
        _set_phi_full(st, rng.uniform(0, 1, size=(2, 2)))
        obs = _to_np(st.clock_obs())
        assert np.all(obs >= -1.0 - 1e-12) and np.all(obs <= 1.0 + 1e-12)
        np.testing.assert_allclose(
            obs, v71_obs(_to_np(st.phi)[:, 0], _to_np(st.phi)[:, 1]),
            atol=1e-12)


def test_targets():
    for backend in BACKENDS:
        st = PRGState(3, backend=backend)
        v = np.array([0.0, 1.0, 2.5])
        _set_phi_full(st, [[0.30, 0.65], [0.30, 0.65], [0.30, 0.65]])
        tg = st.targets(v)
        np.testing.assert_allclose(
            _to_np(tg["duty"]), [v71_duty(0.0), v71_duty(1.0), v71_duty(2.5)],
            atol=1e-12)
        np.testing.assert_allclose(
            _to_np(tg["h"]), [v71_h(0.0), v71_h(1.0), v71_h(2.5)], atol=1e-12)
        stance = _to_np(tg["stance"])
        swing = _to_np(tg["swing"])
        # duty: env0=0.68, env1=0.61, env2=0.55 -> 0.30 always stance;
        # 0.65 only stance in env0
        np.testing.assert_array_equal(
            stance, [[True, True], [True, False], [True, False]])
        np.testing.assert_array_equal(swing, ~stance)


def test_repr_and_input_coercion():
    st = PRGState(2, backend="numpy")
    # python list inputs are accepted
    st.step(0.02, [1.0, 0.5], [[False, False], [True, False]],
            delta_f=[0.1, -0.1])
    assert st.phi.shape == (2, 2)
    # scalar v_cmd broadcasts
    st2 = PRGState(2, backend="numpy", reset_mode="none", freeze_alpha=True)
    st2.step(0.02, 1.0, np.zeros((2, 2), bool))
    np.testing.assert_allclose(_to_np(st2.phi[:, 0]), [DT / v71_T(1.0)] * 2)
    assert "PRGState" in repr(st)
    # invalid shapes rejected
    try:
        st.step(0.02, np.zeros(3), np.zeros((2, 2), bool))
        raise RuntimeError("should have raised")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def _main():
    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print("PASS %s" % name)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("FAIL %s: %r" % (name, exc))
            failures.append(name)
    print("-" * 60)
    print("%d/%d tests passed (backends: %s, torch=%s)"
          % (len(tests) - len(failures), len(tests), BACKENDS, HAS_TORCH))
    if failures:
        print("FAILED: %s" % ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
