"""prg_clock.py — C1 PRG (Contact-triggered Phase-Reset Gait) adaptive gait clock.

Upgrades the v71 exogenous gait clock (pure time function phi = t / T*(v) % 1)
into a stateful hybrid clock for the Unitree G1 humanoid:

  1. Contact-triggered phase reset  — a foot touchdown event re-anchors that
     foot's phase to 0 (stance start).  Two strengths:
       "hard"  phi[foot] := 0            (instant re-anchor)
       "soft"  phi[foot] += corr         (rate-limited pull toward 0,
                                          corr = clip(-beta*dt*wrap_err(phi),
                                                      +/- PRG_SOFT_CLAMP*dt*beta))
       "none"  no reset (v71-equivalent open-loop integration)
  2. Policy frequency modulation    — the policy outputs delta_f in [-1, 1];
     the effective frequency blends the command frequency f_cmd = 1/T*(v_cmd)
     with the policy frequency f_pol = f_cmd*(1 + PRG_FMOD_SCALE*delta_f):
       f_eff = (1-alpha)*f_cmd + alpha*f_pol
     alpha follows a linear curriculum over episode time:
       alpha = clamp(t_ep / PRG_ALPHA_RAMP_S, 0, 1) * PRG_ALPHA_MAX
     (alpha == 0 for the whole episode when freeze_alpha is set -> exact v71).

v71 gait curves (authoritative, from the task spec):
  T*(v)   = clamp(1.46 - 0.36 v, 0.70, 1.75)   single-foot cycle period  [s]
  d*(v)   = clip (0.68 - 0.07 v, 0.55, 0.70)   duty factor (stance share)
  h*(v)   = clip (0.025 + 0.025 v, 0.025, 0.075) swing apex height       [m]
  phiL = (t/T*)%1 ; phiR = (phiL + 0.5)%1 ; stance: phi in [0, d*), swing: [d*, 1)
  clock obs = [sin 2pi phiL, cos 2pi phiL, sin 2pi phiR, cos 2pi phiR]   (4,)

Isaaclab-free: only numpy is required.  If torch is importable the same class
runs natively on torch tensors (CPU or CUDA) with an identical API; without
torch a numpy isomorphic layer is used (signatures keep the torch style).

All PRG_* constants can be overridden with environment variables of the same
name (read once at import time).
"""

import os

import numpy as np

try:  # torch is optional: numpy-only deployments (sim2sim twin, CI) work fine
    import torch as _torch
except Exception:  # pragma: no cover - exercised only on torch-less hosts
    _torch = None

__all__ = [
    "PRG_RESET_MODE",
    "PRG_RESET_GAIN",
    "PRG_SYNC_WINDOW",
    "PRG_SYNC_HEAD",
    "PRG_ALPHA_MAX",
    "PRG_ALPHA_RAMP_S",
    "PRG_FMOD_SCALE",
    "PRG_SOFT_CLAMP",
    "PRG_T_INTERCEPT",
    "PRG_T_VSLOPE",
    "PRG_T_LO",
    "PRG_T_HI",
    "PRG_D_INTERCEPT",
    "PRG_D_VSLOPE",
    "PRG_D_LO",
    "PRG_D_HI",
    "PRG_H_INTERCEPT",
    "PRG_H_VSLOPE",
    "PRG_H_LO",
    "PRG_H_HI",
    "PRG_BACKEND",
    "PRG_FREEZE_ALPHA",
    "t_star",
    "duty_star",
    "h_star",
    "f_cmd_star",
    "PRGState",
]


def _env_float(name, default):
    return float(os.environ.get(name, default))


# ---------------------------------------------------------------------------
# Module-level PRG constants (envvar-overridable, same-name env vars)
# ---------------------------------------------------------------------------
PRG_RESET_MODE = os.environ.get("PRG_RESET_MODE", "hard")
if PRG_RESET_MODE not in ("hard", "soft", "none"):
    raise ValueError(
        "PRG_RESET_MODE must be one of {'hard','soft','none'}, got %r"
        % (PRG_RESET_MODE,)
    )

PRG_RESET_GAIN = _env_float("PRG_RESET_GAIN", 2.0)      # beta [1/s], soft reset rate
PRG_SYNC_WINDOW = _env_float("PRG_SYNC_WINDOW", 0.25)   # gamma, swing-tail window
PRG_SYNC_HEAD = _env_float("PRG_SYNC_HEAD", 0.05)       # stance-head window head
PRG_ALPHA_MAX = _env_float("PRG_ALPHA_MAX", 0.5)        # max policy-frequency blend
PRG_ALPHA_RAMP_S = _env_float("PRG_ALPHA_RAMP_S", 10.0)  # linear curriculum time [s]
PRG_FMOD_SCALE = _env_float("PRG_FMOD_SCALE", 0.5)      # delta_f -> +/-50% f_cmd
PRG_SOFT_CLAMP = _env_float("PRG_SOFT_CLAMP", 0.15)     # soft-reset max corr [cycles]

# v71 gait curve constants (envvar-overridable for experiment sweeps)
PRG_T_INTERCEPT = _env_float("PRG_T_INTERCEPT", 1.46)   # T*(v) = clamp(a - b*v, lo, hi)
PRG_T_VSLOPE = _env_float("PRG_T_VSLOPE", 0.36)
PRG_T_LO = _env_float("PRG_T_LO", 0.70)
PRG_T_HI = _env_float("PRG_T_HI", 1.75)
PRG_D_INTERCEPT = _env_float("PRG_D_INTERCEPT", 0.68)   # d*(v) = clip(a - b*v, lo, hi)
PRG_D_VSLOPE = _env_float("PRG_D_VSLOPE", 0.07)
PRG_D_LO = _env_float("PRG_D_LO", 0.55)
PRG_D_HI = _env_float("PRG_D_HI", 0.70)
PRG_H_INTERCEPT = _env_float("PRG_H_INTERCEPT", 0.025)  # h*(v) = clip(a + b*v, lo, hi)
PRG_H_VSLOPE = _env_float("PRG_H_VSLOPE", 0.025)
PRG_H_LO = _env_float("PRG_H_LO", 0.025)
PRG_H_HI = _env_float("PRG_H_HI", 0.075)

# Backend selection: explicit ctor arg > PRG_BACKEND envvar > auto
#   auto = torch if torch is importable, else numpy.
PRG_BACKEND = os.environ.get("PRG_BACKEND", "auto")
# Freeze the alpha curriculum at 0 for the whole process ("1"/"0").
PRG_FREEZE_ALPHA = os.environ.get("PRG_FREEZE_ALPHA", "0") == "1"


# ---------------------------------------------------------------------------
# Backend-neutral helpers (identical semantics on numpy ndarray / torch.Tensor)
# ---------------------------------------------------------------------------
def _is_torch(x):
    return _torch is not None and isinstance(x, _torch.Tensor)


def _clamp(x, lo, hi):
    if _is_torch(x):
        return _torch.clamp(x, min=lo, max=hi)
    return np.clip(x, lo, hi)


def _where(cond, a, b):
    if _is_torch(cond) or _is_torch(a) or _is_torch(b):
        return _torch.where(cond, a, b)
    return np.where(cond, a, b)


def _logical_not(x):
    if _is_torch(x):
        return _torch.logical_not(x)
    return np.logical_not(x)


def _stack(cols, axis_or_dim):
    if any(_is_torch(c) for c in cols):
        return _torch.stack(cols, dim=axis_or_dim)
    return np.stack(cols, axis=axis_or_dim)


# ---------------------------------------------------------------------------
# v71 gait curves (backend-neutral: accept python scalars, numpy, or torch)
# ---------------------------------------------------------------------------
def t_star(v):
    """Single-foot cycle period T*(v) in seconds (v71)."""
    return _clamp(PRG_T_INTERCEPT - PRG_T_VSLOPE * v, PRG_T_LO, PRG_T_HI)


def duty_star(v):
    """Stance duty factor d*(v) (v71)."""
    return _clamp(PRG_D_INTERCEPT - PRG_D_VSLOPE * v, PRG_D_LO, PRG_D_HI)


def h_star(v):
    """Swing apex height h*(v) in meters (v71)."""
    return _clamp(PRG_H_INTERCEPT + PRG_H_VSLOPE * v, PRG_H_LO, PRG_H_HI)


def f_cmd_star(v):
    """Command step frequency f_cmd(v) = 1/T*(v) in Hz."""
    return 1.0 / t_star(v)


# ---------------------------------------------------------------------------
# v71-mirrored per-phase target functions (used by the PRG reward terms).
# Same semantics as mdp/gait_clock.py::{target_height, swing_mid_mask,
# stance_mid_mask} but operating on an EXTERNAL phi array (the PRG state's
# phase), so rewards can score against the ADAPTIVE clock instead of the
# pure-time one. Bitwise-identical formulas.
# ---------------------------------------------------------------------------
def _col2(v):
    """(N,) -> (N,1) column for broadcasting against (N,2) phi."""
    if _is_torch(v):
        return v[:, None]
    return np.asarray(v)[:, None]


def target_height(phi, v):
    """Desired lowest-sole-point height z*(phi_i) per foot, shape (N, 2).

    0 through the stance window; raised C0 bell peaking at h*(v) mid-swing
    (v71 formula, evaluated at the PRG state's phi).
    """
    d = _clamp(_col2(duty_star(v)), PRG_D_LO, PRG_D_HI)
    h = _clamp(_col2(h_star(v)), PRG_H_LO, PRG_H_HI)
    frac = (phi - d) / _clamp(1.0 - d, 1e-6, 1.0)   # <0 stance, [0,1) swing
    pi = _torch.pi if _is_torch(frac) else np.pi
    sin = _torch.sin if _is_torch(frac) else np.sin
    zeros = (_torch.zeros_like if _is_torch(frac) else np.zeros_like)
    bell = sin(pi * _clamp(frac, 0.0, 1.0))
    return _where(frac > 0.0, h * bell, zeros(bell))


# hinge windows (fractions INSIDE each desired phase span) — v71 constants
SWING_MID_LO, SWING_MID_HI = 0.2, 0.8     # of the swing span
STANCE_MID_LO, STANCE_MID_HI = 0.15, 0.85  # of the stance span


def swing_mid_mask(phi, v):
    """True where the foot SHOULD be mid-swing (contact here = violation)."""
    d = _clamp(_col2(duty_star(v)), PRG_D_LO, PRG_D_HI)
    span = _clamp(1.0 - d, 1e-6, 1.0)
    return (phi >= d + SWING_MID_LO * span) & (phi < d + SWING_MID_HI * span)


def stance_mid_mask(phi, v):
    """True where the foot SHOULD be mid-stance (airborne here = violation)."""
    d = _clamp(_col2(duty_star(v)), PRG_D_LO, PRG_D_HI)
    return (phi >= STANCE_MID_LO * d) & (phi < STANCE_MID_HI * d)


# ---------------------------------------------------------------------------
# PRG state machine
# ---------------------------------------------------------------------------
class PRGState:
    """Stateful PRG gait clock for N parallel environments.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments N.
    device : optional
        torch device (ignored on the numpy backend).
    dtype : optional
        Floating dtype for the phase state.  Defaults to float64 so that the
        v71 degradation gate (atol=1e-6 over thousands of integration steps)
        holds regardless of platform; pass float32 for production speed.
    reset_mode : {"hard","soft","none"}, optional
        Per-instance override of PRG_RESET_MODE.
    freeze_alpha : bool, optional
        Per-instance override of PRG_FREEZE_ALPHA: when True the alpha
        curriculum is pinned to 0 (exact v71 blending, used by the B0
        degradation gate).
    backend : {"torch","numpy","auto"}, optional
        Array backend override.  "auto" (default) picks torch when
        importable, else numpy.

    State (public attributes)
    -------------------------
    phi : (N, 2) float        per-foot phase in [0,1); column 0 = left, 1 = right
    contact_prev : (N, 2) bool  last-step contact state (touchdown edge memory)
    t_ep : (N,) float         episode time [s]
    f_eff : (N,) float        last-step effective step frequency [Hz] (property)

    Notes
    -----
    Reset semantics: a touchdown (contact rising edge) on a foot re-anchors
    THAT foot's phase to 0 = stance start, matching the v71 convention where
    phi = 0 at touchdown.  Resets only fire when the phase at touchdown lies
    inside the sync window [1-gamma, 1) U [0, PRG_SYNC_HEAD).
    """

    def __init__(
        self,
        num_envs,
        device=None,
        dtype=None,
        reset_mode=None,
        freeze_alpha=None,
        backend=None,
    ):
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")

        backend = backend or PRG_BACKEND or "auto"
        if backend == "auto":
            backend = "torch" if _torch is not None else "numpy"
        if backend not in ("torch", "numpy"):
            raise ValueError("backend must be 'torch', 'numpy' or 'auto'")
        if backend == "torch" and _torch is None:
            raise RuntimeError("backend='torch' but torch is not importable")
        self.backend = backend

        if reset_mode is None:
            reset_mode = PRG_RESET_MODE
        if reset_mode not in ("hard", "soft", "none"):
            raise ValueError("reset_mode must be one of {'hard','soft','none'}")
        self.reset_mode = reset_mode

        if freeze_alpha is None:
            freeze_alpha = PRG_FREEZE_ALPHA
        self.freeze_alpha = bool(freeze_alpha)

        n = self.num_envs
        if backend == "torch":
            self._lib = _torch
            self._fdtype = _torch.promote_types(
                dtype if dtype is not None else _torch.float64,
                _torch.float32,
            )
            dev = device if device is not None else "cpu"
            self._dev = dev
            self.phi = _torch.zeros((n, 2), dtype=self._fdtype, device=dev)
            self.contact_prev = _torch.zeros((n, 2), dtype=_torch.bool, device=dev)
            self.t_ep = _torch.zeros((n,), dtype=self._fdtype, device=dev)
            self._f_eff = _torch.zeros((n,), dtype=self._fdtype, device=dev)
            self._alpha = _torch.zeros((n,), dtype=self._fdtype, device=dev)
        else:
            self._lib = np
            self._fdtype = np.dtype(dtype if dtype is not None else np.float64)
            if self._fdtype.kind != "f":
                raise ValueError("dtype must be a floating dtype")
            self._dev = None
            self.phi = np.zeros((n, 2), dtype=self._fdtype)
            self.contact_prev = np.zeros((n, 2), dtype=bool)
            self.t_ep = np.zeros((n,), dtype=self._fdtype)
            self._f_eff = np.zeros((n,), dtype=self._fdtype)
            self._alpha = np.zeros((n,), dtype=self._fdtype)

        self.reset()

    # -- input coercion ----------------------------------------------------
    def _to_float(self, x, broadcast_shape=None):
        """Coerce scalar/list/array input to this backend's float tensor/array."""
        if self.backend == "torch":
            t = x if _is_torch(x) else _torch.as_tensor(x)
            t = t.to(device=self._dev, dtype=self._fdtype)
            if broadcast_shape is not None and t.dim() == 0:
                t = _torch.full(
                    broadcast_shape, float(t), dtype=self._fdtype,
                    device=self._dev,
                )
            return t
        arr = np.asarray(x)
        if arr.shape == () and broadcast_shape is not None:
            arr = np.full(broadcast_shape, arr.item())
        return arr.astype(self._fdtype)

    def _to_bool(self, x):
        if self.backend == "torch":
            t = x if _is_torch(x) else _torch.as_tensor(x)
            return t.to(device=self._dev, dtype=_torch.bool)
        return np.asarray(x).astype(bool)

    # -- lifecycle ---------------------------------------------------------
    def reset(self, env_ids=None):
        """Re-anchor the given envs to the episode start state.

        phi[env_ids, 0] = 0.0 (left foot touchdown), phi[env_ids, 1] = 0.5
        (right foot antiphase), contact_prev = False, t_ep = 0.
        ``env_ids=None`` resets every environment.
        """
        if env_ids is None:
            self.phi[:, 0] = 0.0
            self.phi[:, 1] = 0.5
            self.contact_prev[:] = False
            self.t_ep[:] = 0.0
            self._f_eff[:] = 0.0
            self._alpha[:] = 0.0
            return
        ids = self._to_index(env_ids)
        self.phi[ids, 0] = 0.0
        self.phi[ids, 1] = 0.5
        self.contact_prev[ids] = False
        self.t_ep[ids] = 0.0
        self._f_eff[ids] = 0.0
        self._alpha[ids] = 0.0

    def _to_index(self, env_ids):
        if isinstance(env_ids, slice):
            return env_ids
        if self.backend == "torch":
            if _is_torch(env_ids):
                return env_ids.to(device=self._dev, dtype=_torch.long)
            return _torch.as_tensor(
                list(env_ids), dtype=_torch.long, device=self._dev
            )
        return env_ids  # numpy handles list / slice / ndarray uniformly

    # -- core step ----------------------------------------------------------
    def step(self, dt, v_cmd, contact, delta_f=None):
        """Advance the clock one control step.

        Parameters
        ----------
        dt : float                    control step duration [s]
        v_cmd : (N,) array-like       commanded planar speed [m/s]
        contact : (N, 2) array-like   current foot contact flags (bool)
        delta_f : (N,) array-like or None
            policy frequency modulation in [-1, 1] (clamped); None = v71 mode

        Order of operations (per spec):
          a) f_cmd = 1/T*(v_cmd); alpha from t_ep curriculum;
             f_pol = f_cmd*(1 + PRG_FMOD_SCALE*delta_f) (or f_cmd if None);
             f_eff = (1-alpha)*f_cmd + alpha*f_pol
          b) phi += f_eff*dt   (both feet share f_eff)
          c) touchdown = contact & ~contact_prev; hard: phi[td & sync] = 0;
             soft: phi[td & sync] += clip(-beta*dt*wrap_err, +/- clamp)
          d) phi %= 1; contact_prev = contact; t_ep += dt
        """
        n = self.num_envs
        dt = float(dt)
        v = self._to_float(v_cmd, broadcast_shape=(n,))
        if v.shape != (n,):
            raise ValueError("v_cmd must have shape (%d,)" % n)
        c = self._to_bool(contact)
        if c.shape != (n, 2):
            raise ValueError("contact must have shape (%d, 2)" % n)

        # (a) frequency blend -------------------------------------------------
        f_cmd = 1.0 / t_star(v)
        if self.freeze_alpha:
            alpha = self._lib.zeros_like(f_cmd)
        else:
            alpha = _clamp(self.t_ep / PRG_ALPHA_RAMP_S, 0.0, 1.0) * PRG_ALPHA_MAX
        if delta_f is None:
            f_pol = f_cmd
        else:
            df = self._to_float(delta_f, broadcast_shape=(n,))
            if df.shape != (n,):
                raise ValueError("delta_f must have shape (%d,)" % n)
            df = _clamp(df, -1.0, 1.0)
            f_pol = f_cmd * (1.0 + PRG_FMOD_SCALE * df)
        f_eff = (1.0 - alpha) * f_cmd + alpha * f_pol

        # (b) advance phase (both feet share f_eff) ---------------------------
        self.phi = self.phi + f_eff[:, None] * dt

        # (c) touchdown-triggered reset ---------------------------------------
        touchdown = c & _logical_not(self.contact_prev)
        if self.reset_mode != "none" and bool(touchdown.any()):
            q = self.phi % 1.0
            in_sync = (q >= (1.0 - PRG_SYNC_WINDOW)) | (q < PRG_SYNC_HEAD)
            mask = touchdown & in_sync
            if self.reset_mode == "hard" and bool(mask.any()):
                self.phi[mask] = 0.0  # re-anchor this foot at stance start
            elif self.reset_mode == "soft" and bool(mask.any()):
                # signed wrap error of phi around 0 on the unit cycle:
                #   err in [-0.5, 0.5); 0.97 -> -0.03 (slightly behind 0)
                err = _where(q < 0.5, q, q - 1.0)
                corr = _clamp(
                    -PRG_RESET_GAIN * dt * err,
                    -PRG_SOFT_CLAMP * dt * PRG_RESET_GAIN,
                    PRG_SOFT_CLAMP * dt * PRG_RESET_GAIN,
                )
                self.phi = self.phi + _where(
                    mask, corr, self._lib.zeros_like(corr))

        # (d) wrap + bookkeeping ----------------------------------------------
        self.phi = self.phi % 1.0
        self.contact_prev = c
        self.t_ep = self.t_ep + dt
        self._f_eff = f_eff
        self._alpha = alpha
        return self.phi

    # -- outputs -------------------------------------------------------------
    def clock_obs(self):
        """(N, 4) [sin 2pi phiL, sin 2pi phiR, cos 2pi phiL, cos 2pi phiR].

        Column order matches the ACTUAL v71 implementation
        (mdp/gait_clock.py::clock_obs = cat([sin(2pi phi), cos(2pi phi)]) with
        phi of shape (N,2), i.e. both sines first, then both cosines) so the
        same network input layout and the sim2sim/deploy twin carry over
        unchanged.  (The task spec text said [sinL,cosL,sinR,cosR]; lead
        corrected to the code-authoritative order at merge time.)
        """
        ang = 2.0 * np.pi * self.phi
        sin, cos = (self._lib.sin, self._lib.cos)
        return _stack(
            [sin(ang[:, 0]), sin(ang[:, 1]), cos(ang[:, 0]), cos(ang[:, 1])],
            axis_or_dim=-1,
        )

    def targets(self, v_cmd):
        """Per-env gait targets at the CURRENT phi (for rewards / metrics).

        Returns a dict:
          "duty"   (N,)   d*(v)   stance duty factor
          "h"      (N,)   h*(v)   swing apex height [m]
          "stance" (N,2)  bool    phi < duty          (v71 stance window)
          "swing"  (N,2)  bool    phi >= duty
        """
        n = self.num_envs
        v = self._to_float(v_cmd, broadcast_shape=(n,))
        duty = duty_star(v)
        h = h_star(v)
        stance = self.phi < duty[:, None]
        swing = _logical_not(stance)
        return {"duty": duty, "h": h, "stance": stance, "swing": swing}

    # -- properties -----------------------------------------------------------
    @property
    def alpha(self):
        """(N,) current alpha curriculum value (0 when frozen)."""
        return self._alpha

    @property
    def f_eff(self):
        """(N,) effective step frequency of the last step() [Hz]."""
        return self._f_eff

    def __repr__(self):
        return (
            "PRGState(num_envs=%d, backend=%s, reset_mode=%r, freeze_alpha=%r,"
            " phi=%s)" % (
                self.num_envs,
                self.backend,
                self.reset_mode,
                self.freeze_alpha,
                getattr(self.phi, "shape", None),
            )
        )
