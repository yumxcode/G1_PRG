# Notation and Symbol↔Code Mapping

Companion to `method_section.md`. Middle column: equation number in the method section. Right column:
the authoritative location in `prg_clock.py` (task-input version, post lead-merge commit 8fcc4da) —
every formula in the method section is required to match the code line-for-line.

## Clock state and phases

| Symbol | Def. | Meaning / units | Code |
|---|---|---|---|
| $\phi_i$, $i \in \{L, R\}$ | — | per-foot gait phase in $[0,1)$; $\phi=0$ ≡ touchdown (stance onset); column 0 = left, 1 = right | `PRGState.phi` (N,2) |
| $t^{\mathrm{ep}}$ | — | episode time [s] | `PRGState.t_ep` |
| $\bar c_{i}$ | — | previous-step contact flag (edge memory) | `PRGState.contact_prev` (N,2) bool |
| $c_{i,t}$ | — | measured contact flag of foot $i$ (ground truth in sim; force sensor / estimator on hardware) | `step(contact)` arg |
| $\mathrm{TD}_{i,t}$ | (9) | touchdown event = rising edge $c \wedge \neg \bar c$ | `touchdown = c & ~contact_prev` |
| $g_t$ | (1) | clock observation, 4-dim; **column order v71-authoritative: [sinL, sinR, cosL, cosR]** | `PRGState.clock_obs()` |
| $q_{i,t}$ | (10) | wrapped phase used for gating, $q = \phi \bmod 1$ | `q = self.phi % 1.0` |
| $\Delta t$ | — | control step (50 Hz ⇒ 0.02 s) | `step(dt)` arg |

## Nominal gait curves (v71, shared by PRG)

| Symbol | Def. | Meaning / units | Code |
|---|---|---|---|
| $v$ / $v^{\mathrm{cmd}}$ | — | planar command speed [m/s] | `v_cmd` (N,) |
| $T^\*(v)$ | (2) | cycle period $\operatorname{clip}(1.46-0.36v,\,0.70,\,1.75)$ [s] | `t_star`, `PRG_T_*` |
| $f_{\mathrm{cmd}}(v)$ | §3.2 | $1/T^\*(v)$ [Hz] | `f_cmd_star` |
| $d^\*(v)$ | (3) | duty factor $\operatorname{clip}(0.68-0.07v,\,0.55,\,0.70)$ | `duty_star`, `PRG_D_*` |
| $h^\*(v)$ | (4) | swing apex $\operatorname{clip}(0.025+0.025v,\,0.025,\,0.075)$ [m] | `h_star`, `PRG_H_*` |
| $\mathrm{frac}(\phi_i)$ | §3.2 | swing progress $(\phi_i - d^\*)/(1 - d^\*)$ | `frac` in `target_height` |
| $z^\*(\phi_i; v)$ | (5) | per-foot sole-height target bell [m]; 0 on stance | `target_height(phi, v)` |

## PRG hybrid dynamics

| Symbol | Def. | Meaning / units | Code |
|---|---|---|---|
| $\delta f$ | (6) | policy frequency output, clipped to $[-1,1]$; `None` ⇒ v71 mode | `step(delta_f)` |
| $\kappa$ | (6) | frequency modulation scale = 0.5 (±50% band) | `PRG_FMOD_SCALE = 0.5` |
| $f^{\mathrm{pol}}$ | (6) | policy frequency $f_{\mathrm{cmd}}(1 + \kappa\,\delta f)$ | `f_pol` |
| $\alpha_t$ | (8) | blend curriculum $\operatorname{clip}(t^{\mathrm{ep}}/T_{\mathrm{ramp}},0,1)\cdot\alpha_{\max}$ | `alpha = clamp(t_ep/PRG_ALPHA_RAMP_S,0,1)*PRG_ALPHA_MAX` |
| $\alpha_{\max}$ | (8) | max blend weight = 0.5 | `PRG_ALPHA_MAX = 0.5` |
| $T_{\mathrm{ramp}}$ | (8) | curriculum ramp = 10 s | `PRG_ALPHA_RAMP_S = 10.0` |
| `freeze_alpha` | §3.3.1 | switch: α ≡ 0 whole episode (exact v71 degradation) | `PRG_FREEZE_ALPHA`, ctor arg |
| $f^{\mathrm{eff}}$ | (6) | $(1-\alpha)f_{\mathrm{cmd}} + \alpha f^{\mathrm{pol}}$ [Hz]; both feet share it | `f_eff`; property `PRGState.f_eff` |
| $\gamma$ | (10) | sync window tail = 0.25 (window $[1-\gamma,1)$) | `PRG_SYNC_WINDOW = 0.25` |
| $\gamma_{\mathrm{head}}$ | (10) | sync window head = 0.05 (window $[0,\gamma_{\mathrm{head}})$) | `PRG_SYNC_HEAD = 0.05` |
| $\mathcal{R}_{\mathrm{hard}}$ | (11) | hard reset: $\phi_i \leftarrow 0$ | `phi[mask] = 0.0` |
| $\mathcal{R}_{\mathrm{soft}}$ | (12) | soft reset: $\phi_i + \operatorname{clip}(-\beta\Delta t\, e(\phi_i),\,\pm c_{\max}\beta\Delta t)$ | `corr = clip(-PRG_RESET_GAIN*dt*err, ±PRG_SOFT_CLAMP*dt*PRG_RESET_GAIN)` |
| $e(q)$ | (12) | signed wrap error, $e \in [-0.5, 0.5)$ | `err = where(q < 0.5, q, q - 1.0)` |
| $\beta$ | (12) | soft reset gain = 2.0 s⁻¹ | `PRG_RESET_GAIN = 2.0` |
| $c_{\max}$ | (12) | soft correction clamp = 0.15 cycles | `PRG_SOFT_CLAMP = 0.15` |
| `reset_mode` | (13) | {hard, soft, none}; none ⇒ v71 open loop | `PRG_RESET_MODE`, ctor arg |
| (a)–(d) | (13) | fixed step order: blend → advance → reset → wrap/bookkeep | `PRGState.step` body order |
| episode init | §3.3.3 | $\phi_L=0$, $\phi_R=0.5$, $\bar c=0$, $t^{\mathrm{ep}}=0$ | `PRGState.reset()` |

## Reward-side phase targets (contact–clock co-synchronization)

| Symbol | Def. | Meaning | Code |
|---|---|---|---|
| $W^{\mathrm{sw}}_i$ | §3.4 | mid-swing window $[d^\*+0.2(1-d^\*),\ d^\*+0.8(1-d^\*))$ | `swing_mid_mask`, `SWING_MID_LO/HI = 0.2/0.8` |
| $W^{\mathrm{st}}_i$ | §3.4 | mid-stance window $[0.15\,d^\*,\ 0.85\,d^\*)$ | `stance_mid_mask`, `STANCE_MID_LO/HI = 0.15/0.85` |
| stance / swing masks | §3.4 | stance $\Leftrightarrow \phi_i < d^\*$ | `PRGState.targets(v)["stance"/"swing"]` |

## GaitTiming metrics (method §3.6)

| Symbol | Def. | Meaning | Metric |
|---|---|---|---|
| $\tau^i_k$ | (14) | measured touchdown times of foot $i$ | contact segmentation |
| $e^{\mathrm{td}}_{i,k}$ | (14) | touchdown anchoring error $\operatorname{wrap}(\phi_i(\tau^i_k))$ [cycles] | sync error (i) |
| $\bar e^{\mathrm{td}}_i$, $\overline{\|e^{\mathrm{td}}\|}$ | (14) | circular mean (lead/lag) and MAE | sync error (i) |
| $\tilde\phi_i(t)$, $\rho_i(t)$ | §3.6(i) | contact-derived phase; continuous drift $\operatorname{wrap}(\phi - \tilde\phi)$ | sync error (i) |
| $\hat d_{i,k}$, $\Delta d_{i,k}$ | (15) | measured per-stride duty; retention error vs. $d^\*$ | duty retention (ii) |
| $\mathrm{Fr}$ | (16 ctx) | Froude number $\bar v^2/(gL_{\mathrm{leg}})$, $L_{\mathrm{leg}}$ = \todo{} | ground-time-structure gate |
| $[v_-, v_+]$, $w_{\mathrm{tr}}$, $v_{\mathrm{tr}}$, $\Delta v_{\mathrm{hys}}$ | §3.6(iii) | transition band, width, center, hysteresis (changepoints of $\hat f(v)$, $\hat d(v)$ fits) | transition band (iii) |
| $\mathrm{SR}(\tau)$, $\mathrm{DM}$ | (16) | success rate under latency τ; delay margin $\sup\{\tau: \mathrm{SR} \ge 0.9\}$ | delay margin (iv) |

## Environment / deployment quantities

| Symbol | Meaning | Value / status |
|---|---|---|
| $a$, $q_{\mathrm{cmd}}$ | action, joint command $q_{\mathrm{default}} + 0.25a$ | 29-dim |
| $o_t$ | proprio obs (96/frame × 3 frames) + 4 clock ch. | 300-dim input |
| $\delta f$ action augmentation | policy head producing $\delta f$ | **v2 extension**: clock side implemented (`step(delta_f)`); env-side action wiring is a reserved switch, not active in v1 |

## Terminology (fixed usage across the paper)

- **gait clock** — the phase signal $\phi$ + its observation encoding $g$ (never "CPG"; PRG is not an
  oscillator network).
- **phase reset / re-anchoring** — the jump map (Eqs. 11–12); "reset" alone always means this, never
  environment reset (use "episode reset" explicitly for the latter).
- **exogenous / endogenous / hybrid clock** — timing set by command only / by policy only / by
  commanded flow + contact events (+ bounded policy rate modulation).
- **touchdown** — contact rising edge (Eq. 9); **stance onset** — its intended phase meaning ($\phi=0$).
- **sync window** — the gating set of Eq. (10); not to be confused with the reward hinge windows.
- **v71** — the internal open-loop baseline clock (pure time function); **B0** — the training arm that
  reproduces v71 exactly.
- **sim2sim** — Isaac (training) → MuJoCo (external evaluation) transfer of the same policy.
