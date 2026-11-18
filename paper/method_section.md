# Method: Contact-Triggered Adaptive Gait Clock for Humanoid Locomotion

> **Draft status (v0.1)** — Method section of the G1 paper, contribution C1 = PRG (Phase-Reset Gait clock).
> All experiment-dependent numbers are placeholders `\todo{}`. All equations are cross-checked line-by-line
> against the authoritative implementation `prg_clock.py` (commit at task input; see Appendix A/B and
> `notation.md` for the symbol↔code mapping). Citations use the verified arXiv ids from the innovation
> survey (132 sources, ids batch-verified 2026-11-17).

---

## 3.1 Problem Formulation

**Locomotion POMDP.** We model humanoid locomotion as a partially observable Markov decision process
(POMDP) $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, T, \Omega, r, \gamma)$, where $\mathcal{S}$
is the physical state of the Unitree G1 (29 revolute joints, base pose and velocities) and its
environment, $\mathcal{A}$ is the 29-dimensional joint-position-offset action space, and the policy is
trained with PPO on GPU-parallel simulation (IsaacLab/rsl_rl, 4096 environments).

The agent does not observe $\mathcal{S}$ directly but an observation
$o_t \in \mathcal{O}$ consisting of proprioceptive channels (joint positions/velocities, base angular
velocity, projected gravity, last action), each stacked over 3 frames, plus the velocity command
$v^{\mathrm{cmd}}_t$.

**Gait-clock conditioning.** Following the gait-clock paradigm established for quadrupeds by
Walk-These-Ways [arXiv:2212.03238] and adopted for humanoids by HugWBC [arXiv:2502.03206], the policy is
additionally conditioned on a gait clock: a per-foot phase variable $\phi_{i,t} \in [0,1)$,
$i \in \{L, R\}$, presented to the policy through the encoding

$$
g_t \;=\; \big[\sin 2\pi\phi_{L,t},\; \sin 2\pi\phi_{R,t},\; \cos 2\pi\phi_{L,t},\; \cos 2\pi\phi_{R,t}\big]
\in \mathbb{R}^4,
\tag{1}
$$

so the policy is $\pi_\theta\!\left(a_t \,\middle|\, o_t,\, v^{\mathrm{cmd}}_t,\, g_t\right)$. The phase
$\phi_i = 0$ is defined as the *stance onset* (touchdown) of foot $i$; the stance window of a cycle is
$\phi_i \in [0, d^\*)$ and the swing window is $\phi_i \in [d^\*, 1)$, where $d^\*$ is the commanded duty
factor defined below.

Our internal baseline (v71) provides the empirical motivation for conditioning on an *exogenous* clock
rather than letting the policy time its steps endogenously: across 18 falsified training rounds
(V64–V70) on the same stack, no reward-side anchoring overturned the low-step-frequency failure mode —
whenever the policy alone chooses *when* to step, micro-stepping remains the locally easiest balance
solution. Making the timing exogenous (the policy observes the clock and is scored against per-foot
phase targets) was the first intervention that produced true stepping at the commanded cadence. This
replicates at humanoid scale the qualitative finding that timing structure is better *conditioned on*
than *shaped through rewards* [arXiv:2212.03238, arXiv:2502.03206].

**Limitation of the open-loop clock.** The v71 clock is a pure function of episode time and command,
$\phi_{L,t} = (t / T^\*(v^{\mathrm{cmd}})) \bmod 1$, $\phi_{R,t} = (\phi_{L,t} + 0.5) \bmod 1$: it is
open-loop with respect to the physical robot. Any mismatch between the clock's phase and the actual
contact pattern — caused by early/late touchdowns under perturbation, terrain interaction, or
simulator-level contact differences — accumulates without correction, and the reward then scores the
robot against a rhythm it can no longer physically realize. Prior work either keeps the clock fully
commanded (WTW [arXiv:2212.03238], HugWBC [arXiv:2502.03206]) or moves timing fully into the policy
(CRA-LoCo [arXiv:2509.12858], HRP-5P [arXiv:2504.13619], [arXiv:2512.07464]); the first lacks a
feedback channel from contacts, the second reopens the endogenous-timing failure mode and removes the
user-commandable timing interface (see Table 1).

**Goal.** We seek a *hybrid* clock that (i) keeps an exogenous, speed-scheduled nominal rhythm (so
timing remains commandable and the micro-stepping attractor stays closed), (ii) admits contact events
as *phase feedback* that re-anchors the rhythm to what the robot physically did, and (iii) optionally
lets the policy modulate the rhythm's *rate* within a bounded band, blended in through a curriculum.
This is the Phase-Reset Gait clock (PRG).

---

## 3.2 Speed-Scheduled Nominal Gait Curves (v71 baseline)

The nominal rhythm is scheduled by the planar command speed $v \equiv v^{\mathrm{cmd}}$
(units m/s) through three clipped-affine curves (v71, shared verbatim by PRG):

$$
T^\*(v) = \operatorname{clip}\!\big(1.46 - 0.36\,v,\; 0.70,\; 1.75\big)
\tag{2}
$$

$$
d^\*(v) = \operatorname{clip}\!\big(0.68 - 0.07\,v,\; 0.55,\; 0.70\big)
\tag{3}
$$

$$
h^\*(v) = \operatorname{clip}\!\big(0.025 + 0.025\,v,\; 0.025,\; 0.075\big)
\tag{4}
$$

where $T^\*$ is the single-foot cycle period [s], $d^\*$ the stance duty factor, and $h^\*$ the swing
apex height [m]. The commanded step frequency is $f_{\mathrm{cmd}}(v) = 1/T^\*(v)$ [Hz]. The phase
convention: $\phi_i \in [0, d^\*)$ = stance, $\phi_i \in [d^\*, 1)$ = swing; the swing progress fraction
is $\mathrm{frac}(\phi_i) = (\phi_i - d^\*)/(1 - d^\*)$.

The desired lowest-sole-point height profile per foot is a $C^0$ bell that is zero through stance and
peaks at $h^\*$ mid-swing:

$$
z^\*(\phi_i; v) \;=\;
\begin{cases}
h^\*(v)\,\sin\!\big(\pi \cdot \operatorname{clip}(\mathrm{frac}(\phi_i),\, 0,\, 1)\big), & \phi_i \ge d^\*(v) \\
0, & \text{otherwise.}
\end{cases}
\tag{5}
$$

---

## 3.3 PRG: Hybrid Clock Dynamics

PRG upgrades (2)–(5) from a pure time function to a *stateful hybrid system*: continuous phase-flow
dynamics interrupted by event-triggered jumps. Its state is
$x_t = (\phi_{L,t}, \phi_{R,t},\, \bar{c}_{t-1},\, t^{\mathrm{ep}}_t) \in [0,1)^2 \times \{0,1\}^2 \times \mathbb{R}_{\ge 0}$,
comprising the two foot phases, the previous-step contact flags $\bar{c}$ (edge memory), and episode
time $t^{\mathrm{ep}}$.

### 3.3.1 Continuous flow: blended frequency with $\alpha$ curriculum

Between events, both feet share a common phase rate (the two phases remain exactly anti-phase outside
resets). With the policy frequency output $\delta f_t \in [-1, 1]$ (clipped), the effective frequency
is a convex blend of the commanded and policy frequencies:

$$
f^{\mathrm{eff}}_t \;=\; (1-\alpha_t)\, f_{\mathrm{cmd}}(v^{\mathrm{cmd}}_t)
\;+\; \alpha_t\, f_{\mathrm{cmd}}(v^{\mathrm{cmd}}_t)\big(1 + \kappa\, \delta f_t\big),
\tag{6}
$$

with modulation scale $\kappa = 0.5$ (so the policy can shift the cadence by at most $\pm 50\%$).
Equivalently, in angular form the clock rate is

$$
\dot\varphi \;=\; 2\pi f^{\mathrm{eff}} \;=\; (1-\alpha)\,2\pi f_{\mathrm{cmd}} \;+\; \alpha\, 2\pi f_{\mathrm{cmd}}\big(1 + 0.5\,\delta f\big)
\quad \text{[rad/s]},
\tag{7}
$$

where $\varphi = 2\pi\phi$ denotes the angular phase (the cycle phase $\phi$ used everywhere else
advances at $f^{\mathrm{eff}}$ cycles/s), and the per-step discrete update is
$\phi_{i,t+1} = (\phi_{i,t} + f^{\mathrm{eff}}_t \Delta t) \bmod 1$ with control step
$\Delta t = 0.02$ s (50 Hz). Throughout, $\operatorname{wrap}(q) = q - \lfloor q + 0.5 \rfloor \in [-0.5, 0.5)$
denotes the signed circular wrap error on the unit cycle.

The blend weight follows a **linear episode-time curriculum**:

$$
\alpha_t \;=\; \operatorname{clip}\!\left(\frac{t^{\mathrm{ep}}_t}{T_{\mathrm{ramp}}},\, 0,\, 1\right)\cdot \alpha_{\max},
\qquad T_{\mathrm{ramp}} = 10\,\text{s},\quad \alpha_{\max} = 0.5,
\tag{8}
$$

i.e. $\alpha = \min(t^{\mathrm{ep}}/10,\, 1)\cdot 0.5$. During the first 10 s of each episode the clock is
purely commanded; the policy's frequency authority grows linearly to at most $50\%$ blend and never
fully replaces the command. Two degenerate switches are exact by construction: $\delta f = \varnothing$
(the policy outputs no modulation) gives $f^{\mathrm{pol}} = f_{\mathrm{cmd}}$; freezing
$\alpha \equiv 0$ recovers the v71 clock bit-for-bit (verified to circular error $\le 10^{-6}$ cycles
over $\ge$1500-step rollouts in the module's regression suite).

> **v2 extension note (implementation status).** The clock-side mechanism (6)–(8) is fully implemented
> in `prg_clock.py` (`step(dt, v_cmd, contact, delta_f)`). The *policy action-space augmentation* that
> produces $\delta f$ (appending one output head to the network and wiring it into the environment) is
> implemented behind a reserved environment switch and is **not active in the v1 experiments**; v1
> trains PRG with contact-triggered reset only ($\delta f = \varnothing$), and the frequency-modulation
> arm is pre-registered as a v2 extension (ablation A2 in `ablation_plan.md`).

### 3.3.2 Event set: touchdown detection and the synchronization window

Let $c_{i,t} \in \{0,1\}$ be the measured contact state of foot $i$. A **touchdown event** is a rising
edge:

$$
\mathrm{TD}_{i,t} \;=\; c_{i,t} \wedge \neg\, \bar{c}_{i,t-1},
\tag{9}
$$

where $\bar{c}_{i,t-1}$ is the edge memory carried in the clock state. Not every touchdown triggers a
reset: a reset fires only if the clock phase is already *close to* the stance onset, i.e. inside the
**synchronization window**

$$
q_{i,t} = \phi_{i,t} \bmod 1 \;\in\; \underbrace{[1-\gamma,\, 1) \;\cup\; [0,\, \gamma_{\mathrm{head}})}_{\text{sync window}},\qquad
\gamma = 0.25,\;\; \gamma_{\mathrm{head}} = 0.05.
\tag{10}
$$

The window makes the reset an *anchoring correction*, not a disturbance: contacts that land mid-cycle
(e.g. a stumble catch, a perturbation-induced emergency step) do not yank the rhythm; only touchdowns
that occur near where stance *should* begin pull the phase fully onto the contact event. The asymmetric
head ($\gamma_{\mathrm{head}} = 0.05 \ll \gamma$) suppresses double-counting of a single touchdown that
straddles the wrap-around.

### 3.3.3 Jump map: hard and soft reset

On a gated touchdown, the phase of the triggering foot is re-anchored by a reset operator
$\mathcal{R}$. Two strengths are implemented:

$$
\mathcal{R}_{\mathrm{hard}}(\phi_i) \;=\; 0
\tag{11}
$$

$$
\mathcal{R}_{\mathrm{soft}}(\phi_i) \;=\; \phi_i + \operatorname{clip}\!\Big(-\beta\, \Delta t\, e(\phi_i),\; -c_{\max}\beta\Delta t,\; +c_{\max}\beta\Delta t\Big),
\qquad e(q) = \begin{cases} q, & q < 0.5 \\ q - 1, & \text{else,} \end{cases}
\tag{12}
$$

with soft gain $\beta = 2.0\ \mathrm{s^{-1}}$ and correction clamp $c_{\max} = 0.15$ cycles, so the
soft reset pulls the phase toward 0 (on the unit circle, $e(q) \in [-0.5, 0.5)$ is the signed wrap
error) at a bounded rate of at most $c_{\max}\beta\Delta t$ cycles per step. A third mode
$\mathcal{R} = \mathrm{none}$ disables jumps entirely (v71 equivalent).

**Full hybrid update.** One control step of PRG executes in the fixed order (a)–(d):

$$
\begin{aligned}
\text{(a)}\quad & f^{\mathrm{eff}}_t \leftarrow \text{Eq. (6) with } \alpha_t \text{ from Eq. (8)};\\
\text{(b)}\quad & \phi_{i} \leftarrow \phi_{i} + f^{\mathrm{eff}}_t\,\Delta t \quad \forall i \in \{L,R\};\\
\text{(c)}\quad & \phi_{i} \leftarrow \mathcal{R}(\phi_{i}) \quad \forall i:\ \mathrm{TD}_{i,t} \wedge \big(q_{i,t} \in \text{sync window}\big);\\
\text{(d)}\quad & \phi_{i} \leftarrow \phi_{i} \bmod 1,\quad \bar{c}_{i} \leftarrow c_{i},\quad t^{\mathrm{ep}} \leftarrow t^{\mathrm{ep}} + \Delta t.
\end{aligned}
\tag{13}
$$

Episode initialization: $\phi_L = 0$ (left touchdown), $\phi_R = 0.5$ (anti-phase),
$\bar c = \mathbf{0}$, $t^{\mathrm{ep}} = 0$. Note the anti-phase offset is *state*, not flow: both
feet share the same $f^{\mathrm{eff}}$, and only resets act per-foot, so a per-foot asymmetry injected
by a reset persists (bounded by the window) until the next touchdown — this is exactly the mechanism by
which a perturbation on one foot shifts that foot's rhythm without destroying the global anti-phase
structure.

**Deployment consistency.** The clock state $(\phi, \bar c, t^{\mathrm{ep}})$ is a function of
robot-side signals only — contact flags, command, and time — so the sim2sim twin (MuJoCo rollout) and
the deployment runtime reconstruct the clock identically from the same inputs, with no simulator-private
information. Contact flags are ground truth in both simulators; on hardware they come from the foot
force sensors / contact estimator, which is the only sensing assumption PRG adds over the v71 baseline.
\todo{hardware contact-estimator note if a real-robot section is added}

---

## 3.4 Reward Synchronization Design

The reward terms that shape gait timing are *scored against the adaptive clock phase* — the same
$\phi$ the policy observes and the same $\phi$ the contacts re-anchor. This closes a three-way
synchronization loop: **clock → policy** (phase observation, Eq. 1), **policy → robot → contacts**,
**contacts → clock** (reset, Eq. 13c). The per-foot phase targets are the v71 formulas evaluated at
the PRG phase:

- **Swing height tracking** — the lowest point of each sole tracks the bell profile (5):
  $r_{\mathrm{h},i} = \exp\!\big(-\|z^{\mathrm{sole}_i} - z^\*(\phi_i; v)\|^2 / \sigma_h^2\big)$, with
  band width $\sigma_h$ = \todo{}.
- **Contact-timing penalties (hinge windows)** — contact during the *mid-swing* window is penalized,
  and lack of contact during the *mid-stance* window is penalized:
  $W^{\mathrm{sw}}_i = [d^\* + 0.2\,(1-d^\*),\ d^\* + 0.8\,(1-d^\*))$ and
  $W^{\mathrm{st}}_i = [0.15\,d^\*,\ 0.85\,d^\*)$; i.e.
  $r_{\mathrm{csw},i} = -\lambda_{\mathrm{csw}} \cdot \mathbb{1}[c_i \wedge \phi_i \in W^{\mathrm{sw}}_i]$,
  $r_{\mathrm{cst},i} = -\lambda_{\mathrm{cst}} \cdot \mathbb{1}[\neg c_i \wedge \phi_i \in W^{\mathrm{st}}_i]$,
  with the hinge constants (0.2/0.8 and 0.15/0.85) identical to v71 so that the mid-phase windows are
  insensitive to boundary chatter.
- **Command gating** — all clock-conditioned reward terms are gated off below their own
  $\ v_{\min}$ \todo{}; the clock itself keeps running at all speeds (at $v=0$, $T^\* = 1.46$ s), and
  the policy learns to ignore the rotating clock when the command says stand.

Reward weights $\lambda_{\mathrm{csw}}, \lambda_{\mathrm{cst}}, \sigma_h$ and the remaining
task-level terms (velocity tracking, pose regularization, action rate, survival) follow the base
environment configuration: \todo{weight table from the frozen G1 env config}. The design principle we
commit to here — and test by ablation (A3) — is that *the clock is the single source of phase truth*:
no reward term references wall-clock time, so the reward structure cannot fight the reset mechanism.

---

## 3.5 Relation to Existing Clock Mechanisms

Table 1 positions PRG against the closest gait-clock mechanisms. Data are from the innovation survey
(132 sources; arXiv ids batch-verified 2026-11-17).

**Table 1 — Gait-clock mechanism comparison.**

| Mechanism | Platform | Clock type | Timing source | Contact feedback | Frequency authority | Gait spectrum & transitions |
|---|---|---|---|---|---|---|
| **PRG (ours)** | G1 (29-DoF) | hybrid: commanded flow + event jumps | speed-scheduled command $f_{\mathrm{cmd}}(v)$, Eq. (2) | **touchdown-triggered phase reset, hard/soft, sync-window gated** (Eqs. 10–12) | policy $\delta f$ blends $\pm 50\%$ via $\alpha$-curriculum, Eq. (6)–(8) | continuous speed spectrum; per-foot resynchronization under perturbation |
| v71 (internal baseline) | X1/G1 (29-DoF) | open-loop exogenous | pure time $\phi = (t/T^\*) \bmod 1$ | none | none (fixed schedule) | fixed walk schedule; no feedback |
| WTW [arXiv:2212.03238], CoRL 2022 | A1 quadruped | open-loop, commanded | gait params (freq, duty, offsets) as *commands* | none (fixed commanded clock) | user command only | rich command space; quadruped; clock not adaptive |
| HugWBC [arXiv:2502.03206], RSS 2025 | H1 (19-DoF) | open-loop, commanded | sinusoidal clock; $f \in [1.5, 3.5]$ Hz, phase offset, duty | none | user command only; duty fixed at 0.5 in practice; run only footnoted, no transition experiments | command interface on H1; no auto gait selection |
| CRA-LoCo [arXiv:2509.12858] | Adam Lite | endogenous, learned | policy outputs freq-update $\varphi(t\!+\!\Delta t) = \varphi(t) + (1+\alpha\delta_0)\Delta t$ + phase offsets; contact-pattern reward syncs feet | **no contact-triggered reset** | fully policy-owned (not user-commandable) | walking on rough terrain (real robot, 9 h/4090); no run, no speed-spectrum transitions |
| HRP-5P [arXiv:2504.13619], Humanoids 2024 | HRP-5P | modified observed clock | policy may edit the observed clock signal → adaptive cadence | rate-level only (no phase re-anchor) | policy-owned | walking only; clock benefit validated in sim |
| Stepping-phase policy [arXiv:2512.07464] | 1.65 m humanoid | endogenous | policy outputs global stepping phase | no explicit reset operator; no ablation | policy-owned | perception-driven; no user timing interface |

Reading of the table: existing mechanisms occupy the two open ends — fully commanded (v71, WTW,
HugWBC: commandable but blind to contacts) or fully endogenous (CRA-LoCo, HRP-5P, [arXiv:2512.07464]:
adaptive but not user-commandable, and in the endogenous family the timing reverts to being an
optimization variable, the failure mode that motivated exogenous clocks). PRG fills the middle: the
commanded schedule defines the rhythm; contact events *re-anchor* it; the policy modulates its rate
within a curriculum-bounded band. The nearest mechanism-level precedent for the reset itself is in
exoskeleton gait assistance [arXiv:2609.38824] (phase synchronization via reset-like correction,
phase MAE $95.55° \to 10.99°$) and quadruped adaptive gait timing [arXiv:2608.07328] (learnable
frequency in the action space); neither applies contact-triggered phase reset to humanoid RL
locomotion — the double-checked survey finding that defines this paper's contribution slot. MPC-side
evidence that timing adaptivity is a high-leverage degree of freedom for perturbation recovery
(+131% recoverable impulse with adaptive step frequency, [arXiv:2409.14342]) motivates evaluating PRG
under pushes (Section 5), and the speed-spectrum-without-timing-interface line
(RuN [arXiv:2509.20696], FORGE [arXiv:2610.37974], [arXiv:2605.18611]) defines the baselines our
clock interface must beat or match with less data and an inspectable, commandable timing signal.

---

## 3.6 GaitTiming: Evaluation Protocol for Timing Structure

We formalize the timing-quality metrics used throughout the experiments (Section 5). All are computed
from rollout logs containing, per control step: the clock phase $\phi_{i,t}$, the measured contact
flags $c_{i,t}$, the command $v^{\mathrm{cmd}}_t$, and the realized base velocity. Metrics are
evaluated in the training simulator (Isaac) and in the external simulator (MuJoCo sim2sim), after a
harness self-check (floor contact, timestep, action-pipeline latency) to exclude harness-induced
pseudo-gaps, following the protocol of [arXiv:2610.32277].

**(i) Phase–contact synchronization error.** Let $\tau^i_k$ be the measured touchdown times of foot
$i$ (rising edges of $c_i$), $k = 1, 2, \dots$. The **touchdown anchoring error** is the circular error
of the clock phase at touchdown:

$$
e^{\mathrm{td}}_{i,k} \;=\; \operatorname{wrap}\!\big(\phi_i(\tau^i_k)\big) \in [-0.5, 0.5)\ \text{[cycles]},
\qquad \bar e^{\mathrm{td}}_i = \operatorname{atan2}\!\Big(\textstyle\sum_k \sin 2\pi e^{\mathrm{td}}_{i,k},\ \sum_k \cos 2\pi e^{\mathrm{td}}_{i,k}\Big)\Big/ 2\pi,
\tag{14}
$$

reported as circular mean (phase lead/lag, signed) and as mean absolute error
$\overline{|e^{\mathrm{td}}|}$ (anisochrony). A **continuous drift measure** complements it: reconstruct
the contact-derived phase $\tilde\phi_i(t) = (t - \tau^i_k)/(\tau^i_{k+1} - \tau^i_k)$ for
$t \in [\tau^i_k, \tau^i_{k+1})$ and report $\rho_i(t) = \operatorname{wrap}(\phi_i(t) - \tilde\phi_i(t))$
(slope = frequency mismatch, curvature = within-cycle distortion).

**(ii) Duty retention.** Per stride $k$ of foot $i$ (touchdown to touchdown), the measured duty factor
is the contact share $\hat d_{i,k} = \big(\text{stance duration of stride } k\big) / \big(\tau^i_{k+1} - \tau^i_k\big)$;
the retention error is

$$
\Delta d_{i,k} \;=\; \hat d_{i,k} - d^\*\big(\bar v^{\mathrm{cmd}}_k\big),
\tag{15}
$$

aggregated as mean ± CI over strides, per speed bin. Duty is reported jointly with the Froude number
$\mathrm{Fr} = \bar v^2 / (g L_{\mathrm{leg}})$ (G1 leg length $L_{\mathrm{leg}}$ = \todo{hip height
from URDF FK; standing height 0.79 m}) so that ground-contact-time structure is checked against the
human walk–run scaling envelope (ground-time-structure gate: one-vote veto, per our internal data
standard).

**(iii) Transition band.** For the up/down command sweep $v^{\mathrm{cmd}}: 0 \to v_{\max} \to 0$
(\todo{v_max}), fit a two-segment piecewise-linear model to the stride-parameter curves
$v \mapsto \hat f(v)$ (measured stride frequency) and $v \mapsto \hat d(v)$ (measured duty); the
**transition band** $[v_-, v_+]$ is the interval between the changepoints of the two fits (width
$w_{\mathrm{tr}} = v_+ - v_-$, center $v_{\mathrm{tr}}$), and **hysteresis** is
$\Delta v_{\mathrm{hys}} = v^{\mathrm{up}}_{\mathrm{tr}} - v^{\mathrm{down}}_{\mathrm{tr}}$ between
sweep directions. A regime change is declared only if both the frequency relation slope and the duty
level change at compatible speeds — the operational definition of *automatic gait selection*.

**(iv) Delay margin.** Following [arXiv:2610.32277], we inject a constant action/observation latency
$\tau$ and measure the success rate $\mathrm{SR}(\tau)$ over \todo{N} rollouts per speed bin (success
= \todo{criterion}); the **delay margin** is

$$
\mathrm{DM} \;=\; \sup\{\tau \le \tau_{\max} : \mathrm{SR}(\tau) \ge 0.9\},
\tag{16}
$$

estimated by a bisection sweep on $\tau$ with resolution \todo{}. Delay margin is the timing-structure
robustness summary: a policy whose gait timing is well-synchronized tolerates larger loop latency.

**(v) Supporting quantities.** Velocity-tracking error per speed bin; measured
frequency–speed relation $\hat f(v)$ vs the schedule $f_{\mathrm{cmd}}(v)$ and vs Froude scaling;
push-recovery: number of steps and time to re-synchronize (all touchdown errors
$|e^{\mathrm{td}}| < 0.1$ cycles for \todo{K} consecutive strides) after a lateral/forward push of
magnitude $F$ for duration \todo{}; fall rate. Cross-simulator retention: metrics (i)–(iv) evaluated
identically in Isaac and MuJoCo, with relative differences reported per metric.

---

## Appendix A — PRG State Machine Pseudocode

Faithful to `prg_clock.py::PRGState.step` (order (a)–(d), Eqs. 6–13). Constants:
`PRG_FMOD_SCALE=0.5, PRG_ALPHA_RAMP_S=10.0, PRG_ALPHA_MAX=0.5, PRG_SYNC_WINDOW=0.25, PRG_SYNC_HEAD=0.05,
PRG_RESET_GAIN=2.0, PRG_SOFT_CLAMP=0.15`; curves per Eqs. 2–4; obs per Eq. 1 with the v71-authoritative
column order `[sinL, sinR, cosL, cosR]`.

```text
# state: phi (N,2) in [0,1); contact_prev (N,2) bool; t_ep (N,); episode init: phi[:,0]=0, phi[:,1]=0.5,
#        contact_prev=False, t_ep=0
procedure STEP(dt, v_cmd (N,), contact (N,2) bool, delta_f (N,) or None):
    # (a) blended frequency, Eqs. (6)-(8)
    f_cmd = 1 / clip(1.46 - 0.36*v_cmd, 0.70, 1.75)            # T*(v), Eq. (2)
    if freeze_alpha: alpha = 0                                   # v71 degradation switch
    else:            alpha = clip(t_ep / 10.0, 0, 1) * 0.5       # Eq. (8)
    if delta_f is None: f_pol = f_cmd                            # v71 mode
    else:               f_pol = f_cmd * (1 + 0.5 * clip(delta_f, -1, 1))
    f_eff = (1 - alpha) * f_cmd + alpha * f_pol                  # Eq. (6)

    # (b) advance both feet at the shared rate, Eq. (13b)
    phi = phi + f_eff[:,None] * dt

    # (c) touchdown-triggered reset, Eqs. (9)-(12)
    touchdown = contact & ~contact_prev                          # rising edges
    if reset_mode != "none" and touchdown.any():
        q = phi % 1.0
        in_sync = (q >= 1 - 0.25) | (q < 0.05)                   # Eq. (10)
        mask = touchdown & in_sync
        if reset_mode == "hard" and mask.any():
            phi[mask] = 0.0                                      # Eq. (11)
        elif reset_mode == "soft" and mask.any():
            err = where(q < 0.5, q, q - 1.0)                     # signed wrap error
            corr = clip(-2.0*dt*err, -0.15*dt*2.0, +0.15*dt*2.0) # Eq. (12)
            phi = phi + where(mask, corr, 0)

    # (d) wrap + bookkeeping, Eq. (13d)
    phi = phi % 1.0
    contact_prev = contact
    t_ep = t_ep + dt
    return phi

procedure CLOCK_OBS():        # (N,4), Eq. (1); column order is v71-authoritative
    ang = 2*pi*phi
    return [sin(ang[:,0]), sin(ang[:,1]), cos(ang[:,0]), cos(ang[:,1])]

procedure TARGETS(v_cmd):     # reward-side phase targets, Eqs. (3)-(5)
    duty = clip(0.68 - 0.07*v_cmd, 0.55, 0.70)                   # d*(v)
    h    = clip(0.025 + 0.025*v_cmd, 0.025, 0.075)               # h*(v)
    stance = phi < duty[:,None]                                  # stance: phi in [0, d*)
    return {duty, h, stance, ~stance}
```

Integration with the training environment: `step` is called once per control step before reward
evaluation; `clock_obs` is concatenated to the proprioceptive observation; `targets` feeds the
height/contact-timing reward terms (Section 3.4). The sim2sim/deploy twin instantiates the same
procedure with `backend="numpy"`.

## Appendix B — Environment and Training Hyperparameters (known values)

Values marked \todo{} are not yet frozen at the time of writing and will be filled from the final
training configuration.

| Group | Parameter | Value |
|---|---|---|
| Platform | GPU / environments | 1× RTX 4090D 24 GB (gradmotion), 4096 parallel envs |
| Platform | stack | IsaacLab + rsl_rl PPO (robolab base); measured 55k steps/s at 4096 envs |
| Robot | platform / DoF | Unitree G1, 29 revolute joints; standing height 0.7897 m (URDF FK) |
| Control | policy rate / PD | 50 Hz policy; PD at \todo{200 Hz–1 kHz, G1 final value} |
| Control | action | $q_{\mathrm{cmd}} = q_{\mathrm{default}} + 0.25\,a$, $a \in \mathbb{R}^{29}$ |
| Observation | layout | proprioception 96-dim/frame × 3 frames + 4 clock channels (Eq. 1) = 300-dim input |
| Clock (v71 curves) | $T^\*(v)$ / $d^\*(v)$ / $h^\*(v)$ | Eqs. (2)/(3)/(4) with the constants therein |
| Clock (PRG) | reset modes | hard / soft / none (ablation switch) |
| Clock (PRG) | sync window $\gamma$, head $\gamma_{\mathrm{head}}$ | 0.25, 0.05 |
| Clock (PRG) | soft gain $\beta$, clamp $c_{\max}$ | 2.0 s⁻¹, 0.15 cycles |
| Clock (PRG) | $\alpha$ curriculum $T_{\mathrm{ramp}}$, $\alpha_{\max}$ | 10 s, 0.5 |
| Clock (PRG) | freq-mod scale $\kappa$ | 0.5 (δf action augmentation: **v2 extension**, reserved switch) |
| Reward | hinge windows | swing-mid [0.2, 0.8) of swing span; stance-mid [0.15, 0.85) of stance span |
| Reward | weights, gating $v_{\min}$ | \todo{} |
| PPO | lr, batch, minibatch, iters, entropy, γ, λ_GAE | \todo{} |
| Commands | speed sampling, episode length, resets | \todo{} |

---

### Placeholder reference list (ids verified 2026-11-17)

[arXiv:2212.03238] Walk These Ways (CoRL 2022) · [arXiv:2502.03206] HugWBC (RSS 2025) ·
[arXiv:2509.12858] CRA-LoCo · [arXiv:2504.13619] Robust Humanoid Walking (HRP-5P) ·
[arXiv:2512.07464] Gait-Adaptive Perceptive Locomotion · [arXiv:2609.38824] PhaseSync-Exo ·
[arXiv:2608.07328] Adaptive Gait Timing (quadruped, IROS 2026) · [arXiv:2409.14342] H-MPC (ICRA 2025) ·
[arXiv:2509.20696] RuN · [arXiv:2610.37974] FORGE · [arXiv:2605.18611] Unified W-R-R ·
[arXiv:2610.32277] Delay Randomization · [arXiv:2510.01708] PolySim · [arXiv:2502.01143] ASAP ·
[arXiv:2404.05695] Humanoid-Gym
