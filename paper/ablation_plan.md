# Ablation Plan — Pre-Registered (C1 / PRG)

> **Pre-registration status.** This document is written *before* the main C1 training results exist.
> Hypotheses, primary metrics, judgment gates, and decision branches are fixed here and recorded in the
> project lab notebook before each run; outcome cells are `\todo{}` and may only be filled from run
> artifacts. Any post-hoc change requires an explicit amendment entry (dated, reasoned) in the lab
> notebook — silent re-gating is forbidden. All arms share the frozen base config; each arm changes
> exactly one switch (single-variable chain), and every arm is seeded \todo{N≥3} times with mean ± CI
> reported.

## 0. Arm inventory and run order

| Arm | Switch (vs. C1 full) | What it isolates | Run |
|---|---|---|---|
| **B0** | `reset=none`, `freeze_alpha=1`, `delta_f=None` | fixed v71 clock baseline (exact degradation, verified ≤1e-6 circular error vs analytic v71) | W3 |
| **B1** | v71 clock + HugWBC-style timing commands (f, duty as user commands instead of speed schedule) | commanded-fixed-clock baseline | W3 |
| **B2** | no clock channels at all (velocity conditioning only) | value of any clock conditioning | W3 (budget permitting) |
| **C1** | full PRG: `reset=hard`, `delta_f=None` (v1), α-curriculum per Eq. (8) | main method (v1 scope: reset only) | W3 |
| **A1** | C1 with `reset=none` | **contact-triggered reset** | W3/W4 |
| **A2** | C1 + `delta_f` action augmentation active (**v2 extension**, reserved env switch) | policy frequency modulation | W4 (v2) |
| **A3** | C1 with `reset=soft` | hard vs. soft reset strength | W4 |
| **A4** | C1 with α frozen at `alpha_max` from t=0 (A4a) and curriculum disabled with δf arm (A2+A4b) | α curriculum | W4 (v2; A4a valid in v1 only if A2's switch ships) |
| **A5** | C1 with contact-timing hinge rewards ($r_{\mathrm{csw}}, r_{\mathrm{cst}}$) removed | reward co-synchronization | W4 |
| **A6** | C1 with sync window γ ∈ {0.05, 0.50} (default 0.25) | reset gating width | W4 (budget permitting) |

Global guarantee: B0 ≡ v71 *bit-for-bit at the clock level* (regression-tested: constant-v analytic
error ≤1e-6 cycles over ≥1500 steps; time-varying-v vs. integral form 1e-9), so C1-vs-B0 differences
are attributable to the PRG mechanism alone.

## 1. Pre-registered hypotheses and gates

### A1 — Contact-triggered phase reset (core mechanism)

- **Hypothesis H1.** Touchdown-triggered re-anchoring reduces phase–contact mismatch and improves
  disturbance re-synchronization, at matched or better velocity tracking, vs. `reset=none`.
- **Primary metrics.** (i) touchdown anchoring error $\overline{|e^{\mathrm{td}}|}$ (Eq. 14, method
  §3.6) at steady state, per speed bin; (ii) post-push re-synchronization steps (definition in method
  §3.6(v)); (iii) fall rate under push set \todo{push magnitudes/durations}.
- **Gate (confirmed if).** C1 improves $\overline{|e^{\mathrm{td}}|}$ by ≥ \todo{30\%} relative AND
  reduces re-synchronization steps by ≥ \todo{1 stride} on ≥ \todo{2/3} speed bins, with velocity
  tracking not worse than \todo{0.05 m/s} mean absolute error vs. A1.
- **Falsified if.** $\overline{|e^{\mathrm{td}}|}$ and recovery steps are statistically
  indistinguishable (overlapping CIs across seeds) on all bins, or velocity tracking degrades beyond
  the gate while sync gains stay below it.
- **Decision branches.** Confirmed → reset is a first-class mechanism (paper core claim).
  Indistinguishable → investigate reset firing statistics (fraction of touchdowns inside the sync
  window, log probe) before any re-run; if the policy simply never mistimes, re-scope the claim to
  perturbation regimes only (push/pull and delay arms).

### A2 — Policy frequency modulation (v2 extension arm)

- **Hypothesis H2.** Bounded policy frequency authority (±50% band via α-blend, Eq. 6–8) improves
  speed-spectrum coverage and gait-selection smoothness over δf=None, without reopening the
  micro-stepping attractor.
- **Primary metrics.** (i) velocity-tracking error across the sweep, especially top bin
  \todo{v_max}; (ii) measured frequency–speed relation $\hat f(v)$ deviation from
  $f_{\mathrm{cmd}}(v)$; (iii) transition band width $w_{\mathrm{tr}}$ and hysteresis (method §3.6(iii));
  (iv) **micro-stepping guard**: stride-frequency ratio $\hat f / f_{\mathrm{cmd}}$ distribution —
  mass above \todo{1.5×} indicates attractor reopening.
- **Gate.** A2 improves mean tracking on ≥ \todo{2/3} bins vs. C1 AND micro-stepping guard mass
  < \todo{5\%} of strides.
- **Falsified if.** Tracking unchanged/worse, or guard violated (policy abuses the frequency channel).
- **Decision branches.** Confirmed → v2 headline extension. Guard violated → tighten $\kappa$ or
  $\alpha_{\max}$ (one knob, one re-run, logged as amendment); both fail → report as negative result
  with the mechanism analysis (the clock stays reset-based).

### A3 — Hard vs. soft reset

- **Hypothesis H3.** Hard reset anchors contacts more precisely but injects phase discontinuities that
  hurt training stability; soft reset trades anchoring sharpness for smoothness and better final
  synchronization under sustained perturbation.
- **Primary metrics.** $\overline{|e^{\mathrm{td}}|}$ distribution (sharpness: 90th percentile),
  post-push recovery steps, training-curve stability (final-return variance across seeds), fall rate.
- **Gate.** Report both; prefer the variant with better recovery-vs-sharpness Pareto position
  (recovery steps primary, $|e^{\mathrm{td}}|$ p90 secondary). No single winner is pre-committed —
  this arm is *characterization*, and both results are publishable as mechanism analysis.
- **Decision branches.** If hard ≈ soft everywhere → simplify to hard in the main results and state
  robustness to reset strength; if they differ → keep both in the main table.

### A4 — α curriculum

- **Hypothesis H4.** Ramping α from 0 (pure command) to 0.5 over 10 s stabilizes early training
  relative to full authority from t=0.
- **Primary metrics.** Time-to-reward-threshold \todo{threshold}, final return, early-episode
  (t < 10 s) fall rate.
- **Gate.** Curriculum arm reaches the threshold ≥ \todo{20\%} earlier (wall-clock or samples) with
  final return within CI of the no-curriculum arm.
- **Decision branches.** Confirmed → keep curriculum (paper Eq. 8 stays). Indistinguishable → simplify
  (constant α) and note; worse → drop curriculum, amend method §3.3.1.

### A5 — Reward co-synchronization (contact-timing hinge terms)

- **Hypothesis H5.** Scoring contacts against clock windows is necessary for phase–contact lock-in;
  removing it degrades duty retention and sync even with resets active.
- **Primary metrics.** Duty retention $\Delta d$ (Eq. 15) mean and CI; $\overline{|e^{\mathrm{td}}|}$;
  ground-time-structure gate (duty×Froude envelope, one-vote veto).
- **Gate.** A5 degrades $\overline{|\Delta d|}$ by ≥ \todo{50\%} relative or violates the
  ground-time-structure gate.
- **Decision branches.** Confirmed → reward co-sync is load-bearing (closes the loop argument,
  method §3.4). Indistinguishable → resets alone suffice; reframe §3.4 accordingly (mechanism claim
  shifts to reset + conditioning only).

### A6 — Sync window width γ (optional, budget permitting)

- **Hypothesis H6.** γ=0.25 (default) balances anchor coverage vs. spurious resets; γ=0.05 starves the
  reset (few firings), γ=0.50 lets mid-cycle contacts corrupt the rhythm.
- **Primary metrics.** Reset firing rate and firing-phase histogram (probe); $\overline{|e^{\mathrm{td}}|}$;
  fall rate under pushes.
- **Gate.** U-shape: both extremes worse than default on ≥ 1 primary metric each.
- **Decision branches.** U-shape confirmed → report sensitivity curve (mechanism-depth evidence).
  Flat → state insensitivity, drop from main text to appendix.

## 2. Main-experiment gates (C1 vs. baselines) — pre-registered

- **Primary comparison.** C1 vs. B0 and B1 on: velocity-spectrum tracking error (all bins), post-push
  recovery steps, fall rate; secondary: GaitTiming family (i)–(iv) of method §3.6.
- **Success gate (paper-level).** C1 ≥ tie on both primary metric families vs. B0/B1 AND at least one
  ablation mechanism (A1 or A5) confirmed with effect ≥ gate.
- **Stop-loss gate (route decision).** If C1 does not beat B1 on the walk spectrum (tracking and
  recovery both worse, CI-separated), the route downgrades to C2 (command-spectrum paper) per the
  project plan v2; this decision is logged in the lab notebook with the artifact paths.
- **Sim2sim one-vote veto.** A policy that fails the ground-time-structure gate (duty×Froude) or the
  harness self-check in MuJoCo does not advance to result tables regardless of training metrics
  (protocol per [arXiv:2610.32277]).

## 3. Reporting commitments

- Every table cell carries seed count and CI; single-seed deltas are never decision inputs.
- Reset firing statistics (rate, phase histogram, window hit fraction) are logged for every arm —
  mechanism claims cite these probes, not just end metrics.
- All gates evaluated on the *pre-registered* metric definitions of method §3.6; no metric redefinition
  after unblinding. Outcome table: \todo{filled from run artifacts}.
