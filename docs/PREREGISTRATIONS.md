# Preregistration — certified difficulty-routing pilot

Frozen 2026-08-22, BEFORE any simulation was run. House rules apply: verdicts
PASS / FAIL / PARTIAL / NO POWER; the weaker verdict governs; nothing above an
amendment line is edited after data is seen.

## Host specification (locked, from arXiv:2603.21177v1 §4.1 + Algorithm 1 + Table 1)

Prompt Replay routing, abstracted to the per-prompt decision our measurement
work addresses:

* Actions: `REPLAY` (prioritized reuse) vs `FRESH` (standard sampler).
* Rule: `f(p̂) = REPLAY iff p̂ ∈ [p_min, p_max] = [0.25, 0.75]`.
* Host estimator: raw fraction of the current step's G=16 rollouts; no prior,
  no smoothing; replaced on every reuse.
* Priority within REPLAY: ascending |p̂ − 0.5|, ties uniform.
* Mixing: up to ε=0.75 of a 32-prompt batch from the buffer top; cooldown C=10;
  max reuse R=15; eviction when recomputed p̂ leaves the band or R exhausted.

The B1 audit fixes f to this band rule and audits the ESTIMATION step only:
what the host computes at N=16 (and N=8/32 for budget sensitivity) versus what
a certification layer would compute on the same observations.

## Reference ("oracle") definition

* Population: all 1023 prompts of `simplelr_qwen_level1to4_sub1k`.
* Truth proxy: pooled pass counts over TWO independent stored evaluation passes
  of baseline checkpoint seed-1/global_step_120, T=0.7, cap 1024:
  pass A = `qwen_0.5b_ckpt1_hi_*` (464 noise subset) ∪ `qwen_0.5b_ckpt1_hi_rest_*`
  (559 complement); pass B = `qwen_0.5b_ckpt1_n128_rep2_*`. n=256 per prompt.
  Scores used RAW (`verification.score==1`), matching the host's verifier.
* Reference action: `a*(x) = REPLAY iff p̃ := k/256 ∈ [0.25, 0.75]`.
* Oracle uncertainty: under Beta(k+α₀, 256−k+β₀), prompt flagged
  ORACLE-UNCERTAIN if posterior mass on the reference action < 0.99.
  Only ORACLE-UNCERTAIN prompts are eligible for additional real sampling.
* Results reported TWICE: all prompts (reference action as truth), and the
  oracle-unambiguous subset. Gates below are judged on BOTH readings and the
  weaker governs.

## Arms (all share the band rule unless stated)

Prior for every Bayesian arm: empirical-Bayes Beta(α₀, β₀) moment-matched to
the 1023 reference p̃ values. Sensitivity: uniform Beta(1,1).

1. **plug-in fixed-N**, N ∈ {8, 16, 32}: act on k/N. N=16 is the host's actual
   operating point.
2. **posterior-mean fixed-N**, N ∈ {8, 16, 32}: act on E[p|k,N].
3. **posterior-expected-utility fixed-N** (primary at N=16, also 8/32):
   REPLAY iff E[p(1−p)|k,N] > ṽ, where ṽ = mean of p̃(1−p̃) over the population
   (the expected value of one unprioritized fresh draw).
4. **certified sequential**: start N₀=8, increments ΔN=4, cap 64, stop when
   C_x = max_j P(f(p)=j | k,N) ≥ δ; then act argmax_j. δ grid {0.90, 0.95, 0.99}.
5. **certified abstention**: identical stopping; if uncertified at cap,
   abstain → FRESH (do not prioritize). Recorded separately so abstention cost
   is visible.
6. **KGPS-style PEU proxy (NOT actual KGPS)**: adaptive allocation — repeatedly
   grant ΔN=4 extra rollouts to the prompt with the largest posterior sd of
   v(p)=p(1−p), until its total budget equals arm 4's mean consumption at the
   same δ; then act by the arm-3 rule. A stand-in for utility-driven adaptive
   estimators only.

Simulation draws: parametric Binomial(N, p̃) (primary). Sensitivity:
nonparametric subsampling of the real stored rollouts of passes A/B without
replacement within each pass (preserves verifier contamination).

Seeds: numpy default_rng(20260822); rerun stream offset by 1. Replications:
200 independent draw-streams per arm per metric pair (rerun metrics use
stream pairs).

## Metrics (all computed in b1_sim.py; no hand-typed numbers downstream)

Primary:
* action accuracy vs a* ;
* rerun reproducibility: Jaccard of REPLAY sets between two streams, plus raw
  action agreement;
* false-route rate P(REPLAY|a*=FRESH) and missed-route rate P(FRESH|a*=REPLAY);
* mean rollouts consumed per prompt (certification fully charged).

Secondary: PR-objective regret Σ_x |v(a*_x) − v(a_x)| with v(REPLAY)(x) =
p̃(1−p̃) and v(FRESH) = ṽ (our operationalization of the host's Σ v(x)
objective — NOT taken from the paper); ranking error (Kendall τ of estimated
vs reference |p−0.5| ordering among REPLAY sets); sample-count distribution.

## Gate (provisional thresholds per user instruction)

Proceed to B2 only if BOTH hold on both populations (weaker governs):

* G1: certified (best δ) cuts action error ≥30% vs plug-in N=16 at ≤110% of
  its mean rollout cost;
* G2: certified rerun Jaccard ≥ 0.90 at that operating point.

The complete cost–error Pareto curve is reported regardless; single-threshold
optimization is forbidden. If the gate fails: method direction stops; the
measurement paper stands alone.

## What is explicitly NOT done yet

B3 training, KL-triggered recertification, VOI stopping, second host, learned
estimators, model sweeps, and any claim about trained-model outcomes (E3's
null warns exactly against assuming decision differences propagate).

---

# AMENDMENT A82 (2026-08-22, post-B1, before any top-up or real generation)

## Recorded verdicts on the ORIGINAL B1 gates — not relabelled

* G1 cost criterion (≤110% of plug-in N=16): **FAIL** (113% at δ=0.90).
* G1 error criterion: PASS (45% reduction). G1 overall: PARTIAL.
* G2 (certified rerun Jaccard ≥ 0.90): **FAIL** (best 0.749; unmeetable by
  every arm including fixed-N baselines).
* **B1 overall: PARTIAL** — follow-up justified by the observed cost-error
  Pareto frontier, not by the registered gates.

## Labelling corrections

* `kgps_proxy` is a **posterior-expected-utility diagnostic**, not a
  comparison against actual KGPS. All reports must carry that label.
* Rerun **action agreement** becomes a prospectively registered primary
  metric from this amendment onward. Jaccard demotes to secondary.

## Prospective registration — real-generation certification (B2)

Frozen before any B2 generation.

Population/checkpoint/band: all 1023 prompts; checkpoint seed-1/global_step_120;
band [0.25, 0.75]; T=0.7, top_p=0.95, max_new_tokens=1024 (matching every
stored pass); verifier raw `verification.score==1`.

Generation design: **two independent executions**, each a fresh 64-sample
pass over all 1023 prompts (sglang, engine-warm). Arms are computed by
consuming prefixes of each execution's per-prompt draw stream:

* fixed N=16, fixed N=32: consume the first 16 / 32 draws;
* certified sequential, δ ∈ {0.90, 0.95, 0.99}: rounds of 8 starting at 8;
  after each round recompute C_x under the EB prior fitted on the REFINED
  oracle; uncertified prompts receive another round until C_x ≥ δ or 64;
  then act on the higher-mass side.

Prefix consumption at multiples of 8 is exactly the registered rounds-of-8
rule; within an execution all arms share common draws (paired), while the two
executions are fully independent generation streams. Every drawn rollout is
charged to the consuming arm's budget.

Primary metrics (registered): routing error vs the refined oracle;
rerun action agreement between the two executions; average rollouts per
prompt; false-REPLAY and missed-REPLAY rates; Prompt-Replay-native decision
regret (v(REPLAY)=p̃(1−p̃) at refined p̃, v(FRESH)=ṽ refined).

Secondary: replay-set Jaccard; distribution of per-prompt N; stopping
fractions at 8/16/…/64; error by distance-to-boundary bins around 0.25/0.75.

**Prospective success criterion** (replaces nothing; old gates stand recorded):
proceed to training only if some certified δ has simultaneously
(i) lower routing error than fixed N=32,
(ii) average cost ≤ fixed N=32,
(iii) higher rerun action agreement than fixed N=32,
and the error improvement holds on the refined-oracle unambiguous subset.
If no δ satisfies all four, the method direction stops.

## Oracle refinement protocol (frozen)

Top-up: +256 fresh draws, same checkpoint/config, ONLY the 133 prompts whose
N=256 reference-action mass < 0.99. Reference becomes pooled n=512 there,
n=256 elsewhere. Report: action changes among the 133; remaining ambiguous
count at n=512; regenerated headline tables on BOTH oracles. δ values are
NOT retuned against the refined oracle.

---

# AMENDMENT A83 (2026-08-22) — B3 pilot registration, frozen before any training run

## Scope

Pilot only: does the certified routing layer change ONLINE TRAINING behaviour
of a literal Prompt Replay host? Full grid deferred until this reads out.

## Arms

1. `pr_vanilla` — Prompt Replay exactly per locked spec (B1_REPORT §1):
   after each gen batch, p̂ = that batch's group mean reward per prompt;
   insert/update if p̂ ∈ [0.25, 0.75] and uses < R else remove; priority
   ascending |p̂−0.5| ties-uniform; next batch mixes up to ε=0.75 of its 32
   prompts from eligible buffer top; cooldown C=10 steps; R=15 reuses.
2. `pr_cert09` — identical except the insert/update decision is gated by an
   accumulated Beta posterior over EVERY group that prompt has produced so
   far in the run: insert only if C_x = P(p ∈ band | history) ≥ 0.90;
   removal when posterior says out-of-band with the same confidence or R is
   exhausted. Uncertified prompts never prioritised (stay FRESH-only).

No fresh vanilla-GRPO baseline is run: E3's baseline evals
(math500 31.40%, pool2000 45.06%, five seeds, guarded) are the reference.

## Registered deviations from the paper / B2 (stated, not hidden)

* **G=8 rollouts** (project-standard config, comparable with E0/E3) instead
  of the paper's 16 → band test k ∈ {2,…,6} of 8.
* Certified arm uses a **uniform Beta(1,1) online prior**, not B2's
  refined-oracle-fitted EB prior — at training time no oracle exists; this
  is the deployable choice.
* Buffer updates read the FIRST `compute_prompt_reward_stats` call of each
  gen batch (pre-replay/pre-filter group means = the trained group).
* Swaps take effect at the NEXT batch assembly (no mid-batch mutation).

## Config (E0-identical otherwise)

dataset simplelr_qwen_level1to4_sub1k@train; batch 32; rollout_n 8; lr 5e-7;
kl_loss 1e-4; entropy 1e-3; temperature 1.0 rollouts; seed 1; target 120
optimizer steps (watchdog stop-on-save); save_freq 40.
Checkpoints to `CHECKPOINTS_ROOT=/home/rmedu/pr_b3_ckpts` (internal NVMe,
98 GB free); delete-after-eval policy — results JSONs are the durable record.

## Pilot metrics (primary)

* mean |advantage| per step trajectory (PR's headline quantity);
* zero-variance discard fraction per step;
* held-out guarded accuracy on `e3_math500` and `e3_pool2000` at step 120,
  compared against the E3 baseline numbers above;
* routing telemetry JSONL: step, buffer size, insertions, removals, reuse
  hits, uncertified-blocked count (cert arm).

## Pre-registered branch table

* **Branch A** — certified differs from vanilla on any primary metric beyond
  noise bands (±2σ bootstrap over steps): certification matters online →
  full-grid justification.
* **Branch B** — arms statistically indistinguishable AND both ≈ baseline:
  routing differences do not propagate (E3-consistent robustness); method
  direction stops at measurement layer.
* **Branch C** — either arm materially worse than baseline: replay itself
  underperforms here (consistent with the paper's plateau report); report
  regardless of which arm wins between them.
* The weaker verdict governs; no mid-run tuning of δ, ε, C, R.

---

# AMENDMENT A84 (2026-08-23) — VOI pilot, frozen before implementation

## Motivation (registered)

B3 showed δ-confidence certification optimizes the wrong objective: it blocks
high-signal band-edge prompts whose misclassification is nearly free under the
host's v(p)=p(1−p). The VOI arm prices both errors explicitly instead.

## Arm `pr_voi` — expected-asymmetric-loss routing

Per prompt, maintain accumulated posterior Beta(1+K, 1+N−K) over ALL groups
observed so far (same history as cert09). Utility s(p) = p(1−p)·c8(p) where
c8(p) = 1 − p^8 − (1−p)^8 (GRPO G=8 contributing probability; the project's
own collapse machinery). Two online running estimates: ṽ_obs = EWMA(mean s of
observed groups, α=0.05) and v_top = EWMA(s of the current highest-priority
eligible buffer prompt, α=0.05).

INSERT/UPDATE iff  P(in)·w·E[s]  ≥  P(out)·max(0, v_top − E[s]) ,
where E[s]=∫s(p)Post(dp) on a 101-point grid; P(in/out) = posterior mass on
either side of [0.25,0.75]; w = 3.0 FROZEN (discounted multi-reuse value of a
true member vs single-slot displacement cost; no tuning permitted).

Removal: same inequality failing AND P(out) ≥ 0.90 (confident eviction only),
plus the usual R-exhaustion removal. Cooldown C=10, R=15, ε=0.75 unchanged.
Everything else identical to A83 arms.

## Runs

pr_voi seeds 1 and 2; pr_vanilla seed 2 (seed 1 exists from B3). Identical
config surface to A83. Telemetry gains a per-tick voi_decisions block.

## Pre-registered readout and KILL CRITERION

Primary: guarded MATH500 accuracy. Proceed toward a method claim ONLY IF
mean over seeds of (VOI − vanilla) ≥ +0.50 pts AND VOI − baseline ≥ 0 on
pool2000 within 0.30 pts. Otherwise the method direction STOPS permanently
and Front B closes as a negative-result chapter inside the measurement paper.
Secondary: pool2000 deltas, buffer telemetry, fraction of decisions flipped
vs cert09's decisions on the same ticks.

---

---

*This file contains the preregistrations for the work reported in the paper. Registrations for
follow-up experiments that are not part of this paper are held back pending their own write-up.*
