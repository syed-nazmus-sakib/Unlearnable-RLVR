# Reproduction notes — 1× RTX 3090 Ti

Reproduction of the **Qwen2.5-0.5B + MATH levels 1–4** walkthrough from the root
[README](../README.md) on a single-GPU workstation.

## Environment: the `vllm` conda env

Installed into the **`vllm`** env via [`setup_env.sh`](setup_env.sh).

Despite the name, that env was *empty* (8 packages — pip, setuptools, requests;
`import vllm` failed), so it was effectively a blank Python 3.11 slate and filling
it displaced nothing.

The other candidate, `taton`, was rejected: it is a populated env (259 packages) on
**torch 2.9.1 / transformers 4.57.3 / numpy 2.2.6**, whereas this repo pins
**torch 2.6.0 / transformers 4.51.1 / numpy 1.26.4**. Installing here would have
downgraded three core libraries and likely broken whatever `taton` is for.

Verified versions:

| Package | Version |
|---|---|
| Python | 3.11.14 |
| torch | 2.6.0+cu124 |
| sglang | 0.4.6.post1 |
| ray | 2.10.0 |
| transformers | 4.51.1 |
| numpy | 1.26.4 |
| flash-attn | 2.7.4.post1 |

Two dependency notes that `requirements.txt` does not make obvious:

- **flash-attn is required, not optional.**
  [`verl/workers/actor/dp_actor.py:35`](../verl/workers/actor/dp_actor.py#L35) imports
  `flash_attn.bert_padding` at module top level, so the FSDP actor path cannot start
  without it. Installed from the prebuilt
  `cu12torch2.6cxx11abiFALSE-cp311` wheel — torch 2.6.0+cu124 from
  download.pytorch.org reports `_GLIBCXX_USE_CXX11_ABI == False`, so the `abiTRUE`
  wheel is the wrong one. A source build takes hours and more RAM than this box has.
- **flashinfer is skippable.** `flashinfer-python==0.2.3` is sdist-only (no wheel), and
  sglang falls back to a working attention backend on sm86 without it. Verified by a
  generation smoke test.

## Hardware constraint

| | Paper walkthrough | This box |
|---|---|---|
| GPUs | ≥2, driver allocates 4 | 1× RTX 3090 Ti, 24 GB |
| RAM | — | 30 GB |

The paper config (batch 256, `rollout_n 8`, 5120-token responses, `rollout_tp 2`,
`--num-gpus 4`) does not fit one 24 GB card, so training is rescaled by
[`train_1gpu.sh`](train_1gpu.sh).

### Scaled training config

The guiding principle: **preserve exposures per prompt and every hyperparameter that
shapes the GRPO signal**; shrink only what is forced by the single GPU.

| Knob | Paper | Here | Why |
|---|---|---|---|
| `n_gpu_per_node` | 4 | 1 | one GPU |
| `rollout_tp` | 2 | 1 | TP needs ≥2 GPUs |
| `train_batch_size` | 256 | 32 | fits alongside the colocated sglang engine |
| `ppo_mini_batch_size` | 64 | 16 | keeps 2 optimizer steps per rollout batch |
| `ppo_micro_batch_size` | 32 | 2 | LM-head logits dominate memory (see below) |
| `log_prob_micro_batch_size` | 128 | 4 | same |
| `max_response_length` | 5120 | 1024 | KV cache + logits |
| training prompts | 8139 | 1023 | stratified subset, [`make_subset.py`](make_subset.py) |
| `total_epochs` | 50 (ran to step 120) | 12 (~120 opt. steps) | see below |
| **`rollout_n`** | **8** | **8** | unchanged — drives the GRPO group signal |
| **learning rate** | **5e-7** | **5e-7** | unchanged |
| **`kl_loss_coef`** | **1e-4** | **1e-4** | unchanged |
| **`entropy_coefficient`** | **1e-3** | **1e-3** | unchanged |

**On step accounting — two counters, two different units.** This is easy to get wrong,
and getting it wrong silently changes how long training runs:

- Termination is `gen_steps >= total_training_steps`
  ([`ray_trainer.py:1079`](../verl/trainer/ppo/ray_trainer.py#L1079)), where
  `total_training_steps = len(dataloader) * total_epochs`. That counter tracks
  **generation batches**.
- `save_freq` / `test_freq` are `global_steps % freq`
  ([`ray_trainer.py:1314`](../verl/trainer/ppo/ray_trainer.py#L1314)), and `global_steps`
  is the **optimizer** step counter — the number in `global_step_120`.

The two diverge because of dynamic sampling: the `std > 0` filter discards prompts whose
8 rollouts all agree (zero advantage) and pulls another generation batch, so one
optimizer step consumes **~3 generation batches** (observed `num_gen_batches=3` at
step 1). The paper's `global_step_120` therefore corresponds to ~360 generation batches
and ≈ 11 exposures per prompt (120 × 3 × 256 ÷ 8139) — *not* the ≈ 3.8 a naive
`120 × 256 ÷ 8139` suggests.

Matching that: 12 epochs × 31 batches = 372 generation batches → **≈ 120 optimizer
steps** and **≈ 11.6 exposures per prompt** (372 × 32 ÷ 1023). Both axes line up with
the paper. An earlier 4-epoch attempt was aborted after 2 steps once this was noticed;
it would have produced only ~41 optimizer steps, likely too little optimization for any
prompt to cross the threshold — which would have emptied the `learnable` set and made
the group comparison meaningless.

**On micro-batch sizing.** Qwen2.5's vocabulary is 151936 tokens, so the LM-head
logits are ~0.6 MB fp32 *per token* — far larger than the 0.5B model's parameters.
Micro batches here are sized against logits, not weights. The same arithmetic caused
the inference OOMs below.

## Deviations from the root README

### 1. Evaluate the `train` split, not `test`

The README's step 1b runs `--test-data simplelr_qwen_level1to4@test`. That does not
line up with classification, which joins two id spaces:

- `verl/utils/dataset/rl_dataset.py:200` — `prompt_id = extra_info["index"]`
  → `prompt_reward_stats.csv`
- `inference/evaluate_model.py:118` — `id = extra_info["index"]`
  → `question_id` in the results JSON
- `classification/finalize_unlearnable_example_id.py` joins those two

`extra_info["index"]` runs 0–8138 on the train split and 0–499 on the test split, over
*different problems*. Feeding `@test` results to the aggregator therefore matches
unrelated prompts against the training reward stats, silently. Unlearnability is a
claim about training prompts, so this reproduction uses `@train` throughout
([`eval.sh`](eval.sh)).

### 2. `example_data/example_ids/` does not ship

The README says you can skip step 1 and "simply use data in
`./example_data/example_ids`". That directory is not in the release: `.gitignore`
line 27 lists `example_ids/`, which excludes it under `example_data/` too, and
`git log --all --name-only` shows it was never committed. The shipped
`example_data/` contains only the 8 augmentation JSONs and `prompt_reward_stats.csv`.

Consequence: the gradient-analysis shortcut is unavailable, and the classification
step must actually be run to produce the `easy` / `learnable` / `unlearnable` id
lists. `prompt_reward_stats.csv` *is* shipped (306,176 data rows), so it remains usable as
a reference for what a full-scale run's reward stats look like.

## Training run (seed 1)

Stopped deliberately at **`global_step_120`**, the paper's comparison point, with
`prompt_reward_stats.csv` complete over steps 1–120 and all 1023 prompts covered.

The observed generation-batch : optimizer-step ratio was not constant — 2.8 early
(gen 14 → step 5), falling to 1.76 by gen 211 → step 120, as the sharpening policy
produced fewer all-agreeing rollout groups for the `std > 0` filter to discard. Left to
run out its 372 generation batches the job would have reached ~210 optimizer steps,
well past the paper; the extra ~90 steps were not useful, so the run was stopped once
step 120 was checkpointed.

Training reward (temperature 1.0, so not directly comparable to eval pass@1):

| optimizer step | 1 | 20 | 40 | 80 | 120 |
|---|---|---|---|---|---|
| `critic/score/mean` | 0.238 | 0.316 | 0.414 | 0.402 | 0.398 |
| `actor/entropy_loss` | 1.576 | 0.776 | 0.341 | 0.224 | 0.199 |

Reward rises steeply to ~step 40 then plateaus while entropy keeps contracting — the
policy sharpens on what it already solves rather than acquiring new prompts. That
plateau-with-correct-rollouts-still-arriving is the regime the paper's unlearnability
claim is about.

## Results

pass@k on the 1023 training prompts, 32 samples each, temperature 0.7:

| | pass@1 | pass@4 | pass@8 | pass@32 |
|---|---|---|---|---|
| Qwen2.5-0.5B (initial) | 0.251 | 0.511 | 0.627 | 0.796 |
| + GRPO, `global_step_120` | **0.473** | 0.657 | 0.732 | 0.841 |

pass@1 nearly doubles while pass@32 moves only +0.045 — the usual RLVR pattern of
sharpening the policy rather than extending capability.

### Classification (τ = 0.1, single seed)

| group | count | share |
|---|---|---|
| unlearnable | 156 | 15.2% |
| learnable | 159 | 15.5% |
| no_reward | 159 | 15.5% |
| easy | 586 | 57.3% |

### The unlearnability phenomenon reproduces

| group | n | init pass@1 | final pass@1 | Δ | correct rollouts / prompt (training) | % with ≥1 correct |
|---|---|---|---|---|---|---|
| easy | 586 | 0.416 | 0.727 | +0.311 | 24.0 | 99.1% |
| learnable | 159 | 0.052 | 0.323 | +0.271 | 5.5 | 92.5% |
| **unlearnable** | **156** | **0.049** | **0.038** | **−0.011** | **3.2** | **100.0%** |
| no_reward | 159 | 0.015 | 0.043 | +0.028 | 0.0 | 0.0% |

The load-bearing comparison is **learnable vs unlearnable**. They begin at
indistinguishable difficulty (0.052 vs 0.049) and both receive positive reward during
training, yet learnable gains +0.271 while unlearnable moves −0.011. Every one of the
156 unlearnable prompts received at least one correct rollout, so the failure is not an
absence of learning signal. `no_reward` separates cleanly at 0% correct, confirming it
isolates verifier/capability failures rather than learning failures.

Caveat: this is a **single-seed** partition. The paper intersects three seeds, which
removes prompts that merely got unlucky in one run, so 15.2% here is an upper bound on
the paper's stricter definition.

### The groups are not disjoint

The root README says the classification "partitions all evaluated prompts into four
groups", but they overlap by construction: `easy` is defined on **initial** pass@1
(`initial_all - initial_flagged`) while `unlearnable` is defined on **final** pass@1
(`intersect_flagged - no_reward`), so a prompt that starts above τ and ends below it
lands in both.

| overlap | count |
|---|---|
| easy ∩ unlearnable | 20 |
| learnable ∩ no_reward | 12 |
| easy ∩ no_reward | 5 |

1023 distinct ids across groups whose sizes sum to 1060. This also explains why
`sample_grad_sim_prompts.py` yields 44 "easy" and 39 "unlearnable" gradient points from
40-per-group sampling — `plot_grad_sim_heatmap.py` assigns each qid to whichever group
it matches first.

Repeating the table above on strictly disjoint groups (dropping all 37 multiply-assigned
ids) leaves the conclusion intact:

| group | n | init pass@1 | final pass@1 | Δ | correct rollouts / prompt | % ≥1 correct |
|---|---|---|---|---|---|---|
| easy | 561 | 0.426 | 0.755 | +0.329 | 24.8 | 100.0% |
| learnable | 147 | 0.052 | 0.322 | +0.270 | 6.0 | 100.0% |
| unlearnable | 136 | 0.029 | 0.037 | +0.008 | 2.5 | 100.0% |
| no_reward | 142 | 0.008 | 0.011 | +0.003 | 0.0 | 0.0% |

### Controlling for initial difficulty

On disjoint groups the unlearnable set starts somewhat harder than the learnable set
(0.029 vs 0.052), so the contrast above could in principle just be difficulty. Binning
both groups by initial pass@1 rules that out — within *every* band, learnable prompts
gain ~+0.24 to +0.30 while unlearnable prompts are flat or negative:

| init pass@1 band | learnable n | Δ learnable | unlearnable n | Δ unlearnable |
|---|---|---|---|---|
| 0.000 (never solved) | 28 | +0.244 | 66 | +0.024 |
| 0.000–0.031 | 33 | +0.242 | 33 | +0.009 |
| 0.031–0.063 | 47 | +0.277 | 19 | −0.002 |
| 0.063–0.100 | 39 | +0.304 | 18 | −0.045 |

In the lowest band both groups receive comparable positive reward (3.0 vs 2.0 correct
rollouts per prompt) yet differ tenfold in improvement. So the failure is not "harder
prompts" and not "no learning signal".

### Gradient similarity

120 prompts (40 per group) × 200 rollouts at the initial checkpoint, per-prompt GRPO
gradients over correct rollouts only (LoRA, r=16), 6903 pairwise cosine similarities.

| block | n pairs | mean cos | median |
|---|---|---|---|
| easy × easy | 946 | 0.724 | 0.751 |
| easy × learnable | 1760 | 0.691 | 0.720 |
| learnable × learnable | 780 | 0.666 | 0.690 |
| easy × unlearnable | 1496 | 0.622 | 0.634 |
| learnable × unlearnable | 1360 | 0.600 | 0.615 |
| **unlearnable × unlearnable** | 561 | **0.551** | **0.550** |

Monotone ordering, reproducing the paper's gradient-outlier claim: unlearnable prompts
have the lowest gradient agreement with every group *including themselves*. They do not
form a coherent direction that a shared update could satisfy. Heatmap:
`figures/grad_sim_heatmap.png`.

## Verifier reliability

`classification/finalize_unlearnable_example_id.py` treats prompts that never received a
positive reward as *verifier failures* (the `no_reward` group) rather than learning
failures, so it matters how often the verifier actually misgrades.

Measured over the first ~27,000 training rollouts of the seed-1 run: **1** caught
exception in `math_verify`'s grader (a sympy failure comparing two interval unions,
logged as `ERROR:...:Error comparing ...` and returning False), i.e. ~0.004%. It is
caught inside `math_verify` and does not interrupt training.

So on this configuration `no_reward` is dominated by genuinely-never-solved prompts, not
grading bugs. Worth re-measuring on datasets with more set- or interval-valued answers,
where that sympy path is hit far more often.

## Bugs found and fixed

All three are on the sglang inference path and all three broke this reproduction.
Diff is confined to `inference/`.

1. **`--gpu-memory-util` was silently dropped on the sglang path**
   ([`evaluate_model.py:376`](../inference/evaluate_model.py#L376)). It was passed only to
   the vllm generator, so sglang used its own ~0.88 default → CUDA OOM. Now forwarded
   as `mem_fraction_static`. (`--lora` is still ignored on this path; not needed here.)

2. **Model-size dispatch was case-sensitive**
   ([`gen_utils_sglang.py:56`](../inference/gen_utils_sglang.py#L56)). It tests
   `"0.5B" in model_name`, but the README's own checkpoint paths are lowercase
   (`qwen_0.5b_baseline_seed1/global_step_120/actor/huggingface`), so evaluating any RL
   checkpoint raised `ValueError: Unsupported model`. Now compares case-insensitively.

3. **Unbounded decode batch OOMs small GPUs.** `generate_responses()` submits 5000
   prompts per `generate()` call, letting sglang form ~3600-sequence batches; at 151936
   vocab that is a single 2 GiB logits allocation. Added an opt-in
   `SGLANG_MAX_RUNNING_REQUESTS` env cap (192 here). Unset preserves upstream behaviour.

Non-blocking observations, not fixed (they are the upstream code's behaviour, and
changing them would change the published methodology):

4. `save_freq` / `test_freq` count optimizer steps while the termination check counts
   generation batches — see "step accounting" above.
5. The four classification groups overlap despite being described as a partition — see
   "the groups are not disjoint" above.
6. `--lora` is accepted by `inference/test.sh` but ignored on the sglang path (it is
   only forwarded to the vllm generator). Not needed for this walkthrough.

## Running it

```bash
# 0. env (once)
bash repro/setup_env.sh vllm

# 1. model + data
hf download Qwen/Qwen2.5-0.5B --local-dir ../models/Qwen2.5-0.5B
hf download hkust-nlp/SimpleRL-Zoo-Data --repo-type dataset \
    --include "simplelr_qwen_level1to4/*" --local-dir ./data/SimpleRL-Zoo-Data
python repro/make_subset.py

# 2. baseline pass@1 before training
bash repro/eval.sh ../models/Qwen2.5-0.5B qwen_0.5b_initial

# 3. GRPO training (repeat per seed; the paper intersects 3 seeds)
bash repro/train_1gpu.sh 1

# 4. pass@1 after training
bash repro/eval.sh \
    checkpoints/<run>/global_step_124/actor/huggingface qwen_0.5b_run1

# 5. partition into easy / learnable / unlearnable / no_reward
python classification/finalize_unlearnable_example_id.py \
    --run          qwen_0.5b_run1='results/qwen_0.5b_run1_*.json' \
    --reward-stats qwen_0.5b_run1=checkpoints/<run>/prompt_reward_stats.csv \
    --initial-run  qwen_0.5b_initial='results/qwen_0.5b_initial_*.json' \
    --threshold 0.1 --output-dir ./example_ids \
    --output-prefix qwen_0.5b_math_level1to4
```
