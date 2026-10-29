# A50 — effective constants in our harness against the original baseline

Original tracked sources are read at HEAD `1dce7fec92e7685929a5efbe73ff2091e7d474a1`. Their training side is the documented `train_baseline.sh`, with omitted flags inherited from `train.sh`; their evaluation side is `inference/test.sh`. Ours are `repro/train_1gpu.sh` and `repro/eval.sh`.

**Coverage is generated, not allow-listed.** The training universe is every shell variable interpolated into the Hydra command; the evaluation universe is every data/sampling variable forwarded to `evaluate_model.py`. Assertions require exact key equality on both sides and explicitly require temperature at both stages.

**17 of 47 effective constants differ: 4 DIRECT-SCIENCE, 3 OPTIMIZATION, 7 EXECUTION-NOT-INVARIANT, and 3 OPERATIONAL.**

| stage | constant | ours | theirs | class | our source | their source |
|---|---|---|---|---|---|---|
| training | `dataset_name` | **simplelr_qwen_level1to4_sub1k** | simplelr_qwen_level1to4 | **DIRECT-SCIENCE** | `repro/train_1gpu.sh:77` | `train_baseline.sh:29` |
| training | `log_prob_micro_batch_size` | **4** | 128 | **EXECUTION-NOT-INVARIANT** | `repro/train_1gpu.sh:82` | `train_baseline.sh:64` |
| training | `max_response_length` | **1024** | 5120 | **DIRECT-SCIENCE** | `repro/train_1gpu.sh:78` | `train_baseline.sh:51` |
| training | `micro_rollout_batch_size` | **128** | 1024 | **EXECUTION-NOT-INVARIANT** | `repro/train_1gpu.sh:83` | `train_baseline.sh:59` |
| training | `n_gpu_per_node` | **1** | 4 | **EXECUTION-NOT-INVARIANT** | `repro/train_1gpu.sh:86` | `train_baseline.sh:65` |
| training | `ppo_micro_batch_size` | **2** | 32 | **EXECUTION-NOT-INVARIANT** | `repro/train_1gpu.sh:81` | `train_baseline.sh:63` |
| training | `ppo_mini_batch_size` | **16** | 64 | **OPTIMIZATION** | `repro/train_1gpu.sh:80` | `train_baseline.sh:68` |
| training | `rollout_gpu_memory_util` | **0.45** | 0.75 | **EXECUTION-NOT-INVARIANT** | `repro/train_1gpu.sh:87` | `train_baseline.sh:56` |
| training | `rollout_tp` | **1** | 2 | **EXECUTION-NOT-INVARIANT** | `repro/train_1gpu.sh:85` | `train_baseline.sh:57` |
| training | `save_freq` | **40** | 20 | **OPERATIONAL** | `repro/train_1gpu.sh:91` | `train_baseline.sh:58` |
| training | `test_freq` | **100000** | 5 | **OPERATIONAL** | `repro/train_1gpu.sh:92` | `train_baseline.sh:49 (inherits train.sh:41)` |
| training | `total_epochs` | **12** | 50 | **OPTIMIZATION** | `repro/train_1gpu.sh:94` | `train_baseline.sh:60` |
| training | `train_batch_size` | **32** | 256 | **OPTIMIZATION** | `repro/train_1gpu.sh:79` | `train_baseline.sh:52` |
| training | `val_batch_size` | **200** | 1000 | **OPERATIONAL** | `repro/train_1gpu.sh:93` | `train_baseline.sh:49 (inherits train.sh:19)` |
| evaluation | `gpu_memory_util` | **0.60** | 0.75 | **EXECUTION-NOT-INVARIANT** | `repro/eval.sh:42` | `inference/test.sh:28` |
| evaluation | `max_length` | **1024** | 5120 | **DIRECT-SCIENCE** | `repro/eval.sh:41` | `inference/test.sh:25` |
| evaluation | `testdata` | **simplelr_qwen_level1to4_sub1k@train** | simplelr_qwen_level1to4@test | **DIRECT-SCIENCE** | `repro/eval.sh:39` | `inference/test.sh:24` |

## Temperature — the accidentally discovered omission

| stage | ours | theirs | classification |
|---|---|---|---|
| training | 1.0 (`repro/train_1gpu.sh:75 (inherits train.sh:30)`) | 1.0 (`train_baseline.sh:49 (inherits train.sh:30)`) | **SCIENCE, unchanged** |
| evaluation | 0.7 (`repro/eval.sh:44 (inherits inference/test.sh:29)`) | 0.7 (`inference/test.sh:29`) | **SCIENCE, unchanged** |

Temperature would be SCIENCE if it differed because it changes the rollout distribution. It does not differ here. It was absent from the former table because that script iterated a hand-written `MAP`; settings outside the map were never examined. In addition, its `ol is None` rule labelled a listed but omitted training flag as unchanged without resolving and displaying the inherited default. Thus the former claim that the table was parsed rather than typed was false in the relevant sense: values were parsed, but the key universe was typed. This is a systematic blind spot.

It also compared training against bare `train.sh` defaults even though HEAD contains and the README invokes `train_baseline.sh`. Resolving the documented wrapper removes the alleged `kl_loss_coef` mismatch: both are 0.0001. It also corrects several compute/schedule reference values.

## Direct experimental-definition differences

* **`dataset_name`** — ours simplelr_qwen_level1to4_sub1k, theirs simplelr_qwen_level1to4. intentional 1023-prompt subset instead of the full level-1-to-4 training set
* **`max_response_length`** — ours 1024, theirs 5120. the A47 token-cap difference, baked into training
* **`max_length`** — ours 1024, theirs 5120. the A47 token-cap difference on evaluation
* **`testdata`** — ours simplelr_qwen_level1to4_sub1k@train, theirs simplelr_qwen_level1to4@test. intentional evaluation of our 1023 training prompts; theirs evaluates the full dataset test split

## Optimization differences: not scientifically inert

* **`ppo_mini_batch_size`** — ours 16, theirs 64. after rollout/world-size normalization the local value is 128 on both sides, but the global FSDP optimizer minibatch is 128 versus 512 and there are 2 versus 4 optimizer minibatches per trainer update
* **`total_epochs`** — ours 12, theirs 50. sets the training horizon; selecting step 120 narrows the comparison but does not make the final-run protocols equivalent
* **`train_batch_size`** — ours 32, theirs 256. sets the prompt/rollout cohort per trainer update (32x8 versus 256x8), so the stochastic gradient is not equivalent

`EXECUTION-NOT-INVARIANT` means the intended algorithmic setting may be the same, but the implementation gives no bitwise or stochastic invariance guarantee across batching, GPU count, tensor parallelism, or engine scheduling. It must not be paraphrased as 'cannot affect a result.' `OPERATIONAL` is limited to cadence or a setting ignored in this exact code path; it is not a universal invariance claim.

## Unchanged (30)

`training:clip_ratio`, `training:entropy_beta`, `training:entropy_coefficient`, `training:entropy_ctrl_type`, `training:entropy_min_coefficient`, `training:grad_clip`, `training:high_clip_ratio`, `training:keep_all_negative`, `training:keep_old_advantage_before_downsample`, `training:kl_coef`, `training:kl_loss_coef`, `training:kl_loss_type`, `training:learning_rate`, `training:low_clip_ratio`, `training:max_prompt_length`, `training:model_name`, `training:oversample_n`, `training:remove_clip`, `training:remove_previous_ckpt`, `training:replay`, `training:replay_n`, `training:resume_data_state`, `training:rollout_n`, `training:rollout_name`, `training:seed`, `training:temperature`, `training:use_kl_loss`, `evaluation:num_samples`, `evaluation:prompt_type`, `evaluation:temperature`.

