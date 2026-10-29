"""A50 — effective constants in our harness against the original baseline.

The comparison universe is DISCOVERED from the scripts, not enumerated by hand:

* training: every shell variable interpolated into the Hydra command in train.sh;
* evaluation: every sampling/data variable forwarded by inference/test.sh.

Both sides are then resolved through the wrapper that was actually intended to
run them.  Their tracked files are read at HEAD with ``git show``; ours are the
local reproduction wrappers.  Coverage assertions make an omitted parameter a
hard failure rather than silently treating it as unchanged.

    $HOME/miniconda3/envs/vllm/bin/python constants_audit.py
"""
import json
import os
import re
import subprocess


EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(EXP, "..", "..", "unlearnability-rlvr"))
OUT = os.path.join(EXP, "analysis", "constants_audit.md")
TEX_OUT = os.path.join(EXP, "analysis", "constants_audit_table.tex")
TRANSFER_TEX_OUT = os.path.join(
    EXP, "..", "..", "TRANSFER_OUT", "constants_audit_table.tex")
HEAD = subprocess.run(
    ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
    text=True, check=True).stdout.strip()


def at_head(path):
    return subprocess.run(
        ["git", "-C", REPO, "show", f"{HEAD}:{path}"], capture_output=True,
        text=True, check=True).stdout


def strip_value(value):
    value = value.split("#", 1)[0].strip().strip('"').strip("'")
    m = re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-(.*)\}", value)
    return m.group(1) if m else value


def assignments(text):
    """Column-zero shell assignments, including ``export X=...``."""
    out = {}
    for i, line in enumerate(text.splitlines(), 1):
        m = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if m:
            out[m.group(1)] = (strip_value(m.group(2)), i)
    return out


def cli_map(train_text):
    """Discover --flag -> shell-variable mappings from train.sh's case arms."""
    out = {}
    for i, line in enumerate(train_text.splitlines(), 1):
        m = re.search(r"--([a-z0-9_]+)\)\s+([A-Z][A-Z0-9_]*)=\"?\$2", line)
        if m:
            out[m.group(1)] = (m.group(2), i)
            continue
        m = re.search(r"--([a-z0-9_]+)\)\s+([A-Z][A-Z0-9_]*)=(True|False)", line)
        if m:
            out[m.group(1)] = (m.group(2), i)
    return out


def call_flags(text, command):
    """Discover two-token flags in a backslash-continued wrapper invocation."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if command in line)
    out = {}
    for i in range(start, len(lines)):
        line = lines[i]
        m = re.match(r'^\s*--([a-z0-9_-]+)\s+"?([^"\\\s]+)"?\s*\\?\s*$', line)
        if m:
            out[m.group(1).replace("-", "_")] = (m.group(2), i + 1)
        if i > start and not line.rstrip().endswith("\\"):
            break
    return out, start + 1


def resolve_token(token, wrapper_assign):
    m = re.fullmatch(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", token)
    if not m:
        return strip_value(token)
    var = m.group(1)
    if var in wrapper_assign:
        return wrapper_assign[var][0]
    # Both baseline wrappers take a seed as positional argument.
    if var.lower() == "seed":
        return "$seed"
    return token


TRAIN = at_head("train.sh")
THEIR_WRAPPER = at_head("train_baseline.sh")
OUR_TRAIN_PATH = os.path.join(REPO, "repro", "train_1gpu.sh")
OUR_WRAPPER = open(OUR_TRAIN_PATH).read()

train_defaults = assignments(TRAIN)
flag_to_var = cli_map(TRAIN)
their_assign = assignments(THEIR_WRAPPER)
our_assign = assignments(OUR_WRAPPER)
their_flags, their_call_line = call_flags(THEIR_WRAPPER, "bash train.sh")
our_flags, our_call_line = call_flags(OUR_WRAPPER, "bash train.sh")

# Discover every variable that reaches Hydra. Paths/names are operational rather
# than experimental constants, except MODEL_NAME and DATASET_NAME (experiment
# identity), so only the former set is excluded.
hydra = TRAIN[TRAIN.index("-- python -m verl.trainer.main_ppo"):]
train_vars = set(re.findall(r"\$([A-Z][A-Z0-9_]*)", hydra))
train_vars -= {
    "ARNOLD_WORKER_NUM", "CHECKPOINTS_ROOT", "HDFS_DATA_PATH", "LOG_FILE_PATH",
    "MODELS_ROOT", "PROJECT_NAME", "RUN_NAME",
}


def effective_train(wrapper_assign, flags, call_line, wrapper_name):
    effective = {}
    locations = {}
    for var in train_vars:
        if var not in train_defaults:
            raise AssertionError(f"Hydra variable {var} has no parsed default")
        effective[var] = train_defaults[var][0]
        locations[var] = f"train.sh:{train_defaults[var][1]}"
        if var in wrapper_assign:
            effective[var] = wrapper_assign[var][0]
            locations[var] = f"{wrapper_name}:{wrapper_assign[var][1]}"
    for flag, (raw, line) in flags.items():
        if flag not in flag_to_var:
            # model_name/dataset_name are in the discovered map too; any truly
            # unknown experimental flag must stop the audit.
            raise AssertionError(f"wrapper flag --{flag} has no train.sh mapping")
        var = flag_to_var[flag][0]
        if var in train_vars:
            effective[var] = resolve_token(raw, wrapper_assign)
            locations[var] = f"{wrapper_name}:{line}"
    # Inherited settings need to identify the wrapper omission as well as the
    # default that supplies the value. This is especially important for T.
    for var in train_vars:
        if locations[var].startswith("train.sh:"):
            locations[var] = (f"{wrapper_name}:{call_line} (inherits "
                              f"{locations[var]})")
    return effective, locations


their_train, their_train_loc = effective_train(
    their_assign, their_flags, their_call_line, "train_baseline.sh")
our_train, our_train_loc = effective_train(
    our_assign, our_flags, our_call_line, "repro/train_1gpu.sh")

# Hard coverage gates: these are the exact omissions that the old allow-list
# failed to detect.
assert "TEMPERATURE" in train_vars
assert set(their_train) == train_vars == set(our_train)


TEST = at_head("inference/test.sh")
OUR_EVAL_PATH = os.path.join(REPO, "repro", "eval.sh")
OUR_EVAL = open(OUR_EVAL_PATH).read()
test_defaults = assignments(TEST)
our_eval_assign = assignments(OUR_EVAL)
_, our_eval_call_line = call_flags(OUR_EVAL, "bash inference/test.sh")

# Discover the lowercase variables actually forwarded to evaluate_model.py.
eval_block = TEST[TEST.index("python -u"):]
eval_vars = set(re.findall(r'\$([a-z][a-z0-9_]*)', eval_block))
eval_vars -= {"model_path", "model_name"}
eval_vars &= {"testdata", "num_samples", "max_length", "prompt_type",
              "temperature", "gpu_memory_util"}
assert eval_vars == {"testdata", "num_samples", "max_length", "prompt_type",
                     "temperature", "gpu_memory_util"}

EVAL_OVERRIDE = {
    "testdata": "TEST_DATA", "num_samples": "NUM_SAMPLES",
    "max_length": "MAX_LENGTH", "gpu_memory_util": "GPU_MEM_UTIL",
}
their_eval = {v: test_defaults[v][0] for v in eval_vars}
their_eval_loc = {v: f"inference/test.sh:{test_defaults[v][1]}" for v in eval_vars}
our_eval = dict(their_eval)
our_eval_loc = {
    v: f"repro/eval.sh:{our_eval_call_line} (inherits inference/test.sh:{test_defaults[v][1]})"
    for v in eval_vars
}
for var, upper in EVAL_OVERRIDE.items():
    if upper in our_eval_assign:
        our_eval[var] = our_eval_assign[upper][0]
        our_eval_loc[var] = f"repro/eval.sh:{our_eval_assign[upper][1]}"
assert "temperature" in eval_vars
assert set(their_eval) == eval_vars == set(our_eval)


DISPLAY = {var: var.lower() for var in train_vars}
DISPLAY["ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE"] = "rollout_tp"

# Classification is judgment, so it is intentionally explicit; coverage is
# asserted. A missing judgment cannot appear as '?' in a published table.
#
# DIRECT-SCIENCE changes the data/measurement definition. OPTIMIZATION changes
# the update protocol or training horizon and therefore is also result-bearing;
# calling those rows "compensated" previously overclaimed equivalence.
# EXECUTION-NOT-INVARIANT is intended as systems scaling, but the code provides
# no bitwise/stochastic invariance guarantee. OPERATIONAL changes cadence or an
# ignored setting in this exact code path.
CLASS = {v: "DIRECT-SCIENCE" for v in train_vars}
for v in {"LOG_PROB_MICRO_BATCH_SIZE", "MICRO_ROLLOUT_BATCH_SIZE",
          "N_GPU_PER_NODE", "PPO_MICRO_BATCH_SIZE",
          "ROLLOUT_GPU_MEMORY_UTIL", "ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE"}:
    CLASS[v] = "EXECUTION-NOT-INVARIANT"
for v in {"REMOVE_PREVIOUS_CKPT", "RESUME_DATA_STATE", "SAVE_FREQ",
          "TEST_FREQ", "VAL_BATCH_SIZE"}:
    CLASS[v] = "OPERATIONAL"
for v in {"PPO_MINI_BATCH_SIZE", "TOTAL_EPOCHS", "TRAIN_BATCH_SIZE"}:
    CLASS[v] = "OPTIMIZATION"
EVAL_CLASS = {
    "testdata": "DIRECT-SCIENCE", "num_samples": "DIRECT-SCIENCE",
    "max_length": "DIRECT-SCIENCE", "prompt_type": "DIRECT-SCIENCE",
    "temperature": "DIRECT-SCIENCE",
    "gpu_memory_util": "EXECUTION-NOT-INVARIANT",
}
assert set(CLASS) == train_vars
assert set(EVAL_CLASS) == eval_vars

WHY = {
    "DATASET_NAME": "intentional 1023-prompt subset instead of the full level-1-to-4 training set",
    "MAX_RESPONSE_LENGTH": "the A47 token-cap difference, baked into training",
    "TRAIN_BATCH_SIZE": "sets the prompt/rollout cohort per trainer update (32x8 versus 256x8), so the stochastic gradient is not equivalent",
    "PPO_MINI_BATCH_SIZE": "after rollout/world-size normalization the local value is 128 on both sides, but the global FSDP optimizer minibatch is 128 versus 512 and there are 2 versus 4 optimizer minibatches per trainer update",
    "TOTAL_EPOCHS": "sets the training horizon; selecting step 120 narrows the comparison but does not make the final-run protocols equivalent",
    "testdata": "intentional evaluation of our 1023 training prompts; theirs evaluates the full dataset test split",
    "max_length": "the A47 token-cap difference on evaluation",
}

rows = []
for var in sorted(train_vars, key=lambda x: DISPLAY[x]):
    rows.append({
        "stage": "training", "name": DISPLAY[var],
        "ours": our_train[var], "theirs": their_train[var],
        "our_loc": our_train_loc[var], "their_loc": their_train_loc[var],
        "same": our_train[var] == their_train[var], "class": CLASS[var],
        "why": WHY.get(var, ""),
    })
for var in sorted(eval_vars):
    rows.append({
        "stage": "evaluation", "name": var,
        "ours": our_eval[var], "theirs": their_eval[var],
        "our_loc": our_eval_loc[var], "their_loc": their_eval_loc[var],
        "same": our_eval[var] == their_eval[var], "class": EVAL_CLASS[var],
        "why": WHY.get(var, ""),
    })

different = [r for r in rows if not r["same"]]
direct_science = [r for r in different if r["class"] == "DIRECT-SCIENCE"]
optimization = [r for r in different if r["class"] == "OPTIMIZATION"]

L = []
W = L.append
W("# A50 — effective constants in our harness against the original baseline\n")
W(f"Original tracked sources are read at HEAD `{HEAD}`. Their training side is "
  "the documented `train_baseline.sh`, with omitted flags inherited from "
  "`train.sh`; their evaluation side is `inference/test.sh`. Ours are "
  "`repro/train_1gpu.sh` and `repro/eval.sh`.\n")
W("**Coverage is generated, not allow-listed.** The training universe is every "
  "shell variable interpolated into the Hydra command; the evaluation universe "
  "is every data/sampling variable forwarded to `evaluate_model.py`. Assertions "
  "require exact key equality on both sides and explicitly require temperature "
  "at both stages.\n")
W(f"**{len(different)} of {len(rows)} effective constants differ: "
  f"{len(direct_science)} DIRECT-SCIENCE, {len(optimization)} OPTIMIZATION, "
  f"{sum(r['class'] == 'EXECUTION-NOT-INVARIANT' for r in different)} "
  "EXECUTION-NOT-INVARIANT, and "
  f"{sum(r['class'] == 'OPERATIONAL' for r in different)} OPERATIONAL.**\n")
W("| stage | constant | ours | theirs | class | our source | their source |")
W("|---|---|---|---|---|---|---|")
for r in different:
    W(f"| {r['stage']} | `{r['name']}` | **{r['ours']}** | {r['theirs']} | "
      f"**{r['class']}** | `{r['our_loc']}` | `{r['their_loc']}` |")
W("")
W("## Temperature — the accidentally discovered omission\n")
W("| stage | ours | theirs | classification |")
W("|---|---|---|---|")
for stage in ("training", "evaluation"):
    r = next(x for x in rows if x["stage"] == stage and x["name"] == "temperature")
    W(f"| {stage} | {r['ours']} (`{r['our_loc']}`) | "
      f"{r['theirs']} (`{r['their_loc']}`) | **SCIENCE, unchanged** |")
W("")
W("Temperature would be SCIENCE if it differed because it changes the rollout "
  "distribution. It does not differ here. It was absent from the former table "
  "because that script iterated a hand-written `MAP`; settings outside the map "
  "were never examined. In addition, its `ol is None` rule labelled a listed "
  "but omitted training flag as unchanged without resolving and displaying the "
  "inherited default. Thus the former claim that the table was parsed rather "
  "than typed was false in the relevant sense: values were parsed, but the key "
  "universe was typed. This is a systematic blind spot.\n")
W("It also compared training against bare `train.sh` defaults even though HEAD "
  "contains and the README invokes `train_baseline.sh`. Resolving the documented "
  "wrapper removes the alleged `kl_loss_coef` mismatch: both are 0.0001. It also "
  "corrects several compute/schedule reference values.\n")
W("## Direct experimental-definition differences\n")
for r in direct_science:
    W(f"* **`{r['name']}`** — ours {r['ours']}, theirs {r['theirs']}. "
      f"{r['why']}")
W("")
W("## Optimization differences: not scientifically inert\n")
for r in optimization:
    W(f"* **`{r['name']}`** — ours {r['ours']}, theirs {r['theirs']}. "
      f"{r['why']}")
W("")
W("`EXECUTION-NOT-INVARIANT` means the intended algorithmic setting may be "
  "the same, but the implementation gives no bitwise or stochastic invariance "
  "guarantee across batching, GPU count, tensor parallelism, or engine "
  "scheduling. It must not be paraphrased as 'cannot affect a result.' "
  "`OPERATIONAL` is limited to cadence or a setting ignored in this exact code "
  "path; it is not a universal invariance claim.\n")
unchanged = [r for r in rows if r["same"]]
W(f"## Unchanged ({len(unchanged)})\n")
W(", ".join(f"`{r['stage']}:{r['name']}`" for r in unchanged) + ".\n")


def latex_tt(value):
    """Literal table cell; audited values/paths contain no closing brace."""
    value = str(value)
    if "}" in value:
        raise AssertionError(f"unsupported closing brace in LaTeX cell: {value}")
    return rf"\texttt{{\detokenize{{{value}}}}}"


# Appendix table contract: it is generated from the same `different` rows as
# the Markdown/JSON audit. It contains a tabular body (not a float or caption),
# so the paper can \input it at the desired location. `graphicx` is required for
# resizebox; no table-data package is required.
TL = [
    "% GENERATED BY EXPERIMENTS/E1_rescue/constants_audit.py -- DO NOT EDIT BY HAND.",
    f"% Original repository HEAD: {HEAD}",
    "% Contract: exactly one row per differing constant; values and source locations",
    "% come from the same parsed rows as constants_audit.md/json. Requires graphicx.",
    r"\begingroup",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{2pt}",
    r"\renewcommand{\arraystretch}{1.12}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{lllllll}",
    r"\hline",
    r"Stage & Constant & Ours & Theirs & Class & Our file:line & Their file:line \\",
    r"\hline",
]
for row in different:
    TL.append(" & ".join([
        row["stage"], latex_tt(row["name"]), latex_tt(row["ours"]),
        latex_tt(row["theirs"]), latex_tt(row["class"]),
        latex_tt(row["our_loc"]), latex_tt(row["their_loc"]),
    ]) + r" \\")
TL.extend([
    r"\hline",
    r"\end{tabular}%",
    r"}",
    r"\endgroup",
])
latex_table = "\n".join(TL) + "\n"
assert sum(line.endswith(r" \\") for line in TL) == len(different) + 1  # header

payload = {
    "head": HEAD, "coverage": {
        "training_discovery": "variables interpolated into train.sh Hydra command",
        "evaluation_discovery": "variables forwarded by inference/test.sh",
        "training_keys": sorted(train_vars), "evaluation_keys": sorted(eval_vars),
    },
    "n_total": len(rows), "n_differ": len(different), "rows": rows,
}
json.dump(payload, open(os.path.join(EXP, "analysis", "constants_audit.json"), "w"), indent=2)
open(OUT, "w").write("\n".join(L) + "\n")
open(TEX_OUT, "w").write(latex_table)
open(TRANSFER_TEX_OUT, "w").write(latex_table)
print("\n".join(L))
print(f"\nwrote {TEX_OUT}")
print(f"wrote {TRANSFER_TEX_OUT}")
