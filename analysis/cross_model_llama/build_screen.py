"""A34.2 gate — build the 800-prompt MATH-Hard screening set.

Prompt construction is fixed by A34.6: the `question` field carries the SAME
instruction as the 0.5B run, and Llama's own chat template is applied later by
its tokenizer. Same instruction, model-native chat format, so the gate's pass
rates and the main run's are the same quantity.

The eval harness reads `question` (stripping a Qwen chat template if it finds
one, which we do not add) and re-wraps it, so nothing Qwen-specific is baked in
here.

  $HOME/miniconda3/envs/vllm/bin/python build_screen.py
"""
import json
import os

import numpy as np
from huggingface_hub import hf_hub_download

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = "/media/rmedu/3039-8E90/LLM/unlearnability-rlvr"
N_SCREEN = 800
SEED = 20260812

# A39.3: at most TWO revisions, both recorded here so the sequence is visible
# and a third cannot be slipped in. ATTEMPT selects which is built.
#
# v1 is Chen's instruction verbatim, used for the 0.5B runs (A34.6). It gave a
# 25.26 % [invalid] share on Llama-3.2-3B, correlated with pass rate at -0.464.
#
# v2 was written from OVERALL failure diagnostics only -- never from per-tier
# rates (A39.2). Those diagnostics: 74.90 % of rollouts contain "\boxed" at all
# and 25.26 % are invalid, and only 0.63 % of invalid rollouts contain "\boxed"
# anywhere. So the failure is NOT a truncated box -- the model simply never
# emits one. Invalid rollouts are also longer (2024 vs 1434 chars) with 15.2 %
# at the length cap, so brevity is the second lever. Code fences appear in just
# 4.34 % of rollouts and are a minor contributor, addressed only indirectly.
INSTRUCTIONS = {
    1: "\nPlease reason step by step, and put your final answer within \\boxed{}.",
    2: ("\nSolve the problem. Keep your reasoning brief. End your response with "
        "the final answer in the form \\boxed{answer} -- exactly one \\boxed{}, "
        "containing only the final answer, with nothing after it."),
}
# CORRECTION (A39.6): the instruction does NOT come from this file. The harness
# strips a trailing copy of Chen's instruction from `question`
# (inference/evaluate_model.py:260) and then re-appends the OUTPUT INSTRUCTION
# from the prompt template (inference/config.py). v1 only looked correct because
# our string was identical to the template's, so strip-then-append was a no-op.
# Putting a different string here produced BOTH instructions in the realised
# prompt, ours then Chen's. So the data file always carries Chen's string, which
# the harness strips, and the attempt is selected by --prompt-type instead:
#     attempt 1 -> "qwen"    attempt 2 -> "e4v2"
# The two data files are therefore byte-identical; only the template differs.
ATTEMPT = int(os.environ.get("E4_ATTEMPT", "1"))
INSTR = INSTRUCTIONS[1]
PROMPT_TYPE = {1: "qwen", 2: "e4v2"}[ATTEMPT]
TAG = "e4_screen800" if ATTEMPT == 1 else f"e4_screen800_v{ATTEMPT}"
DST = os.path.join(REPO, "data", TAG)

SUBJ = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
        "number_theory", "prealgebra", "precalculus"]

rows = []
for s in SUBJ:
    p = hf_hub_download("lighteval/MATH-Hard", f"train/{s}.jsonl", repo_type="dataset")
    for r in json.load(open(p)):          # the .jsonl files are JSON arrays
        rows.append(r)
print(f"MATH-Hard train pool: {len(rows)} problems")
assert all(r.get("level") == "Level 5" for r in rows), "pool is not uniformly Level 5"

rng = np.random.default_rng(SEED)
idx = rng.choice(len(rows), size=min(N_SCREEN, len(rows)), replace=False)
sample = [rows[i] for i in sorted(idx)]


def answer_of(r):
    """MATH-Hard carries the answer inside \\boxed{} in the solution."""
    s = r["solution"]
    k = s.rfind("\\boxed")
    if k < 0:
        return None
    i = s.find("{", k)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j]
        j += 1
    return None


out, dropped = [], 0
for i, r in enumerate(sample):
    a = answer_of(r)
    if a is None:
        dropped += 1
        continue
    out.append({
        "answer": a, "gt_answer": a, "target": a,
        "subject": r.get("type"), "level": r.get("level"),
        "question": r["problem"] + INSTR,
        "data_source": "lighteval/MATH-Hard",
        "prompt": [{"role": "user", "content": r["problem"] + INSTR}],
        "ability": "math",
        "reward_model": {"ground_truth": a, "style": "rule"},
        "extra_info": {"answer": a, "index": int(idx[i]) if i < len(idx) else i,
                       "question": r["problem"], "split": "screen",
                       "level": r.get("level"), "subject": r.get("type")},
    })

os.makedirs(DST, exist_ok=True)
p = os.path.join(DST, "test.jsonl")
with open(p, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
json.dump({"attempt": ATTEMPT, "prompt_type": PROMPT_TYPE,
           "instruction_realised": INSTRUCTIONS[ATTEMPT], "instruction_in_file": INSTR, "seed": SEED, "n_requested": N_SCREEN, "n_written": len(out),
           "dropped_no_boxed_answer": dropped, "pool": len(rows)},
          open(os.path.join(EXP, f"screen_manifest_v{ATTEMPT}.json"), "w"), indent=2)
print(f"wrote {len(out)} prompts -> {p}")
print(f"dropped for no extractable \\boxed answer: {dropped}")
