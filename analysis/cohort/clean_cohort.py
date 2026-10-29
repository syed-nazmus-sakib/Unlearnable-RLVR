"""A70.4 — the clean-cohort recomputation of Section 6's two published numbers.

A70 found that 11 of the 55 natural D_u prompts carry a label defined on TRAINED
runs while their gradients are measured on the INITIAL policy, and that their
initial-policy pass rate contradicts the label. This script recomputes, on the
prompts whose initial behaviour MATCHES their label:

  * the natural-count ratio, cohort B
  * the matched ratio at K=11, cohort B
  * the share of the log gap that matching removes  (the "48.9 %" figure)

all three with intervals, on the SAME bootstrap family as the published ones
(seed 20260808, 5000 replicates, percentile 2.5/97.5) -- see guard_attrition.py
for the natural term and decompose_move.py:boot_ratio for the matched term.

ONE DELIBERATE DIFFERENCE FROM THE PUBLISHED BOOTSTRAPS, because the share is a
function of BOTH terms: within a replicate the easy prompt draw is shared
between the natural and the matched easy means, and likewise for D_u. The
published code drew them in separate calls, which is fine for two separate
intervals and wrong for an interval on their ratio. The full-cohort numbers are
recomputed here under this same procedure so the reader can see the procedure is
not doing the work.

EVERYTHING IS ON COHORT B (28 D_u / 28 learnable / 38 easy). A28.1 forbids
mixing cohorts and A70's first pass did exactly that -- see A70.5.

  $HOME/miniconda3/envs/vllm/bin/python clean_cohort.py
"""
import json
import os
from collections import defaultdict

import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(EXP, "..", ".."))
RES = os.path.join(ROOT, "unlearnability-rlvr", "results")
AN = os.path.join(EXP, "analysis")

GROUPS = ["du71", "learnable", "easy"]
SUBS = {1, 2, 3, 4, 5, 6}
TAU = 0.1
SEED, NBOOT = 20260808, 5000
PCT = [2.5, 97.5]

att = json.load(open(os.path.join(AN, "attenuation.json")))
nat = json.load(open(os.path.join(AN, "figure5_natural_per_prompt.json")))
Ks = sorted({int(k) for g in att for k in att[g]})


def cells(g, K):
    return [e for e in att[g][str(K)] if "per_prompt" in e and e.get("sub") in SUBS]


cohB = {g: sorted(set.intersection(*[set(e["per_prompt"]) for K in Ks
                                     for e in cells(g, K)]), key=int)
        for g in GROUPS}
k11own = {g: sorted(set.intersection(*[set(e["per_prompt"]) for e in cells(g, 11)]),
                    key=int) for g in GROUPS}
natset = {g: sorted(nat[g], key=int) for g in GROUPS}

# ---- the initial-policy guarded pass rate, the SAME pool the gradients use ---
ok, tot = defaultdict(int), defaultdict(int)
for r in json.load(open(os.path.join(RES, "e1_gradsim_initial_GUARDED.json"))):
    q = str(int(r["question_id"]))
    v = r.get("verification", {}) or {}
    s = int(v.get("score", 0)) == 1
    tot[q] += 1
    ok[q] += s and v.get("extracted_answer") != "[invalid]"
rate = {q: ok[q] / tot[q] for q in tot}

# The direction of the mismatch is GROUP-SPECIFIC and must not be shared:
#   du71 / learnable are labelled on TRAINED runs (pass@1 <= tau, > tau) -- the
#     initial policy is a different policy, so a rate ABOVE tau contradicts the
#     unlearnable label. `learnable` is listed for completeness only.
#   easy is labelled on the INITIAL policy itself, so its mismatch is a rate AT
#     OR BELOW tau, and there is no timing gap for it to open (A70.3).
def mismatched(g, ids):
    if g == "easy":
        return [q for q in ids if rate[q] <= TAU]
    return [q for q in ids if rate[q] > TAU]


mm = {"natural": {g: mismatched(g, natset[g]) for g in GROUPS},
      "cohortB": {g: mismatched(g, cohB[g]) for g in GROUPS},
      "k11own": {g: mismatched(g, k11own[g]) for g in GROUPS}}
clean = {g: [q for q in cohB[g] if q not in set(mm["cohortB"][g])] for g in GROUPS}

# ---- the two terms, as matrices, so one prompt draw serves both --------------
NATV = {g: np.array([nat[g][q] for q in cohB[g]], float) for g in ("easy", "du71")}
MATV = {g: np.array([[e["per_prompt"][q] for q in cohB[g]] for e in cells(g, 11)],
                    float) for g in ("easy", "du71")}
KEEP = {g: np.array([q not in set(mm["cohortB"][g]) for q in cohB[g]]) for g in
        ("easy", "du71")}


def stats(ie, idd, se, sd):
    """natural ratio, matched K=11 ratio and the log-gap share for one draw."""
    n_e, n_d = NATV["easy"][ie].mean(), NATV["du71"][idd].mean()
    m_e = MATV["easy"][np.ix_(se, ie)].mean()
    m_d = MATV["du71"][np.ix_(sd, idd)].mean()
    if min(n_d, m_d) < 1e-9:
        return None
    nr, mr = n_e / n_d, m_e / m_d
    if nr <= 1.0 + 1e-9:
        return None                      # share undefined if there is no gap
    return nr, mr, (np.log(nr) - np.log(mr)) / np.log(nr)


def point(sel_e, sel_d):
    ne, nd = len(sel_e), len(sel_d)
    return stats(sel_e, sel_d, np.arange(MATV["easy"].shape[0]),
                 np.arange(MATV["du71"].shape[0]))


def boot(sel_e, sel_d, rng):
    ne, nd = len(sel_e), len(sel_d)
    ns_e, ns_d = MATV["easy"].shape[0], MATV["du71"].shape[0]
    out = []
    for _ in range(NBOOT):
        ie = sel_e[rng.integers(0, ne, ne)]
        idd = sel_d[rng.integers(0, nd, nd)]
        se = rng.integers(0, ns_e, ns_e)
        sd = rng.integers(0, ns_d, ns_d)
        v = stats(ie, idd, se, sd)
        if v is not None:
            out.append(v)
    a = np.array(out)
    return {k: (float(np.percentile(a[:, i], PCT[0])),
                float(np.percentile(a[:, i], PCT[1])))
            for i, k in enumerate(("nat", "matched", "share"))}, len(out)


IDX = {g: np.arange(len(cohB[g])) for g in ("easy", "du71")}
CLEAN_IDX = {g: IDX[g][KEEP[g]] for g in ("easy", "du71")}

rng = np.random.default_rng(SEED)
res = {}
for tag, sel in (("published", IDX), ("clean", CLEAN_IDX)):
    p = point(sel["easy"], sel["du71"])
    ci, nrep = boot(sel["easy"], sel["du71"], rng)
    res[tag] = {"n_easy": int(len(sel["easy"])), "n_du": int(len(sel["du71"])),
                "nat": p[0], "matched": p[1], "share": p[2],
                "ci": ci, "boot_kept": nrep}

# ---- the PAIRED difference: same draw, full cohort, then subset to clean -----
rngp = np.random.default_rng(SEED)
dm, ds = [], []
ns_e, ns_d = MATV["easy"].shape[0], MATV["du71"].shape[0]
ne, nd = len(cohB["easy"]), len(cohB["du71"])
for _ in range(NBOOT):
    ie = rngp.integers(0, ne, ne)
    idd = rngp.integers(0, nd, nd)
    se = rngp.integers(0, ns_e, ns_e)
    sd = rngp.integers(0, ns_d, ns_d)
    ce, cd = ie[KEEP["easy"][ie]], idd[KEEP["du71"][idd]]
    if len(ce) == 0 or len(cd) == 0:
        continue
    A = stats(ie, idd, se, sd)
    B = stats(ce, cd, se, sd)
    if A is None or B is None:
        continue
    dm.append(B[1] - A[1])
    ds.append(B[2] - A[2])
res["paired_delta"] = {
    "matched": [float(np.mean(dm)), float(np.percentile(dm, PCT[0])),
                float(np.percentile(dm, PCT[1]))],
    "share": [float(np.mean(ds)), float(np.percentile(ds, PCT[0])),
              float(np.percentile(ds, PCT[1]))],
    "kept": len(dm)}

# ---- the mismatch shares, all populations in one place ----------------------
res["mismatch"] = {}
for pop, sets in (("natural", natset), ("cohortB", cohB), ("k11own", k11own)):
    res["mismatch"][pop] = {g: {"n": len(sets[g]), "bad": len(mm[pop][g]),
                                "pct": 100 * len(mm[pop][g]) / len(sets[g]),
                                "ids": sorted(mm[pop][g], key=int)}
                            for g in GROUPS}
res["mismatch_rates_du_natural"] = sorted(
    round(rate[q], 4) for q in mm["natural"]["du71"])
res["tau"] = TAU
res["seed"], res["nboot"] = SEED, NBOOT

# ---- A70.5: the 1.5350 / 1.539 difference, and what it actually was ---------
def cellfirst(qe, qd, K):
    e = np.mean([np.mean([c["per_prompt"][q] for q in qe]) for c in cells("easy", K)])
    d = np.mean([np.mean([c["per_prompt"][q] for q in qd]) for c in cells("du71", K)])
    return float(e / d)


def promptfirst(qe, qd, K):
    e = np.mean([np.mean([c["per_prompt"][q] for c in cells("easy", K)]) for q in qe])
    d = np.mean([np.mean([c["per_prompt"][q] for c in cells("du71", K)]) for q in qd])
    return float(e / d)


res["convention"] = {
    "published_cohortB_cellfirst": cellfirst(cohB["easy"], cohB["du71"], 11),
    "published_cohortB_promptfirst": promptfirst(cohB["easy"], cohB["du71"], 11),
    "k11own_cellfirst": cellfirst(k11own["easy"], k11own["du71"], 11),
    "k11own_promptfirst": promptfirst(k11own["easy"], k11own["du71"], 11),
    "k11own_n": {g: len(k11own[g]) for g in GROUPS},
    "easy_only_in_k11own": sorted(set(k11own["easy"]) - set(cohB["easy"]), key=int),
    "du_sets_identical": set(k11own["du71"]) == set(cohB["du71"]),
    "clean_k11own": None}

ck = {g: [q for q in k11own[g] if q not in set(mm["k11own"][g])] for g in GROUPS}
res["convention"]["clean_k11own"] = cellfirst(ck["easy"], ck["du71"], 11)
res["convention"]["clean_k11own_n"] = {g: len(ck[g]) for g in GROUPS}

# Every level A70 printed, reproduced exactly, so the ladder from the published
# number to the primary clean one has no unexplained step. A70's clean levels
# cleaned the D_u side ONLY; the primary here cleans both.
natmean = lambda g, ids: float(np.mean([nat[g][q] for q in ids]))
du_clean_nat = [q for q in natset["du71"] if q not in set(mm["natural"]["du71"])]
du_clean_k11 = [q for q in k11own["du71"] if q not in set(mm["k11own"]["du71"])]
res["ladder"] = [
    ("natural, full included set", len(natset["easy"]), len(natset["du71"]),
     natmean("easy", natset["easy"]) / natmean("du71", natset["du71"])),
    ("natural, full set, D_u side cleaned", len(natset["easy"]), len(du_clean_nat),
     natmean("easy", natset["easy"]) / natmean("du71", du_clean_nat)),
    ("natural, cohort B (published)", len(cohB["easy"]), len(cohB["du71"]),
     res["published"]["nat"]),
    ("natural, cohort B, both sides cleaned", len(clean["easy"]), len(clean["du71"]),
     res["clean"]["nat"]),
    ("matched K=11, K=11-own set", len(k11own["easy"]), len(k11own["du71"]),
     cellfirst(k11own["easy"], k11own["du71"], 11)),
    ("matched K=11, K=11-own, D_u side cleaned", len(k11own["easy"]),
     len(du_clean_k11), cellfirst(k11own["easy"], du_clean_k11, 11)),
    ("matched K=11, cohort B (published)", len(cohB["easy"]), len(cohB["du71"]),
     res["published"]["matched"]),
    ("matched K=11, cohort B, both sides cleaned", len(clean["easy"]),
     len(clean["du71"]), res["clean"]["matched"]),
]

json.dump(res, open(os.path.join(AN, "a70_clean_cohort.json"), "w"), indent=2)

# ---------------------------------- report -----------------------------------
f = lambda t, k: (res[t][k], *res[t]["ci"][k])
print(f"tau = {TAU}, guarded initial pass rate, N=200 pool "
      f"(e1_gradsim_initial_GUARDED.json)\n")
print("MISMATCH SHARES")
print(f"{'population':16s} {'group':11s} {'n':>4s} {'bad':>4s} {'pct':>7s}")
for pop in ("natural", "cohortB", "k11own"):
    for g in GROUPS:
        d = res["mismatch"][pop][g]
        print(f"{pop:16s} {g:11s} {d['n']:4d} {d['bad']:4d} {d['pct']:6.1f}%")
print()
print("COHORT B, PUBLISHED AGAINST CLEAN  (seed %d, %d reps, %s pct)"
      % (SEED, NBOOT, PCT))
print(f"{'':10s} {'n e/du':>9s} {'natural':>26s} {'matched K=11':>26s} {'share':>26s}")
for t in ("published", "clean"):
    r = res[t]
    cell = lambda k, s=1: (f"{r[k]*s:8.4f} [{r['ci'][k][0]*s:.4f},"
                           f" {r['ci'][k][1]*s:.4f}]")
    print(f"{t:10s} {r['n_easy']:4d}/{r['n_du']:<4d} {cell('nat'):>26s} "
          f"{cell('matched'):>26s} {cell('share', 100):>26s}")
d = res["paired_delta"]
print(f"\npaired delta (same draws): matched {d['matched'][0]:+.4f} "
      f"[{d['matched'][1]:+.4f}, {d['matched'][2]:+.4f}]   "
      f"share {100*d['share'][0]:+.2f}pp "
      f"[{100*d['share'][1]:+.2f}, {100*d['share'][2]:+.2f}]")
c = res["convention"]
print(f"\nCONVENTION CHECK (A70.5)")
print(f"  cohort B    cell-first {c['published_cohortB_cellfirst']:.4f}  "
      f"prompt-first {c['published_cohortB_promptfirst']:.4f}")
print(f"  K=11 own    cell-first {c['k11own_cellfirst']:.4f}  "
      f"prompt-first {c['k11own_promptfirst']:.4f}")
print(f"  the two conventions are identical on a fixed cohort; the 0.004 is the "
      f"easy prompts {c['easy_only_in_k11own']}, D_u sets identical: "
      f"{c['du_sets_identical']}")
print(f"  clean on the K=11-own set: {c['clean_k11own']:.4f} "
      f"(n {c['clean_k11own_n']})")
print("\nRECONCILIATION LADDER — every level A70 printed, reproduced")
for name, ne, nd, v in res["ladder"]:
    print(f"  {name:44s} {ne:3d}/{nd:<3d} {v:.4f}")
print(f"\nwrote {os.path.join(AN, 'a70_clean_cohort.json')}")
