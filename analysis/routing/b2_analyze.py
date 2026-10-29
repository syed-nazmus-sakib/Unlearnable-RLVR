"""B2 real-generation certification analysis (A82 protocol).

Each independent execution is one fresh 64-draw pass over all 1023 prompts.
Arms consume prefixes of that stream; certified arms take rounds of 8 until
C_x >= delta or 64. All primary metrics are computed against the REFINED
oracle (reference_cache_v2). Constants frozen in REGISTERED.md amendment A82.
"""
import glob
import json

import numpy as np

import route_core as rc

REF = "analysis/reference_cache_v2.json"
EXEC_PATTERNS = {
    "exec1": "../../unlearnability-rlvr/results/b2_exec1_*_101_qwen.json",
    "exec2": "../../unlearnability-rlvr/results/b2_exec2_*_202_qwen.json",
}
OUT = "analysis/b2_results.json"
DELTAS = rc.DELTAS
N0 = 8
DN = 8
NMAX = 64
BOUNDARY_BINS = [("d<0.05", 0.0, 0.05), ("0.05<=d<0.10", 0.05, 0.10),
                 ("0.10<=d<0.20", 0.10, 0.20), ("d>=0.20", 0.20, 1.01)]


def load_stream(pattern):
    paths = sorted(glob.glob(pattern))
    assert len(paths) == 1, f"expected exactly one file for {pattern}: {paths}"
    per_q = {}
    for rec in json.load(open(paths[0])):
        per_q.setdefault(int(rec["question_id"]), []).append(
            int(rec["verification"]["score"]))
    return paths[0], per_q


def load_refined():
    d = json.load(open(REF))
    qids = sorted(int(q) for q in d["prompts"])
    k_tot = np.array([d["prompts"][str(q)]["kA"] + d["prompts"][str(q)]["kB"]
                      + d["prompts"][str(q)]["kC"] for q in qids], dtype=float)
    n_tot = np.array([d["prompts"][str(q)]["n_tot"] for q in qids], dtype=float)
    return np.array(qids), k_tot, n_tot


def prefix_actions(stream_lists, order, n_fixed):
    k = np.array([sum(stream_lists[q][:n_fixed]) for q in order], dtype=float)
    acts = rc.act_band(k, np.full(len(order), n_fixed))
    return acts, np.full(len(order), float(n_fixed))


def certified_actions(stream_lists, order, p_true, prior, delta):
    a0, b0 = prior
    P = len(order)
    k = np.array([sum(stream_lists[q][:N0]) for q in order], dtype=np.int64)
    n = np.full(P, N0, dtype=np.int64)
    lens = np.array([len(stream_lists[q]) for q in order])
    assert (lens >= NMAX).all(), "execution stream shorter than cap"
    active = np.ones(P, dtype=bool)
    while True:
        idx = np.where(active)[0]
        conf, _ = rc.confidence(k[idx], n[idx], a0, b0)
        done = conf >= delta
        at_cap = n[idx] >= NMAX
        stop_idx = idx[done | at_cap]
        active[stop_idx] = False
        go = idx[(~done) & (~at_cap)]
        if go.size == 0:
            break
        for i in go:
            q = order[i]
            lo, hi = int(n[i]), min(int(n[i]) + DN, NMAX)
            k[i] += sum(stream_lists[q][lo:hi])
            n[i] += hi - lo
    _, inb = rc.confidence(k, n, a0, b0)
    return rc.act_from_mass(inb), n


def main():
    qids, k_tot, n_tot = load_refined()
    p_ref = k_tot / n_tot
    ref = rc.ref_action(p_ref)
    v_replay = p_ref * (1.0 - p_ref)
    vbar = float(v_replay.mean())
    prior = rc.fit_beta_moment(p_ref)
    post_ref = rc.band_mass(k_tot, n_tot, *prior)
    mass_on_ref = np.where(ref == 1, post_ref, 1 - post_ref)
    unamb = mass_on_ref >= rc.ORACLE_UNCERTAIN_MAXP

    streams = {}
    for name, pattern in EXEC_PATTERNS.items():
        path, per_q = load_stream(pattern)
        assert set(per_q.keys()) == set(int(q) for q in qids), name
        assert all(len(v) == NMAX for v in per_q.values()), name
        streams[name] = {q: per_q[q] for q in sorted(per_q)}
        print(f"{name}: {path}")

    def metrics(a1, a2, n_used):
        acc1 = float((a1 == ref).mean())
        accu1 = float((a1[unamb] == ref[unamb]).mean())
        false = float(((a1 == 1) & (ref == 0)).mean())
        missed = float(((a1 == 0) & (ref == 1)).mean())
        false2 = float(((a2 == 1) & (ref == 0)).mean())
        missed2 = float(((a2 == 0) & (ref == 1)).mean())
        v_act1 = np.where(a1 == 1, v_replay, vbar)
        v_ref = np.where(ref == 1, v_replay, vbar)
        regret = float(np.abs(v_act1 - v_ref).mean())
        agree = float((a1 == a2).mean())
        inter = float(np.sum((a1 == 1) & (a2 == 1)))
        union = float(np.sum((a1 == 1) | (a2 == 1)))
        jac = inter / union if union else 1.0
        return {
            "mean_N": float(np.mean(n_used)),
            "err_all": 1 - acc1,
            "err_unamb": 1 - accu1,
            "false_route": (false + false2) / 2,
            "missed_route": (missed + missed2) / 2,
            "regret_mean": regret,
            "action_agreement": agree,
            "jaccard_rerun": jac,
        }

    rows = []
    aux = {}
    for n_fixed in (16, 32):
        a1, n1 = prefix_actions(streams["exec1"], qids, n_fixed)
        a2, _ = prefix_actions(streams["exec2"], qids, n_fixed)
        rows.append({"arm": "fixed_plugin", "config": f"N={n_fixed}",
                     **metrics(a1, a2, np.full(len(qids), float(n_fixed)))})

    stop_fracs = {}
    for delta in DELTAS:
        a1, n1 = certified_actions(streams["exec1"], qids, p_ref, prior, delta)
        a2, n2 = certified_actions(streams["exec2"], qids, p_ref, prior, delta)
        rows.append({"arm": "certified_seq", "config": f"delta={delta}",
                     **metrics(a1, a2, (n1 + n2) / 2)})
        levels = list(range(N0, NMAX + 1, DN))
        stop_fracs[f"delta={delta}"] = {
            str(l): float((n1 <= l).mean()) for l in levels}
        aux[f"certified_seq|delta={delta}"] = {"n_used_exec1": n1.tolist()}

    dist = np.minimum(np.abs(p_ref - rc.BAND[0]), np.abs(p_ref - rc.BAND[1]))
    boundary_table = {}
    for r in rows:
        key = f"{r['arm']}|{r['config']}"
        boundary_table[key] = {}
        if r["arm"] == "fixed_plugin":
            nf = int(r["config"].split("=")[1])
            a1, _ = prefix_actions(streams["exec1"], qids, nf)
        else:
            dl = float(r["config"].split("=")[1])
            a1, _ = certified_actions(streams["exec1"], qids, p_ref, prior, dl)
        errs = (a1 != ref)
        for label, lo, hi in BOUNDARY_BINS:
            m = (dist >= lo) & (dist < hi)
            boundary_table[key][label] = {
                "n_prompts": int(m.sum()),
                "err": float(errs[m].mean()),
            }

    fixed32 = next(r for r in rows if r["config"] == "N=32")
    criterion_rows = []
    overall_pass = False
    passing = []
    for r in rows:
        if r["arm"] != "certified_seq":
            continue
        checks = {
            "err_lower_than_fixed32": r["err_all"] < fixed32["err_all"],
            "cost_le_32": r["mean_N"] <= 32.0,
            "agreement_higher_than_fixed32":
                r["action_agreement"] > fixed32["action_agreement"],
            "unamb_improvement_holds": r["err_unamb"] < fixed32["err_unamb"],
        }
        ok = all(checks.values())
        criterion_rows.append({"delta": r["config"], **checks, "all_pass": ok})
        if ok:
            overall_pass = True
            passing.append(r["config"])

    result = {
        "meta": {
            "protocol": "A82",
            "band": list(rc.BAND),
            "prior_refined": {"alpha0": prior[0], "beta0": prior[1]},
            "vbar_refined": vbar,
            "n_unambiguous": int(unamb.sum()),
            "rounds": {"n0": N0, "dn": DN, "nmax": NMAX},
            "exec_files": {k: load_stream(v)[0] for k, v in EXEC_PATTERNS.items()},
        },
        "rows": rows,
        "stopping_fractions_certified": stop_fracs,
        "boundary_bins": boundary_table,
        "prospective_criterion": {
            "definition": ("err<fixed32 AND meanN<=32 AND agreement>fixed32 "
                           "AND unamb improvement"),
            "per_delta": criterion_rows,
            "verdict": "PASS" if overall_pass else "FAIL",
            "passing_deltas": passing,
        },
    }
    json.dump(result, open(OUT, "w"), indent=1)

    hdr = (f"{'arm':<16} {'cfg':<12} {'meanN':>6} {'errAll':>7} "
           f"{'errUnamb':>8} {'agree':>6} {'J':>6} {'false':>6} "
           f"{'missed':>7} {'regret':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["mean_N"]):
        print(f"{r['arm']:<16} {r['config']:<12} {r['mean_N']:>6.1f} "
              f"{r['err_all']:>7.3f} {r['err_unamb']:>8.3f} "
              f"{r['action_agreement']:>6.3f} {r['jaccard_rerun']:>6.3f} "
              f"{r['false_route']:>6.3f} {r['missed_route']:>7.3f} "
              f"{r['regret_mean']:>8.4f}")
    print()
    for c in criterion_rows:
        print(c)


if __name__ == "__main__":
    main()
