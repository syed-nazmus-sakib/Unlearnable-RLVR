"""Merge the oracle top-up pass into a refined reference (A82 protocol)."""
import json

import numpy as np

import route_core as rc

ORIG = "analysis/reference_cache.json"
TOPUP_GLOB = None
OUT_CACHE = "analysis/reference_cache_v2.json"
OUT_STATS = "analysis/refinement_stats.json"


def load_scores_by_qid(paths):
    out = {}
    for path in paths:
        for rec in json.load(open(path)):
            qid = int(rec["question_id"])
            out.setdefault(qid, []).append(int(rec["verification"]["score"]))
    return out


def main(topup_paths):
    qids, kA, kB, sA, sB = rc.load_reference(ORIG)
    idx = {int(q): i for i, q in enumerate(qids)}
    kc = np.zeros(len(qids), dtype=np.int64)
    nc = np.zeros(len(qids), dtype=np.int64)
    top = load_scores_by_qid(topup_paths)
    for q, scores in top.items():
        kc[idx[q]] = sum(scores)
        nc[idx[q]] = len(scores)

    uncertain = set(json.load(open("analysis/b1_results.json"))
                    ["oracle_uncertainty"]["uncertain_qids"])
    mask_top = np.array([q in uncertain for q in qids])
    assert set(int(q) for q in np.array(qids)[mask_top]) == set(top.keys()), \
        "top-up coverage mismatch"

    k_old = (kA + kB)[mask_top]
    n_old = 256
    a0_old, b0_old = rc.fit_beta_moment((kA + kB) / 256.0)
    post_old = rc.band_mass(k_old, np.full(mask_top.sum(), n_old), a0_old, b0_old)
    ref_old = rc.ref_action((kA + kB) / 256.0)
    act_old = ref_old[mask_top]

    k_tot = kA + kB + kc
    n_tot = np.where(mask_top, 256 + nc, 256)
    p_new = k_tot / n_tot
    ref_new = rc.ref_action(p_new)
    changed_mask = mask_top & (ref_new != ref_old)
    n_changed = int(changed_mask.sum())

    a0_new, b0_new = rc.fit_beta_moment(p_new)
    post_new = rc.band_mass(k_tot[mask_top], n_tot[mask_top], a0_new, b0_new)
    mass_on_ref = np.where(ref_new[mask_top] == 1, post_new, 1 - post_new)
    still_amb = int((mass_on_ref < rc.ORACLE_UNCERTAIN_MAXP).sum())
    unamb_new = np.ones(len(qids), dtype=bool)
    unamb_new[mask_top] = mass_on_ref >= rc.ORACLE_UNCERTAIN_MAXP

    changed_rows = [
        {"qid": int(qids[i]), "p_old": float((kA[i] + kB[i]) / 256),
         "p_new": float(p_new[i]),
         "action_old": int(ref_old[i]), "action_new": int(ref_new[i]),
         "kc": int(kc[i]), "nc": int(nc[i])}
        for i in np.where(changed_mask)[0]]

    cache_v2 = {
        "meta": {
            "base": ORIG,
            "topup_files": list(topup_paths),
            "n_uncertain_topped": int(mask_top.sum()),
            "prior_refined": {"alpha0": a0_new, "beta0": b0_new},
        },
        "prompts": {str(q): {"kA": int(kA[i]), "kB": int(kB[i]),
                             "kC": int(kc[i]), "nC": int(nc[i]),
                             "n_tot": int(n_tot[i])}
                    for i, q in enumerate(qids)},
    }
    json.dump(cache_v2, open(OUT_CACHE, "w"))

    stats = {
        "topped": int(mask_top.sum()),
        "changed_action": n_changed,
        "still_ambiguous_at_refined_n": still_amb,
        "unambiguous_total": int(unamb_new.sum()),
        "prior_refined": {"alpha0": a0_new, "beta0": b0_new},
        "changed_examples": changed_rows[:20],
        "changed_qids": [int(qids[i]) for i in np.where(changed_mask)[0]],
    }
    json.dump(stats, open(OUT_STATS, "w"), indent=1)
    print(f"topped={mask_top.sum()} changed_action={n_changed} "
          f"still_ambiguous={still_amb} unambiguous_total={unamb_new.sum()}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
