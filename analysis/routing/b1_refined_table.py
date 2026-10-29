"""Regenerate B1 headline metrics against the REFINED oracle (A82).

Same arm schedules and seed as the original registered run; only the truth
labels change (n=512 for the topped 133, n=256 elsewhere). No delta tuning.
"""
import json

import numpy as np

import route_core as rc

REF = "analysis/reference_cache_v2.json"
OUT = "analysis/b1_refined_table.json"


def main():
    d = json.load(open(REF))
    qids = sorted(int(q) for q in d["prompts"])
    k_tot = np.array([d["prompts"][str(q)]["kA"] + d["prompts"][str(q)]["kB"]
                      + d["prompts"][str(q)]["kC"] for q in qids], dtype=float)
    n_tot = np.array([d["prompts"][str(q)]["n_tot"] for q in qids], dtype=float)
    p_ref = k_tot / n_tot
    ref = rc.ref_action(p_ref)
    vbar = float((p_ref * (1 - p_ref)).mean())
    prior = rc.fit_beta_moment(p_ref)
    post = rc.band_mass(k_tot, n_tot, *prior)
    mass_on_ref = np.where(ref == 1, post, 1 - post)
    unamb = mass_on_ref >= rc.ORACLE_UNCERTAIN_MAXP

    rng = np.random.default_rng(rc.SEED)
    rows = []

    for n_fixed in (8, 16, 32):
        k1, k2, _ = rc.run_fixed(p_ref, rng, n_fixed)
        a1, a2 = rc.act_band(k1, n_fixed), rc.act_band(k2, n_fixed)
        rows.append(_row("plugin", f"N={n_fixed}", a1, a2,
                         np.full(rc.REPS, float(n_fixed)), ref, unamb,
                         p_ref, vbar))

    for delta in rc.DELTAS:
        a1, a2, cons, unc = rc.run_certified(
            p_ref, rng, delta, abstain=False, prior=prior)
        r = _row("certified_seq", f"delta={delta}", a1, a2, cons,
                 ref, unamb, p_ref, vbar)
        r["abstained_frac"] = 0.0
        rows.append(r)

    json.dump({"meta": {"seed": rc.SEED, "reps": rc.REPS,
                        "prior": {"alpha0": prior[0], "beta0": prior[1]},
                        "vbar": vbar,
                        "n_unambiguous": int(unamb.sum())},
               "rows": rows}, open(OUT, "w"), indent=1)

    hdr = (f"{'arm':<16} {'cfg':<12} {'meanN':>6} {'errAll':>7} "
           f"{'errUnamb':>9} {'J':>6} {'agree':>6} {'false':>6} "
           f"{'missed':>7} {'regret':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["mean_N"]):
        print(f"{r['arm']:<16} {r['config']:<12} {r['mean_N']:>6.1f} "
              f"{r['err_all']:>7.3f} {r['err_unamb']:>9.3f} "
              f"{r['jaccard_rerun']:>6.3f} {r['agreement_rerun']:>6.3f} "
              f"{r['false_route']:>6.3f} {r['missed_route']:>7.3f} "
              f"{r['regret_mean']:>8.4f}")


def _row(name, cfg, a1, a2, cons, ref, unamb, p_true, vbar):
    err_all = 1 - float((a1 == ref[None, :]).mean())
    err_unamb = 1 - float((a1[:, unamb] == ref[unamb][None, :]).mean())
    false = float(((a1 == 1) & (ref == 0)[None, :]).mean())
    missed = float(((a1 == 0) & (ref == 1)[None, :]).mean())
    v_replay = p_true * (1 - p_true)
    v_act = np.where(a1 == 1, v_replay[None, :], vbar)
    v_ref = np.where(ref[None, :] == 1, v_replay[None, :], vbar)
    return {
        "arm": name, "config": cfg,
        "mean_N": float(np.mean(cons)),
        "err_all": err_all, "err_unamb": err_unamb,
        "false_route": false, "missed_route": missed,
        "jaccard_rerun": float(rc.jaccard(a1, a2).mean()),
        "agreement_rerun": float((a1 == a2).mean()),
        "regret_mean": float(np.abs(v_act - v_ref).mean()),
    }


if __name__ == "__main__":
    main()
