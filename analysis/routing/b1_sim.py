"""B1 offline certification simulation. Gates and constants per REGISTERED.md."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import route_core as rc

OUT = "analysis/b1_results.json"
REF_CACHE = "analysis/reference_cache.json"


def summarize(name, cfg, acts1, acts2, consumed, ref, unamb,
              p_true, vbar, extra=None):
    def m(x):
        return float(np.mean(x))

    err_all = 1 - m(acts1 == ref[None, :])
    err_unamb = 1 - m(acts1[:, unamb] == ref[unamb][None, :])
    false = m((acts1 == 1) & (ref == 0)[None, :])
    missed = m((acts1 == 0) & (ref == 1)[None, :])
    false_b = m((acts2 == 1) & (ref == 0)[None, :])
    missed_b = m((acts2 == 0) & (ref == 1)[None, :])
    jac = rc.jaccard(acts1, acts2)
    reg = rc.regret_mean(acts1, ref, p_true, vbar)
    row = {
        "arm": name,
        "config": cfg,
        "mean_N": float(np.mean(consumed)),
        "acc_all": 1 - err_all,
        "acc_unamb": 1 - err_unamb,
        "false_route": (false + false_b) / 2,
        "missed_route": (missed + missed_b) / 2,
        "jaccard_rerun": float(jac.mean()),
        "agreement_rerun": float(rc.agreement(acts1, acts2).mean()),
        "regret_mean": float(reg.mean()),
        "err_all": float(err_all),
        "err_unamb": float(err_unamb),
    }
    if extra:
        row.update(extra)
    return row


def main():
    t0 = time.time()
    qids, kA, kB, sA, sB = rc.load_reference(REF_CACHE)
    k_pool = kA + kB
    n_pool = kA.shape[0] * 0 + 256
    p_true = k_pool / 256.0
    ref = rc.ref_action(p_true)
    vbar = float(np.mean(p_true * (1 - p_true)))

    rng_fit = np.random.default_rng(0)
    del rng_fit
    a0_eb, b0_eb = rc.fit_beta_moment(p_true)

    post_ref = rc.band_mass(k_pool, np.full(len(p_true), 256), a0_eb, b0_eb)
    mass_on_ref = np.where(ref == 1, post_ref, 1 - post_ref)
    unamb = mass_on_ref >= rc.ORACLE_UNCERTAIN_MAXP
    n_uncertain = int((~unamb).sum())

    rows = []
    uncert_payload = {
        "n_uncertain": n_uncertain,
        "threshold_mass": rc.ORACLE_UNCERTAIN_MAXP,
        "prior": {"alpha0": a0_eb, "beta0": b0_eb, "kind": "moment_matched_EB"},
        "uncertain_qids": [int(q) for q, u in zip(qids, unamb) if not u],
        "mass_on_ref_stats": {
            "min": float(mass_on_ref.min()),
            "p05": float(np.quantile(mass_on_ref, 0.05)),
            "median": float(np.median(mass_on_ref)),
        },
    }

    rng = np.random.default_rng(rc.SEED)

    for n_fixed in (8, 16, 32):
        k1, k2, cons = rc.run_fixed(p_true, rng, n_fixed)
        a1 = rc.act_band(k1, n_fixed)
        a2 = rc.act_band(k2, n_fixed)
        rows.append(summarize(f"plugin", f"N={n_fixed}", a1, a2, cons,
                              ref, unamb, p_true, vbar))

    for n_fixed in (8, 16, 32):
        nn = np.full(len(p_true), n_fixed, dtype=float)
        k1, k2, cons = rc.run_fixed(p_true, rng, n_fixed)

        def pm_act(k):
            a = a0_eb + k
            b = b0_eb + n_fixed - k
            ep = a / (a + b)
            return ((ep >= rc.BAND[0]) & (ep <= rc.BAND[1])).astype(np.int8)

        a1, a2 = pm_act(k1), pm_act(k2)
        rows.append(summarize(f"postmean", f"N={n_fixed}", a1, a2, cons,
                              ref, unamb, p_true, vbar))

    for n_fixed in (8, 16, 32):
        a1, a2, cons = rc.run_peu_fixed(p_true, rng, n_fixed, vbar, (a0_eb, b0_eb))
        rows.append(summarize(f"peu", f"N={n_fixed}", a1, a2, cons,
                              ref, unamb, p_true, vbar))

    cert_rows = {}
    for delta in rc.DELTAS:
        a1, a2, cons, unc_frac = rc.run_certified(
            p_true, rng, delta, abstain=False, prior=(a0_eb, b0_eb))
        r = summarize("certified_seq", f"delta={delta}", a1, a2, cons,
                      ref, unamb, p_true, vbar,
                      extra={"abstained_frac": 0.0})
        rows.append(r)
        cert_rows[f"certified_seq|delta={delta}"] = r

        a1b, a2b, consb, unc_frac_b = rc.run_certified(
            p_true, rng, delta, abstain=True, prior=(a0_eb, b0_eb))
        rb = summarize("certified_abstain", f"delta={delta}", a1b, a2b, consb,
                       ref, unamb, p_true, vbar,
                       extra={"abstained_frac": float(unc_frac_b.mean())})
        rows.append(rb)

    for delta in rc.DELTAS:
        key = f"certified_seq|delta={delta}"
        budget = round(cert_rows[key]["mean_N"])
        a1, a2, cons = rc.run_kgps_proxy(p_true, rng, budget, vbar, (a0_eb, b0_eb))
        rows.append(summarize("peu_diagnostic", f"budget={budget} (matches {key})",
                              a1, a2, cons, ref, unamb, p_true, vbar,
                              extra={"label_note": "posterior-expected-utility "
                                                   "diagnostic; NOT a comparison "
                                                   "against actual KGPS"}))

    np_rows = []
    rng_k = np.random.default_rng(rc.SEED + 5)
    k16, _, _ = rc.run_fixed(p_true, rng_k, 16, reps=rc.REPS)
    a16 = rc.act_band(k16, 16)
    dhat16 = np.abs(k16 / 16 - 0.5)
    tau_plugin = float(np.mean([rc.kendall_replay(dhat16[r], p_true, a16[r])
                                for r in range(rc.REPS)]))
    ranking_block = {
        "kendall_priority_plugin_N16": tau_plugin,
        "kendall_priority_certified": None,
        "note": ("ordering of estimated |p-0.5| vs reference among REPLAY-routed "
                 "sets; certified-arm ranking deferred to B2 where the buffer "
                 "priority rule is simulated end-to-end"),
    }
    for n_fixed in (8, 16, 32):
        rng_np = np.random.default_rng(rc.SEED + 777)
        k1, k2, ok = rc.nonparametric_streams(sA, sB, rng_np, n_fixed, reps=100)
        a1 = rc.act_band(k1, n_fixed)
        a2 = rc.act_band(k2, n_fixed)
        jac = rc.jaccard(a1[:, ok], a2[:, ok])
        acc_np = (a1[:, ok] == ref[ok][None, :]).mean(axis=1)
        np_rows.append({
            "arm": "nonparametric_plugin_real_draws",
            "config": f"N={n_fixed}",
            "n_prompts_used": int(ok.sum()),
            "err_all_nonparam": float(1 - acc_np.mean()),
            "jaccard_rerun_nonparam": float(jac.mean()),
        })

    result = {
        "meta": {
            "seed": rc.SEED,
            "reps": rc.REPS,
            "n_prompts": int(len(p_true)),
            "n_reference_samples_per_prompt": 256,
            "band": list(rc.BAND),
            "sequential": {"n0": rc.N0, "delta_n": rc.DELTA_N, "nmax": rc.NMAX},
            "vbar": vbar,
            "runtime_s": time.time() - t0,
        },
        "oracle_uncertainty": uncert_payload,
        "rows": rows,
        "ranking_secondary": ranking_block,
        "nonparametric_sensitivity": np_rows,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)

    hdr = f"{'arm':<18} {'cfg':<26} {'meanN':>6} {'errAll':>7} {'errUnamb':>8} {'J':>6} {'agree':>6} {'false':>6} {'missed':>7} {'regret':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["mean_N"]):
        print(f"{r['arm']:<18} {r['config']:<26} {r['mean_N']:>6.1f} "
              f"{r['err_all']:>7.3f} {r['err_unamb']:>8.3f} {r['jaccard_rerun']:>6.3f} "
              f"{r['agreement_rerun']:>6.3f} {r['false_route']:>6.3f} "
              f"{r['missed_route']:>7.3f} {r['regret_mean']:>7.4f}")
    print()
    print("oracle-uncertain prompts:", n_uncertain, "/", len(p_true))
    for r in np_rows:
        print(r)


if __name__ == "__main__":
    main()
