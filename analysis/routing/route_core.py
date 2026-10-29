"""Certified-routing core for the F_route pilot. Constants frozen in REGISTERED.md."""
import numpy as np
from scipy import stats

BAND = (0.25, 0.75)
N0 = 8
DELTA_N = 4
NMAX = 64
DELTAS = (0.90, 0.95, 0.99)
SEED = 20260822
REPS = 200
ORACLE_UNCERTAIN_MAXP = 0.99


def load_reference(path):
    import json
    d = json.load(open(path))
    qids = sorted(int(q) for q in d["prompts"])
    kA = np.array([d["prompts"][str(q)]["kA"] for q in qids], dtype=np.int64)
    kB = np.array([d["prompts"][str(q)]["kB"] for q in qids], dtype=np.int64)
    sA = [np.array(d["prompts"][str(q)]["scoresA"], dtype=np.int8) for q in qids]
    sB = [np.array(d["prompts"][str(q)]["scoresB"], dtype=np.int8) for q in qids]
    return np.array(qids), kA, kB, sA, sB


def ref_action(p_true):
    return ((p_true >= BAND[0]) & (p_true <= BAND[1])).astype(np.int8)


def fit_beta_moment(p):
    m = p.mean()
    v = p.var()
    t = m * (1.0 - m) / v - 1.0
    return float(m * t), float((1.0 - m) * t)


def band_mass(k, n, a0, b0):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    rv = stats.beta(a0 + k, b0 + n - k)
    return rv.cdf(BAND[1]) - rv.cdf(BAND[0])


def confidence(k, n, a0, b0):
    inb = band_mass(k, n, a0, b0)
    return np.maximum(inb, 1.0 - inb), inb


def act_from_mass(inb):
    return (inb >= 0.5).astype(np.int8)


def act_band(k, n):
    kn = np.asarray(k, dtype=float) / np.asarray(n)
    return ((kn >= BAND[0]) & (kn <= BAND[1])).astype(np.int8)


def beta_raw_moments(a, b, upto=4):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = []
    num = np.ones_like(a)
    den = np.ones_like(a)
    for j in range(upto):
        num = num * (a + j)
        den = den * (a + b + j)
        out.append(num / den)
    return out


def posterior_mean_v(k, n, a0, b0):
    a = a0 + np.asarray(k, dtype=float)
    b = b0 + np.asarray(n, dtype=float) - np.asarray(k, dtype=float)
    e1, e2, _, _ = beta_raw_moments(a, b)
    return e1 - e2


def posterior_sd_v(k, n, a0, b0):
    a = a0 + np.asarray(k, dtype=float)
    b = b0 + np.asarray(n, dtype=float) - np.asarray(k, dtype=float)
    e1, e2, e3, e4 = beta_raw_moments(a, b)
    ev = e1 - e2
    ev2 = e2 - 2.0 * e3 + e4
    return np.sqrt(np.maximum(ev2 - ev ** 2, 0.0))


def run_fixed(p_true, rng, n, act_fn=None, reps=REPS):
    p = np.asarray(p_true, dtype=float)
    k1 = rng.binomial(n, p, size=(reps, len(p)))
    k2 = rng.binomial(n, p, size=(reps, len(p)))
    return k1, k2, np.full(reps, float(n))


def run_certified(p_true, rng, delta, abstain, prior, reps=REPS,
                  n0=N0, dn=DELTA_N, nmax=NMAX):
    a0, b0 = prior
    p = np.asarray(p_true, dtype=float)
    P = len(p)
    acts1 = np.zeros((reps, P), dtype=np.int8)
    acts2 = np.zeros((reps, P), dtype=np.int8)
    cons = np.zeros(2)
    unc_frac = np.zeros(reps)
    for r in range(reps):
        for s in range(2):
            k = rng.binomial(n0, p)
            n = np.full(P, n0, dtype=np.int64)
            active = np.ones(P, dtype=bool)
            uncert_mask = np.zeros(P, dtype=bool)
            while True:
                idx = np.where(active)[0]
                conf, _ = confidence(k[idx], n[idx], a0, b0)
                certified = conf >= delta
                at_cap = n[idx] >= nmax
                stop_idx = idx[certified | at_cap]
                uncert_mask[idx[at_cap & ~certified]] = True
                active[stop_idx] = False
                go = idx[(~certified) & (~at_cap)]
                if go.size == 0:
                    break
                k[go] += rng.binomial(dn, p[go])
                n[go] += dn
            _, inb_final = confidence(k, n, a0, b0)
            a = act_from_mass(inb_final)
            if abstain:
                a[uncert_mask] = 0
            (acts1 if s == 0 else acts2)[r] = a
            cons[s] += n.mean() / reps
            if s == 1:
                unc_frac[r] = uncert_mask.mean()
    return acts1, acts2, cons.mean(), unc_frac


def run_peu_fixed(p_true, rng, n, vbar, prior, reps=REPS):
    a0, b0 = prior
    p = np.asarray(p_true, dtype=float)
    nn = np.full(len(p), n, dtype=float)
    k1 = rng.binomial(n, p, size=(reps, len(p)))
    k2 = rng.binomial(n, p, size=(reps, len(p)))
    a1 = (posterior_mean_v(k1, nn, a0, b0) > vbar).astype(np.int8)
    a2 = (posterior_mean_v(k2, nn, a0, b0) > vbar).astype(np.int8)
    return a1, a2, np.full(reps, float(n))


def run_kgps_proxy(p_true, rng, budget, vbar, prior, reps=REPS, n0=N0, dn=DELTA_N):
    a0, b0 = prior
    p = np.asarray(p_true, dtype=float)
    P = len(p)
    grants_total = max(0, int(round(budget * P)) - n0 * P)
    n_iters = grants_total // dn
    acts1 = np.zeros((reps, P), dtype=np.int8)
    acts2 = np.zeros((reps, P), dtype=np.int8)
    for r in range(reps):
        for s in range(2):
            k = rng.binomial(n0, p).astype(np.float64)
            n = np.full(P, float(n0))
            for _ in range(n_iters):
                sdv = posterior_sd_v(k, n, a0, b0)
                i = int(np.argmax(sdv))
                k[i] += rng.binomial(dn, p[i])
                n[i] += dn
            mv = posterior_mean_v(k, n, a0, b0)
            (acts1 if s == 0 else acts2)[r] = (mv > vbar).astype(np.int8)
    return acts1, acts2, np.full(reps, float(budget))


def jaccard(a1, a2):
    inter = (a1 & a2).sum(axis=1).astype(float)
    union = (a1 | a2).sum(axis=1).astype(float)
    both_empty = ((a1.sum(1) == 0) & (a2.sum(1) == 0)).astype(float)
    j = np.where(union > 0, inter / np.maximum(union, 1e-12), both_empty)
    return j


def agreement(a1, a2):
    return (a1 == a2).mean(axis=1)


def rates_vs_ref(acts, ref):
    false = ((acts == 1) & (ref == 0)).mean(axis=1)
    missed = ((acts == 0) & (ref == 1)).mean(axis=1)
    acc = (acts == ref).mean(axis=1)
    return acc, false, missed


def regret_mean(acts, ref, p_true, vbar):
    v_replay = p_true * (1.0 - p_true)
    v_of_act = np.where(acts == 1, v_replay[None, :], vbar)
    v_of_ref = np.where(ref[None, :] == 1, v_replay[None, :], vbar)
    return np.abs(v_of_act - v_of_ref).mean(axis=1)


def kendall_replay(dist_est, p_true, acts_row):
    sel = acts_row == 1
    if sel.sum() < 3:
        return np.nan
    tau = stats.kendalltau(np.asarray(dist_est)[sel], np.abs(p_true - 0.5)[sel])
    return float(tau.statistic)


def nonparametric_streams(sA, sB, rng, n, reps=REPS):
    P = len(sA)
    lensA = np.array([len(x) for x in sA])
    lensB = np.array([len(x) for x in sB])
    ok = (lensA >= n) & (lensB >= n)
    k1 = np.zeros((reps, P), dtype=np.int64)
    k2 = np.zeros((reps, P), dtype=np.int64)
    idx_ok = np.where(ok)[0]
    for r in range(reps):
        for i in idx_ok:
            k1[r, i] = sA[i][rng.choice(len(sA[i]), size=n, replace=False)].sum()
            k2[r, i] = sB[i][rng.choice(len(sB[i]), size=n, replace=False)].sum()
    return k1, k2, ok
