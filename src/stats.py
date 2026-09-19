"""Nonparametric statistics: Cliff's delta, BCa bootstrap, paired bootstrap, BH-FDR.

Why nonparametric throughout: dependency-length distributions are heavily
right-skewed and bounded below at zero. A t-test on them is not merely
imprecise, it tests the wrong thing. Cliff's delta asks the question we
actually care about -- "how often is a real sentence shorter than its random
counterpart?" -- without assuming any distribution shape.

THE PAIRED-BOOTSTRAP TRAP (H2): to compare two effect sizes measured on the
same sentences, you must resample the SAME sentence indices for both metrics on
every replicate. Bootstrapping the two deltas independently throws away the
correlation between them and inflates the CI on their difference, typically
turning a real effect into a null one. `paired_delta_difference_ci` below does
it correctly; there is no independent-resampling path in this module.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: P(x > y) - P(x < y), in [-1, 1].

    delta < 0 means x tends to be SMALLER than y. Since we pass
    (real, baseline), the DLM prediction is delta < 0.

    Computed in O(n log n) via rank statistics rather than the naive O(n*m)
    all-pairs comparison -- with 20k sentences the latter is 4e8 comparisons
    per treebank per metric.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")

    # Mann-Whitney U relates directly to Cliff's delta:
    #   delta = 2U/(nx*ny) - 1, with U counting (x > y) pairs plus half the ties.
    combined = np.concatenate([x, y])
    ranks = sp_stats.rankdata(combined)
    rank_x = ranks[:nx].sum()
    u_x = rank_x - nx * (nx + 1) / 2.0
    return float(2.0 * u_x / (nx * ny) - 1.0)


def paired_cliffs_delta(real: np.ndarray, base: np.ndarray) -> float:
    """Cliff's delta for matched pairs (same sentence, two conditions).

    With genuine pairing, the natural statistic is the sign balance of the
    within-pair differences: P(real > base) - P(real < base). This is the
    matched-pairs analogue and is what we use for H1/H2, because every real
    sentence has its own baseline built from its own tree.
    """
    real = np.asarray(real, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    if len(real) != len(base):
        raise ValueError("paired_cliffs_delta requires equal-length arrays")
    if len(real) == 0:
        return float("nan")
    diff = real - base
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / len(diff))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bca_ci(
    data: np.ndarray,
    statistic,
    n_resamples: int = 5000,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bias-corrected and accelerated bootstrap CI.

    `data` is a 2-D array of shape (n_sentences, k); rows are the independent
    resampling unit. `statistic(rows) -> float`.

    Returns (point_estimate, lo, hi).

    BCa rather than percentile because Cliff's delta is a bounded, typically
    skewed statistic near the edges of its range; a plain percentile interval
    is noticeably off there.
    """
    rng = rng or np.random.default_rng(0)
    data = np.atleast_2d(np.asarray(data, dtype=np.float64))
    if data.shape[0] == 1 and data.shape[1] > 1:
        data = data.T
    n = data.shape[0]
    theta_hat = float(statistic(data))

    if n < 3:
        return theta_hat, float("nan"), float("nan")

    # --- bootstrap replicates ---
    boot = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot[b] = statistic(data[idx])
    boot = boot[np.isfinite(boot)]
    if len(boot) < 10:
        return theta_hat, float("nan"), float("nan")

    # --- bias correction z0 ---
    prop_less = np.mean(boot < theta_hat)
    prop_less = min(max(prop_less, 1.0 / (2 * len(boot))), 1 - 1.0 / (2 * len(boot)))
    z0 = sp_stats.norm.ppf(prop_less)

    # --- acceleration via jackknife ---
    jack = np.empty(n, dtype=np.float64)
    all_idx = np.arange(n)
    for i in range(n):
        jack[i] = statistic(data[all_idx != i])
    jack = jack[np.isfinite(jack)]
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = 0.0 if den == 0 else float(num / den)

    # --- adjusted percentiles ---
    alpha = 1.0 - ci_level
    z_lo, z_hi = sp_stats.norm.ppf(alpha / 2), sp_stats.norm.ppf(1 - alpha / 2)

    def adjust(z):
        denom = 1 - a * (z0 + z)
        if denom == 0:
            return 0.5
        return float(sp_stats.norm.cdf(z0 + (z0 + z) / denom))

    p_lo, p_hi = adjust(z_lo), adjust(z_hi)
    lo = float(np.quantile(boot, np.clip(p_lo, 0.0, 1.0)))
    hi = float(np.quantile(boot, np.clip(p_hi, 0.0, 1.0)))
    return theta_hat, lo, hi


def jackknife_free_bca_ci(
    data: np.ndarray,
    statistic,
    n_resamples: int = 5000,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
    max_jackknife: int = 400,
) -> tuple[float, float, float]:
    """BCa with the jackknife subsampled for large n.

    The full jackknife is O(n) statistic evaluations, which at n=20000 dominates
    runtime. Acceleration `a` is a third-moment quantity that stabilizes quickly,
    so estimating it from a random subsample of leave-one-out replicates is
    accurate enough and turns hours into seconds.
    """
    rng = rng or np.random.default_rng(0)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    n = data.shape[0]
    theta_hat = float(statistic(data))
    if n < 3:
        return theta_hat, float("nan"), float("nan")

    boot = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        boot[b] = statistic(data[rng.integers(0, n, size=n)])
    boot = boot[np.isfinite(boot)]
    if len(boot) < 10:
        return theta_hat, float("nan"), float("nan")

    prop_less = np.mean(boot < theta_hat)
    prop_less = min(max(prop_less, 1.0 / (2 * len(boot))), 1 - 1.0 / (2 * len(boot)))
    z0 = sp_stats.norm.ppf(prop_less)

    jack_idx = np.arange(n) if n <= max_jackknife else rng.choice(n, max_jackknife, replace=False)
    jack = np.empty(len(jack_idx), dtype=np.float64)
    mask = np.ones(n, dtype=bool)
    for k, i in enumerate(jack_idx):
        mask[i] = False
        jack[k] = statistic(data[mask])
        mask[i] = True
    jack = jack[np.isfinite(jack)]
    if len(jack) < 3:
        a = 0.0
    else:
        jm = jack.mean()
        num = np.sum((jm - jack) ** 3)
        den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
        a = 0.0 if den == 0 else float(num / den)

    alpha = 1.0 - ci_level
    z_lo, z_hi = sp_stats.norm.ppf(alpha / 2), sp_stats.norm.ppf(1 - alpha / 2)

    def adjust(z):
        denom = 1 - a * (z0 + z)
        return 0.5 if denom == 0 else float(sp_stats.norm.cdf(z0 + (z0 + z) / denom))

    lo = float(np.quantile(boot, np.clip(adjust(z_lo), 0.0, 1.0)))
    hi = float(np.quantile(boot, np.clip(adjust(z_hi), 0.0, 1.0)))
    return theta_hat, lo, hi


def paired_delta_difference_ci(
    real_word: np.ndarray,
    base_word: np.ndarray,
    real_head: np.ndarray,
    base_head: np.ndarray,
    n_resamples: int = 5000,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict:
    """H2: CI on  D = |delta_head| - |delta_word|, correctly paired.

    All four arrays are indexed by sentence. On each bootstrap replicate we draw
    ONE set of sentence indices and apply it to all four arrays, so the
    correlation between the two metrics is preserved. This is the difference
    between an honest CI and a badly inflated one.
    """
    rng = rng or np.random.default_rng(0)
    real_word = np.asarray(real_word, dtype=np.float64)
    base_word = np.asarray(base_word, dtype=np.float64)
    real_head = np.asarray(real_head, dtype=np.float64)
    base_head = np.asarray(base_head, dtype=np.float64)

    n = len(real_word)
    if not (len(base_word) == len(real_head) == len(base_head) == n):
        raise ValueError("paired_delta_difference_ci requires four equal-length arrays")

    def stat(idx: np.ndarray) -> tuple[float, float, float]:
        dw = paired_cliffs_delta(real_word[idx], base_word[idx])
        dh = paired_cliffs_delta(real_head[idx], base_head[idx])
        return dw, dh, abs(dh) - abs(dw)

    base_idx = np.arange(n)
    d_word, d_head, d_diff = stat(base_idx)

    boot = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)          # SAME indices for both metrics
        boot[b] = stat(idx)[2]
    boot = boot[np.isfinite(boot)]

    alpha = 1.0 - ci_level
    lo = float(np.quantile(boot, alpha / 2)) if len(boot) else float("nan")
    hi = float(np.quantile(boot, 1 - alpha / 2)) if len(boot) else float("nan")

    # Two-sided bootstrap p-value for H0: D = 0
    if len(boot):
        p = 2.0 * min(np.mean(boot <= 0), np.mean(boot >= 0))
        p = float(min(p, 1.0))
    else:
        p = float("nan")

    return {
        "delta_word": d_word,
        "delta_head": d_head,
        "delta_diff": d_diff,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_value": p,
        "n_sentences": n,
    }


# ---------------------------------------------------------------------------
# H2 co-primary statistics (pre-registered alongside Cliff's delta)
# ---------------------------------------------------------------------------
#
# Why two more statistics. Cliff's delta counts only the SIGN of each within-pair
# difference, so every tie is scored as non-support. head_DL is much coarser than
# word_DL (its real-vs-baseline tie rate runs 6-12x higher on UD data), which
# penalizes it structurally regardless of how well it tracks word order.
#
# The two additions were chosen to fail in DIFFERENT ways, so that agreement
# between them would be informative:
#
#   * P(advantage) with ties split. Intended as a tie-immune rank measure.
#     THE FULL RUN SHOWED THIS INTENTION WAS WRONG: P(adv) = (1 - delta)/2
#     identically, so it is Cliff's delta rescaled and carries no independent
#     information. See `prob_advantage` for the algebra. Retained because it was
#     pre-registered, but not counted as separate evidence.
#
#   * Median per-sentence relative reduction. A ratio of SUMMED dependency
#     length -- the obvious formulation -- is an effect on the corpus total and
#     is dominated by long sentences. Taking the median of the per-sentence
#     ratio gives the typical sentence one vote each.
#
# What survives is a RANK-vs-MAGNITUDE contrast, not a tie-structure one. On the
# full run these diverge completely under B1: the magnitude measure favours
# head_DL in 17/17 treebanks while the rank measures favour word_DL in 17/17.
# head_DL wins a smaller share of pairs but wins them by proportionally more.
# That divergence, not either verdict alone, is the H2 result.


def prob_advantage(real: np.ndarray, base: np.ndarray) -> float:
    """P(real < base) + 0.5 * P(real == base): "how often is real shorter?"

    Lower DL is better, so values ABOVE 0.5 mean the real sentence usually wins.

    IMPORTANT -- this is NOT independent of `paired_cliffs_delta`. It is that
    statistic rescaled:

        delta  = (pos - neg) / n           with pos + neg + ties = n
        P(adv) = (neg + ties/2) / n
               = (1 - delta) / 2           EXACTLY, ties included

    The full run confirmed this empirically: corr(dP(adv), d_delta) = 1.0 at a
    ratio of exactly 0.5 across all 34 cells. Splitting ties evenly does NOT
    rescue the coarser metric, because delta already splits them -- a tie
    contributes to neither `pos` nor `neg`, which is arithmetically the same as
    contributing half to each.

    It is kept because it was pre-registered and because it is the more legible
    way to report a rank comparison, but it must not be counted as independent
    evidence alongside delta. The contrast that carries real information is
    `median_relative_reduction`, which is a magnitude rather than a rank.
    """
    real = np.asarray(real, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    if len(real) != len(base):
        raise ValueError("prob_advantage requires equal-length arrays")
    if len(real) == 0:
        return float("nan")
    return float((np.sum(real < base) + 0.5 * np.sum(real == base)) / len(real))


def median_relative_reduction(real: np.ndarray, base: np.ndarray) -> float:
    """Median of (base - real) / base over sentences with base > 0.

    Unweighted by length, unlike a ratio of sums. Pairs whose baseline DL is
    zero carry no proportional information and are dropped rather than being
    counted as zero improvement.
    """
    real = np.asarray(real, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    if len(real) != len(base):
        raise ValueError("median_relative_reduction requires equal-length arrays")
    ok = base > 0
    if not ok.any():
        return float("nan")
    return float(np.median((base[ok] - real[ok]) / base[ok]))


def paired_h2_costatistics_ci(
    real_word: np.ndarray,
    base_word: np.ndarray,
    real_head: np.ndarray,
    base_head: np.ndarray,
    n_resamples: int = 5000,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict:
    """CIs on the head-minus-word contrast for both co-primary statistics.

    Same pairing discipline as `paired_delta_difference_ci`: ONE set of sentence
    indices per replicate, applied to all four arrays, so the correlation
    between the metrics is preserved. Resampling them independently would
    inflate both CIs.

    Returns the two contrasts (head minus word; positive favours head_DL), their
    CIs and bootstrap p-values, plus the underlying per-metric values.
    """
    rng = rng or np.random.default_rng(0)
    rw = np.asarray(real_word, dtype=np.float64)
    bw = np.asarray(base_word, dtype=np.float64)
    rh = np.asarray(real_head, dtype=np.float64)
    bh = np.asarray(base_head, dtype=np.float64)

    n = len(rw)
    if not (len(bw) == len(rh) == len(bh) == n):
        raise ValueError("paired_h2_costatistics_ci requires four equal-length arrays")

    def contrasts(idx: np.ndarray) -> tuple[float, float]:
        pa = prob_advantage(rh[idx], bh[idx]) - prob_advantage(rw[idx], bw[idx])
        mr = (median_relative_reduction(rh[idx], bh[idx])
              - median_relative_reduction(rw[idx], bw[idx]))
        return pa, mr

    base_idx = np.arange(n)
    pa_obs, mr_obs = contrasts(base_idx)

    boot_pa = np.empty(n_resamples, dtype=np.float64)
    boot_mr = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)          # SAME indices for both metrics
        boot_pa[b], boot_mr[b] = contrasts(idx)

    alpha = 1.0 - ci_level

    def summarize(obs: float, boot: np.ndarray) -> tuple[float, float, float]:
        boot = boot[np.isfinite(boot)]
        if not len(boot):
            return float("nan"), float("nan"), float("nan")
        lo = float(np.quantile(boot, alpha / 2))
        hi = float(np.quantile(boot, 1 - alpha / 2))
        p = 2.0 * min(np.mean(boot <= 0), np.mean(boot >= 0))
        return lo, hi, float(min(p, 1.0))

    pa_lo, pa_hi, pa_p = summarize(pa_obs, boot_pa)
    mr_lo, mr_hi, mr_p = summarize(mr_obs, boot_mr)

    return {
        "prob_adv_word": prob_advantage(rw, bw),
        "prob_adv_head": prob_advantage(rh, bh),
        "prob_adv_diff": pa_obs,
        "prob_adv_ci_lo": pa_lo,
        "prob_adv_ci_hi": pa_hi,
        "prob_adv_p": pa_p,
        "med_rel_word": median_relative_reduction(rw, bw),
        "med_rel_head": median_relative_reduction(rh, bh),
        "med_rel_diff": mr_obs,
        "med_rel_ci_lo": mr_lo,
        "med_rel_ci_hi": mr_hi,
        "med_rel_p": mr_p,
        "tie_rate_word": float(np.mean(rw == bw)),
        "tie_rate_head": float(np.mean(rh == bh)),
    }


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvalues: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """BH-FDR. Returns (rejected, adjusted_pvalues).

    Applied within each hypothesis family (H1, H2, H3) across treebanks, not
    across everything at once -- the three hypotheses are separate questions.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    finite = np.isfinite(p)
    m = int(finite.sum())
    adj = np.full(len(p), np.nan)
    rej = np.zeros(len(p), dtype=bool)
    if m == 0:
        return rej, adj

    idx = np.flatnonzero(finite)
    order = idx[np.argsort(p[idx])]
    ranked = p[order]

    adj_sorted = ranked * m / np.arange(1, m + 1)
    # enforce monotonicity from the largest p downward
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)

    adj[order] = adj_sorted
    rej[order] = adj_sorted <= q
    return rej, adj


def sign_test_p(diff: np.ndarray) -> float:
    """Exact two-sided sign test on paired differences (ties dropped).

    Used as the per-treebank p-value for H1: does the real sentence beat its
    baseline more often than chance?
    """
    diff = np.asarray(diff, dtype=np.float64)
    pos = int(np.sum(diff > 0))
    neg = int(np.sum(diff < 0))
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    p = 2.0 * sp_stats.binom.cdf(k, n, 0.5)
    return float(min(p, 1.0))
