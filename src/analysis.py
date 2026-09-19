"""H1/H2/H3 drivers and verdict emission.

The three hypotheses, with directions stated before any data is read:

  H1  Real sentences minimize DL vs. random linearization, on BOTH metrics.
      Statistic: paired Cliff's delta (real vs. baseline). Predicted delta < 0.
      This is a SANITY CHECK. If H1 fails the pipeline is broken, not the theory.

  H2  The head-based metric shows a LARGER real-vs-random effect than the
      word-based metric. Statistic: D = |delta_head| - |delta_word|, with a
      correctly paired bootstrap CI. Predicted D > 0.

  H3  Word-based DL is DERIVATIVE of head-based DL. Per-sentence logistic
      regression over {real} u {one random baseline per real sentence}:

          P(real | pair) ~ z(word_DL) + z(head_DL)        [conditional logit]

      Predicted: beta(word_DL) falls to zero once head_DL enters the model,
      while beta(head_DL) stays significant. Pair-constant covariates
      (n_tokens, arity, depth) are absorbed by the strata, not fitted.

Note on H3's design: "realness" is a BETWEEN-CONDITION factor over matched
pairs, not a per-sentence property of a treebank (every UD sentence is real).
Making realness the binary outcome over real-vs-baseline pairs is what makes
this a decidable head-to-head test rather than an illustration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import linearize as lin
from . import metrics as mx
from . import stats as st
from .tree import Sentence, is_projective, mean_arity, tree_depth

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-sentence feature extraction
# ---------------------------------------------------------------------------

def sentence_row(sent: Sentence, scope: str, include_endpoints: bool, is_real: int) -> dict:
    """One row of the analysis table: DL metrics plus structural covariates."""
    m = mx.sentence_metrics(sent, scope=scope, include_endpoints=include_endpoints)
    return {
        "sent_id": sent.sent_id,
        "is_real": is_real,
        "n_tokens": m["n_tokens"],
        "n_arcs": m["n_arcs"],
        "word_dl": m["word_dl_sum"],
        "head_dl": m["head_dl_sum"],
        "word_dl_mean": m["word_dl_mean"],
        "head_dl_mean": m["head_dl_mean"],
        "mean_arity": mean_arity(sent),
        "tree_depth": float(tree_depth(sent)),
    }


def build_treebank_table(
    sentences,
    treebank_id: str,
    cfg: dict,
    n_samples: int,
    projective_only: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Stream sentences -> per-sentence real/baseline metric table.

    Memory note: sentences arrive as a generator and we keep only the extracted
    numeric rows, never the parsed trees. Peak RAM stays well under the 2 GB
    budget even for cs_pdt.
    """
    scope = cfg["head_count"]["scope"]
    alt_scope = "within_span" if scope == "global" else "global"
    include_endpoints = cfg["head_count"]["include_endpoints"]
    master_seed = cfg["run"]["seed"]
    kinds = cfg["baselines"]["kinds"]

    rows: list[dict] = []
    n_nonproj = 0
    n_total = 0

    for sent in sentences:
        n_total += 1
        proj = is_projective(sent)
        if not proj:
            n_nonproj += 1
            if projective_only:
                # Non-projective originals are not comparable to projective
                # baselines; they are counted and reported, never silently mixed.
                continue

        real = sentence_row(sent, scope, include_endpoints, is_real=1)
        real["treebank_id"] = treebank_id
        real["baseline_kind"] = "REAL"
        real["is_projective"] = int(proj)
        # robustness: the same sentence under the alternative scope definition
        real["head_dl_alt"] = float(
            mx.head_dl_arcs(sent, scope=alt_scope, include_endpoints=include_endpoints).sum()
        )
        rows.append(real)

        for kind in kinds:
            samples = lin.sample_baselines(sent, kind, n_samples, master_seed)
            # Average over samples so each real sentence has exactly ONE matched
            # baseline value per metric -- this is what makes the pairing valid.
            acc = [sentence_row(s, scope, include_endpoints, is_real=0) for s in samples]
            mean_row = {
                "sent_id": sent.sent_id,
                "treebank_id": treebank_id,
                "baseline_kind": kind,
                "is_real": 0,
                "is_projective": 1,
                "n_tokens": acc[0]["n_tokens"],
                "n_arcs": acc[0]["n_arcs"],
                "word_dl": float(np.mean([a["word_dl"] for a in acc])),
                "head_dl": float(np.mean([a["head_dl"] for a in acc])),
                "word_dl_mean": float(np.mean([a["word_dl_mean"] for a in acc])),
                "head_dl_mean": float(np.mean([a["head_dl_mean"] for a in acc])),
                "mean_arity": acc[0]["mean_arity"],
                "tree_depth": acc[0]["tree_depth"],
                "head_dl_alt": float(
                    np.mean([
                        mx.head_dl_arcs(s, scope=alt_scope,
                                        include_endpoints=include_endpoints).sum()
                        for s in samples
                    ])
                ),
            }
            rows.append(mean_row)

    df = pd.DataFrame(rows)
    info = {
        "n_sentences_total": n_total,
        "n_nonprojective": n_nonproj,
        "nonprojective_rate": (n_nonproj / n_total) if n_total else 0.0,
        "n_analyzed": int((df["baseline_kind"] == "REAL").sum()) if len(df) else 0,
    }
    return df, info


# ---------------------------------------------------------------------------
# H1 and H2
# ---------------------------------------------------------------------------

def run_h1_h2(df: pd.DataFrame, treebank_id: str, cfg: dict, n_resamples: int) -> tuple[list[dict], list[dict]]:
    """Per-treebank H1 (delta per metric per baseline) and H2 (the comparison)."""
    ci_level = cfg["thresholds"]["ci_level"]
    rng = np.random.default_rng(cfg["run"]["seed"] + 991)

    real = df[df["baseline_kind"] == "REAL"].set_index("sent_id").sort_index()
    h1_rows, h2_rows = [], []

    for kind in cfg["baselines"]["kinds"]:
        base = df[df["baseline_kind"] == kind].set_index("sent_id").sort_index()
        common = real.index.intersection(base.index)
        if len(common) < 10:
            log.warning("%s/%s: only %d paired sentences, skipping", treebank_id, kind, len(common))
            continue

        r = real.loc[common]
        b = base.loc[common]

        for metric in ("word_dl", "head_dl"):
            rv = r[metric].to_numpy(dtype=np.float64)
            bv = b[metric].to_numpy(dtype=np.float64)
            diff = rv - bv

            def stat(rows: np.ndarray) -> float:
                return st.paired_cliffs_delta(rows[:, 0], rows[:, 1])

            data = np.column_stack([rv, bv])
            delta, lo, hi = st.jackknife_free_bca_ci(
                data, stat, n_resamples=n_resamples, ci_level=ci_level, rng=rng
            )
            h1_rows.append({
                "treebank_id": treebank_id,
                "baseline_kind": kind,
                "metric": metric,
                "delta": delta,
                "ci_lo": lo,
                "ci_hi": hi,
                "p_value": st.sign_test_p(diff),
                "n_sentences": int(len(common)),
                "n_arcs": float(r["n_arcs"].sum()),
                "mean_real": float(rv.mean()),
                "mean_baseline": float(bv.mean()),
                # H1 is supported for this cell if delta < 0 and the CI excludes 0
                "supports_h1": bool(delta < 0 and np.isfinite(hi) and hi < 0),
            })

        # --- H2: the correctly paired comparison of the two deltas ---------
        h2 = st.paired_delta_difference_ci(
            real_word=r["word_dl"].to_numpy(dtype=np.float64),
            base_word=b["word_dl"].to_numpy(dtype=np.float64),
            real_head=r["head_dl"].to_numpy(dtype=np.float64),
            base_head=b["head_dl"].to_numpy(dtype=np.float64),
            n_resamples=n_resamples,
            ci_level=ci_level,
            rng=rng,
        )
        # --- H2 co-primaries: tie-immune and length-unweighted ------------
        # Pre-registered alongside delta, not as replacements for it. Both are
        # computed on the same bootstrap pairing so all three statistics are
        # directly comparable within a cell.
        co = st.paired_h2_costatistics_ci(
            real_word=r["word_dl"].to_numpy(dtype=np.float64),
            base_word=b["word_dl"].to_numpy(dtype=np.float64),
            real_head=r["head_dl"].to_numpy(dtype=np.float64),
            base_head=b["head_dl"].to_numpy(dtype=np.float64),
            n_resamples=n_resamples,
            ci_level=ci_level,
            rng=rng,
        )
        h2.update(co)

        h2["treebank_id"] = treebank_id
        h2["baseline_kind"] = kind
        h2["supports_h2"] = bool(h2["delta_diff"] > 0 and np.isfinite(h2["ci_lo"]) and h2["ci_lo"] > 0)
        h2["contradicts_h2"] = bool(h2["delta_diff"] < 0 and np.isfinite(h2["ci_hi"]) and h2["ci_hi"] < 0)
        # Each co-primary is scored by its own CI excluding zero, on the same
        # rule as delta. They are reported separately and never pooled: the
        # point of pre-registering two is to see whether they agree.
        h2["supports_h2_prob_adv"] = bool(
            h2["prob_adv_diff"] > 0 and np.isfinite(h2["prob_adv_ci_lo"]) and h2["prob_adv_ci_lo"] > 0)
        h2["supports_h2_med_rel"] = bool(
            h2["med_rel_diff"] > 0 and np.isfinite(h2["med_rel_ci_lo"]) and h2["med_rel_ci_lo"] > 0)
        h2["coprimaries_agree"] = bool(
            h2["supports_h2_prob_adv"] == h2["supports_h2_med_rel"])
        h2_rows.append(h2)

    return h1_rows, h2_rows


# ---------------------------------------------------------------------------
# H3
# ---------------------------------------------------------------------------

def run_h3(df: pd.DataFrame, treebank_id: str, cfg: dict) -> list[dict]:
    """Nested CONDITIONAL logistic models predicting realness within matched pairs.

    Model A (reduced):  P(real | pair) ~ word_dl
    Model B (full):     P(real | pair) ~ word_dl + head_dl

    Two design points, both corrections to an earlier version of this function:

    1. CONDITIONAL, not plain, logit. The data are matched pairs (a real
       sentence and a relinearization of its own tree), so the pair is the
       stratum. n_tokens, mean_arity and tree_depth are constant within a pair
       by construction; in a plain logit they are not controls at all, they are
       just collinear noise. Conditional logit drops them automatically along
       with the pair-level intercepts, which is the honest way to say "tree
       shape is held fixed by the design, not by adjustment".

    2. NO SHRINKAGE STATISTIC. The previous criterion compared |beta(word_dl)|
       between nested models. Logistic coefficients are not collapsible: adding
       a predictor with real signal rescales the remaining coefficients even
       when nothing is confounded. Simulation shows the shrinkage statistic
       fails in BOTH directions -- it reports up to +0.73 shrinkage when the
       two predictors are independent causes (derivativeness false), and
       reports -0.81 "suppression" when word_dl's true effect is exactly zero
       (derivativeness true). It measures noncollapsibility, not derivativeness.

    H3 is therefore scored on whether word_dl retains an independent effect once
    head_dl is in the model: derivativeness predicts beta(word_dl) -> 0 and no
    incremental discrimination from word_dl over head_dl alone.
    """
    try:
        import statsmodels.api as sm  # noqa: F401 - availability check
        from statsmodels.discrete.conditional_models import ConditionalLogit
    except ImportError:
        log.error("statsmodels not installed; H3 cannot be fitted")
        return []

    out = []
    alpha = cfg["thresholds"].get("h3_alpha", 0.05)

    for kind in cfg["baselines"]["kinds"]:
        sub = df[df["baseline_kind"].isin(["REAL", kind])].copy()
        if len(sub) < 40 or sub["is_real"].nunique() < 2:
            continue

        # z-score WITHIN treebank: coefficients are then comparable across
        # treebanks with wildly different sentence lengths.
        #
        # A predictor with zero variance carries no information AND makes the
        # design matrix singular (an all-zero column beside add_constant's
        # all-ones column), which would make sm.Logit raise and silently drop
        # this whole treebank from H3 -- shrinking the pre-registered
        # denominator without any record. Such predictors are dropped from the
        # model instead, and the drop is recorded in the output row.
        preds = ["word_dl", "head_dl", "n_tokens", "mean_arity", "tree_depth"]
        degenerate: list[str] = []
        for c in preds:
            sd = sub[c].std()
            if sd == 0 or not np.isfinite(sd):
                degenerate.append(c)
                sub[f"z_{c}"] = 0.0
            else:
                sub[f"z_{c}"] = (sub[c] - sub[c].mean()) / sd

        # word_dl and head_dl are the hypothesis itself; if either is constant
        # there is no H3 to test for this treebank.
        if "word_dl" in degenerate or "head_dl" in degenerate:
            log.warning(
                "%s/%s: word_dl or head_dl has zero variance (%s); H3 not testable",
                treebank_id, kind, degenerate,
            )
            continue

        # Pair-constant covariates are NOT dropped for being degenerate here --
        # conditional logit removes them structurally, because they cannot vary
        # within a stratum. Recording them keeps the audit trail.
        pair_constant = [
            c for c in ("n_tokens", "mean_arity", "tree_depth")
            if sub.groupby("sent_id")[c].nunique().max() <= 1
        ]

        y = sub["is_real"].to_numpy(dtype=np.float64)
        groups = sub["sent_id"].to_numpy()

        # Only complete, discordant pairs contribute to a conditional
        # likelihood; strata with both members on the same side of the outcome
        # are uninformative and are dropped by the estimator itself.
        sizes = sub.groupby("sent_id")["is_real"].agg(["size", "nunique"])
        n_pairs_used = int(((sizes["size"] == 2) & (sizes["nunique"] == 2)).sum())
        if n_pairs_used < 20:
            log.warning("%s/%s: only %d usable pairs; H3 not fitted",
                        treebank_id, kind, n_pairs_used)
            continue

        def fit(cols: list[str]):
            X = sub[cols].to_numpy(dtype=np.float64)
            try:
                return ConditionalLogit(y, X, groups=groups).fit(disp=0)
            except Exception as exc:  # noqa: BLE001 - separation / singularity
                log.warning("%s/%s: conditional logit failed (%s)",
                            treebank_id, kind, exc)
                return None

        reduced_cols = ["z_word_dl"]
        full_cols = ["z_word_dl", "z_head_dl"]

        m_red = fit(reduced_cols)
        m_full = fit(full_cols)
        if m_red is None or m_full is None:
            continue

        # No intercept in a conditional logit, so coefficients line up with
        # `cols` directly -- index 0 is the first predictor.
        def _at(model, col_list: list[str], name: str, attr: str = "params") -> float:
            return float(getattr(model, attr)[col_list.index(name)])

        beta_word_reduced = _at(m_red, reduced_cols, "z_word_dl")
        beta_word_full = _at(m_full, full_cols, "z_word_dl")
        beta_head_full = _at(m_full, full_cols, "z_head_dl")
        p_head_full = _at(m_full, full_cols, "z_head_dl", "pvalues")
        p_word_full = _at(m_full, full_cols, "z_word_dl", "pvalues")

        # Likelihood-ratio test for adding head_dl (nested conditional models).
        lr_stat = float(2 * (m_full.llf - m_red.llf))
        from scipy import stats as sp
        lr_p = float(sp.chi2.sf(max(lr_stat, 0.0), df=1))

        # Incremental discrimination: within-pair accuracy of head_dl alone vs.
        # head_dl + word_dl. Scale-free, and unlike a beta comparison it is not
        # affected by noncollapsibility.
        # Within-pair collinearity. The conditional likelihood sees only the
        # within-pair DIFFERENCES, so this -- not the correlation of the raw
        # metrics -- is what governs how separable the two coefficients are.
        # It runs 0.95-0.98 on UD data (VIF 11-23), which is why delta_auc and
        # the LR test are reported: the individual betas are jointly
        # identified but individually fragile, and should not be read as
        # standalone effect sizes.
        _r = sub[sub.is_real == 1].set_index("sent_id").sort_index()
        _b = sub[sub.is_real == 0].set_index("sent_id").sort_index()
        _c = _r.index.intersection(_b.index)
        if len(_c) > 2:
            _dw = (_r.loc[_c, "word_dl"] - _b.loc[_c, "word_dl"]).to_numpy(float)
            _dh = (_r.loc[_c, "head_dl"] - _b.loc[_c, "head_dl"]).to_numpy(float)
            _corr = (float(np.corrcoef(_dw, _dh)[0, 1])
                     if _dw.std() > 0 and _dh.std() > 0 else np.nan)
        else:
            _corr = np.nan
        within_pair_vif = (float(1.0 / (1.0 - _corr ** 2))
                           if np.isfinite(_corr) and abs(_corr) < 1 else np.nan)

        m_head_only = fit(["z_head_dl"])
        auc_full = _pair_auc(sub, ["z_word_dl", "z_head_dl"],
                             m_full.params if m_full is not None else None)
        auc_head = _pair_auc(sub, ["z_head_dl"],
                             m_head_only.params if m_head_only is not None else None)
        d_auc = (float(auc_full - auc_head)
                 if np.isfinite(auc_full) and np.isfinite(auc_head) else np.nan)

        out.append({
            "treebank_id": treebank_id,
            "baseline_kind": kind,
            "n_obs": int(len(sub)),
            "n_pairs_used": n_pairs_used,
            "model": "conditional_logit",
            "beta_word_reduced": beta_word_reduced,
            "se_word_reduced": _at(m_red, reduced_cols, "z_word_dl", "bse"),
            "beta_word_full": beta_word_full,
            "se_word_full": _at(m_full, full_cols, "z_word_dl", "bse"),
            "p_word_full": p_word_full,
            "beta_head_full": beta_head_full,
            "se_head_full": _at(m_full, full_cols, "z_head_dl", "bse"),
            "p_head_full": p_head_full,
            "pair_constant_covariates": ";".join(pair_constant),
            "dropped_covariates": ";".join(degenerate),
            "lr_stat": lr_stat,
            "lr_p": lr_p,
            "auc_head_only": auc_head,
            "auc_full": auc_full,
            "delta_auc": d_auc,
            "within_pair_corr": _corr,
            "within_pair_vif": within_pair_vif,
            # Derivativeness predicts word_dl carries NO independent signal once
            # head_dl is controlled. Support = word_dl's own effect is not
            # distinguishable from zero, while head_dl's is.
            "supports_h3": bool(
                np.isfinite(p_word_full) and np.isfinite(p_head_full)
                and p_word_full >= alpha and p_head_full < alpha
            ),
        })

    return out


def _pair_auc(sub: pd.DataFrame, cols: list[str], params) -> float:
    """Within-pair discrimination: share of pairs whose real member scores higher.

    This is the matched-pair analogue of AUC -- with one real and one baseline
    per stratum, "rank the real one first" is the whole question. Ties count a
    half, as in the usual AUC convention.
    """
    if params is None:
        return float("nan")
    score = sub[cols].to_numpy(dtype=np.float64) @ np.asarray(params, dtype=np.float64)
    tmp = pd.DataFrame({"sent_id": sub["sent_id"].to_numpy(),
                        "is_real": sub["is_real"].to_numpy(), "score": score})
    real = tmp[tmp.is_real == 1].set_index("sent_id")["score"]
    base = tmp[tmp.is_real == 0].set_index("sent_id")["score"]
    common = real.index.intersection(base.index)
    if len(common) == 0:
        return float("nan")
    r, b = real.loc[common].to_numpy(), base.loc[common].to_numpy()
    return float((np.sum(r > b) + 0.5 * np.sum(r == b)) / len(common))


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _classify(n_support: int, n_total: int, cfg: dict) -> str:
    """Apply the pre-registered 70/30 rule. EQUIVOCAL is a real outcome."""
    if n_total == 0:
        return "INSUFFICIENT_DATA"
    frac = n_support / n_total
    if frac >= cfg["thresholds"]["support_fraction"]:
        return "SUPPORTED"
    if frac < cfg["thresholds"]["contradict_fraction"]:
        return "CONTRADICTED"
    return "EQUIVOCAL"


def _power_warning(h1: pd.DataFrame, min_sentences: int = 500) -> dict | None:
    """Flag a run too small for a CONTRADICTED verdict to mean anything.

    A null result from an underpowered run is an absence of evidence, not
    evidence of absence. The fixture run (~30 sentences) reports CONTRADICTED
    on all three hypotheses purely because no CI can exclude zero at that n --
    the point estimates still point the predicted way. Without this flag a
    reader could mistake a smoke test for a refutation of DLM.
    """
    if h1.empty or "n_sentences" not in h1:
        return None
    n = int(h1["n_sentences"].max())
    if n >= min_sentences:
        return None
    return {
        "underpowered": True,
        "max_sentences_per_treebank": n,
        "advisory": (
            f"Only {n} sentences per treebank were analyzed (< {min_sentences}). "
            "Confidence intervals at this scale cannot exclude zero, so a "
            "CONTRADICTED verdict here reflects LOW POWER, not evidence against "
            "the hypothesis. Inspect the point estimates ('delta' in "
            "per_treebank.csv) rather than the verdict labels. This is the "
            "expected state for a --fixture smoke test."
        ),
    }


def build_verdict(
    h1: pd.DataFrame,
    h2: pd.DataFrame,
    h3: pd.DataFrame,
    cfg: dict,
    run_meta: dict,
) -> dict:
    """Assemble results/verdict.json with FDR correction applied per family."""
    q = cfg["thresholds"]["fdr_q"]
    verdict: dict = {
        "run": run_meta,
        "pre_registered_thresholds": cfg["thresholds"],
        "hypotheses": {},
    }

    warning = _power_warning(h1)
    if warning is not None:
        verdict["power_advisory"] = warning

    # --- H1: needs BOTH metrics negative in a treebank to count as support ---
    #
    # The verdict rests on B1 only. B2 holds each node's left/right dependent
    # COUNTS to the real values, which constrains word order so tightly that on
    # short sentences it frequently reproduces the original order exactly (~21%
    # of samples on the fixture). Those samples have a DL difference of exactly
    # zero by construction, which drags the paired delta toward 0 regardless of
    # whether DLM holds. B2 is therefore a valid *conservative* control for H2
    # and H3 -- where both conditions are affected equally -- but it is not a
    # usable sanity check for "is there a DLM effect at all". Scoring H1 on B2
    # would report a broken pipeline whenever the control is merely strict.
    if len(h1):
        h1 = h1.copy()
        rej, adj = st.benjamini_hochberg(h1["p_value"].to_numpy(), q=q)
        h1["p_adj"] = adj
        h1["significant_fdr"] = rej

        per_tb = []
        for (tb, kind), g in h1.groupby(["treebank_id", "baseline_kind"]):
            both = bool(g["supports_h1"].all() and g["significant_fdr"].all())
            per_tb.append({"treebank_id": tb, "baseline_kind": kind, "supports": both})

        primary = [r for r in per_tb if r["baseline_kind"] == "B1_uniform"]
        secondary = [r for r in per_tb if r["baseline_kind"] != "B1_uniform"]
        scored = primary if primary else per_tb
        n_sup = sum(1 for r in scored if r["supports"])

        verdict["hypotheses"]["H1"] = {
            "statement": "Real sentences minimize dependency length vs. random "
                         "projective linearization, on both metrics.",
            "role": "sanity check -- failure indicates a broken pipeline, not a refuted theory",
            "predicted_direction": "delta < 0 on both metrics",
            "scored_on_baseline": "B1_uniform",
            "baseline_note": (
                "B2_arity_preserving is reported but not scored: it preserves each "
                "node's left/right dependent counts, so it often reproduces the real "
                "order exactly and cannot detect a DLM effect."
            ),
            "n_conditions": len(scored),
            "n_supporting": n_sup,
            "fraction_supporting": (n_sup / len(scored)) if scored else 0.0,
            "verdict": _classify(n_sup, len(scored), cfg),
            "secondary_B2_n_supporting": sum(1 for r in secondary if r["supports"]),
            "secondary_B2_n_conditions": len(secondary),
        }

    # --- H2: head metric shows a larger effect than word metric --------------
    if len(h2):
        h2 = h2.copy()
        rej, adj = st.benjamini_hochberg(h2["p_value"].to_numpy(), q=q)
        h2["p_adj"] = adj
        h2["significant_fdr"] = rej
        support = (h2["supports_h2"] & h2["significant_fdr"]).sum()
        contra = (h2["contradicts_h2"] & h2["significant_fdr"]).sum()
        verdict["hypotheses"]["H2"] = {
            "statement": "The head-based metric shows a larger real-vs-random effect "
                         "than the word-based metric.",
            "predicted_direction": "|delta_head| - |delta_word| > 0",
            "n_conditions": int(len(h2)),
            "n_supporting": int(support),
            "n_contradicting": int(contra),
            "n_equivocal": int(len(h2) - support - contra),
            "fraction_supporting": float(support / len(h2)),
            "median_delta_diff": float(h2["delta_diff"].median()),
            "verdict": _classify(int(support), int(len(h2)), cfg),
            # Co-primaries, pre-registered alongside delta. Reported with their
            # own verdicts and never pooled with it: delta is tie-sensitive,
            # these are not, and the comparison between them is the point.
            "coprimary_prob_advantage": {
                "statistic": "P(real shorter) with ties split, head minus word",
                "n_supporting": int((h2["supports_h2_prob_adv"]).sum()),
                "fraction_supporting": float(h2["supports_h2_prob_adv"].mean()),
                "median_diff": float(h2["prob_adv_diff"].median()),
                "verdict": _classify(int(h2["supports_h2_prob_adv"].sum()),
                                     int(len(h2)), cfg),
            },
            "coprimary_median_relative_reduction": {
                "statistic": "median per-sentence (base-real)/base, head minus word",
                "n_supporting": int((h2["supports_h2_med_rel"]).sum()),
                "fraction_supporting": float(h2["supports_h2_med_rel"].mean()),
                "median_diff": float(h2["med_rel_diff"].median()),
                "verdict": _classify(int(h2["supports_h2_med_rel"].sum()),
                                     int(len(h2)), cfg),
            },
            "coprimaries_agree_n": int(h2["coprimaries_agree"].sum()),
            "median_tie_rate_word": float(h2["tie_rate_word"].median()),
            "median_tie_rate_head": float(h2["tie_rate_head"].median()),
        }

    # --- H3: word_DL is derivative of head_DL --------------------------------
    if len(h3):
        h3 = h3.copy()
        rej, adj = st.benjamini_hochberg(h3["lr_p"].to_numpy(), q=q)
        h3["lr_p_adj"] = adj
        h3["lr_significant_fdr"] = rej
        support = (h3["supports_h3"] & h3["lr_significant_fdr"]).sum()
        verdict["hypotheses"]["H3"] = {
            "statement": "Word-based dependency length is derivative of head-based "
                         "dependency length.",
            "model": "conditional_logit_on_matched_pairs",
            "predicted_direction": "beta(word_DL) indistinguishable from zero once "
                                   "head_DL is in the model, with beta(head_DL) "
                                   "significant and the nested LR test passing FDR",
            "statistic_note": "Scored on whether word_DL retains an independent "
                              "effect, NOT on shrinkage in |beta(word_DL)|. "
                              "Logistic coefficients are not collapsible, so a "
                              "between-model beta comparison measures rescaling "
                              "rather than derivativeness.",
            "n_conditions": int(len(h3)),
            "n_supporting": int(support),
            "fraction_supporting": float(support / len(h3)),
            "median_beta_word_reduced": float(h3["beta_word_reduced"].median()),
            "median_beta_word_full": float(h3["beta_word_full"].median()),
            "median_beta_head_full": float(h3["beta_head_full"].median()),
            "median_delta_auc": float(h3["delta_auc"].median(skipna=True)),
            "n_beta_head_wrong_sign": int((h3["beta_head_full"] > 0).sum()),
            "max_within_pair_vif": float(h3["within_pair_vif"].max(skipna=True)),
            "collinearity_note": "word_DL and head_DL differences correlate "
                                 "0.95-0.98 within pairs. The LR test and "
                                 "delta_AUC are the interpretable quantities; "
                                 "individual betas are fragile under this VIF.",
            "verdict": _classify(int(support), int(len(h3)), cfg),
        }

    return verdict


def write_verdict(verdict: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", path)
