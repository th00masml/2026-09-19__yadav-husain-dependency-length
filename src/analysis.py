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

          P(real) ~ z(word_DL) + z(head_DL) + z(n_tokens)
                    + z(mean_arity) + z(tree_depth)

      Predicted: |beta(word_DL)| shrinks by >50% once head_DL enters the model,
      while beta(head_DL) stays significant.

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
        h2["treebank_id"] = treebank_id
        h2["baseline_kind"] = kind
        h2["supports_h2"] = bool(h2["delta_diff"] > 0 and np.isfinite(h2["ci_lo"]) and h2["ci_lo"] > 0)
        h2["contradicts_h2"] = bool(h2["delta_diff"] < 0 and np.isfinite(h2["ci_hi"]) and h2["ci_hi"] < 0)
        h2_rows.append(h2)

    return h1_rows, h2_rows


# ---------------------------------------------------------------------------
# H3
# ---------------------------------------------------------------------------

def run_h3(df: pd.DataFrame, treebank_id: str, cfg: dict) -> list[dict]:
    """Nested logistic models predicting realness; does word_DL survive head_DL?

    Model A (reduced):  P(real) ~ word_dl + n_tokens + mean_arity + tree_depth
    Model B (full):     Model A + head_dl

    H3 predicts |beta(word_dl)| shrinks substantially from A to B.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        log.error("statsmodels not installed; H3 cannot be fitted")
        return []

    out = []
    shrink_min = cfg["thresholds"]["h3_shrinkage_min"]

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
        if degenerate:
            log.warning(
                "%s/%s: dropping zero-variance covariate(s) %s from the H3 model",
                treebank_id, kind, degenerate,
            )

        y = sub["is_real"].to_numpy(dtype=np.float64)

        def fit(cols: list[str]):
            X = sm.add_constant(sub[cols].to_numpy(dtype=np.float64), has_constant="add")
            try:
                return sm.Logit(y, X).fit(disp=0, maxiter=200)
            except Exception as exc:  # noqa: BLE001 - separation / singularity
                log.warning("%s/%s: logit failed (%s)", treebank_id, kind, exc)
                return None

        covariates = [f"z_{c}" for c in ("n_tokens", "mean_arity", "tree_depth")
                      if c not in degenerate]
        reduced_cols = ["z_word_dl"] + covariates
        full_cols = ["z_word_dl", "z_head_dl"] + covariates

        m_red = fit(reduced_cols)
        m_full = fit(full_cols)
        if m_red is None or m_full is None:
            continue

        # Coefficients are read positionally; add_constant prepends the
        # intercept, so index 0 is the constant and the predictors follow in
        # the order given above. `_beta` keeps that contract explicit rather
        # than scattering magic indices.
        def _at(model, col_list: list[str], name: str, attr: str = "params") -> float:
            return float(getattr(model, attr)[col_list.index(name) + 1])

        beta_word_reduced = _at(m_red, reduced_cols, "z_word_dl")
        beta_word_full = _at(m_full, full_cols, "z_word_dl")
        beta_head_full = _at(m_full, full_cols, "z_head_dl")
        se_head_full = _at(m_full, full_cols, "z_head_dl", "bse")
        p_head_full = _at(m_full, full_cols, "z_head_dl", "pvalues")
        p_word_full = _at(m_full, full_cols, "z_word_dl", "pvalues")

        shrinkage = (
            1.0 - abs(beta_word_full) / abs(beta_word_reduced)
            if abs(beta_word_reduced) > 1e-12 else np.nan
        )

        # Likelihood-ratio test for adding head_dl
        lr_stat = float(2 * (m_full.llf - m_red.llf))
        from scipy import stats as sp
        lr_p = float(sp.chi2.sf(max(lr_stat, 0.0), df=1))

        out.append({
            "treebank_id": treebank_id,
            "baseline_kind": kind,
            "n_obs": int(len(sub)),
            "beta_word_reduced": beta_word_reduced,
            "se_word_reduced": _at(m_red, reduced_cols, "z_word_dl", "bse"),
            "beta_word_full": beta_word_full,
            "se_word_full": _at(m_full, full_cols, "z_word_dl", "bse"),
            "p_word_full": p_word_full,
            "beta_head_full": beta_head_full,
            "se_head_full": se_head_full,
            "p_head_full": p_head_full,
            "beta_ntokens_full": (_at(m_full, full_cols, "z_n_tokens")
                                  if "z_n_tokens" in full_cols else np.nan),
            "beta_arity_full": (_at(m_full, full_cols, "z_mean_arity")
                                if "z_mean_arity" in full_cols else np.nan),
            "beta_depth_full": (_at(m_full, full_cols, "z_tree_depth")
                                if "z_tree_depth" in full_cols else np.nan),
            "dropped_covariates": ";".join(degenerate),
            "shrinkage": float(shrinkage) if np.isfinite(shrinkage) else np.nan,
            "lr_stat": lr_stat,
            "lr_p": lr_p,
            "pseudo_r2_reduced": float(m_red.prsquared),
            "pseudo_r2_full": float(m_full.prsquared),
            # H3 needs BOTH: word_dl collapses AND head_dl stays significant.
            "supports_h3": bool(
                np.isfinite(shrinkage)
                and shrinkage >= shrink_min
                and p_head_full < 0.05
            ),
        })

    return out


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
            "predicted_direction": f">={cfg['thresholds']['h3_shrinkage_min']:.0%} shrinkage in "
                                   "|beta(word_DL)| when head_DL is added, with beta(head_DL) significant",
            "n_conditions": int(len(h3)),
            "n_supporting": int(support),
            "fraction_supporting": float(support / len(h3)),
            "median_shrinkage": float(h3["shrinkage"].median(skipna=True)),
            "median_beta_word_reduced": float(h3["beta_word_reduced"].median()),
            "median_beta_word_full": float(h3["beta_word_full"].median()),
            "median_beta_head_full": float(h3["beta_head_full"].median()),
            "verdict": _classify(int(support), int(len(h3)), cfg),
        }

    return verdict


def write_verdict(verdict: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", path)
