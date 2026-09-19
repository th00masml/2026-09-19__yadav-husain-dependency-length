"""Reviewer-added regression tests for yadav-husain-dependency-length.

These complement tests/test_main.py rather than replacing it. Each test here
was written against a specific defect or untested edge case found during code
review; the docstring names the defect so a future failure is diagnosable.

Every test runs with ZERO network access.

Run:  python -m pytest tests/ -v
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN_DIR))

from src import analysis, linearize as lin, metrics as mx, stats as st, ud_loader  # noqa: E402
from src.tree import (  # noqa: E402
    ROOT,
    Sentence,
    is_projective,
    mean_arity,
    tree_depth,
    validate,
)

FIXTURE = RUN_DIR / "tests" / "fixtures" / "mini.conllu"
RESULTS = RUN_DIR / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mk(heads: list[int], sent_id: str = "s") -> Sentence:
    """Build a Sentence from a head vector alone (tokens/deprels are filler)."""
    n = len(heads)
    return Sentence(
        tokens=tuple(f"w{i}" for i in range(n)),
        heads=np.array(heads, dtype=np.int32),
        deprels=tuple("dep" for _ in range(n)),
        upos=tuple("X" for _ in range(n)),
        treebank_id="t",
        sent_id=sent_id,
    )


def all_valid_trees(n: int):
    """Every well-formed tree on n nodes, as Sentence objects."""
    for heads in itertools.product(range(ROOT, n), repeat=n):
        arr = np.array(heads, dtype=np.int32)
        if validate(arr, n) is None:
            yield mk(list(heads))


def synthetic_pair_table(
    n_sent: int = 200,
    constant_col: str | None = None,
    seed: int = 1,
) -> pd.DataFrame:
    """A real/baseline analysis table shaped exactly like build_treebank_table's.

    `constant_col` forces one covariate to be constant across every row, which
    is what makes the z-score degenerate and the design matrix singular.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sent):
        nt = 5.0 if constant_col == "n_tokens" else float(rng.integers(5, 25))
        ar = 1.0 if constant_col == "mean_arity" else float(rng.normal(2.0, 0.3))
        for is_real, kind in ((1, "REAL"), (0, "B1_uniform")):
            rows.append({
                "sent_id": f"s{i}",
                "treebank_id": "tb",
                "baseline_kind": kind,
                "is_real": is_real,
                "n_tokens": nt,
                "n_arcs": nt - 1,
                "word_dl": float(rng.normal(20.0 if is_real else 30.0, 5.0)),
                "head_dl": float(rng.normal(8.0 if is_real else 14.0, 3.0)),
                "mean_arity": ar,
                "tree_depth": float(rng.integers(2, 6)),
            })
    return pd.DataFrame(rows)


MIN_CFG = {
    "thresholds": {
        "support_fraction": 0.70,
        "contradict_fraction": 0.30,
        "h3_shrinkage_min": 0.50,
        "fdr_q": 0.05,
        "ci_level": 0.95,
    },
    "baselines": {"kinds": ["B1_uniform"]},
}


# ===========================================================================
# TEST A -- BUG 1: a constant covariate silently deletes a treebank from H3
# ===========================================================================

def test_a_h3_survives_a_degenerate_covariate():
    """A constant covariate must not make a treebank vanish from H3.

    DEFECT: analysis.run_h3 z-scores each predictor and, when sd == 0, assigns
    the scalar 0.0. That leaves an all-zero column in the design matrix next to
    add_constant's all-ones column, so the matrix is singular, sm.Logit raises
    LinAlgError, `fit()` catches it and returns None, and `continue` drops the
    ENTIRE treebank from H3 -- silently, with only a log line.

    The failure is data-dependent and realistic: any treebank whose sentences
    were length-filtered to a single value, or a small/fixture treebank where
    mean_arity happens to be constant, triggers it. The verdict's n_conditions
    then shrinks without anything marking those treebanks as unanalyzable, which
    changes the denominator of the pre-registered support fraction.

    A degenerate covariate carries no information and should simply be dropped
    from the model, leaving the scientifically meaningful coefficients on
    word_dl and head_dl estimable.
    """
    healthy = analysis.run_h3(synthetic_pair_table(), "tb", MIN_CFG)
    assert len(healthy) == 1, "the healthy control case should fit"

    degenerate = analysis.run_h3(
        synthetic_pair_table(constant_col="n_tokens"), "tb", MIN_CFG
    )
    assert len(degenerate) == 1, (
        "a treebank with one constant covariate produced NO H3 row at all; it "
        "was silently dropped from the hypothesis denominator instead of being "
        "fitted without the uninformative predictor"
    )
    row = degenerate[0]
    assert np.isfinite(row["beta_word_full"]), "beta(word_DL) must still be estimable"
    assert np.isfinite(row["beta_head_full"]), "beta(head_DL) must still be estimable"


def test_a2_h3_reports_coefficients_in_the_documented_positions():
    """m.params[1] must be word_dl and m.params[2] must be head_dl.

    run_h3 reads its coefficients POSITIONALLY out of the fitted params vector,
    which is only correct because add_constant prepends the intercept and the
    column order in `full_cols` is [word, head, ntokens, arity, depth]. If
    anyone reorders full_cols, every reported beta silently shifts by one column
    and the whole H3 result becomes wrong-but-plausible. This pins the contract.
    """
    df = synthetic_pair_table(n_sent=400, seed=7)
    # Make word_dl strongly separating and head_dl weakly so the two betas are
    # distinguishable by magnitude, not just by position.
    out = analysis.run_h3(df, "tb", MIN_CFG)
    assert out, "fit failed"
    row = out[0]

    import statsmodels.api as sm

    sub = df[df["baseline_kind"].isin(["REAL", "B1_uniform"])].copy()
    for c in ["word_dl", "head_dl", "n_tokens", "mean_arity", "tree_depth"]:
        sd = sub[c].std()
        sub[f"z_{c}"] = 0.0 if sd == 0 else (sub[c] - sub[c].mean()) / sd
    cols = ["z_word_dl", "z_head_dl", "z_n_tokens", "z_mean_arity", "z_tree_depth"]
    X = sm.add_constant(sub[cols].to_numpy(dtype=np.float64), has_constant="add")
    m = sm.Logit(sub["is_real"].to_numpy(dtype=np.float64), X).fit(disp=0, maxiter=200)

    assert np.all(X[:, 0] == 1.0), "add_constant no longer prepends the intercept"
    assert row["beta_word_full"] == pytest.approx(float(m.params[1]))
    assert row["beta_head_full"] == pytest.approx(float(m.params[2]))


# ===========================================================================
# TEST B -- BUG 2: the test suite overwrites the real results/
# ===========================================================================

def test_b_pipeline_run_does_not_clobber_real_results(config_with_tmp_results):
    """An end-to-end test run must write to a sandbox, never to results/.

    DEFECT: test_main.py::test_8 shells out to `main.py --fixture --fast
    --offline` with cwd=RUN_DIR and no output redirection. main.py resolves its
    output paths from config.yaml relative to RUN_DIR, so that subprocess
    OVERWRITES results/verdict.json, per_treebank.csv, h2_comparison.csv,
    h3_models.csv, treebank_manifest.csv and every figure with 30-sentence
    fixture output.

    The real deliverable -- the 5-treebank / ~37.8k-sentence run that README.md
    reports (H1 SUPPORTED 5/5, H2 CONTRADICTED 2/10, H3 EQUIVOCAL 3/10) -- is
    destroyed by simply running `pytest`. It was in fact already destroyed on
    disk at review time: results/verdict.json recorded mode.fixture=true and
    n_sentences=30, contradicting every number in README.md.

    This test proves output redirection works, so test_8 can adopt it.
    """
    cfg_path, out_dir = config_with_tmp_results

    before = {}
    if RESULTS.exists():
        before = {
            p.name: p.stat().st_mtime_ns
            for p in RESULTS.iterdir()
            if p.is_file()
        }

    result = subprocess.run(
        [sys.executable, "main.py", "--fixture", "--fast", "--offline",
         "--config", str(cfg_path)],
        cwd=str(RUN_DIR),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"sandboxed pipeline exited {result.returncode}\n{result.stdout[-3000:]}\n"
        f"{result.stderr[-3000:]}"
    )

    assert (out_dir / "verdict.json").exists(), "sandbox run wrote no verdict"

    after = {}
    if RESULTS.exists():
        after = {
            p.name: p.stat().st_mtime_ns
            for p in RESULTS.iterdir()
            if p.is_file()
        }
    assert before == after, (
        "running the pipeline modified the real results/ directory: "
        f"{sorted(set(before) ^ set(after)) or 'mtimes changed'}"
    )


def test_b2_committed_results_are_internally_consistent():
    """results/verdict.json must not claim fixture mode while README cites UD.

    Guards the exact inconsistency found at review time. A verdict produced by
    `--fixture` describes 30 toy sentences and must never be the artifact left
    sitting in results/ while README.md reports a five-treebank study. If this
    fails, someone ran the test suite (or `main.py --fixture`) last and the real
    results need regenerating with `python main.py --fast --offline`.
    """
    verdict_path = RESULTS / "verdict.json"
    if not verdict_path.exists():
        pytest.skip("no results/verdict.json yet; run main.py first")

    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    mode = verdict.get("run", {}).get("mode", {})

    assert not mode.get("fixture", False), (
        "results/verdict.json was produced by a --fixture smoke run (30 toy "
        "sentences). It does not support the findings in README.md. Regenerate "
        "with: python main.py --fast --offline"
    )

    if "power_advisory" in verdict:
        pytest.fail(
            "results/verdict.json carries a power_advisory, meaning fewer than "
            "500 sentences/treebank were analyzed. The committed deliverable "
            "must come from the full data run."
        )


# ===========================================================================
# TEST C -- head-based metric: exhaustive fast-vs-naive over ALL small trees
# ===========================================================================

@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_c_head_dl_fast_matches_naive_exhaustively(n):
    """Exhaustive, not random: every well-formed tree on n nodes.

    test_main.py::test_3 samples 200 random trees, which leaves the
    combinatorial corners of the prefix-sum arithmetic (empty spans, adjacent
    head/dependent, root-adjacent arcs) to chance. Here we enumerate EVERY valid
    tree for n <= 6 and check both scopes and both endpoint settings, so an
    off-by-one in the cum[] indexing cannot hide.

    This also pins `within_span`, whose fast path uses a prev_child/next_child
    precomputation whose published rationale is subtler than the code: the code
    actually evaluates the predicate for every token in the span, so it is
    correct, but it is exactly the kind of "clever" path that silently rots.
    """
    checked = 0
    for s in all_valid_trees(n):
        for scope in ("global", "within_span"):
            for endpoints in (False, True):
                fast = mx.head_dl_arcs(s, scope=scope, include_endpoints=endpoints)
                slow = mx.head_dl_arcs_naive(s, scope=scope, include_endpoints=endpoints)
                assert np.array_equal(fast, slow), (
                    f"heads={s.heads.tolist()} scope={scope} endpoints={endpoints}: "
                    f"fast={fast.tolist()} naive={slow.tolist()}"
                )
        checked += 1
    assert checked > 0, f"no valid trees enumerated for n={n}"


def test_c2_head_dl_rejects_an_unknown_scope():
    """A typo'd scope must raise, not silently fall through to a default.

    Silently defaulting would mean config.yaml's head_count.scope could be
    misspelled and the run would report results for the wrong metric definition
    -- the contested quantity of the entire experiment.
    """
    s = mk([-1, 0, 1])
    with pytest.raises(ValueError, match="scope"):
        mx.head_dl_arcs(s, scope="globl")


def test_c3_adjacent_arc_has_zero_head_dl():
    """|h-d| == 1 leaves no tokens strictly inside, so head_dl must be 0.

    The empty-span boundary is where prefix-sum code goes wrong (cum[hi] -
    cum[lo+1] with hi == lo+1 must be 0, never -1).
    """
    s = mk([-1, 0, 1, 2])          # a chain: every arc is adjacent
    head = mx.head_dl_arcs(s, scope="global", include_endpoints=False)
    assert head.tolist() == [0, 0, 0], f"adjacent arcs gave {head.tolist()}"
    assert np.all(head >= 0), "negative head count from an empty span"


# ===========================================================================
# TEST D -- tree validation and loader robustness
# ===========================================================================

def test_d_validate_catches_every_malformed_head_vector():
    """Each rejection reason must fire on a tree that exhibits exactly it.

    The manifest's rejection breakdown is a data-quality signal readers use to
    judge whether a treebank's annotation is trustworthy. If a reason is
    mislabeled the breakdown is misleading, so pin each one.
    """
    assert validate(np.array([-1, 0, 1], dtype=np.int32), 3) is None

    assert validate(np.array([0, 1, 2], dtype=np.int32), 3) == "no_root"
    assert validate(np.array([-1, -1, 0], dtype=np.int32), 3) == "multiple_roots"
    assert validate(np.array([-1, 9, 0], dtype=np.int32), 3) == "head_out_of_range"
    assert validate(np.array([-1, 1, 0], dtype=np.int32), 3) == "self_loop"
    assert validate(np.array([-1, 0], dtype=np.int32), 3) == "length_mismatch"

    # A genuine cycle that still has a root elsewhere: 0 is root, 1<->2 cycle.
    assert validate(np.array([-1, 2, 1], dtype=np.int32), 3) == "cycle"


def test_d2_loader_rejects_head_pointing_at_a_dropped_token():
    """A HEAD referring to a multiword range must reject, not silently remap.

    Dropping MWT lines renumbers every token. If a HEAD column pointed at the
    range id itself, naive remapping would attach the dependent to whatever
    token inherited that index -- a wrong arc that never raises. The loader must
    reject the sentence instead.
    """
    text = (
        "# sent_id = trap\n"
        "1\tvi\t_\tVERB\t_\t_\t0\troot\t_\t_\n"
        "2-3\tal\t_\t_\t_\t_\t_\t_\t_\t_\n"
        "2\ta\t_\tADP\t_\t_\t4\tcase\t_\t_\n"
        "3\tel\t_\tDET\t_\t_\t4\tdet\t_\t_\n"
        "4\tperro\t_\tNOUN\t_\t_\t1\tobj\t_\t_\n"
        "\n"
    )
    stats = ud_loader.LoadStats("t")
    sents = list(ud_loader.parse_conllu_text(text, "t", 3, 100, stats))

    assert len(sents) == 1
    s = sents[0]
    assert s.n_tokens == 4, "the 2-3 range line became a token"
    assert s.tokens == ("vi", "a", "el", "perro")
    # 1-based ids 1,2,3,4 -> 0-based 0,1,2,3; heads 0,4,4,1 -> -1,3,3,0
    assert s.heads.tolist() == [-1, 3, 3, 0]


def test_d3_loader_is_robust_to_crlf_and_missing_trailing_newline():
    """Windows line endings and a missing final blank line must still parse.

    This run was produced on Windows, where a CoNLL-U file round-tripped through
    a text editor acquires \\r\\n. If the parser treated \\r as content the ID
    column would stop matching and every length would be wrong.
    """
    base = (
        "# sent_id = x1\n"
        "1\tthe\t_\tDET\t_\t_\t2\tdet\t_\t_\n"
        "2\tdog\t_\tNOUN\t_\t_\t3\tnsubj\t_\t_\n"
        "3\tran\t_\tVERB\t_\t_\t0\troot\t_\t_\n"
        "\n"
    )

    for label, text in [
        ("lf", base),
        ("crlf", base.replace("\n", "\r\n")),
        ("no-trailing-newline", base.rstrip("\n")),
    ]:
        stats = ud_loader.LoadStats("t")
        sents = list(ud_loader.parse_conllu_text(text, "t", 3, 100, stats))
        assert len(sents) == 1, f"{label}: parsed {len(sents)} sentences"
        assert sents[0].tokens == ("the", "dog", "ran"), f"{label}: {sents[0].tokens}"
        assert sents[0].heads.tolist() == [1, 2, -1], f"{label}: bad heads"


def test_d4_loader_length_filters_are_inclusive_bounds():
    """min_tokens/max_tokens must be inclusive, matching the config comments.

    An off-by-one here changes which sentences enter the study -- a silent
    change to the sampling frame.
    """
    def n_token_sentence(k: int) -> str:
        lines = [f"# sent_id = n{k}"]
        for i in range(1, k + 1):
            head = 0 if i == 1 else 1
            rel = "root" if i == 1 else "dep"
            lines.append(f"{i}\tw{i}\t_\tX\t_\t_\t{head}\t{rel}\t_\t_")
        return "\n".join(lines) + "\n\n"

    for k, lo, hi, expected in [
        (3, 3, 5, 1),     # exactly at the lower bound -> kept
        (2, 3, 5, 0),     # below -> dropped
        (5, 3, 5, 1),     # exactly at the upper bound -> kept
        (6, 3, 5, 0),     # above -> dropped
    ]:
        stats = ud_loader.LoadStats("t")
        got = list(ud_loader.parse_conllu_text(n_token_sentence(k), "t", lo, hi, stats))
        assert len(got) == expected, (
            f"n={k} with bounds [{lo},{hi}] kept {len(got)}, expected {expected}"
        )


# ===========================================================================
# TEST E -- statistics: directions, pairing, and edge cases
# ===========================================================================

def test_e_paired_bootstrap_ci_brackets_its_own_estimate():
    """H2's CI must contain its point estimate and have the right sign.

    If the paired bootstrap drifted (e.g. resampling the two metrics with
    different indices), the CI would stop bracketing delta_diff. Here head_dl
    separates real from baseline much more sharply than word_dl, so
    |delta_head| - |delta_word| must be positive and the CI must sit above zero.
    """
    rng = np.random.default_rng(4)
    n = 400
    real_word = rng.normal(10.0, 3.0, n)
    base_word = real_word + rng.normal(1.0, 3.0, n)      # weak separation
    real_head = rng.normal(10.0, 1.0, n)
    base_head = real_head + rng.normal(6.0, 1.0, n)      # strong separation

    out = st.paired_delta_difference_ci(
        real_word=real_word, base_word=base_word,
        real_head=real_head, base_head=base_head,
        n_resamples=600, rng=rng,
    )

    assert out["ci_lo"] <= out["delta_diff"] <= out["ci_hi"], (
        f"CI [{out['ci_lo']}, {out['ci_hi']}] excludes the estimate "
        f"{out['delta_diff']}"
    )
    assert out["delta_diff"] > 0, "the sharply-separating metric did not win"
    assert out["ci_lo"] > 0, "CI failed to exclude zero on a clear-cut case"
    assert 0.0 <= out["p_value"] <= 1.0


def test_e2_paired_bootstrap_rejects_mismatched_lengths():
    """Four arrays indexed by sentence must all be the same length.

    A length mismatch means the pairing is broken; it must raise rather than
    broadcast into a meaningless answer.
    """
    with pytest.raises(ValueError):
        st.paired_delta_difference_ci(
            real_word=np.zeros(10), base_word=np.zeros(10),
            real_head=np.zeros(10), base_head=np.zeros(9),
            n_resamples=10,
        )


def test_e3_statistics_degrade_gracefully_on_empty_and_tied_input():
    """Empty/degenerate input must yield nan or a neutral value, never a crash.

    A treebank that filters down to nothing must not take the whole run with it.
    """
    assert np.isnan(st.cliffs_delta(np.array([]), np.array([1.0])))
    assert np.isnan(st.paired_cliffs_delta(np.array([]), np.array([])))
    assert st.sign_test_p(np.zeros(10)) == 1.0        # all ties -> no evidence

    rej, adj = st.benjamini_hochberg(np.array([]))
    assert len(rej) == 0 and len(adj) == 0

    rej, adj = st.benjamini_hochberg(np.array([np.nan, np.nan]))
    assert not rej.any(), "NaN p-values must never be counted as rejections"


def test_e4_cliffs_delta_sign_convention_matches_the_dlm_prediction():
    """delta < 0 must mean 'real is SHORTER than baseline'.

    The entire H1 verdict is `delta < 0 and ci_hi < 0`. A flipped sign
    convention would invert every reported conclusion while every CI still
    looked healthy, so pin the direction explicitly.
    """
    real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])      # shorter
    base = np.array([9.0, 9.0, 9.0, 9.0, 9.0])      # longer
    assert st.paired_cliffs_delta(real, base) == pytest.approx(-1.0)
    assert st.cliffs_delta(real, base) == pytest.approx(-1.0)
    assert st.paired_cliffs_delta(base, real) == pytest.approx(1.0)


# ===========================================================================
# TEST F -- verdict logic: the pre-registered decision rule
# ===========================================================================

@pytest.mark.parametrize(
    "n_support, n_total, expected",
    [
        (0, 0, "INSUFFICIENT_DATA"),
        (10, 10, "SUPPORTED"),
        (7, 10, "SUPPORTED"),        # exactly at the 0.70 bar -> inclusive
        (6, 10, "EQUIVOCAL"),
        (3, 10, "EQUIVOCAL"),        # exactly at the 0.30 bar -> NOT contradicted
        (2, 10, "CONTRADICTED"),
        (0, 10, "CONTRADICTED"),
    ],
)
def test_f_classify_respects_the_preregistered_boundaries(n_support, n_total, expected):
    """The 70/30 rule must be applied with the documented inclusivity.

    config.yaml says SUPPORTED at >= 0.70 and CONTRADICTED at < 0.30. The
    boundary cases (7/10 and 3/10) are where a `>` / `>=` slip would flip a
    published verdict, so they are pinned rather than assumed.
    """
    assert analysis._classify(n_support, n_total, MIN_CFG) == expected


def test_f2_power_advisory_fires_only_on_underpowered_runs():
    """The advisory must appear for a 30-sentence run and vanish for a real one.

    This flag is what stops a smoke test reading as a refutation of DLM, so its
    trigger condition needs a test of its own.
    """
    small = pd.DataFrame([{"n_sentences": 30}])
    large = pd.DataFrame([{"n_sentences": 11108}])

    warn = analysis._power_warning(small)
    assert warn is not None and warn["underpowered"] is True
    assert warn["max_sentences_per_treebank"] == 30

    assert analysis._power_warning(large) is None
    assert analysis._power_warning(pd.DataFrame()) is None


def test_f3_verdict_echoes_the_preregistered_thresholds_verbatim():
    """verdict.json must carry the thresholds it was judged against.

    Pre-registration is only enforceable if the decision rule travels with the
    result. This checks the echo is the config's own values, not a hardcoded
    copy that could drift out of sync.
    """
    import yaml
    cfg = yaml.safe_load((RUN_DIR / "config.yaml").read_text(encoding="utf-8"))

    h1 = pd.DataFrame([{
        "treebank_id": "tb", "baseline_kind": "B1_uniform", "metric": m,
        "delta": -0.5, "ci_lo": -0.6, "ci_hi": -0.4, "p_value": 1e-9,
        "n_sentences": 5000, "n_arcs": 1e4, "mean_real": 1.0,
        "mean_baseline": 2.0, "supports_h1": True,
    } for m in ("word_dl", "head_dl")])

    verdict = analysis.build_verdict(
        h1, pd.DataFrame(), pd.DataFrame(), cfg, {"run_date": "2026-09-19"}
    )

    assert verdict["pre_registered_thresholds"] == cfg["thresholds"], (
        "the verdict's thresholds diverged from config.yaml"
    )
    assert verdict["hypotheses"]["H1"]["verdict"] == "SUPPORTED"
    # H1 is scored on B1 only -- a documented judgment call worth pinning.
    assert verdict["hypotheses"]["H1"]["scored_on_baseline"] == "B1_uniform"


def test_f4_h1_requires_both_metrics_to_support():
    """A treebank counts for H1 only if BOTH metrics show the effect.

    If word_dl supports but head_dl does not, that treebank must NOT be scored
    as supporting -- otherwise H1 stops being the sanity check it claims to be.
    """
    h1 = pd.DataFrame([
        {"treebank_id": "tb", "baseline_kind": "B1_uniform", "metric": "word_dl",
         "delta": -0.5, "ci_lo": -0.6, "ci_hi": -0.4, "p_value": 1e-9,
         "n_sentences": 5000, "n_arcs": 1e4, "mean_real": 1.0,
         "mean_baseline": 2.0, "supports_h1": True},
        {"treebank_id": "tb", "baseline_kind": "B1_uniform", "metric": "head_dl",
         "delta": 0.1, "ci_lo": -0.1, "ci_hi": 0.3, "p_value": 0.4,
         "n_sentences": 5000, "n_arcs": 1e4, "mean_real": 1.0,
         "mean_baseline": 1.0, "supports_h1": False},
    ])
    verdict = analysis.build_verdict(
        h1, pd.DataFrame(), pd.DataFrame(), MIN_CFG, {"run_date": "x"}
    )
    assert verdict["hypotheses"]["H1"]["n_supporting"] == 0, (
        "a treebank supported H1 on only one of the two required metrics"
    )


# ===========================================================================
# TEST G -- linearizer invariants, exhaustively
# ===========================================================================

@pytest.mark.parametrize("n", [4, 5, 6])
def test_g_baselines_preserve_the_tree_over_all_small_trees(n):
    """Over EVERY projective tree on n nodes, both baselines preserve structure.

    test_main.py::test_4 checks 15 fixture sentences. Exhaustive enumeration is
    what proves the linearizer cannot rewire a tree on some shape the fixture
    happens not to contain -- and tree-preservation is the assumption the entire
    matched-pair design rests on.
    """
    checked = 0
    for s in all_valid_trees(n):
        if not is_projective(s):
            continue
        sig_arcs = lin.arc_signature(s)
        sig_arity = lin.arity_signature(s)
        sig_sides = lin.side_signature(s)

        for kind in ("B1_uniform", "B2_arity_preserving"):
            for sample in lin.sample_baselines(s, kind, n_samples=3, master_seed=5):
                assert sample.n_tokens == s.n_tokens
                assert lin.check_projective(sample), (
                    f"{kind} produced a non-projective order for {s.heads.tolist()}"
                )
                assert lin.arc_signature(sample) == sig_arcs, (
                    f"{kind} changed the arc multiset for {s.heads.tolist()}"
                )
                assert lin.arity_signature(sample) == sig_arity
                if kind == "B2_arity_preserving":
                    assert lin.side_signature(sample) == sig_sides, (
                        f"B2 changed left/right counts for {s.heads.tolist()}"
                    )
        checked += 1
    assert checked > 0, f"no projective trees enumerated for n={n}"


def test_g2_matched_pairs_share_every_structural_covariate():
    """A sentence and its relinearization must agree on arity, depth and length.

    These are H3's control covariates. If relinearization perturbed them, the
    logistic model could separate real from baseline using tree shape rather
    than word order, and H3's whole claim would be an artifact.
    """
    for s in itertools.islice(
        (t for t in all_valid_trees(6) if is_projective(t)), 40
    ):
        for kind in ("B1_uniform", "B2_arity_preserving"):
            for sample in lin.sample_baselines(s, kind, n_samples=2, master_seed=13):
                assert sample.n_tokens == s.n_tokens
                assert mean_arity(sample) == pytest.approx(mean_arity(s))
                assert tree_depth(sample) == tree_depth(s), (
                    f"{kind} changed tree depth for {s.heads.tolist()}"
                )


def test_g3_unknown_baseline_kind_raises():
    """A typo'd baseline name must raise, not silently produce no control."""
    s = mk([-1, 0, 1])
    with pytest.raises(ValueError, match="baseline"):
        lin.sample_baselines(s, "B3_nonexistent", n_samples=1, master_seed=1)


# ===========================================================================
# TEST H -- packaging and environment hygiene
# ===========================================================================

def test_h_no_absolute_paths_are_hardcoded_in_source():
    """No source file may hardcode a machine-specific absolute path.

    The run directory must stay relocatable; a baked-in C:\\Users\\... or /home/
    path makes the experiment unreproducible on any other machine.
    """
    offenders = []
    for py in sorted((RUN_DIR / "src").glob("*.py")) + [RUN_DIR / "main.py"]:
        text = py.read_text(encoding="utf-8")
        for marker in ("C:\\Users", "C:/Users", "/home/", "/Users/"):
            if marker in text:
                offenders.append(f"{py.name}: {marker}")
    assert not offenders, f"hardcoded absolute paths: {offenders}"


def test_h2_every_imported_third_party_package_is_declared():
    """Anything the pipeline imports must appear in requirements.txt.

    A missing pin means a fresh `pip install -r requirements.txt` produces an
    environment where the run dies on import -- the classic irreproducible run.
    """
    req_text = (RUN_DIR / "requirements.txt").read_text(encoding="utf-8").lower()
    declared = {
        line.split("==")[0].split(">=")[0].strip()
        for line in req_text.splitlines()
        if line.strip() and not line.startswith("#")
    }

    # import name -> distribution name where they differ
    needed = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "statsmodels": "statsmodels",
        "matplotlib": "matplotlib",
        "yaml": "pyyaml",
        "requests": "requests",
        "conllu": "conllu",
        "pytest": "pytest",
    }

    sources = list((RUN_DIR / "src").glob("*.py")) + [
        RUN_DIR / "main.py", Path(__file__), RUN_DIR / "tests" / "test_main.py"
    ]
    blob = "\n".join(p.read_text(encoding="utf-8") for p in sources if p.exists())

    missing = []
    for import_name, dist in needed.items():
        if f"import {import_name}" in blob and dist not in declared:
            missing.append(f"{import_name} (needs '{dist}')")
    assert not missing, f"imported but not in requirements.txt: {missing}"


def test_h3_gitignore_excludes_the_venv_and_caches():
    """The venv and bytecode caches must never be committable.

    Mandated by the run-isolation rules; a committed .venv is thousands of files.
    """
    ignored = {
        line.strip()
        for line in (RUN_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for required in (".venv/", "__pycache__/", "*.pyc"):
        assert required in ignored, f"{required} missing from .gitignore"


def test_h4_venv_lives_inside_the_run_directory():
    """Isolation rule: the virtualenv belongs at <run-dir>/.venv, nowhere else."""
    venv = RUN_DIR / ".venv"
    if not venv.exists():
        pytest.skip("no .venv in this checkout")
    assert venv.is_dir()
    assert (venv / "Scripts").exists() or (venv / "bin").exists(), (
        "'.venv' exists but holds no interpreter"
    )
