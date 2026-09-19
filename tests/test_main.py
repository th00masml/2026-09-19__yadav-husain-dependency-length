"""Test suite for yadav-husain-dependency-length.

Every test runs with ZERO network access. The mandatory tests from the spec are
marked with their number; tests 1, 3, 4 and 8 are the ones that catch bugs which
would silently invalidate the science rather than crash.

Run:  python -m pytest tests/ -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

RUN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN_DIR))

from src import analysis, linearize as lin, metrics as mx, stats as st, ud_loader  # noqa: E402
from src.tree import Sentence, is_projective, mean_arity, tree_depth, validate  # noqa: E402

FIXTURE = RUN_DIR / "tests" / "fixtures" / "mini.conllu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fixture(min_tokens: int = 3, max_tokens: int = 100):
    """Parse the fixture into (sentences, stats)."""
    stats = ud_loader.LoadStats(treebank_id="fixture")
    text = FIXTURE.read_text(encoding="utf-8")
    sents = list(
        ud_loader.parse_conllu_text(text, "fixture", min_tokens, max_tokens, stats)
    )
    return sents, stats


def by_id(sents, sent_id):
    for s in sents:
        if s.sent_id == sent_id:
            return s
    raise KeyError(f"{sent_id} not in fixture (got {[s.sent_id for s in sents]})")


def random_tree(rng: np.random.Generator, n: int) -> Sentence:
    """Random projective-ish tree: each token attaches to a random earlier token."""
    heads = np.full(n, -1, dtype=np.int32)
    root = int(rng.integers(0, n))
    for i in range(n):
        if i == root:
            continue
        candidates = [j for j in range(n) if j != i]
        # attach toward the root to guarantee acyclicity
        heads[i] = root if not candidates else int(rng.choice(candidates))
    # Rebuild as a proper tree: attach node k to a node closer to root by BFS order
    order = [root] + [i for i in range(n) if i != root]
    rng.shuffle(order[1:])
    pos_in_order = {node: k for k, node in enumerate(order)}
    heads = np.full(n, -1, dtype=np.int32)
    for node in order[1:]:
        k = pos_in_order[node]
        parent = order[int(rng.integers(0, k))]
        heads[node] = parent
    return Sentence(
        tokens=tuple(f"w{i}" for i in range(n)),
        heads=heads,
        deprels=tuple("dep" for _ in range(n)),
        upos=tuple("X" for _ in range(n)),
        treebank_id="synthetic",
        sent_id=f"syn{n}",
    )


# ===========================================================================
# TEST 1 (mandatory): MWT ranges and empty nodes are handled correctly
# ===========================================================================

def test_1_mwt_and_empty_nodes_dropped():
    """Multiword ranges (1-2) and empty nodes (3.1) must not become tokens.

    This is the single most common bug in dependency-length code: leaving these
    lines in shifts every index by one or more and silently corrupts every
    length, without ever raising an error.
    """
    sents, _ = load_fixture()

    # s-es-1 "vi al perro del vecino" has TWO MWT ranges (2-3 and 5-6).
    # 9 CoNLL-U lines - 2 range lines = 7 real tokens.
    es1 = by_id(sents, "s-es-1")
    assert es1.n_tokens == 7, f"expected 7 real tokens, got {es1.n_tokens}: {es1.tokens}"
    assert es1.tokens == ("vi", "a", "el", "perro", "de", "el", "vecino")

    # Heads must be remapped to the new contiguous indices, not the old ids.
    # Original head ids (1-based): vi->0, a->4, el->4, perro->1, de->7, el->7, vecino->4
    # After re-indexing 0-based:    -1,     3,     3,        0,     6,     6,        3
    assert es1.heads.tolist() == [-1, 3, 3, 0, 6, 6, 3]

    # s-en-4 contains an empty node 5.1 -- 6 real tokens from 7 lines.
    en4 = by_id(sents, "s-en-4")
    assert en4.n_tokens == 6
    assert "5.1" not in en4.tokens
    assert en4.tokens == ("Mary", "ate", "rice", "and", "John", "beans")


def test_1b_malformed_sentences_are_rejected():
    """Cycles, multiple roots, and out-of-range heads must all be caught."""
    sents, stats = load_fixture()
    ids = {s.sent_id for s in sents}

    assert "s-bad-cycle" not in ids, "a cyclic 'tree' was accepted"
    assert "s-bad-noroot" not in ids, "a rootless 'tree' was accepted"
    assert "s-bad-two-roots" not in ids, "a two-root 'tree' was accepted"
    assert "s-bad-range" not in ids, "an out-of-range head was accepted"
    assert "s-bad-short" not in ids, "a 1-token sentence passed the min_tokens filter"

    # Each malformed sentence must be rejected for the RIGHT reason -- the
    # manifest's rejection breakdown is only meaningful if these are accurate.
    assert stats.rejections["cycle"] >= 1, "cycle detection never fired"
    assert stats.rejections["no_root"] >= 1
    assert stats.rejections["multiple_roots"] >= 1
    assert stats.rejections["head_out_of_range"] >= 1
    assert stats.rejections["too_short"] >= 1
    assert stats.n_kept < stats.n_seen

    # Comment-only blocks (the fixture's header banner) are not sentences and
    # must not inflate the seen count.
    assert stats.rejections["empty_sentence"] == 0


# ===========================================================================
# TEST 2 (mandatory): hand-computed DL values, under both scope settings
# ===========================================================================

@pytest.mark.parametrize(
    "sent_id, expected_word_dl, expected_head_dl",
    [
        # Values worked out by hand in the fixture's comments.
        ("s-en-1", 5, 0),    # "the dog chased the cat"
        ("s-en-2", 6, 1),    # "a very old man slept"
        ("s-es-1", 12, 0),   # "vi al perro del vecino"
        ("s-ja-1", 8, 1),    # "inu ga neko wo oikaketa"
    ],
)
def test_2_hand_computed_dependency_lengths(sent_id, expected_word_dl, expected_head_dl):
    """Both metrics must reproduce values computed by hand on paper."""
    sents, _ = load_fixture()
    s = by_id(sents, sent_id)

    word = mx.word_dl_arcs(s)
    assert int(word.sum()) == expected_word_dl, (
        f"{sent_id}: word_dl {int(word.sum())} != {expected_word_dl}; "
        f"per-arc {word.tolist()}, heads {s.heads.tolist()}"
    )

    head = mx.head_dl_arcs(s, scope="global", include_endpoints=False)
    assert int(head.sum()) == expected_head_dl, (
        f"{sent_id}: head_dl {int(head.sum())} != {expected_head_dl}; "
        f"per-arc {head.tolist()}"
    )


def test_2b_both_scopes_computable_and_ordered():
    """within_span can never exceed global: it is a strictly stronger condition."""
    sents, _ = load_fixture()
    for s in sents:
        g = mx.head_dl_arcs(s, scope="global", include_endpoints=False)
        w = mx.head_dl_arcs(s, scope="within_span", include_endpoints=False)
        assert np.all(w <= g), f"{s.sent_id}: within_span exceeded global"


def test_2c_include_endpoints_flag_increases_counts():
    """Including endpoints can only add to the count, never subtract."""
    sents, _ = load_fixture()
    for s in sents[:10]:
        excl = mx.head_dl_arcs(s, scope="global", include_endpoints=False)
        incl = mx.head_dl_arcs(s, scope="global", include_endpoints=True)
        assert np.all(incl >= excl), f"{s.sent_id}: endpoints flag decreased the count"


# ===========================================================================
# TEST 3 (mandatory): vectorized prefix-sum == naive O(n^2) reference
# ===========================================================================

def test_3_prefix_sum_matches_naive_reference():
    """The fast path must agree with an obviously-correct slow path.

    200 random trees, both scopes, both endpoint settings. If the prefix-sum
    arithmetic is off by one anywhere, this finds it.
    """
    rng = np.random.default_rng(12345)
    for trial in range(200):
        n = int(rng.integers(3, 18))
        s = random_tree(rng, n)
        for scope in ("global", "within_span"):
            for endpoints in (False, True):
                fast = mx.head_dl_arcs(s, scope=scope, include_endpoints=endpoints)
                slow = mx.head_dl_arcs_naive(s, scope=scope, include_endpoints=endpoints)
                assert np.array_equal(fast, slow), (
                    f"trial {trial} scope={scope} endpoints={endpoints}: "
                    f"fast {fast.tolist()} != naive {slow.tolist()}, heads {s.heads.tolist()}"
                )


def test_3b_prefix_sum_matches_naive_on_fixture():
    """Same check on the real fixture sentences."""
    sents, _ = load_fixture()
    for s in sents:
        for scope in ("global", "within_span"):
            fast = mx.head_dl_arcs(s, scope=scope)
            slow = mx.head_dl_arcs_naive(s, scope=scope)
            assert np.array_equal(fast, slow), f"{s.sent_id} scope={scope}"


# ===========================================================================
# TEST 4 (mandatory): the linearizers preserve the tree
# ===========================================================================

def test_4_linearizers_preserve_the_tree():
    """B1 and B2 must permute ORDER only -- never rewire the tree.

    If a linearizer quietly changed the arc multiset or a node's arity, the
    real-vs-baseline contrast would be comparing two different trees and the
    whole experiment would be meaningless.
    """
    sents, _ = load_fixture()
    projective = [s for s in sents if is_projective(s)]
    assert len(projective) >= 10, "fixture should contain plenty of projective sentences"

    for s in projective[:15]:
        sig_arcs = lin.arc_signature(s)
        sig_arity = lin.arity_signature(s)
        sig_sides = lin.side_signature(s)

        for kind in ("B1_uniform", "B2_arity_preserving"):
            for sample in lin.sample_baselines(s, kind, n_samples=5, master_seed=7):
                assert sample.n_tokens == s.n_tokens
                assert lin.arc_signature(sample) == sig_arcs, (
                    f"{s.sent_id}/{kind}: arc multiset changed"
                )
                assert lin.arity_signature(sample) == sig_arity, (
                    f"{s.sent_id}/{kind}: node arities changed"
                )
                if kind == "B2_arity_preserving":
                    assert lin.side_signature(sample) == sig_sides, (
                        f"{s.sent_id}: B2 failed to preserve left/right dependent counts"
                    )


def test_4b_b1_actually_varies_order():
    """A 'baseline' that always returns the original order tests nothing."""
    sents, _ = load_fixture()
    s = next(x for x in sents if is_projective(x) and x.n_tokens >= 7)
    samples = lin.sample_baselines(s, "B1_uniform", n_samples=20, master_seed=3)
    orders = {tuple(x.tokens) for x in samples}
    assert len(orders) > 1, "B1 produced a single order across 20 samples"


# ===========================================================================
# TEST 5 (mandatory): every baseline is projective
# ===========================================================================

def test_5_baselines_are_always_projective():
    """Projectivity is the control's defining property.

    A non-projective baseline would be a weaker (and incomparable) control,
    inflating every effect size.
    """
    sents, _ = load_fixture()
    projective = [s for s in sents if is_projective(s)]

    for s in projective[:15]:
        for kind in ("B1_uniform", "B2_arity_preserving"):
            for sample in lin.sample_baselines(s, kind, n_samples=5, master_seed=11):
                assert lin.check_projective(sample), (
                    f"{s.sent_id}/{kind}: produced a NON-projective baseline"
                )


def test_5b_projectivity_detector_is_correct():
    """The detector must flag the fixture's known non-projective sentence."""
    sents, _ = load_fixture()
    en3 = by_id(sents, "s-en-3")     # "a man came who smiled" -- crossing arcs
    assert not is_projective(en3), "failed to detect a known non-projective sentence"
    en1 = by_id(sents, "s-en-1")
    assert is_projective(en1), "flagged a plainly projective sentence as crossing"


# ===========================================================================
# TEST 6 (mandatory): seeding is deterministic
# ===========================================================================

def test_6_seeding_is_deterministic():
    """Same seed => bit-identical baselines; different seed => different."""
    sents, _ = load_fixture()
    s = next(x for x in sents if is_projective(x) and x.n_tokens >= 7)

    a = lin.sample_baselines(s, "B1_uniform", n_samples=10, master_seed=42)
    b = lin.sample_baselines(s, "B1_uniform", n_samples=10, master_seed=42)
    for x, y in zip(a, b):
        assert x.tokens == y.tokens
        assert np.array_equal(x.heads, y.heads)

    c = lin.sample_baselines(s, "B1_uniform", n_samples=10, master_seed=43)
    assert any(x.tokens != y.tokens for x, y in zip(a, c)), (
        "a different master seed produced identical baselines"
    )


def test_6b_rng_derivation_is_order_independent():
    """A sentence's RNG must depend on its identity, not on iteration order."""
    r1 = lin.derive_rng(5, "en_ewt", "s-99", 0).integers(0, 10**9, 5).tolist()
    r2 = lin.derive_rng(5, "en_ewt", "s-99", 0).integers(0, 10**9, 5).tolist()
    r3 = lin.derive_rng(5, "en_ewt", "s-98", 0).integers(0, 10**9, 5).tolist()
    assert r1 == r2
    assert r1 != r3


# ===========================================================================
# TEST 7 (mandatory): Cliff's delta against known cases
# ===========================================================================

def test_7_cliffs_delta_known_values():
    """delta = -1 when fully separated below, +1 above, 0 when identical."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 11.0, 12.0])
    assert st.cliffs_delta(x, y) == pytest.approx(-1.0)
    assert st.cliffs_delta(y, x) == pytest.approx(1.0)
    assert st.cliffs_delta(x, x) == pytest.approx(0.0)


def test_7b_paired_cliffs_delta_known_values():
    """The matched-pairs variant on hand-constructed cases."""
    real = np.array([1.0, 2.0, 3.0, 4.0])
    base = np.array([5.0, 6.0, 7.0, 8.0])
    assert st.paired_cliffs_delta(real, base) == pytest.approx(-1.0)
    assert st.paired_cliffs_delta(base, real) == pytest.approx(1.0)
    assert st.paired_cliffs_delta(real, real) == pytest.approx(0.0)

    # 3 of 4 pairs favor 'real' -> (1 - 3)/4 = -0.5
    mixed_base = np.array([5.0, 6.0, 7.0, 0.0])
    assert st.paired_cliffs_delta(real, mixed_base) == pytest.approx(-0.5)


def test_7c_paired_bootstrap_preserves_correlation():
    """H2's CI must use the SAME resampled indices for both metrics.

    With two perfectly correlated metrics the difference of deltas is exactly
    zero on every replicate, so a correctly paired bootstrap returns a
    degenerate (zero-width) CI. An independently-resampled bootstrap would
    return a wide one -- which is the bug this guards against.
    """
    rng = np.random.default_rng(0)
    n = 200
    real = rng.normal(10, 2, n)
    base = real + 3.0                       # baseline always larger
    out = st.paired_delta_difference_ci(
        real_word=real, base_word=base,
        real_head=real, base_head=base,     # identical metrics
        n_resamples=300, rng=rng,
    )
    assert out["delta_word"] == pytest.approx(out["delta_head"])
    assert out["delta_diff"] == pytest.approx(0.0, abs=1e-9)
    assert abs(out["ci_hi"] - out["ci_lo"]) < 1e-6, (
        "identical metrics produced a non-degenerate CI -- pairing is broken"
    )


def test_7d_benjamini_hochberg():
    """BH-FDR: monotone, bounded, and rejects the obvious cases."""
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.9])
    rej, adj = st.benjamini_hochberg(p, q=0.05)
    assert rej[0] and rej[1]
    assert not rej[4]
    assert np.all(adj[np.isfinite(adj)] <= 1.0)
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12), "adjusted p-values not monotone"


def test_7e_bca_ci_brackets_the_estimate():
    """A sane CI contains its own point estimate."""
    rng = np.random.default_rng(1)
    real = rng.normal(5, 1, 120)
    base = rng.normal(8, 1, 120)
    data = np.column_stack([real, base])

    def stat(rows):
        return st.paired_cliffs_delta(rows[:, 0], rows[:, 1])

    theta, lo, hi = st.jackknife_free_bca_ci(data, stat, n_resamples=400, rng=rng)
    assert lo <= theta <= hi
    assert theta < 0, "real values drawn strictly lower should give a negative delta"


# ===========================================================================
# TEST 8 (mandatory): full offline pipeline end-to-end
# ===========================================================================

def test_8_full_pipeline_offline_emits_valid_verdict(config_with_tmp_results):
    """`python main.py --fixture --fast --offline` must run with no network.

    This is the reproducibility guarantee: if UD is unreachable, the pipeline
    still executes end to end and produces a well-formed verdict rather than
    silently wrong numbers.

    Output is redirected into a tmp_path sandbox via --config. Previously this
    test wrote straight into results/, so every `pytest` run replaced the real
    five-treebank findings with 30-sentence fixture output.
    """
    cfg_path, out_dir = config_with_tmp_results

    result = subprocess.run(
        [sys.executable, "main.py", "--fixture", "--fast", "--offline",
         "--config", str(cfg_path)],
        cwd=str(RUN_DIR),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"pipeline exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n--- stderr ---\n{result.stderr[-4000:]}"
    )

    verdict_path = out_dir / "verdict.json"
    assert verdict_path.exists(), "verdict.json was not written"

    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert "hypotheses" in verdict
    assert "pre_registered_thresholds" in verdict

    # Thresholds in the output must match the pre-registered config values.
    thr = verdict["pre_registered_thresholds"]
    assert thr["support_fraction"] == 0.70
    assert thr["contradict_fraction"] == 0.30
    # h3_shrinkage_min was RETIRED after pre-registration: logistic
    # coefficients are not collapsible, so the shrinkage comparison measured
    # rescaling rather than derivativeness. It stays in config.yaml as an
    # explicit null with the reasoning attached, rather than being deleted, so
    # that the change is visible instead of silent. H3 is now scored on
    # h3_alpha via a conditional logit; see src/analysis.py::run_h3.
    assert thr["h3_shrinkage_min"] is None, (
        "h3_shrinkage_min must stay retired; reinstating the shrinkage "
        "criterion reintroduces a statistic that fails in both directions"
    )
    assert thr["h3_alpha"] == 0.05

    valid = {"SUPPORTED", "CONTRADICTED", "EQUIVOCAL", "INSUFFICIENT_DATA"}
    assert verdict["hypotheses"], "no hypotheses were evaluated"
    for hid, h in verdict["hypotheses"].items():
        assert h["verdict"] in valid, f"{hid} has an invalid verdict {h['verdict']!r}"
        assert "statement" in h and "predicted_direction" in h


def test_8b_results_files_written(config_with_tmp_results):
    """A pipeline run emits the CSV deliverables.

    Runs in its own sandbox rather than asserting on results/, so it neither
    depends on another test's side effects nor touches the real deliverables.
    """
    cfg_path, out_dir = config_with_tmp_results

    result = subprocess.run(
        [sys.executable, "main.py", "--fixture", "--fast", "--offline",
         "--config", str(cfg_path)],
        cwd=str(RUN_DIR),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert (out_dir / "treebank_manifest.csv").exists()
    assert (out_dir / "per_treebank.csv").exists()


# ===========================================================================
# Supporting tests: structural helpers and the analysis table
# ===========================================================================

def test_9_tree_structural_helpers():
    """depth, arity, and validation on a hand-checked tree."""
    sents, _ = load_fixture()
    s = by_id(sents, "s-en-1")        # the dog chased the cat
    # root=chased(2); children det(0)<-dog(1), dog(1)<-chased, the(3)<-cat(4)
    # depth: chased(0) -> dog(1)/cat(1) -> the(2)/the(2)  => 2
    assert tree_depth(s) == 2
    assert mean_arity(s) > 0
    assert validate(s.heads, s.n_tokens) is None


def test_10_with_order_roundtrip():
    """Relinearizing by the identity permutation is a no-op."""
    sents, _ = load_fixture()
    s = by_id(sents, "s-en-2")
    same = s.with_order(list(range(s.n_tokens)))
    assert same.tokens == s.tokens
    assert np.array_equal(same.heads, s.heads)

    with pytest.raises(ValueError):
        s.with_order([0, 0, 1, 2, 3])       # not a permutation


def test_11_analysis_table_shape_and_pairing():
    """Every real sentence gets exactly one matched row per baseline kind."""
    import yaml
    cfg = yaml.safe_load((RUN_DIR / "config.yaml").read_text(encoding="utf-8"))
    sents, _ = load_fixture()
    df, info = analysis.build_treebank_table(iter(sents), "fixture", cfg, n_samples=3)

    n_real = int((df["baseline_kind"] == "REAL").sum())
    assert n_real == info["n_analyzed"]
    for kind in cfg["baselines"]["kinds"]:
        n_base = int((df["baseline_kind"] == kind).sum())
        assert n_base == n_real, f"{kind}: {n_base} baseline rows for {n_real} real sentences"

    assert {"word_dl", "head_dl", "mean_arity", "tree_depth", "is_real"} <= set(df.columns)
    # Matched pairs share the tree, so arity and depth must be identical.
    real = df[df["baseline_kind"] == "REAL"].set_index("sent_id").sort_index()
    b1 = df[df["baseline_kind"] == "B1_uniform"].set_index("sent_id").sort_index()
    assert np.allclose(real["mean_arity"], b1["mean_arity"]), (
        "arity differs between a sentence and its own relinearization"
    )
    assert np.allclose(real["tree_depth"], b1["tree_depth"])


def test_12_nonprojective_excluded_from_headline():
    """Non-projective originals are counted but kept out of the paired analysis."""
    import yaml
    cfg = yaml.safe_load((RUN_DIR / "config.yaml").read_text(encoding="utf-8"))
    sents, _ = load_fixture()
    df, info = analysis.build_treebank_table(iter(sents), "fixture", cfg, n_samples=2)

    assert info["n_nonprojective"] >= 1, "fixture should contain a non-projective sentence"
    analyzed_ids = set(df[df["baseline_kind"] == "REAL"]["sent_id"])
    assert "s-en-3" not in analyzed_ids, "a non-projective sentence entered the headline analysis"
