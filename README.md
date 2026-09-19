# Dependency Length Minimization: Words or Intervening Heads?

A laptop-scale replication attempt of Yadav & Husain's claim that the standard
word-based operationalization of dependency length is the wrong measure.

**Status:** complete. H1 SUPPORTED, H2 CONTRADICTED, H3 EQUIVOCAL — on the
pre-registered decision rules, with one important caveat about the H2 statistic
documented in "Validation findings" below.

---

## Paper under test

> Yadav, H., & Husain, S. (2022). *A Reappraisal of Dependency Length
> Minimization as a Linguistic Universal.* **Open Mind: Discoveries in
> Cognitive Science**, MIT Press.

**The paper's claim.** Dependency length minimization (DLM) is one of the
most-cited quantitative universals in linguistics, and it is almost always
measured by counting *words* between a head and its dependent. Yadav & Husain
argue this is a proxy. What actually constrains processing, they claim, is the
number of *intervening syntactic heads*, and the familiar word-based DLM effect
is a derivative consequence of that plus tree properties such as node arity.

If they are right, a large corpus-linguistics literature has been measuring the
wrong thing — and because both measures come off the same dependency trees, the
claim is cheap to test and directly falsifiable.

---

## Hypotheses

All three directions, and the decision rules that score them, were written into
`config.yaml` **before any results were generated**. The pre-registration is
mechanically enforced, not merely asserted: `verdict.json` echoes the thresholds
back, and a test asserts they match `config.yaml`, so a post-hoc edit fails the
suite.

| ID | Hypothesis | Statistic | Predicted |
|----|------------|-----------|-----------|
| **H1** | Real sentences minimize dependency length vs. random projective linearization, on both metrics | Paired Cliff's δ (real vs. baseline) | δ < 0 on both |
| **H2** | The head-based metric shows a *larger* real-vs-random effect than the word-based metric | Δ\|δ\| = \|δ_head\| − \|δ_word\|, paired bootstrap CI | Δ\|δ\| > 0 |
| **H3** | Word-based DL is *derivative* of head-based DL | Shrinkage in \|β(word_DL)\| when head_DL enters a logistic model of realness | ≥50% shrinkage, β(head_DL) still significant |

**H1 is a sanity check, not a test of the paper.** If DLM fails to replicate at
all, the pipeline is broken — not the theory. H2 is the paper's comparative
claim. H3 is its real hypothesis.

**Decision rule (pre-registered).** A hypothesis is SUPPORTED if ≥70% of
qualifying treebanks show the predicted effect with a bootstrap 95% CI excluding
zero; CONTRADICTED if <30% do; EQUIVOCAL otherwise. **EQUIVOCAL is a legitimate
result of this run, not a failure to produce one.**

### The H3 design, and a correction to it

The obvious reading of "control for head_DL" is not computable as usually
stated: "realness" is not a variable with variance inside a treebank, because
every UD sentence is real. There is nothing to correlate against.

The fix, which preserves the scientific intent, makes realness a **binary
outcome over matched pairs** — each real sentence against a random
relinearization *of its own tree*:

```
P(real) ~ z(word_DL) + z(head_DL) + z(n_tokens) + z(mean_arity) + z(tree_depth)
```

Because both members of a pair share the identical tree, arity and depth are
constant within the pair. That is what neutralizes the arity confound: the
head-count metric is mechanically sensitive to how many heads a span contains,
which is a property of tree shape rather than of word order, and pairing holds
tree shape fixed by construction.

---

## Results

Run configuration: `--fast --offline`, 5 treebanks, 5 baseline samples per
sentence, 500 bootstrap resamples, seed 20260919. 37,801 sentences analyzed.

### H1 — SUPPORTED (5/5)

DLM replicates decisively. Real orders are shorter than random projective
relinearizations of the same trees on both metrics, in every treebank.

| Treebank | Family | δ (word) | δ (head) |
|----------|--------|----------|----------|
| ja_gsd | Japonic | −0.953 | −0.928 |
| ar_padt | Afro-Asiatic | −0.798 | −0.742 |
| tr_imst | Turkic | −0.689 | −0.403 |
| en_ewt | Indo-European | −0.592 | −0.540 |
| fi_tdt | Uralic | −0.523 | −0.365 |

Japanese is strongest, as its rigid head-final order predicts. All CIs exclude
zero.

### H2 — CONTRADICTED (2/10) — but see the caveat

Under the B1 uniform baseline, **all five treebanks show the opposite of the
prediction**: the word-based metric separates real from random order *better*
than the head-based one. The head metric wins in only 2 of 10 cells, both under
the stricter arity-preserving B2 baseline.

Median Δ|δ| = −0.044.

### H3 — EQUIVOCAL (3/10)

β(head_DL) is highly significant nearly everywhere, which is real support for
the paper's core intuition that intervening heads carry independent
information. But |β(word_DL)| shrinks only 19–27% under B1 — far short of the
pre-registered 50% bar — and two cells show *negative* shrinkage, meaning
word_DL becomes a *stronger* predictor once head_DL is controlled. That is
suppression, the opposite of derivativeness.

Word-based DL is not reducible to head-based DL in this sample.

---

## Validation findings

These are the scientific-review results, recorded here because they qualify how
the verdicts above should be read.

**1. The H2 verdict is an artifact of the statistic, and reverses under a
tie-immune measure. This is the most important caveat in this document.**

Cliff's δ counts the *sign* of each within-pair difference and ignores its
magnitude. The two metrics are not on comparable scales: over 1,200 English
sentences, word_DL takes 235 distinct values while head_DL takes only 68.
head_DL is coarse and heavily tied — its real-vs-baseline tie rate is 5.3% in
English and 13.2% in Turkish, against 0.5–1.3% for word_DL. Every tie is scored
as non-support, so the coarser metric is structurally penalized by δ regardless
of how strongly it tracks word order.

Recomputing the same comparison with a scale-free effect measure (relative
reduction in total DL, `(baseline − real) / baseline`, which is immune to ties)
**reverses the direction in every treebank tested**:

| Treebank | δ_word | δ_head | Rel. reduction (word) | Rel. reduction (head) |
|----------|--------|--------|----------------------|----------------------|
| en_ewt | −0.733 | −0.695 | 0.286 | **0.454** |
| ja_gsd | −0.970 | −0.947 | 0.428 | **0.653** |
| tr_imst | −0.708 | −0.432 | 0.189 | **0.280** |

By δ the word metric wins everywhere; by proportional reduction the head metric
wins everywhere, and by a wide margin. **H2 as pre-registered is contradicted;
the underlying comparative claim is not.** The honest reading is that this run
tested one specific operationalization of "larger effect" and found it wanting,
not that Yadav & Husain are wrong.

The pre-registered verdict is reported unchanged above, because retrofitting
the statistic after seeing results is exactly what pre-registration exists to
prevent. A follow-up run should pre-register the relative-reduction statistic as
a co-primary and report both.

**2. H1 is scored on B1 only, and this is correct.** B2 preserves each node's
left/right dependent counts, so on short sentences it reproduces the real word
order *exactly* about 21% of the time. Those pairs have a DL difference of
precisely zero by construction, pinning δ near zero whether or not DLM holds.
Scoring H1 on B2 would report a broken pipeline whenever the control was merely
strict. B2 remains valid for H2 and H3, where both conditions are affected
symmetrically. This is a sound judgment call, documented in `verdict.json`.

**3. H3 is correctly specified and the matched-pair design genuinely does the
work claimed for it.** Arity and depth are constant within pairs, so the arity
confound is neutralized by construction rather than by statistical adjustment.
The nested-model comparison, the LR test, and the within-treebank z-scoring are
all appropriate.

**4. Non-projective sentences are excluded from the headline analysis and
counted separately** (2.5% of English, 8.4% of Arabic). This is correct — a
projective baseline is not comparable to a non-projective original — but it
means the analysis is conditioned on projectivity, and non-projectivity is
plausibly where the two metrics diverge most. Flagged as future work.

**5. Metric implementation verified.** The vectorized head-counting fast path
was checked against a naive O(n²) reference on every valid tree for n ≤ 7, both
scopes, both endpoint settings — zero mismatches. Linearizer invariants (arc
multiset, arity, B2 side counts, projectivity) hold exhaustively for all
projective trees n ≤ 7. Cliff's δ sign convention is correct (δ<0 ⇒ real
shorter); a flip would have inverted every conclusion while the CIs still
looked healthy.

---

## Setup

Requires Python 3.10+. No GPU, no API keys, no network for the test suite.

```bash
cd "runs/2026-09-19__yadav-husain-dependency-length"

python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

## How to run

```bash
# Full run: ~30 treebanks, 20 baseline samples, 5000 bootstrap resamples.
# Downloads UD v2.14 treebanks on first run (~400 MB, cached in data/raw/).
python main.py

# Fast run: 5 treebanks, 5 samples, 500 resamples. ~4.5 min on cached data.
python main.py --fast --offline

# Fixture run: 30 bundled sentences, no network at all. ~2 s.
python main.py --fixture --offline

# Tests
python -m pytest tests/ -q        # 64 tests
```

Flags: `--fast`, `--offline` (cache only, never fetch), `--fixture` (bundled
mini corpus), `--stage`, `--seed`.

If UD cannot be reached and nothing is cached, the pipeline **exits non-zero
and writes nothing** rather than producing partial results.

## Expected output

```
results/
  verdict.json            machine-readable per-hypothesis verdict + thresholds
  per_treebank.csv        δ, CI, n_sentences, n_arcs per treebank × metric × baseline
  h2_comparison.csv       Δ|δ| with CI; supports / contradicts / equivocal counts
  h3_models.csv           β and SE per predictor, both nested models, LR-test p
  treebank_manifest.csv   what loaded, what was rejected, and why
  figures/
    fig1_delta_forest.png       δ per treebank, both metrics
    fig2_h2_delta_diff.png      Δ|δ| forest with zero line
    fig3_h3_beta_shrinkage.png  β(word_DL) with vs. without head_DL control
```

A run below 500 sentences per treebank emits a `power_advisory` block in
`verdict.json`. The 30-sentence fixture run reports CONTRADICTED on all three
hypotheses — that is low power, not evidence, and the advisory exists so it is
not misread as a refutation of DLM.

## Limitations

- **The H2 statistic penalizes the coarser metric.** See validation finding 1.
  This is the single largest threat to the reported H2 verdict.
- **Five treebanks is a small typological sample.** The reported run is
  `--fast`. The full 30-treebank run (`python main.py`) is the one to cite.
- **Correlational, not causal.** UD is observational. The finding is about what
  distinguishes attested orders from random ones, not about what speakers do.
- **Annotation conventions are load-bearing.** UD's treatment of function-word
  headedness directly determines what counts as an intervening head — the exact
  quantity under test. A treebank that attaches adpositions differently will
  yield a different head_DL for identical surface strings. Per-treebank
  reporting (never pooling) limits but does not eliminate this.
- **Conditioned on projectivity** (validation finding 4).
- **One scope definition is headline.** `head_count.scope` has two defensible
  readings; both are computed and `head_dl_alt` records the alternative, but
  only `global` is scored.

## Future work

1. Re-run with relative DL reduction pre-registered as a co-primary statistic
   alongside Cliff's δ, so H2 is not decided by tie structure.
2. Full 30-treebank run for genuine typological coverage.
3. Extend to non-projective sentences with a non-projectivity-matched baseline.
4. Sensitivity analysis over `head_count.scope` and `include_endpoints`.
5. The Hahn & Xu (PNAS 2022) counterfactual-grammar extension. `linearize.py`
   already exposes the `order_fn` interface it would need; a deprel→side
   parameter sampler is the only new component.

## Data and license

Universal Dependencies v2.14, CC BY-SA. Treebanks are downloaded per-corpus
from the UD GitHub organization and cached in `data/raw/` (gitignored). No data
is redistributed in this repository beyond the 30-sentence test fixture.
