# Dependency Length Minimization: Words or Intervening Heads?

A laptop-scale replication attempt of Yadav & Husain's claim that the standard
word-based operationalization of dependency length is the wrong measure.

**Status:** full run complete, plus a controlled follow-up experiment.
26 treebanks, 430,113 sentences, 20 linearizations per sentence, 5,000
bootstrap resamples, seed 20260919. H1 SUPPORTED (26/26), H2 CONTRADICTED on
the primary statistic (12/52) and SUPPORTED on the magnitude co-primary
(44/52), H3 CONTRADICTED (2/52).

The substantive result is not any of those verdicts. It is that **the direction
of an H2-style comparison is controlled by how hard the baseline is**: rank and
magnitude effect measures disagree in 26/26 treebanks against the weak B1
baseline and in 6/26 against the strong B2 one. A λ-sweep over baselines of
tunable difficulty reproduces the entire effect by varying difficulty alone
(divergence rises on 9/9 steps in all 26 treebanks, corr = −0.96), so the
mechanism is demonstrated rather than inferred. Two of the three H2 statistics also turned
out to be algebraically the same measure, and one of the sweep's own
predictions was mis-specified. All of that is recorded in place rather than
tidied away.

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
wrong thing. Because both measures come off the same dependency trees, the
claim is cheap to test and directly falsifiable.

---

## Hypotheses

All three directions, and the decision rules that score them, were written into
`config.yaml` **before any results were generated**. The pre-registration is mechanically
enforced: `verdict.json` echoes the thresholds back, and a test asserts they
match `config.yaml`, so a post-hoc edit fails the suite. That guard caught the
H3 threshold change described below, which is why the retirement is recorded
rather than silent.

| ID | Hypothesis | Statistic | Predicted |
|----|------------|-----------|-----------|
| **H1** | Real sentences minimize dependency length vs. random projective linearization, on both metrics | Paired Cliff's δ (real vs. baseline) | δ < 0 on both |
| **H2** | The head-based metric shows a *larger* real-vs-random effect than the word-based metric | Δ\|δ\| = \|δ_head\| − \|δ_word\|, paired bootstrap CI; **co-primaries:** ΔP(advantage) and Δ median relative reduction | all three > 0 |
| **H3** | Word-based DL is *derivative* of head-based DL | β(word_DL) in a conditional logit over matched pairs, with nested LR test and within-pair ΔAUC | β(word_DL) → 0, β(head_DL) significant |

The H3 statistic shown is the corrected one. As pre-registered it was "≥50%
shrinkage in \|β(word_DL)\|"; that criterion was retired as invalid before
scoring — see "Correction 2" below.

**H1 is a sanity check, not a test of the paper.** If DLM fails to replicate at
all, the pipeline is broken, not the theory. H2 is the paper's comparative
claim. H3 is its real hypothesis.

**Decision rule (pre-registered).** A hypothesis is SUPPORTED if ≥70% of
qualifying *conditions* show the predicted effect with a bootstrap 95% CI
excluding zero; CONTRADICTED if <30% do; EQUIVOCAL otherwise. **EQUIVOCAL is a
legitimate result of this run, not a failure to produce one.**

A "condition" is a treebank for H1, which is scored on B1 only, and a
treebank × baseline cell for H2 and H3, which are scored on both baselines.
Hence 26 conditions for H1 and 52 for H2/H3 on the reported run. The rule was
written as "treebanks" and applied to cells; the wording is corrected here to
match the implementation, with no change to the arithmetic of any verdict.

### The H2 co-primaries

Cliff's δ counts only the *sign* of each within-pair difference, so every tie is
scored as non-support. head_DL is the coarser metric and ties far more often
(roughly 20× the word_DL rate under B1), which was believed to penalize it
structurally regardless of how well it tracks word order. Validation finding 1 established this on the
`--fast` run *after* seeing the results, so it could not be scored.

Two co-primary statistics were therefore pre-registered in `config.yaml` before
the full run was executed, chosen to fail in **different** ways so that their
agreement is informative:

- **ΔP(advantage)** — P(real shorter) with ties split evenly, head minus word.
  Intended as a tie-immune rank measure.
- **Δ median relative reduction** — median per-sentence `(base − real) / base`,
  head minus word. A ratio of *summed* DL was considered and rejected: it is an
  effect on the corpus total and is dominated by long sentences, so the median
  gives each sentence one vote.

δ remains the primary and its verdict is reported unchanged. Each co-primary is
scored by the same 70/30 rule on its own bootstrap CI, reported separately and
never pooled.

**One of the two co-primaries was a mistake, and the full run proved it.**
P(advantage) is not independent of Cliff's δ — it is that statistic rescaled:

```
δ      = (pos − neg) / n        with pos + neg + ties = n
P(adv) = (neg + ties/2) / n
       = (1 − δ) / 2            exactly, ties included
```

Measured across all 52 cells, `corr(ΔP(adv), Δδ) = 1.0` at a ratio of exactly
0.5. The premise was wrong: δ *already* splits ties, because a tie contributes
to neither `pos` nor `neg`, which is arithmetically the same as contributing
half to each. Splitting ties explicitly changes nothing.

So the pre-registered design did not deliver a tie-structure contrast. What it
delivered is a **rank-versus-magnitude** contrast, which turns out to be the
more interesting one. A regression test now pins the identity so the two are
never again reported as independent evidence.

### The H3 design, and two corrections to it

The obvious reading of "control for head_DL" is not computable as usually
stated: "realness" is not a variable with variance inside a treebank, because
every UD sentence is real. There is nothing to correlate against.

The fix, which preserves the scientific intent, makes realness a **binary
outcome over matched pairs** — each real sentence against a random
relinearization *of its own tree*:

```
P(real | pair) ~ z(word_DL) + z(head_DL)        [conditional logit]
```

Because both members of a pair share the identical tree, arity and depth are
constant within the pair. That is what neutralizes the arity confound: the
head-count metric is mechanically sensitive to how many heads a span contains,
which is a property of tree shape rather than of word order, and pairing holds
tree shape fixed by construction.

**Correction 1 — conditional, not plain, logit.** An earlier version fitted an
ordinary logistic regression with `n_tokens`, `mean_arity` and `tree_depth` as
covariates. Those are constant within a pair, so they controlled nothing; the
pairing, not their presence in the equation, is what neutralizes the confound.
The stratum is the pair, so the model is a conditional logit and the
pair-constant covariates drop out by construction.

**Correction 2 — H3 is no longer scored on shrinkage.** The pre-registered rule
asked for ≥50% shrinkage in |β(word_DL)| when head_DL entered. Logistic
coefficients are **not collapsible**: adding a predictor that carries real
signal rescales the remaining coefficients even when nothing is confounded, so
a between-model β comparison measures rescaling, not derivativeness. Simulation
shows the statistic fails in *both* directions. It reports up to **+73%**
shrinkage when word_DL and head_DL are independent causes (derivativeness
false), and **−81%** "suppression" when word_DL's true effect is exactly zero
(derivativeness true). The negative-shrinkage cells reported by the earlier run
were this artifact, not evidence of suppression.

H3 is now scored on whether word_DL retains an independent effect once head_DL
is in the model, with a nested LR test under FDR control and within-pair ΔAUC
as a scale-free companion. `h3_shrinkage_min` remains in `config.yaml` as an
explicit `null` with the reasoning attached, and a test asserts it stays
retired. A pre-registered threshold should be visibly withdrawn, not quietly
deleted.

---

## Results

Run configuration: full run, 20 baseline samples per sentence, 5,000 bootstrap
resamples, seed 20260919. 26 of 29 configured treebanks were analyzed,
totalling 430,113 sentences. (An earlier run reached only 17 treebanks: two
large ones 404'd because UD ships their training data as multiple shards, and
the sentence-count floor was set at an unjustified 5,000. Both are fixed; see
`config.yaml`.)

### H1 — SUPPORTED (26/26)

DLM replicates decisively. Real orders are shorter than random projective
relinearizations of the same trees on both metrics, in every treebank, with all
CIs excluding zero.

| Treebank | Family | δ (word) | δ (head) |
|----------|--------|----------|----------|
| ja_gsd | Japonic | −0.971 | −0.944 |
| fr_gsd | Indo-European | −0.900 | −0.873 |
| id_gsd | Austronesian | −0.876 | −0.819 |
| ko_kaist | Koreanic | −0.876 | −0.622 |
| es_ancora | Indo-European | −0.869 | −0.827 |
| ar_padt | Afro-Asiatic | −0.829 | −0.775 |
| he_htb | Afro-Asiatic | −0.806 | −0.774 |
| el_gdt | Indo-European | −0.801 | −0.776 |
| ug_udt | Turkic | −0.797 | −0.455 |
| pt_bosque | Indo-European | −0.793 | −0.786 |
| eu_bdt | isolate | −0.769 | −0.594 |
| it_isdt | Indo-European | −0.762 | −0.730 |
| tr_imst | Turkic | −0.746 | −0.462 |
| hi_hdtb | Indo-European | −0.698 | −0.569 |
| ru_syntagrus | Indo-European | −0.662 | −0.534 |
| en_ewt | Indo-European | −0.638 | −0.579 |
| lv_lvtb | Indo-European | −0.638 | −0.495 |
| pl_pdb | Indo-European | −0.637 | −0.561 |
| fi_tdt | Uralic | −0.593 | −0.418 |
| et_edt | Uralic | −0.564 | −0.373 |
| zh_gsd | Sino-Tibetan | −0.509 | −0.452 |
| la_ittb | Indo-European | −0.498 | −0.308 |
| ga_idt | Indo-European | −0.498 | −0.425 |
| cs_pdt | Indo-European | −0.495 | −0.421 |
| fa_seraji | Indo-European | −0.479 | −0.284 |
| de_gsd | Indo-European | −0.412 | −0.290 |

Japanese is strongest, as its rigid head-final order predicts. Note that δ(word)
exceeds δ(head) in all 26 treebanks — the rank comparison behind H2.

### H2 — the statistics disagree under B1, and agree under B2

| Statistic | Kind | Verdict | Cells |
|-----------|------|---------|-------|
| Δ\|δ\| (primary) | rank | **CONTRADICTED** | 12/52 |
| ΔP(advantage) (co-primary) | rank — *identical to δ* | EQUIVOCAL | 17/52 |
| Δ median rel. reduction (co-primary) | magnitude | **SUPPORTED** | 44/52 |

Under the **B1 uniform** baseline the split is total: the magnitude measure
favours head_DL in 26/26 treebanks, the rank measures favour word_DL in 26/26.
Every single cell points both ways at once.

**That split does not survive the second baseline.** Under **B2
arity-preserving**, rank and magnitude point in opposite directions in only
**6 of 26** treebanks. `verify_h2_split.py` reproduces this from the committed
results.

| Baseline | rank favours head | magnitude favours head | cells pointing opposite |
|----------|------------------|------------------------|------------------------|
| B1_uniform | 0/26 | 26/26 | **26/26** |
| B2_arity_preserving | 12/26 | 18/26 | **6/26** |

The reason is a **ceiling effect, not a property of the metrics**. B1 is a weak
baseline: both metrics beat it in most pairs (median P(real shorter) 0.861 for
word_DL, 0.783 for head_DL), which pushes any rank statistic toward its ceiling
where it can no longer separate the two. The magnitude measure still has room to
move — median per-sentence reduction is 0.190 for word_DL against 0.325 for
head_DL. Under B2 ranks fall to ~0.55, both measures regain resolution, and they
largely agree.

So the defensible claim is narrower than "rank and magnitude disagree about
word_DL versus head_DL". It is:

> Against a weak baseline, a rank statistic saturates and a magnitude statistic
> does not, so the two can be made to disagree completely. Against a strong
> baseline they agree.

That is a claim about **choosing baselines and effect measures in DLM work**,
and it generalizes past this particular pair of metrics. It is not the claim the
pre-registered H2 was testing. It is also no longer correlational: see "The
baseline-difficulty experiment" below, which reproduces the whole effect by
varying difficulty alone.

On the paper's own question, the honest summary is: **head_DL shows the larger
effect on the magnitude reading under both baselines** (26/26 under B1, 18/26
under B2), and the rank reading is unreliable under B1 for the reason above.
That is qualified support for Yadav & Husain's comparative claim, arrived at by
a route the pre-registration did not anticipate, and it should not be reported
as a clean confirmation.

### H3 — CONTRADICTED (2/52)

Conditional logit over matched pairs, across all 26 treebanks:

- β(word_DL) is significant at α = 0.05 in **50 of 52** cells once head_DL is
  controlled. Derivativeness predicts it should vanish. It does not.
- Median β(word_DL) in the full model is **−8.74**, against median β(head_DL)
  of **−0.17**.
- Median within-pair ΔAUC from adding word_DL to head_DL is **+0.036**: word_DL
  carries incremental discriminative information that head_DL does not.

**β(head_DL) has the wrong sign in 25 of 52 cells, spanning 18 of the 26
languages.** In those cells *more* intervening heads predicts the real order —
the opposite of the paper's direction. This too is baseline-dependent (18/26
cells under B1 against 7/26 under B2), which points at the same ceiling
mechanism rather than at a linguistic fact about those languages.

**Read ΔAUC and the LR test, not the individual βs.** Within-pair, the two
metrics' differences correlate 0.95–0.98 (VIF up to 41). The coefficients are
jointly identified but individually fragile.

---

## The baseline-difficulty experiment

The H2 result above says that rank and magnitude effect measures disagree
completely against B1 and barely at all against B2, and blames a ceiling
effect. On its own that is **correlational**: B1 and B2 differ in more than
difficulty, since B2 also preserves branching direction. A reviewer could
reasonably answer "that might be branching, not difficulty".

`experiment_ceiling.py` settles it with a baseline of tunable difficulty. At
every node the real local arrangement is kept with probability λ and shuffled
otherwise, so **λ=0 is exactly B1_uniform and λ=1 is exactly the real
sentence**, with the construction held fixed throughout. Only difficulty
varies. Both endpoints are asserted by tests; λ=1 is an exact identity on the
projective sentences the analysis is conditioned on.

Sweeping λ over 10 values on **all 26 analyzed treebanks**:

| λ | P(advantage), best metric | rank separation | magnitude separation | divergence |
|---|--------------------------|-----------------|---------------------|------------|
| 0.0 | 0.817 | −0.114 | +0.137 | 0.251 |
| 0.2 | 0.807 | −0.098 | +0.127 | 0.226 |
| 0.4 | 0.791 | −0.081 | +0.114 | 0.196 |
| 0.6 | 0.765 | −0.059 | +0.089 | 0.148 |
| 0.8 | 0.714 | −0.025 | +0.041 | 0.066 |
| 0.9 | 0.664 | −0.002 | +0.013 | 0.015 |

**The divergence rises on 9 of 9 steps as the baseline gets easier**, with
corr(λ, divergence) = −0.96. The B1-vs-B2 contrast was not a quirk of B2's
construction: difficulty alone reproduces the whole effect.

There are no exceptions across the sample:

| | holds in |
|---|---|
| corr(λ, rank separation) > 0 | **26/26** (median +0.96, range +0.49 to +1.00) |
| corr(λ, magnitude separation) < 0 | **26/26** (median −0.94, range −0.84 to −0.99) |
| divergence monotone in λ (corr < −0.8) | **26/26** |

Two of the three stated predictions survived, and the third was mis-specified
in a way worth recording.

**What held.** Rank separation moves monotonically toward zero as the baseline
hardens, in every treebank. Magnitude separation moves the opposite way.
Within a treebank, rank separation tracks the ceiling coordinate at corr =
**−0.90**: the closer the best metric gets to P(advantage) = 1, the less the
rank statistic can distinguish the two metrics.

**What did not.** The prediction was written as "|rank_sep| shrinks", and by
that test it fails (0.106 easy vs 0.059 hard). Two things were wrong with it.
`rank_sep` is *signed*, so the absolute value hides the actual pattern — the
rank statistic says word_DL wins by a lot against an easy baseline and that
advantage decays toward zero as the baseline hardens. And pooling across
treebanks destroys the ceiling correlation (−0.16 pooled against −0.90
within-treebank), because baseline difficulty is comparable within a language
but confounded with branching profile across languages. The script prints both
the failed original test and the corrected one.

**What this licenses, and what it does not.** The mechanism is now
demonstrated rather than suggested: baseline difficulty causes the divergence,
and the effect is monotone and present in every language tested. But the
finding is slightly different from the original "ceiling" wording. Difficulty
does not saturate one measure and leave the other alone — it drives *both*, in
opposite directions. A DLM result computed against a single baseline with a
single effect measure is therefore not merely imprecise; its direction is a
function of a design choice that is rarely reported.

---

## Validation findings

These are the scientific-review results, recorded here because they qualify how
the verdicts above should be read.

**1. The H2 verdict depends on whether you measure rank or magnitude. This is
the most important caveat in this document.** Superseded in part by the full
run: the tie-structure diagnosis below was *half right*, and the half that was
wrong is recorded in "The H2 co-primaries" above.

Cliff's δ counts the *sign* of each within-pair difference and ignores its
magnitude. head_DL is the coarser metric and ties far more often than word_DL
(median 4.3% vs 0.2% under B1 on the full run).

The `--fast` run inferred from this that δ *structurally penalizes* the coarser
metric, and proposed a tie-splitting statistic as the fix. **The full run showed
that inference was wrong**: P(advantage) = (1 − δ)/2 identically, so δ already
splits ties and tie rates cannot be what drives the reversal.

What actually drives it is rank versus magnitude. Under B1, head_DL wins on
median per-sentence relative reduction in 26/26 treebanks while losing on δ in
26/26 — it separates real from random in fewer pairs, but by roughly 1.7× the
proportional margin when it does. Under B2 that opposition mostly disappears
(6/26), which is why the split is reported as a baseline artifact rather than a
property of the metrics.

**H2 as pre-registered is contradicted on the primary statistic and supported on
the magnitude co-primary.** The honest reading is that "larger effect" was never
a single well-defined quantity, and the two readings of it disagree
systematically rather than noisily.

Numbers in this finding are reproduced by `verify_finding1.py`, which rebuilds
the comparison from the pipeline's own loader, linearizer and metrics. Its δ
column matches `per_treebank.csv` to four decimals.

**2. H1 is scored on B1 only, and this is correct.** B2 preserves each node's
left/right dependent counts, so on short sentences it reproduces the real word
order *exactly* about 21% of the time. Those pairs have a DL difference of
precisely zero by construction, pinning δ near zero whether or not DLM holds.
Scoring H1 on B2 would report a broken pipeline whenever the control was merely
strict. B2 remains valid for H2 and H3, where both conditions are affected
symmetrically. This is a sound judgment call, documented in `verdict.json`.

**3. H3's matched-pair design is sound; its original model and statistic were
not. Both have been replaced; this finding was wrong in the earlier
write-up.** The pairing genuinely does neutralize the arity confound by
construction. But the analysis then fitted a plain logit with pair-constant
covariates that controlled nothing, and scored the hypothesis on a between-model
β comparison that is invalid for logistic coefficients. See "The H3 design, and
two corrections to it" above; the verdict moves from EQUIVOCAL to CONTRADICTED.
The claim that "the nested-model comparison, the LR test, and the within-treebank
z-scoring are all appropriate" was an error of review, and is retracted here
rather than silently edited away.

**4. Non-projective sentences are excluded from the headline analysis and
counted separately** (2.5% of English, 8.4% of Arabic). This is correct, since a
projective baseline is not comparable to a non-projective original, but it
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
git clone <this repository>
cd 2026-09-19__yadav-husain-dependency-length

python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

## How to run

```bash
# Full run: 30 configured treebanks (17 met the 5,000-sentence minimum in the
# reported run), 20 baseline samples, 5000 bootstrap resamples. ~34 min.
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
    fig_a_forest_delta_<baseline>.png   δ per treebank, both metrics
    fig_b_h2_delta_difference.png       Δ|δ| forest with zero line
    fig_c_h3_beta_shrinkage.png         β(word_DL) with vs. without head_DL
```

A run below 500 sentences per treebank emits a `power_advisory` block in
`verdict.json`. The 30-sentence fixture run reports CONTRADICTED on all three
hypotheses — that is low power, not evidence, and the advisory exists so it is
not misread as a refutation of DLM.

## Limitations

- **The headline H2 disagreement is a baseline artifact.** Rank and magnitude
  point opposite ways in 17/17 treebanks under B1 but only 3/17 under B2. B1 is
  weak enough to saturate rank statistics, which manufactures the split. Any
  DLM result resting on a single baseline and a single effect measure should be
  treated as provisional for this reason.
- **B1 is too weak to be the headline baseline.** Both metrics beat it in
  80–88% of pairs. It remains the H1 sanity check, where that weakness is
  harmless, but comparative claims should be read off B2.
- **One co-primary was redundant.** P(advantage) is Cliff's δ rescaled, so the
  pre-registered design bought one contrast, not two. Pre-registering a
  statistic does not make it independent of the statistic it was meant to
  check — that has to be verified algebraically first.
- **word_DL and head_DL are near-collinear within pairs** (r = 0.95–0.98, VIF up
  to 41). Individual H3 coefficients are unstable at this level; the LR test and
  ΔAUC are the interpretable quantities. A genuinely null LR result would be
  hard to distinguish from insufficient power to separate the two metrics.
- **The sample still skews Indo-European: 15 of 26.** Raising the treebank
  count from 17 to 26 did not fix this, because most of the newly admitted
  treebanks are also Indo-European. Non-IE families are represented by one or
  two treebanks each, so any claim about typological generality rests on very
  few independent observations per family. Three configured treebanks remain
  unanalyzed: `am_att` is test-only upstream, `ta_ttb` (400 sentences) and
  `hu_szeged` (908) fall below the 1,500-sentence floor.
- **β(head_DL) carries the wrong sign in 18 of 26 languages under B1.** Whatever
  the head-count metric is measuring against a uniform baseline, it is not
  cleanly "more intervening heads is worse". This is concentrated under B1
  (18/26 against 7/26 under B2), so the difficulty mechanism established by the
  λ sweep is the first thing to rule out before reading it as linguistic.
- **λ interpolates toward the attested order, not toward a principled "harder
  baseline".** It establishes that difficulty drives the divergence; it does
  not by itself say which difficulty level a DLM study ought to use.
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
