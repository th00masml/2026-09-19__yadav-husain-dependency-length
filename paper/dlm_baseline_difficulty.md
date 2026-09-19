# The Direction of a Dependency-Length Comparison Depends on How Hard the Baseline Is

**Draft — not submitted.**

---

## Abstract

Dependency length minimization (DLM) is usually demonstrated by comparing
attested word orders against random relinearizations of the same trees. Such a
comparison requires two choices that are rarely reported together: which
baseline the attested order is measured against, and which effect measure
summarizes the comparison. We show that these choices are not independent of
the answer.

Testing Yadav and Husain's (2022) claim that word-based dependency length is a
proxy for the number of intervening syntactic heads, we find that rank-based and
magnitude-based effect measures disagree about which metric shows the larger
effect in 26 of 26 treebanks under a uniform random baseline, and in 6 of 26
under a stricter arity-preserving one. Because those two baselines differ in more
than difficulty, we construct a baseline of tunable difficulty that interpolates
between them and sweep it. The disagreement rises monotonically as the baseline
becomes easier to beat (9 of 9 steps; *r* = −0.96 between the interpolation
parameter and the divergence), and the trend holds in all 26 treebanks
individually. Baseline difficulty alone reproduces the entire effect.

We also report two statistics that generate their own results. A shrinkage
criterion comparing logistic coefficients across nested models fails in both
directions under simulation, because logistic coefficients are not collapsible.
A statistic pre-registered as a tie-immune alternative to Cliff's δ turns out to
equal (1 − δ)/2 identically. Neither failure is specific to this dataset.

The narrow conclusion is that Yadav and Husain's comparative claim holds on a
magnitude reading and fails on a rank reading. The general conclusion is that a
DLM result computed against one baseline with one effect measure is not merely
imprecise: its direction is a function of a design choice.

Code, data pipeline and all reported numbers:
`github.com/th00masml/yadav-husain-dependency-length`

---

## 1. Introduction

Dependency length minimization is among the most-cited quantitative universals
in linguistics. Across languages, attested word orders keep syntactic heads and
their dependents closer together than chance arrangements of the same trees
would.

The standard operationalization counts **words** between head and dependent.
Yadav and Husain (2022) argue that this is a proxy for something else: the
number of **intervening syntactic heads**, with the familiar word-based effect
following from that plus tree properties such as node arity. If they are right,
a large corpus literature has been measuring a correlate rather than the
quantity of interest.

The claim is unusually cheap to test. Both measures come off the same dependency
trees, so no new annotation is required, only a decision about what to compare
them against, and how.

This paper is about those decisions. We set out to test the claim and found that
we could not answer it without first answering a prior question: how much of a
DLM comparison is determined by the comparison's own design? The answer turns
out to be: enough to flip it.

### 1.1 What we claim

1. Rank-based and magnitude-based effect measures disagree systematically about
   the word-versus-head comparison, and the disagreement is controlled by
   baseline difficulty rather than by any property of the two metrics.
2. This is demonstrated by a controlled sweep, not inferred from contrasting two
   fixed baselines.
3. Two statistics that look reasonable, a nested-model shrinkage criterion and
   a tie-splitting rank measure, produce results without corresponding facts,
   and we characterize both failures.

### 1.2 What we do not claim

We do not resolve whether head-based dependency length is the better construct.
Our results are consistent with head-based DL showing a larger effect on a
magnitude reading under both baselines, and a smaller one on a rank reading, and
nothing in corpus data alone decides which reading is the right one.

---

## 2. Setup

### 2.1 Data

Universal Dependencies v2.14, training splits. Of 29 configured treebanks, 26
cleared a 1,500-sentence floor after validation, giving **430,113 sentences** and
**7,118,952 arcs**. Sentences are filtered to 3–100 tokens; multiword tokens and
empty nodes are dropped.

The sample is typologically uneven: 15 of 26 treebanks are Indo-European, with
Afro-Asiatic, Uralic and Turkic represented by two each and Japonic, Koreanic,
Sino-Tibetan, Austronesian and Basque by one. We return to this in §7.

### 2.2 Metrics

For a sentence with arcs *(h, d)*:

- **word_DL**: Σ |*h* − *d*|, the standard measure.
- **head_DL**: Σ (number of syntactic heads strictly between *h* and *d*).

Both are computed from the same trees. A vectorized head-counting path was
verified against a naive O(*n*²) reference on every valid tree for *n* ≤ 7, in
both scope settings, with zero mismatches.

### 2.3 Baselines

Each real sentence is paired with random projective relinearizations **of its own
tree**, so tree shape is held fixed by construction:

- **B1 (uniform)**: at each node, the head and its subtree blocks are shuffled
  uniformly. Every projective arrangement of the local family is equally likely.
- **B2 (arity-preserving)**: the shuffle preserves each node's left/right
  dependent counts, holding branching direction fixed.

Twenty relinearizations per sentence are averaged, so each real sentence has
exactly one matched baseline value per metric.

Non-projective sentences are excluded from the headline analysis and counted
separately, since a projective baseline is not comparable to a non-projective
original. All results are therefore conditioned on projectivity.

### 2.4 Effect measures

Three summaries of the same paired comparison:

| | definition | kind |
|---|---|---|
| Cliff's δ | (#real>base − #real<base)/*n* | rank |
| P(advantage) | (#real<base + ½·#ties)/*n* | rank |
| median relative reduction | median of (base − real)/base | magnitude |

The comparison of interest is the **contrast** between metrics: Δ\|δ\| =
\|δ_head\| − \|δ_word\|, and analogously for the others. Positive favours
head_DL.

Confidence intervals use a paired bootstrap that draws **one** set of sentence
indices per replicate and applies it to all four arrays, preserving the
correlation between metrics. Resampling them independently inflates the interval
on their difference.

### 2.5 Pre-registration

Decision rules were written to `config.yaml` before results were generated, and
`verdict.json` echoes them back with a test asserting the two match. A
hypothesis is SUPPORTED if ≥70% of conditions show the predicted effect with a
95% CI excluding zero, CONTRADICTED if <30% do, EQUIVOCAL otherwise. A condition
is a treebank for the sanity check and a treebank × baseline cell for the
comparative hypotheses.

We flag one consequence of this mechanism in §5: it caught us changing a
threshold, which is why the change is recorded rather than silent.

---

## 3. The comparison disagrees with itself

DLM replicates decisively as a sanity check: real orders are shorter than random
projective relinearizations on both metrics in **26 of 26 treebanks**, all CIs
excluding zero. Japanese is strongest (δ_word = −0.971), German weakest
(−0.412), consistent with their branching rigidity.

The comparative question gives three different answers depending on the
statistic:

| statistic | kind | verdict | cells |
|---|---|---|---|
| Δ\|δ\| | rank | CONTRADICTED | 12/52 |
| ΔP(advantage) | rank | EQUIVOCAL | 17/52 |
| Δ median rel. reduction | magnitude | SUPPORTED | 44/52 |

Under B1 the split is total. The magnitude measure favours head_DL in **26/26**
treebanks; the rank measures favour word_DL in **26/26**. Not one treebank goes
the other way on either.

Both readings describe something real:

- **By rank**, word_DL wins. Median P(real shorter) is 0.861 for word_DL against
  0.783 for head_DL. word_DL identifies the attested order in a larger *share*
  of pairs.
- **By magnitude**, head_DL wins. Median per-sentence reduction is 0.325 against
  0.190: when head_DL separates attested from random, it separates them by
  roughly 1.7× the proportional margin.

head_DL wins fewer pairs but wins them bigger.

Under B2 the picture changes. Rank and magnitude point in opposite directions in
only **6 of 26** treebanks; rank favours head_DL in 12/26 and magnitude in
18/26. The same data, the same metrics, a different baseline, and the
disagreement mostly evaporates.

---

## 4. Baseline difficulty, controlled

§3 is suggestive but correlational. B1 and B2 differ in more than difficulty —
B2 also preserves branching direction, so the contrast cannot distinguish
"difficulty causes the divergence" from "branching does".

### 4.1 A baseline of tunable difficulty

We interpolate. At every node independently, the attested local arrangement is
kept with probability λ and replaced by a uniform shuffle otherwise. Thus λ = 0
**is** B1 and λ = 1 **is** the attested sentence, with the construction held
fixed throughout. Only difficulty varies.

Interpolating per node rather than per sentence matters: mixing whole sentences
would give a bimodal mixture of attested and fully scrambled items, whose rank
statistic is an average of two extremes rather than a genuinely intermediate
difficulty.

Both endpoints are asserted by tests. λ = 0 reproduces B1 sample for sample;
λ = 1 is an exact identity on projective sentences; the arc multiset is preserved
throughout; and mean word_DL is monotone in λ.

### 4.2 Predictions

Stated in the script before it ran:

- **P1**: rank separation shrinks toward zero as P(advantage) approaches its
  ceiling.
- **P2**: magnitude separation does not shrink the same way.
- **P3**: the two diverge monotonically, so the sign flip is a property of
  difficulty rather than of B1's construction.

### 4.3 Result

Sweeping λ over ten values on all 26 treebanks:

| λ | P(adv), best metric | rank sep. | magnitude sep. | divergence |
|---|---|---|---|---|
| 0.0 | 0.817 | −0.114 | +0.137 | 0.251 |
| 0.2 | 0.807 | −0.098 | +0.127 | 0.226 |
| 0.4 | 0.791 | −0.081 | +0.114 | 0.196 |
| 0.6 | 0.765 | −0.059 | +0.089 | 0.148 |
| 0.8 | 0.714 | −0.025 | +0.041 | 0.066 |
| 0.9 | 0.664 | −0.002 | +0.013 | 0.015 |

**The divergence rises on 9 of 9 steps as the baseline gets easier**, with
*r*(λ, divergence) = **−0.964**. There are no exceptions across the sample:

| | holds in |
|---|---|
| *r*(λ, rank separation) > 0 | **26/26** (median +0.96, min +0.49) |
| *r*(λ, magnitude separation) < 0 | **26/26** (median −0.94, max −0.84) |
| divergence monotone in λ (*r* < −0.8) | **26/26** |

**P3 held decisively.** The B1-vs-B2 contrast was not a quirk of B2's
construction. Difficulty alone reproduces the whole effect.

**P1 as written was wrong**, and we report the failure rather than substituting
the corrected test for it. By the literal prediction, |rank_sep| does not shrink
(0.106 easy vs 0.059 hard). Two things were wrong with the formulation.
`rank_sep` is *signed*, so the absolute value hides the pattern: the rank
statistic says word_DL wins by a lot against an easy baseline, and that advantage
decays toward zero as the baseline hardens. And pooling across treebanks
destroys the ceiling correlation entirely — *r* = −0.16 pooled against **−0.90**
once each treebank is centred on its own mean. Baseline difficulty is comparable
within a language but confounded with branching profile between languages.

**P2 held, but not as assumed.** Difficulty does not saturate one measure and
spare the other. It drives both, in opposite directions.

### 4.4 Interpretation

A rank statistic asks how *often* the attested order wins. Against an easy
baseline it wins almost always on both metrics, so the statistic approaches its
ceiling and loses the resolution to separate them. A magnitude statistic asks by
*how much*, and retains resolution there. As the baseline hardens, the rank
statistic regains discrimination while the magnitude gap narrows, and the two
converge.

This is not a subtle interaction. It is large enough to reverse the reported
direction of the comparison, and it is present in every language we tested.

---

## 5. Two statistics that produce their own results

### 5.1 Shrinkage across nested logistic models

The natural test of "word_DL is derivative of head_DL" is whether word_DL's
coefficient collapses once head_DL is controlled. We pre-registered a ≥50%
shrinkage criterion on |β(word_DL)| across nested models.

The criterion is invalid. **Logistic coefficients are not collapsible**: adding a
predictor carrying real signal rescales the remaining coefficients even when
nothing is confounded. Simulation with known ground truth shows the statistic
failing in *both* directions:

- Independent causes, derivativeness **false**: up to **+73%** shrinkage
  reported, falsely confirming derivativeness.
- Head-only cause, word's true effect exactly zero, derivativeness **true**:
  **−81%** shrinkage reported, read as "suppression".

An earlier version of our own write-up interpreted negative-shrinkage cells as
suppression. They were this artifact.

A second defect compounded it. The model was a plain logistic regression
carrying `n_tokens`, `mean_arity` and `tree_depth` as covariates. Because the
design is matched pairs, all three are constant within a pair and control
nothing; the pairing does that work. The correct model is a **conditional
logit** stratified on the pair, where such covariates drop out by construction.

Rescored on the conditional logit with a nested likelihood-ratio test under FDR
control and within-pair ΔAUC, derivativeness is contradicted: β(word_DL) remains
significant in **50 of 52** cells once head_DL is controlled, median β(word_DL) =
−8.74 against β(head_DL) = −0.17, and adding word_DL to head_DL improves
within-pair discrimination by ΔAUC = +0.036.

Two caveats on reading those coefficients. Within-pair, the two metrics'
differences correlate 0.95–0.98 (VIF up to 41), so individual βs are jointly
identified but individually fragile; ΔAUC and the LR test are the interpretable
quantities. And β(head_DL) carries the **wrong sign** in 25 of 52 cells across 18
of the 26 languages, concentrated under B1 (18/26 against 7/26 under B2), which
points back at the difficulty mechanism of §4 rather than at a linguistic fact.

The pre-registration guard caught the threshold change. We record the retirement
in `config.yaml` as an explicit null with reasoning attached, and a test asserts
it stays retired.

### 5.2 A tie-splitting rank measure that is not a new measure

head_DL is much coarser than word_DL and ties with its baseline far more often
(median 4.25% vs 0.21% under B1). Since Cliff's δ appears to score every tie as
non-support, we pre-registered P(advantage) with ties split evenly as a
tie-immune co-primary.

The premise was wrong. With *pos* + *neg* + *ties* = *n*:

```
δ      = (pos − neg)/n
P(adv) = (neg + ties/2)/n
       = (1 − δ)/2                exactly, ties included
```

δ *already* splits ties: a tie contributes to neither *pos* nor *neg*, which is
arithmetically identical to contributing half to each. Measured across all 52
cells, *r*(ΔP(adv), Δδ) = **1.0** at a ratio of exactly 0.5.

So the pre-registered design bought one contrast, not two. What it delivered was
a **rank-versus-magnitude** contrast, which is the one that carried the result in
§4. Pre-registering a statistic does not make it independent of the statistic it
is meant to check; that has to be verified algebraically first.

---

## 6. What this means for DLM work

The narrow finding: Yadav and Husain's comparative claim **holds on a magnitude
reading** (head_DL shows the larger effect in 26/26 treebanks under B1 and 18/26
under B2) and **fails on a rank reading**. We do not think corpus data can
adjudicate between those readings, because they answer different questions: how
often, versus by how much.

The general finding is the one we would press. A DLM comparison requires a
baseline and an effect measure, and the pairing of those two choices can
determine the direction of the result. This is not a precision problem that more
data fixes; the divergence in §4 is monotone and present in every language at
*n* > 400,000 sentences.

Three concrete recommendations:

1. **Report both a rank and a magnitude effect measure.** They answer different
   questions and can disagree completely. Reporting one is reporting a choice.
2. **Report baseline difficulty explicitly**, e.g. as P(attested shorter). A
   result against a baseline beaten in 86% of pairs is near a ceiling, and rank
   statistics behave badly there.
3. **Do not compare coefficients across nested logistic models.** Use a
   likelihood-ratio test or a discrimination measure. Shrinkage measures
   rescaling, not mediation.

---

## 7. Limitations

**Typological coverage.** 15 of 26 treebanks are Indo-European. Raising the
treebank count from 17 to 26 did not fix this, because most newly admitted
treebanks are also Indo-European. Non-IE families are represented by one or two
treebanks each, so claims about typological generality rest on few independent
observations per family. The λ-sweep trend holds in all 26 individually, which
is the strongest form in which we can state it.

**Conditioned on projectivity.** Non-projective sentences (2.5% of English, 12.3%
of Czech) are excluded, since a projective baseline is not comparable.
Non-projectivity is plausibly where the two metrics diverge most.

**λ interpolates toward the attested order**, not toward a principled notion of
"harder baseline". It establishes that difficulty drives the divergence; it does
not say which difficulty level a study ought to use.

**Annotation conventions are load-bearing.** UD's treatment of function-word
headedness directly determines what counts as an intervening head, the exact
quantity under test. Per-treebank reporting limits but does not eliminate this.

**Collinearity limits the H3 analysis.** At VIF up to 41, a null LR result would
be hard to distinguish from insufficient power to separate the two metrics.

**Correlational, not causal.** UD is observational. The findings concern what
distinguishes attested orders from random ones, not what speakers do.

---

## 8. Reproducibility

All numbers are produced by the linked repository under a fixed seed (20260919).
Three scripts regenerate the reported claims from the committed results:

- `main.py`: the full analysis (26 treebanks, ~67 min).
- `verify_h2_split.py`: the baseline-dependence table of §3.
- `experiment_ceiling.py`: the λ sweep of §4.

The test suite (77 tests, no network) pins the contracts that matter: that the
λ endpoints are exact, that relinearization preserves the arc multiset, that the
conditional logit matches an independent difference-logit formulation, that the
retired shrinkage criterion stays retired, and that P(advantage) equals
(1 − δ)/2.

An earlier write-up of this work cited validation numbers that no code in the
repository produced; recomputation matched two of three treebanks and failed on
the third by 0.14 in δ. Those numbers were replaced with reproducible ones, and
`verify_finding1.py` exists so the claim can be checked rather than trusted.

---

## References

Yadav, H., & Husain, S. (2022). A Reappraisal of Dependency Length Minimization
as a Linguistic Universal. *Open Mind: Discoveries in Cognitive Science*, MIT
Press.

Universal Dependencies v2.14 (CC BY-SA).
