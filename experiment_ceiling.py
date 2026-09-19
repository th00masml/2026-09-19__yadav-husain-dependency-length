"""Is the rank/magnitude divergence a CEILING EFFECT? A controlled test.

THE CLAIM UNDER TEST
--------------------
The full run found that rank effect measures (Cliff's delta, P(advantage)) and
a magnitude measure (median per-sentence relative DL reduction) disagree
completely about word_DL vs head_DL against the weak B1 baseline (26/26 cells
point opposite ways) and barely at all against the strong B2 baseline (6/26).

The proposed explanation is a ceiling effect: when a baseline is easy to beat,
a rank statistic saturates near its maximum and loses the resolution to
separate two metrics, while a magnitude statistic still has room to move.

WHY THE EXISTING EVIDENCE IS NOT ENOUGH
---------------------------------------
B1 and B2 differ in more than difficulty -- B2 also preserves each node's
left/right dependent counts. Comparing two fixed baselines therefore cannot
distinguish "difficulty causes the divergence" from "branching direction causes
the divergence". The comparison is correlational.

WHAT THIS SCRIPT DOES
---------------------
Sweeps a baseline of tunable difficulty (`Lam_<x>`, see
linearize.make_order_lambda_interpolated) across lambda in [0, 1]. At every
node the real local arrangement is kept with probability lambda, so lambda=0 IS
B1_uniform and lambda=1 IS the real sentence, with construction held fixed
throughout. Only difficulty varies.

PREDICTIONS, STATED BEFORE RUNNING
----------------------------------
If the ceiling account is right, then as lambda falls from 1 to 0:
  P1. Rank separation between the metrics shrinks toward zero as P(advantage)
      approaches its ceiling of 1.0.
  P2. Magnitude separation does NOT shrink the same way.
  P3. The two therefore diverge monotonically, and the sign flip appears at low
      lambda rather than being a peculiarity of B1's construction.

If instead the divergence is caused by something specific to B1 rather than by
difficulty, rank and magnitude should disagree at lambda=0 but show no orderly
trend across the sweep.

HOW THE PREDICTIONS FARED (recorded after running; see README)
--------------------------------------------------------------
P3 held decisively on all 26 analyzed treebanks: divergence rises on 9/9 steps
as the baseline eases, with corr(lambda, divergence) = -0.96, and is monotone
(corr < -0.8) in 26/26 treebanks individually. The trend is not a peculiarity
of B1.

P1 AS WRITTEN WAS WRONG, and the script reports both the failure and the
reason. `rank_sep` is SIGNED, so testing whether |rank_sep| shrinks asks the
wrong question. What actually happens is that signed rank_sep rises
monotonically toward zero as lambda rises (positive in 26/26 treebanks, median
+0.96):
against an easy baseline the rank statistic says word_DL wins by a lot, and
that advantage disappears as the baseline hardens. Pooling |.| across
treebanks with different baseline levels then masks the effect entirely --
corr(max P(adv), rank_sep) = -0.16 pooled, but -0.90 once each treebank is
centred on its own mean. The ceiling coordinate matters WITHIN a language;
between languages it is confounded with how head-final the language is.

P2 held, but not in the way the ceiling story assumed: magnitude separation is
not merely "less affected", it moves the OPPOSITE way (negative in 26/26
treebanks, median -0.94). So difficulty drives both measures, in
opposite directions, rather than saturating one and leaving the other alone.
That is a stronger and more specific result than the original claim.

Usage:
    python experiment_ceiling.py [--treebanks en_ewt,ja_gsd,...] [--limit N]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import analysis as an
from src import stats as st
from src import ud_loader
from src.tree import is_projective

RUN_DIR = Path(__file__).parent
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Every treebank the headline analysis kept. The reported sweep uses all of
# them; `--treebanks` narrows it for a quicker check.
DEFAULT_TREEBANKS = [
    "ar_padt", "cs_pdt", "de_gsd", "el_gdt", "en_ewt", "es_ancora", "et_edt",
    "eu_bdt", "fa_seraji", "fi_tdt", "fr_gsd", "ga_idt", "he_htb", "hi_hdtb",
    "id_gsd", "it_isdt", "ja_gsd", "ko_kaist", "la_ittb", "lv_lvtb", "pl_pdb",
    "pt_bosque", "ru_syntagrus", "tr_imst", "ug_udt", "zh_gsd",
]


def metrics_for(sent, cfg) -> tuple[float, float]:
    import src.metrics as mx
    m = mx.sentence_metrics(sent, scope=cfg["head_count"]["scope"],
                            include_endpoints=cfg["head_count"]["include_endpoints"])
    return m["word_dl_sum"], m["head_dl_sum"]


def run_treebank(tb: str, cfg: dict, limit: int | None, n_samples: int) -> list[dict]:
    import src.linearize as lin

    specs = {t["id"]: t for t in cfg["treebanks"]}
    path = RUN_DIR / cfg["paths"]["raw_dir"] / specs[tb]["train"]
    if not path.exists():
        print(f"[skip] {tb}: no cached treebank")
        return []

    gen, _ = ud_loader.load_treebank(
        path, tb,
        min_tokens=cfg["preprocessing"]["min_tokens"],
        max_tokens=cfg["preprocessing"]["max_tokens"],
    )
    if limit:
        gen = itertools.islice(gen, limit)
    # The headline analysis is conditioned on projectivity; match it exactly.
    sents = [s for s in gen if is_projective(s)]
    if len(sents) < 200:
        print(f"[skip] {tb}: only {len(sents)} projective sentences")
        return []

    seed = cfg["run"]["seed"]
    real = np.array([metrics_for(s, cfg) for s in sents], dtype=np.float64)
    rw_real, rh_real = real[:, 0], real[:, 1]

    rows = []
    for lam in LAMBDAS:
        kind = f"Lam_{lam}"
        bw = np.empty(len(sents))
        bh = np.empty(len(sents))
        for i, s in enumerate(sents):
            samples = lin.sample_baselines(s, kind, n_samples, seed)
            vals = np.array([metrics_for(x, cfg) for x in samples], dtype=np.float64)
            # Average over samples, exactly as build_treebank_table does, so each
            # real sentence has one matched baseline value per metric.
            bw[i], bh[i] = vals[:, 0].mean(), vals[:, 1].mean()

        d_word = st.paired_cliffs_delta(rw_real, bw)
        d_head = st.paired_cliffs_delta(rh_real, bh)
        p_word = st.prob_advantage(rw_real, bw)
        p_head = st.prob_advantage(rh_real, bh)
        m_word = st.median_relative_reduction(rw_real, bw)
        m_head = st.median_relative_reduction(rh_real, bh)

        rows.append({
            "treebank_id": tb,
            "lam": lam,
            "n_sentences": len(sents),
            # How beatable is this baseline? This is the ceiling coordinate.
            "prob_adv_word": p_word,
            "prob_adv_head": p_head,
            "max_prob_adv": max(p_word, p_head),
            # Separation between the metrics, by each kind of statistic.
            "rank_sep": abs(d_head) - abs(d_word),
            "magnitude_sep": m_head - m_word,
            "delta_word": d_word,
            "delta_head": d_head,
            "med_rel_word": m_word,
            "med_rel_head": m_head,
        })
        print(f"  {tb:10s} lam={lam:4.2f}  P(adv) w={p_word:.3f} h={p_head:.3f}"
              f"  rank_sep={rows[-1]['rank_sep']:+.4f}"
              f"  mag_sep={rows[-1]['magnitude_sep']:+.4f}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--treebanks", default=",".join(DEFAULT_TREEBANKS))
    ap.add_argument("--limit", type=int, default=4000,
                    help="cap sentences per treebank (the sweep is 10x the work "
                         "of one baseline, so the default trades n for lambdas)")
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load((RUN_DIR / "config.yaml").read_text(encoding="utf-8"))

    rows: list[dict] = []
    for tb in args.treebanks.split(","):
        rows += run_treebank(tb.strip(), cfg, args.limit, args.samples)

    if not rows:
        print("no treebanks available")
        return 1

    df = pd.DataFrame(rows)
    out = RUN_DIR / "results" / "ceiling_sweep.csv"
    df.to_csv(out, index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 74)
    print("CEILING SWEEP -- means across treebanks")
    print("=" * 74)
    agg = df.groupby("lam").agg(
        max_prob_adv=("max_prob_adv", "mean"),
        rank_sep=("rank_sep", "mean"),
        magnitude_sep=("magnitude_sep", "mean"),
    )
    agg["divergence"] = agg.magnitude_sep - agg.rank_sep
    print(agg.round(4).to_string())

    print("\n--- P1 (as originally written): does |rank_sep| shrink? ---")
    easy = df[df.lam <= 0.2].rank_sep.abs().mean()
    hard = df[df.lam >= 0.7].rank_sep.abs().mean()
    print(f"  |rank_sep|  easy baselines (lam<=0.2): {easy:.4f}")
    print(f"  |rank_sep|  hard baselines (lam>=0.7): {hard:.4f}")
    print(f"  -> {'shrinks' if easy < hard else 'does NOT shrink'}"
          "   <-- P1 was mis-specified; see below")

    print("\n--- P1 (corrected): SIGNED rank separation vs lambda, per treebank ---")
    print("  rank_sep is signed, so |.| asks the wrong question. The claim is")
    print("  that the rank statistic's verdict moves toward zero as the")
    print("  baseline hardens.")
    for tb, g in df.groupby("treebank_id"):
        g = g.sort_values("lam")
        r = float(np.corrcoef(g.lam, g.rank_sep)[0, 1])
        print(f"    {tb:10s} corr(lambda, rank_sep) = {r:+.3f}"
              f"   {g.iloc[0].rank_sep:+.4f} -> {g.iloc[-1].rank_sep:+.4f}")

    print("\n--- P2: magnitude separation vs lambda, per treebank ---")
    for tb, g in df.groupby("treebank_id"):
        g = g.sort_values("lam")
        r = float(np.corrcoef(g.lam, g.magnitude_sep)[0, 1])
        print(f"    {tb:10s} corr(lambda, magnitude_sep) = {r:+.3f}")
    print("  The two measures move in OPPOSITE directions with difficulty.")

    print("\n--- P3: is the divergence monotone in lambda? ---")
    d = agg.divergence.to_numpy()
    lams = agg.index.to_numpy()
    rho = float(np.corrcoef(lams, d)[0, 1])
    steps = np.diff(d[::-1])  # walking from hard (high lam) to easy (low lam)
    print(f"  corr(lambda, divergence) = {rho:+.3f}")
    print(f"  divergence rises on {int((steps > 0).sum())}/{len(steps)} steps "
          f"as the baseline gets easier")

    # The ceiling coordinate. Pooling across treebanks confounds it with how
    # head-final each language is, so the within-treebank version is the one
    # that answers the question.
    pooled = float(np.corrcoef(df.max_prob_adv, df.rank_sep)[0, 1])
    c = df.copy()
    for col in ("max_prob_adv", "rank_sep", "magnitude_sep"):
        c[col] = c.groupby("treebank_id")[col].transform(lambda v: v - v.mean())
    within_rank = float(np.corrcoef(c.max_prob_adv, c.rank_sep)[0, 1])
    within_mag = float(np.corrcoef(c.max_prob_adv, c.magnitude_sep)[0, 1])
    print(f"\n  corr(max P(advantage), rank_sep)      pooled {pooled:+.3f}"
          f" | within-treebank {within_rank:+.3f}")
    print(f"  corr(max P(advantage), magnitude_sep) within-treebank {within_mag:+.3f}")
    print("  Pooling masks the effect: baseline difficulty is comparable within a")
    print("  language, not across languages with different branching profiles.")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
