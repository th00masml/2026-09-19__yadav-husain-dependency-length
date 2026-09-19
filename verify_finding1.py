"""Recompute README "Validation finding 1" from the repo's own pipeline.

Finding 1 claims, for en_ewt / ja_gsd / tr_imst:
  * delta_word / delta_head          (a SECOND set, different from the H1 table)
  * relative reduction (baseline-real)/baseline for both metrics
  * tie rates: 5.3% (en), 13.2% (tr) for head_dl; 0.5-1.3% for word_dl
  * distinct values over 1,200 English sentences: 235 word vs 68 head

None of these numbers are produced by any code in the repository. This script
recomputes all of them with the repo's own loader / linearizer / metrics, under
the same config and seed, so the claim can be checked rather than trusted.

Usage:
    python verify_finding1.py [--limit N]
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

RUN_DIR = Path(__file__).parent
CITED = ["en_ewt", "ja_gsd", "tr_imst"]

# What the README's validation table asserts.
README_CLAIM = {
    "en_ewt": {"delta_word": -0.733, "delta_head": -0.695, "rel_word": 0.286, "rel_head": 0.454},
    "ja_gsd": {"delta_word": -0.970, "delta_head": -0.947, "rel_word": 0.428, "rel_head": 0.653},
    "tr_imst": {"delta_word": -0.708, "delta_head": -0.432, "rel_word": 0.189, "rel_head": 0.280},
}
README_TIES = {"en_ewt": 0.053, "tr_imst": 0.132}          # head_dl tie rates
README_DISTINCT = {"n_sentences": 1200, "word": 235, "head": 68}


def rel_reduction(real: np.ndarray, base: np.ndarray) -> float:
    """(baseline - real) / baseline, as a ratio of SUMS (README's wording)."""
    tot = base.sum()
    return float((tot - real.sum()) / tot) if tot > 0 else float("nan")


def rel_reduction_per_sentence(real: np.ndarray, base: np.ndarray) -> float:
    """Median of the per-sentence ratio -- the length-unweighted alternative."""
    ok = base > 0
    return float(np.median((base[ok] - real[ok]) / base[ok])) if ok.any() else float("nan")


def tie_rate(real: np.ndarray, base: np.ndarray) -> float:
    return float(np.mean(real == base))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap sentences per treebank (default: all, as the main run does)")
    args = ap.parse_args()

    cfg = yaml.safe_load((RUN_DIR / "config.yaml").read_text(encoding="utf-8"))
    raw_dir = RUN_DIR / cfg["paths"]["raw_dir"]
    specs = {t["id"]: t for t in cfg["treebanks"]}
    n_samples = cfg["baselines"]["fast_n_samples"]

    rows, dist_rows = [], []

    for tb in CITED:
        path = raw_dir / specs[tb]["train"]
        if not path.exists():
            print(f"[skip] {tb}: no cached treebank at {path}")
            continue

        sents, _ = ud_loader.load_treebank(
            path, tb,
            min_tokens=cfg["preprocessing"]["min_tokens"],
            max_tokens=cfg["preprocessing"]["max_tokens"],
        )
        if args.limit:
            sents = itertools.islice(sents, args.limit)
        df, _ = an.build_treebank_table(sents, tb, cfg, n_samples=n_samples)

        real = df[df.baseline_kind == "REAL"].set_index("sent_id").sort_index()
        base = df[df.baseline_kind == "B1_uniform"].set_index("sent_id").sort_index()
        common = real.index.intersection(base.index)
        real, base = real.loc[common], base.loc[common]

        rec = {"treebank_id": tb, "n_sentences": len(common)}
        for metric in ("word_dl", "head_dl"):
            r, b = real[metric].to_numpy(float), base[metric].to_numpy(float)
            short = metric.split("_")[0]
            rec[f"delta_{short}"] = st.paired_cliffs_delta(r, b)
            rec[f"rel_{short}"] = rel_reduction(r, b)
            rec[f"relmed_{short}"] = rel_reduction_per_sentence(r, b)
            rec[f"tie_{short}"] = tie_rate(r, b)
        rows.append(rec)

        # Distinct-value counts on the first 1,200 sentences (README's claim).
        head = real.head(README_DISTINCT["n_sentences"])
        dist_rows.append({
            "treebank_id": tb,
            "n_sentences": len(head),
            "distinct_word": int(head["word_dl"].nunique()),
            "distinct_head": int(head["head_dl"].nunique()),
        })

    if not rows:
        print("No treebanks available. Download them into data/raw/ first.")
        return 1

    res = pd.DataFrame(rows).set_index("treebank_id")
    dist = pd.DataFrame(dist_rows).set_index("treebank_id")

    pd.set_option("display.width", 200)
    print("\n=== RECOMPUTED (B1_uniform, this repo's own code) ===")
    print(res[["n_sentences", "delta_word", "delta_head",
               "rel_word", "rel_head", "relmed_word", "relmed_head",
               "tie_word", "tie_head"]].round(4).to_string())

    print("\n=== vs README validation table ===")
    print(f"{'treebank':10s} {'quantity':12s} {'README':>9s} {'recomputed':>11s} {'diff':>9s}")
    print("-" * 56)
    for tb, claim in README_CLAIM.items():
        if tb not in res.index:
            continue
        for key, want in claim.items():
            got = float(res.loc[tb, key])
            print(f"{tb:10s} {key:12s} {want:9.3f} {got:11.3f} {got - want:+9.3f}")

    print("\n=== tie rates (head_dl) ===")
    for tb, want in README_TIES.items():
        if tb in res.index:
            got = float(res.loc[tb, "tie_head"])
            print(f"{tb:10s} README {want:6.3f}   recomputed {got:6.3f}   diff {got - want:+6.3f}")

    print("\n=== distinct values (first 1,200 sentences) ===")
    print(dist.to_string())
    print(f"README claims for en_ewt: word={README_DISTINCT['word']}, "
          f"head={README_DISTINCT['head']}")

    print("\n=== direction check: which metric wins under each statistic? ===")
    for tb in res.index:
        d_win = "head" if abs(res.loc[tb, "delta_head"]) > abs(res.loc[tb, "delta_word"]) else "word"
        r_win = "head" if res.loc[tb, "rel_head"] > res.loc[tb, "rel_word"] else "word"
        m_win = "head" if res.loc[tb, "relmed_head"] > res.loc[tb, "relmed_word"] else "word"
        print(f"{tb:10s} by delta: {d_win:5s} | by rel-reduction(sums): {r_win:5s} "
              f"| by per-sentence median: {m_win:5s}")

    out = RUN_DIR / "results" / "finding1_recomputed.csv"
    res.join(dist[["distinct_word", "distinct_head"]]).to_csv(out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
