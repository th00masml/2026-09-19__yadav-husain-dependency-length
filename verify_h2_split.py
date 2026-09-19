"""Does the H2 rank-vs-magnitude split survive the second baseline?

The headline H2 result is that rank measures favour word_DL while the magnitude
measure favours head_DL. Under B1 that split is total: 17/17 each way. This
script asks whether it is a property of the two METRICS or of the B1 BASELINE,
by recomputing the same contrast under B2_arity_preserving.

Answer: it is a property of B1. Under B2 the two measures point the same way in
14 of 17 treebanks. The mechanism is a ceiling effect -- B1 is weak enough that
both metrics beat it in ~80-88% of pairs, which saturates any rank statistic and
leaves only the magnitude measure with resolution.

Reads results/h2_comparison.csv; recomputes nothing, so it is cheap and cannot
drift from the committed run.

Usage:
    python verify_h2_split.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).parent
BASELINES = ["B1_uniform", "B2_arity_preserving"]


def main() -> int:
    path = RUN_DIR / "results" / "h2_comparison.csv"
    if not path.exists():
        print(f"missing {path}; run `python main.py` first")
        return 1
    h2 = pd.read_csv(path)

    print("=" * 72)
    print("H2: does the rank-vs-magnitude split depend on the baseline?")
    print("=" * 72)

    rows = []
    for kind in BASELINES:
        d = h2[h2.baseline_kind == kind]
        if d.empty:
            continue
        # A cell "splits" when rank and magnitude point in OPPOSITE directions.
        opposite = ((d.delta_diff < 0) & (d.med_rel_diff > 0)) | (
            (d.delta_diff > 0) & (d.med_rel_diff < 0)
        )
        rows.append({
            "baseline": kind,
            "n": len(d),
            "rank_supports_head": int(d.supports_h2.sum()),
            "magnitude_supports_head": int(d.supports_h2_med_rel.sum()),
            "cells_pointing_opposite": int(opposite.sum()),
            "corr_rank_magnitude": float(np.corrcoef(d.delta_diff, d.med_rel_diff)[0, 1]),
            # How beatable is this baseline? A rank statistic saturates as this
            # approaches 1.0 and then cannot separate the two metrics.
            "median_Padv_word": float(d.prob_adv_word.median()),
            "median_Padv_head": float(d.prob_adv_head.median()),
            "median_medrel_word": float(d.med_rel_word.median()),
            "median_medrel_head": float(d.med_rel_head.median()),
        })

    res = pd.DataFrame(rows).set_index("baseline")
    pd.set_option("display.width", 200)

    print("\n--- the split, per baseline ---")
    print(res[["n", "rank_supports_head", "magnitude_supports_head",
               "cells_pointing_opposite", "corr_rank_magnitude"]].to_string())

    print("\n--- how beatable is each baseline? (rank saturation) ---")
    print(res[["median_Padv_word", "median_Padv_head",
               "median_medrel_word", "median_medrel_head"]].round(3).to_string())

    b1 = res.loc["B1_uniform"] if "B1_uniform" in res.index else None
    b2 = res.loc["B2_arity_preserving"] if "B2_arity_preserving" in res.index else None

    print("\n--- verdict ---")
    if b1 is not None and b2 is not None:
        print(f"  B1: {b1.cells_pointing_opposite}/{b1.n} cells point opposite")
        print(f"  B2: {b2.cells_pointing_opposite}/{b2.n} cells point opposite")
        survives = b2.cells_pointing_opposite >= 0.7 * b2.n
        print()
        if survives:
            print("  The split SURVIVES the second baseline: it is a property of")
            print("  the two metrics, not of how the baseline was constructed.")
        else:
            print("  The split does NOT survive. It is a property of B1, not of the")
            print("  metrics. B1 is weak enough that both metrics beat it in most")
            print("  pairs, which saturates any rank statistic; only the magnitude")
            print("  measure retains resolution. Under the harder B2 baseline both")
            print("  measures regain resolution and then largely AGREE.")
            print()
            print("  The claim 'rank and magnitude disagree systematically' must")
            print("  therefore be stated as a claim about weak baselines, not as a")
            print("  general claim about word_DL vs head_DL.")

    out = RUN_DIR / "results" / "h2_split_by_baseline.csv"
    res.to_csv(out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
