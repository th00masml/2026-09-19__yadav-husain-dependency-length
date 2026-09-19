"""Three figures. Each is designed to make a verdict readable at a glance.

Figures are diagnostic instruments, not decoration: every one has an explicit
zero line or reference, because the question in each case is "is this effect
distinguishable from nothing?"
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

WORD_COLOR = "#4C72B0"
HEAD_COLOR = "#C44E52"
GRID = "#D9D9D9"


def _setup():
    import matplotlib
    matplotlib.use("Agg")          # headless: no display needed
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def forest_delta(h1: pd.DataFrame, out_path: Path, baseline_kind: str) -> None:
    """(a) Cliff's delta per treebank, both metrics, with CIs.

    Negative = real sentences are shorter than random. The zero line is the
    "word order does nothing" null.
    """
    plt = _setup()
    sub = h1[h1["baseline_kind"] == baseline_kind]
    if sub.empty:
        return

    order = (
        sub[sub["metric"] == "head_dl"]
        .sort_values("delta")["treebank_id"].tolist()
    )
    if not order:
        order = sorted(sub["treebank_id"].unique())

    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.32 * len(order) + 1.4)))
    ypos = np.arange(len(order))

    for metric, color, offset, label in (
        ("word_dl", WORD_COLOR, -0.16, "word-based DL"),
        ("head_dl", HEAD_COLOR, +0.16, "head-based DL"),
    ):
        m = sub[sub["metric"] == metric].set_index("treebank_id")
        vals, los, his, ys = [], [], [], []
        for i, tb in enumerate(order):
            if tb not in m.index:
                continue
            row = m.loc[tb]
            vals.append(row["delta"])
            los.append(row["delta"] - row["ci_lo"])
            his.append(row["ci_hi"] - row["delta"])
            ys.append(i + offset)
        ax.errorbar(
            vals, ys, xerr=[np.abs(los), np.abs(his)],
            fmt="o", ms=4, lw=0, elinewidth=1.2, capsize=2,
            color=color, ecolor=color, label=label,
        )

    ax.axvline(0, color="#333333", lw=1.0, ls="--", zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Cliff's $\\delta$  (real vs. random linearization)")
    ax.set_title(
        f"H1: dependency-length minimization by metric\nbaseline = {baseline_kind}",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.text(
        0.01, -0.06, "negative = real sentences shorter than random",
        transform=ax.transAxes, fontsize=7.5, color="#666666",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def forest_h2(h2: pd.DataFrame, out_path: Path) -> None:
    """(b) Delta|delta| with the zero line -- the H2 verdict, per treebank."""
    plt = _setup()
    if h2.empty:
        return

    sub = h2.sort_values("delta_diff")
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.3 * len(sub) + 1.4)))
    ys = np.arange(len(sub))

    colors = [
        "#55A868" if r.supports_h2 else ("#C44E52" if r.contradicts_h2 else "#999999")
        for r in sub.itertuples()
    ]
    ax.errorbar(
        sub["delta_diff"], ys,
        xerr=[
            np.abs(sub["delta_diff"] - sub["ci_lo"]),
            np.abs(sub["ci_hi"] - sub["delta_diff"]),
        ],
        fmt="none", elinewidth=1.2, capsize=2, ecolor="#888888", zorder=2,
    )
    ax.scatter(sub["delta_diff"], ys, c=colors, s=26, zorder=3)
    ax.axvline(0, color="#333333", lw=1.0, ls="--", zorder=1)

    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{r.treebank_id} ({r.baseline_kind.split('_')[0]})" for r in sub.itertuples()],
        fontsize=7.5,
    )
    ax.set_xlabel("$|\\delta_{head}| - |\\delta_{word}|$   (paired bootstrap 95% CI)")
    ax.set_title(
        "H2: does the head-based metric show a larger effect?\n"
        "green = supports, red = contradicts, grey = equivocal",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def h3_scatter(h3: pd.DataFrame, out_path: Path) -> None:
    """(c) beta(word_DL) with vs. without the head_DL control, paired.

    Points falling toward the horizontal axis are treebanks where word-based DL
    lost its predictive power once head-based DL was in the model -- which is
    exactly what H3 predicts.
    """
    plt = _setup()
    if h3.empty:
        return

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    x = h3["beta_word_reduced"].to_numpy(dtype=float)
    y = h3["beta_word_full"].to_numpy(dtype=float)

    colors = ["#55A868" if s else "#999999" for s in h3["supports_h3"]]
    ax.scatter(x, y, c=colors, s=34, zorder=3, edgecolor="white", linewidth=0.6)

    lim = float(np.nanmax(np.abs(np.concatenate([x, y])))) * 1.15 if len(x) else 1.0
    lim = max(lim, 0.1)
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="#333333", lw=1.0,
            label="no change", zorder=1)
    ax.axhline(0, color="#C44E52", lw=1.0, ls=":", label="$\\beta$ = 0 (fully derivative)", zorder=1)

    for r in h3.itertuples():
        ax.annotate(
            r.treebank_id,
            (r.beta_word_reduced, r.beta_word_full),
            fontsize=6.5, color="#555555",
            xytext=(3, 3), textcoords="offset points",
        )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("$\\beta$(word_DL)  without head_DL control")
    ax.set_ylabel("$\\beta$(word_DL)  with head_DL control")
    ax.set_title("H3: is word-based DL derivative?\n"
                 "green = word_DL loses its independent effect", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def make_all(h1: pd.DataFrame, h2: pd.DataFrame, h3: pd.DataFrame, figdir: Path) -> list[Path]:
    """Render every figure; never let a plotting failure kill a completed run."""
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        for kind in h1["baseline_kind"].unique() if len(h1) else []:
            p = figdir / f"fig_a_forest_delta_{kind}.png"
            forest_delta(h1, p, kind)
            written.append(p)
        if len(h2):
            p = figdir / "fig_b_h2_delta_difference.png"
            forest_h2(h2, p)
            written.append(p)
        if len(h3):
            p = figdir / "fig_c_h3_beta_shrinkage.png"
            h3_scatter(h3, p)
            written.append(p)
    except Exception as exc:  # noqa: BLE001
        log.error("figure generation failed (results are still valid): %s", exc)
    return written
