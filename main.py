#!/usr/bin/env python3
"""Orchestrator for the yadav-husain-dependency-length experiment.

Usage:
    python main.py                 # full run (downloads UD, ~15-40 min)
    python main.py --fast          # 5 treebanks, 5 baseline samples (~90 s cached)
    python main.py --offline       # cache only, no network
    python main.py --fixture       # run entirely on tests/fixtures/mini.conllu
    python main.py --stage analyze # skip acquisition, reuse cache

Failure policy: if nothing can be acquired, exit non-zero having written
nothing. A network failure must degrade to "no results", never to "wrong
results".
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from src import analysis, plots, ud_download, ud_loader  # noqa: E402

log = logging.getLogger("main")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        log.error("PyYAML not installed. Run:  pip install -r requirements.txt")
        raise SystemExit(2)
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="5 treebanks, fewer bootstrap resamples")
    ap.add_argument("--offline", action="store_true", help="use cached data only, never download")
    ap.add_argument("--fixture", action="store_true", help="run on the bundled mini.conllu fixture")
    ap.add_argument("--stage", choices=["all", "download", "analyze"], default="all")
    ap.add_argument("--seed", type=int, default=None, help="override the master seed")
    ap.add_argument("--config", default=str(RUN_DIR / "config.yaml"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)
    t_start = time.time()

    cfg = load_config(Path(args.config))
    if args.seed is not None:
        cfg["run"]["seed"] = args.seed

    n_samples = cfg["baselines"]["fast_n_samples"] if args.fast else cfg["baselines"]["n_samples"]
    n_resamples = cfg["bootstrap"]["fast_n_resamples"] if args.fast else cfg["bootstrap"]["n_resamples"]

    results_dir = RUN_DIR / cfg["paths"]["results_dir"]
    figures_dir = RUN_DIR / cfg["paths"]["figures_dir"]
    raw_dir = RUN_DIR / cfg["paths"]["raw_dir"]

    log.info("=" * 68)
    log.info("yadav-husain-dependency-length")
    log.info("head-based vs word-based dependency length minimization")
    log.info("=" * 68)
    log.info("mode: %s%s%s", "fast " if args.fast else "full ",
             "offline " if args.offline else "", "fixture" if args.fixture else "")
    log.info("baseline samples/sentence: %d | bootstrap resamples: %d", n_samples, n_resamples)
    log.info("seed: %d", cfg["run"]["seed"])

    # ---------------------------------------------------------------- acquire
    if args.fixture:
        fixture = RUN_DIR / cfg["paths"]["fixture"]
        if not fixture.exists():
            log.error("fixture not found at %s", fixture)
            return 2
        sources = {"fixture": fixture}
        log.info("using fixture: %s", fixture)
    else:
        specs = cfg["treebanks"]
        if args.fast:
            keep = set(cfg["fast_treebanks"])
            specs = [s for s in specs if s["id"] in keep]
        log.info("acquiring %d treebanks ...", len(specs))
        sources = ud_download.fetch_all(
            specs, raw_dir, cfg["download"], offline=args.offline
        )
        if not sources:
            log.error("")
            log.error("ACQUISITION FAILED: no treebank could be downloaded or read from cache.")
            log.error("Nothing has been written. Options:")
            log.error("  * check network access to raw.githubusercontent.com")
            log.error("  * run with --fixture to exercise the pipeline offline")
            log.error("")
            return 1
        log.info("acquired %d/%d treebanks", len(sources), len(specs))

    if args.stage == "download":
        log.info("stage=download complete; stopping before analysis")
        return 0

    # ---------------------------------------------------------------- analyze
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    h1_all: list[dict] = []
    h2_all: list[dict] = []
    h3_all: list[dict] = []

    min_sentences = 0 if (args.fixture or args.fast) else cfg["filters"]["min_sentences"]
    if args.fixture:
        log.info("fixture mode: min_sentences filter disabled")
    elif args.fast:
        log.info("fast mode: min_sentences filter disabled")

    for tb_id, path in sorted(sources.items()):
        log.info("-" * 60)
        log.info("[%s] loading", tb_id)
        try:
            gen, stats = ud_loader.load_treebank(
                path, tb_id,
                min_tokens=cfg["preprocessing"]["min_tokens"],
                max_tokens=cfg["preprocessing"]["max_tokens"],
            )
            df, info = analysis.build_treebank_table(gen, tb_id, cfg, n_samples)
        except Exception as exc:  # noqa: BLE001 - one bad treebank must not kill the run
            log.error("[%s] failed: %s", tb_id, exc)
            manifest_rows.append({"treebank_id": tb_id, "status": "ERROR", "error": str(exc)})
            continue

        row = stats.as_row()
        row.update(info)
        row["path"] = str(path.name)

        if stats.n_kept < min_sentences:
            log.info("[%s] %d sentences < %d threshold -- excluded",
                     tb_id, stats.n_kept, min_sentences)
            row["status"] = "EXCLUDED_TOO_SMALL"
            manifest_rows.append(row)
            continue

        if stats.rejection_rate > cfg["preprocessing"]["max_rejection_rate_warn"]:
            log.warning("[%s] HIGH REJECTION RATE %.1f%% -- possible annotation-convention "
                        "problem, inspect before trusting", tb_id, 100 * stats.rejection_rate)
            row["high_rejection_warning"] = True

        if info["nonprojective_rate"] > cfg["filters"]["nonprojective_report_threshold"]:
            log.warning("[%s] %.1f%% non-projective sentences -- excluded from the headline "
                        "analysis and reported separately",
                        tb_id, 100 * info["nonprojective_rate"])
            row["high_nonprojectivity"] = True

        row["status"] = "ANALYZED"
        manifest_rows.append(row)
        log.info("[%s] %d kept / %d seen (%.1f%% rejected), %d projective analyzed, "
                 "%.1f%% non-projective",
                 tb_id, stats.n_kept, stats.n_seen, 100 * stats.rejection_rate,
                 info["n_analyzed"], 100 * info["nonprojective_rate"])

        if info["n_analyzed"] < 10:
            log.warning("[%s] too few projective sentences for statistics", tb_id)
            continue

        log.info("[%s] H1/H2 ...", tb_id)
        h1_rows, h2_rows = analysis.run_h1_h2(df, tb_id, cfg, n_resamples)
        h1_all.extend(h1_rows)
        h2_all.extend(h2_rows)

        log.info("[%s] H3 ...", tb_id)
        h3_all.extend(analysis.run_h3(df, tb_id, cfg))

    # ---------------------------------------------------------------- report
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(results_dir / "treebank_manifest.csv", index=False)
    log.info("wrote %s", results_dir / "treebank_manifest.csv")

    h1_df = pd.DataFrame(h1_all)
    h2_df = pd.DataFrame(h2_all)
    h3_df = pd.DataFrame(h3_all)

    if h1_df.empty and h2_df.empty and h3_df.empty:
        log.error("no treebank produced usable statistics; nothing to report")
        return 1

    if not h1_df.empty:
        h1_df.to_csv(results_dir / "per_treebank.csv", index=False)
        log.info("wrote %s", results_dir / "per_treebank.csv")
    if not h2_df.empty:
        h2_df.to_csv(results_dir / "h2_comparison.csv", index=False)
        log.info("wrote %s", results_dir / "h2_comparison.csv")
    if not h3_df.empty:
        h3_df.to_csv(results_dir / "h3_models.csv", index=False)
        log.info("wrote %s", results_dir / "h3_models.csv")

    run_meta = {
        "run_date": "2026-09-19",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": {"fast": args.fast, "offline": args.offline, "fixture": args.fixture},
        "seed": cfg["run"]["seed"],
        "n_baseline_samples": n_samples,
        "n_bootstrap_resamples": n_resamples,
        "n_treebanks_analyzed": int((manifest["status"] == "ANALYZED").sum())
        if "status" in manifest else 0,
        "head_count_scope": cfg["head_count"]["scope"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "elapsed_seconds": None,
    }

    verdict = analysis.build_verdict(h1_df, h2_df, h3_df, cfg, run_meta)
    verdict["run"]["elapsed_seconds"] = round(time.time() - t_start, 1)
    analysis.write_verdict(verdict, results_dir / "verdict.json")

    plots.make_all(h1_df, h2_df, h3_df, figures_dir)

    # ---------------------------------------------------------------- summary
    log.info("=" * 68)
    log.info("VERDICTS  (pre-registered thresholds: support>=%.0f%%, contradict<%.0f%%)",
             100 * cfg["thresholds"]["support_fraction"],
             100 * cfg["thresholds"]["contradict_fraction"])
    log.info("=" * 68)
    for hid, h in verdict.get("hypotheses", {}).items():
        log.info("%s: %-16s  %d/%d conditions supporting (%.0f%%)",
                 hid, h["verdict"], h["n_supporting"], h["n_conditions"],
                 100 * h["fraction_supporting"])
    log.info("=" * 68)
    if "power_advisory" in verdict:
        log.warning("UNDERPOWERED RUN (%d sentences/treebank): the verdicts above "
                    "reflect low power, NOT evidence against the hypotheses. "
                    "Read the point estimates in per_treebank.csv instead.",
                    verdict["power_advisory"]["max_sentences_per_treebank"])
        log.info("=" * 68)
    log.info("EQUIVOCAL and CONTRADICTED are legitimate outcomes, not failures.")
    log.info("elapsed: %.1f s", time.time() - t_start)
    log.info("results: %s", results_dir)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.error("interrupted")
        sys.exit(130)
