"""Shared pytest fixtures.

The important one here is `isolated_results`, which exists because of a defect
found in review: the end-to-end test invoked `main.py --fixture` with the run
directory as cwd, so every `pytest` run silently overwrote `results/` with
30-sentence smoke-test output, destroying the real 5-treebank findings that
README.md cites. Tests must never clobber deliverables.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

RUN_DIR = Path(__file__).resolve().parent.parent
RESULTS = RUN_DIR / "results"


@pytest.fixture
def isolated_results(tmp_path: Path):
    """Give a subprocess run its own results/figures dirs under tmp_path.

    Returns a dict of CLI args to append, so the pipeline writes nowhere near
    the real deliverables.
    """
    res = tmp_path / "results"
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    return {"results_dir": res, "figures_dir": figs}


@pytest.fixture
def config_with_tmp_results(tmp_path: Path):
    """A copy of config.yaml whose output paths point inside tmp_path.

    `main.py` takes --config, and all output locations are read from the config,
    so redirecting them there is enough to sandbox a full end-to-end run.
    """
    import yaml

    cfg = yaml.safe_load((RUN_DIR / "config.yaml").read_text(encoding="utf-8"))
    out = tmp_path / "out"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # config paths are resolved relative to RUN_DIR by main.py, so we need paths
    # that resolve to tmp_path. Absolute paths survive the `RUN_DIR / p` join on
    # both POSIX and Windows.
    cfg["paths"]["results_dir"] = str(out)
    cfg["paths"]["figures_dir"] = str(out / "figures")
    cfg["paths"]["fixture"] = str(RUN_DIR / "tests" / "fixtures" / "mini.conllu")

    cfg_path = tmp_path / "config_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path, out


@pytest.fixture
def preserve_results():
    """Back up results/ and restore it afterwards.

    Belt-and-braces guard for any test that still shells out to main.py without
    redirecting its output. If the backup/restore ever becomes a no-op because
    the pipeline was fixed to respect --config paths, this stays harmless.
    """
    backup = None
    if RESULTS.exists():
        backup = shutil.copytree(RESULTS, RESULTS.parent / "_results_backup", dirs_exist_ok=True)
    try:
        yield RESULTS
    finally:
        if backup is not None:
            shutil.rmtree(RESULTS, ignore_errors=True)
            shutil.copytree(backup, RESULTS, dirs_exist_ok=True)
            shutil.rmtree(backup, ignore_errors=True)
