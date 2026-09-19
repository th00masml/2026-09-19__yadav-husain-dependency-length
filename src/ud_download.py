"""Universal Dependencies acquisition: fetch, validate, cache.

Design stance: the network is assumed hostile. Every path is fallible, nothing
is counted as acquired until it has actually parsed as CoNLL-U, and total
failure exits cleanly rather than leaving a half-written run. The fixture at
tests/fixtures/mini.conllu means the pipeline and the whole test suite run with
no network at all.

Two acquisition paths, in order:
  1. per-treebank from GitHub raw  (selective, resumable, ~5-40 MB each)
  2. the LINDAT bulk tarball       (~2.5 GB; only if (1) fails entirely)
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path

log = logging.getLogger(__name__)


class AcquisitionError(RuntimeError):
    """Raised when no treebank could be acquired. Caller should exit non-zero."""


def _looks_like_conllu(path: Path, min_sentences: int = 5) -> bool:
    """Cheap structural validation: does this file actually parse?

    Checks the first few sentence blocks rather than the whole file -- enough to
    catch an HTML error page, a truncated download, or an LFS pointer stub,
    which are the realistic failure modes.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if not text.strip():
        return False
    if text.lstrip().startswith("<"):        # HTML error page
        return False
    if "version https://git-lfs" in text[:200]:
        return False

    blocks = 0
    saw_tab_row = False
    for line in text.splitlines():
        if not line.strip():
            if saw_tab_row:
                blocks += 1
                saw_tab_row = False
            if blocks >= min_sentences:
                return True
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) >= 8:
            saw_tab_row = True
    return blocks + int(saw_tab_row) >= 1


def fetch_treebank(
    spec: dict,
    raw_dir: Path,
    cfg_download: dict,
    offline: bool = False,
) -> Path | None:
    """Fetch one treebank's train file. Returns the cached path, or None.

    Skips the download entirely if a valid cached copy already exists, which
    makes the whole pipeline resumable after an interrupted run.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / spec["train"]

    if dest.exists() and _looks_like_conllu(dest):
        log.debug("cached: %s", spec["id"])
        return dest

    if offline:
        log.warning("offline mode and no valid cache for %s -- skipping", spec["id"])
        return None

    try:
        import requests
    except ImportError:
        log.error("`requests` not installed; cannot download. Install requirements.txt.")
        return None

    url = cfg_download["github_raw_template"].format(repo=spec["repo"], fname=spec["train"])
    timeout = cfg_download.get("timeout_seconds", 120)
    retries = cfg_download.get("max_retries", 3)

    for attempt in range(1, retries + 1):
        try:
            log.info("downloading %s (attempt %d/%d)", spec["id"], attempt, retries)
            resp = requests.get(url, timeout=timeout, stream=True)
            if resp.status_code != 200:
                log.warning("  HTTP %s for %s", resp.status_code, spec["id"])
                continue

            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)

            # Validate BEFORE promoting the .part file to the real name, so a
            # corrupt download never poisons the cache for later runs.
            if _looks_like_conllu(tmp):
                tmp.replace(dest)
                log.info("  ok: %s (%.1f MB)", spec["id"], dest.stat().st_size / 1e6)
                return dest
            log.warning("  downloaded %s but it does not parse as CoNLL-U", spec["id"])
            tmp.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - any network error is just a retry
            log.warning("  error fetching %s: %s", spec["id"], exc)

    return None


def fetch_all(
    treebanks: list[dict],
    raw_dir: Path,
    cfg_download: dict,
    offline: bool = False,
) -> dict[str, Path]:
    """Fetch every treebank we can. Returns {treebank_id: path} for successes."""
    acquired: dict[str, Path] = {}
    for spec in treebanks:
        path = fetch_treebank(spec, raw_dir, cfg_download, offline=offline)
        if path is not None:
            acquired[spec["id"]] = path

    if not acquired and not offline:
        log.warning("per-treebank acquisition acquired nothing; trying bulk tarball")
        acquired = _try_bulk(treebanks, raw_dir, cfg_download)

    return acquired


def _try_bulk(
    treebanks: list[dict],
    raw_dir: Path,
    cfg_download: dict,
) -> dict[str, Path]:
    """Fallback path: the ~2.5 GB LINDAT tarball, extracting only what we need."""
    try:
        import requests
    except ImportError:
        return {}

    url = cfg_download.get("lindat_bulk_url")
    if not url:
        return {}

    tarball = Path(raw_dir) / "ud-treebanks.tgz"
    if not tarball.exists():
        try:
            log.info("downloading bulk tarball (this is ~2.5 GB and slow)")
            resp = requests.get(url, timeout=cfg_download.get("timeout_seconds", 120), stream=True)
            if resp.status_code != 200:
                log.error("bulk tarball HTTP %s", resp.status_code)
                return {}
            with open(tarball, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
        except Exception as exc:  # noqa: BLE001
            log.error("bulk download failed: %s", exc)
            tarball.unlink(missing_ok=True)
            return {}

    wanted = {spec["train"]: spec["id"] for spec in treebanks}
    acquired: dict[str, Path] = {}
    try:
        with tarfile.open(tarball, "r:gz") as tf:
            for member in tf:
                name = Path(member.name).name
                if name in wanted:
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    dest = Path(raw_dir) / name
                    dest.write_bytes(fh.read())
                    if _looks_like_conllu(dest):
                        acquired[wanted[name]] = dest
                    else:
                        dest.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        log.error("bulk extraction failed: %s", exc)

    return acquired
