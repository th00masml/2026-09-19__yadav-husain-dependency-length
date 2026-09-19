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


def _fetch_multipart(
    spec: dict,
    dest: Path,
    cfg_download: dict,
    offline: bool = False,
) -> Path | None:
    """Fetch several train shards and concatenate them into `dest`.

    All-or-nothing: if any shard fails, nothing is written. A treebank that is
    silently short a third of its sentences is worse than one that is absent,
    because only the second is visible in the manifest.
    """
    if offline:
        log.warning("offline mode and no valid cache for %s -- skipping", spec["id"])
        return None

    try:
        import requests
    except ImportError:
        log.error("`requests` not installed; cannot download. Install requirements.txt.")
        return None

    parts = spec["train_parts"]
    timeout = cfg_download.get("timeout_seconds", 120)
    retries = cfg_download.get("max_retries", 3)
    tmp = dest.with_suffix(dest.suffix + ".part")

    log.info("downloading %s (%d parts)", spec["id"], len(parts))
    complete = False
    try:
        # The temp file must be CLOSED before it is unlinked or replaced;
        # Windows refuses both on an open handle.
        with open(tmp, "wb") as out:
            for fname in parts:
                url = cfg_download["github_raw_template"].format(
                    repo=spec["repo"], fname=fname)
                for attempt in range(1, retries + 1):
                    try:
                        resp = requests.get(url, timeout=timeout, stream=True)
                        if resp.status_code != 200:
                            log.warning("  HTTP %s for %s", resp.status_code, fname)
                            continue
                        for chunk in resp.iter_content(chunk_size=1 << 16):
                            if chunk:
                                out.write(chunk)
                        # CoNLL-U sentences are blank-line delimited; guarantee a
                        # separator so the last sentence of one shard and the
                        # first of the next never merge into one bad block.
                        out.write(b"\n\n")
                        break
                    except Exception as exc:  # noqa: BLE001 - retry any network error
                        log.warning("  error fetching %s: %s", fname, exc)
                else:
                    log.warning("  %s: part %s unavailable; abandoning treebank",
                                spec["id"], fname)
                    break
            else:
                complete = True

        if complete and _looks_like_conllu(tmp):
            tmp.replace(dest)
            log.info("  ok: %s (%.1f MB from %d parts)",
                     spec["id"], dest.stat().st_size / 1e6, len(parts))
            return dest
        if complete:
            log.warning("  %s: concatenated parts do not parse as CoNLL-U", spec["id"])
    finally:
        tmp.unlink(missing_ok=True)
    return None


def fetch_treebank(
    spec: dict,
    raw_dir: Path,
    cfg_download: dict,
    offline: bool = False,
) -> Path | None:
    """Fetch one treebank's train file. Returns the cached path, or None.

    Skips the download entirely if a valid cached copy already exists, which
    makes the whole pipeline resumable after an interrupted run.

    Some treebanks ship their training data as several files rather than one
    (UD splits anything too large for a comfortable single download -- Russian
    SynTagRus is three parts, Czech PDT is eleven). A spec may therefore carry
    `train_parts`, a list of remote filenames that are fetched and concatenated
    into `train`. CoNLL-U is blank-line delimited, so concatenation with a
    separating blank line is a valid merge. Without this, those treebanks 404
    and are silently missing from the typological sample.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / spec["train"]

    if dest.exists() and _looks_like_conllu(dest):
        log.debug("cached: %s", spec["id"])
        return dest

    if spec.get("train_parts"):
        return _fetch_multipart(spec, dest, cfg_download, offline=offline)

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
