"""CoNLL-U -> validated Sentence objects.

THIS MODULE IS WHERE THE BUGS LIVE. Two classes of CoNLL-U line are not real
tokens and must be dropped before any index arithmetic:

    1-2   fuse    _    ...     <- multiword token range (e.g. Spanish "del")
    3.1   _       _    ...     <- empty node (ellipsis in enhanced deps)

If you leave these in, the ID column no longer matches the row position and
every dependency length silently comes out wrong -- wrong, not crashed, which
is the dangerous kind. After dropping them we re-index 0..n-1 contiguously and
remap every HEAD through the same map.

Sentences are yielded lazily so a treebank is never fully resident in memory.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .tree import ROOT, Sentence, validate

log = logging.getLogger(__name__)


@dataclass
class LoadStats:
    """Per-treebank counts, written to results/treebank_manifest.csv."""

    treebank_id: str
    n_seen: int = 0
    n_kept: int = 0
    rejections: Counter = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rejections is None:
            self.rejections = Counter()

    @property
    def rejection_rate(self) -> float:
        return 0.0 if self.n_seen == 0 else 1.0 - (self.n_kept / self.n_seen)

    def as_row(self) -> dict:
        row = {
            "treebank_id": self.treebank_id,
            "n_seen": self.n_seen,
            "n_kept": self.n_kept,
            "rejection_rate": round(self.rejection_rate, 4),
        }
        for reason, count in sorted(self.rejections.items()):
            row[f"reject_{reason}"] = count
        return row


def _is_real_token_id(token_id) -> bool:
    """False for multiword ranges (tuple with '-') and empty nodes (with '.').

    The `conllu` package represents these as tuples: (1, '-', 2) for a range and
    (3, '.', 1) for an empty node. A plain int is a real token.
    """
    return isinstance(token_id, int)


def parse_conllu_text(
    text: str,
    treebank_id: str,
    min_tokens: int = 3,
    max_tokens: int = 100,
    stats: LoadStats | None = None,
) -> Iterator[Sentence]:
    """Parse CoNLL-U text into validated Sentences.

    Uses the `conllu` package when available; otherwise falls back to a compact
    built-in parser so the fixture path works even before the venv exists.
    """
    try:
        import conllu  # noqa: F401
        blocks = _iter_sentences_conllu(text)
    except ImportError:  # pragma: no cover - exercised only without the venv
        log.debug("conllu package unavailable; using built-in fallback parser")
        blocks = _iter_sentences_fallback(text)

    for sent_id, rows in blocks:
        if stats is not None:
            stats.n_seen += 1

        sentence = _rows_to_sentence(
            rows, treebank_id, sent_id, min_tokens, max_tokens, stats
        )
        if sentence is not None:
            if stats is not None:
                stats.n_kept += 1
            yield sentence


def _rows_to_sentence(
    rows: list[tuple],
    treebank_id: str,
    sent_id: str,
    min_tokens: int,
    max_tokens: int,
    stats: LoadStats | None,
) -> Sentence | None:
    """rows: list of (id, form, upos, head, deprel) with raw CoNLL-U ids."""

    def reject(reason: str) -> None:
        if stats is not None:
            stats.rejections[reason] += 1

    # --- 1. Drop MWT ranges and empty nodes -------------------------------
    real = [r for r in rows if _is_real_token_id(r[0])]
    if not real:
        reject("empty_sentence")
        return None

    n = len(real)
    if n < min_tokens:
        reject("too_short")
        return None
    if n > max_tokens:
        reject("too_long")
        return None

    # --- 2. Re-index 0..n-1 and remap HEAD --------------------------------
    # old CoNLL-U id (1-based) -> new contiguous 0-based position
    id_map = {int(r[0]): i for i, r in enumerate(real)}

    heads = np.full(n, ROOT, dtype=np.int32)
    for i, r in enumerate(real):
        raw_head = r[3]
        if raw_head is None:
            reject("missing_head")
            return None
        raw_head = int(raw_head)
        if raw_head == 0:
            heads[i] = ROOT          # 0 means root in CoNLL-U
        elif raw_head in id_map:
            heads[i] = id_map[raw_head]
        else:
            # Head points at a token we dropped, or outside the sentence.
            reject("head_out_of_range")
            return None

    # --- 3. Validate tree well-formedness ---------------------------------
    reason = validate(heads, n)
    if reason is not None:
        reject(reason)
        return None

    return Sentence(
        tokens=tuple(str(r[1]) for r in real),
        heads=heads,
        deprels=tuple(str(r[4]) for r in real),
        upos=tuple(str(r[2]) for r in real),
        treebank_id=treebank_id,
        sent_id=sent_id,
    )


def _iter_sentences_conllu(text: str) -> Iterator[tuple[str, list[tuple]]]:
    """Sentence blocks via the `conllu` package.

    Parsed per blank-line-separated block rather than with `parse_incr`, so a
    treebank is never fully materialized as TokenList objects at once -- the
    memory budget only allows one sentence resident at a time.
    """
    import conllu

    buf: list[str] = []
    index = 0

    def flush(block: list[str], i: int):
        raw = "\n".join(block)
        if not raw.strip():
            return None
        parsed = conllu.parse(raw)
        if not parsed:
            return None
        tl = parsed[0]
        sent_id = tl.metadata.get("sent_id", f"s{i}") if tl.metadata else f"s{i}"
        rows = [
            (t["id"], t["form"], t.get("upos") or "_", t.get("head"), t.get("deprel") or "_")
            for t in tl
        ]
        return str(sent_id), rows

    for line in text.splitlines():
        if line.strip():
            buf.append(line)
            continue
        # A block of nothing but comments (e.g. a file header banner) is not a
        # sentence and must not be counted as one seen-and-rejected.
        if buf and any(not ln.startswith("#") for ln in buf):
            got = flush(buf, index)
            if got:
                index += 1
                yield got
        buf = []

    if buf and any(not ln.startswith("#") for ln in buf):
        got = flush(buf, index)
        if got:
            yield got


def _iter_sentences_fallback(text: str) -> Iterator[tuple[str, list[tuple]]]:
    """Minimal CoNLL-U reader used when the `conllu` package is not installed.

    Handles exactly what we need: comments, blank-line separation, the 10-column
    layout, and the two irregular id forms.
    """
    rows: list[tuple] = []
    sent_id = None
    counter = 0

    def flush():
        # Mirrors the conllu-path guard: a comment-only block yields nothing,
        # so both parsers report identical seen/rejected counts.
        nonlocal rows, sent_id, counter
        if rows:
            sid = sent_id if sent_id else f"s{counter}"
            counter += 1
            out = (str(sid), rows)
            rows, sent_id = [], None
            return out
        rows, sent_id = [], None
        return None

    for line in text.splitlines():
        line = line.rstrip("\n\r")
        if not line.strip():
            got = flush()
            if got:
                yield got
            continue
        if line.startswith("#"):
            if "sent_id" in line and "=" in line:
                sent_id = line.split("=", 1)[1].strip()
            continue
        cols = line.split("\t")
        if len(cols) < 8:
            continue
        raw_id = cols[0]
        if "-" in raw_id:
            lo, hi = raw_id.split("-", 1)
            tid = (int(lo), "-", int(hi))
        elif "." in raw_id:
            lo, hi = raw_id.split(".", 1)
            tid = (int(lo), ".", int(hi))
        else:
            tid = int(raw_id)
        head = None if cols[6] in ("_", "") else int(cols[6])
        rows.append((tid, cols[1], cols[3], head, cols[7]))

    got = flush()
    if got:
        yield got


def load_treebank(
    path: Path,
    treebank_id: str,
    min_tokens: int = 3,
    max_tokens: int = 100,
) -> tuple[Iterator[Sentence], LoadStats]:
    """Stream a treebank file. Returns (generator, stats).

    The stats object fills in as the generator is consumed, so read it only
    after exhausting the iterator.
    """
    stats = LoadStats(treebank_id=treebank_id)
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    gen = parse_conllu_text(text, treebank_id, min_tokens, max_tokens, stats)
    return gen, stats
