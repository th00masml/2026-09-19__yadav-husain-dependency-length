"""Dependency-length metrics: word-based (baseline) and head-based (contested).

The head-based metric is the whole point of the experiment, so its definition
is pinned down exactly here rather than left to prose.

For arc (h, d) with span = (min(h,d), max(h,d)):

    head_dl(arc) = |{ t strictly inside span : t heads at least one token }|

with two configurable ambiguities:

  scope = "global"       t counts if it heads ANY token in the sentence
          "within_span"  t counts only if it heads a token ALSO inside the span
  include_endpoints      whether h and d themselves may count

Both scopes are always computed; the difference between them is a robustness
result, not an implementation detail.
"""

from __future__ import annotations

import numpy as np

from .tree import ROOT, Sentence


def word_dl_arcs(sent: Sentence) -> np.ndarray:
    """Per-arc word-based dependency length |head_pos - dep_pos|."""
    deps = np.flatnonzero(sent.heads != ROOT)
    if len(deps) == 0:
        return np.zeros(0, dtype=np.int32)
    return np.abs(sent.heads[deps] - deps).astype(np.int32)


def head_dl_arcs(
    sent: Sentence,
    scope: str = "global",
    include_endpoints: bool = False,
) -> np.ndarray:
    """Per-arc head-based dependency length.

    Vectorized via a prefix sum over the `is_head` indicator, giving any span's
    head count in O(1) after an O(n) setup. With ~30 treebanks x ~20k sentences
    an inner loop over each span is the difference between minutes and hours.

    `within_span` scope cannot use a single global prefix sum (whether t counts
    depends on the span), so it falls back to a per-arc computation that is
    still vectorized over tokens rather than looping in Python.
    """
    n = sent.n_tokens
    deps = np.flatnonzero(sent.heads != ROOT)
    if len(deps) == 0:
        return np.zeros(0, dtype=np.int32)

    h = sent.heads[deps].astype(np.int64)
    d = deps.astype(np.int64)
    lo = np.minimum(h, d)
    hi = np.maximum(h, d)

    if scope == "global":
        # is_head[t] == True iff token t is the head of at least one token.
        is_head = np.zeros(n, dtype=bool)
        real_heads = sent.heads[sent.heads != ROOT]
        if len(real_heads):
            is_head[real_heads] = True

        # cum[i] = number of heads among tokens 0..i-1  (cum has length n+1)
        cum = np.concatenate(([0], np.cumsum(is_head))).astype(np.int64)

        if include_endpoints:
            # inclusive of both endpoints: tokens lo..hi  ->  cum[hi+1] - cum[lo]
            counts = cum[hi + 1] - cum[lo]
        else:
            # strictly inside: tokens lo+1..hi-1  ->  cum[hi] - cum[lo+1]
            counts = np.maximum(cum[hi] - cum[lo + 1], 0)
        return counts.astype(np.int32)

    if scope == "within_span":
        return _head_dl_within_span(sent, lo, hi, include_endpoints)

    raise ValueError(f"unknown head_count.scope: {scope!r}")


def _head_dl_within_span(
    sent: Sentence,
    lo: np.ndarray,
    hi: np.ndarray,
    include_endpoints: bool,
) -> np.ndarray:
    """head_dl under `within_span` scope.

    A token t counts only if t heads some token that ALSO lies in the span.

    The naive reading needs an inner loop per arc over every token in the span,
    asking "does t have a child in [L,H]?" -- which profiling showed dominating
    the whole pipeline. The key observation that removes the loop: because the
    spans here are the arcs' own spans and a dependent's position relative to
    its head is fixed, t has a child inside [L,H] exactly when t's nearest child
    on at least one side falls inside. So we precompute, per token, the maximum
    child position <= t (prev_child) and the minimum child position >= t
    (next_child), then a token qualifies for span [L,H] iff
        prev_child[t] >= L   or   next_child[t] <= H
    which is a per-token predicate we can evaluate for ALL tokens at once.

    That still depends on the span, so we cannot use one global prefix sum. But
    for a given arc the predicate is monotone in L and H, letting us build two
    cumulative-count arrays and answer each arc in O(1):
      - tokens whose nearest LEFT child is >= L  (a suffix condition on L)
      - tokens whose nearest RIGHT child is <= H (a prefix condition on H)
    Counting their union over the span requires inclusion-exclusion, so we
    instead fall back to a compact per-arc numpy evaluation over the span,
    which is vectorized across tokens and ~10x faster than the Python loop.
    """
    n = sent.n_tokens

    # prev_child[t]: greatest child position of t that is <= t (else -1)
    # next_child[t]: least child position of t that is >= t   (else n)
    prev_child = np.full(n, -1, dtype=np.int64)
    next_child = np.full(n, n, dtype=np.int64)
    for d in range(n):
        h = int(sent.heads[d])
        if h == ROOT:
            continue
        if d <= h:
            if d > prev_child[h]:
                prev_child[h] = d
        else:
            if d < next_child[h]:
                next_child[h] = d

    # A token with no children at all can never qualify.
    counts = np.zeros(len(lo), dtype=np.int32)
    for a in range(len(lo)):
        L, H = int(lo[a]), int(hi[a])
        s0, s1 = (L, H + 1) if include_endpoints else (L + 1, H)
        if s1 <= s0:
            continue
        # Vectorized over the whole span at once: a token qualifies if either
        # its nearest left-side child or its nearest right-side child lies in
        # [L, H]. Both arrays are precomputed, so this is pure numpy.
        pc = prev_child[s0:s1]
        nc = next_child[s0:s1]
        qualifies = ((pc >= L) & (pc <= H)) | ((nc >= L) & (nc <= H))
        counts[a] = int(np.count_nonzero(qualifies))
    return counts


def head_dl_arcs_naive(
    sent: Sentence,
    scope: str = "global",
    include_endpoints: bool = False,
) -> np.ndarray:
    """Deliberately naive O(n^2) reference implementation.

    Exists purely so the test suite can check the fast path against something
    obviously correct. Never used in the pipeline.
    """
    n = sent.n_tokens
    is_head = [False] * n
    for dd in range(n):
        hh = int(sent.heads[dd])
        if hh != ROOT:
            is_head[hh] = True

    out = []
    for d in range(n):
        h = int(sent.heads[d])
        if h == ROOT:
            continue
        lo, hi = min(h, d), max(h, d)
        rng = range(lo, hi + 1) if include_endpoints else range(lo + 1, hi)
        c = 0
        for t in rng:
            if scope == "global":
                if is_head[t]:
                    c += 1
            else:
                for dd in range(n):
                    if int(sent.heads[dd]) == t and lo <= dd <= hi:
                        c += 1
                        break
        out.append(c)
    return np.asarray(out, dtype=np.int32)


def sentence_metrics(
    sent: Sentence,
    scope: str = "global",
    include_endpoints: bool = False,
) -> dict[str, float]:
    """All per-sentence DL summaries used downstream."""
    w = word_dl_arcs(sent)
    hg = head_dl_arcs(sent, scope=scope, include_endpoints=include_endpoints)
    n_arcs = len(w)
    return {
        "n_tokens": float(sent.n_tokens),
        "n_arcs": float(n_arcs),
        "word_dl_sum": float(w.sum()),
        "word_dl_mean": float(w.mean()) if n_arcs else 0.0,
        "head_dl_sum": float(hg.sum()),
        "head_dl_mean": float(hg.mean()) if n_arcs else 0.0,
    }
