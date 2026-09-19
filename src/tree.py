"""Sentence representation and tree-structural properties.

The `Sentence` dataclass is the single currency of this pipeline. It is
deliberately minimal and immutable-by-convention: every downstream module
(metrics, linearizers, analysis) reads it and never mutates it.

Index convention, fixed once here and relied on everywhere else:
  - tokens are indexed 0..n-1, contiguously
  - heads[i] is the 0-based index of token i's head, or -1 if i is the root
  - there is exactly one root
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

ROOT = -1


@dataclass(frozen=True)
class Sentence:
    """One validated dependency tree with a linear order."""

    tokens: tuple[str, ...]
    heads: np.ndarray            # int array, len n, -1 for root
    deprels: tuple[str, ...]
    upos: tuple[str, ...]
    treebank_id: str
    sent_id: str

    def __post_init__(self) -> None:
        n = len(self.tokens)
        if not isinstance(self.heads, np.ndarray):
            object.__setattr__(self, "heads", np.asarray(self.heads, dtype=np.int32))
        if self.heads.dtype != np.int32:
            object.__setattr__(self, "heads", self.heads.astype(np.int32))
        if len(self.heads) != n or len(self.deprels) != n or len(self.upos) != n:
            raise ValueError(
                f"Sentence field length mismatch in {self.sent_id}: "
                f"tokens={n} heads={len(self.heads)} "
                f"deprels={len(self.deprels)} upos={len(self.upos)}"
            )

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def root(self) -> int:
        """Index of the unique root token."""
        roots = np.flatnonzero(self.heads == ROOT)
        if len(roots) != 1:
            raise ValueError(f"{self.sent_id}: expected exactly 1 root, got {len(roots)}")
        return int(roots[0])

    def arcs(self) -> list[tuple[int, int]]:
        """All non-root arcs as (head_index, dependent_index) pairs."""
        return [(int(self.heads[d]), d) for d in range(self.n_tokens) if self.heads[d] != ROOT]

    def children(self) -> list[list[int]]:
        """children()[h] = sorted list of h's dependents, by linear position."""
        kids: list[list[int]] = [[] for _ in range(self.n_tokens)]
        for d in range(self.n_tokens):
            h = int(self.heads[d])
            if h != ROOT:
                kids[h].append(d)
        return kids

    def with_order(self, new_positions: Sequence[int]) -> "Sentence":
        """Relinearize: `new_positions[i]` is the new position of old token i.

        The tree is preserved exactly -- only linear order changes. This is the
        operation both random baselines are built from, so it is worth being
        precise: we permute the token array and remap every head index through
        the same permutation, so the arc multiset is bit-identical.
        """
        n = self.n_tokens
        new_positions = np.asarray(new_positions, dtype=np.int32)
        if sorted(new_positions.tolist()) != list(range(n)):
            raise ValueError(f"{self.sent_id}: new_positions is not a permutation of 0..n-1")

        # inverse[p] = the old index of whatever token now sits at position p
        inverse = np.empty(n, dtype=np.int32)
        inverse[new_positions] = np.arange(n, dtype=np.int32)

        new_heads = np.full(n, ROOT, dtype=np.int32)
        for p in range(n):
            old = int(inverse[p])
            old_head = int(self.heads[old])
            new_heads[p] = ROOT if old_head == ROOT else int(new_positions[old_head])

        return replace(
            self,
            tokens=tuple(self.tokens[int(inverse[p])] for p in range(n)),
            heads=new_heads,
            deprels=tuple(self.deprels[int(inverse[p])] for p in range(n)),
            upos=tuple(self.upos[int(inverse[p])] for p in range(n)),
        )


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------

def subtree_spans(sent: Sentence) -> list[list[int]]:
    """For each node, the sorted list of indices in its subtree (incl. itself).

    Iterative post-order so deep trees cannot blow the Python stack -- UD has
    some pathologically deep annotation in e.g. la_ittb.
    """
    kids = sent.children()
    order = _postorder(sent, kids)
    sub: list[list[int]] = [[] for _ in range(sent.n_tokens)]
    for node in order:
        acc = [node]
        for c in kids[node]:
            acc.extend(sub[c])
        sub[node] = sorted(acc)
    return sub


def _postorder(sent: Sentence, kids: list[list[int]]) -> list[int]:
    """Iterative post-order traversal from the root."""
    out: list[int] = []
    stack = [(sent.root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            out.append(node)
            continue
        stack.append((node, True))
        for c in kids[node]:
            stack.append((c, False))
    return out


def is_projective(sent: Sentence) -> bool:
    """True if no two arcs cross.

    Standard definition: arc (h,d) is projective iff every token strictly
    between h and d is a descendant of h. We test it via subtree contiguity,
    which is equivalent and O(n) after the spans are built: a tree is
    projective iff every subtree occupies a contiguous span of positions.
    """
    for span in subtree_spans(sent):
        if span[-1] - span[0] + 1 != len(span):
            return False
    return True


def n_nonprojective_arcs(sent: Sentence) -> int:
    """Count arcs that cross at least one other arc. Used for reporting only."""
    arcs = sent.arcs()
    count = 0
    for h, d in arcs:
        lo, hi = (h, d) if h < d else (d, h)
        crossed = False
        for h2, d2 in arcs:
            lo2, hi2 = (h2, d2) if h2 < d2 else (d2, h2)
            # crossing == exactly one endpoint of the other arc lies strictly inside
            inside2_lo = lo < lo2 < hi
            inside2_hi = lo < hi2 < hi
            if inside2_lo != inside2_hi:
                crossed = True
                break
        count += int(crossed)
    return count


def tree_depth(sent: Sentence) -> int:
    """Maximum root-to-leaf path length (root has depth 0)."""
    depths = np.full(sent.n_tokens, -1, dtype=np.int32)
    kids = sent.children()
    stack = [(sent.root, 0)]
    while stack:
        node, d = stack.pop()
        depths[node] = d
        for c in kids[node]:
            stack.append((c, d + 1))
    return int(depths.max())


def mean_arity(sent: Sentence) -> float:
    """Mean number of dependents over non-leaf nodes (0.0 if none)."""
    kids = sent.children()
    arities = [len(k) for k in kids if k]
    return float(np.mean(arities)) if arities else 0.0


def validate(sent_heads: np.ndarray, n: int) -> str | None:
    """Return a rejection reason string, or None if the tree is well-formed.

    Checked here rather than in the loader so the same rules apply to fixtures
    and to downloaded data, and so the reasons are enumerable for the manifest.
    """
    if len(sent_heads) != n:
        return "length_mismatch"
    roots = int(np.sum(sent_heads == ROOT))
    if roots == 0:
        return "no_root"
    if roots > 1:
        return "multiple_roots"
    if np.any((sent_heads >= n) | (sent_heads < ROOT)):
        return "head_out_of_range"
    if np.any(sent_heads == np.arange(n)):
        return "self_loop"

    # Cycle detection: walk each node to the root with a step budget.
    for start in range(n):
        node, steps = start, 0
        while sent_heads[node] != ROOT:
            node = int(sent_heads[node])
            steps += 1
            if steps > n:
                return "cycle"
    return None
