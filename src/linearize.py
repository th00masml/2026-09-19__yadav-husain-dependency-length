"""Random projective linearizations of a fixed dependency tree.

Both baselines preserve the tree EXACTLY (same arcs, same arity, same deprels)
and permute only linear order. That is what makes the real-vs-baseline contrast
a test of word order rather than of tree shape -- and it is why arity and depth
are constant within a matched pair, neutralizing the confound that would
otherwise sink the head-count metric.

Why projective baselines: an unconstrained permutation is far too weak a
control. It produces wildly crossing, unnatural orders that inflate every
effect size and would make even a broken pipeline look like it confirmed DLM.
Restricting to projective orders is the honest comparison.

Shared interface (the hedge for a later Hahn & Xu extension):

    linearize(sentence, order_fn, rng) -> Sentence
    order_fn(node, children, rng) -> reordered sequence placing `node` among
                                    its children (node marked by the sentinel)

B1 and B2 are two order_fns. A counterfactual-grammar sampler taking a
deprel->side parameter dict would be a third, and costs nothing to add.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Sequence

import numpy as np

from .tree import Sentence, subtree_spans

# Sentinel marking "the head itself" inside an ordering of head + subtree blocks.
HEAD = object()

OrderFn = Callable[[int, Sequence[int], np.random.Generator], list]


def derive_rng(master_seed: int, treebank_id: str, sent_id: str, sample_idx: int) -> np.random.Generator:
    """Deterministic per-(treebank, sentence, sample) RNG.

    Hashing the identifiers rather than counting means results are reproducible
    regardless of iteration order or parallelism -- a sentence gets the same
    baselines whether it is processed first or thousandth.
    """
    key = f"{master_seed}|{treebank_id}|{sent_id}|{sample_idx}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


# ---------------------------------------------------------------------------
# Order functions
# ---------------------------------------------------------------------------

def order_b1_uniform(node: int, children: Sequence[int], rng: np.random.Generator) -> list:
    """B1: uniform random placement of the head among its subtree blocks.

    The head and its k subtree blocks form k+1 units; we shuffle all of them
    uniformly. Every projective arrangement of this node's local family is
    equally likely.
    """
    units: list = [HEAD] + list(children)
    perm = rng.permutation(len(units))
    return [units[i] for i in perm]


def make_order_b2_arity_preserving(sent: Sentence) -> OrderFn:
    """B2: shuffle order but keep each node's left/right dependent COUNTS.

    This controls for branching direction. If a language puts 3 dependents
    before the head and 1 after, every B2 sample does too -- only *which*
    dependents, and their relative order, changes. Isolates pure ordering from
    head-directionality, which B1 conflates.
    """
    kids = sent.children()
    n_left = {
        h: sum(1 for c in kids[h] if c < h)
        for h in range(sent.n_tokens)
        if kids[h]
    }

    def order_fn(node: int, children: Sequence[int], rng: np.random.Generator) -> list:
        k_left = n_left.get(node, 0)
        k_left = min(k_left, len(children))
        shuffled = list(rng.permutation(list(children)))
        left = shuffled[:k_left]
        right = shuffled[k_left:]
        return left + [HEAD] + right

    return order_fn


# ---------------------------------------------------------------------------
# The linearizer
# ---------------------------------------------------------------------------

def linearize(sent: Sentence, order_fn: OrderFn, rng: np.random.Generator) -> Sentence:
    """Produce a projective relinearization of `sent` using `order_fn`.

    Builds the new order by an iterative pre-order walk: each node emits its
    subtree as [blocks and head in the order chosen by order_fn], recursively.
    Because every subtree is emitted as one contiguous run, the result is
    projective by construction.
    """
    kids = sent.children()

    # new_order[i] = old token index sitting at new position i
    new_order: list[int] = []

    # Explicit stack of work items to avoid recursion depth limits.
    # An item is either ("emit", node) meaning "place this node itself" or
    # ("expand", node) meaning "lay out node's family".
    stack: list[tuple[str, int]] = [("expand", sent.root)]

    while stack:
        kind, node = stack.pop()
        if kind == "emit":
            new_order.append(node)
            continue

        children = kids[node]
        if not children:
            new_order.append(node)
            continue

        units = order_fn(node, children, rng)

        # Push in reverse so the leftmost unit is processed first.
        for unit in reversed(units):
            if unit is HEAD:
                stack.append(("emit", node))
            else:
                stack.append(("expand", int(unit)))

    if len(new_order) != sent.n_tokens:
        raise AssertionError(
            f"{sent.sent_id}: linearizer emitted {len(new_order)} of {sent.n_tokens} tokens"
        )

    # new_order maps position -> old index; with_order wants old index -> position.
    new_positions = np.empty(sent.n_tokens, dtype=np.int32)
    for pos, old in enumerate(new_order):
        new_positions[old] = pos

    return sent.with_order(new_positions)


def make_order_lambda_interpolated(sent: Sentence, lam: float) -> OrderFn:
    """A baseline of TUNABLE difficulty: keep the real local order w.p. `lam`.

    At every node independently, the real arrangement of {head, subtree blocks}
    is kept with probability `lam` and replaced by a uniform shuffle otherwise.
    So `lam = 0` reproduces B1_uniform exactly and `lam = 1` reproduces the real
    sentence exactly, with a continuum in between.

    This exists to turn a correlational claim into a controlled one. The run
    found that rank and magnitude effect measures disagree completely against
    the weak B1 baseline and barely at all against the strong B2 baseline, and
    attributed that to a CEILING EFFECT: when a baseline is easy to beat, a rank
    statistic saturates near its maximum and can no longer separate two metrics,
    while a magnitude statistic still has room to move. Comparing two fixed
    baselines cannot establish that, because B1 and B2 differ in more than
    difficulty -- B2 also preserves branching direction.

    Sweeping `lam` holds the construction fixed and varies ONLY difficulty, so
    if the divergence tracks `lam` monotonically the ceiling account is
    demonstrated rather than merely consistent with the data.

    Interpolating per node rather than per sentence matters: mixing whole
    sentences would produce a bimodal mixture of "real" and "fully scrambled"
    items, whose rank statistic is an average of two extremes rather than a
    genuinely intermediate difficulty.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    def order_fn(node: int, children: Sequence[int], rng: np.random.Generator) -> list:
        if lam > 0.0 and rng.random() < lam:
            # Real local arrangement: head and blocks in their attested order.
            # A child block stands at the position of the child's own token.
            units = sorted([(node, HEAD)] + [(int(c), int(c)) for c in children])
            return [u for _, u in units]
        return order_b1_uniform(node, children, rng)

    return order_fn


def sample_baselines(
    sent: Sentence,
    kind: str,
    n_samples: int,
    master_seed: int,
) -> list[Sentence]:
    """n_samples random linearizations of `sent` under baseline `kind`.

    `kind` may also be "Lam_<x>" (e.g. "Lam_0.25") to request the tunable
    baseline described in `make_order_lambda_interpolated`. That form is used by
    the ceiling-effect sweep, not by the pre-registered hypotheses.
    """
    if kind == "B1_uniform":
        order_fn: OrderFn = order_b1_uniform
    elif kind == "B2_arity_preserving":
        order_fn = make_order_b2_arity_preserving(sent)
    elif kind.startswith("Lam_"):
        order_fn = make_order_lambda_interpolated(sent, float(kind[4:]))
    else:
        raise ValueError(f"unknown baseline kind: {kind!r}")

    out = []
    for i in range(n_samples):
        rng = derive_rng(master_seed, sent.treebank_id, sent.sent_id, i)
        out.append(linearize(sent, order_fn, rng))
    return out


def arc_signature(sent: Sentence) -> tuple:
    """Multiset of (head_deprel, dep_deprel) arcs -- invariant under relinearization.

    Used by the tests to prove the linearizers preserve the tree rather than
    quietly rewiring it.
    """
    sig = []
    for h, d in sent.arcs():
        sig.append((sent.deprels[h], sent.deprels[d], sent.upos[h], sent.upos[d]))
    return tuple(sorted(sig))


def arity_signature(sent: Sentence) -> tuple:
    """Sorted multiset of node arities -- also invariant."""
    return tuple(sorted(len(k) for k in sent.children()))


def side_signature(sent: Sentence) -> tuple:
    """Sorted multiset of (n_left, n_right) per node -- invariant under B2 only."""
    kids = sent.children()
    out = []
    for h in range(sent.n_tokens):
        if kids[h]:
            left = sum(1 for c in kids[h] if c < h)
            out.append((left, len(kids[h]) - left))
    return tuple(sorted(out))


def check_projective(sent: Sentence) -> bool:
    """Convenience re-export used in tests."""
    for span in subtree_spans(sent):
        if span[-1] - span[0] + 1 != len(span):
            return False
    return True
