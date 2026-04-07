"""
structure.py — Dot-bracket parser and loop decomposition.

Responsibilities:
  1. Parse dot-bracket notation into a Structure object
  2. Validate: balanced parentheses, no pseudoknots
  3. Decompose a structure into its constituent loops

Reference: Standard RNA secondary structure representation.
"""

from ribo_switch.types import (
    Structure, LoopType,
    HairpinLoop, StackLoop, InteriorLoop, BulgeLoop, MultiLoop, ExternalLoop,
)

def parse_dot_bracket(db: str) -> Structure:
    """Parse a dot-bracket string into a Structure.

    Args:
        db: Dot-bracket notation string using '.', '(', ')'.
             Example: "(((...)))"

    Returns:
        Structure with pair_table and sorted pairs list.

    Raises:
        ValueError: If the string contains invalid characters,
                    has unbalanced parentheses, or is empty.
    """
    if not db:
        raise ValueError("Empty dot-bracket string")

    n = len(db)
    pair_table: list[int] = [-1] * n  # -1 means unpaired
    stack: list[int] = []

    for i, ch in enumerate(db):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if not stack:
                raise ValueError(
                    f"Unbalanced parentheses: extra ')' at position {i}"
                )
            j = stack.pop()
            pair_table[j] = i
            pair_table[i] = j
        elif ch == '.':
            pass  # unpaired
        else:
            raise ValueError(
                f"Invalid character '{ch}' at position {i}. "
                f"Only '.', '(', ')' are allowed."
            )

    if stack:
        raise ValueError(
            f"Unbalanced parentheses: {len(stack)} unmatched '(' "
            f"(first at position {stack[0]})"
        )

    # Collect pairs as (i, j) with i < j, sorted by opening position
    pairs = sorted(
        [(i, pair_table[i]) for i in range(n) if pair_table[i] > i]
    )

    return Structure(length=n, pair_table=pair_table, pairs=pairs)


def decompose_loops(structure: Structure) -> list[LoopType]:
    """Decompose a structure into its constituent loops.

    Algorithm:
        For each base pair (i, j) considered as a closing pair, find
        all enclosed pairs at the next depth level.  The number of
        enclosed pairs determines the loop type:
          - 0 enclosed  →  hairpin
          - 1 enclosed (p, q):
              * p == i+1 and q == j-1  →  stacking pair
              * one side has no unpaired  →  bulge
              * both sides have unpaired  →  interior loop
          - 2+ enclosed  →  multiloop

        Pairs not enclosed by anything contribute to the external loop.

    Returns:
        List of LoopType objects (one per loop in the structure).
    """
    pt = structure.pair_table
    n = structure.length
    loops: list[LoopType] = []

    # ── Find enclosed pairs for each closing pair ──
    for i, j in structure.pairs:
        enclosed = _find_enclosed_pairs(pt, i, j)
        loop = _classify_and_build_loop(pt, i, j, enclosed)
        loops.append(loop)

    # ── External loop ──
    ext = _build_external_loop(pt, n, structure.pairs)
    loops.append(ext)

    return loops


def _find_enclosed_pairs(
    pair_table: list[int], i: int, j: int
) -> list[tuple[int, int]]:
    """Find all base pairs directly enclosed by pair (i, j).

    "Directly enclosed" means pairs (p, q) where i < p < q < j and
    there is no other pair (a, b) with i < a < p < q < b < j that
    also encloses (p, q).  In other words, these are the pairs at
    depth exactly one level deeper than (i, j).

    Algorithm:
        Walk from i+1 to j-1.  Whenever we hit a position k that is
        paired to some l > k (i.e., k is an opening bracket), record
        (k, l) as an enclosed pair and skip ahead to l+1.  Whenever
        we hit an unpaired position or a position paired to something
        before k, just move forward.
    """
    enclosed: list[tuple[int, int]] = []
    k = i + 1
    while k < j:
        partner = pair_table[k]
        if partner > k:  # k opens a pair, partner closes it
            enclosed.append((k, partner))
            k = partner + 1  # skip the entire enclosed region
        else:
            k += 1
    return enclosed


def _classify_and_build_loop(
    pair_table: list[int],
    i: int, j: int,
    enclosed: list[tuple[int, int]],
) -> LoopType:
    """Given a closing pair (i, j) and its enclosed pairs, build the loop."""

    if len(enclosed) == 0:
        # ── Hairpin ──
        unpaired = list(range(i + 1, j))
        return HairpinLoop(closing_pair=(i, j), unpaired=unpaired)

    if len(enclosed) == 1:
        p, q = enclosed[0]

        # Unpaired positions on the left side (between i and p)
        left_unp = list(range(i + 1, p))
        # Unpaired positions on the right side (between q and j)
        right_unp = list(range(q + 1, j))

        n_left = len(left_unp)
        n_right = len(right_unp)

        if n_left == 0 and n_right == 0:
            # ── Stacking pair ──
            return StackLoop(outer_pair=(i, j), inner_pair=(p, q))

        if n_left == 0:
            # ── Bulge on the right side ──
            return BulgeLoop(
                outer_pair=(i, j), inner_pair=(p, q),
                unpaired=right_unp, side="right",
            )

        if n_right == 0:
            # ── Bulge on the left side ──
            return BulgeLoop(
                outer_pair=(i, j), inner_pair=(p, q),
                unpaired=left_unp, side="left",
            )

        # ── Interior loop ──
        return InteriorLoop(
            outer_pair=(i, j), inner_pair=(p, q),
            left_unpaired=left_unp, right_unpaired=right_unp,
        )

    # ── Multiloop (2+ enclosed pairs) ──
    # Collect all unpaired positions inside the multiloop
    unpaired: list[int] = []
    prev_end = i + 1
    for p, q in enclosed:
        unpaired.extend(range(prev_end, p))
        prev_end = q + 1
    unpaired.extend(range(prev_end, j))

    return MultiLoop(
        closing_pair=(i, j),
        branches=enclosed,
        unpaired=unpaired,
    )


def _build_external_loop(
    pair_table: list[int],
    n: int,
    pairs: list[tuple[int, int]],
) -> ExternalLoop:
    """Build the external loop from top-level pairs and unpaired positions.

    The external loop contains:
      - All base pairs that are not enclosed by any other pair (top-level pairs)
      - All unpaired positions not inside any pair
    """
    # Find top-level pairs: pairs (i, j) where no other pair encloses them
    top_level: list[tuple[int, int]] = []
    for i, j in pairs:
        # Check if any other pair encloses (i, j)
        is_top = pair_table[i] == j  # it's a valid pair
        # A pair is top-level if there's no position k < i with pair_table[k] > j
        enclosed_by_other = False
        for a, b in pairs:
            if a < i and b > j:
                enclosed_by_other = True
                break
        if not enclosed_by_other:
            top_level.append((i, j))

    # Unpaired positions not inside any top-level pair
    unpaired: list[int] = []
    pos = 0
    for i, j in top_level:
        # Positions before this pair
        for k in range(pos, i):
            if pair_table[k] == -1:
                unpaired.append(k)
        pos = j + 1
    # Positions after the last pair
    for k in range(pos, n):
        if pair_table[k] == -1:
            unpaired.append(k)

    return ExternalLoop(unpaired=unpaired, closing_pairs=top_level)
