"""
topology.py — Structure pair topology analysis for bicompatible sampling.

Given two RNA secondary structures S1 and S2, computes:
  1. Loop decomposition of each structure
  2. Position-to-loop mapping for each structure
  3. Exposed positions — nucleotides in loops of BOTH structures
  4. κ (kappa) — count of exposed positions (complexity parameter)
  5. DP traversal order for the bicompatible partition function

The exposed vertices are the key concept from Huang, Barrett & Reidys (2021):
they are the positions where the DP must enumerate all 4^κ nucleotide assignments
because those positions affect energy calculations in both structures.

Reference: Huang, Barrett & Reidys (2021)
"""

from dataclasses import dataclass, field

from ribo_switch.types import (
    Structure, LoopType,
    HairpinLoop, StackLoop, InteriorLoop, BulgeLoop, MultiLoop, ExternalLoop,
)
from ribo_switch.structure import decompose_loops


@dataclass
class DPNode:
    """A node in the DP traversal order.

    Attributes:
        kind: "loop1" (only S1 loops), "loop2" (only S2 loops), or "joint"
        loop_indices: indices into loops1 or loops2 list
        positions: nucleotide positions involved in this node
        exposed: exposed positions that must be enumerated at this node
    """
    kind: str
    loop_indices: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    exposed: list[int] = field(default_factory=list)


@dataclass
class StructurePairTopology:
    """Result of topology analysis for a pair of structures.

    Attributes:
        structure1, structure2: The input structures.
        loops1, loops2: Loop decomposition of each structure.
        pos_to_loops1, pos_to_loops2: For each position, which loop indices contain it.
        exposed_positions: Boolean mask — True if position is in loops of both structures.
        kappa: Total number of exposed positions.
        dp_order: Traversal order for the bicompatible DP.
    """
    structure1: Structure
    structure2: Structure
    loops1: list[LoopType]
    loops2: list[LoopType]
    pos_to_loops1: list[set[int]]
    pos_to_loops2: list[set[int]]
    exposed_positions: list[bool]
    kappa: int
    dp_order: list[DPNode]


# Maximum κ before we refuse to compute (complexity is O(4^κ × n))
MAX_KAPPA = 20


def analyze_topology(s1: Structure, s2: Structure) -> StructurePairTopology:
    """Analyze the topology of two structures of the same length.

    Args:
        s1: First structure (e.g., ON-state).
        s2: Second structure (e.g., OFF-state).

    Returns:
        StructurePairTopology with all computed fields.

    Raises:
        ValueError: If structures have different lengths or κ > MAX_KAPPA.
    """
    if s1.length != s2.length:
        raise ValueError(
            f"Structure lengths differ: {s1.length} vs {s2.length}"
        )

    n = s1.length
    loops1 = decompose_loops(s1)
    loops2 = decompose_loops(s2)

    # Map each position to the set of loop indices it belongs to
    pos_to_loops1 = _build_position_map(loops1, n)
    pos_to_loops2 = _build_position_map(loops2, n)

    # A position is "exposed" if it participates in loops from BOTH structures
    # More precisely: it's in at least one loop of S1 AND at least one loop of S2,
    # AND the loops it belongs to in S1 differ from the loops in S2 (different
    # pairing context). For the initial implementation, we use the simpler
    # criterion: a position is exposed if its pairing status differs between
    # the two structures (paired in one, unpaired in the other, or paired
    # to different partners).
    exposed = _find_exposed_positions(s1, s2, n)
    kappa = sum(exposed)

    if kappa > MAX_KAPPA:
        raise ValueError(
            f"κ = {kappa} exceeds maximum ({MAX_KAPPA}). "
            f"The DP complexity O(4^κ × n) would be intractable. "
            f"These two structures are too different for bicompatible sampling."
        )

    # Build DP traversal order
    dp_order = _build_dp_order(
        loops1, loops2, pos_to_loops1, pos_to_loops2, exposed, n
    )

    return StructurePairTopology(
        structure1=s1,
        structure2=s2,
        loops1=loops1,
        loops2=loops2,
        pos_to_loops1=pos_to_loops1,
        pos_to_loops2=pos_to_loops2,
        exposed_positions=exposed,
        kappa=kappa,
        dp_order=dp_order,
    )


def _get_loop_positions(loop: LoopType) -> set[int]:
    """Get all nucleotide positions involved in a loop.

    This includes both unpaired positions AND the closing pair positions.
    """
    positions: set[int] = set()

    if isinstance(loop, HairpinLoop):
        positions.add(loop.closing_pair[0])
        positions.add(loop.closing_pair[1])
        positions.update(loop.unpaired)

    elif isinstance(loop, StackLoop):
        positions.add(loop.outer_pair[0])
        positions.add(loop.outer_pair[1])
        positions.add(loop.inner_pair[0])
        positions.add(loop.inner_pair[1])

    elif isinstance(loop, InteriorLoop):
        positions.add(loop.outer_pair[0])
        positions.add(loop.outer_pair[1])
        positions.add(loop.inner_pair[0])
        positions.add(loop.inner_pair[1])
        positions.update(loop.left_unpaired)
        positions.update(loop.right_unpaired)

    elif isinstance(loop, BulgeLoop):
        positions.add(loop.outer_pair[0])
        positions.add(loop.outer_pair[1])
        positions.add(loop.inner_pair[0])
        positions.add(loop.inner_pair[1])
        positions.update(loop.unpaired)

    elif isinstance(loop, MultiLoop):
        positions.add(loop.closing_pair[0])
        positions.add(loop.closing_pair[1])
        for p, q in loop.branches:
            positions.add(p)
            positions.add(q)
        positions.update(loop.unpaired)

    elif isinstance(loop, ExternalLoop):
        for p, q in loop.closing_pairs:
            positions.add(p)
            positions.add(q)
        positions.update(loop.unpaired)

    return positions


def _build_position_map(
    loops: list[LoopType], n: int
) -> list[set[int]]:
    """Map each position to the set of loop indices it belongs to."""
    pos_to_loops: list[set[int]] = [set() for _ in range(n)]

    for loop_idx, loop in enumerate(loops):
        for pos in _get_loop_positions(loop):
            pos_to_loops[pos].add(loop_idx)

    return pos_to_loops


def _find_exposed_positions(
    s1: Structure, s2: Structure, n: int
) -> list[bool]:
    """Find positions where the pairing context differs between structures.

    A position is exposed if:
      - It is paired in S1 but unpaired in S2 (or vice versa)
      - It is paired in both but to different partners
    These are the positions where nucleotide assignment in one structure's
    loop context constrains the other structure's energy calculation.
    """
    exposed = [False] * n

    for i in range(n):
        p1 = s1.pair_table[i]
        p2 = s2.pair_table[i]

        if p1 != p2:
            # Different pairing status → exposed
            exposed[i] = True

    return exposed


def _build_dp_order(
    loops1: list[LoopType],
    loops2: list[LoopType],
    pos_to_loops1: list[set[int]],
    pos_to_loops2: list[set[int]],
    exposed: list[bool],
    n: int,
) -> list[DPNode]:
    """Build a DP traversal order for the bicompatible partition function.

    Strategy:
      1. Find groups of loops that share exposed positions (connected components)
      2. Loops that share no exposed positions with the other structure
         can be processed independently
      3. Loops connected through exposed positions form "joint" nodes
         where we must enumerate nucleotide assignments

    For the initial implementation, we use a simpler linear ordering:
      - Process all loops of S1 in order, then all loops of S2
      - Mark exposed positions at each node
    """
    dp_order: list[DPNode] = []

    # Collect exposed positions per loop
    for idx, loop in enumerate(loops1):
        loop_pos = _get_loop_positions(loop)
        loop_exposed = sorted(p for p in loop_pos if exposed[p])
        dp_order.append(DPNode(
            kind="loop1",
            loop_indices=[idx],
            positions=sorted(loop_pos),
            exposed=loop_exposed,
        ))

    for idx, loop in enumerate(loops2):
        loop_pos = _get_loop_positions(loop)
        loop_exposed = sorted(p for p in loop_pos if exposed[p])
        dp_order.append(DPNode(
            kind="loop2",
            loop_indices=[idx],
            positions=sorted(loop_pos),
            exposed=loop_exposed,
        ))

    return dp_order


def get_compatible_bases(
    s1: Structure, s2: Structure, pos: int
) -> list[tuple[int, ...]]:
    """Get all base assignments compatible with both structures at a position.

    Returns list of valid base values (0=A, 1=C, 2=G, 3=U) for this position,
    considering pairing constraints from both structures.

    For paired position (i,j): bases[i] and bases[j] must form a canonical pair.
    If a position is paired in both structures to different partners, the
    intersection of allowed base sets may be restricted.
    """
    from ribo_switch.types import Base, CANONICAL_PAIRS

    p1 = s1.pair_table[pos]
    p2 = s2.pair_table[pos]

    if p1 == -1 and p2 == -1:
        # Unpaired in both → any base
        return [0, 1, 2, 3]

    # Collect constraints
    allowed = {0, 1, 2, 3}  # start with all bases

    if p1 != -1:
        # Paired in S1 — base at pos must be able to pair with base at p1
        if pos < p1:
            # pos is 5' end of pair
            valid = {b.value for b, bp in CANONICAL_PAIRS if (b, bp) in CANONICAL_PAIRS}
            allowed &= {b.value for b in Base}  # all bases can be 5' in some pair
        else:
            allowed &= {b.value for b in Base}

    if p2 != -1:
        # Similarly for S2
        pass  # constraint applied during sampling, not here

    return sorted(allowed)


def get_pair_compatible_assignments(
    s1: Structure, s2: Structure
) -> dict[int, set[int]]:
    """For each position, find which base values are compatible with both structures.

    A base value at position i is compatible if:
      - For every pair (i,j) in S1: (base[i], base[j]) can form a canonical pair
      - For every pair (i,j) in S2: (base[i], base[j]) can form a canonical pair

    Since we don't know base[j] yet, we return the set of bases that CAN
    participate in at least one canonical pair on the correct side.

    Returns:
        Dict mapping position → set of allowed base values.
    """
    from ribo_switch.types import Base, CANONICAL_PAIRS

    n = s1.length
    result: dict[int, set[int]] = {}

    # Bases that can be the 5' partner of a canonical pair
    can_be_5prime = {b.value for b, _ in CANONICAL_PAIRS}
    # Bases that can be the 3' partner
    can_be_3prime = {b.value for _, b in CANONICAL_PAIRS}

    for i in range(n):
        allowed = {0, 1, 2, 3}  # A, C, G, U

        for struct in (s1, s2):
            partner = struct.pair_table[i]
            if partner != -1:
                if i < partner:
                    allowed &= can_be_5prime
                else:
                    allowed &= can_be_3prime

        result[i] = allowed

    return result
