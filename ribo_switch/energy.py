"""
energy.py — Free energy evaluation for RNA secondary structures.

Given a sequence and a structure, compute ΔG (free energy change) by
summing the nearest-neighbor energy contributions from every loop.

All energies are in units of 0.01 kcal/mol (integer arithmetic).

Reference: Turner & Mathews (2010), Mathews et al. (2004)
"""

import math
from ribo_switch.types import (
    Base, Energy, Sequence, Structure,
    HairpinLoop, StackLoop, InteriorLoop, BulgeLoop, MultiLoop, ExternalLoop,
    LoopType, CANONICAL_PAIRS,
)
from ribo_switch.structure import decompose_loops
from ribo_switch.turner import TurnerParams, INF


def eval_energy(
    seq: Sequence, structure: Structure, params: TurnerParams
) -> Energy:
    """Compute total free energy of seq folded into structure.

    Decomposes the structure into loops and sums the energy
    contribution of each loop.

    Args:
        seq: RNA sequence.
        structure: RNA secondary structure (same length as seq).
        params: Turner 2004 energy parameters.

    Returns:
        Total free energy in 0.01 kcal/mol.
    """
    if len(seq) != structure.length:
        raise ValueError(
            f"Sequence length ({len(seq)}) != structure length ({structure.length})"
        )

    loops = decompose_loops(structure)
    total = 0

    for loop in loops:
        total += _loop_energy(seq, loop, params)

    return total


def _loop_energy(
    seq: Sequence, loop: LoopType, params: TurnerParams
) -> Energy:
    """Dispatch to the appropriate energy function for the loop type."""
    if isinstance(loop, HairpinLoop):
        return hairpin_energy(seq, loop.closing_pair, params)
    elif isinstance(loop, StackLoop):
        return stack_energy(seq, loop.outer_pair, loop.inner_pair, params)
    elif isinstance(loop, InteriorLoop):
        return interior_energy(
            seq, loop.outer_pair, loop.inner_pair,
            len(loop.left_unpaired), len(loop.right_unpaired), params,
        )
    elif isinstance(loop, BulgeLoop):
        return bulge_energy(
            seq, loop.outer_pair, loop.inner_pair,
            len(loop.unpaired), params,
        )
    elif isinstance(loop, MultiLoop):
        return multiloop_energy(
            seq, loop.closing_pair, loop.branches,
            len(loop.unpaired), params,
        )
    elif isinstance(loop, ExternalLoop):
        return external_energy(seq, loop, params)
    else:
        raise TypeError(f"Unknown loop type: {type(loop)}")

def _pair_index(seq: Sequence, i: int, j: int) -> int:
    """Return the BasePair index (0-5) for the pair at positions (i, j).

    AU=0, UA=1, CG=2, GC=3, GU=4, UG=5

    Raises ValueError if the pair is non-canonical.
    """
    bi, bj = seq.bases[i], seq.bases[j]
    _PAIR_MAP = {
        (Base.A, Base.U): 0,
        (Base.U, Base.A): 1,
        (Base.C, Base.G): 2,
        (Base.G, Base.C): 3,
        (Base.G, Base.U): 4,
        (Base.U, Base.G): 5,
    }
    idx = _PAIR_MAP.get((bi, bj))
    if idx is None:
        raise ValueError(
            f"Non-canonical pair ({bi.name}, {bj.name}) at ({i}, {j})"
        )
    return idx

def hairpin_energy(
    seq: Sequence, closing: tuple[int, int], params: TurnerParams
) -> Energy:
    """Energy of a hairpin loop closed by pair (i, j).

    Rules (from NNDB):
      1. size < 3 → impossible (return INF)
      2. Initiation by size (or Jacobson-Stockmayer extrapolation if > 30)
      3. size == 3: check triloop bonus, add terminal AU penalty, NO mismatch
      4. size == 4: check tetraloop bonus
      5. size >= 4: add terminal mismatch
      6. Special: UU/GA first mismatch bonus, GG first mismatch bonus
      7. Special: GU closure preceded by GG
      8. All-C loop penalty
    """
    i, j = closing
    size = j - i - 1

    if size < 3:
        return INF

    # 1. Initiation
    if size <= 30:
        energy = int(params.hairpin_init[size])
    else:
        # Jacobson-Stockmayer extrapolation
        energy = int(params.hairpin_init[30]) + int(
            round(params.loop_extrapolation_coeff * 100 * math.log(size / 30.0))
        )

    pair_idx = _pair_index(seq, i, j)

    # 2. Triloop check (size == 3)
    if size == 3:
        loop_seq = ''.join(seq.bases[k].name for k in range(i, j + 1))
        bonus = params.hairpin_triloop.get(loop_seq, 0)
        if bonus:
            energy = bonus  # special triloops replace initiation
        # Terminal AU/GU penalty for size-3 hairpins
        if pair_idx in (0, 1, 4, 5):  # AU, UA, GU, UG
            energy += params.terminal_au_penalty
        # All-C penalty for triloops
        if all(seq.bases[k] == Base.C for k in range(i + 1, j)):
            energy += params.hairpin_c3
        return energy

    # 3. Tetraloop check (size == 4)
    if size == 4:
        loop_seq = ''.join(seq.bases[k].name for k in range(i, j + 1))
        bonus = params.hairpin_tetraloop.get(loop_seq, 0)
        if bonus:
            energy = bonus  # special tetraloops replace initiation

    # 4. Terminal mismatch (size >= 4)
    b5 = seq.bases[i + 1].value  # base after 5' closing base
    b3 = seq.bases[j - 1].value  # base before 3' closing base
    energy += int(params.hairpin_mismatch[pair_idx][b5][b3])

    # 5. UU or GA first mismatch bonus
    first_mm = (seq.bases[i + 1], seq.bases[j - 1])
    if first_mm in ((Base.U, Base.U), (Base.G, Base.A)):
        energy += params.hairpin_uu_ga_bonus

    # 6. GG first mismatch bonus
    if first_mm == (Base.G, Base.G):
        energy += params.hairpin_gg_bonus

    # 7. Special GU closure: GU pair (not UG) preceded by two Gs
    if pair_idx == 4:  # GU pair
        if (i >= 2 and seq.bases[i - 1] == Base.G
                and seq.bases[i - 2] == Base.G):
            energy += params.hairpin_special_gu

    # 8. All-C loop penalty (size > 3)
    if all(seq.bases[k] == Base.C for k in range(i + 1, j)):
        energy += params.hairpin_c_slope * size + params.hairpin_c_intercept

    return energy


def stack_energy(
    seq: Sequence,
    outer: tuple[int, int],
    inner: tuple[int, int],
    params: TurnerParams,
) -> Energy:
    """Energy of a stacking pair: outer (i,j) directly enclosing inner (p,q)."""
    outer_idx = _pair_index(seq, outer[0], outer[1])
    inner_idx = _pair_index(seq, inner[0], inner[1])
    return int(params.stack[outer_idx][inner_idx])


def interior_energy(
    seq: Sequence,
    outer: tuple[int, int],
    inner: tuple[int, int],
    n_left: int,
    n_right: int,
    params: TurnerParams,
) -> Energy:
    """Energy of an interior loop with n_left and n_right unpaired bases.

    Special cases:
      - 1×1: use interior_1x1 lookup table
      - 1×2, 2×2: use general formula (full tables not yet implemented)
      - General: initiation + Ninio asymmetry + terminal mismatches
    """
    i, j = outer
    p, q = inner
    outer_idx = _pair_index(seq, i, j)
    inner_idx = _pair_index(seq, p, q)
    total_unpaired = n_left + n_right

    # 1×1 special case
    if n_left == 1 and n_right == 1:
        mm5 = seq.bases[i + 1].value  # mismatch on 5' side
        mm3 = seq.bases[j - 1].value  # mismatch on 3' side
        val = int(params.interior_1x1[outer_idx][inner_idx][mm5][mm3])
        if val < INF:
            return val
        # Fall through to general if INF (shouldn't happen with full table)

    # General interior loop formula
    if total_unpaired > 30:
        energy = int(params.interior_init[30]) + int(
            round(params.loop_extrapolation_coeff * 100
                  * math.log(total_unpaired / 30.0))
        )
    else:
        energy = int(params.interior_init[total_unpaired])

    # Ninio asymmetry penalty
    asymmetry = abs(n_left - n_right)
    ninio = min(params.ninio_max, params.ninio_m * asymmetry)
    energy += ninio

    # Terminal mismatches on both sides (only for loops > 1×1)
    if not (n_left == 1 and n_right == 1):
        # Outer side mismatch
        b5_outer = seq.bases[i + 1].value
        b3_outer = seq.bases[j - 1].value
        energy += int(params.interior_mismatch[outer_idx][b5_outer][b3_outer])

        # Inner side mismatch (reversed direction)
        b5_inner = seq.bases[q + 1].value if q + 1 < len(seq) else 0
        b3_inner = seq.bases[p - 1].value if p - 1 >= 0 else 0
        energy += int(params.interior_mismatch[inner_idx][b3_inner][b5_inner])

    # Terminal AU/GU penalty for both pairs
    if outer_idx in (0, 1, 4, 5):
        energy += params.terminal_au_penalty
    if inner_idx in (0, 1, 4, 5):
        energy += params.terminal_au_penalty

    return energy


def bulge_energy(
    seq: Sequence,
    outer: tuple[int, int],
    inner: tuple[int, int],
    n_unpaired: int,
    params: TurnerParams,
) -> Energy:
    """Energy of a bulge loop with n_unpaired unpaired bases.

    Rules:
      - Initiation by size (or extrapolation)
      - Size 1: add stacking energy of flanking pairs
      - Terminal AU/GU penalty on both closing pairs
    """
    if n_unpaired > 30:
        energy = int(params.bulge_init[30]) + int(
            round(params.loop_extrapolation_coeff * 100
                  * math.log(n_unpaired / 30.0))
        )
    else:
        energy = int(params.bulge_init[n_unpaired])

    outer_idx = _pair_index(seq, outer[0], outer[1])
    inner_idx = _pair_index(seq, inner[0], inner[1])

    # Size-1 bulge: stacking contribution
    if n_unpaired == 1:
        energy += int(params.stack[outer_idx][inner_idx])

    # Terminal AU/GU penalties
    if outer_idx in (0, 1, 4, 5):
        energy += params.terminal_au_penalty
    if inner_idx in (0, 1, 4, 5):
        energy += params.terminal_au_penalty

    return energy


def multiloop_energy(
    seq: Sequence,
    closing: tuple[int, int],
    branches: list[tuple[int, int]],
    n_unpaired: int,
    params: TurnerParams,
) -> Energy:
    """Energy of a multiloop.

    E = offset + per_branch × n_branches + per_unpaired × n_unpaired
        + dangling end contributions
        + terminal AU/GU penalties
    """
    n_branches = len(branches)
    energy = (params.ml_offset
              + params.ml_per_branch * n_branches
              + params.ml_per_unpaired * n_unpaired)

    # Terminal AU/GU penalty for closing pair
    closing_idx = _pair_index(seq, closing[0], closing[1])
    if closing_idx in (0, 1, 4, 5):
        energy += params.terminal_au_penalty

    # Terminal AU/GU penalty for each branch pair
    for p, q in branches:
        branch_idx = _pair_index(seq, p, q)
        if branch_idx in (0, 1, 4, 5):
            energy += params.terminal_au_penalty

    return energy


def external_energy(
    seq: Sequence,
    loop: ExternalLoop,
    params: TurnerParams,
) -> Energy:
    """Energy of the external loop (unpaired regions + dangling ends).

    Only terminal AU/GU penalties for top-level pairs, plus dangling ends.
    """
    energy = 0

    for i, j in loop.closing_pairs:
        pair_idx = _pair_index(seq, i, j)

        # Terminal AU/GU penalty
        if pair_idx in (0, 1, 4, 5):
            energy += params.terminal_au_penalty

        # 5' dangling end (base before the pair, if it exists and is unpaired)
        if i > 0 and i - 1 in loop.unpaired:
            d5_base = seq.bases[i - 1].value
            energy += int(params.dangle5[pair_idx][d5_base])

        # 3' dangling end (base after the pair, if it exists and is unpaired)
        if j + 1 < len(seq) and j + 1 in loop.unpaired:
            d3_base = seq.bases[j + 1].value
            energy += int(params.dangle3[pair_idx][d3_base])

    return energy
