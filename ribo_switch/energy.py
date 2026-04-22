import math
from ribo_switch.types import Base, Energy, Sequence, Structure, HairpinLoop, StackLoop, InteriorLoop, BulgeLoop, MultiLoop, ExternalLoop, LoopType, CANONICAL_PAIRS
from ribo_switch.structure import decompose_loops
from ribo_switch.turner import TurnerParams, INF

def eval_energy(seq: Sequence, structure: Structure, params: TurnerParams) -> Energy:
    if len(seq) != structure.length:
        raise ValueError(f'Sequence length ({len(seq)}) != structure length ({structure.length})')
    loops = decompose_loops(structure)
    total = 0
    for loop in loops:
        total += _loop_energy(seq, loop, params)
    return total

def _loop_energy(seq: Sequence, loop: LoopType, params: TurnerParams) -> Energy:
    if isinstance(loop, HairpinLoop):
        return hairpin_energy(seq, loop.closing_pair, params)
    elif isinstance(loop, StackLoop):
        return stack_energy(seq, loop.outer_pair, loop.inner_pair, params)
    elif isinstance(loop, InteriorLoop):
        return interior_energy(seq, loop.outer_pair, loop.inner_pair, len(loop.left_unpaired), len(loop.right_unpaired), params)
    elif isinstance(loop, BulgeLoop):
        return bulge_energy(seq, loop.outer_pair, loop.inner_pair, len(loop.unpaired), params)
    elif isinstance(loop, MultiLoop):
        return multiloop_energy(seq, loop.closing_pair, loop.branches, len(loop.unpaired), params)
    elif isinstance(loop, ExternalLoop):
        return external_energy(seq, loop, params)
    else:
        raise TypeError(f'Unknown loop type: {type(loop)}')

def _pair_index(seq: Sequence, i: int, j: int) -> int:
    bi, bj = (seq.bases[i], seq.bases[j])
    _PAIR_MAP = {(Base.A, Base.U): 0, (Base.U, Base.A): 1, (Base.C, Base.G): 2, (Base.G, Base.C): 3, (Base.G, Base.U): 4, (Base.U, Base.G): 5}
    idx = _PAIR_MAP.get((bi, bj))
    if idx is None:
        raise ValueError(f'Non-canonical pair ({bi.name}, {bj.name}) at ({i}, {j})')
    return idx

def hairpin_energy(seq: Sequence, closing: tuple[int, int], params: TurnerParams) -> Energy:
    i, j = closing
    size = j - i - 1
    if size < 3:
        return INF
    if size <= 30:
        energy = int(params.hairpin_init[size])
    else:
        energy = int(params.hairpin_init[30]) + int(round(params.loop_extrapolation_coeff * 100 * math.log(size / 30.0)))
    pair_idx = _pair_index(seq, i, j)
    if size == 3:
        loop_seq = ''.join((seq.bases[k].name for k in range(i, j + 1)))
        bonus = params.hairpin_triloop.get(loop_seq, 0)
        if bonus:
            energy = bonus
        if pair_idx in (0, 1, 4, 5):
            energy += params.terminal_au_penalty
        if all((seq.bases[k] == Base.C for k in range(i + 1, j))):
            energy += params.hairpin_c3
        return energy
    if size == 4:
        loop_seq = ''.join((seq.bases[k].name for k in range(i, j + 1)))
        bonus = params.hairpin_tetraloop.get(loop_seq, 0)
        if bonus:
            energy = bonus
    b5 = seq.bases[i + 1].value
    b3 = seq.bases[j - 1].value
    energy += int(params.hairpin_mismatch[pair_idx][b5][b3])
    first_mm = (seq.bases[i + 1], seq.bases[j - 1])
    if first_mm in ((Base.U, Base.U), (Base.G, Base.A)):
        energy += params.hairpin_uu_ga_bonus
    if first_mm == (Base.G, Base.G):
        energy += params.hairpin_gg_bonus
    if pair_idx == 4:
        if i >= 2 and seq.bases[i - 1] == Base.G and (seq.bases[i - 2] == Base.G):
            energy += params.hairpin_special_gu
    if all((seq.bases[k] == Base.C for k in range(i + 1, j))):
        energy += params.hairpin_c_slope * size + params.hairpin_c_intercept
    return energy

def stack_energy(seq: Sequence, outer: tuple[int, int], inner: tuple[int, int], params: TurnerParams) -> Energy:
    outer_idx = _pair_index(seq, outer[0], outer[1])
    inner_idx = _pair_index(seq, inner[0], inner[1])
    return int(params.stack[outer_idx][inner_idx])

def interior_energy(seq: Sequence, outer: tuple[int, int], inner: tuple[int, int], n_left: int, n_right: int, params: TurnerParams) -> Energy:
    i, j = outer
    p, q = inner
    outer_idx = _pair_index(seq, i, j)
    inner_idx = _pair_index(seq, p, q)
    total_unpaired = n_left + n_right
    if n_left == 1 and n_right == 1:
        mm5 = seq.bases[i + 1].value
        mm3 = seq.bases[j - 1].value
        val = int(params.interior_1x1[outer_idx][inner_idx][mm5][mm3])
        if val < INF:
            return val
    if total_unpaired > 30:
        energy = int(params.interior_init[30]) + int(round(params.loop_extrapolation_coeff * 100 * math.log(total_unpaired / 30.0)))
    else:
        energy = int(params.interior_init[total_unpaired])
    asymmetry = abs(n_left - n_right)
    ninio = min(params.ninio_max, params.ninio_m * asymmetry)
    energy += ninio
    if not (n_left == 1 and n_right == 1):
        b5_outer = seq.bases[i + 1].value
        b3_outer = seq.bases[j - 1].value
        energy += int(params.interior_mismatch[outer_idx][b5_outer][b3_outer])
        b5_inner = seq.bases[q + 1].value if q + 1 < len(seq) else 0
        b3_inner = seq.bases[p - 1].value if p - 1 >= 0 else 0
        energy += int(params.interior_mismatch[inner_idx][b3_inner][b5_inner])
    if outer_idx in (0, 1, 4, 5):
        energy += params.terminal_au_penalty
    if inner_idx in (0, 1, 4, 5):
        energy += params.terminal_au_penalty
    return energy

def bulge_energy(seq: Sequence, outer: tuple[int, int], inner: tuple[int, int], n_unpaired: int, params: TurnerParams) -> Energy:
    if n_unpaired > 30:
        energy = int(params.bulge_init[30]) + int(round(params.loop_extrapolation_coeff * 100 * math.log(n_unpaired / 30.0)))
    else:
        energy = int(params.bulge_init[n_unpaired])
    outer_idx = _pair_index(seq, outer[0], outer[1])
    inner_idx = _pair_index(seq, inner[0], inner[1])
    if n_unpaired == 1:
        energy += int(params.stack[outer_idx][inner_idx])
    if outer_idx in (0, 1, 4, 5):
        energy += params.terminal_au_penalty
    if inner_idx in (0, 1, 4, 5):
        energy += params.terminal_au_penalty
    return energy

def multiloop_energy(seq: Sequence, closing: tuple[int, int], branches: list[tuple[int, int]], n_unpaired: int, params: TurnerParams) -> Energy:
    n_branches = len(branches)
    energy = params.ml_offset + params.ml_per_branch * (n_branches + 1) + params.ml_per_unpaired * n_unpaired
    closing_idx = _pair_index(seq, closing[0], closing[1])
    if closing_idx in (0, 1, 4, 5):
        energy += params.terminal_au_penalty
    for p, q in branches:
        branch_idx = _pair_index(seq, p, q)
        if branch_idx in (0, 1, 4, 5):
            energy += params.terminal_au_penalty
    return energy

def external_energy(seq: Sequence, loop: ExternalLoop, params: TurnerParams) -> Energy:
    energy = 0
    n = len(seq)
    for i, j in loop.closing_pairs:
        pair_idx = _pair_index(seq, i, j)
        if pair_idx in (0, 1, 4, 5):
            energy += params.terminal_au_penalty
        if i > 0:
            d5_base = seq.bases[i - 1].value
            energy += int(params.dangle5[pair_idx][d5_base])
        if j + 1 < n:
            d3_base = seq.bases[j + 1].value
            energy += int(params.dangle3[pair_idx][d3_base])
    return energy
