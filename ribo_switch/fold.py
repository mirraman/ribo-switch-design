import math
import numpy as np
from dataclasses import dataclass
from ribo_switch.types import Base, Energy, Sequence, CANONICAL_PAIRS
from ribo_switch.turner import TurnerParams, INF

@dataclass
class FoldResult:
    mfe_energy: Energy
    mfe_structure: str
    v: np.ndarray
    w: np.ndarray
MIN_HAIRPIN = 3

def fold_mfe(seq: Sequence, params: TurnerParams) -> FoldResult:
    n = len(seq)
    v = np.full((n, n), INF, dtype=np.int64)
    wm = np.full((n, n), INF, dtype=np.int64)
    for span in range(MIN_HAIRPIN + 2, n + 1):
        for i in range(n - span + 1):
            j = i + span - 1
            if _can_pair(seq, i, j):
                v[i][j] = _fill_v(seq, params, v, wm, i, j)
            wm[i][j] = _fill_wm(seq, params, v, wm, i, j)
    f5 = _fill_f5(seq, params, v, n)
    mfe = int(f5[n])
    structure = _traceback(seq, params, v, f5, wm, n)
    return FoldResult(mfe_energy=mfe, mfe_structure=structure, v=v, w=f5)

def _fill_f5(seq: Sequence, params: TurnerParams, v: np.ndarray, n: int) -> np.ndarray:
    f5 = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        best = int(f5[i - 1])
        for k in range(0, i):
            if i - 1 - k < MIN_HAIRPIN + 1:
                continue
            if not _can_pair(seq, k, i - 1):
                continue
            if v[k][i - 1] >= INF:
                continue
            pi = _pair_index(seq, k, i - 1)
            e = int(f5[k]) + int(v[k][i - 1])
            if pi in (0, 1, 4, 5):
                e += params.terminal_au_penalty
            if k > 0:
                e += int(params.dangle5[pi][seq.bases[k - 1].value])
            if i < n:
                e += int(params.dangle3[pi][seq.bases[i].value])
            if e < best:
                best = e
        f5[i] = best
    return f5

def _can_pair(seq: Sequence, i: int, j: int) -> bool:
    return (seq.bases[i], seq.bases[j]) in CANONICAL_PAIRS

def _pair_index(seq: Sequence, i: int, j: int) -> int:
    bi, bj = (seq.bases[i], seq.bases[j])
    _MAP = {(Base.A, Base.U): 0, (Base.U, Base.A): 1, (Base.C, Base.G): 2, (Base.G, Base.C): 3, (Base.G, Base.U): 4, (Base.U, Base.G): 5}
    return _MAP[bi, bj]

def _fill_v(seq: Sequence, params: TurnerParams, v: np.ndarray, wm: np.ndarray, i: int, j: int) -> int:
    best = INF
    n = len(seq)
    pair_ij = _pair_index(seq, i, j)
    best = min(best, _hairpin_energy(seq, params, i, j, pair_ij))
    max_interior = min(j - i - 2, 30)
    for p in range(i + 1, j):
        n_left = p - i - 1
        if n_left > max_interior:
            break
        for q in range(j - 1, p, -1):
            n_right = j - q - 1
            total_unp = n_left + n_right
            if total_unp > max_interior:
                continue
            if not _can_pair(seq, p, q):
                continue
            if q - p - 1 < MIN_HAIRPIN:
                continue
            pair_pq = _pair_index(seq, p, q)
            if n_left == 0 and n_right == 0:
                e = int(params.stack[pair_ij][pair_pq]) + int(v[p][q])
                best = min(best, e)
            elif n_left == 0 or n_right == 0:
                bulge_size = n_left + n_right
                e = _bulge_e(params, pair_ij, pair_pq, bulge_size) + int(v[p][q])
                best = min(best, e)
            else:
                e = _interior_e(seq, params, i, j, p, q, pair_ij, pair_pq, n_left, n_right) + int(v[p][q])
                best = min(best, e)
    if j - i - 1 >= 2 * (MIN_HAIRPIN + 2):
        ml_e = params.ml_offset + params.ml_per_branch
        if pair_ij in (0, 1, 4, 5):
            ml_e += params.terminal_au_penalty
        for k in range(i + 2 + MIN_HAIRPIN, j - MIN_HAIRPIN - 1):
            branch_e = ml_e + int(wm[i + 1][k]) + int(wm[k + 1][j - 1])
            best = min(best, branch_e)
    return best

def _fill_wm(seq: Sequence, params: TurnerParams, v: np.ndarray, wm: np.ndarray, i: int, j: int) -> int:
    best = INF
    if _can_pair(seq, i, j) and v[i][j] < INF:
        branch_e = int(v[i][j]) + params.ml_per_branch
        pair_idx = _pair_index(seq, i, j)
        if pair_idx in (0, 1, 4, 5):
            branch_e += params.terminal_au_penalty
        best = min(best, branch_e)
    if i + 1 <= j and wm[i + 1][j] < INF:
        best = min(best, int(wm[i + 1][j]) + params.ml_per_unpaired)
    if i <= j - 1 and wm[i][j - 1] < INF:
        best = min(best, int(wm[i][j - 1]) + params.ml_per_unpaired)
    for k in range(i + 1, j):
        if wm[i][k] < INF and wm[k + 1][j] < INF:
            best = min(best, int(wm[i][k]) + int(wm[k + 1][j]))
    return best

def _hairpin_energy(seq: Sequence, params: TurnerParams, i: int, j: int, pair_idx: int) -> int:
    size = j - i - 1
    if size < MIN_HAIRPIN:
        return INF
    if size <= 30:
        energy = int(params.hairpin_init[size])
    else:
        energy = int(params.hairpin_init[30]) + int(round(params.loop_extrapolation_coeff * 100 * math.log(size / 30.0)))
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
    mm = (seq.bases[i + 1], seq.bases[j - 1])
    if mm in ((Base.U, Base.U), (Base.G, Base.A)):
        energy += params.hairpin_uu_ga_bonus
    if mm == (Base.G, Base.G):
        energy += params.hairpin_gg_bonus
    if pair_idx == 4:
        if i >= 2 and seq.bases[i - 1] == Base.G and (seq.bases[i - 2] == Base.G):
            energy += params.hairpin_special_gu
    if all((seq.bases[k] == Base.C for k in range(i + 1, j))):
        energy += params.hairpin_c_slope * size + params.hairpin_c_intercept
    return energy

def _bulge_e(params: TurnerParams, pair_ij: int, pair_pq: int, size: int) -> int:
    if size <= 30:
        energy = int(params.bulge_init[size])
    else:
        energy = int(params.bulge_init[30]) + int(round(params.loop_extrapolation_coeff * 100 * math.log(size / 30.0)))
    if size == 1:
        energy += int(params.stack[pair_ij][pair_pq])
    if pair_ij in (0, 1, 4, 5):
        energy += params.terminal_au_penalty
    if pair_pq in (0, 1, 4, 5):
        energy += params.terminal_au_penalty
    return energy

def _interior_e(seq: Sequence, params: TurnerParams, i: int, j: int, p: int, q: int, pair_ij: int, pair_pq: int, n_left: int, n_right: int) -> int:
    total = n_left + n_right
    if n_left == 1 and n_right == 1:
        mm5 = seq.bases[i + 1].value
        mm3 = seq.bases[j - 1].value
        val = int(params.interior_1x1[pair_ij][pair_pq][mm5][mm3])
        if val < INF:
            return val
    if total <= 30:
        energy = int(params.interior_init[total])
    else:
        energy = int(params.interior_init[30]) + int(round(params.loop_extrapolation_coeff * 100 * math.log(total / 30.0)))
    asym = abs(n_left - n_right)
    energy += min(params.ninio_max, params.ninio_m * asym)
    if not (n_left == 1 and n_right == 1):
        b5o = seq.bases[i + 1].value
        b3o = seq.bases[j - 1].value
        energy += int(params.interior_mismatch[pair_ij][b5o][b3o])
        b3i = seq.bases[p - 1].value if p > 0 else 0
        b5i = seq.bases[q + 1].value if q + 1 < len(seq) else 0
        energy += int(params.interior_mismatch[pair_pq][b3i][b5i])
    if pair_ij in (0, 1, 4, 5):
        energy += params.terminal_au_penalty
    if pair_pq in (0, 1, 4, 5):
        energy += params.terminal_au_penalty
    return energy

def _traceback(seq: Sequence, params: TurnerParams, v: np.ndarray, f5: np.ndarray, wm: np.ndarray, n: int) -> str:
    pairs: list[tuple[int, int]] = []
    _trace_f5(seq, params, v, f5, wm, n, pairs)
    db = ['.'] * n
    for i, j in pairs:
        db[i] = '('
        db[j] = ')'
    return ''.join(db)

def _trace_f5(seq: Sequence, params: TurnerParams, v: np.ndarray, f5: np.ndarray, wm: np.ndarray, n: int, pairs: list[tuple[int, int]]) -> None:
    i = n
    while i > 0:
        if int(f5[i]) == int(f5[i - 1]):
            i -= 1
            continue
        matched = False
        for k in range(0, i):
            if i - 1 - k < MIN_HAIRPIN + 1:
                continue
            if not _can_pair(seq, k, i - 1):
                continue
            if v[k][i - 1] >= INF:
                continue
            pi = _pair_index(seq, k, i - 1)
            e = int(f5[k]) + int(v[k][i - 1])
            if pi in (0, 1, 4, 5):
                e += params.terminal_au_penalty
            if k > 0:
                e += int(params.dangle5[pi][seq.bases[k - 1].value])
            if i < n:
                e += int(params.dangle3[pi][seq.bases[i].value])
            if e == int(f5[i]):
                pairs.append((k, i - 1))
                _trace_v(seq, params, v, f5, wm, k, i - 1, pairs)
                i = k
                matched = True
                break
        if not matched:
            i -= 1

def _trace_v(seq: Sequence, params: TurnerParams, v: np.ndarray, w: np.ndarray, wm: np.ndarray, i: int, j: int, pairs: list[tuple[int, int]]) -> None:
    target = int(v[i][j])
    pair_ij = _pair_index(seq, i, j)
    hp_e = _hairpin_energy(seq, params, i, j, pair_ij)
    if hp_e == target:
        return
    max_interior = min(j - i - 2, 30)
    for p in range(i + 1, j):
        n_left = p - i - 1
        if n_left > max_interior:
            break
        for q in range(j - 1, p, -1):
            n_right = j - q - 1
            total_unp = n_left + n_right
            if total_unp > max_interior:
                continue
            if not _can_pair(seq, p, q):
                continue
            if q - p - 1 < MIN_HAIRPIN:
                continue
            pair_pq = _pair_index(seq, p, q)
            if n_left == 0 and n_right == 0:
                e = int(params.stack[pair_ij][pair_pq]) + int(v[p][q])
            elif n_left == 0 or n_right == 0:
                bulge_size = n_left + n_right
                e = _bulge_e(params, pair_ij, pair_pq, bulge_size) + int(v[p][q])
            else:
                e = _interior_e(seq, params, i, j, p, q, pair_ij, pair_pq, n_left, n_right) + int(v[p][q])
            if e == target:
                pairs.append((p, q))
                _trace_v(seq, params, v, w, wm, p, q, pairs)
                return
    if j - i - 1 >= 2 * (MIN_HAIRPIN + 2):
        ml_e = params.ml_offset + params.ml_per_branch
        if pair_ij in (0, 1, 4, 5):
            ml_e += params.terminal_au_penalty
        for k in range(i + 2 + MIN_HAIRPIN, j - MIN_HAIRPIN - 1):
            if ml_e + int(wm[i + 1][k]) + int(wm[k + 1][j - 1]) == target:
                _trace_wm(seq, params, v, w, wm, i + 1, k, pairs)
                _trace_wm(seq, params, v, w, wm, k + 1, j - 1, pairs)
                return

def _trace_wm(seq: Sequence, params: TurnerParams, v: np.ndarray, w: np.ndarray, wm: np.ndarray, i: int, j: int, pairs: list[tuple[int, int]]) -> None:
    if i > j:
        return
    target = int(wm[i][j])
    if target >= INF:
        return
    if _can_pair(seq, i, j) and v[i][j] < INF:
        branch_e = int(v[i][j]) + params.ml_per_branch
        pair_idx = _pair_index(seq, i, j)
        if pair_idx in (0, 1, 4, 5):
            branch_e += params.terminal_au_penalty
        if branch_e == target:
            pairs.append((i, j))
            _trace_v(seq, params, v, w, wm, i, j, pairs)
            return
    if i + 1 <= j and wm[i + 1][j] < INF:
        if int(wm[i + 1][j]) + params.ml_per_unpaired == target:
            _trace_wm(seq, params, v, w, wm, i + 1, j, pairs)
            return
    if i <= j - 1 and wm[i][j - 1] < INF:
        if int(wm[i][j - 1]) + params.ml_per_unpaired == target:
            _trace_wm(seq, params, v, w, wm, i, j - 1, pairs)
            return
    for k in range(i + 1, j):
        if wm[i][k] < INF and wm[k + 1][j] < INF:
            if int(wm[i][k]) + int(wm[k + 1][j]) == target:
                _trace_wm(seq, params, v, w, wm, i, k, pairs)
                _trace_wm(seq, params, v, w, wm, k + 1, j, pairs)
                return
