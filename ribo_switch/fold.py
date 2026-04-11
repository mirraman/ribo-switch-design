"""
fold.py — Zuker MFE (Minimum Free Energy) folding algorithm.

Given an RNA sequence, find the secondary structure with the lowest
free energy using O(n³) dynamic programming.

Reference: Zuker & Stiegler (1981), Zuker (2003)
"""

import math
import numpy as np
from dataclasses import dataclass

from ribo_switch.types import Base, Energy, Sequence, CANONICAL_PAIRS
from ribo_switch.turner import TurnerParams, INF


@dataclass
class FoldResult:
    """Result of MFE folding."""
    mfe_energy: Energy       # MFE in 0.01 kcal/mol
    mfe_structure: str       # dot-bracket notation
    v: np.ndarray            # V[i][j] table (paired)
    w: np.ndarray            # W[i][j] table (optimal subsequence)


# Minimum hairpin loop size
MIN_HAIRPIN = 3


def fold_mfe(seq: Sequence, params: TurnerParams) -> FoldResult:
    """Compute the MFE structure for a sequence.

    Uses three DP tables:
      V[i][j] — min energy assuming (i,j) form a base pair
      W[i][j] — min energy of subsequence i..j
      WM[i][j] — min energy of a multiloop segment i..j

    Args:
        seq: RNA sequence.
        params: Turner 2004 parameters.

    Returns:
        FoldResult with MFE energy and structure.
    """
    n = len(seq)
    v = np.full((n, n), INF, dtype=np.int64)
    w = np.full((n, n), 0, dtype=np.int64)
    wm = np.full((n, n), INF, dtype=np.int64)

    # Fill DP tables bottom-up: increasing span length
    for span in range(MIN_HAIRPIN + 2, n + 1):  # +2 for closing pair
        for i in range(n - span + 1):
            j = i + span - 1

            # ── V[i][j]: energy assuming (i,j) paired ──
            if _can_pair(seq, i, j):
                v[i][j] = _fill_v(seq, params, v, wm, i, j)

            # ── WM[i][j]: multiloop segment ──
            wm[i][j] = _fill_wm(seq, params, v, wm, i, j)

            # ── W[i][j]: optimal subsequence ──
            w[i][j] = _fill_w(v, w, i, j)

    # MFE is W[0][n-1]
    mfe = int(w[0][n - 1])

    # Traceback
    structure = _traceback(seq, params, v, w, wm, n)

    return FoldResult(
        mfe_energy=mfe,
        mfe_structure=structure,
        v=v,
        w=w,
    )


def _can_pair(seq: Sequence, i: int, j: int) -> bool:
    """Check if positions i and j can form a canonical base pair."""
    return (seq.bases[i], seq.bases[j]) in CANONICAL_PAIRS


def _pair_index(seq: Sequence, i: int, j: int) -> int:
    """Return BasePair index for (i,j). Assumes pair is canonical."""
    bi, bj = seq.bases[i], seq.bases[j]
    _MAP = {
        (Base.A, Base.U): 0, (Base.U, Base.A): 1,
        (Base.C, Base.G): 2, (Base.G, Base.C): 3,
        (Base.G, Base.U): 4, (Base.U, Base.G): 5,
    }
    return _MAP[(bi, bj)]

def _fill_v(
    seq: Sequence, params: TurnerParams,
    v: np.ndarray, wm: np.ndarray,
    i: int, j: int,
) -> int:
    """Compute V[i][j] — min energy with (i,j) paired.

    V[i][j] = min {
        hairpin(i, j),
        min over enclosed (p,q): stack/interior/bulge + V[p][q],
        multiloop(i, j)
    }
    """
    best = INF
    n = len(seq)
    pair_ij = _pair_index(seq, i, j)

    # Case 1: Hairpin
    best = min(best, _hairpin_energy(seq, params, i, j, pair_ij))

    # Case 2: Stacking / Interior / Bulge
    # Try all enclosed pairs (p, q) with i < p < q < j
    max_interior = min(j - i - 2, 30)  # cap interior loop size
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
                # Stacking pair: (i,j) directly on (p,q)
                e = int(params.stack[pair_ij][pair_pq]) + int(v[p][q])
                best = min(best, e)
            elif n_left == 0 or n_right == 0:
                # Bulge loop
                bulge_size = n_left + n_right
                e = _bulge_e(params, pair_ij, pair_pq, bulge_size) + int(v[p][q])
                best = min(best, e)
            else:
                # Interior loop (includes 1×1 special case inside _interior_e)
                e = _interior_e(seq, params, i, j, p, q,
                                pair_ij, pair_pq,
                                n_left, n_right) + int(v[p][q])
                best = min(best, e)

    # Case 3: Multiloop closed by (i, j)
    # E_multi = ml_offset + ml_per_branch + AU_penalty(i,j) + WM[i+1][j-1]
    if j - i - 1 >= 2 * (MIN_HAIRPIN + 2):  # need room for ≥2 branches
        ml_e = params.ml_offset + params.ml_per_branch
        if pair_ij in (0, 1, 4, 5):
            ml_e += params.terminal_au_penalty
        # Split WM into at least 2 branches via bifurcation
        for k in range(i + 2 + MIN_HAIRPIN, j - MIN_HAIRPIN - 1):
            branch_e = ml_e + int(wm[i + 1][k]) + int(wm[k + 1][j - 1])
            best = min(best, branch_e)

    return best


def _fill_wm(
    seq: Sequence, params: TurnerParams,
    v: np.ndarray, wm: np.ndarray,
    i: int, j: int,
) -> int:
    """Compute WM[i][j] — multiloop segment energy.

    WM[i][j] = min {
        V[i][j] + ml_per_branch + AU_penalty,   // (i,j) is a branch
        WM[i+1][j] + ml_per_unpaired,           // i is unpaired
        WM[i][j-1] + ml_per_unpaired,           // j is unpaired
        min over k: WM[i][k] + WM[k+1][j]       // bifurcation
    }
    """
    best = INF

    # Case 1: (i,j) paired as a branch
    if _can_pair(seq, i, j) and v[i][j] < INF:
        branch_e = int(v[i][j]) + params.ml_per_branch
        pair_idx = _pair_index(seq, i, j)
        if pair_idx in (0, 1, 4, 5):
            branch_e += params.terminal_au_penalty
        best = min(best, branch_e)

    # Case 2: i unpaired
    if i + 1 <= j and wm[i + 1][j] < INF:
        best = min(best, int(wm[i + 1][j]) + params.ml_per_unpaired)

    # Case 3: j unpaired
    if i <= j - 1 and wm[i][j - 1] < INF:
        best = min(best, int(wm[i][j - 1]) + params.ml_per_unpaired)

    # Case 4: bifurcation
    for k in range(i + 1, j):
        if wm[i][k] < INF and wm[k + 1][j] < INF:
            best = min(best, int(wm[i][k]) + int(wm[k + 1][j]))

    return best


def _fill_w(v: np.ndarray, w: np.ndarray, i: int, j: int) -> int:
    """Compute W[i][j] — optimal energy of subsequence i..j.

    W[i][j] = min {
        0,                        // all unpaired
        V[i][j],                  // (i,j) paired
        W[i+1][j],                // i unpaired
        W[i][j-1],                // j unpaired
        min over k: W[i][k] + W[k+1][j]  // bifurcation
    }
    """
    best = 0  # all unpaired baseline

    # (i,j) paired
    if v[i][j] < INF:
        best = min(best, int(v[i][j]))

    # i unpaired
    if i + 1 <= j:
        best = min(best, int(w[i + 1][j]))

    # j unpaired
    if i <= j - 1:
        best = min(best, int(w[i][j - 1]))

    # bifurcation
    for k in range(i + 1, j):
        best = min(best, int(w[i][k]) + int(w[k + 1][j]))

    return best

def _hairpin_energy(
    seq: Sequence, params: TurnerParams,
    i: int, j: int, pair_idx: int,
) -> int:
    """Hairpin energy for fold DP (simplified — mirrors energy.py logic)."""
    size = j - i - 1
    if size < MIN_HAIRPIN:
        return INF

    if size <= 30:
        energy = int(params.hairpin_init[size])
    else:
        energy = int(params.hairpin_init[30]) + int(
            round(params.loop_extrapolation_coeff * 100
                  * math.log(size / 30.0))
        )

    # Triloop (size 3)
    if size == 3:
        loop_seq = ''.join(seq.bases[k].name for k in range(i, j + 1))
        bonus = params.hairpin_triloop.get(loop_seq, 0)
        if bonus:
            energy = bonus
        if pair_idx in (0, 1, 4, 5):
            energy += params.terminal_au_penalty
        if all(seq.bases[k] == Base.C for k in range(i + 1, j)):
            energy += params.hairpin_c3
        return energy

    # Tetraloop (size 4)
    if size == 4:
        loop_seq = ''.join(seq.bases[k].name for k in range(i, j + 1))
        bonus = params.hairpin_tetraloop.get(loop_seq, 0)
        if bonus:
            energy = bonus

    # Terminal mismatch (size >= 4)
    b5 = seq.bases[i + 1].value
    b3 = seq.bases[j - 1].value
    energy += int(params.hairpin_mismatch[pair_idx][b5][b3])

    # UU/GA bonus
    mm = (seq.bases[i + 1], seq.bases[j - 1])
    if mm in ((Base.U, Base.U), (Base.G, Base.A)):
        energy += params.hairpin_uu_ga_bonus
    if mm == (Base.G, Base.G):
        energy += params.hairpin_gg_bonus

    # Special GU closure: GU pair preceded by two Gs
    if pair_idx == 4:  # GU
        if i >= 2 and seq.bases[i - 1] == Base.G and seq.bases[i - 2] == Base.G:
            energy += params.hairpin_special_gu

    # All-C penalty
    if all(seq.bases[k] == Base.C for k in range(i + 1, j)):
        energy += params.hairpin_c_slope * size + params.hairpin_c_intercept

    return energy


def _bulge_e(
    params: TurnerParams, pair_ij: int, pair_pq: int, size: int,
) -> int:
    """Bulge energy (sequence-independent parts)."""
    if size <= 30:
        energy = int(params.bulge_init[size])
    else:
        energy = int(params.bulge_init[30]) + int(
            round(params.loop_extrapolation_coeff * 100
                  * math.log(size / 30.0))
        )
    if size == 1:
        energy += int(params.stack[pair_ij][pair_pq])
    if pair_ij in (0, 1, 4, 5):
        energy += params.terminal_au_penalty
    if pair_pq in (0, 1, 4, 5):
        energy += params.terminal_au_penalty
    return energy


def _interior_e(
    seq: Sequence, params: TurnerParams,
    i: int, j: int, p: int, q: int,
    pair_ij: int, pair_pq: int,
    n_left: int, n_right: int,
) -> int:
    """Interior loop energy."""
    total = n_left + n_right

    # 1×1 special case
    if n_left == 1 and n_right == 1:
        mm5 = seq.bases[i + 1].value
        mm3 = seq.bases[j - 1].value
        val = int(params.interior_1x1[pair_ij][pair_pq][mm5][mm3])
        if val < INF:
            return val

    # General formula
    if total <= 30:
        energy = int(params.interior_init[total])
    else:
        energy = int(params.interior_init[30]) + int(
            round(params.loop_extrapolation_coeff * 100
                  * math.log(total / 30.0))
        )

    # Ninio asymmetry
    asym = abs(n_left - n_right)
    energy += min(params.ninio_max, params.ninio_m * asym)

    # Terminal mismatches (not for 1×1)
    if not (n_left == 1 and n_right == 1):
        b5o = seq.bases[i + 1].value
        b3o = seq.bases[j - 1].value
        energy += int(params.interior_mismatch[pair_ij][b5o][b3o])
        b3i = seq.bases[p - 1].value if p > 0 else 0
        b5i = seq.bases[q + 1].value if q + 1 < len(seq) else 0
        energy += int(params.interior_mismatch[pair_pq][b3i][b5i])

    # AU/GU penalties
    if pair_ij in (0, 1, 4, 5):
        energy += params.terminal_au_penalty
    if pair_pq in (0, 1, 4, 5):
        energy += params.terminal_au_penalty

    return energy

def _traceback(
    seq: Sequence, params: TurnerParams,
    v: np.ndarray, w: np.ndarray, wm: np.ndarray,
    n: int,
) -> str:
    """Reconstruct MFE structure from DP tables."""
    pairs: list[tuple[int, int]] = []
    _trace_w(seq, params, v, w, wm, 0, n - 1, pairs)

    # Build dot-bracket
    db = ['.'] * n
    for i, j in pairs:
        db[i] = '('
        db[j] = ')'
    return ''.join(db)


def _trace_w(
    seq: Sequence, params: TurnerParams,
    v: np.ndarray, w: np.ndarray, wm: np.ndarray,
    i: int, j: int,
    pairs: list[tuple[int, int]],
) -> None:
    """Traceback through W table."""
    if i >= j:
        return

    target = int(w[i][j])
    if target == 0:
        return  # all unpaired

    # Check V[i][j]
    if v[i][j] < INF and int(v[i][j]) == target:
        pairs.append((i, j))
        _trace_v(seq, params, v, w, wm, i, j, pairs)
        return

    # Check W[i+1][j]
    if i + 1 <= j and int(w[i + 1][j]) == target:
        _trace_w(seq, params, v, w, wm, i + 1, j, pairs)
        return

    # Check W[i][j-1]
    if i <= j - 1 and int(w[i][j - 1]) == target:
        _trace_w(seq, params, v, w, wm, i, j - 1, pairs)
        return

    # Bifurcation
    for k in range(i + 1, j):
        if int(w[i][k]) + int(w[k + 1][j]) == target:
            _trace_w(seq, params, v, w, wm, i, k, pairs)
            _trace_w(seq, params, v, w, wm, k + 1, j, pairs)
            return


def _trace_v(
    seq: Sequence, params: TurnerParams,
    v: np.ndarray, w: np.ndarray, wm: np.ndarray,
    i: int, j: int,
    pairs: list[tuple[int, int]],
) -> None:
    """Traceback through V table — (i,j) are paired."""
    target = int(v[i][j])
    pair_ij = _pair_index(seq, i, j)

    # Hairpin?
    hp_e = _hairpin_energy(seq, params, i, j, pair_ij)
    if hp_e == target:
        return  # leaf — no further pairs

    # Stack / Interior / Bulge?
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
                # Stack
                e = int(params.stack[pair_ij][pair_pq]) + int(v[p][q])
            elif n_left == 0 or n_right == 0:
                # Bulge
                bulge_size = n_left + n_right
                e = _bulge_e(params, pair_ij, pair_pq, bulge_size) + int(v[p][q])
            else:
                # Interior
                e = _interior_e(seq, params, i, j, p, q,
                                pair_ij, pair_pq,
                                n_left, n_right) + int(v[p][q])

            if e == target:
                pairs.append((p, q))
                _trace_v(seq, params, v, w, wm, p, q, pairs)
                return

    # Multiloop?
    if j - i - 1 >= 2 * (MIN_HAIRPIN + 2):
        ml_e = params.ml_offset + params.ml_per_branch
        if pair_ij in (0, 1, 4, 5):
            ml_e += params.terminal_au_penalty
        for k in range(i + 2 + MIN_HAIRPIN, j - MIN_HAIRPIN - 1):
            if ml_e + int(wm[i + 1][k]) + int(wm[k + 1][j - 1]) == target:
                _trace_wm(seq, params, v, w, wm, i + 1, k, pairs)
                _trace_wm(seq, params, v, w, wm, k + 1, j - 1, pairs)
                return


def _trace_wm(
    seq: Sequence, params: TurnerParams,
    v: np.ndarray, w: np.ndarray, wm: np.ndarray,
    i: int, j: int,
    pairs: list[tuple[int, int]],
) -> None:
    """Traceback through WM table (multiloop segments)."""
    if i > j:
        return

    target = int(wm[i][j])
    if target >= INF:
        return

    # Branch: V[i][j] + ml_per_branch
    if _can_pair(seq, i, j) and v[i][j] < INF:
        branch_e = int(v[i][j]) + params.ml_per_branch
        pair_idx = _pair_index(seq, i, j)
        if pair_idx in (0, 1, 4, 5):
            branch_e += params.terminal_au_penalty
        if branch_e == target:
            pairs.append((i, j))
            _trace_v(seq, params, v, w, wm, i, j, pairs)
            return

    # i unpaired
    if i + 1 <= j and wm[i + 1][j] < INF:
        if int(wm[i + 1][j]) + params.ml_per_unpaired == target:
            _trace_wm(seq, params, v, w, wm, i + 1, j, pairs)
            return

    # j unpaired
    if i <= j - 1 and wm[i][j - 1] < INF:
        if int(wm[i][j - 1]) + params.ml_per_unpaired == target:
            _trace_wm(seq, params, v, w, wm, i, j - 1, pairs)
            return

    # Bifurcation
    for k in range(i + 1, j):
        if wm[i][k] < INF and wm[k + 1][j] < INF:
            if int(wm[i][k]) + int(wm[k + 1][j]) == target:
                _trace_wm(seq, params, v, w, wm, i, k, pairs)
                _trace_wm(seq, params, v, w, wm, k + 1, j, pairs)
                return
