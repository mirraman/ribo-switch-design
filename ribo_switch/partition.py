"""
partition.py — McCaskill partition function for RNA.

Computes the partition function:

    Z = Σ_s exp(-ΔG(seq, s) / kT)

summed over all secondary structures s. From Z, the Boltzmann probability
of any target structure S at equilibrium is:

    P(S) = exp(-ΔG(seq, S) / kT) / Z

Three DP tables mirror fold.py's Zuker tables, replacing min-energy with
Boltzmann-weighted sums:

    QB[i][j]  (≈ V[i][j])   — (i,j) paired
    QM[i][j]  (≈ WM[i][j])  — multiloop segment i..j
    Q[i][j]   (≈ W[i][j])   — total partition function for i..j

kT is in units of 0.01 kcal/mol (same as all energies in this package).
At T = 37 °C:  kT = 0.1987204 × 310.15 ≈ 61.6  (i.e. ~0.616 kcal/mol).

Note on multiloop QM: QM uses the same four-case recursion as WM in
fold.py (branch, i-unpaired, j-unpaired, bifurcation).  This is
consistent with the Turner energy model and may slightly overcount
structures with unpaired flanks in multiloops — an accepted approximation
used in RNA folding (Mathews 2004).  Because both P(S_ON) and P(S_OFF)
share the same Z, relative probabilities are still meaningful for
riboswitch design.

Reference: McCaskill (1990), Biopolymers 29:1105-1119.
"""

import math
import numpy as np
from dataclasses import dataclass

from ribo_switch.types import Energy, Sequence
from ribo_switch.turner import TurnerParams, INF
from ribo_switch.fold import (
    MIN_HAIRPIN,
    _can_pair,
    _pair_index,
    _hairpin_energy,
    _bulge_e,
    _interior_e,
)


@dataclass
class PartitionResult:
    """Result of McCaskill partition function computation.

    Attributes:
        Z:  total partition function (sum of Boltzmann weights over all
            secondary structures).
        kT: thermal energy in 0.01 kcal/mol units (same scale as
            eval_energy output).
    """
    Z: float
    kT: float

    def structure_prob(self, energy: Energy) -> float:
        """Boltzmann probability of a structure with given free energy.

        P(S) = exp(-E(S) / kT) / Z

        Args:
            energy: free energy in 0.01 kcal/mol, as returned by
                    eval_energy(seq, structure, params).

        Returns:
            Probability in [0, 1].  Returns 0.0 if Z is negligible.
        """
        if self.Z <= 0.0:
            return 0.0
        return math.exp(-energy / self.kT) / self.Z


def partition_fn(
    seq: Sequence,
    params: TurnerParams,
    T: float = 37.0,
) -> PartitionResult:
    """Compute the McCaskill partition function for an RNA sequence.

    Uses three DP tables filled by span-increasing order:
      QB[i][j] — Boltzmann sum assuming (i,j) are base-paired
      QM[i][j] — Boltzmann sum for a multiloop segment i..j
      Q[i][j]  — total partition function for subsequence i..j

    Q is filled via the right-extension recursion (avoids double-counting
    the unfolded baseline):
        Q[i][j] = Q[i][j-1]  +  Σ_{k=i}^{j-1} Q[i][k-1] × QB[k][j]

    where Q[i][k-1] = 1 when k = i (empty left context).

    Args:
        seq:    RNA sequence.
        params: Turner 2004 nearest-neighbour parameters.
        T:      Temperature in Celsius (default 37.0).

    Returns:
        PartitionResult with Z = Q[0][n-1] and kT.
    """
    # kT in units of 0.01 kcal/mol
    # R = 1.987204 cal/(mol·K) = 0.001987204 kcal/(mol·K)
    # kT [0.01 kcal/mol] = 0.1987204 × (T + 273.15)
    kT = 0.1987204 * (T + 273.15)

    n = len(seq)
    if n < MIN_HAIRPIN + 2:
        # Sequence too short to form any secondary structure.
        return PartitionResult(Z=1.0, kT=kT)

    # QB[i][j]: Boltzmann sum with (i,j) paired.
    qb = np.zeros((n, n), dtype=np.float64)

    # QM[i][j]: multiloop segment Boltzmann sum (≥1 branch).
    qm = np.zeros((n, n), dtype=np.float64)

    # Q[i][j]: total partition function for i..j.
    # Initialized to 1.0 — the "all unpaired" baseline for every span.
    # Cells for span ≥ MIN_HAIRPIN+2 are overwritten by the fill loop.
    q = np.ones((n, n), dtype=np.float64)

    # --- Precomputed Boltzmann factors for multiloop parameters ---
    # These are loop-invariant so we compute them once.
    b_unp       = math.exp(-params.ml_per_unpaired / kT)
    b_branch    = math.exp(-params.ml_per_branch / kT)
    b_branch_au = math.exp(-(params.ml_per_branch + params.terminal_au_penalty) / kT)
    b_au        = math.exp(-params.terminal_au_penalty / kT)
    # ml_offset + ml_per_branch paid by the closing pair of a multiloop
    b_ml_init   = math.exp(-(params.ml_offset + params.ml_per_branch) / kT)

    def _q(i: int, j: int) -> float:
        """Q[i][j] with Q[i][j] = 1.0 for j < i (empty subsequence)."""
        return 1.0 if j < i else q[i][j]

    # --- Fill DP tables by increasing span ---
    for span in range(MIN_HAIRPIN + 2, n + 1):
        for i in range(n - span + 1):
            j = i + span - 1

            # ── QB[i][j]: Boltzmann sum with (i,j) paired ────────────────
            if _can_pair(seq, i, j):
                pair_ij = _pair_index(seq, i, j)

                # Hairpin loop
                hp_e = _hairpin_energy(seq, params, i, j, pair_ij)
                qb_val = 0.0 if hp_e >= INF else math.exp(-hp_e / kT)

                # Stack / Bulge / Interior loop
                max_int = min(j - i - 2, 30)
                for p in range(i + 1, j):
                    n_left = p - i - 1
                    if n_left > max_int:
                        break
                    for r in range(j - 1, p, -1):
                        n_right = j - r - 1
                        if n_left + n_right > max_int:
                            continue
                        if not _can_pair(seq, p, r):
                            continue
                        if r - p - 1 < MIN_HAIRPIN:
                            continue
                        qb_pr = qb[p][r]
                        if qb_pr == 0.0:
                            continue

                        pair_pr = _pair_index(seq, p, r)
                        if n_left == 0 and n_right == 0:
                            loop_e = int(params.stack[pair_ij][pair_pr])
                        elif n_left == 0 or n_right == 0:
                            loop_e = _bulge_e(
                                params, pair_ij, pair_pr, n_left + n_right
                            )
                        else:
                            loop_e = _interior_e(
                                seq, params, i, j, p, r,
                                pair_ij, pair_pr, n_left, n_right,
                            )
                        qb_val += math.exp(-loop_e / kT) * qb_pr

                # Multiloop closed by (i, j)
                # ml_offset + ml_per_branch paid by closing pair; each interior
                # branch pays ml_per_branch (via QM Case 1).
                if j - i - 1 >= 2 * (MIN_HAIRPIN + 2):
                    b_ml = b_ml_init * (b_au if pair_ij in (0, 1, 4, 5) else 1.0)
                    for k in range(i + 2 + MIN_HAIRPIN, j - MIN_HAIRPIN - 1):
                        qm_left  = qm[i + 1][k]
                        qm_right = qm[k + 1][j - 1]
                        if qm_left > 0.0 and qm_right > 0.0:
                            qb_val += b_ml * qm_left * qm_right

                qb[i][j] = qb_val

            # ── QM[i][j]: multiloop segment ──────────────────────────────
            qm_val = 0.0

            # Case 1: (i,j) is a multiloop branch
            if _can_pair(seq, i, j) and qb[i][j] > 0.0:
                pair_idx = _pair_index(seq, i, j)
                b_br = b_branch_au if pair_idx in (0, 1, 4, 5) else b_branch
                qm_val += qb[i][j] * b_br

            # Case 2: i is unpaired inside multiloop
            qm_ip1_j = qm[i + 1][j]
            if qm_ip1_j > 0.0:
                qm_val += qm_ip1_j * b_unp

            # Case 3: j is unpaired inside multiloop
            qm_i_jm1 = qm[i][j - 1]
            if qm_i_jm1 > 0.0:
                qm_val += qm_i_jm1 * b_unp

            # Case 4: bifurcation into two multiloop segments
            for k in range(i + 1, j):
                qm_left  = qm[i][k]
                qm_right = qm[k + 1][j]
                if qm_left > 0.0 and qm_right > 0.0:
                    qm_val += qm_left * qm_right

            qm[i][j] = qm_val

            # ── Q[i][j]: total partition function ────────────────────────
            # Q[i][j] = Q[i][j-1]              (j is unpaired)
            #         + Σ_{k=i}^{j-1} Q[i][k-1] × QB[k][j]  (j paired with k)
            q_val = _q(i, j - 1)
            for k in range(i, j):       # k = i..j-1
                qb_kj = qb[k][j]
                if qb_kj > 0.0:
                    q_val += _q(i, k - 1) * qb_kj
            q[i][j] = q_val

    return PartitionResult(Z=float(q[0][n - 1]), kT=kT)
