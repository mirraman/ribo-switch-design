import math
import numpy as np
from dataclasses import dataclass
from ribo_switch.types import Energy, Sequence
from ribo_switch.turner import TurnerParams, INF
from ribo_switch.fold import MIN_HAIRPIN, _can_pair, _pair_index, _hairpin_energy, _bulge_e, _interior_e

@dataclass
class PartitionResult:
    Z: float
    kT: float

    def structure_prob(self, energy: Energy) -> float:
        if self.Z <= 0.0:
            return 0.0
        return math.exp(-energy / self.kT) / self.Z

def partition_fn(seq: Sequence, params: TurnerParams, T: float=37.0) -> PartitionResult:
    kT = 0.1987204 * (T + 273.15)
    n = len(seq)
    if n < MIN_HAIRPIN + 2:
        return PartitionResult(Z=1.0, kT=kT)
    qb = np.zeros((n, n), dtype=np.float64)
    qm = np.zeros((n, n), dtype=np.float64)
    q = np.ones((n, n), dtype=np.float64)
    b_unp = math.exp(-params.ml_per_unpaired / kT)
    b_branch = math.exp(-params.ml_per_branch / kT)
    b_branch_au = math.exp(-(params.ml_per_branch + params.terminal_au_penalty) / kT)
    b_au = math.exp(-params.terminal_au_penalty / kT)
    b_ml_init = math.exp(-(params.ml_offset + params.ml_per_branch) / kT)

    def _q(i: int, j: int) -> float:
        return 1.0 if j < i else q[i][j]
    for span in range(MIN_HAIRPIN + 2, n + 1):
        for i in range(n - span + 1):
            j = i + span - 1
            if _can_pair(seq, i, j):
                pair_ij = _pair_index(seq, i, j)
                hp_e = _hairpin_energy(seq, params, i, j, pair_ij)
                qb_val = 0.0 if hp_e >= INF else math.exp(-hp_e / kT)
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
                            loop_e = _bulge_e(params, pair_ij, pair_pr, n_left + n_right)
                        else:
                            loop_e = _interior_e(seq, params, i, j, p, r, pair_ij, pair_pr, n_left, n_right)
                        qb_val += math.exp(-loop_e / kT) * qb_pr
                if j - i - 1 >= 2 * (MIN_HAIRPIN + 2):
                    b_ml = b_ml_init * (b_au if pair_ij in (0, 1, 4, 5) else 1.0)
                    for k in range(i + 2 + MIN_HAIRPIN, j - MIN_HAIRPIN - 1):
                        qm_left = qm[i + 1][k]
                        qm_right = qm[k + 1][j - 1]
                        if qm_left > 0.0 and qm_right > 0.0:
                            qb_val += b_ml * qm_left * qm_right
                qb[i][j] = qb_val
            qm_val = 0.0
            if _can_pair(seq, i, j) and qb[i][j] > 0.0:
                pair_idx = _pair_index(seq, i, j)
                b_br = b_branch_au if pair_idx in (0, 1, 4, 5) else b_branch
                qm_val += qb[i][j] * b_br
            qm_ip1_j = qm[i + 1][j]
            if qm_ip1_j > 0.0:
                qm_val += qm_ip1_j * b_unp
            qm_i_jm1 = qm[i][j - 1]
            if qm_i_jm1 > 0.0:
                qm_val += qm_i_jm1 * b_unp
            for k in range(i + 1, j):
                qm_left = qm[i][k]
                qm_right = qm[k + 1][j]
                if qm_left > 0.0 and qm_right > 0.0:
                    qm_val += qm_left * qm_right
            qm[i][j] = qm_val
            q_val = _q(i, j - 1)
            for k in range(i, j):
                qb_kj = qb[k][j]
                if qb_kj > 0.0:
                    q_val += _q(i, k - 1) * qb_kj
            q[i][j] = q_val
    return PartitionResult(Z=float(q[0][n - 1]), kT=kT)
