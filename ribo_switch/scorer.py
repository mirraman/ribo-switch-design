"""
scorer.py — Candidate sequence scoring.

Implements the Huang & Reidys quality criterion: a good riboswitch
candidate has both S1 and S2 ranking close to MFE for the designed
sequence. The combined score captures:
  - How stable each target structure is (energy)
  - How close each target structure is to the MFE (energy gap)
  - How balanced the two structures are (energy difference)
"""

from dataclasses import dataclass

from ribo_switch.types import Base, Sequence, Structure
from ribo_switch.turner import TurnerParams
from ribo_switch.energy import eval_energy
from ribo_switch.fold import fold_mfe


@dataclass
class CandidateResult:
    """Scored riboswitch candidate.

    Attributes:
        sequence: RNA sequence as string.
        energy_s1: ΔG(seq, S1) in kcal/mol.
        energy_s2: ΔG(seq, S2) in kcal/mol.
        mfe_energy: MFE energy of the sequence in kcal/mol.
        gap_s1: E(S1) - MFE — how far S1 is from optimal (0 = S1 is MFE).
        gap_s2: E(S2) - MFE — how far S2 is from optimal.
        combined_score: Overall quality metric (lower is better).
    """
    sequence: str
    energy_s1: float
    energy_s2: float
    mfe_energy: float
    gap_s1: float
    gap_s2: float
    combined_score: float


def score_candidate(
    seq: Sequence,
    struct1: Structure,
    struct2: Structure,
    params: TurnerParams,
    alpha: float = 1.0,
    beta: float = 0.5,
    compute_mfe: bool = True,
) -> CandidateResult:
    """Score a candidate sequence against two target structures.

    Scoring formula:
        combined = E1 + E2 + α × (gap1 + gap2) + β × |E1 - E2|

    Where:
        E1, E2 = energy of seq in S1, S2
        gap1 = E1 - MFE, gap2 = E2 - MFE
        α weights how important MFE proximity is
        β weights energy balance between the two structures

    Lower combined score → better candidate.

    Args:
        seq: Candidate RNA sequence.
        struct1: Target structure S1.
        struct2: Target structure S2.
        params: Turner 2004 parameters.
        alpha: Weight for MFE gap penalty.
        beta: Weight for energy balance penalty.
        compute_mfe: If False, skip MFE folding (faster, gaps set to 0).

    Returns:
        CandidateResult with all scores.
    """
    # Energy in each target structure
    e1 = eval_energy(seq, struct1, params) / 100.0
    e2 = eval_energy(seq, struct2, params) / 100.0

    # MFE folding (expensive — can be skipped for initial filtering)
    if compute_mfe:
        mfe_result = fold_mfe(seq, params)
        mfe = mfe_result.mfe_energy / 100.0
    else:
        mfe = min(e1, e2)  # approximate lower bound

    gap1 = e1 - mfe
    gap2 = e2 - mfe

    # Combined score (lower = better)
    combined = e1 + e2 + alpha * (gap1 + gap2) + beta * abs(e1 - e2)

    seq_str = ''.join(b.name for b in seq.bases)

    return CandidateResult(
        sequence=seq_str,
        energy_s1=e1,
        energy_s2=e2,
        mfe_energy=mfe,
        gap_s1=gap1,
        gap_s2=gap2,
        combined_score=combined,
    )


def score_batch(
    sequences: list[Sequence],
    struct1: Structure,
    struct2: Structure,
    params: TurnerParams,
    alpha: float = 1.0,
    beta: float = 0.5,
    compute_mfe: bool = False,
) -> list[CandidateResult]:
    """Score a batch of sequences (default: no MFE for speed).

    Returns sorted list (best candidates first).
    """
    results = [
        score_candidate(seq, struct1, struct2, params, alpha, beta, compute_mfe)
        for seq in sequences
    ]
    results.sort(key=lambda r: r.combined_score)
    return results
