"""
scorer.py — Candidate sequence scoring.

Implements scoring metrics for riboswitch candidates:
  - MFE gap: how close target structures are to MFE
  - Ensemble defect: probability-weighted structural distance (optional)
  - Pareto rank integration: works with NSGA-II results

Reference: Huang & Reidys (2021), Dirks et al. (2004)
"""

from dataclasses import dataclass
from typing import Optional

from ribo_switch.types import Base, Sequence, Structure
from ribo_switch.turner import TurnerParams
from ribo_switch.rust_bridge import eval_energy, fold_mfe


@dataclass
class CandidateResult:
    """Scored riboswitch candidate.

    Attributes:
        sequence:        RNA sequence as string.
        energy_s1:       ΔG(seq, S1) in kcal/mol.
        energy_s2:       ΔG(seq, S2) in kcal/mol.
        mfe_energy:      MFE energy of the sequence in kcal/mol.
        mfe_structure:   MFE structure in dot-bracket notation.
        gap_s1:          E(S1) - MFE (0 means S1 is the MFE structure).
        gap_s2:          E(S2) - MFE.
        combined_score:  Overall quality metric (lower is better).
        pareto_rank:     Rank in Pareto front (0 = non-dominated, None if not set).
        stability:       E(S1) + E(S2) in kcal/mol.
        switching_score: Two-state Boltzmann P(S_ON) in {S_ON, S_OFF} ∈ (0,1);
                         None if not computed.
    """
    sequence: str
    energy_s1: float
    energy_s2: float
    mfe_energy: float
    mfe_structure: str
    gap_s1: float
    gap_s2: float
    combined_score: float
    pareto_rank: Optional[int] = None
    stability: float = 0.0
    switching_score: Optional[float] = None

    def __post_init__(self):
        self.stability = self.energy_s1 + self.energy_s2


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
        mfe_struct = mfe_result.mfe_structure
    else:
        mfe = min(e1, e2)  # approximate lower bound
        mfe_struct = ""

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
        mfe_structure=mfe_struct,
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


def score_from_nsga2_candidate(candidate) -> CandidateResult:
    """
    Convert an NSGA-II Candidate to a CandidateResult.
    
    This allows integration between the NSGA-II output and the scoring system.
    
    Args:
        candidate: A Candidate object from nsga2.py
        
    Returns:
        CandidateResult with equivalent data
    """
    seq_str = str(candidate.sequence)
    
    return CandidateResult(
        sequence=seq_str,
        energy_s1=candidate.e_on / 100.0,
        energy_s2=candidate.e_off / 100.0,
        mfe_energy=candidate.mfe / 100.0,
        mfe_structure=candidate.mfe_structure,
        gap_s1=candidate.gap_on / 100.0,
        gap_s2=candidate.gap_off / 100.0,
        combined_score=(candidate.e_on + candidate.e_off + candidate.gap_on + candidate.gap_off) / 100.0,
        pareto_rank=candidate.rank,
        stability=(candidate.e_on + candidate.e_off) / 100.0,
        switching_score=candidate.switching_score,
    )


def summarize_results(results: list[CandidateResult]) -> dict:
    """
    Generate summary statistics for a list of scored candidates.
    
    Args:
        results: List of CandidateResult objects
        
    Returns:
        Dictionary with statistics
    """
    if not results:
        return {"count": 0}
    
    gap1s = [r.gap_s1 for r in results]
    gap2s = [r.gap_s2 for r in results]
    stabilities = [r.stability for r in results]
    scores = [r.combined_score for r in results]
    
    # Count candidates with both gaps at 0 (ideal)
    ideal_count = sum(1 for r in results if r.gap_s1 == 0 and r.gap_s2 == 0)
    
    # Count Pareto-optimal (rank 0)
    pareto_count = sum(1 for r in results if r.pareto_rank == 0)
    
    return {
        "count": len(results),
        "gap_s1_min": min(gap1s),
        "gap_s1_max": max(gap1s),
        "gap_s1_mean": sum(gap1s) / len(gap1s),
        "gap_s2_min": min(gap2s),
        "gap_s2_max": max(gap2s),
        "gap_s2_mean": sum(gap2s) / len(gap2s),
        "stability_min": min(stabilities),
        "stability_max": max(stabilities),
        "stability_mean": sum(stabilities) / len(stabilities),
        "score_min": min(scores),
        "score_max": max(scores),
        "score_mean": sum(scores) / len(scores),
        "ideal_count": ideal_count,
        "pareto_optimal_count": pareto_count,
    }
