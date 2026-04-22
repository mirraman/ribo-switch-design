from dataclasses import dataclass
from typing import Optional
from ribo_switch.types import Base, Sequence, Structure
from ribo_switch.turner import TurnerParams
from ribo_switch.rust_bridge import eval_energy, fold_mfe

@dataclass
class CandidateResult:
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

def score_candidate(seq: Sequence, struct1: Structure, struct2: Structure, params: TurnerParams, alpha: float=1.0, beta: float=0.5, compute_mfe: bool=True) -> CandidateResult:
    e1 = eval_energy(seq, struct1, params) / 100.0
    e2 = eval_energy(seq, struct2, params) / 100.0
    if compute_mfe:
        mfe_result = fold_mfe(seq, params)
        mfe = mfe_result.mfe_energy / 100.0
        mfe_struct = mfe_result.mfe_structure
    else:
        mfe = min(e1, e2)
        mfe_struct = ''
    gap1 = e1 - mfe
    gap2 = e2 - mfe
    combined = e1 + e2 + alpha * (gap1 + gap2) + beta * abs(e1 - e2)
    seq_str = ''.join((b.name for b in seq.bases))
    return CandidateResult(sequence=seq_str, energy_s1=e1, energy_s2=e2, mfe_energy=mfe, mfe_structure=mfe_struct, gap_s1=gap1, gap_s2=gap2, combined_score=combined)

def score_batch(sequences: list[Sequence], struct1: Structure, struct2: Structure, params: TurnerParams, alpha: float=1.0, beta: float=0.5, compute_mfe: bool=False) -> list[CandidateResult]:
    results = [score_candidate(seq, struct1, struct2, params, alpha, beta, compute_mfe) for seq in sequences]
    results.sort(key=lambda r: r.combined_score)
    return results

def score_from_nsga2_candidate(candidate) -> CandidateResult:
    seq_str = str(candidate.sequence)
    return CandidateResult(sequence=seq_str, energy_s1=candidate.e_on / 100.0, energy_s2=candidate.e_off / 100.0, mfe_energy=candidate.mfe / 100.0, mfe_structure=candidate.mfe_structure, gap_s1=candidate.gap_on / 100.0, gap_s2=candidate.gap_off / 100.0, combined_score=(candidate.e_on + candidate.e_off + candidate.gap_on + candidate.gap_off) / 100.0, pareto_rank=candidate.rank, stability=(candidate.e_on + candidate.e_off) / 100.0, switching_score=candidate.switching_score)

def summarize_results(results: list[CandidateResult]) -> dict:
    if not results:
        return {'count': 0}
    gap1s = [r.gap_s1 for r in results]
    gap2s = [r.gap_s2 for r in results]
    stabilities = [r.stability for r in results]
    scores = [r.combined_score for r in results]
    ideal_count = sum((1 for r in results if r.gap_s1 == 0 and r.gap_s2 == 0))
    pareto_count = sum((1 for r in results if r.pareto_rank == 0))
    return {'count': len(results), 'gap_s1_min': min(gap1s), 'gap_s1_max': max(gap1s), 'gap_s1_mean': sum(gap1s) / len(gap1s), 'gap_s2_min': min(gap2s), 'gap_s2_max': max(gap2s), 'gap_s2_mean': sum(gap2s) / len(gap2s), 'stability_min': min(stabilities), 'stability_max': max(stabilities), 'stability_mean': sum(stabilities) / len(stabilities), 'score_min': min(scores), 'score_max': max(scores), 'score_mean': sum(scores) / len(scores), 'ideal_count': ideal_count, 'pareto_optimal_count': pareto_count}
