"""
optimizer.py — Multi-stage riboswitch sequence optimizer.

Orchestrates the full design pipeline:
  1. Boltzmann sampling via bicompat sampler
  2. Scoring and filtering
  3. Local search refinement
  4. Final ranking

Usage:
    optimizer = RiboswitchOptimizer(on_struct, off_struct)
    results = optimizer.run()
"""

import random
import math
from dataclasses import dataclass, field

from ribo_switch.types import Base, Sequence, CANONICAL_PAIRS
from ribo_switch.structure import parse_dot_bracket
from ribo_switch.turner import TurnerParams
from ribo_switch.bicompat import BicompatSampler
from ribo_switch.scorer import CandidateResult, score_candidate, score_batch


@dataclass
class OptimizerConfig:
    """Configuration for the riboswitch optimizer.

    Attributes:
        n_samples: Number of Boltzmann samples to generate.
        top_k: Number of top candidates to keep after initial scoring.
        n_mutations: Number of mutation rounds per seed in local search.
        temperature: Temperature in Celsius for energy calculations.
        sa_temperature: Simulated annealing temperature for local search.
        alpha: Weight for MFE gap penalty in scoring.
        beta: Weight for energy balance penalty in scoring.
        seed: Random seed (None for random).
    """
    n_samples: int = 1000
    top_k: int = 20
    n_mutations: int = 50
    temperature: float = 37.0
    sa_temperature: float = 2.0
    alpha: float = 1.0
    beta: float = 0.5
    seed: int | None = None


class RiboswitchOptimizer:
    """Multi-stage riboswitch sequence design optimizer."""

    def __init__(
        self,
        s1: str,
        s2: str,
        params: TurnerParams | None = None,
        config: OptimizerConfig | None = None,
    ):
        self.s1_db = s1
        self.s2_db = s2
        self.struct1 = parse_dot_bracket(s1)
        self.struct2 = parse_dot_bracket(s2)
        self.params = params or TurnerParams.turner2004()
        self.config = config or OptimizerConfig()

        if self.config.seed is not None:
            random.seed(self.config.seed)

        self.sampler = BicompatSampler(
            s1, s2, self.params, self.config.temperature
        )

    def run(self, verbose: bool = False) -> list[CandidateResult]:
        """Run the full optimization pipeline.

        Returns:
            List of CandidateResult, sorted by combined score (best first).
        """
        cfg = self.config

        # Stage 1: Boltzmann sampling
        if verbose:
            print(f"Stage 1: Generating {cfg.n_samples} bicompatible samples...")
        self.sampler.precompute()
        candidates = self.sampler.sample(cfg.n_samples)
        if verbose:
            print(f"  Generated {len(candidates)} valid sequences")

        if not candidates:
            return []

        # Stage 2: Score and filter (without MFE for speed)
        if verbose:
            print(f"Stage 2: Scoring and filtering to top {cfg.top_k}...")
        scored = score_batch(
            candidates, self.struct1, self.struct2, self.params,
            cfg.alpha, cfg.beta, compute_mfe=False,
        )
        top_candidates = scored[:cfg.top_k]

        if verbose:
            best = top_candidates[0]
            print(f"  Best initial: E1={best.energy_s1:.2f} E2={best.energy_s2:.2f} "
                  f"score={best.combined_score:.2f}")

        # Stage 3: Local search refinement
        if verbose:
            print(f"Stage 3: Refining top {cfg.top_k} with local search "
                  f"({cfg.n_mutations} mutations each)...")
        refined = []
        for candidate in top_candidates:
            seq = Sequence([Base[c] for c in candidate.sequence])
            improved = self._local_search(seq)
            refined.append(improved)

        # Stage 4: Final ranking with MFE
        if verbose:
            print("Stage 4: Final ranking with MFE computation...")
        final_seqs = [Sequence([Base[c] for c in r.sequence]) for r in refined]
        final = [
            score_candidate(
                seq, self.struct1, self.struct2, self.params,
                cfg.alpha, cfg.beta, compute_mfe=True,
            )
            for seq in final_seqs
        ]
        final.sort(key=lambda r: r.combined_score)

        if verbose:
            print(f"  Best final: seq={final[0].sequence} "
                  f"E1={final[0].energy_s1:.2f} E2={final[0].energy_s2:.2f} "
                  f"MFE={final[0].mfe_energy:.2f} score={final[0].combined_score:.2f}")

        return final

    def _local_search(self, seed_seq: Sequence) -> CandidateResult:
        """Refine a seed sequence via single-point mutations.

        Uses simulated annealing: accept improving mutations always,
        accept worsening mutations with probability exp(-ΔScore / T).
        """
        cfg = self.config
        pt1 = self.struct1.pair_table
        pt2 = self.struct2.pair_table
        n = len(seed_seq)

        current = list(seed_seq.bases)
        current_score = score_candidate(
            Sequence(current), self.struct1, self.struct2, self.params,
            cfg.alpha, cfg.beta, compute_mfe=False,
        )

        best = current_score
        best_bases = list(current)

        for step in range(cfg.n_mutations):
            # Pick a random position to mutate
            pos = random.randint(0, n - 1)
            old_base = current[pos]

            # Determine valid mutations at this position
            valid_bases = self._get_valid_bases(pos, current, pt1, pt2)
            if len(valid_bases) <= 1:
                continue

            # Pick a random new base (different from current)
            new_bases = [b for b in valid_bases if b != old_base]
            if not new_bases:
                continue
            new_base = random.choice(new_bases)

            # Apply mutation
            current[pos] = new_base

            # Also update paired partners to maintain compatibility
            partner_updated = self._fix_partners(pos, current, pt1, pt2)
            if not partner_updated:
                current[pos] = old_base
                continue

            # Score new sequence
            new_score = score_candidate(
                Sequence(current), self.struct1, self.struct2, self.params,
                cfg.alpha, cfg.beta, compute_mfe=False,
            )

            # Accept/reject (simulated annealing)
            delta = new_score.combined_score - current_score.combined_score
            if delta < 0 or random.random() < math.exp(-delta / cfg.sa_temperature):
                current_score = new_score
                if new_score.combined_score < best.combined_score:
                    best = new_score
                    best_bases = list(current)
            else:
                # Reject — revert
                current = list(best_bases)

        return best

    def _get_valid_bases(
        self, pos: int, current: list[Base],
        pt1: list[int], pt2: list[int],
    ) -> list[Base]:
        """Get valid bases for a position given pairing constraints."""
        valid = set(Base)

        for pt in (pt1, pt2):
            partner = pt[pos]
            if partner != -1:
                partner_base = current[partner]
                if pos < partner:
                    # pos is 5', partner is 3'
                    allowed = {b for b in Base
                               if (b, partner_base) in CANONICAL_PAIRS}
                else:
                    # pos is 3', partner is 5'
                    allowed = {b for b in Base
                               if (partner_base, b) in CANONICAL_PAIRS}
                valid &= allowed

        return list(valid)

    def _fix_partners(
        self, pos: int, bases: list[Base],
        pt1: list[int], pt2: list[int],
    ) -> bool:
        """After mutating pos, fix partner bases to maintain canonical pairing.

        Returns True if successful, False if no valid assignment exists.
        """
        for pt in (pt1, pt2):
            partner = pt[pos]
            if partner != -1:
                base_pos = bases[pos]
                # Find a compatible base for the partner
                if pos < partner:
                    # pos is 5', partner needs to be valid 3'
                    options = [b for b in Base
                               if (base_pos, b) in CANONICAL_PAIRS]
                else:
                    # pos is 3', partner needs to be valid 5'
                    options = [b for b in Base
                               if (b, base_pos) in CANONICAL_PAIRS]

                if not options:
                    return False

                # Check if current partner base works
                if bases[partner] in options:
                    continue

                # Check cross-constraints: partner might also be paired
                # in the other structure
                other_pt = pt2 if pt is pt1 else pt1
                other_partner = other_pt[partner]
                if other_partner != -1:
                    # partner is also paired in the other structure
                    # need a base that works for both
                    other_base = bases[other_partner]
                    if partner < other_partner:
                        options = [b for b in options
                                   if (b, other_base) in CANONICAL_PAIRS
                                   or other_base == Base.A]  # simplification
                    else:
                        options = [b for b in options
                                   if (other_base, b) in CANONICAL_PAIRS
                                   or other_base == Base.A]

                if not options:
                    return False

                bases[partner] = random.choice(options)

        return True
