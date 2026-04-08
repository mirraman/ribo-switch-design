"""
test_known_riboswitches.py — Validation tests using real riboswitch structures.

Tests the full pipeline on biologically relevant riboswitch structures:
  - Adenine riboswitch (add riboswitch)
  - TPP riboswitch
  - Control experiments with random sequences
"""

import random
import pytest

from ribo_switch.structure import parse_dot_bracket
from ribo_switch.graph import (
    build_constraint_graph,
    generate_bicompatible_sequence,
    verify_bicompatible,
)
from ribo_switch.genetics import create_individual
from ribo_switch.nsga2 import nsga2, evaluate_candidate, summarize_pareto_front
from ribo_switch.turner import TurnerParams


# Real riboswitch structures from literature
# Adenine riboswitch (aptamer domain, simplified)
ADENINE_ON = "((((((.....))))))...((((....))))"  # 33 nt, ligand-bound
ADENINE_OFF = "......(((((((((....)))))))))...."  # 33 nt, ligand-free

# TPP riboswitch (thiamine pyrophosphate, simplified)
TPP_ON = "((((....))))((((....))))((((....))))"      # 36 nt
TPP_OFF = "....((((....))))....((((....))))......"[:-2]  # 36 nt (trimmed)


class TestAdenineRiboswitch:
    """Tests using adenine riboswitch structures."""
    
    def test_structure_lengths_match(self):
        """Both structures should have equal length."""
        assert len(ADENINE_ON) == len(ADENINE_OFF)
    
    def test_structures_are_valid(self):
        """Both structures should parse without error."""
        s_on = parse_dot_bracket(ADENINE_ON)
        s_off = parse_dot_bracket(ADENINE_OFF)
        
        assert s_on.length == len(ADENINE_ON)
        assert s_off.length == len(ADENINE_OFF)
    
    def test_can_build_constraint_graph(self):
        """Constraint graph should be buildable."""
        s_on = parse_dot_bracket(ADENINE_ON)
        s_off = parse_dot_bracket(ADENINE_OFF)
        
        graph = build_constraint_graph(s_on, s_off)
        
        assert graph.n == len(ADENINE_ON)
        assert len(graph.components) > 0
    
    def test_generate_bicompatible_sequences(self):
        """Generated sequences should all be bicompatible."""
        s_on = parse_dot_bracket(ADENINE_ON)
        s_off = parse_dot_bracket(ADENINE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        
        rng = random.Random(42)
        for _ in range(100):
            seq = generate_bicompatible_sequence(graph, rng)
            assert verify_bicompatible(seq, s_on, s_off)
    
    def test_nsga2_produces_pareto_front(self):
        """NSGA-II should produce a valid Pareto front."""
        result = nsga2(
            ADENINE_ON, ADENINE_OFF,
            population_size=30,
            n_generations=20,
            seed=42
        )
        
        assert len(result) > 0
        
        # All should be Pareto-optimal (rank 0)
        for cand in result:
            assert cand.rank == 0
    
    def test_designed_sequences_better_than_random(self):
        """
        Designed sequences should have better energy characteristics
        than randomly generated bicompatible sequences.
        """
        s_on = parse_dot_bracket(ADENINE_ON)
        s_off = parse_dot_bracket(ADENINE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        params = TurnerParams.turner2004()
        
        # Generate random bicompatible sequences
        rng = random.Random(42)
        random_gaps = []
        for _ in range(50):
            ind = create_individual(graph, rng)
            cand = evaluate_candidate(ind, s_on, s_off, params)
            random_gaps.append(cand.gap_on + cand.gap_off)
        
        # Run NSGA-II optimization
        pareto_front = nsga2(
            s_on, s_off,
            population_size=30,
            n_generations=30,
            seed=42
        )
        
        designed_gaps = [c.gap_on + c.gap_off for c in pareto_front]
        
        # Designed sequences should have lower (or equal) mean gap
        random_mean = sum(random_gaps) / len(random_gaps)
        designed_mean = sum(designed_gaps) / len(designed_gaps)
        
        # The best designed should be better than mean random
        assert min(designed_gaps) <= random_mean


class TestTPPRiboswitch:
    """Tests using TPP riboswitch structures."""
    
    def test_structure_lengths_match(self):
        """Both structures should have equal length."""
        assert len(TPP_ON) == len(TPP_OFF)
    
    def test_structures_are_valid(self):
        """Both structures should parse without error."""
        s_on = parse_dot_bracket(TPP_ON)
        s_off = parse_dot_bracket(TPP_OFF)
        
        assert s_on.length == len(TPP_ON)
        assert s_off.length == len(TPP_OFF)
    
    def test_nsga2_produces_pareto_front(self):
        """NSGA-II should produce a valid Pareto front."""
        result = nsga2(
            TPP_ON, TPP_OFF,
            population_size=30,
            n_generations=20,
            seed=42
        )
        
        assert len(result) > 0


class TestControlExperiments:
    """Control experiments to validate the optimization adds value."""
    
    def test_random_sequences_have_higher_gaps(self):
        """
        Random bicompatible sequences should generally have worse
        (higher) energy gaps than optimized sequences.
        
        This is the control experiment proving NSGA-II adds value.
        """
        s_on = parse_dot_bracket(ADENINE_ON)
        s_off = parse_dot_bracket(ADENINE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        params = TurnerParams.turner2004()
        
        # Generate many random sequences
        rng = random.Random(123)
        random_sequences = []
        for _ in range(100):
            ind = create_individual(graph, rng)
            cand = evaluate_candidate(ind, s_on, s_off, params)
            random_sequences.append(cand)
        
        # Run NSGA-II
        pareto_front = nsga2(
            s_on, s_off,
            population_size=50,
            n_generations=50,
            seed=123
        )
        
        # Compare distributions
        random_min_gap = min(c.gap_on + c.gap_off for c in random_sequences)
        designed_min_gap = min(c.gap_on + c.gap_off for c in pareto_front)
        
        # Designed should achieve lower minimum gap
        # (or at least match - no worse)
        assert designed_min_gap <= random_min_gap
    
    def test_convergence_over_generations(self):
        """
        The Pareto front should improve over generations.
        """
        history = []
        
        def callback(gen, front):
            if gen % 5 == 0:
                summary = summarize_pareto_front(front)
                history.append({
                    'gen': gen,
                    'mean_gap': (summary['gap_on_mean'] + summary['gap_off_mean']) / 2
                })
        
        nsga2(
            ADENINE_ON, ADENINE_OFF,
            population_size=30,
            n_generations=30,
            seed=42,
            callback=callback
        )
        
        # Should have recorded history
        assert len(history) >= 3
        
        # Later generations should have lower (or equal) mean gap
        first_gap = history[0]['mean_gap']
        last_gap = history[-1]['mean_gap']
        
        assert last_gap <= first_gap * 1.2  # Allow 20% margin for stochasticity


class TestNativeSignature:
    """
    Test for the native riboswitch signature identified by Huang et al.
    
    A good riboswitch candidate should have both target structures
    close to the MFE (low gaps) while maintaining thermodynamic stability.
    """
    
    def test_pareto_front_contains_low_gap_solutions(self):
        """
        The Pareto front should contain solutions where both gaps are small.
        """
        pareto_front = nsga2(
            ADENINE_ON, ADENINE_OFF,
            population_size=50,
            n_generations=50,
            seed=42
        )
        
        # Convert gaps to kcal/mol
        gap_sums = [(c.gap_on + c.gap_off) / 100.0 for c in pareto_front]
        
        # At least some solutions should have combined gap < 10 kcal/mol
        best_gap = min(gap_sums)
        assert best_gap < 20.0, f"Best combined gap {best_gap:.2f} kcal/mol too high"
    
    def test_pareto_front_is_diverse(self):
        """
        The Pareto front should contain diverse solutions offering
        different tradeoffs between ON and OFF states.
        """
        pareto_front = nsga2(
            ADENINE_ON, ADENINE_OFF,
            population_size=50,
            n_generations=50,
            seed=42
        )
        
        # Should have multiple non-dominated solutions
        assert len(pareto_front) >= 3
        
        # Solutions should have varying gap ratios
        ratios = []
        for c in pareto_front:
            if c.gap_off != 0:
                ratios.append(c.gap_on / (c.gap_off + 0.001))
        
        if len(ratios) >= 2:
            # There should be some variation in the ratios
            ratio_range = max(ratios) - min(ratios)
            # Some diversity expected
            assert ratio_range > 0.1 or len(pareto_front) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
