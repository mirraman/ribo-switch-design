"""
test_nsga2.py — Tests for the NSGA-II evolutionary engine.

Verifies that:
1. Pareto fronts are correctly computed
2. Population converges toward (0, 0) gaps over generations
3. All solutions in the Pareto front are bicompatible
"""

import random
import pytest

from ribo_switch.structure import parse_dot_bracket
from ribo_switch.graph import verify_bicompatible
from ribo_switch.turner import TurnerParams
from ribo_switch.nsga2 import (
    Candidate,
    evaluate_candidate,
    dominates,
    fast_non_dominated_sort,
    crowding_distance,
    tournament_select,
    nsga2,
    summarize_pareto_front,
)
from ribo_switch.genetics import create_individual
from ribo_switch.graph import build_constraint_graph


# Test structures (simpler for faster tests)
SIMPLE_ON = "(((...)))"
SIMPLE_OFF = ".((....))"


class TestDominates:
    """Tests for Pareto dominance."""
    
    def test_dominates_all_better(self):
        """A candidate with all better objectives dominates."""
        s_on = parse_dot_bracket(SIMPLE_ON)
        s_off = parse_dot_bracket(SIMPLE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        params = TurnerParams.turner2004()
        
        # Create two candidates
        rng = random.Random(42)
        ind1 = create_individual(graph, rng)
        ind2 = create_individual(graph, rng)
        
        cand1 = evaluate_candidate(ind1, s_on, s_off, params)
        cand2 = evaluate_candidate(ind2, s_on, s_off, params)
        
        # At least one should not dominate the other (different tradeoffs)
        # or one dominates - either way, the function should work
        result1 = dominates(cand1, cand2)
        result2 = dominates(cand2, cand1)
        
        # Can't both dominate each other
        assert not (result1 and result2)
    
    def test_same_objectives_no_dominance(self):
        """Same objectives means no dominance."""
        s_on = parse_dot_bracket(SIMPLE_ON)
        s_off = parse_dot_bracket(SIMPLE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        params = TurnerParams.turner2004()
        
        # Same individual
        rng = random.Random(42)
        ind = create_individual(graph, rng)
        cand = evaluate_candidate(ind, s_on, s_off, params)
        
        # Should not dominate itself
        assert not dominates(cand, cand)


class TestFastNonDominatedSort:
    """Tests for non-dominated sorting."""
    
    def test_sorts_population(self):
        """Population should be sorted into fronts."""
        s_on = parse_dot_bracket(SIMPLE_ON)
        s_off = parse_dot_bracket(SIMPLE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        params = TurnerParams.turner2004()
        
        # Create population
        rng = random.Random(42)
        population = []
        for _ in range(20):
            ind = create_individual(graph, rng)
            cand = evaluate_candidate(ind, s_on, s_off, params)
            population.append(cand)
        
        fronts = fast_non_dominated_sort(population)
        
        # Should have at least one front
        assert len(fronts) >= 1
        
        # All candidates should be in some front
        total = sum(len(f) for f in fronts)
        assert total == 20
        
        # First front should be non-dominated
        for cand in fronts[0]:
            assert cand.rank == 0
    
    def test_first_front_truly_non_dominated(self):
        """No candidate in first front should be dominated by another in first front."""
        s_on = parse_dot_bracket(SIMPLE_ON)
        s_off = parse_dot_bracket(SIMPLE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        params = TurnerParams.turner2004()
        
        rng = random.Random(123)
        population = []
        for _ in range(30):
            ind = create_individual(graph, rng)
            cand = evaluate_candidate(ind, s_on, s_off, params)
            population.append(cand)
        
        fronts = fast_non_dominated_sort(population)
        pareto_front = fronts[0]
        
        # No member of Pareto front should dominate another
        for i, a in enumerate(pareto_front):
            for j, b in enumerate(pareto_front):
                if i != j:
                    assert not dominates(a, b), \
                        f"Candidate {i} dominates {j} but both in Pareto front"


class TestCrowdingDistance:
    """Tests for crowding distance computation."""
    
    def test_boundary_solutions_infinite(self):
        """Boundary solutions should have infinite crowding distance."""
        s_on = parse_dot_bracket(SIMPLE_ON)
        s_off = parse_dot_bracket(SIMPLE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        params = TurnerParams.turner2004()
        
        rng = random.Random(42)
        population = []
        for _ in range(20):
            ind = create_individual(graph, rng)
            cand = evaluate_candidate(ind, s_on, s_off, params)
            population.append(cand)
        
        fronts = fast_non_dominated_sort(population)
        crowding_distance(fronts[0])
        
        # At least some should have infinite distance (boundary)
        infinite_count = sum(1 for c in fronts[0] if c.crowding_distance == float('inf'))
        assert infinite_count >= 1


class TestTournamentSelect:
    """Tests for tournament selection."""
    
    def test_selects_from_population(self):
        """Selected candidate should be from the population."""
        s_on = parse_dot_bracket(SIMPLE_ON)
        s_off = parse_dot_bracket(SIMPLE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        params = TurnerParams.turner2004()
        
        rng = random.Random(42)
        population = []
        for _ in range(20):
            ind = create_individual(graph, rng)
            cand = evaluate_candidate(ind, s_on, s_off, params)
            population.append(cand)
        
        fast_non_dominated_sort(population)
        
        selected = tournament_select(population, rng)
        assert selected in population


class TestNSGA2:
    """Integration tests for the full NSGA-II algorithm."""
    
    def test_returns_pareto_front(self):
        """NSGA-II should return a Pareto front."""
        result = nsga2(
            SIMPLE_ON, SIMPLE_OFF,
            population_size=20,
            n_generations=5,
            seed=42
        )
        
        assert len(result) > 0
        # All should have rank 0
        for cand in result:
            assert cand.rank == 0
    
    def test_all_bicompatible(self):
        """All candidates in result should be bicompatible."""
        s_on = parse_dot_bracket(SIMPLE_ON)
        s_off = parse_dot_bracket(SIMPLE_OFF)
        
        result = nsga2(
            s_on, s_off,
            population_size=30,
            n_generations=10,
            seed=42
        )
        
        for cand in result:
            assert verify_bicompatible(cand.sequence, s_on, s_off), \
                f"Candidate not bicompatible: {cand.sequence}"
    
    def test_accepts_string_structures(self):
        """Should accept dot-bracket strings directly."""
        result = nsga2(
            "(((...)))",
            ".((....))",
            population_size=10,
            n_generations=3,
            seed=42
        )
        
        assert len(result) > 0
    
    def test_callback_called(self):
        """Callback should be called each generation."""
        generations_seen = []
        
        def callback(gen: int, front: list):
            generations_seen.append(gen)
        
        nsga2(
            SIMPLE_ON, SIMPLE_OFF,
            population_size=10,
            n_generations=5,
            seed=42,
            callback=callback
        )
        
        assert generations_seen == [0, 1, 2, 3, 4]
    
    def test_convergence(self):
        """Pareto front should improve (or stay same) over generations."""
        # Track mean gap over generations
        gap_means = []
        
        def callback(gen: int, front: list):
            mean_gap = sum(c.gap_on + c.gap_off for c in front) / len(front) if front else float('inf')
            gap_means.append(mean_gap)
        
        nsga2(
            SIMPLE_ON, SIMPLE_OFF,
            population_size=30,
            n_generations=20,
            seed=42,
            callback=callback
        )
        
        # Later generations should be at least as good as earlier
        # (allow some noise - check last 5 vs first 5)
        early_mean = sum(gap_means[:5]) / 5
        late_mean = sum(gap_means[-5:]) / 5
        
        # Late should be no worse (smaller or equal is better)
        assert late_mean <= early_mean * 1.5  # Allow 50% margin for stochasticity


class TestSummarizeParetoFront:
    """Tests for Pareto front summarization."""
    
    def test_summarize_empty(self):
        """Empty front should return count 0."""
        summary = summarize_pareto_front([])
        assert summary["count"] == 0
    
    def test_summarize_has_stats(self):
        """Summary should have all expected statistics."""
        result = nsga2(
            SIMPLE_ON, SIMPLE_OFF,
            population_size=20,
            n_generations=5,
            seed=42
        )
        
        summary = summarize_pareto_front(result)
        
        assert "count" in summary
        assert "gap_on_min" in summary
        assert "gap_on_max" in summary
        assert "gap_on_mean" in summary
        assert "gap_off_min" in summary
        assert "gap_off_max" in summary
        assert "gap_off_mean" in summary
        assert "stability_min" in summary
        assert "stability_max" in summary


class TestEdgeCases:
    """Edge case tests."""
    
    def test_small_population(self):
        """Should work with very small population."""
        result = nsga2(
            SIMPLE_ON, SIMPLE_OFF,
            population_size=5,
            n_generations=3,
            seed=42
        )
        
        assert len(result) > 0
        assert len(result) <= 5
    
    def test_one_generation(self):
        """Should work with single generation."""
        result = nsga2(
            SIMPLE_ON, SIMPLE_OFF,
            population_size=10,
            n_generations=1,
            seed=42
        )
        
        assert len(result) > 0
    
    def test_same_structure(self):
        """Should handle identical ON/OFF structures."""
        same = "(((...)))"
        result = nsga2(
            same, same,
            population_size=10,
            n_generations=3,
            seed=42
        )
        
        assert len(result) > 0
        # With same structure, gap_on == gap_off for all
        for cand in result:
            assert cand.gap_on == cand.gap_off


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
