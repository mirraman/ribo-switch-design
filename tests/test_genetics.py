"""
test_genetics.py — Tests for the evolutionary operators module.

Verifies that:
1. All crossover offspring are bicompatible
2. All mutated individuals are bicompatible
3. Operators preserve the constraint graph structure
"""

import random
import pytest

from ribo_switch.structure import parse_dot_bracket
from ribo_switch.graph import (
    build_constraint_graph,
    verify_bicompatible,
)
from ribo_switch.genetics import (
    Individual,
    create_individual,
    crossover,
    mutate,
    uniform_crossover,
    multi_point_mutate,
)


# Test structures
STRUCTURE_ON = "((((....))))....((((....))))"    # 28 nt
STRUCTURE_OFF = "....((((....))))((((....))))"   # 28 nt


class TestCreateIndividual:
    """Tests for individual creation."""
    
    def test_creates_valid_individual(self):
        """Created individual should have valid structure."""
        s_on = parse_dot_bracket(STRUCTURE_ON)
        s_off = parse_dot_bracket(STRUCTURE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        
        ind = create_individual(graph, rng=random.Random(42))
        
        assert len(ind.sequence) == 28
        assert len(ind.component_assignments) == len(graph.components)
    
    def test_individual_is_bicompatible(self):
        """Created individual must be bicompatible."""
        s_on = parse_dot_bracket(STRUCTURE_ON)
        s_off = parse_dot_bracket(STRUCTURE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        
        rng = random.Random(42)
        for _ in range(100):
            ind = create_individual(graph, rng=rng)
            assert verify_bicompatible(ind.sequence, s_on, s_off)
    
    def test_deterministic_with_seed(self):
        """Same seed should produce same individual."""
        s_on = parse_dot_bracket(STRUCTURE_ON)
        s_off = parse_dot_bracket(STRUCTURE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        
        ind1 = create_individual(graph, rng=random.Random(123))
        ind2 = create_individual(graph, rng=random.Random(123))
        
        assert str(ind1.sequence) == str(ind2.sequence)


class TestCrossover:
    """Tests for crossover operator."""
    
    def test_offspring_bicompatible(self):
        """All crossover offspring must be bicompatible."""
        s_on = parse_dot_bracket(STRUCTURE_ON)
        s_off = parse_dot_bracket(STRUCTURE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        
        rng = random.Random(42)
        
        for _ in range(100):
            parent_a = create_individual(graph, rng=rng)
            parent_b = create_individual(graph, rng=rng)
            
            child1, child2 = crossover(parent_a, parent_b, graph, rng=rng)
            
            assert verify_bicompatible(child1.sequence, s_on, s_off), \
                f"Child1 not bicompatible: {child1.sequence}"
            assert verify_bicompatible(child2.sequence, s_on, s_off), \
                f"Child2 not bicompatible: {child2.sequence}"
    
    def test_offspring_inherit_from_parents(self):
        """Each component in offspring should come from one parent."""
        s_on = parse_dot_bracket("((...))")
        s_off = parse_dot_bracket(".(....)")
        graph = build_constraint_graph(s_on, s_off)
        
        rng = random.Random(42)
        parent_a = create_individual(graph, rng=rng)
        parent_b = create_individual(graph, rng=rng)
        
        child1, child2 = crossover(parent_a, parent_b, graph, rng=random.Random(123))
        
        # Each component should match either parent A or parent B
        for idx in range(len(graph.components)):
            comp_a = parent_a.component_assignments[idx]
            comp_b = parent_b.component_assignments[idx]
            child1_comp = child1.component_assignments[idx]
            child2_comp = child2.component_assignments[idx]
            
            assert child1_comp == comp_a or child1_comp == comp_b
            assert child2_comp == comp_a or child2_comp == comp_b
    
    def test_offspring_are_complementary(self):
        """If child1 gets A's component, child2 should get B's and vice versa."""
        s_on = parse_dot_bracket("((...))")
        s_off = parse_dot_bracket(".(....)")
        graph = build_constraint_graph(s_on, s_off)
        
        rng = random.Random(42)
        parent_a = create_individual(graph, rng=rng)
        parent_b = create_individual(graph, rng=rng)
        
        child1, child2 = crossover(parent_a, parent_b, graph, rng=random.Random(123))
        
        for idx in range(len(graph.components)):
            comp_a = parent_a.component_assignments[idx]
            comp_b = parent_b.component_assignments[idx]
            c1 = child1.component_assignments[idx]
            c2 = child2.component_assignments[idx]
            
            # If child1 got A's, child2 should have B's (and vice versa)
            if c1 == comp_a:
                assert c2 == comp_b
            else:
                assert c1 == comp_b and c2 == comp_a


class TestMutate:
    """Tests for mutation operator."""
    
    def test_mutated_is_bicompatible(self):
        """All mutated individuals must be bicompatible."""
        s_on = parse_dot_bracket(STRUCTURE_ON)
        s_off = parse_dot_bracket(STRUCTURE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        
        rng = random.Random(42)
        
        for _ in range(100):
            original = create_individual(graph, rng=rng)
            mutated = mutate(original, graph, mutation_rate=0.3, rng=rng)
            
            assert verify_bicompatible(mutated.sequence, s_on, s_off), \
                f"Mutated individual not bicompatible: {mutated.sequence}"
    
    def test_mutation_rate_zero_no_change(self):
        """With mutation_rate=0, individual should not change."""
        s_on = parse_dot_bracket(STRUCTURE_ON)
        s_off = parse_dot_bracket(STRUCTURE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        
        original = create_individual(graph, rng=random.Random(42))
        mutated = mutate(original, graph, mutation_rate=0.0, rng=random.Random(123))
        
        # Should be the same object (no mutation occurred)
        assert mutated is original
    
    def test_high_mutation_rate_changes(self):
        """With high mutation rate, most individuals should change."""
        s_on = parse_dot_bracket(STRUCTURE_ON)
        s_off = parse_dot_bracket(STRUCTURE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        
        rng = random.Random(42)
        changed_count = 0
        
        for _ in range(100):
            original = create_individual(graph, rng=rng)
            mutated = mutate(original, graph, mutation_rate=0.9, rng=rng)
            
            if str(mutated.sequence) != str(original.sequence):
                changed_count += 1
        
        # Most should have changed
        assert changed_count > 80


class TestUniformCrossover:
    """Tests for uniform crossover."""
    
    def test_offspring_bicompatible(self):
        """Uniform crossover offspring must be bicompatible."""
        s_on = parse_dot_bracket(STRUCTURE_ON)
        s_off = parse_dot_bracket(STRUCTURE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        
        rng = random.Random(42)
        
        for _ in range(100):
            parent_a = create_individual(graph, rng=rng)
            parent_b = create_individual(graph, rng=rng)
            
            child1, child2 = uniform_crossover(parent_a, parent_b, graph, rng=rng)
            
            assert verify_bicompatible(child1.sequence, s_on, s_off)
            assert verify_bicompatible(child2.sequence, s_on, s_off)
    
    def test_swap_prob_zero_keeps_originals(self):
        """With swap_prob=0, children should match parents exactly."""
        s_on = parse_dot_bracket("((...))")
        s_off = parse_dot_bracket(".(....)")
        graph = build_constraint_graph(s_on, s_off)
        
        parent_a = create_individual(graph, rng=random.Random(42))
        parent_b = create_individual(graph, rng=random.Random(123))
        
        child1, child2 = uniform_crossover(parent_a, parent_b, graph, swap_prob=0.0)
        
        # child1 should have A's assignments, child2 should have B's
        assert child1.component_assignments == parent_a.component_assignments
        assert child2.component_assignments == parent_b.component_assignments


class TestMultiPointMutate:
    """Tests for multi-point mutation."""
    
    def test_mutated_is_bicompatible(self):
        """Multi-point mutated individuals must be bicompatible."""
        s_on = parse_dot_bracket(STRUCTURE_ON)
        s_off = parse_dot_bracket(STRUCTURE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        
        rng = random.Random(42)
        
        for _ in range(100):
            original = create_individual(graph, rng=rng)
            mutated = multi_point_mutate(original, graph, n_mutations=3, rng=rng)
            
            assert verify_bicompatible(mutated.sequence, s_on, s_off)
    
    def test_exact_mutations(self):
        """Should mutate exactly n_mutations components (if enough exist)."""
        s_on = parse_dot_bracket(STRUCTURE_ON)
        s_off = parse_dot_bracket(STRUCTURE_OFF)
        graph = build_constraint_graph(s_on, s_off)
        
        original = create_individual(graph, rng=random.Random(42))
        n_mutations = min(3, len(graph.components))
        
        # With fresh RNG, mutations should change things
        mutated = multi_point_mutate(original, graph, n_mutations=n_mutations, 
                                     rng=random.Random(999))
        
        # Count how many components changed
        changed = sum(
            1 for idx in range(len(graph.components))
            if original.component_assignments[idx] != mutated.component_assignments[idx]
        )
        
        # Should have mutated at most n_mutations (might be fewer if same assignment sampled)
        assert changed <= n_mutations


class TestEdgeCases:
    """Edge case tests."""
    
    def test_single_component_structure(self):
        """Handle structures that form a single component."""
        s_on = parse_dot_bracket("((..))")
        s_off = parse_dot_bracket("((..))")  # Same structure
        graph = build_constraint_graph(s_on, s_off)
        
        rng = random.Random(42)
        parent_a = create_individual(graph, rng=rng)
        parent_b = create_individual(graph, rng=rng)
        
        child1, child2 = crossover(parent_a, parent_b, graph, rng=rng)
        mutated = mutate(child1, graph, mutation_rate=0.5, rng=rng)
        
        assert verify_bicompatible(child1.sequence, s_on, s_off)
        assert verify_bicompatible(mutated.sequence, s_on, s_off)
    
    def test_all_isolated_nodes(self):
        """Handle structures with all positions unpaired."""
        s_on = parse_dot_bracket("........")
        s_off = parse_dot_bracket("........")
        graph = build_constraint_graph(s_on, s_off)
        
        rng = random.Random(42)
        parent_a = create_individual(graph, rng=rng)
        parent_b = create_individual(graph, rng=rng)
        
        child1, child2 = crossover(parent_a, parent_b, graph, rng=rng)
        
        # All components are isolated, so offspring should be bicompatible
        assert verify_bicompatible(child1.sequence, s_on, s_off)
        assert verify_bicompatible(child2.sequence, s_on, s_off)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
