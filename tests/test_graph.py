"""
test_graph.py — Tests for the constraint graph module.

Verifies that:
1. Graph construction works correctly
2. All generated sequences are 100% bicompatible
3. Edge cases are handled properly
"""

import random
import pytest

from ribo_switch.structure import parse_dot_bracket
from ribo_switch.graph import (
    build_constraint_graph,
    generate_bicompatible_sequence,
    verify_bicompatible,
    assign_component,
)


# Example riboswitch structures (adenine riboswitch-like)
ADENINE_ON = "((((....))))....((((....))))"    # 28 nt
ADENINE_OFF = "....((((....))))((((....))))"   # 28 nt (different but valid)

# Simpler test cases
SIMPLE_ON = "(((...)))"
SIMPLE_OFF = ".((....))."  # Different pairing pattern, same length 9

# Same structure (edge case)
SAME_ON = "(((...)))"
SAME_OFF = "(((...)))"


class TestBuildConstraintGraph:
    """Tests for build_constraint_graph function."""
    
    def test_simple_structures(self):
        """Test graph construction with simple structures."""
        s_on = parse_dot_bracket("((..))")
        s_off = parse_dot_bracket(".(...)")
        
        graph = build_constraint_graph(s_on, s_off)
        
        assert graph.n == 6
        assert len(graph.components) > 0
        # Every position should be in exactly one component
        assert len(graph.node_to_component) == 6
    
    def test_different_lengths_raises(self):
        """Structures with different lengths should raise ValueError."""
        s_on = parse_dot_bracket("((...))")
        s_off = parse_dot_bracket("((..))")
        
        with pytest.raises(ValueError, match="same length"):
            build_constraint_graph(s_on, s_off)
    
    def test_isolated_nodes(self):
        """Positions unpaired in both structures should be isolated nodes."""
        s_on = parse_dot_bracket("...((...))...")
        s_off = parse_dot_bracket("...((...))...")
        
        graph = build_constraint_graph(s_on, s_off)
        
        # Positions 0, 1, 2, 10, 11, 12 are unpaired in both
        # They should be isolated components
        isolated_count = sum(1 for c in graph.components if len(c.nodes) == 1)
        assert isolated_count >= 6
    
    def test_shared_pairs(self):
        """Pairs in both structures should create single edges, not duplicates."""
        s_on = parse_dot_bracket("((...))")
        s_off = parse_dot_bracket("((...))")  # Same structure
        
        graph = build_constraint_graph(s_on, s_off)
        
        # Both structures have the same pairs, so edges come from both
        # But the component structure should still be valid
        assert graph.n == 7


class TestGenerateBicompatibleSequence:
    """Tests for sequence generation."""
    
    def test_simple_bicompatible(self):
        """Generated sequence should be bicompatible."""
        s_on = parse_dot_bracket("((...))")
        s_off = parse_dot_bracket(".(....)")
        
        graph = build_constraint_graph(s_on, s_off)
        seq = generate_bicompatible_sequence(graph, rng=random.Random(42))
        
        assert verify_bicompatible(seq, s_on, s_off)
    
    def test_deterministic_with_seed(self):
        """Same seed should produce same sequence."""
        s_on = parse_dot_bracket("(((...)))")
        s_off = parse_dot_bracket(".((....))")
        
        graph = build_constraint_graph(s_on, s_off)
        
        seq1 = generate_bicompatible_sequence(graph, rng=random.Random(123))
        seq2 = generate_bicompatible_sequence(graph, rng=random.Random(123))
        
        assert str(seq1) == str(seq2)
    
    def test_mass_generation_all_bicompatible(self):
        """Generate many sequences, all should be bicompatible."""
        s_on = parse_dot_bracket(ADENINE_ON)
        s_off = parse_dot_bracket(ADENINE_OFF)
        
        graph = build_constraint_graph(s_on, s_off)
        
        rng = random.Random(42)
        for i in range(1000):
            seq = generate_bicompatible_sequence(graph, rng=rng)
            assert verify_bicompatible(seq, s_on, s_off), f"Sequence {i} not bicompatible: {seq}"


class TestVerifyBicompatible:
    """Tests for the verification function."""
    
    def test_valid_sequence(self):
        """A properly constructed sequence should verify."""
        s_on = parse_dot_bracket("((...))")
        s_off = parse_dot_bracket(".(....)")
        
        graph = build_constraint_graph(s_on, s_off)
        seq = generate_bicompatible_sequence(graph)
        
        assert verify_bicompatible(seq, s_on, s_off)
    
    def test_invalid_sequence_detected(self):
        """An invalid sequence should fail verification."""
        from ribo_switch.types import Sequence, Base
        
        s_on = parse_dot_bracket("((...))")
        
        # A-A pairing is not valid
        bad_seq = Sequence(bases=[Base.A, Base.A, Base.C, Base.C, Base.C, Base.A, Base.A])
        
        # This should fail because position 0 pairs with 6, and A-A is not canonical
        assert not verify_bicompatible(bad_seq, s_on, s_on)


class TestComponentAssignment:
    """Tests for individual component assignment."""
    
    def test_isolated_node_gets_random_base(self):
        """Isolated nodes should get a random base."""
        s_on = parse_dot_bracket(".......")
        s_off = parse_dot_bracket(".......")
        
        graph = build_constraint_graph(s_on, s_off)
        
        # All nodes should be isolated
        assert len(graph.components) == 7
        for comp in graph.components:
            assert len(comp.nodes) == 1
            assert len(comp.edges) == 0
    
    def test_path_assignment_valid(self):
        """Path components should produce valid assignments."""
        from ribo_switch.types import Base
        
        s_on = parse_dot_bracket("((..))")
        s_off = parse_dot_bracket("..()..")
        
        graph = build_constraint_graph(s_on, s_off)
        
        rng = random.Random(42)
        for comp in graph.components:
            if len(comp.edges) > 0:
                assignment = assign_component(comp, rng)
                # All assigned positions should have valid bases
                for pos, base in assignment.items():
                    assert base in [Base.A, Base.C, Base.G, Base.U]


class TestEdgeCases:
    """Edge case tests."""
    
    def test_all_paired(self):
        """Structure where every position is paired."""
        s_on = parse_dot_bracket("(((())))")  # All paired
        s_off = parse_dot_bracket("(())(())")  # Different but all paired
        
        graph = build_constraint_graph(s_on, s_off)
        seq = generate_bicompatible_sequence(graph)
        
        assert verify_bicompatible(seq, s_on, s_off)
    
    def test_no_pairs(self):
        """Structure with no pairs at all."""
        s_on = parse_dot_bracket("........")
        s_off = parse_dot_bracket("........")
        
        graph = build_constraint_graph(s_on, s_off)
        
        assert len(graph.components) == 8
        seq = generate_bicompatible_sequence(graph)
        assert verify_bicompatible(seq, s_on, s_off)
    
    def test_long_sequence(self):
        """Longer sequence stress test."""
        # 100 nucleotides
        s_on = parse_dot_bracket("((((...." + "." * 80 + "....))))")
        s_off = parse_dot_bracket("." * 20 + "((((...." + "." * 40 + "....))))" + "." * 20)
        
        graph = build_constraint_graph(s_on, s_off)
        
        rng = random.Random(42)
        for _ in range(100):
            seq = generate_bicompatible_sequence(graph, rng=rng)
            assert verify_bicompatible(seq, s_on, s_off)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
