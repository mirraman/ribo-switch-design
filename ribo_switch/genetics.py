"""
genetics.py — Evolutionary operators for bicompatible sequence design.

This module defines crossover and mutation operators that work at the level of 
constraint graph components, not individual nucleotides. This guarantees every 
offspring sequence is bicompatible by construction — no rejection step needed.

Key insight: Because the constraint graph decomposes into independent components 
(paths and cycles), we can swap entire component assignments between parents 
without breaking bicompatibility.
"""

from __future__ import annotations
from dataclasses import dataclass
import random

from ribo_switch.types import Base, Sequence
from ribo_switch.graph import ConstraintGraph, Component, assign_component


@dataclass
class Individual:
    """
    An individual in the genetic algorithm population.
    
    Stores both the sequence and its component-level representation for
    efficient crossover and mutation operations.
    """
    sequence: Sequence
    # Component assignments: maps component index -> {position: base}
    component_assignments: dict[int, dict[int, Base]]


def create_individual(graph: ConstraintGraph, rng: random.Random | None = None) -> Individual:
    """
    Create a new random individual that is guaranteed bicompatible.
    
    Args:
        graph: The constraint graph from the two target structures
        rng: Random number generator for reproducibility
        
    Returns:
        A new Individual with random but valid component assignments
    """
    if rng is None:
        rng = random.Random()
    
    bases: list[Base] = [Base.A] * graph.n
    component_assignments: dict[int, dict[int, Base]] = {}
    
    for idx, component in enumerate(graph.components):
        assignment = assign_component(component, rng)
        component_assignments[idx] = assignment
        for pos, base in assignment.items():
            bases[pos] = base
    
    return Individual(
        sequence=Sequence(bases=bases),
        component_assignments=component_assignments
    )


def crossover(
    parent_a: Individual,
    parent_b: Individual,
    graph: ConstraintGraph,
    rng: random.Random | None = None
) -> tuple[Individual, Individual]:
    """
    Perform crossover between two parents at the component level.
    
    For each connected component, randomly inherit the assignment from parent A 
    or parent B. Both offspring are guaranteed bicompatible because each component
    is inherited whole — not split.
    
    Args:
        parent_a: First parent individual
        parent_b: Second parent individual  
        graph: The constraint graph
        rng: Random number generator
        
    Returns:
        Tuple of two offspring individuals
    """
    if rng is None:
        rng = random.Random()
    
    n = graph.n
    n_components = len(graph.components)
    
    # For each component, randomly choose which parent to inherit from
    # Create two complementary children
    inherit_from_a = [rng.random() < 0.5 for _ in range(n_components)]
    
    # Child 1: follows inherit_from_a pattern
    # Child 2: follows inverse pattern
    child1_assignments: dict[int, dict[int, Base]] = {}
    child2_assignments: dict[int, dict[int, Base]] = {}
    
    child1_bases: list[Base] = [Base.A] * n
    child2_bases: list[Base] = [Base.A] * n
    
    for idx in range(n_components):
        if inherit_from_a[idx]:
            child1_assignments[idx] = parent_a.component_assignments[idx]
            child2_assignments[idx] = parent_b.component_assignments[idx]
        else:
            child1_assignments[idx] = parent_b.component_assignments[idx]
            child2_assignments[idx] = parent_a.component_assignments[idx]
        
        # Apply to sequence
        for pos, base in child1_assignments[idx].items():
            child1_bases[pos] = base
        for pos, base in child2_assignments[idx].items():
            child2_bases[pos] = base
    
    child1 = Individual(
        sequence=Sequence(bases=child1_bases),
        component_assignments=child1_assignments
    )
    child2 = Individual(
        sequence=Sequence(bases=child2_bases),
        component_assignments=child2_assignments
    )
    
    return child1, child2


def mutate(
    individual: Individual,
    graph: ConstraintGraph,
    mutation_rate: float = 0.1,
    rng: random.Random | None = None
) -> Individual:
    """
    Mutate an individual by re-sampling component assignments.
    
    Select each component independently with probability mutation_rate.
    Re-sample that component's nucleotide assignment from scratch.
    Result is guaranteed bicompatible.
    
    Args:
        individual: The individual to mutate
        graph: The constraint graph
        mutation_rate: Probability of mutating each component (0.0 to 1.0)
        rng: Random number generator
        
    Returns:
        A new (potentially mutated) individual
    """
    if rng is None:
        rng = random.Random()
    
    n = graph.n
    n_components = len(graph.components)
    
    # Copy the current assignments
    new_assignments = dict(individual.component_assignments)
    mutated = False
    
    for idx, component in enumerate(graph.components):
        if rng.random() < mutation_rate:
            # Re-sample this component
            new_assignments[idx] = assign_component(component, rng)
            mutated = True
    
    if not mutated:
        # Return the same individual if no mutation occurred
        return individual
    
    # Rebuild the sequence from assignments
    new_bases: list[Base] = [Base.A] * n
    for idx, assignment in new_assignments.items():
        for pos, base in assignment.items():
            new_bases[pos] = base
    
    return Individual(
        sequence=Sequence(bases=new_bases),
        component_assignments=new_assignments
    )


def uniform_crossover(
    parent_a: Individual,
    parent_b: Individual,
    graph: ConstraintGraph,
    swap_prob: float = 0.5,
    rng: random.Random | None = None
) -> tuple[Individual, Individual]:
    """
    Alternative crossover where each component is swapped with given probability.
    
    Unlike the standard crossover, this allows control over how much mixing
    happens between parents.
    
    Args:
        parent_a: First parent
        parent_b: Second parent
        graph: The constraint graph
        swap_prob: Probability of swapping each component (0.5 = uniform crossover)
        rng: Random number generator
        
    Returns:
        Two offspring individuals
    """
    if rng is None:
        rng = random.Random()
    
    n = graph.n
    n_components = len(graph.components)
    
    child1_assignments: dict[int, dict[int, Base]] = {}
    child2_assignments: dict[int, dict[int, Base]] = {}
    
    child1_bases: list[Base] = [Base.A] * n
    child2_bases: list[Base] = [Base.A] * n
    
    for idx in range(n_components):
        if rng.random() < swap_prob:
            # Swap: child1 gets B's, child2 gets A's
            child1_assignments[idx] = parent_b.component_assignments[idx]
            child2_assignments[idx] = parent_a.component_assignments[idx]
        else:
            # No swap: child1 gets A's, child2 gets B's
            child1_assignments[idx] = parent_a.component_assignments[idx]
            child2_assignments[idx] = parent_b.component_assignments[idx]
        
        for pos, base in child1_assignments[idx].items():
            child1_bases[pos] = base
        for pos, base in child2_assignments[idx].items():
            child2_bases[pos] = base
    
    return (
        Individual(Sequence(child1_bases), child1_assignments),
        Individual(Sequence(child2_bases), child2_assignments)
    )


def multi_point_mutate(
    individual: Individual,
    graph: ConstraintGraph,
    n_mutations: int = 1,
    rng: random.Random | None = None
) -> Individual:
    """
    Mutate exactly n_mutations components (or all if fewer exist).
    
    Unlike rate-based mutation, this guarantees a specific number of
    mutations will occur.
    
    Args:
        individual: The individual to mutate
        graph: The constraint graph  
        n_mutations: Number of components to mutate
        rng: Random number generator
        
    Returns:
        A new mutated individual
    """
    if rng is None:
        rng = random.Random()
    
    n = graph.n
    n_components = len(graph.components)
    
    # Select components to mutate
    n_to_mutate = min(n_mutations, n_components)
    indices_to_mutate = set(rng.sample(range(n_components), n_to_mutate))
    
    # Copy and modify assignments
    new_assignments = dict(individual.component_assignments)
    
    for idx in indices_to_mutate:
        new_assignments[idx] = assign_component(graph.components[idx], rng)
    
    # Rebuild sequence
    new_bases: list[Base] = [Base.A] * n
    for idx, assignment in new_assignments.items():
        for pos, base in assignment.items():
            new_bases[pos] = base
    
    return Individual(
        sequence=Sequence(bases=new_bases),
        component_assignments=new_assignments
    )
