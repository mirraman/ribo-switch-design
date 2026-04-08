"""
nsga2.py — Multi-objective evolutionary engine using NSGA-II.

Implements Non-dominated Sorting Genetic Algorithm II (Deb et al., 2002)
for riboswitch inverse design. Returns a Pareto front of candidate sequences
with varying tradeoffs between ON-state and OFF-state stability.

Objectives (all minimized):
    - Gap_ON:   E(seq, S_ON) - MFE(seq)    -- how far S_ON is from MFE
    - Gap_OFF:  E(seq, S_OFF) - MFE(seq)   -- how far S_OFF is from MFE
    - Stability: E(seq, S_ON) + E(seq, S_OFF)  -- prefer structured sequences
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import random

from ribo_switch.types import Base, Energy, Sequence, Structure
from ribo_switch.structure import parse_dot_bracket
from ribo_switch.graph import ConstraintGraph, build_constraint_graph, verify_bicompatible
from ribo_switch.genetics import Individual, create_individual, crossover, mutate
from ribo_switch.energy import eval_energy
from ribo_switch.fold import fold_mfe, FoldResult
from ribo_switch.turner import TurnerParams


@dataclass
class Candidate:
    """
    A riboswitch candidate with computed objective values.
    
    Attributes:
        individual: The underlying Individual from genetics.py
        e_on: Energy of sequence in ON structure
        e_off: Energy of sequence in OFF structure
        mfe: Minimum free energy of sequence (any structure)
        mfe_structure: The MFE structure in dot-bracket notation
        gap_on: E_ON - MFE (how far ON is from optimal)
        gap_off: E_OFF - MFE (how far OFF is from optimal)
        stability: E_ON + E_OFF (lower = more stable)
        rank: Pareto rank (0 = non-dominated)
        crowding_distance: Diversity measure within Pareto front
    """
    individual: Individual
    e_on: Energy
    e_off: Energy
    mfe: Energy
    mfe_structure: str
    gap_on: int = field(init=False)
    gap_off: int = field(init=False)
    stability: int = field(init=False)
    rank: int = 0
    crowding_distance: float = 0.0
    
    def __post_init__(self):
        self.gap_on = self.e_on - self.mfe
        self.gap_off = self.e_off - self.mfe
        self.stability = self.e_on + self.e_off
    
    @property
    def sequence(self) -> Sequence:
        return self.individual.sequence
    
    @property
    def objectives(self) -> tuple[int, int, int]:
        """Return tuple of objectives (all to be minimized)."""
        return (self.gap_on, self.gap_off, self.stability)


def evaluate_candidate(
    individual: Individual,
    s_on: Structure,
    s_off: Structure,
    params: TurnerParams,
) -> Candidate:
    """
    Evaluate an individual to create a Candidate with computed objectives.
    
    Args:
        individual: The sequence with component assignments
        s_on: ON-state structure
        s_off: OFF-state structure
        params: Turner energy parameters
        
    Returns:
        Candidate with all energy values computed
    """
    seq = individual.sequence
    
    # Compute energies for both target structures
    e_on = eval_energy(seq, s_on, params)
    e_off = eval_energy(seq, s_off, params)
    
    # Compute MFE
    fold_result = fold_mfe(seq, params)
    mfe = fold_result.mfe_energy
    mfe_struct = fold_result.mfe_structure
    
    return Candidate(
        individual=individual,
        e_on=e_on,
        e_off=e_off,
        mfe=mfe,
        mfe_structure=mfe_struct,
    )


def dominates(a: Candidate, b: Candidate) -> bool:
    """
    Check if candidate `a` Pareto-dominates candidate `b`.
    
    a dominates b iff:
        - a is at least as good as b on all objectives
        - a is strictly better than b on at least one objective
    """
    obj_a = a.objectives
    obj_b = b.objectives
    
    at_least_as_good = all(oa <= ob for oa, ob in zip(obj_a, obj_b))
    strictly_better = any(oa < ob for oa, ob in zip(obj_a, obj_b))
    
    return at_least_as_good and strictly_better


def fast_non_dominated_sort(population: list[Candidate]) -> list[list[Candidate]]:
    """
    Sort population into Pareto fronts using fast non-dominated sorting.
    
    Front 0 = non-dominated (best)
    Front 1 = dominated only by front 0
    etc.
    
    Time complexity: O(M * N^2) where M = objectives, N = population size
    
    Args:
        population: List of candidates to sort
        
    Returns:
        List of fronts, where fronts[0] is the Pareto front
    """
    n = len(population)
    if n == 0:
        return []
    
    # For each candidate, track who it dominates and how many dominate it
    domination_count = [0] * n  # number of solutions that dominate this one
    dominated_set: list[list[int]] = [[] for _ in range(n)]  # solutions this dominates
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(population[i], population[j]):
                dominated_set[i].append(j)
            elif dominates(population[j], population[i]):
                domination_count[i] += 1
    
    # First front: candidates with domination_count == 0
    fronts: list[list[Candidate]] = []
    current_front_indices: list[int] = []
    
    for i in range(n):
        if domination_count[i] == 0:
            population[i].rank = 0
            current_front_indices.append(i)
    
    fronts.append([population[i] for i in current_front_indices])
    
    # Generate subsequent fronts
    front_idx = 0
    while current_front_indices:
        next_front_indices: list[int] = []
        
        for i in current_front_indices:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = front_idx + 1
                    next_front_indices.append(j)
        
        front_idx += 1
        current_front_indices = next_front_indices
        
        if next_front_indices:
            fronts.append([population[i] for i in next_front_indices])
    
    return fronts


def crowding_distance(front: list[Candidate]) -> None:
    """
    Compute crowding distance for each candidate in a Pareto front.
    
    Crowding distance measures how isolated a solution is from its neighbors
    in objective space. Higher distance = more diversity = preferred.
    
    Modifies candidates in-place, setting their crowding_distance attribute.
    
    Args:
        front: A list of candidates at the same Pareto rank
    """
    n = len(front)
    if n == 0:
        return
    
    # Initialize distances
    for c in front:
        c.crowding_distance = 0.0
    
    # For each objective
    num_objectives = 3
    for m in range(num_objectives):
        # Sort by this objective
        front.sort(key=lambda c: c.objectives[m])
        
        # Boundary solutions get infinite distance
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        
        # Range for normalization
        obj_min = front[0].objectives[m]
        obj_max = front[-1].objectives[m]
        obj_range = obj_max - obj_min
        
        if obj_range == 0:
            continue
        
        # Interior solutions: add normalized distance
        for i in range(1, n - 1):
            distance = (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / obj_range
            front[i].crowding_distance += distance


def tournament_select(
    population: list[Candidate],
    rng: random.Random,
    tournament_size: int = 2
) -> Candidate:
    """
    Select a candidate using tournament selection.
    
    Compare random candidates and return the one with:
    1. Better (lower) Pareto rank, or
    2. If same rank, higher crowding distance
    
    Args:
        population: Population to select from
        rng: Random number generator
        tournament_size: Number of candidates in each tournament
        
    Returns:
        The winning candidate
    """
    contestants = rng.sample(population, min(tournament_size, len(population)))
    
    def key_fn(c: Candidate) -> tuple[int, float]:
        # Lower rank is better, higher crowding distance is better
        return (c.rank, -c.crowding_distance)
    
    return min(contestants, key=key_fn)


def evolve(
    population: list[Candidate],
    graph: ConstraintGraph,
    s_on: Structure,
    s_off: Structure,
    params: TurnerParams,
    mutation_rate: float,
    rng: random.Random,
) -> list[Candidate]:
    """
    Perform one generation of NSGA-II evolution.
    
    Steps:
    1. Select parents via tournament selection
    2. Create offspring via crossover + mutation
    3. Combine parents + offspring (size 2N)
    4. Non-dominated sort
    5. Compute crowding distances
    6. Select top N by (rank, crowding distance)
    
    Args:
        population: Current population
        graph: Constraint graph for genetic operators
        s_on: ON-state structure
        s_off: OFF-state structure
        params: Energy parameters
        mutation_rate: Probability of mutating each component
        rng: Random number generator
        
    Returns:
        New population of same size
    """
    pop_size = len(population)
    offspring: list[Candidate] = []
    
    # Generate offspring
    while len(offspring) < pop_size:
        # Select parents
        parent1 = tournament_select(population, rng)
        parent2 = tournament_select(population, rng)
        
        # Crossover
        child1_ind, child2_ind = crossover(
            parent1.individual, parent2.individual, graph, rng
        )
        
        # Mutation
        child1_ind = mutate(child1_ind, graph, mutation_rate, rng)
        child2_ind = mutate(child2_ind, graph, mutation_rate, rng)
        
        # Evaluate
        child1 = evaluate_candidate(child1_ind, s_on, s_off, params)
        child2 = evaluate_candidate(child2_ind, s_on, s_off, params)
        
        offspring.append(child1)
        if len(offspring) < pop_size:
            offspring.append(child2)
    
    # Combine parents and offspring
    combined = population + offspring
    
    # Non-dominated sort
    fronts = fast_non_dominated_sort(combined)
    
    # Fill new population from fronts
    new_population: list[Candidate] = []
    front_idx = 0
    
    while len(new_population) + len(fronts[front_idx]) <= pop_size:
        # Add entire front
        crowding_distance(fronts[front_idx])
        new_population.extend(fronts[front_idx])
        front_idx += 1
        if front_idx >= len(fronts):
            break
    
    # If we need more, select from the next front by crowding distance
    if len(new_population) < pop_size and front_idx < len(fronts):
        remaining_front = fronts[front_idx]
        crowding_distance(remaining_front)
        
        # Sort by crowding distance (descending) and take what we need
        remaining_front.sort(key=lambda c: -c.crowding_distance)
        needed = pop_size - len(new_population)
        new_population.extend(remaining_front[:needed])
    
    return new_population


def nsga2(
    structure_on: str | Structure,
    structure_off: str | Structure,
    population_size: int = 100,
    n_generations: int = 200,
    mutation_rate: float = 0.1,
    params: TurnerParams | None = None,
    seed: int | None = None,
    callback: Callable[[int, list[Candidate]], None] | None = None,
) -> list[Candidate]:
    """
    Run NSGA-II to design riboswitch sequences.
    
    Args:
        structure_on: ON-state structure (dot-bracket string or Structure)
        structure_off: OFF-state structure (dot-bracket string or Structure)
        population_size: Number of individuals in population
        n_generations: Number of generations to evolve
        mutation_rate: Probability of mutating each component (0.0 to 1.0)
        params: Turner parameters (defaults to Turner 2004)
        seed: Random seed for reproducibility
        callback: Optional function called each generation with (gen_num, pareto_front)
        
    Returns:
        The final Pareto front of riboswitch candidates
    """
    # Parse structures if needed
    if isinstance(structure_on, str):
        structure_on = parse_dot_bracket(structure_on)
    if isinstance(structure_off, str):
        structure_off = parse_dot_bracket(structure_off)
    
    # Initialize
    if params is None:
        params = TurnerParams.turner2004()
    
    rng = random.Random(seed)
    
    # Build constraint graph
    graph = build_constraint_graph(structure_on, structure_off)
    
    # Initialize population
    population: list[Candidate] = []
    for _ in range(population_size):
        ind = create_individual(graph, rng)
        candidate = evaluate_candidate(ind, structure_on, structure_off, params)
        population.append(candidate)
    
    # Initial sort
    fronts = fast_non_dominated_sort(population)
    for front in fronts:
        crowding_distance(front)
    
    # Evolution loop
    for gen in range(n_generations):
        population = evolve(
            population, graph, structure_on, structure_off,
            params, mutation_rate, rng
        )
        
        # Callback with current Pareto front
        if callback is not None:
            pareto_front = [c for c in population if c.rank == 0]
            callback(gen, pareto_front)
    
    # Return final Pareto front
    fronts = fast_non_dominated_sort(population)
    crowding_distance(fronts[0])
    
    return fronts[0]


def summarize_pareto_front(front: list[Candidate]) -> dict:
    """
    Generate summary statistics for a Pareto front.
    
    Args:
        front: List of Pareto-optimal candidates
        
    Returns:
        Dictionary with statistics
    """
    if not front:
        return {"count": 0}
    
    gap_ons = [c.gap_on for c in front]
    gap_offs = [c.gap_off for c in front]
    stabilities = [c.stability for c in front]
    
    return {
        "count": len(front),
        "gap_on_min": min(gap_ons),
        "gap_on_max": max(gap_ons),
        "gap_on_mean": sum(gap_ons) / len(gap_ons),
        "gap_off_min": min(gap_offs),
        "gap_off_max": max(gap_offs),
        "gap_off_mean": sum(gap_offs) / len(gap_offs),
        "stability_min": min(stabilities),
        "stability_max": max(stabilities),
        "stability_mean": sum(stabilities) / len(stabilities),
        "ideal_count": sum(1 for c in front if c.gap_on == 0 and c.gap_off == 0),
    }
