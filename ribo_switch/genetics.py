from __future__ import annotations
from dataclasses import dataclass
import random
from ribo_switch.types import Base, Sequence
from ribo_switch.graph import ConstraintGraph, Component, assign_component

@dataclass
class Individual:
    sequence: Sequence
    component_assignments: dict[int, dict[int, Base]]

def create_individual(graph: ConstraintGraph, rng: random.Random | None=None) -> Individual:
    if rng is None:
        rng = random.Random()
    bases: list[Base] = [Base.A] * graph.n
    component_assignments: dict[int, dict[int, Base]] = {}
    for idx, component in enumerate(graph.components):
        assignment = assign_component(component, rng)
        component_assignments[idx] = assignment
        for pos, base in assignment.items():
            bases[pos] = base
    return Individual(sequence=Sequence(bases=bases), component_assignments=component_assignments)

def crossover(parent_a: Individual, parent_b: Individual, graph: ConstraintGraph, rng: random.Random | None=None) -> tuple[Individual, Individual]:
    if rng is None:
        rng = random.Random()
    n = graph.n
    n_components = len(graph.components)
    inherit_from_a = [rng.random() < 0.5 for _ in range(n_components)]
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
        for pos, base in child1_assignments[idx].items():
            child1_bases[pos] = base
        for pos, base in child2_assignments[idx].items():
            child2_bases[pos] = base
    child1 = Individual(sequence=Sequence(bases=child1_bases), component_assignments=child1_assignments)
    child2 = Individual(sequence=Sequence(bases=child2_bases), component_assignments=child2_assignments)
    return (child1, child2)

def mutate(individual: Individual, graph: ConstraintGraph, mutation_rate: float=0.1, rng: random.Random | None=None) -> Individual:
    if rng is None:
        rng = random.Random()
    n = graph.n
    n_components = len(graph.components)
    new_assignments = dict(individual.component_assignments)
    mutated = False
    for idx, component in enumerate(graph.components):
        if rng.random() < mutation_rate:
            new_assignments[idx] = assign_component(component, rng)
            mutated = True
    if not mutated:
        return individual
    new_bases: list[Base] = [Base.A] * n
    for idx, assignment in new_assignments.items():
        for pos, base in assignment.items():
            new_bases[pos] = base
    return Individual(sequence=Sequence(bases=new_bases), component_assignments=new_assignments)

def uniform_crossover(parent_a: Individual, parent_b: Individual, graph: ConstraintGraph, swap_prob: float=0.5, rng: random.Random | None=None) -> tuple[Individual, Individual]:
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
            child1_assignments[idx] = parent_b.component_assignments[idx]
            child2_assignments[idx] = parent_a.component_assignments[idx]
        else:
            child1_assignments[idx] = parent_a.component_assignments[idx]
            child2_assignments[idx] = parent_b.component_assignments[idx]
        for pos, base in child1_assignments[idx].items():
            child1_bases[pos] = base
        for pos, base in child2_assignments[idx].items():
            child2_bases[pos] = base
    return (Individual(Sequence(child1_bases), child1_assignments), Individual(Sequence(child2_bases), child2_assignments))

def multi_point_mutate(individual: Individual, graph: ConstraintGraph, n_mutations: int=1, rng: random.Random | None=None) -> Individual:
    if rng is None:
        rng = random.Random()
    n = graph.n
    n_components = len(graph.components)
    n_to_mutate = min(n_mutations, n_components)
    indices_to_mutate = set(rng.sample(range(n_components), n_to_mutate))
    new_assignments = dict(individual.component_assignments)
    for idx in indices_to_mutate:
        new_assignments[idx] = assign_component(graph.components[idx], rng)
    new_bases: list[Base] = [Base.A] * n
    for idx, assignment in new_assignments.items():
        for pos, base in assignment.items():
            new_bases[pos] = base
    return Individual(sequence=Sequence(bases=new_bases), component_assignments=new_assignments)
