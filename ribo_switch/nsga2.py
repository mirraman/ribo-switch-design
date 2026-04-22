from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import random
import numpy as np
from ribo_switch.types import Base, Energy, Sequence, Structure
from ribo_switch.structure import parse_dot_bracket
from ribo_switch.graph import ConstraintGraph, build_constraint_graph, verify_bicompatible
from ribo_switch.genetics import Individual, create_individual, crossover, mutate
from ribo_switch.rust_bridge import eval_energy, fold_mfe, evaluate_candidate as _rs_evaluate_candidate, evaluate_batch as _rs_evaluate_batch
from ribo_switch.fold import FoldResult
from ribo_switch.brpf import two_state_score, kT_at
from ribo_switch.turner import TurnerParams
from ribo_switch.verify import bp_distance_from_tables, bp_f1_from_tables

def _mfe_db_to_pair_table(db: str) -> list[int]:
    n = len(db)
    pt: list[int] = [-1] * n
    stack: list[int] = []
    for i, ch in enumerate(db):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            j = stack.pop()
            pt[j] = i
            pt[i] = j
    return pt

@dataclass
class Candidate:
    individual: Individual
    e_on: Energy
    e_off: Energy
    mfe: Energy
    mfe_structure: str
    switching_score: float
    s_on_pt: list[int] = field(default_factory=list, repr=False)
    s_off_pt: list[int] = field(default_factory=list, repr=False)
    include_structure_objective: bool = True
    gap_on: int = field(init=False)
    gap_off: int = field(init=False)
    stability: int = field(init=False)
    bp_dist_on: int = field(init=False)
    bp_dist_off: int = field(init=False)
    bp_f1_on: float = field(init=False)
    bp_f1_off: float = field(init=False)
    rank: int = 0
    crowding_distance: float = 0.0

    def __post_init__(self):
        self.gap_on = self.e_on - self.mfe
        self.gap_off = self.e_off - self.mfe
        self.stability = self.e_on + self.e_off
        if self.s_on_pt and self.s_off_pt and self.mfe_structure:
            mfe_pt = _mfe_db_to_pair_table(self.mfe_structure)
            self.bp_dist_on = bp_distance_from_tables(mfe_pt, self.s_on_pt)
            self.bp_dist_off = bp_distance_from_tables(mfe_pt, self.s_off_pt)
            self.bp_f1_on = bp_f1_from_tables(mfe_pt, self.s_on_pt)
            self.bp_f1_off = bp_f1_from_tables(mfe_pt, self.s_off_pt)
        else:
            self.bp_dist_on = 0
            self.bp_dist_off = 0
            self.bp_f1_on = 1.0
            self.bp_f1_off = 1.0

    @property
    def sequence(self) -> Sequence:
        return self.individual.sequence

    @property
    def objectives(self) -> tuple:
        base = (self.gap_on, self.gap_off, -self.switching_score)
        if self.include_structure_objective:
            return base + (self.bp_dist_on,)
        return base

def evaluate_candidate(individual: Individual, s_on: Structure, s_off: Structure, params: TurnerParams, include_structure_objective: bool=True) -> Candidate:
    seq = individual.sequence
    e_on, e_off, mfe, mfe_struct = _rs_evaluate_candidate(seq, s_on, s_off, params)
    kT = kT_at(37.0)
    score = two_state_score(e_on, e_off, kT)
    return Candidate(individual=individual, e_on=e_on, e_off=e_off, mfe=mfe, mfe_structure=mfe_struct, switching_score=score, s_on_pt=s_on.pair_table, s_off_pt=s_off.pair_table, include_structure_objective=include_structure_objective)

def evaluate_individuals_batch(individuals: list[Individual], s_on: Structure, s_off: Structure, params: TurnerParams, include_structure_objective: bool=True) -> list[Candidate]:
    if not individuals:
        return []
    seqs = [ind.sequence for ind in individuals]
    raw = _rs_evaluate_batch(seqs, s_on, s_off, params)
    kT = kT_at(37.0)
    return [Candidate(individual=ind, e_on=e_on, e_off=e_off, mfe=mfe, mfe_structure=mfe_db, switching_score=two_state_score(e_on, e_off, kT), s_on_pt=s_on.pair_table, s_off_pt=s_off.pair_table, include_structure_objective=include_structure_objective) for ind, (e_on, e_off, mfe, mfe_db) in zip(individuals, raw)]

def dominates(a: Candidate, b: Candidate) -> bool:
    obj_a = a.objectives
    obj_b = b.objectives
    at_least_as_good = all((oa <= ob for oa, ob in zip(obj_a, obj_b)))
    strictly_better = any((oa < ob for oa, ob in zip(obj_a, obj_b)))
    return at_least_as_good and strictly_better

def fast_non_dominated_sort(population: list[Candidate]) -> list[list[Candidate]]:
    n = len(population)
    if n == 0:
        return []
    obj = np.asarray([c.objectives for c in population], dtype=np.float64)
    diff = obj[:, None, :] - obj[None, :, :]
    leq = np.all(diff <= 0.0, axis=2)
    lt = np.any(diff < 0.0, axis=2)
    dom = leq & lt
    np.fill_diagonal(dom, False)
    domination_count = dom.sum(axis=0).astype(np.int32)
    remaining = np.ones(n, dtype=bool)
    fronts: list[list[Candidate]] = []
    front_idx = 0
    while remaining.any():
        front_mask = remaining & (domination_count == 0)
        if not front_mask.any():
            straggler_idx = np.where(remaining)[0]
            for i in straggler_idx:
                population[int(i)].rank = front_idx
            fronts.append([population[int(i)] for i in straggler_idx])
            break
        front_indices = np.where(front_mask)[0]
        for i in front_indices:
            population[int(i)].rank = front_idx
        fronts.append([population[int(i)] for i in front_indices])
        domination_count -= dom[front_indices].sum(axis=0, dtype=np.int32)
        remaining[front_indices] = False
        front_idx += 1
    return fronts

def crowding_distance(front: list[Candidate]) -> None:
    n = len(front)
    if n == 0:
        return
    for c in front:
        c.crowding_distance = 0.0
    num_objectives = len(front[0].objectives)
    for m in range(num_objectives):
        front.sort(key=lambda c: c.objectives[m])
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        obj_min = front[0].objectives[m]
        obj_max = front[-1].objectives[m]
        obj_range = obj_max - obj_min
        if obj_range == 0:
            continue
        for i in range(1, n - 1):
            distance = (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / obj_range
            front[i].crowding_distance += distance

def tournament_select(population: list[Candidate], rng: random.Random, tournament_size: int=2) -> Candidate:
    contestants = rng.sample(population, min(tournament_size, len(population)))

    def key_fn(c: Candidate) -> tuple[int, float]:
        return (c.rank, -c.crowding_distance)
    return min(contestants, key=key_fn)

def evolve(population: list[Candidate], graph: ConstraintGraph, s_on: Structure, s_off: Structure, params: TurnerParams, mutation_rate: float, rng: random.Random, include_structure_objective: bool=True) -> list[Candidate]:
    pop_size = len(population)
    child_inds: list[Individual] = []
    while len(child_inds) < pop_size:
        parent1 = tournament_select(population, rng)
        parent2 = tournament_select(population, rng)
        child1_ind, child2_ind = crossover(parent1.individual, parent2.individual, graph, rng)
        child1_ind = mutate(child1_ind, graph, mutation_rate, rng)
        child2_ind = mutate(child2_ind, graph, mutation_rate, rng)
        child_inds.append(child1_ind)
        if len(child_inds) < pop_size:
            child_inds.append(child2_ind)
    offspring = evaluate_individuals_batch(child_inds, s_on, s_off, params, include_structure_objective=include_structure_objective)
    combined = population + offspring
    fronts = fast_non_dominated_sort(combined)
    new_population: list[Candidate] = []
    front_idx = 0
    while len(new_population) + len(fronts[front_idx]) <= pop_size:
        crowding_distance(fronts[front_idx])
        new_population.extend(fronts[front_idx])
        front_idx += 1
        if front_idx >= len(fronts):
            break
    if len(new_population) < pop_size and front_idx < len(fronts):
        remaining_front = fronts[front_idx]
        crowding_distance(remaining_front)
        remaining_front.sort(key=lambda c: -c.crowding_distance)
        needed = pop_size - len(new_population)
        new_population.extend(remaining_front[:needed])
    return new_population

def filter_by_structure(candidates: list[Candidate], max_bp_dist_on: int=0, max_bp_dist_off: int | None=None) -> list[Candidate]:
    result = [c for c in candidates if c.bp_dist_on <= max_bp_dist_on]
    if max_bp_dist_off is not None:
        result = [c for c in result if c.bp_dist_off <= max_bp_dist_off]
    return result

def nsga2(structure_on: str | Structure, structure_off: str | Structure, population_size: int=100, n_generations: int=200, mutation_rate: float=0.1, params: TurnerParams | None=None, seed: int | None=None, callback: Callable[[int, list[Candidate]], None] | None=None, include_structure_objective: bool=True) -> list[Candidate]:
    if isinstance(structure_on, str):
        structure_on = parse_dot_bracket(structure_on)
    if isinstance(structure_off, str):
        structure_off = parse_dot_bracket(structure_off)
    if params is None:
        params = TurnerParams.turner2004()
    rng = random.Random(seed)
    graph = build_constraint_graph(structure_on, structure_off)
    initial_inds = [create_individual(graph, rng) for _ in range(population_size)]
    population = evaluate_individuals_batch(initial_inds, structure_on, structure_off, params, include_structure_objective=include_structure_objective)
    fronts = fast_non_dominated_sort(population)
    for front in fronts:
        crowding_distance(front)
    for gen in range(n_generations):
        population = evolve(population, graph, structure_on, structure_off, params, mutation_rate, rng, include_structure_objective=include_structure_objective)
        if callback is not None:
            pareto_front = [c for c in population if c.rank == 0]
            callback(gen, pareto_front)
    fronts = fast_non_dominated_sort(population)
    crowding_distance(fronts[0])
    return fronts[0]

def summarize_pareto_front(front: list[Candidate]) -> dict:
    if not front:
        return {'count': 0}
    gap_ons = [c.gap_on for c in front]
    gap_offs = [c.gap_off for c in front]
    stabilities = [c.stability for c in front]
    scores = [c.switching_score for c in front]
    return {'count': len(front), 'gap_on_min': min(gap_ons), 'gap_on_max': max(gap_ons), 'gap_on_mean': sum(gap_ons) / len(gap_ons), 'gap_off_min': min(gap_offs), 'gap_off_max': max(gap_offs), 'gap_off_mean': sum(gap_offs) / len(gap_offs), 'stability_min': min(stabilities), 'stability_max': max(stabilities), 'stability_mean': sum(stabilities) / len(stabilities), 'switching_score_max': max(scores), 'switching_score_mean': sum(scores) / len(scores), 'switching_score_min': min(scores), 'ideal_count': sum((1 for c in front if c.gap_on == 0 and c.gap_off == 0))}
