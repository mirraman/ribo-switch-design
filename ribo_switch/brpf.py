from __future__ import annotations
import math
from dataclasses import dataclass
from ribo_switch.types import Base, Sequence, Structure, CANONICAL_PAIRS
from ribo_switch.graph import Component, ConstraintGraph
from ribo_switch.energy import eval_energy
from ribo_switch.turner import TurnerParams
_ALL_BASES = [Base.A, Base.C, Base.G, Base.U]

def kT_at(T: float=37.0) -> float:
    return 0.1987204 * (T + 273.15)

def two_state_score(e_on: int, e_off: int, kT: float) -> float:
    delta = (e_on - e_off) / kT
    if delta > 500.0:
        return 0.0
    if delta < -500.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(delta))

def enumerate_component_assignments(component: Component) -> list[dict[int, Base]]:
    nodes = component.nodes
    edges = component.edges
    if not edges:
        return [{nodes[0]: b} for b in _ALL_BASES]
    adj: dict[int, list[int]] = {n: [] for n in nodes}
    for e in edges:
        adj[e.i].append(e.j)
        adj[e.j].append(e.i)
    results: list[dict[int, Base]] = []

    def backtrack(idx: int, assignment: dict[int, Base]) -> None:
        if idx == len(nodes):
            results.append(dict(assignment))
            return
        node = nodes[idx]
        valid = list(_ALL_BASES)
        for nb in adj[node]:
            if nb in assignment:
                nb_base = assignment[nb]
                valid = [b for b in valid if (nb_base, b) in CANONICAL_PAIRS or (b, nb_base) in CANONICAL_PAIRS]
        for b in valid:
            assignment[node] = b
            backtrack(idx + 1, assignment)
            del assignment[node]
    backtrack(0, {})
    return results

@dataclass
class BRPFResult:
    log_Z_on: float
    log_Z_off: float
    switching_score: float
    kT: float
    n_components: int

def brpf(seq: Sequence, graph: ConstraintGraph, s_on: Structure, s_off: Structure, params: TurnerParams, T: float=37.0) -> BRPFResult:
    kT = kT_at(T)
    ref_bases = list(seq.bases)
    log_Z_on = 0.0
    log_Z_off = 0.0
    for component in graph.components:
        all_assignments = enumerate_component_assignments(component)
        on_terms: list[float] = []
        off_terms: list[float] = []
        for assignment in all_assignments:
            mod_bases = list(ref_bases)
            for pos, base in assignment.items():
                mod_bases[pos] = base
            mod_seq = Sequence(bases=mod_bases)
            e_on = eval_energy(mod_seq, s_on, params)
            e_off = eval_energy(mod_seq, s_off, params)
            on_terms.append(-e_on / kT)
            off_terms.append(-e_off / kT)
        log_Z_on += _log_sum_exp(on_terms)
        log_Z_off += _log_sum_exp(off_terms)
    delta = log_Z_off - log_Z_on
    if delta > 500.0:
        switching_score = 0.0
    elif delta < -500.0:
        switching_score = 1.0
    else:
        switching_score = 1.0 / (1.0 + math.exp(delta))
    return BRPFResult(log_Z_on=log_Z_on, log_Z_off=log_Z_off, switching_score=switching_score, kT=kT, n_components=len(graph.components))

def _log_sum_exp(values: list[float]) -> float:
    if not values:
        return float('-inf')
    m = max(values)
    if m == float('-inf'):
        return float('-inf')
    return m + math.log(sum((math.exp(v - m) for v in values)))
