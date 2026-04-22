from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator
import random
from ribo_switch.types import Base, Sequence, Structure, CANONICAL_PAIRS
VALID_PAIRS: list[tuple[Base, Base]] = [(Base.A, Base.U), (Base.U, Base.A), (Base.C, Base.G), (Base.G, Base.C), (Base.G, Base.U), (Base.U, Base.G)]

@dataclass
class Edge:
    i: int
    j: int
    source: str

@dataclass
class Component:
    nodes: list[int]
    edges: list[Edge]
    is_cycle: bool

@dataclass
class ConstraintGraph:
    n: int
    components: list[Component]
    node_to_component: dict[int, int]

def build_constraint_graph(structure_on: Structure, structure_off: Structure) -> ConstraintGraph:
    if structure_on.length != structure_off.length:
        raise ValueError(f'Structures must have same length: {structure_on.length} vs {structure_off.length}')
    n = structure_on.length
    adjacency: dict[int, list[tuple[int, Edge]]] = {i: [] for i in range(n)}
    for i, j in structure_on.pairs:
        edge = Edge(i=i, j=j, source='on')
        adjacency[i].append((j, edge))
        adjacency[j].append((i, edge))
    for i, j in structure_off.pairs:
        edge = Edge(i=i, j=j, source='off')
        adjacency[i].append((j, edge))
        adjacency[j].append((i, edge))
    visited: set[int] = set()
    components: list[Component] = []
    for start in range(n):
        if start in visited:
            continue
        component = _trace_component(start, adjacency, visited)
        components.append(component)
    node_to_component: dict[int, int] = {}
    for idx, comp in enumerate(components):
        for node in comp.nodes:
            node_to_component[node] = idx
    return ConstraintGraph(n=n, components=components, node_to_component=node_to_component)

def _trace_component(start: int, adjacency: dict[int, list[tuple[int, Edge]]], visited: set[int]) -> Component:
    current = start
    if len(adjacency[start]) == 0:
        visited.add(start)
        return Component(nodes=[start], edges=[], is_cycle=False)
    found_endpoint = None
    temp_visited: set[int] = set()
    queue = [start]
    while queue:
        node = queue.pop()
        if node in temp_visited:
            continue
        temp_visited.add(node)
        if len(adjacency[node]) == 1:
            found_endpoint = node
            break
        for neighbor, _ in adjacency[node]:
            if neighbor not in temp_visited:
                queue.append(neighbor)
    trace_start = found_endpoint if found_endpoint is not None else start
    nodes: list[int] = []
    edges: list[Edge] = []
    current = trace_start
    prev = -1
    while current not in visited:
        visited.add(current)
        nodes.append(current)
        next_node = -1
        next_edge = None
        for neighbor, edge in adjacency[current]:
            if neighbor != prev:
                if neighbor not in visited:
                    next_node = neighbor
                    next_edge = edge
                    break
                elif neighbor == trace_start and len(nodes) > 2:
                    edges.append(edge)
                    return Component(nodes=nodes, edges=edges, is_cycle=True)
        if next_node == -1:
            break
        edges.append(next_edge)
        prev = current
        current = next_node
    return Component(nodes=nodes, edges=edges, is_cycle=False)

def assign_component(component: Component, rng: random.Random | None=None) -> dict[int, Base]:
    if rng is None:
        rng = random.Random()
    nodes = component.nodes
    edges = component.edges
    if len(edges) == 0:
        base = rng.choice([Base.A, Base.C, Base.G, Base.U])
        return {nodes[0]: base}
    node_edges: dict[int, list[Edge]] = {n: [] for n in nodes}
    for edge in edges:
        node_edges[edge.i].append(edge)
        node_edges[edge.j].append(edge)
    if component.is_cycle:
        return _assign_cycle(nodes, edges, node_edges, rng)
    else:
        return _assign_path(nodes, edges, node_edges, rng)

def _assign_path(nodes: list[int], edges: list[Edge], node_edges: dict[int, list[Edge]], rng: random.Random) -> dict[int, Base]:
    n = len(nodes)
    assignment: dict[int, Base] = {}
    for _ in range(100):
        assignment.clear()
        first_base = rng.choice([Base.A, Base.C, Base.G, Base.U])
        assignment[nodes[0]] = first_base
        success = True
        for idx in range(len(edges)):
            edge = edges[idx]
            node_i, node_j = (edge.i, edge.j)
            if node_i in assignment and node_j not in assignment:
                known, unknown = (node_i, node_j)
            elif node_j in assignment and node_i not in assignment:
                known, unknown = (node_j, node_i)
            else:
                if (assignment[node_i], assignment[node_j]) not in CANONICAL_PAIRS:
                    success = False
                    break
                continue
            known_base = assignment[known]
            valid_bases = [b for b in [Base.A, Base.C, Base.G, Base.U] if (known_base, b) in CANONICAL_PAIRS or (b, known_base) in CANONICAL_PAIRS]
            if not valid_bases:
                success = False
                break
            other_constraints: list[Base] = []
            for other_edge in node_edges[unknown]:
                if other_edge is edge:
                    continue
                other_node = other_edge.i if other_edge.j == unknown else other_edge.j
                if other_node in assignment:
                    other_base = assignment[other_node]
                    compatible = [b for b in valid_bases if (other_base, b) in CANONICAL_PAIRS or (b, other_base) in CANONICAL_PAIRS]
                    if not compatible:
                        success = False
                        break
                    valid_bases = compatible
            if not success or not valid_bases:
                success = False
                break
            assignment[unknown] = rng.choice(valid_bases)
        if success:
            for node in nodes:
                if node not in assignment:
                    assignment[node] = rng.choice([Base.A, Base.C, Base.G, Base.U])
            return assignment
    raise RuntimeError(f'Could not find valid assignment for path component: {nodes}')

def _assign_cycle(nodes: list[int], edges: list[Edge], node_edges: dict[int, list[Edge]], rng: random.Random) -> dict[int, Base]:
    n = len(nodes)
    for _ in range(100):
        assignment: dict[int, Base] = {}
        first_base = rng.choice([Base.A, Base.C, Base.G, Base.U])
        assignment[nodes[0]] = first_base
        success = True
        for idx, node in enumerate(nodes[1:], 1):
            prev_node = nodes[idx - 1]
            connecting_edge = None
            for edge in edges:
                if edge.i == prev_node and edge.j == node or (edge.i == node and edge.j == prev_node):
                    connecting_edge = edge
                    break
            if connecting_edge is None:
                valid_bases = [Base.A, Base.C, Base.G, Base.U]
            else:
                prev_base = assignment[prev_node]
                valid_bases = [b for b in [Base.A, Base.C, Base.G, Base.U] if (prev_base, b) in CANONICAL_PAIRS or (b, prev_base) in CANONICAL_PAIRS]
            if not valid_bases:
                success = False
                break
            assignment[node] = rng.choice(valid_bases)
        if not success:
            continue
        closing_edge = None
        for edge in edges:
            if edge.i == nodes[-1] and edge.j == nodes[0] or (edge.i == nodes[0] and edge.j == nodes[-1]):
                closing_edge = edge
                break
        if closing_edge:
            last_base = assignment[nodes[-1]]
            first_base = assignment[nodes[0]]
            if (last_base, first_base) not in CANONICAL_PAIRS and (first_base, last_base) not in CANONICAL_PAIRS:
                continue
        all_valid = True
        for edge in edges:
            base_i = assignment.get(edge.i)
            base_j = assignment.get(edge.j)
            if base_i and base_j:
                if (base_i, base_j) not in CANONICAL_PAIRS and (base_j, base_i) not in CANONICAL_PAIRS:
                    all_valid = False
                    break
        if all_valid:
            return assignment
    raise RuntimeError(f'Could not find valid assignment for cycle component: {nodes}')

def generate_bicompatible_sequence(graph: ConstraintGraph, rng: random.Random | None=None) -> Sequence:
    if rng is None:
        rng = random.Random()
    bases: list[Base] = [Base.A] * graph.n
    for component in graph.components:
        assignment = assign_component(component, rng)
        for pos, base in assignment.items():
            bases[pos] = base
    return Sequence(bases=bases)

def verify_bicompatible(seq: Sequence, s_on: Structure, s_off: Structure) -> bool:
    bases = seq.bases
    for i, j in s_on.pairs:
        if (bases[i], bases[j]) not in CANONICAL_PAIRS:
            return False
    for i, j in s_off.pairs:
        if (bases[i], bases[j]) not in CANONICAL_PAIRS:
            return False
    return True
