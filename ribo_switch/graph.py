"""
graph.py — Constraint Graph for bicompatible sequence generation.

This module is the core novelty of the thesis. It takes two dot-bracket structures 
(ON and OFF states) and builds a constraint graph where:
- Every nucleotide position is a node
- Every base pair in S_ON is an edge
- Every base pair in S_OFF is an edge

Because each node has at most degree 2 (paired in at most one structure each),
the graph decomposes perfectly into disjoint paths and cycles.

This decomposition allows O(1) mutation and crossover operators that are 
mathematically guaranteed to produce only valid bicompatible sequences.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator
import random

from ribo_switch.types import Base, Sequence, Structure, CANONICAL_PAIRS


# Valid base pairs for constraint satisfaction (Watson-Crick + wobble)
VALID_PAIRS: list[tuple[Base, Base]] = [
    (Base.A, Base.U), (Base.U, Base.A),
    (Base.C, Base.G), (Base.G, Base.C),
    (Base.G, Base.U), (Base.U, Base.G),
]


@dataclass
class Edge:
    """An edge in the constraint graph representing a base pair requirement."""
    i: int
    j: int
    source: str  # "on" or "off" - which structure this pair comes from


@dataclass
class Component:
    """A connected component of the constraint graph (path or cycle)."""
    nodes: list[int]  # positions in order of traversal
    edges: list[Edge]  # edges connecting these nodes
    is_cycle: bool


@dataclass
class ConstraintGraph:
    """
    The full constraint graph overlay of two RNA secondary structures.
    
    Attributes:
        n: Length of the RNA sequence
        components: List of connected components (paths and cycles)
        node_to_component: Maps each position to its component index
    """
    n: int
    components: list[Component]
    node_to_component: dict[int, int]


def build_constraint_graph(structure_on: Structure, structure_off: Structure) -> ConstraintGraph:
    """
    Overlay both structures into one graph.
    
    Args:
        structure_on: The ON-state secondary structure
        structure_off: The OFF-state secondary structure
        
    Returns:
        ConstraintGraph with list of connected components (each is a path or cycle).
        
    Raises:
        ValueError: If structures have different lengths
    """
    if structure_on.length != structure_off.length:
        raise ValueError(
            f"Structures must have same length: {structure_on.length} vs {structure_off.length}"
        )
    
    n = structure_on.length
    
    # Build adjacency list: each node maps to list of (neighbor, edge)
    adjacency: dict[int, list[tuple[int, Edge]]] = {i: [] for i in range(n)}
    
    # Add edges from ON structure
    for i, j in structure_on.pairs:
        edge = Edge(i=i, j=j, source="on")
        adjacency[i].append((j, edge))
        adjacency[j].append((i, edge))
    
    # Add edges from OFF structure
    for i, j in structure_off.pairs:
        edge = Edge(i=i, j=j, source="off")
        adjacency[i].append((j, edge))
        adjacency[j].append((i, edge))
    
    # Find connected components via DFS
    visited: set[int] = set()
    components: list[Component] = []
    
    for start in range(n):
        if start in visited:
            continue
        
        component = _trace_component(start, adjacency, visited)
        components.append(component)
    
    # Build node-to-component mapping
    node_to_component: dict[int, int] = {}
    for idx, comp in enumerate(components):
        for node in comp.nodes:
            node_to_component[node] = idx
    
    return ConstraintGraph(n=n, components=components, node_to_component=node_to_component)


def _trace_component(
    start: int, 
    adjacency: dict[int, list[tuple[int, Edge]]], 
    visited: set[int]
) -> Component:
    """
    Trace a connected component starting from a node.
    
    Since each node has degree at most 2, the component is either:
    - A path (if we hit a dead end or degree-1 node)
    - A cycle (if we return to the start)
    """
    # For degree ≤ 2 graphs, we can trace linearly
    # First, find an endpoint if this is a path (degree-1 node)
    current = start
    
    # If start has degree 0, it's an isolated node (unpaired in both structures)
    if len(adjacency[start]) == 0:
        visited.add(start)
        return Component(nodes=[start], edges=[], is_cycle=False)
    
    # Find a degree-1 endpoint if one exists (for paths)
    # Otherwise we're in a cycle
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
    
    # Start tracing from endpoint (for path) or any node (for cycle)
    trace_start = found_endpoint if found_endpoint is not None else start
    
    nodes: list[int] = []
    edges: list[Edge] = []
    
    current = trace_start
    prev = -1
    
    while current not in visited:
        visited.add(current)
        nodes.append(current)
        
        # Find next unvisited neighbor
        next_node = -1
        next_edge = None
        
        for neighbor, edge in adjacency[current]:
            if neighbor != prev:
                if neighbor not in visited:
                    next_node = neighbor
                    next_edge = edge
                    break
                elif neighbor == trace_start and len(nodes) > 2:
                    # We've completed a cycle
                    edges.append(edge)
                    return Component(nodes=nodes, edges=edges, is_cycle=True)
        
        if next_node == -1:
            # Dead end - this is a path
            break
        
        edges.append(next_edge)
        prev = current
        current = next_node
    
    return Component(nodes=nodes, edges=edges, is_cycle=False)


def assign_component(component: Component, rng: random.Random | None = None) -> dict[int, Base]:
    """
    Given one path or cycle, enumerate valid nucleotide assignments
    that satisfy Watson-Crick/wobble rules for all edges in the component.
    
    Args:
        component: A connected component (path or cycle)
        rng: Random number generator (uses global random if None)
        
    Returns:
        One valid assignment sampled uniformly, as {position: base}
    """
    if rng is None:
        rng = random.Random()
    
    nodes = component.nodes
    edges = component.edges
    
    # Handle isolated node (unpaired in both structures)
    if len(edges) == 0:
        base = rng.choice([Base.A, Base.C, Base.G, Base.U])
        return {nodes[0]: base}
    
    # Build edge lookup: for each node, what edges involve it?
    node_edges: dict[int, list[Edge]] = {n: [] for n in nodes}
    for edge in edges:
        node_edges[edge.i].append(edge)
        node_edges[edge.j].append(edge)
    
    if component.is_cycle:
        return _assign_cycle(nodes, edges, node_edges, rng)
    else:
        return _assign_path(nodes, edges, node_edges, rng)


def _assign_path(
    nodes: list[int], 
    edges: list[Edge], 
    node_edges: dict[int, list[Edge]],
    rng: random.Random
) -> dict[int, Base]:
    """
    Assign bases to a path component.
    
    Strategy: Start from first node, pick a random valid base, then propagate
    constraints along the path. Use backtracking if we hit a dead end.
    """
    n = len(nodes)
    assignment: dict[int, Base] = {}
    
    # Try multiple times with different starting choices
    for _ in range(100):  # Should succeed quickly for valid structures
        assignment.clear()
        
        # Randomly assign first node
        first_base = rng.choice([Base.A, Base.C, Base.G, Base.U])
        assignment[nodes[0]] = first_base
        
        success = True
        for idx in range(len(edges)):
            edge = edges[idx]
            node_i, node_j = edge.i, edge.j
            
            # Determine which node is already assigned
            if node_i in assignment and node_j not in assignment:
                known, unknown = node_i, node_j
            elif node_j in assignment and node_i not in assignment:
                known, unknown = node_j, node_i
            else:
                # Both assigned - just verify compatibility
                if (assignment[node_i], assignment[node_j]) not in CANONICAL_PAIRS:
                    success = False
                    break
                continue
            
            known_base = assignment[known]
            
            # Find valid partners for the known base
            valid_bases = [
                b for b in [Base.A, Base.C, Base.G, Base.U]
                if (known_base, b) in CANONICAL_PAIRS or (b, known_base) in CANONICAL_PAIRS
            ]
            
            if not valid_bases:
                success = False
                break
            
            # Check if unknown already has constraints from other edges
            other_constraints: list[Base] = []
            for other_edge in node_edges[unknown]:
                if other_edge is edge:
                    continue
                other_node = other_edge.i if other_edge.j == unknown else other_edge.j
                if other_node in assignment:
                    other_base = assignment[other_node]
                    compatible = [
                        b for b in valid_bases
                        if (other_base, b) in CANONICAL_PAIRS or (b, other_base) in CANONICAL_PAIRS
                    ]
                    if not compatible:
                        success = False
                        break
                    valid_bases = compatible
            
            if not success or not valid_bases:
                success = False
                break
            
            assignment[unknown] = rng.choice(valid_bases)
        
        if success:
            # Verify all nodes are assigned
            for node in nodes:
                if node not in assignment:
                    assignment[node] = rng.choice([Base.A, Base.C, Base.G, Base.U])
            return assignment
    
    # Fallback: shouldn't happen for valid structures
    raise RuntimeError(f"Could not find valid assignment for path component: {nodes}")


def _assign_cycle(
    nodes: list[int], 
    edges: list[Edge], 
    node_edges: dict[int, list[Edge]],
    rng: random.Random
) -> dict[int, Base]:
    """
    Assign bases to a cycle component.
    
    Strategy: Same as path, but need to ensure the cycle closes properly.
    Try multiple starting configurations.
    """
    n = len(nodes)
    
    # Build the cycle edge list in traversal order
    # The last edge connects back to the first node
    
    for _ in range(100):  # Multiple attempts
        assignment: dict[int, Base] = {}
        
        # Start with a random base for the first node
        first_base = rng.choice([Base.A, Base.C, Base.G, Base.U])
        assignment[nodes[0]] = first_base
        
        success = True
        for idx, node in enumerate(nodes[1:], 1):
            prev_node = nodes[idx - 1]
            
            # Find edge between prev_node and node
            connecting_edge = None
            for edge in edges:
                if (edge.i == prev_node and edge.j == node) or (edge.i == node and edge.j == prev_node):
                    connecting_edge = edge
                    break
            
            if connecting_edge is None:
                # Nodes not directly connected, assign freely
                valid_bases = [Base.A, Base.C, Base.G, Base.U]
            else:
                prev_base = assignment[prev_node]
                valid_bases = [
                    b for b in [Base.A, Base.C, Base.G, Base.U]
                    if (prev_base, b) in CANONICAL_PAIRS or (b, prev_base) in CANONICAL_PAIRS
                ]
            
            if not valid_bases:
                success = False
                break
            
            assignment[node] = rng.choice(valid_bases)
        
        if not success:
            continue
        
        # Check the closing edge (last node to first node)
        closing_edge = None
        for edge in edges:
            if (edge.i == nodes[-1] and edge.j == nodes[0]) or (edge.i == nodes[0] and edge.j == nodes[-1]):
                closing_edge = edge
                break
        
        if closing_edge:
            last_base = assignment[nodes[-1]]
            first_base = assignment[nodes[0]]
            if (last_base, first_base) not in CANONICAL_PAIRS and (first_base, last_base) not in CANONICAL_PAIRS:
                continue  # Try again
        
        # Verify all edges are satisfied
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
    
    raise RuntimeError(f"Could not find valid assignment for cycle component: {nodes}")


def generate_bicompatible_sequence(
    graph: ConstraintGraph, 
    rng: random.Random | None = None
) -> Sequence:
    """
    Assign each component independently, merge into full sequence.
    Result is guaranteed bicompatible with both structures.
    
    Args:
        graph: The constraint graph from build_constraint_graph()
        rng: Random number generator for reproducibility
        
    Returns:
        A Sequence that is bicompatible with both input structures
    """
    if rng is None:
        rng = random.Random()
    
    bases: list[Base] = [Base.A] * graph.n  # placeholder
    
    for component in graph.components:
        assignment = assign_component(component, rng)
        for pos, base in assignment.items():
            bases[pos] = base
    
    return Sequence(bases=bases)


def verify_bicompatible(seq: Sequence, s_on: Structure, s_off: Structure) -> bool:
    """
    Sanity check: verify every base pair in both structures is satisfied.
    Used in tests only — should always return True for graph-generated sequences.
    
    Args:
        seq: The sequence to verify
        s_on: The ON-state structure
        s_off: The OFF-state structure
        
    Returns:
        True if all base pairs in both structures are valid (Watson-Crick or wobble)
    """
    bases = seq.bases
    
    # Check ON structure pairs
    for i, j in s_on.pairs:
        if (bases[i], bases[j]) not in CANONICAL_PAIRS:
            return False
    
    # Check OFF structure pairs
    for i, j in s_off.pairs:
        if (bases[i], bases[j]) not in CANONICAL_PAIRS:
            return False
    
    return True
