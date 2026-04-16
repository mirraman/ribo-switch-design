"""
brpf.py — Bicompatible Restricted Partition Function (BRPF).

The problem with McCaskill P(S_ON):
    For a 74-nt sequence there are ~10^30 possible secondary structures.
    P(S_ON) = exp(-E_ON/kT) / Z_McCaskill ≈ 10^-30.
    This is technically correct but completely useless as a design objective —
    it is effectively 0 for every candidate, so the optimizer gets no signal.

What BRPF asks instead:
    "Among all sequences that can form BOTH S_ON and S_OFF (bicompatible
    sequences), how much thermodynamic weight goes to S_ON vs S_OFF?"

Two implementations are provided:

1. two_state_score  [O(1) per candidate]
   ----------------------------------------
   Restricts the partition function to just the two target states:

       switching_score = exp(-E_ON/kT) / (exp(-E_ON/kT) + exp(-E_OFF/kT))
                       = sigmoid((E_OFF - E_ON) / kT)

   This is the exact Boltzmann probability of S_ON in a two-state system
   {S_ON, S_OFF}.  It uses already-computed E_ON and E_OFF, adds zero cost,
   and returns a meaningful value in (0, 1) for any sequence length.
   Used as the third NSGA-II objective.

2. brpf  [O(K × M) per candidate, where K = #components, M = avg assignments]
   -------------------------------------------------------------------------
   Marginalizes over ALL bicompatible sequences using the constraint graph
   component decomposition.  For each component k:

       Z_k(S_ON) = Σ_{valid assignments a of k} exp(-E(seq_k^a, S_ON) / kT)

   where seq_k^a is the candidate sequence with component k substituted by
   assignment a (all other components held fixed).  The per-component
   Boltzmann sums are combined via a mean-field product approximation:

       ln Z_bico(S_ON) ≈ Σ_k ln Z_k(S_ON)

   Note: cross-component stacking terms are included in each E(seq_k^a, ...)
   evaluation (they are not double-counted within a single component's sum,
   but the product formula does over-weight cross-component terms across
   components).  This is an accepted mean-field approximation — it is
   symmetric between S_ON and S_OFF, so the switching score

       switching_score = Z_bico(S_ON) / (Z_bico(S_ON) + Z_bico(S_OFF))

   remains meaningful for ranking candidates.

   This is the thesis-level contribution: a computationally tractable,
   sequence-length-independent switching metric restricted to the
   bicompatible sequence space.

Reference: This work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ribo_switch.types import Base, Sequence, Structure, CANONICAL_PAIRS
from ribo_switch.graph import Component, ConstraintGraph
from ribo_switch.energy import eval_energy
from ribo_switch.turner import TurnerParams

_ALL_BASES = [Base.A, Base.C, Base.G, Base.U]


def kT_at(T: float = 37.0) -> float:
    """Thermal energy kT in units of 0.01 kcal/mol at temperature T (Celsius).

    R = 1.987204 cal/(mol·K) → kT [0.01 kcal/mol] = 0.1987204 × (T + 273.15)
    At 37 °C: kT ≈ 61.63
    """
    return 0.1987204 * (T + 273.15)


# ---------------------------------------------------------------------------
# 1. Two-state switching score  (O(1))
# ---------------------------------------------------------------------------

def two_state_score(e_on: int, e_off: int, kT: float) -> float:
    """Boltzmann probability of S_ON in a two-state {S_ON, S_OFF} model.

        switching_score = exp(-E_ON/kT) / (exp(-E_ON/kT) + exp(-E_OFF/kT))
                        = 1 / (1 + exp((E_ON - E_OFF) / kT))

    Args:
        e_on:  Energy of the sequence in S_ON (integer, 0.01 kcal/mol units).
        e_off: Energy of the sequence in S_OFF (same units).
        kT:    Thermal energy (same units, use kT_at(37.0) for 37 °C).

    Returns:
        Score in (0, 1).  > 0.5 means the molecule thermodynamically prefers
        S_ON over S_OFF.
    """
    delta = (e_on - e_off) / kT   # positive → prefers S_OFF
    if delta > 500.0:
        return 0.0
    if delta < -500.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(delta))


# ---------------------------------------------------------------------------
# 2. Component assignment enumeration
# ---------------------------------------------------------------------------

def enumerate_component_assignments(component: Component) -> list[dict[int, Base]]:
    """Return all valid base assignments for a single constraint graph component.

    A valid assignment satisfies every edge constraint: for each edge (i, j)
    in the component, (seq[i], seq[j]) must be a canonical Watson-Crick or
    wobble pair.

    Isolated nodes (no edges) have 4 valid assignments (any base).
    A 2-node component with one edge has exactly 6 valid assignments.
    Larger components typically have 4–36 valid assignments.

    Args:
        component: A connected component from build_constraint_graph().

    Returns:
        List of dicts mapping position → Base, one entry per valid assignment.
        Never empty (every component has at least one valid assignment).
    """
    nodes = component.nodes
    edges = component.edges

    if not edges:
        return [{nodes[0]: b} for b in _ALL_BASES]

    # Adjacency: position → list of constrained neighbor positions.
    # Works for both paths and cycles (the closing edge for cycles is already
    # in component.edges and will appear here naturally).
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
        # Filter by every already-assigned neighbor's pair constraint.
        valid = list(_ALL_BASES)
        for nb in adj[node]:
            if nb in assignment:
                nb_base = assignment[nb]
                valid = [
                    b for b in valid
                    if (nb_base, b) in CANONICAL_PAIRS or (b, nb_base) in CANONICAL_PAIRS
                ]
        for b in valid:
            assignment[node] = b
            backtrack(idx + 1, assignment)
            del assignment[node]

    backtrack(0, {})
    return results


# ---------------------------------------------------------------------------
# 3. Full component-level BRPF  (O(K × M))
# ---------------------------------------------------------------------------

@dataclass
class BRPFResult:
    """Result of the Bicompatible Restricted Partition Function.

    Attributes:
        log_Z_on:        ln Z_bico(S_ON)  — summed log Boltzmann weight toward ON.
        log_Z_off:       ln Z_bico(S_OFF) — summed log Boltzmann weight toward OFF.
        switching_score: Z_bico(S_ON) / (Z_bico(S_ON) + Z_bico(S_OFF)) ∈ (0, 1).
        kT:              Thermal energy used (0.01 kcal/mol units).
        n_components:    Number of constraint graph components processed.
    """
    log_Z_on: float
    log_Z_off: float
    switching_score: float
    kT: float
    n_components: int


def brpf(
    seq: Sequence,
    graph: ConstraintGraph,
    s_on: Structure,
    s_off: Structure,
    params: TurnerParams,
    T: float = 37.0,
) -> BRPFResult:
    """Compute the Bicompatible Restricted Partition Function for a candidate.

    For each constraint graph component k, enumerates all m_k valid base
    assignments.  For each assignment a, builds the modified sequence
    (component k = a, all other components held at seq's values) and
    evaluates its energy in S_ON and S_OFF.

    The per-component log Boltzmann sums are accumulated via log-sum-exp
    for numerical stability, then summed across components (mean-field
    product approximation).

    Complexity: O(K × M × n²) where K = #components (~n/3 for typical
    riboswitches), M = avg valid assignments per component (~6), n = sequence
    length.  Much cheaper than O(n³) McCaskill per candidate.

    Args:
        seq:    Reference candidate sequence (component assignments follow this).
        graph:  Constraint graph from build_constraint_graph(s_on, s_off).
        s_on:   ON-state target structure.
        s_off:  OFF-state target structure.
        params: Turner 2004 nearest-neighbour parameters.
        T:      Temperature in Celsius (default 37.0).

    Returns:
        BRPFResult with log_Z_on, log_Z_off, switching_score, kT, n_components.
    """
    kT = kT_at(T)
    ref_bases = list(seq.bases)

    log_Z_on = 0.0
    log_Z_off = 0.0

    for component in graph.components:
        all_assignments = enumerate_component_assignments(component)

        on_terms: list[float] = []
        off_terms: list[float] = []

        for assignment in all_assignments:
            # Build modified sequence: substitute this component's assignment.
            mod_bases = list(ref_bases)
            for pos, base in assignment.items():
                mod_bases[pos] = base
            mod_seq = Sequence(bases=mod_bases)

            e_on = eval_energy(mod_seq, s_on, params)
            e_off = eval_energy(mod_seq, s_off, params)

            # Use -E/kT as the exponent (lower energy → larger Boltzmann weight).
            on_terms.append(-e_on / kT)
            off_terms.append(-e_off / kT)

        # log-sum-exp for numerical stability.
        log_Z_on += _log_sum_exp(on_terms)
        log_Z_off += _log_sum_exp(off_terms)

    # switching_score = 1 / (1 + exp(log_Z_off - log_Z_on))
    delta = log_Z_off - log_Z_on
    if delta > 500.0:
        switching_score = 0.0
    elif delta < -500.0:
        switching_score = 1.0
    else:
        switching_score = 1.0 / (1.0 + math.exp(delta))

    return BRPFResult(
        log_Z_on=log_Z_on,
        log_Z_off=log_Z_off,
        switching_score=switching_score,
        kT=kT,
        n_components=len(graph.components),
    )


def _log_sum_exp(values: list[float]) -> float:
    """Numerically stable log(Σ exp(x_i))."""
    if not values:
        return float("-inf")
    m = max(values)
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(sum(math.exp(v - m) for v in values))
