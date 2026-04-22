from __future__ import annotations
from typing import TYPE_CHECKING
import json

if TYPE_CHECKING:
    from ribo_switch.nsga2 import Candidate
    from ribo_switch.scorer import CandidateResult


def pareto_front_data(candidates: list) -> dict:
    data = {
        "gap_on": [],
        "gap_off": [],
        "stability": [],
        "rank": [],
        "sequence": [],
    }
    
    for c in candidates:
        if hasattr(c, 'gap_on'):
            data["gap_on"].append(c.gap_on / 100.0)  # Convert to kcal/mol
            data["gap_off"].append(c.gap_off / 100.0)
            data["stability"].append(c.stability / 100.0)
            data["rank"].append(c.rank)
            data["sequence"].append(str(c.sequence))
        else:
            # CandidateResult
            data["gap_on"].append(c.gap_s1)
            data["gap_off"].append(c.gap_s2)
            data["stability"].append(c.stability)
            data["rank"].append(c.pareto_rank or 0)
            data["sequence"].append(c.sequence)
    
    return data


def convergence_data(history: list[list]) -> dict:
    data = {
        "generation": [],
        "front_size": [],
        "mean_gap_on": [],
        "mean_gap_off": [],
        "min_gap_on": [],
        "min_gap_off": [],
        "mean_combined": [],
    }
    
    for gen, front in enumerate(history):
        if not front:
            continue
            
        data["generation"].append(gen)
        data["front_size"].append(len(front))
        
        # Handle both Candidate and CandidateResult
        if hasattr(front[0], 'gap_on'):
            gaps_on = [c.gap_on / 100.0 for c in front]
            gaps_off = [c.gap_off / 100.0 for c in front]
        else:
            gaps_on = [c.gap_s1 for c in front]
            gaps_off = [c.gap_s2 for c in front]
        
        data["mean_gap_on"].append(sum(gaps_on) / len(gaps_on))
        data["mean_gap_off"].append(sum(gaps_off) / len(gaps_off))
        data["min_gap_on"].append(min(gaps_on))
        data["min_gap_off"].append(min(gaps_off))
        data["mean_combined"].append(
            sum(g1 + g2 for g1, g2 in zip(gaps_on, gaps_off)) / len(gaps_on)
        )
    
    return data


def component_diversity_data(candidates: list, graph) -> dict:

    n_components = len(graph.components)
    
    # For each component, collect all unique assignments
    component_variants: list[set] = [set() for _ in range(n_components)]
    
    for c in candidates:
        if hasattr(c, 'individual'):
            ind = c.individual
        else:
            continue
            
        for idx, assignment in ind.component_assignments.items():
            # Convert assignment to hashable form
            key = tuple(sorted(assignment.items()))
            component_variants[idx].add(key)
    
    return {
        "component_index": list(range(n_components)),
        "component_size": [len(comp.nodes) for comp in graph.components],
        "unique_assignments": [len(variants) for variants in component_variants],
    }


def format_pareto_table(candidates: list, max_rows: int = 10) -> str:

    if not candidates:
        return "No candidates in Pareto front."
    
    # Sort by gap_on + gap_off
    if hasattr(candidates[0], 'gap_on'):
        sorted_cands = sorted(candidates, key=lambda c: c.gap_on + c.gap_off)
    else:
        sorted_cands = sorted(candidates, key=lambda c: c.gap_s1 + c.gap_s2)
    
    lines = []
    header = (
        f"{'#':>3} {'Gap_ON':>8} {'Gap_OFF':>8} "
        f"{'bpΔ_ON':>7} {'bpΔ_OFF':>7} "
        f"{'Stability':>10} {'Sequence':>20}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for i, c in enumerate(sorted_cands[:max_rows]):
        if hasattr(c, 'gap_on'):
            gap_on    = c.gap_on / 100.0
            gap_off   = c.gap_off / 100.0
            stability = c.stability / 100.0
            seq       = str(c.sequence)
            bp_on     = c.bp_dist_on
            bp_off    = c.bp_dist_off
        else:
            gap_on    = c.gap_s1
            gap_off   = c.gap_s2
            stability = c.stability
            seq       = c.sequence
            bp_on     = getattr(c, 'bp_dist_on', "-")
            bp_off    = getattr(c, 'bp_dist_off', "-")

        # Truncate sequence if needed
        if len(seq) > 20:
            seq = seq[:17] + "..."

        bp_on_str  = f"{bp_on:>7}"  if isinstance(bp_on,  int) else f"{'?':>7}"
        bp_off_str = f"{bp_off:>7}" if isinstance(bp_off, int) else f"{'?':>7}"

        lines.append(
            f"{i+1:>3} {gap_on:>8.2f} {gap_off:>8.2f} "
            f"{bp_on_str} {bp_off_str} "
            f"{stability:>10.2f} {seq:>20}"
        )
    
    if len(sorted_cands) > max_rows:
        lines.append(f"... and {len(sorted_cands) - max_rows} more candidates")
    
    return "\n".join(lines)


def export_pareto_json(candidates: list, filepath: str) -> None:

    data = pareto_front_data(candidates)
    
    # Add full sequences
    records = []
    for i in range(len(data["sequence"])):
        records.append({
            "sequence": data["sequence"][i],
            "gap_on_kcal": data["gap_on"][i],
            "gap_off_kcal": data["gap_off"][i],
            "stability_kcal": data["stability"][i],
            "pareto_rank": data["rank"][i],
        })
    
    with open(filepath, 'w') as f:
        json.dump(records, f, indent=2)


def try_matplotlib_plot(candidates: list, output_path: str = None) -> bool:

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    
    data = pareto_front_data(candidates)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        data["gap_on"],
        data["gap_off"],
        c=data["stability"],
        cmap="viridis_r",
        alpha=0.7,
        s=50,
    )
    
    ax.set_xlabel("Gap_ON (kcal/mol)", fontsize=12)
    ax.set_ylabel("Gap_OFF (kcal/mol)", fontsize=12)
    ax.set_title("Riboswitch Design Pareto Front", fontsize=14)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Stability (kcal/mol)", fontsize=10)
    
    # Mark the ideal point
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.plot(0, 0, 'r*', markersize=15, label='Ideal (0, 0)')
    ax.legend()
    
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return True


def try_convergence_plot(history: list[list], output_path: str = None) -> bool:

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    
    data = convergence_data(history)
    
    if not data["generation"]:
        return False
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mean gaps over time
    ax1 = axes[0, 0]
    ax1.plot(data["generation"], data["mean_gap_on"], label="Mean Gap_ON", marker='o', markersize=3)
    ax1.plot(data["generation"], data["mean_gap_off"], label="Mean Gap_OFF", marker='s', markersize=3)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Gap (kcal/mol)")
    ax1.set_title("Mean Gap Values")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Min gaps over time
    ax2 = axes[0, 1]
    ax2.plot(data["generation"], data["min_gap_on"], label="Min Gap_ON", marker='o', markersize=3)
    ax2.plot(data["generation"], data["min_gap_off"], label="Min Gap_OFF", marker='s', markersize=3)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Gap (kcal/mol)")
    ax2.set_title("Best Gap Values")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Combined gap
    ax3 = axes[1, 0]
    ax3.plot(data["generation"], data["mean_combined"], color='purple', marker='o', markersize=3)
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Gap_ON + Gap_OFF (kcal/mol)")
    ax3.set_title("Mean Combined Gap")
    ax3.grid(True, alpha=0.3)
    
    # Front size
    ax4 = axes[1, 1]
    ax4.plot(data["generation"], data["front_size"], color='green', marker='o', markersize=3)
    ax4.set_xlabel("Generation")
    ax4.set_ylabel("Number of Solutions")
    ax4.set_title("Pareto Front Size")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return True
