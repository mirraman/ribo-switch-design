"""
io_utils.py — File I/O utilities for riboswitch design.
"""

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ribo_switch.scorer import CandidateResult
    from ribo_switch.nsga2 import Candidate


def read_structure_pair(filepath: str) -> tuple[str, str]:
    """Read two dot-bracket structures from a file.

    Accepted format (one structure per line, ignoring comments/blanks):
        # ON-state
        ((((....))))........
        # OFF-state
        ........((((....))))

    Returns:
        Tuple of (structure1, structure2) as dot-bracket strings.
    """
    lines: list[str] = []
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                lines.append(stripped)

    if len(lines) < 2:
        raise ValueError(
            f"Need at least 2 structure lines in {filepath}, got {len(lines)}"
        )

    s1, s2 = lines[0], lines[1]

    # Validate characters
    valid = set('.()[]{}')
    for i, s in enumerate((s1, s2), 1):
        bad = set(s) - valid
        if bad:
            raise ValueError(
                f"Invalid characters in structure {i}: {bad}"
            )

    if len(s1) != len(s2):
        raise ValueError(
            f"Structure lengths differ: {len(s1)} vs {len(s2)}"
        )

    return s1, s2


def write_results(
    results: list,
    filepath: str,
    fmt: str = "tsv",
) -> None:
    """Write ranked results to a file.

    Args:
        results: List of CandidateResult (assumed sorted by score).
        filepath: Output file path.
        fmt: "tsv" or "csv".
    """
    delimiter = '\t' if fmt == 'tsv' else ','
    header = [
        'rank', 'sequence', 'energy_s1', 'energy_s2',
        'mfe_energy', 'gap_s1', 'gap_s2', 'combined_score',
    ]

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(header)
        for i, r in enumerate(results, 1):
            writer.writerow([
                i, r.sequence,
                f'{r.energy_s1:.2f}', f'{r.energy_s2:.2f}',
                f'{r.mfe_energy:.2f}', f'{r.gap_s1:.2f}',
                f'{r.gap_s2:.2f}', f'{r.combined_score:.2f}',
            ])


def write_pareto_json(
    candidates: list,
    filepath: str,
    structure_on: str = "",
    structure_off: str = "",
) -> None:
    """Write Pareto front to JSON file.

    Args:
        candidates: List of NSGA-II Candidates
        filepath: Output file path
        structure_on: ON-state structure (for metadata)
        structure_off: OFF-state structure (for metadata)
    """
    # Sort by combined gap
    sorted_cands = sorted(candidates, key=lambda c: c.gap_on + c.gap_off)
    
    output = {
        "metadata": {
            "structure_on": structure_on,
            "structure_off": structure_off,
            "count": len(candidates),
        },
        "candidates": []
    }
    
    for i, c in enumerate(sorted_cands):
        output["candidates"].append({
            "rank": i + 1,
            "sequence": str(c.sequence),
            "energy_on_kcal": c.e_on / 100.0,
            "energy_off_kcal": c.e_off / 100.0,
            "mfe_kcal": c.mfe / 100.0,
            "mfe_structure": c.mfe_structure,
            "gap_on_kcal": c.gap_on / 100.0,
            "gap_off_kcal": c.gap_off / 100.0,
            "stability_kcal": c.stability / 100.0,
            "bp_dist_on": c.bp_dist_on,
            "bp_dist_off": c.bp_dist_off,
            "bp_f1_on": round(c.bp_f1_on, 4),
            "bp_f1_off": round(c.bp_f1_off, 4),
            "mfe_matches_on": c.bp_dist_on == 0,
            "pareto_rank": c.rank,
            "crowding_distance": c.crowding_distance,
        })
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)


def read_pareto_json(filepath: str) -> dict:
    """Read Pareto front from JSON file.

    Args:
        filepath: Input file path
        
    Returns:
        Dictionary with metadata and candidate list
    """
    with open(filepath) as f:
        return json.load(f)


def write_fasta(
    candidates: list,
    filepath: str,
    include_energy: bool = True,
) -> None:
    """Write sequences to FASTA format.

    Args:
        candidates: List of candidates (NSGA-II or CandidateResult)
        filepath: Output file path
        include_energy: If True, include energy info in header
    """
    with open(filepath, 'w') as f:
        for i, c in enumerate(candidates, 1):
            # Get sequence
            if hasattr(c, 'sequence'):
                if hasattr(c.sequence, 'bases'):
                    seq = str(c.sequence)
                else:
                    seq = c.sequence
            else:
                continue
            
            # Build header
            if include_energy:
                if hasattr(c, 'gap_on'):
                    header = f">candidate_{i} gap_on={c.gap_on/100:.2f} gap_off={c.gap_off/100:.2f}"
                else:
                    header = f">candidate_{i} gap_s1={c.gap_s1:.2f} gap_s2={c.gap_s2:.2f}"
            else:
                header = f">candidate_{i}"
            
            f.write(f"{header}\n")
            f.write(f"{seq}\n")
