"""
io_utils.py — File I/O utilities for riboswitch design.
"""

import csv
from pathlib import Path
from ribo_switch.scorer import CandidateResult


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
    results: list[CandidateResult],
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
