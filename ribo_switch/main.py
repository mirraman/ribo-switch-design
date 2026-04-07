"""
main.py — CLI entry point for riboswitch sequence design.

Usage:
    ribo-design "(((....)))..." "....(((....)))" --samples 1000 --top-k 10
    ribo-design --input structures.txt --output results.tsv
"""

import sys
import time
import click

from ribo_switch.turner import TurnerParams
from ribo_switch.optimizer import RiboswitchOptimizer, OptimizerConfig
from ribo_switch.io_utils import read_structure_pair, write_results


@click.command()
@click.argument('structure1', required=False)
@click.argument('structure2', required=False)
@click.option('--input', '-i', 'input_file', help='Read structures from file')
@click.option('--output', '-o', default='results.tsv', help='Output file')
@click.option('--samples', '-n', default=1000, help='Boltzmann samples')
@click.option('--top-k', '-k', default=20, help='Top candidates to report')
@click.option('--mutations', '-m', default=50, help='Mutations per seed')
@click.option('--temperature', '-t', default=37.0, help='Temperature (°C)')
@click.option('--seed', '-s', default=None, type=int, help='Random seed')
@click.option('--verbose', '-v', is_flag=True, help='Show progress')
def design(
    structure1, structure2, input_file, output,
    samples, top_k, mutations, temperature, seed, verbose,
):
    """Design riboswitch sequences for two target structures.

    Provide structures as arguments or via --input file.

    Examples:

        ribo-design "(((....))).." "..(((....).))" -n 500 -k 10 -v

        ribo-design -i my_structures.txt -o results.tsv -n 2000 -v
    """
    # Get structures
    if input_file:
        try:
            s1, s2 = read_structure_pair(input_file)
        except (FileNotFoundError, ValueError) as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    elif structure1 and structure2:
        s1, s2 = structure1, structure2
    else:
        click.echo("Error: provide two structures as arguments or use --input", err=True)
        sys.exit(1)

    # Validate lengths
    if len(s1) != len(s2):
        click.echo(f"Error: structure lengths differ ({len(s1)} vs {len(s2)})", err=True)
        sys.exit(1)

    click.echo(f"Riboswitch Design — {len(s1)} nt")
    click.echo(f"  S1: {s1}")
    click.echo(f"  S2: {s2}")
    click.echo()

    config = OptimizerConfig(
        n_samples=samples,
        top_k=top_k,
        n_mutations=mutations,
        temperature=temperature,
        seed=seed,
    )

    params = TurnerParams.turner2004()

    start = time.time()
    optimizer = RiboswitchOptimizer(s1, s2, params, config)
    results = optimizer.run(verbose=verbose)
    elapsed = time.time() - start

    if not results:
        click.echo("No valid candidates found.", err=True)
        sys.exit(1)

    # Display top results
    click.echo(f"\n{'='*70}")
    click.echo(f"Top {min(len(results), top_k)} candidates ({elapsed:.1f}s)")
    click.echo(f"{'='*70}")
    click.echo(f"{'#':>3}  {'Sequence':<{len(s1)+2}} {'E1':>6} {'E2':>6} "
               f"{'MFE':>6} {'Gap1':>5} {'Gap2':>5} {'Score':>7}")
    click.echo(f"{'-'*70}")

    for i, r in enumerate(results[:top_k], 1):
        click.echo(
            f"{i:3d}  {r.sequence:<{len(s1)+2}} "
            f"{r.energy_s1:6.2f} {r.energy_s2:6.2f} "
            f"{r.mfe_energy:6.2f} {r.gap_s1:5.2f} {r.gap_s2:5.2f} "
            f"{r.combined_score:7.2f}"
        )

    # Write to file
    write_results(results, output)
    click.echo(f"\nResults written to {output}")


def main():
    """Entry point."""
    design()


if __name__ == '__main__':
    main()
