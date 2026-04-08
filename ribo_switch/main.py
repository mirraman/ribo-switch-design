"""
main.py — CLI entry point for riboswitch sequence design.

Usage:
    ribo-design "(((....)))..." "....(((....)))" --population 100 --generations 200
    ribo-design --input structures.txt --output results.json
"""

import sys
import time
import click

from ribo_switch.turner import TurnerParams
from ribo_switch.nsga2 import nsga2, summarize_pareto_front
from ribo_switch.io_utils import read_structure_pair, write_results, write_pareto_json
from ribo_switch.viz import format_pareto_table


@click.command()
@click.argument('structure1', required=False)
@click.argument('structure2', required=False)
@click.option('--input', '-i', 'input_file', help='Read structures from file')
@click.option('--output', '-o', default='results.json', help='Output file (JSON)')
@click.option('--population', '-p', default=100, help='Population size')
@click.option('--generations', '-g', default=200, help='Number of generations')
@click.option('--mutation-rate', '-m', default=0.1, help='Mutation rate (0-1)')
@click.option('--seed', '-s', default=None, type=int, help='Random seed')
@click.option('--verbose', '-v', is_flag=True, help='Show progress')
@click.option('--top-k', '-k', default=20, help='Top candidates to display')
def design(
    structure1, structure2, input_file, output,
    population, generations, mutation_rate, seed, verbose, top_k,
):
    """Design riboswitch sequences using NSGA-II evolutionary optimization.

    Provide structures as arguments or via --input file.

    Examples:

        ribo-design "(((....))).." "..(((....).))" -p 100 -g 200 -v

        ribo-design -i my_structures.txt -o results.json -p 50 -g 100 -v
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

    click.echo(f"Riboswitch Design (NSGA-II) — {len(s1)} nt")
    click.echo(f"  S_ON:  {s1}")
    click.echo(f"  S_OFF: {s2}")
    click.echo(f"  Population: {population}, Generations: {generations}")
    click.echo()

    params = TurnerParams.turner2004()

    # Progress callback
    last_report = [0]
    def progress_callback(gen, front):
        if verbose and (gen % 10 == 0 or gen == generations - 1):
            summary = summarize_pareto_front(front)
            click.echo(f"  Gen {gen:4d}: {summary['count']:3d} solutions, "
                       f"mean gap = {summary.get('gap_on_mean', 0)/100:.2f} + {summary.get('gap_off_mean', 0)/100:.2f}")
            last_report[0] = gen

    start = time.time()
    
    if verbose:
        click.echo("Running NSGA-II optimization...")
    
    pareto_front = nsga2(
        structure_on=s1,
        structure_off=s2,
        population_size=population,
        n_generations=generations,
        mutation_rate=mutation_rate,
        params=params,
        seed=seed,
        callback=progress_callback if verbose else None,
    )
    
    elapsed = time.time() - start

    if not pareto_front:
        click.echo("No valid candidates found.", err=True)
        sys.exit(1)

    # Display summary
    summary = summarize_pareto_front(pareto_front)
    
    click.echo(f"\n{'='*70}")
    click.echo(f"Optimization complete ({elapsed:.1f}s)")
    click.echo(f"{'='*70}")
    click.echo(f"Pareto front: {summary['count']} solutions")
    click.echo(f"  Gap_ON:  min={summary['gap_on_min']/100:.2f}, mean={summary['gap_on_mean']/100:.2f}, max={summary['gap_on_max']/100:.2f} kcal/mol")
    click.echo(f"  Gap_OFF: min={summary['gap_off_min']/100:.2f}, mean={summary['gap_off_mean']/100:.2f}, max={summary['gap_off_max']/100:.2f} kcal/mol")
    click.echo(f"  Ideal (both gaps = 0): {summary['ideal_count']} solutions")
    click.echo()

    # Display top candidates
    click.echo(f"Top {min(len(pareto_front), top_k)} candidates:")
    click.echo(format_pareto_table(pareto_front, max_rows=top_k))
    click.echo()

    # Write to file
    write_pareto_json(pareto_front, output, s1, s2)
    click.echo(f"Results written to {output}")


def main():
    """Entry point."""
    design()


if __name__ == '__main__':
    main()
