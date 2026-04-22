import sys
import time
import click
from ribo_switch.turner import TurnerParams
from ribo_switch.nsga2 import nsga2, summarize_pareto_front, filter_by_structure
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
@click.option('--bp-distance-obj/--no-bp-distance-obj', default=True, show_default=True, help='Include base-pair distance to S_ON as a 4th NSGA-II objective so the optimizer selects sequences whose MFE structure matches S_ON. Use --no-bp-distance-obj to restore the original 3-objective mode.')
@click.option('--require-exact-on', is_flag=True, default=False, help='After optimization, filter the Pareto front to candidates whose MFE structure exactly matches S_ON (bp_dist_on == 0). Falls back to the full front with a warning if the filter leaves nothing.')
def design(structure1, structure2, input_file, output, population, generations, mutation_rate, seed, verbose, top_k, bp_distance_obj, require_exact_on):
    if input_file:
        try:
            s1, s2 = read_structure_pair(input_file)
        except (FileNotFoundError, ValueError) as e:
            click.echo(f'Error: {e}', err=True)
            sys.exit(1)
    elif structure1 and structure2:
        s1, s2 = (structure1, structure2)
    else:
        click.echo('Error: provide two structures as arguments or use --input', err=True)
        sys.exit(1)
    if len(s1) != len(s2):
        click.echo(f'Error: structure lengths differ ({len(s1)} vs {len(s2)})', err=True)
        sys.exit(1)
    click.echo(f'Riboswitch Design (NSGA-II) — {len(s1)} nt')
    click.echo(f'  S_ON:  {s1}')
    click.echo(f'  S_OFF: {s2}')
    click.echo(f'  Population: {population}, Generations: {generations}')
    click.echo(f'  Structure objective: {('ON (4-objective)' if bp_distance_obj else 'OFF (3-objective)')}')
    click.echo()
    params = TurnerParams.turner2004()
    last_report = [0]

    def progress_callback(gen, front):
        if verbose and (gen % 10 == 0 or gen == generations - 1):
            summary = summarize_pareto_front(front)
            exact_on = sum((1 for c in front if c.bp_dist_on == 0))
            click.echo(f'  Gen {gen:4d}: {summary['count']:3d} solutions, mean gap = {summary.get('gap_on_mean', 0) / 100:.2f} + {summary.get('gap_off_mean', 0) / 100:.2f}  exact_ON={exact_on}')
            last_report[0] = gen
    start = time.time()
    if verbose:
        click.echo('Running NSGA-II optimization...')
    pareto_front = nsga2(structure_on=s1, structure_off=s2, population_size=population, n_generations=generations, mutation_rate=mutation_rate, params=params, seed=seed, callback=progress_callback if verbose else None, include_structure_objective=bp_distance_obj)
    elapsed = time.time() - start
    if not pareto_front:
        click.echo('No valid candidates found.', err=True)
        sys.exit(1)
    if require_exact_on:
        filtered = filter_by_structure(pareto_front, max_bp_dist_on=0)
        if filtered:
            click.echo(f'  --require-exact-on: kept {len(filtered)} / {len(pareto_front)} candidates (bp_dist_on == 0)')
            pareto_front = filtered
        else:
            click.echo('  Warning: --require-exact-on found no exact matches; showing full Pareto front instead.', err=True)
    summary = summarize_pareto_front(pareto_front)
    exact_on_count = sum((1 for c in pareto_front if c.bp_dist_on == 0))
    click.echo(f'\n{'=' * 70}')
    click.echo(f'Optimization complete ({elapsed:.1f}s)')
    click.echo(f'{'=' * 70}')
    click.echo(f'Pareto front: {summary['count']} solutions')
    click.echo(f'  Gap_ON:  min={summary['gap_on_min'] / 100:.2f}, mean={summary['gap_on_mean'] / 100:.2f}, max={summary['gap_on_max'] / 100:.2f} kcal/mol')
    click.echo(f'  Gap_OFF: min={summary['gap_off_min'] / 100:.2f}, mean={summary['gap_off_mean'] / 100:.2f}, max={summary['gap_off_max'] / 100:.2f} kcal/mol')
    click.echo(f'  Ideal (both gaps = 0): {summary['ideal_count']} solutions')
    click.echo(f'  MFE matches S_ON exactly (bp_dist_on=0): {exact_on_count} solutions')
    click.echo()
    click.echo(f'Top {min(len(pareto_front), top_k)} candidates:')
    click.echo(format_pareto_table(pareto_front, max_rows=top_k))
    click.echo()
    write_pareto_json(pareto_front, output, s1, s2)
    click.echo(f'Results written to {output}')

@click.command()
@click.argument('sequence')
@click.argument('struct_on')
@click.argument('struct_off', required=False, default=None)
def verify(sequence: str, struct_on: str, struct_off: str | None):
    from ribo_switch.verify import verify_sequence, verify_against_both
    params = TurnerParams.turner2004()
    seq_clean = sequence.upper().replace('T', 'U')
    if len(seq_clean) != len(struct_on):
        click.echo(f'Error: sequence length {len(seq_clean)} != structure length {len(struct_on)}', err=True)
        sys.exit(1)
    if struct_off is not None and len(seq_clean) != len(struct_off):
        click.echo(f'Error: sequence length {len(seq_clean)} != S_OFF length {len(struct_off)}', err=True)
        sys.exit(1)
    sep = '=' * 60
    if struct_off is None:
        report = verify_sequence(seq_clean, struct_on, params=params, label='ON')
        click.echo(sep)
        for line in report.summary_lines():
            click.echo(line)
        click.echo(sep)
    else:
        rep_on, rep_off = verify_against_both(seq_clean, struct_on, struct_off, params=params)
        click.echo(sep)
        click.echo('=== S_ON Verification ===')
        for line in rep_on.summary_lines():
            click.echo(line)
        click.echo()
        click.echo('=== S_OFF Verification ===')
        for line in rep_off.summary_lines():
            click.echo(line)
        on_exact = rep_on.matches_exactly
        off_exact = rep_off.matches_exactly
        if on_exact and off_exact:
            verdict = 'MATCHES_BOTH'
        elif on_exact:
            verdict = 'MATCHES_ON'
        elif off_exact:
            verdict = 'MATCHES_OFF'
        else:
            verdict = 'NO_MATCH'
        click.echo()
        click.echo(f'Overall verdict: {verdict}')
        click.echo(sep)

def main():
    design()

def main_verify():
    verify()
if __name__ == '__main__':
    main()
