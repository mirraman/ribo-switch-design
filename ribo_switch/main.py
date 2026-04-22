import json
import time
import click
from ribo_switch.turner import TurnerParams
from ribo_switch.nsga2 import nsga2, summarize_pareto_front, filter_by_structure


def _is_dot_bracket(line: str) -> bool:
    return bool(line) and set(line) <= {'.', '(', ')'}


def _read_structure_pair(input_file: str) -> tuple[str, str]:
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f'Input file not found: {input_file}')
    structures = [line for line in lines if _is_dot_bracket(line)]
    if len(structures) < 2:
        raise ValueError('Input file must contain at least two dot-bracket structure lines')
    return (structures[0], structures[1])


def _format_pareto_table(front: list, max_rows: int=20) -> str:
    rows = front[:max_rows]
    if not rows:
        return '(no candidates)'
    header = 'idx  sequence                              e_on    e_off    mfe   gap_on gap_off  bp_on bp_off  switch'
    lines = [header]
    for idx, c in enumerate(rows, 1):
        seq = str(c.sequence)
        lines.append(f'{idx:>3d}  {seq:<36.36s} {c.e_on / 100:>6.2f} {c.e_off / 100:>8.2f} {c.mfe / 100:>7.2f} {c.gap_on / 100:>7.2f} {c.gap_off / 100:>7.2f} {c.bp_dist_on:>6d} {c.bp_dist_off:>6d} {c.switching_score:>7.4f}')
    return '\n'.join(lines)


def _summary_to_kcal(summary: dict) -> dict:
    out = dict(summary)
    for key in ('gap_on_min', 'gap_on_mean', 'gap_on_max', 'gap_off_min', 'gap_off_mean', 'gap_off_max', 'stability_min', 'stability_mean', 'stability_max'):
        if key in out:
            out[key] = float(out[key]) / 100.0
    for key in ('switching_score_min', 'switching_score_mean', 'switching_score_max'):
        if key in out:
            out[key] = float(out[key])
    for key in ('count', 'ideal_count'):
        if key in out:
            out[key] = int(out[key])
    return out


def _candidate_to_json(candidate) -> dict:
    crowding_distance = None if candidate.crowding_distance == float('inf') else float(candidate.crowding_distance)
    return {
        'sequence': str(candidate.sequence),
        'mfe_structure': candidate.mfe_structure,
        'energy_on_kcal': float(candidate.e_on) / 100.0,
        'energy_off_kcal': float(candidate.e_off) / 100.0,
        'mfe_energy_kcal': float(candidate.mfe) / 100.0,
        'gap_on_kcal': float(candidate.gap_on) / 100.0,
        'gap_off_kcal': float(candidate.gap_off) / 100.0,
        'stability_kcal': float(candidate.stability) / 100.0,
        'switching_score': float(candidate.switching_score),
        'bp_dist_on': int(candidate.bp_dist_on),
        'bp_dist_off': int(candidate.bp_dist_off),
        'bp_f1_on': float(candidate.bp_f1_on),
        'bp_f1_off': float(candidate.bp_f1_off),
        'rank': int(candidate.rank),
        'crowding_distance': crowding_distance,
    }


def _write_pareto_json(output_path: str, pareto_front: list, structure_on: str, structure_off: str, summary: dict) -> None:
    payload = {
        'structure_on': structure_on,
        'structure_off': structure_off,
        'units': {'energy': 'kcal/mol'},
        'summary': _summary_to_kcal(summary),
        'candidates': [_candidate_to_json(c) for c in pareto_front],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')

@click.command()
@click.argument('structure1', required=False)
@click.argument('structure2', required=False)
@click.option('--input', '-i', 'input_file', help='Read structures from file')
@click.option('--output', '-o', default='results.json', show_default=True, help='Output file (JSON)')
@click.option('--population', '-p', default=100, help='Population size')
@click.option('--generations', '-g', default=200, help='Number of generations')
@click.option('--mutation-rate', '-m', default=0.1, help='Mutation rate (0-1)')
@click.option('--seed', '-s', default=None, type=int, help='Random seed')
@click.option('--verbose', '-v', is_flag=True, help='Show progress')
@click.option('--top-k', '-k', default=20, help='Top candidates to display')
@click.option('--bp-distance-obj/--no-bp-distance-obj', default=False, show_default=True, help='Include base-pair distance to S_ON as a 4th NSGA-II objective so the optimizer selects sequences whose MFE structure matches S_ON. Default mode uses the original 3 objectives; pass --bp-distance-obj to enable this 4th objective.')
@click.option('--require-exact-on', is_flag=True, default=False, help='After optimization, filter the Pareto front to candidates whose MFE structure exactly matches S_ON (bp_dist_on == 0). Falls back to the full front with a warning if the filter leaves nothing.')
def design(structure1, structure2, input_file, output, population, generations, mutation_rate, seed, verbose, top_k, bp_distance_obj, require_exact_on):
    if input_file:
        try:
            s1, s2 = _read_structure_pair(input_file)
        except (FileNotFoundError, ValueError) as e:
            click.echo(f'Error: {e}', err=True)
            raise SystemExit(1)
    elif structure1 and structure2:
        s1, s2 = (structure1, structure2)
    else:
        click.echo('Error: provide two structures as arguments or use --input', err=True)
        raise SystemExit(1)
    if len(s1) != len(s2):
        click.echo(f'Error: structure lengths differ ({len(s1)} vs {len(s2)})', err=True)
        raise SystemExit(1)
    click.echo(f'Riboswitch Design (NSGA-II) — {len(s1)} nt')
    click.echo(f'  S_ON:  {s1}')
    click.echo(f'  S_OFF: {s2}')
    click.echo(f'  Population: {population}, Generations: {generations}')
    click.echo(f'  Structure objective: {('ON (4-objective)' if bp_distance_obj else 'OFF (3-objective)')}')
    click.echo()
    params = TurnerParams.turner2004()
    def progress_callback(gen, front):
        if verbose and (gen % 10 == 0 or gen == generations - 1):
            summary = summarize_pareto_front(front)
            exact_on = sum((1 for c in front if c.bp_dist_on == 0))
            click.echo(f'  Gen {gen:4d}: {summary['count']:3d} solutions, mean gap = {summary.get('gap_on_mean', 0) / 100:.2f} + {summary.get('gap_off_mean', 0) / 100:.2f}  exact_ON={exact_on}')
    start = time.time()
    if verbose:
        click.echo('Running NSGA-II optimization...')
    pareto_front = nsga2(structure_on=s1, structure_off=s2, population_size=population, n_generations=generations, mutation_rate=mutation_rate, params=params, seed=seed, callback=progress_callback if verbose else None, include_structure_objective=bp_distance_obj)
    elapsed = time.time() - start
    if not pareto_front:
        click.echo('No valid candidates found.', err=True)
        raise SystemExit(1)
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
    click.echo(_format_pareto_table(pareto_front, max_rows=top_k))
    click.echo()
    _write_pareto_json(output, pareto_front, s1, s2, summary)
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
        raise SystemExit(1)
    if struct_off is not None and len(seq_clean) != len(struct_off):
        click.echo(f'Error: sequence length {len(seq_clean)} != S_OFF length {len(struct_off)}', err=True)
        raise SystemExit(1)
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


@click.group()
def cli():
    pass


cli.add_command(design, name='design')
cli.add_command(verify, name='verify')

def main():
    cli()

def main_verify():
    verify()
if __name__ == '__main__':
    main()
