import time
from ribo_switch.nsga2 import nsga2, filter_by_structure, summarize_pareto_front
from ribo_switch.verify import verify_against_both
from ribo_switch.viz import format_pareto_table
FILE = 'rand_structure/rand_struc42.in'
lines = [l.strip() for l in open(FILE) if l.strip() and (not l.startswith('>'))]
s_on, s_off, ref_seq = (lines[0], lines[1], lines[2])
print(f'File   : {FILE}')
print(f'Length : {len(s_on)} nt')
print(f'S_ON   : {s_on}')
print(f'S_OFF  : {s_off}')
print()
print('=' * 62)
print('Verifying the reference sequence from the file...')
print('=' * 62)
print(f'Ref seq: {ref_seq}')
print()
try:
    rep_on, rep_off = verify_against_both(ref_seq, s_on, s_off)
    print('--- vs S_ON ---')
    for line in rep_on.summary_lines():
        print(line)
    print()
    print('--- vs S_OFF ---')
    for line in rep_off.summary_lines():
        print(line)
except ValueError as e:
    print(f'Note: reference sequence is not compatible with these structures: {e}')
    print("(The file's sequence was not designed for S_ON/S_OFF -- skipping.)")
print()
print('=' * 62)
print('Running NSGA-II (pop=80, gen=80, seed=7)...')
print('=' * 62)
t0 = time.perf_counter()
front = nsga2(s_on, s_off, population_size=80, n_generations=80, seed=7)
dt = time.perf_counter() - t0
summary = summarize_pareto_front(front)
exact = filter_by_structure(front, max_bp_dist_on=0)
print(f'Done in {dt:.1f}s   |front|={len(front)}   exact_ON (bp_dist=0): {len(exact)}')
print(f'Gap_ON   min={summary['gap_on_min'] / 100:.2f}  mean={summary['gap_on_mean'] / 100:.2f} kcal/mol')
print(f'Gap_OFF  min={summary['gap_off_min'] / 100:.2f}  mean={summary['gap_off_mean'] / 100:.2f} kcal/mol')
print()
print('Top candidates (sorted by Gap_ON+Gap_OFF, with bp-distance columns):')
print(format_pareto_table(front, max_rows=8))
best = sorted(front, key=lambda c: (c.bp_dist_on, c.gap_on + c.gap_off))[0]
seq_str = str(best.sequence)
print()
print('=' * 62)
print('Verifying best designed candidate...')
print('=' * 62)
print(f'Sequence : {seq_str}')
print()
rep_on2, rep_off2 = verify_against_both(seq_str, s_on, s_off)
print('--- vs S_ON ---')
for line in rep_on2.summary_lines():
    print(line)
print()
print('--- vs S_OFF ---')
for line in rep_off2.summary_lines():
    print(line)
on_exact = rep_on2.matches_exactly
off_exact = rep_off2.matches_exactly
if on_exact and off_exact:
    verdict = 'MATCHES_BOTH'
elif on_exact:
    verdict = 'MATCHES_ON'
elif off_exact:
    verdict = 'MATCHES_OFF'
else:
    verdict = 'NO_MATCH'
print()
print(f'Overall verdict: {verdict}')
