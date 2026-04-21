"""Verify Rust eval_energy and fold_mfe match Python on test cases."""
import ribo_rs
from ribo_switch.types import Base, Sequence
from ribo_switch.structure import parse_dot_bracket
from ribo_switch.energy import eval_energy as py_eval_energy
from ribo_switch.fold import fold_mfe as py_fold_mfe
from ribo_switch.turner import TurnerParams
from ribo_switch.genetics import create_individual
from ribo_switch.graph import build_constraint_graph
import random, time

params = TurnerParams.turner2004()

def seq_to_ints(seq):
    return [int(b) for b in seq.bases]

def test_eval_energy(seq, struct):
    pt = struct.pair_table
    py_e = py_eval_energy(seq, struct, params)
    rs_e = ribo_rs.eval_energy(seq_to_ints(seq), pt)
    ok = py_e == rs_e
    if not ok:
        print(f"  MISMATCH  py={py_e}  rs={rs_e}")
    return ok

def test_fold_mfe(seq):
    """Both folders must (a) return the same MFE value (reported by the DP
    itself); (b) produce a structure whose eval_energy matches the DP's MFE
    — the fold/eval consistency invariant we depend on in NSGA-II."""
    py_res  = py_fold_mfe(seq, params)
    rs_mfe, rs_db = ribo_rs.fold_mfe(seq_to_ints(seq))
    py_struct = parse_dot_bracket(py_res.mfe_structure)
    rs_struct = parse_dot_bracket(rs_db)
    py_re = py_eval_energy(seq, py_struct, params)
    rs_re = py_eval_energy(seq, rs_struct, params)

    ok_val   = py_res.mfe_energy == rs_mfe
    ok_py    = py_res.mfe_energy == py_re    # Python fold ≡ Python eval
    ok_rs    = rs_mfe == rs_re                # Rust fold ≡ Python eval
    ok = ok_val and ok_py and ok_rs
    if not ok:
        print(f"  MISMATCH  py_mfe={py_res.mfe_energy}  rs_mfe={rs_mfe}")
        print(f"  py_re={py_re}  rs_re={rs_re}")
        print(f"  py_struct={py_res.mfe_structure}")
        print(f"  rs_struct={rs_db}")
    return ok

print("=== Correctness check ===")
S_ON  = "((((((.....))))))...((((....))))"
S_OFF = "......(((((((((....)))))))))...."
s_on  = parse_dot_bracket(S_ON)
s_off = parse_dot_bracket(S_OFF)
graph = build_constraint_graph(s_on, s_off)
rng   = random.Random(42)

all_ok = True
for i in range(50):
    ind = create_individual(graph, rng)
    seq = ind.sequence
    for struct in [s_on, s_off]:
        ok = test_eval_energy(seq, struct)
        if not ok: all_ok = False
    ok = test_fold_mfe(seq)
    if not ok: all_ok = False

print(f"eval_energy: {'PASS' if all_ok else 'FAIL'}  (50 sequences × 2 structures)")
print(f"fold_mfe:    {'PASS' if all_ok else 'FAIL'}  (50 sequences)")

print()
print("=== Speed benchmark (100 candidates, eval_energy + fold_mfe each) ===")
seqs = []
for _ in range(100):
    ind = create_individual(graph, rng)
    seqs.append(ind.sequence)

t0 = time.perf_counter()
for seq in seqs:
    py_eval_energy(seq, s_on, params)
    py_eval_energy(seq, s_off, params)
    py_fold_mfe(seq, params)
py_t = time.perf_counter() - t0

t0 = time.perf_counter()
for seq in seqs:
    si = seq_to_ints(seq)
    ribo_rs.eval_energy(si, s_on.pair_table)
    ribo_rs.eval_energy(si, s_off.pair_table)
    ribo_rs.fold_mfe(si)
rs_t = time.perf_counter() - t0

print(f"  Python:  {py_t*1000:.0f} ms")
print(f"  Rust:    {rs_t*1000:.0f} ms")
print(f"  Speedup: {py_t/rs_t:.1f}×")
