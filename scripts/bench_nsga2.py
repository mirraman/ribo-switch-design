import time
import random
import cProfile
import pstats
import io
from ribo_switch.nsga2 import nsga2
from ribo_switch.rust_bridge import USING_RUST
CASES: dict[int, tuple[str, str]] = {28: ('((((....))))....((((....))))', '....((((....))))((((....))))'), 60: ('(((((((........)))))))..........(((((((........)))))))......', '..(((((((((((((((........................)))))))))))))))....'), 95: ('((((((.........))))))((((((.........))))))((((((.........))))))' + '.' * 32, '((((((((((((((((...................))))))))))))))))' + '.' * 44)}

def make_pair(length: int) -> tuple[str, str]:
    on, off = CASES[length]
    assert len(on) == length and len(off) == length, f'len mismatch for {length}: on={len(on)} off={len(off)}'
    return (on, off)

def run(length: int, pop: int, gens: int) -> float:
    random.seed(42)
    s_on, s_off = make_pair(length)
    t0 = time.perf_counter()
    front = nsga2(s_on, s_off, population_size=pop, n_generations=gens, mutation_rate=0.1, seed=42)
    dt = time.perf_counter() - t0
    print(f'  len={length:3d}  pop={pop:4d}  gen={gens:4d}  time={dt:7.3f}s  |front|={len(front)}')
    return dt

def profile_one(length: int, pop: int, gens: int) -> None:
    random.seed(42)
    s_on, s_off = make_pair(length)
    pr = cProfile.Profile()
    pr.enable()
    nsga2(s_on, s_off, population_size=pop, n_generations=gens, mutation_rate=0.1, seed=42)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).strip_dirs().sort_stats('cumulative').print_stats(20)
    print(buf.getvalue())

def main() -> None:
    print(f'Rust extension active: {USING_RUST}')
    print()
    print('Wall-clock:')
    for length, pop, gens in [(28, 60, 20), (28, 100, 40), (60, 100, 40), (95, 100, 40), (95, 200, 40)]:
        run(length, pop, gens)
    print()
    print('Profile of len=95 pop=100 gen=20:')
    profile_one(95, 100, 20)
if __name__ == '__main__':
    main()
