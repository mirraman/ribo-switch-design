from ribo_switch.types import Structure, LoopType, HairpinLoop, StackLoop, InteriorLoop, BulgeLoop, MultiLoop, ExternalLoop

def parse_dot_bracket(db: str) -> Structure:
    if not db:
        raise ValueError('Empty dot-bracket string')
    n = len(db)
    pair_table: list[int] = [-1] * n
    stack: list[int] = []
    for i, ch in enumerate(db):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if not stack:
                raise ValueError(f"Unbalanced parentheses: extra ')' at position {i}")
            j = stack.pop()
            pair_table[j] = i
            pair_table[i] = j
        elif ch == '.':
            pass
        else:
            raise ValueError(f"Invalid character '{ch}' at position {i}. Only '.', '(', ')' are allowed.")
    if stack:
        raise ValueError(f"Unbalanced parentheses: {len(stack)} unmatched '(' (first at position {stack[0]})")
    pairs = sorted([(i, pair_table[i]) for i in range(n) if pair_table[i] > i])
    return Structure(length=n, pair_table=pair_table, pairs=pairs)

def decompose_loops(structure: Structure) -> list[LoopType]:
    pt = structure.pair_table
    n = structure.length
    loops: list[LoopType] = []
    for i, j in structure.pairs:
        enclosed = _find_enclosed_pairs(pt, i, j)
        loop = _classify_and_build_loop(pt, i, j, enclosed)
        loops.append(loop)
    ext = _build_external_loop(pt, n, structure.pairs)
    loops.append(ext)
    return loops

def _find_enclosed_pairs(pair_table: list[int], i: int, j: int) -> list[tuple[int, int]]:
    enclosed: list[tuple[int, int]] = []
    k = i + 1
    while k < j:
        partner = pair_table[k]
        if partner > k:
            enclosed.append((k, partner))
            k = partner + 1
        else:
            k += 1
    return enclosed

def _classify_and_build_loop(pair_table: list[int], i: int, j: int, enclosed: list[tuple[int, int]]) -> LoopType:
    if len(enclosed) == 0:
        unpaired = list(range(i + 1, j))
        return HairpinLoop(closing_pair=(i, j), unpaired=unpaired)
    if len(enclosed) == 1:
        p, q = enclosed[0]
        left_unp = list(range(i + 1, p))
        right_unp = list(range(q + 1, j))
        n_left = len(left_unp)
        n_right = len(right_unp)
        if n_left == 0 and n_right == 0:
            return StackLoop(outer_pair=(i, j), inner_pair=(p, q))
        if n_left == 0:
            return BulgeLoop(outer_pair=(i, j), inner_pair=(p, q), unpaired=right_unp, side='right')
        if n_right == 0:
            return BulgeLoop(outer_pair=(i, j), inner_pair=(p, q), unpaired=left_unp, side='left')
        return InteriorLoop(outer_pair=(i, j), inner_pair=(p, q), left_unpaired=left_unp, right_unpaired=right_unp)
    unpaired: list[int] = []
    prev_end = i + 1
    for p, q in enclosed:
        unpaired.extend(range(prev_end, p))
        prev_end = q + 1
    unpaired.extend(range(prev_end, j))
    return MultiLoop(closing_pair=(i, j), branches=enclosed, unpaired=unpaired)

def _build_external_loop(pair_table: list[int], n: int, pairs: list[tuple[int, int]]) -> ExternalLoop:
    top_level: list[tuple[int, int]] = []
    for i, j in pairs:
        is_top = pair_table[i] == j
        enclosed_by_other = False
        for a, b in pairs:
            if a < i and b > j:
                enclosed_by_other = True
                break
        if not enclosed_by_other:
            top_level.append((i, j))
    unpaired: list[int] = []
    pos = 0
    for i, j in top_level:
        for k in range(pos, i):
            if pair_table[k] == -1:
                unpaired.append(k)
        pos = j + 1
    for k in range(pos, n):
        if pair_table[k] == -1:
            unpaired.append(k)
    return ExternalLoop(unpaired=unpaired, closing_pairs=top_level)
