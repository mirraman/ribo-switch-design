"""
turner.py — Turner 2004 nearest-neighbor energy parameters.

All energies are stored as integers in units of 0.01 kcal/mol to avoid
floating-point accumulation errors during DP (standard convention used
by ViennaRNA and RNAstructure).

Source: NNDB (https://rna.urmc.rochester.edu/NNDB/turner04/)
Reference: Mathews et al. (2004), Turner & Mathews (2010)
"""

import numpy as np
from dataclasses import dataclass, field

# Infinity sentinel for impossible configurations
INF = 10**7


@dataclass
class TurnerParams:
    """Container for all Turner 2004 nearest-neighbor parameters.

    Array indexing conventions:
      - Base pair index:  AU=0, UA=1, CG=2, GC=3, GU=4, UG=5
      - Base index:       A=0, C=1, G=2, U=3
      - Energy unit:      0.01 kcal/mol (e.g., -220 = -2.20 kcal/mol)
    """

    # --- Stacking energies: stack[pair_i][pair_j] ---
    # Energy of pair_i stacked directly on pair_j
    stack: np.ndarray  # shape (6, 6), int32

    # --- Loop initiation energies (by loop size, index = size) ---
    hairpin_init: np.ndarray   # shape (31,) — sizes 0..30
    bulge_init: np.ndarray     # shape (31,)
    interior_init: np.ndarray  # shape (31,)

    # --- Special hairpin loops ---
    hairpin_triloop: dict[str, int]    # 5-mer string → energy
    hairpin_tetraloop: dict[str, int]  # 6-mer string → energy

    # --- Terminal mismatch in hairpins: [pair][base_5prime][base_3prime] ---
    hairpin_mismatch: np.ndarray  # shape (6, 4, 4)

    # --- Interior loop mismatch (for loops > 2×2) ---
    interior_mismatch: np.ndarray  # shape (6, 4, 4)

    # --- 1×1 interior loops: [outer_pair][inner_pair][mismatch1][mismatch2] ---
    interior_1x1: np.ndarray  # shape (6, 6, 4, 4)

    # --- Multiloop parameters ---
    ml_offset: int = 0       # a: initiation penalty
    ml_per_branch: int = 0   # b: per helix/branch
    ml_per_unpaired: int = 0 # c: per unpaired nucleotide

    # --- Dangling ends ---
    dangle5: np.ndarray = field(
        default_factory=lambda: np.zeros((6, 4), dtype=np.int32)
    )
    dangle3: np.ndarray = field(
        default_factory=lambda: np.zeros((6, 4), dtype=np.int32)
    )

    # --- Penalties and corrections ---
    terminal_au_penalty: int = 50   # 0.50 kcal/mol for terminal AU/GU pairs
    ninio_max: int = 300            # max asymmetry correction
    ninio_m: int = 60               # per-nucleotide asymmetry

    # --- Hairpin special terms ---
    hairpin_uu_ga_bonus: int = -90  # bonus for UU or GA first mismatch
    hairpin_gg_bonus: int = -80     # bonus for GG first mismatch
    hairpin_special_gu: int = -220  # special GU closure preceded by GG
    hairpin_c3: int = 140           # penalty for all-C triloop
    hairpin_c_slope: int = 30       # A in penalty = A*n + B for all-C loops
    hairpin_c_intercept: int = 160  # B in penalty = A*n + B for all-C loops

    # --- Jacobson-Stockmayer extrapolation ---
    loop_extrapolation_coeff: float = 1.079  # 1.75 * RT at 37°C

    @staticmethod
    def turner2004() -> 'TurnerParams':
        """Return hardcoded Turner 2004 parameters from NNDB."""
        return TurnerParams(
            stack=_build_stack(),
            hairpin_init=_build_hairpin_init(),
            bulge_init=_build_bulge_init(),
            interior_init=_build_interior_init(),
            hairpin_triloop=_build_triloop(),
            hairpin_tetraloop=_build_tetraloop(),
            hairpin_mismatch=_build_hairpin_mismatch(),
            interior_mismatch=_build_interior_mismatch(),
            interior_1x1=_build_interior_1x1(),
            dangle5=_build_dangle5(),
            dangle3=_build_dangle3(),
            ml_offset=340,
            ml_per_branch=40,
            ml_per_unpaired=0,
        )


# ===================================================================
#  Builder functions — each returns one parameter table
# ===================================================================

def _build_stack() -> np.ndarray:
    """Stacking energies from NNDB stack.txt.

    Index order: AU=0, UA=1, CG=2, GC=3, GU=4, UG=5
    stack[i][j] = energy of pair i stacked on pair j.

    Source: https://rna.urmc.rochester.edu/NNDB/turner04/stack.txt
    """
    #          AU    UA    CG    GC    GU    UG
    data = [
        [ -90, -110, -220, -210,  -60, -140],  # AU
        [-130,  -90, -240, -210, -100, -130],  # UA
        [-210, -210, -330, -240, -140, -210],  # CG
        [-240, -220, -340, -330, -150, -250],  # GC
        [-130, -140, -250, -210,  -50,  130],  # GU
        [-100,  -60, -150, -140,   30,  -50],  # UG
    ]
    return np.array(data, dtype=np.int32)


def _build_hairpin_init() -> np.ndarray:
    """Hairpin loop initiation energies by size (0..30).

    Source: https://rna.urmc.rochester.edu/NNDB/turner04/loop.txt
    Sizes 0-2 are impossible (set to INF).
    """
    init = [INF] * 31
    # From NNDB loop.txt (kcal/mol × 100)
    vals = {
        3: 540, 4: 560, 5: 570, 6: 540, 7: 600, 8: 550, 9: 640,
        10: 650, 11: 660, 12: 670, 13: 680, 14: 690, 15: 690,
        16: 700, 17: 710, 18: 710, 19: 720, 20: 720, 21: 730,
        22: 730, 23: 740, 24: 740, 25: 750, 26: 750, 27: 750,
        28: 760, 29: 760, 30: 770,
    }
    for size, energy in vals.items():
        init[size] = energy
    return np.array(init, dtype=np.int32)


def _build_bulge_init() -> np.ndarray:
    """Bulge loop initiation energies by size (0..30).

    Source: https://rna.urmc.rochester.edu/NNDB/turner04/loop.txt
    """
    init = [INF] * 31
    vals = {
        1: 380, 2: 280, 3: 320, 4: 360, 5: 400, 6: 440, 7: 460,
        8: 470, 9: 480, 10: 490, 11: 500, 12: 510, 13: 520,
        14: 530, 15: 540, 16: 540, 17: 550, 18: 550, 19: 560,
        20: 570, 21: 570, 22: 580, 23: 580, 24: 580, 25: 590,
        26: 590, 27: 600, 28: 600, 29: 600, 30: 610,
    }
    for size, energy in vals.items():
        init[size] = energy
    return np.array(init, dtype=np.int32)


def _build_interior_init() -> np.ndarray:
    """Interior loop initiation energies by total unpaired count (0..30).

    Source: https://rna.urmc.rochester.edu/NNDB/turner04/loop.txt
    Sizes 0-3 are impossible or handled by special tables.
    """
    init = [INF] * 31
    vals = {
        3: 200,  # 2×1 and 1×2 loops (approx; proper impl uses int21 table)
        4: 110, 5: 200, 6: 200, 7: 210, 8: 230, 9: 240,
        10: 250, 11: 260, 12: 270, 13: 280, 14: 290, 15: 290,
        16: 300, 17: 310, 18: 310, 19: 320, 20: 330, 21: 330,
        22: 340, 23: 340, 24: 350, 25: 350, 26: 350, 27: 360,
        28: 360, 29: 370, 30: 370,
    }
    for size, energy in vals.items():
        init[size] = energy
    return np.array(init, dtype=np.int32)


def _build_triloop() -> dict[str, int]:
    """Special triloop (3 unpaired nt) bonus energies.

    Key = 5-mer string (closing pair + 3 loop bases).
    Source: https://rna.urmc.rochester.edu/NNDB/turner04/triloop.txt
    """
    return {
        "CAACG": 680,
        "GUUAC": 690,
    }


def _build_tetraloop() -> dict[str, int]:
    """Special tetraloop (4 unpaired nt) bonus energies.

    Key = 6-mer string (closing pair + 4 loop bases).
    Source: https://rna.urmc.rochester.edu/NNDB/turner04/tloop.txt
    """
    return {
        "CUACGG": 280, "CUCCGG": 270, "CUUCGG": 370, "CUUUGG": 370,
        "CCAAGG": 330, "CCCAGG": 340, "CCGAGG": 350, "CCUAGG": 370,
        "CCACGG": 370, "CCGCGG": 360, "CCUCGG": 250, "CUAAGG": 360,
        "CUCAGG": 370, "CUUAGG": 350, "CUGCGG": 280, "CAACGG": 550,
    }


def _build_hairpin_mismatch() -> np.ndarray:
    """Terminal mismatch energies for hairpin loops.

    hairpin_mismatch[pair][base_after_5prime][base_before_3prime]
    Source: https://rna.urmc.rochester.edu/NNDB/turner04/tstack.txt
    """
    m = np.zeros((6, 4, 4), dtype=np.int32)

    # AU (pair 0)
    m[0] = [[-80, -100, -80, -100],
            [-60,  -70, -60,  -70],
            [-80, -100, -80, -100],
            [-60,  -80, -60,  -80]]
    # UA (pair 1)
    m[1] = [[-100, -80, -110, -80],
            [ -70, -60,  -70, -50],
            [-110, -80, -120, -80],
            [ -70, -60,  -70, -50]]
    # CG (pair 2)
    m[2] = [[-150, -150, -140, -150],
            [-100, -110, -100,  -80],
            [-140, -150, -160, -150],
            [-100, -140, -100, -120]]
    # GC (pair 3)
    m[3] = [[-110, -150, -130, -150],
            [-110,  -70, -110,  -50],
            [-160, -150, -140, -150],
            [-110, -100, -110,  -70]]
    # GU (pair 4)
    m[4] = [[ -30, -100,  -80, -100],
            [ -60,  -70,  -60,  -70],
            [ -60, -100,  -80, -100],
            [ -60,  -80,  -60,  -60]]
    # UG (pair 5)
    m[5] = [[-100,  -80, -110,  -80],
            [ -70,  -60,  -70,  -50],
            [ -50,  -80,  -80,  -80],
            [ -70,  -60,  -70,  -50]]
    return m


def _build_interior_mismatch() -> np.ndarray:
    """Terminal mismatch energies for interior loops (≥ 2×3).

    For interior loops, a common approximation is to use the same
    values as hairpin mismatches. Full tstacki tables can replace this.
    """
    # Use hairpin mismatch as approximation (standard in many implementations)
    return _build_hairpin_mismatch().copy()


def _build_interior_1x1() -> np.ndarray:
    """1×1 interior loop energies.

    interior_1x1[outer_pair][inner_pair][mismatch_5prime][mismatch_3prime]
    Source: https://rna.urmc.rochester.edu/NNDB/turner04/int11.txt

    Values in 0.01 kcal/mol. Full 6×6×4×4 table from NNDB.
    """
    t = np.full((6, 6, 4, 4), INF, dtype=np.int32)

    # ── outer=AU(0) ──
    # inner=AU(0)
    t[0][0] = [[ 190, 190, 190, 190],
               [ 190, 190, 190, 190],
               [ 190, 190, -70, 190],
               [ 190, 190, 190, 150]]
    # inner=CG(2)
    t[0][2] = [[ 120, 120, 120, 120],
               [ 120, 120, 120, 120],
               [ 120, 120,-140, 120],
               [ 120, 120, 120,  80]]
    # inner=GC(3)
    t[0][3] = [[ 120, 120, 120, 120],
               [ 120, 120, 120, 120],
               [ 120, 120,-140, 120],
               [ 120, 120, 120,  80]]
    # inner=UA(1)
    t[0][1] = [[ 190, 190, 190, 190],
               [ 190, 190, 190, 190],
               [ 190, 190, -70, 190],
               [ 190, 190, 190, 120]]
    # inner=GU(4)
    t[0][4] = [[ 190, 190, 190, 190],
               [ 190, 190, 190, 190],
               [ 190, 190, -70, 190],
               [ 190, 190, 190, 160]]
    # inner=UG(5)
    t[0][5] = [[ 190, 190, 190, 190],
               [ 190, 190, 190, 190],
               [ 190, 190, -70, 190],
               [ 190, 190, 190, 120]]

    # ── outer=CG(2) ──
    t[2][0] = [[ 120, 120, 120, 120],
               [ 120, 120, 120, 120],
               [ 120, 120,-140, 120],
               [ 120, 120, 120, 120]]
    t[2][2] = [[  90, -40,  50,  50],
               [  30,  50,  50,  60],
               [ -10,  50,-220,  50],
               [  50,   0,  50, -10]]
    t[2][3] = [[  80,  50,  50,  50],
               [  50,  50,  50,  50],
               [  50,  50,-230,  50],
               [  50,  50,  50, -60]]
    t[2][1] = [[ 120, 120, 120, 120],
               [ 120, 120, 120, 120],
               [ 120, 120,-140, 120],
               [ 120, 120, 120,  80]]
    t[2][4] = [[ 220, 130, 120, 120],
               [ 120, 170, 120, 120],
               [ 120, 120,-140, 120],
               [ 120, 120, 120, 110]]
    t[2][5] = [[  60,  50, 120, 120],
               [ 120, 120, 120, 120],
               [ -20, 120,-140, 120],
               [ 120, 100, 120, 110]]

    # ── outer=GC(3) ──
    t[3][0] = [[ 120, 120, 120, 120],
               [ 120, 120, 120, 120],
               [ 120, 120,-140, 120],
               [ 120, 120, 120, 120]]
    t[3][2] = [[  90,  30, -10,  50],
               [ -40,  50,  50,   0],
               [  50,  50,-220,  50],
               [  50,  60,  50, -10]]
    t[3][3] = [[  80,  50,  50,  50],
               [  50,  50,  50,  50],
               [  50,  50,-230,  50],
               [  50,  50,  50, -60]]
    t[3][1] = [[ 120, 120, 120, 120],
               [ 120, 120, 120, 120],
               [ 120, 120,-140, 120],
               [ 120, 120, 120,  80]]
    t[3][4] = [[ 160, 120, 100, 120],
               [ 120, 120, 120, 120],
               [ 120, 120,-140, 120],
               [ 120, 120, 120,  70]]
    t[3][5] = [[ 190, 120, 150, 120],
               [ 120, 120, 120, 120],
               [ 120, 120,-140, 120],
               [ 120, 120, 120, 150]]

    # ── outer=UA(1) ──
    t[1][0] = [[ 190, 190, 190, 190],
               [ 190, 190, 190, 190],
               [ 190, 190, -70, 190],
               [ 190, 190, 190, 170]]
    t[1][2] = [[ 120, 120, 120, 120],
               [ 120, 120, 120, 120],
               [ 120, 120,-140, 120],
               [ 120, 120, 120, 120]]
    t[1][3] = [[ 120, 120, 120, 120],
               [ 120, 120, 120, 120],
               [ 120, 120,-140, 120],
               [ 120, 120, 120, 120]]
    t[1][1] = [[ 190, 190, 190, 190],
               [ 190, 190, 190, 190],
               [ 190, 190, -70, 190],
               [ 190, 190, 190, 150]]
    t[1][4] = [[ 190, 190, 190, 190],
               [ 190, 190, 190, 190],
               [ 190, 190, -70, 190],
               [ 190, 190, 190, 190]]
    t[1][5] = [[ 190, 190, 190, 190],
               [ 190, 190, 190, 190],
               [ 190, 190, -70, 190],
               [ 190, 190, 190, 160]]

    # ── outer=GU(4) ──
    t[4][0] = [[ 190, 190, 190, 190],
               [ 190, 190, 190, 190],
               [ 190, 190, -70, 190],
               [ 190, 190, 190, 160]]
    t[4][2] = [[ 190, 120, 120, 120],
               [ 120, 120, 120, 120],
               [ 150, 120,-140, 120],
               [ 120, 120, 120, 150]]
    t[4][3] = [[  60, 120, -20, 120],
               [  50, 120, 120, 100],
               [ 120, 120,-140, 120],
               [ 120, 120, 120, 110]]
    t[4][1] = [[ 190, 190, 190, 190],
               [ 190, 190, 190, 190],
               [ 190, 190, -70, 190],
               [ 190, 190, 190, 120]]
    t[4][4] = [[ 190, 190, 190, 190],
               [ 190, 190, 190, 190],
               [ 190, 190, -70, 190],
               [ 190, 190, 190, 160]]
    t[4][5] = [[ 190, 190, 190, 190],
               [ 190, 190, 190, 190],
               [ 190, 190, -70, 190],
               [ 190, 190, 190, 120]]

    # ── outer=UG(5) ──
    t[5][0] = [[ 190, 190, 190, 190],
               [ 190, 190, 190, 190],
               [ 190, 190, -70, 190],
               [ 190, 190, 190, 190]]
    t[5][2] = [[ 160, 120, 120, 120],
               [ 120, 120, 120, 120],
               [ 100, 120,-140, 120],
               [ 120, 120, 120,  70]]
    t[5][3] = [[ 220, 120, 120, 120],
               [ 130, 170, 120, 120],
               [ 120, 120,-140, 120],
               [ 120, 120, 120, 110]]
    t[5][1] = [[ 190, 190, 190, 190],
               [ 190, 190, 190, 190],
               [ 190, 190, -70, 190],
               [ 190, 190, 190, 160]]
    t[5][4] = [[ 190, 190, 190, 190],
               [ 190, 190, 190, 190],
               [ 190, 190, -70, 190],
               [ 190, 190, 190, 190]]
    t[5][5] = [[ 190, 190, 190, 190],
               [ 190, 190, 190, 190],
               [ 190, 190, -70, 190],
               [ 190, 190, 190, 160]]

    return t


def _build_dangle3() -> np.ndarray:
    """3' dangling end energies: dangle3[pair][dangling_base].

    The dangling base is on the 3' side of the closing pair.
    Source: https://rna.urmc.rochester.edu/NNDB/turner04/dangle.txt
    """
    d = np.zeros((6, 4), dtype=np.int32)
    #        A    C    G    U
    d[0] = [-80, -50, -80, -60]  # AU
    d[1] = [-70, -10, -70, -10]  # UA
    d[2] = [-170, -80, -170, -120]  # CG
    d[3] = [-110, -40, -130, -60]  # GC
    d[4] = [-80, -50, -80, -60]  # GU
    d[5] = [-70, -10, -70, -10]  # UG
    return d


def _build_dangle5() -> np.ndarray:
    """5' dangling end energies: dangle5[pair][dangling_base].

    The dangling base is on the 5' side of the closing pair.
    Source: https://rna.urmc.rochester.edu/NNDB/turner04/dangle.txt
    """
    d = np.zeros((6, 4), dtype=np.int32)
    #        A    C    G    U
    d[0] = [-30, -10, -20, -20]  # AU
    d[1] = [-30, -30, -40, -20]  # UA
    d[2] = [-20, -30,   0,   0]  # CG
    d[3] = [-50, -30, -20, -10]  # GC
    d[4] = [-30, -10, -20, -20]  # GU
    d[5] = [-30, -30, -40, -20]  # UG
    return d
