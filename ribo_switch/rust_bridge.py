"""
rust_bridge.py — Transparent Rust/Python fallback for hot-path functions.

Tries to use the compiled Rust extension (ribo_rs) for eval_energy and
fold_mfe.  Falls back to pure-Python implementations if the extension is
not available (CI, fresh checkout, unsupported platform).

Public API (drop-in replacements for the Python originals):
    eval_energy(seq, struct, params) -> Energy
    fold_mfe(seq, params)           -> FoldResult
    USING_RUST: bool                 — True when Rust extension is active
"""

from __future__ import annotations
from ribo_switch.types import Sequence, Energy, Structure
from ribo_switch.turner import TurnerParams
from ribo_switch.fold import FoldResult

try:
    import ribo_rs as _ribo_rs
    USING_RUST: bool = True
except ImportError:
    _ribo_rs = None  # type: ignore[assignment]
    USING_RUST = False


if USING_RUST:
    import numpy as _np

    def eval_energy(seq: Sequence, struct: Structure, params: TurnerParams) -> Energy:
        return _ribo_rs.eval_energy([int(b) for b in seq.bases], struct.pair_table)

    def fold_mfe(seq: Sequence, params: TurnerParams) -> FoldResult:
        mfe_e, mfe_db = _ribo_rs.fold_mfe([int(b) for b in seq.bases])
        n = len(seq.bases)
        _dummy = _np.empty((n, n), dtype=_np.int64)
        return FoldResult(mfe_energy=mfe_e, mfe_structure=mfe_db, v=_dummy, w=_dummy)

else:
    from ribo_switch.energy import eval_energy as _py_eval  # type: ignore[assignment]
    from ribo_switch.fold import fold_mfe as _py_fold  # type: ignore[assignment]

    def eval_energy(seq: Sequence, struct: Structure, params: TurnerParams) -> Energy:
        return _py_eval(seq, struct, params)

    def fold_mfe(seq: Sequence, params: TurnerParams) -> FoldResult:
        return _py_fold(seq, params)
