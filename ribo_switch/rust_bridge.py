"""
rust_bridge.py — Transparent Rust/Python fallback for hot-path functions.

Public API (drop-in replacements for the Python originals):
    eval_energy(seq, struct, params)                 -> Energy
    fold_mfe(seq, params)                            -> FoldResult
    evaluate_candidate(seq, s_on, s_off, params)     -> (e_on, e_off, mfe, mfe_db)
    evaluate_batch(seqs, s_on, s_off, params)        -> list[(e_on, e_off, mfe, mfe_db)]
    USING_RUST: bool                                  — True when Rust extension is active

`evaluate_candidate` and `evaluate_batch` combine the three calls that the
NSGA-II hot loop used to make separately (eval on S_ON, eval on S_OFF,
fold_mfe + re-eval on the folded structure). The batch version fans out
across CPU cores via rayon with the GIL released.
"""

from __future__ import annotations

from typing import Iterable

from ribo_switch.types import Sequence, Energy, Structure
from ribo_switch.turner import TurnerParams
from ribo_switch.fold import FoldResult

try:
    import ribo_rs as _ribo_rs
    USING_RUST: bool = True
except ImportError:
    _ribo_rs = None  # type: ignore[assignment]
    USING_RUST = False


def _seq_bytes(seq: Sequence) -> list[int]:
    """Convert Sequence -> list[int] once, cheaply."""
    return [int(b) for b in seq.bases]


if USING_RUST:
    import numpy as _np

    def eval_energy(seq: Sequence, struct: Structure, params: TurnerParams) -> Energy:
        return _ribo_rs.eval_energy(_seq_bytes(seq), struct.pair_table)

    def fold_mfe(seq: Sequence, params: TurnerParams) -> FoldResult:
        mfe_e, mfe_db = _ribo_rs.fold_mfe(_seq_bytes(seq))
        n = len(seq.bases)
        _dummy = _np.empty((n, n), dtype=_np.int64)
        return FoldResult(mfe_energy=mfe_e, mfe_structure=mfe_db, v=_dummy, w=_dummy)

    def evaluate_candidate(
        seq: Sequence,
        s_on: Structure,
        s_off: Structure,
        params: TurnerParams,
    ) -> tuple[int, int, int, str]:
        return _ribo_rs.evaluate_candidate(
            _seq_bytes(seq), s_on.pair_table, s_off.pair_table
        )

    def evaluate_batch(
        seqs: Iterable[Sequence],
        s_on: Structure,
        s_off: Structure,
        params: TurnerParams,
    ) -> list[tuple[int, int, int, str]]:
        seq_lists = [_seq_bytes(s) for s in seqs]
        if not seq_lists:
            return []
        return _ribo_rs.evaluate_batch(
            seq_lists, s_on.pair_table, s_off.pair_table
        )

else:
    from ribo_switch.energy import eval_energy as _py_eval
    from ribo_switch.fold import fold_mfe as _py_fold
    from ribo_switch.structure import parse_dot_bracket as _parse_db

    def eval_energy(seq: Sequence, struct: Structure, params: TurnerParams) -> Energy:
        return _py_eval(seq, struct, params)

    def fold_mfe(seq: Sequence, params: TurnerParams) -> FoldResult:
        return _py_fold(seq, params)

    def evaluate_candidate(
        seq: Sequence,
        s_on: Structure,
        s_off: Structure,
        params: TurnerParams,
    ) -> tuple[int, int, int, str]:
        e_on = _py_eval(seq, s_on, params)
        e_off = _py_eval(seq, s_off, params)
        fr = _py_fold(seq, params)
        mfe_struct = _parse_db(fr.mfe_structure)
        mfe = _py_eval(seq, mfe_struct, params)
        return e_on, e_off, mfe, fr.mfe_structure

    def evaluate_batch(
        seqs: Iterable[Sequence],
        s_on: Structure,
        s_off: Structure,
        params: TurnerParams,
    ) -> list[tuple[int, int, int, str]]:
        return [evaluate_candidate(s, s_on, s_off, params) for s in seqs]
