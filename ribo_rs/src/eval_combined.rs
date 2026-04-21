/// Combined per-candidate and batch evaluation.
///
/// Single-candidate `evaluate_one` runs `eval_energy` on the ON and OFF
/// target pair tables, then `fold_mfe_full`, then re-scores the folded
/// structure with `eval_energy` (defensive: ensures the returned `mfe`
/// uses the same accounting as `e_on` / `e_off`, so `gap_on = e_on - mfe`
/// is guaranteed to be ≥ 0). The folded pair table is reused directly —
/// no dot-bracket → pair-table round-trip.
///
/// Batch variant `evaluate_batch` fans the work out across cores via
/// rayon while the Python GIL is released by the caller.

use crate::energy;
use crate::fold;

/// Result tuple: `(e_on, e_off, mfe, mfe_dot_bracket)` in 0.01 kcal/mol.
pub type EvalResult = (i32, i32, i32, String);

pub fn evaluate_one(
    seq: &[u8],
    pair_table_on: &[i32],
    pair_table_off: &[i32],
) -> EvalResult {
    let e_on = energy::eval_energy(seq, pair_table_on);
    let e_off = energy::eval_energy(seq, pair_table_off);
    let (_fold_e, pairs, db) = fold::fold_mfe_full(seq);
    // Defensive re-eval with the same model used for the targets.
    let mfe = energy::eval_energy(seq, &pairs);
    (e_on, e_off, mfe, db)
}

pub fn evaluate_batch(
    seqs: &[Vec<u8>],
    pair_table_on: &[i32],
    pair_table_off: &[i32],
) -> Vec<EvalResult> {
    use rayon::prelude::*;
    seqs.par_iter()
        .map(|s| evaluate_one(s, pair_table_on, pair_table_off))
        .collect()
}
