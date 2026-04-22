use crate::energy;
use crate::fold;
pub type EvalResult = (i32, i32, i32, String);
pub fn evaluate_one(
    seq: &[u8],
    pair_table_on: &[i32],
    pair_table_off: &[i32],
) -> EvalResult {
    let e_on = energy::eval_energy(seq, pair_table_on);
    let e_off = energy::eval_energy(seq, pair_table_off);
    let (_fold_e, pairs, db) = fold::fold_mfe_full(seq);
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
