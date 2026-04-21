mod params;
mod energy;
mod fold;
mod eval_combined;

use pyo3::prelude::*;

/// Evaluate free energy of a sequence in a given structure.
///
/// Args:
///     seq:        list[int] — bases encoded A=0,C=1,G=2,U=3
///     pair_table: list[int] — pair_table[i]=j if paired, -1 if unpaired
///
/// Returns:
///     int — energy in 0.01 kcal/mol
#[pyfunction]
fn eval_energy(seq: Vec<u8>, pair_table: Vec<i32>) -> PyResult<i32> {
    Ok(energy::eval_energy(&seq, &pair_table))
}

/// Compute MFE structure for a sequence (Zuker algorithm).
///
/// Args:
///     seq: list[int] — bases encoded A=0,C=1,G=2,U=3
///
/// Returns:
///     (int, str) — (mfe_energy in 0.01 kcal/mol, dot-bracket structure)
#[pyfunction]
fn fold_mfe(seq: Vec<u8>) -> PyResult<(i32, String)> {
    Ok(fold::fold_mfe(&seq))
}

/// Fully evaluate one candidate: `e_on`, `e_off`, `mfe`, folded structure.
///
/// Returns `(e_on, e_off, mfe, mfe_dot_bracket)` in 0.01 kcal/mol. The `mfe`
/// is obtained by folding, then re-scoring the folded structure with the
/// same energy model that evaluated the targets — so `e_on - mfe` and
/// `e_off - mfe` are guaranteed ≥ 0.
#[pyfunction]
fn evaluate_candidate(
    seq: Vec<u8>,
    pair_table_on: Vec<i32>,
    pair_table_off: Vec<i32>,
) -> PyResult<(i32, i32, i32, String)> {
    Ok(eval_combined::evaluate_one(&seq, &pair_table_on, &pair_table_off))
}

/// Batch variant: evaluate many candidates in parallel (rayon).
///
/// Releases the Python GIL while folding runs so parallelism actually
/// scales across cores. `seqs` share the same `pair_table_on` /
/// `pair_table_off`.
#[pyfunction]
fn evaluate_batch(
    py: Python<'_>,
    seqs: Vec<Vec<u8>>,
    pair_table_on: Vec<i32>,
    pair_table_off: Vec<i32>,
) -> PyResult<Vec<(i32, i32, i32, String)>> {
    Ok(py.allow_threads(|| {
        eval_combined::evaluate_batch(&seqs, &pair_table_on, &pair_table_off)
    }))
}

#[pymodule]
fn ribo_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eval_energy, m)?)?;
    m.add_function(wrap_pyfunction!(fold_mfe, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_candidate, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_batch, m)?)?;
    Ok(())
}
