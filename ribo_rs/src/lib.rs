mod params;
mod energy;
mod fold;
mod eval_combined;
use pyo3::prelude::*;
#[pyfunction]
fn eval_energy(seq: Vec<u8>, pair_table: Vec<i32>) -> PyResult<i32> {
    Ok(energy::eval_energy(&seq, &pair_table))
}
#[pyfunction]
fn fold_mfe(seq: Vec<u8>) -> PyResult<(i32, String)> {
    Ok(fold::fold_mfe(&seq))
}
#[pyfunction]
fn evaluate_candidate(
    seq: Vec<u8>,
    pair_table_on: Vec<i32>,
    pair_table_off: Vec<i32>,
) -> PyResult<(i32, i32, i32, String)> {
    Ok(eval_combined::evaluate_one(&seq, &pair_table_on, &pair_table_off))
}
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
