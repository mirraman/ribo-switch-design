mod params;
mod energy;
mod fold;

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

#[pymodule]
fn ribo_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eval_energy, m)?)?;
    m.add_function(wrap_pyfunction!(fold_mfe, m)?)?;
    Ok(())
}
