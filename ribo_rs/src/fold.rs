/// fold_mfe — Zuker MFE folding algorithm in Rust.
///
/// Mirrors fold.py exactly. Returns (mfe_energy, dot_bracket_string).
/// Uses i64 for DP tables to avoid overflow during accumulation.

use crate::params::*;

const INF64: i64 = 10_000_000_000;

pub fn fold_mfe(seq: &[u8]) -> (i32, String) {
    let n = seq.len();

    if n < MIN_HAIRPIN + 2 {
        return (0, ".".repeat(n));
    }

    // DP tables: v[i][j], w[i][j], wm[i][j] — flattened row-major
    let mut v  = vec![INF64;   n * n];
    let mut w  = vec![0i64;    n * n];
    let mut wm = vec![INF64;   n * n];

    macro_rules! V  { ($i:expr,$j:expr) => { v [$i*n+$j] } }
    macro_rules! W  { ($i:expr,$j:expr) => { w [$i*n+$j] } }
    macro_rules! WM { ($i:expr,$j:expr) => { wm[$i*n+$j] } }

    for span in (MIN_HAIRPIN + 2)..=n {
        for i in 0..=(n - span) {
            let j = i + span - 1;

            // ── V[i][j] ──────────────────────────────────────────────────────
            if can_pair(seq, i, j) {
                V!(i, j) = fill_v(seq, &v, &wm, i, j, n);
            }

            // ── WM[i][j] ─────────────────────────────────────────────────────
            WM!(i, j) = fill_wm(seq, &v, &wm, i, j, n);

            // ── W[i][j] ──────────────────────────────────────────────────────
            W!(i, j) = fill_w(&v, &w, i, j, n);
        }
    }

    let mfe = W!(0, n - 1) as i32;

    // Traceback
    let mut pairs = vec![-1i32; n];
    trace_w(seq, &v, &w, &wm, 0, n - 1, n, &mut pairs);

    let mut db = vec![b'.'; n];
    for i in 0..n {
        if pairs[i] > i as i32 {
            db[i] = b'(';
            db[pairs[i] as usize] = b')';
        }
    }
    let structure = String::from_utf8(db).unwrap();

    (mfe, structure)
}

// ─── DP fill functions ────────────────────────────────────────────────────────

fn fill_v(seq: &[u8], v: &[i64], wm: &[i64], i: usize, j: usize, n: usize) -> i64 {
    let pi = match pair_index(seq[i], seq[j]) { Some(x) => x, None => return INF64 };
    let mut best = hairpin64(seq, i, j, pi);

    // Stack / Bulge / Interior
    let max_int = (j - i - 2).min(30);
    for p in i + 1..j {
        let nl = p - i - 1;
        if nl > max_int { break; }
        for q in (p + 1..j).rev() {
            let nr = j - q - 1;
            if nl + nr > max_int { continue; }
            if !can_pair(seq, p, q) { continue; }
            if q - p - 1 < MIN_HAIRPIN { continue; }

            let vp = v[p * n + q];
            if vp >= INF64 { continue; }

            let pp = match pair_index(seq[p], seq[q]) { Some(x) => x, None => continue };

            let e = if nl == 0 && nr == 0 {
                STACK[pi][pp] as i64 + vp
            } else if nl == 0 || nr == 0 {
                bulge64(pi, pp, nl + nr) + vp
            } else {
                interior64(seq, i, j, p, q, pi, pp, nl, nr) + vp
            };

            if e < best { best = e; }
        }
    }

    // Multiloop
    if j - i - 1 >= 2 * (MIN_HAIRPIN + 2) {
        let mut ml_base = (ML_OFFSET + ML_PER_BRANCH) as i64;
        if is_au_gu(pi) { ml_base += TERMINAL_AU_PENALTY as i64; }
        for k in (i + 2 + MIN_HAIRPIN)..(j - MIN_HAIRPIN - 1) {
            let wml = wm[(i + 1) * n + k];
            let wmr = wm[(k + 1) * n + j - 1];
            if wml < INF64 && wmr < INF64 {
                let e = ml_base + wml + wmr;
                if e < best { best = e; }
            }
        }
    }

    best
}

fn fill_wm(seq: &[u8], v: &[i64], wm: &[i64], i: usize, j: usize, n: usize) -> i64 {
    let mut best = INF64;

    // Case 1: (i,j) branch
    if can_pair(seq, i, j) {
        let vi = v[i * n + j];
        if vi < INF64 {
            let pi = pair_index(seq[i], seq[j]).unwrap();
            let mut e = vi + ML_PER_BRANCH as i64;
            if is_au_gu(pi) { e += TERMINAL_AU_PENALTY as i64; }
            if e < best { best = e; }
        }
    }
    // Case 2: i unpaired
    if i + 1 <= j {
        let e = wm[(i + 1) * n + j];
        if e < INF64 {
            let e2 = e + ML_PER_UNPAIRED as i64;
            if e2 < best { best = e2; }
        }
    }
    // Case 3: j unpaired
    if j >= 1 {
        let e = wm[i * n + j - 1];
        if e < INF64 {
            let e2 = e + ML_PER_UNPAIRED as i64;
            if e2 < best { best = e2; }
        }
    }
    // Case 4: bifurcation
    for k in i + 1..j {
        let el = wm[i * n + k];
        let er = wm[(k + 1) * n + j];
        if el < INF64 && er < INF64 {
            let e = el + er;
            if e < best { best = e; }
        }
    }

    best
}

fn fill_w(v: &[i64], w: &[i64], i: usize, j: usize, n: usize) -> i64 {
    let mut best = 0i64;

    // (i,j) paired
    let vi = v[i * n + j];
    if vi < INF64 && vi < best { best = vi; }

    // i unpaired
    if i + 1 <= j {
        let e = w[(i + 1) * n + j];
        if e < best { best = e; }
    }
    // j unpaired
    if j >= 1 {
        let e = w[i * n + j - 1];
        if e < best { best = e; }
    }
    // bifurcation
    for k in i + 1..j {
        let e = w[i * n + k] + w[(k + 1) * n + j];
        if e < best { best = e; }
    }

    best
}

// ─── Traceback ────────────────────────────────────────────────────────────────

fn trace_w(
    seq: &[u8], v: &[i64], w: &[i64], wm: &[i64],
    i: usize, j: usize, n: usize,
    pairs: &mut Vec<i32>,
) {
    if i >= j { return; }
    let target = w[i * n + j];
    if target == 0 { return; }

    // Check V[i][j]
    let vi = v[i * n + j];
    if vi < INF64 && vi == target {
        pairs[i] = j as i32;
        pairs[j] = i as i32;
        trace_v(seq, v, w, wm, i, j, n, pairs);
        return;
    }
    // i unpaired
    if i + 1 <= j && w[(i + 1) * n + j] == target {
        trace_w(seq, v, w, wm, i + 1, j, n, pairs);
        return;
    }
    // j unpaired
    if j >= 1 && w[i * n + j - 1] == target {
        trace_w(seq, v, w, wm, i, j - 1, n, pairs);
        return;
    }
    // bifurcation
    for k in i + 1..j {
        if w[i * n + k] + w[(k + 1) * n + j] == target {
            trace_w(seq, v, w, wm, i, k, n, pairs);
            trace_w(seq, v, w, wm, k + 1, j, n, pairs);
            return;
        }
    }
}

fn trace_v(
    seq: &[u8], v: &[i64], w: &[i64], wm: &[i64],
    i: usize, j: usize, n: usize,
    pairs: &mut Vec<i32>,
) {
    let target = v[i * n + j];
    let pi = match pair_index(seq[i], seq[j]) { Some(x) => x, None => return };

    // Hairpin?
    if hairpin64(seq, i, j, pi) == target { return; }

    // Stack / Bulge / Interior?
    let max_int = (j - i - 2).min(30);
    for p in i + 1..j {
        let nl = p - i - 1;
        if nl > max_int { break; }
        for q in (p + 1..j).rev() {
            let nr = j - q - 1;
            if nl + nr > max_int { continue; }
            if !can_pair(seq, p, q) { continue; }
            if q - p - 1 < MIN_HAIRPIN { continue; }
            let vp = v[p * n + q];
            if vp >= INF64 { continue; }
            let pp = match pair_index(seq[p], seq[q]) { Some(x) => x, None => continue };
            let e = if nl == 0 && nr == 0 {
                STACK[pi][pp] as i64 + vp
            } else if nl == 0 || nr == 0 {
                bulge64(pi, pp, nl + nr) + vp
            } else {
                interior64(seq, i, j, p, q, pi, pp, nl, nr) + vp
            };
            if e == target {
                pairs[p] = q as i32;
                pairs[q] = p as i32;
                trace_v(seq, v, w, wm, p, q, n, pairs);
                return;
            }
        }
    }

    // Multiloop?
    if j - i - 1 >= 2 * (MIN_HAIRPIN + 2) {
        let mut ml_base = (ML_OFFSET + ML_PER_BRANCH) as i64;
        if is_au_gu(pi) { ml_base += TERMINAL_AU_PENALTY as i64; }
        for k in (i + 2 + MIN_HAIRPIN)..(j - MIN_HAIRPIN - 1) {
            let wml = wm[(i + 1) * n + k];
            let wmr = wm[(k + 1) * n + j - 1];
            if wml < INF64 && wmr < INF64 && ml_base + wml + wmr == target {
                trace_wm(seq, v, w, wm, i + 1, k, n, pairs);
                trace_wm(seq, v, w, wm, k + 1, j - 1, n, pairs);
                return;
            }
        }
    }
}

fn trace_wm(
    seq: &[u8], v: &[i64], w: &[i64], wm: &[i64],
    i: usize, j: usize, n: usize,
    pairs: &mut Vec<i32>,
) {
    if i > j { return; }
    let target = wm[i * n + j];
    if target >= INF64 { return; }

    // Branch
    if can_pair(seq, i, j) {
        let vi = v[i * n + j];
        if vi < INF64 {
            let pi = pair_index(seq[i], seq[j]).unwrap();
            let mut e = vi + ML_PER_BRANCH as i64;
            if is_au_gu(pi) { e += TERMINAL_AU_PENALTY as i64; }
            if e == target {
                pairs[i] = j as i32;
                pairs[j] = i as i32;
                trace_v(seq, v, w, wm, i, j, n, pairs);
                return;
            }
        }
    }
    // i unpaired
    if i + 1 <= j {
        let e = wm[(i + 1) * n + j];
        if e < INF64 && e + ML_PER_UNPAIRED as i64 == target {
            trace_wm(seq, v, w, wm, i + 1, j, n, pairs);
            return;
        }
    }
    // j unpaired
    if j >= 1 {
        let e = wm[i * n + j - 1];
        if e < INF64 && e + ML_PER_UNPAIRED as i64 == target {
            trace_wm(seq, v, w, wm, i, j - 1, n, pairs);
            return;
        }
    }
    // bifurcation
    for k in i + 1..j {
        let el = wm[i * n + k];
        let er = wm[(k + 1) * n + j];
        if el < INF64 && er < INF64 && el + er == target {
            trace_wm(seq, v, w, wm, i, k, n, pairs);
            trace_wm(seq, v, w, wm, k + 1, j, n, pairs);
            return;
        }
    }
}

// ─── Shared helpers ───────────────────────────────────────────────────────────

#[inline]
fn can_pair(seq: &[u8], i: usize, j: usize) -> bool {
    pair_index(seq[i], seq[j]).is_some()
}

#[inline]
fn hairpin64(seq: &[u8], i: usize, j: usize, pi: usize) -> i64 {
    let e = hairpin_e(seq, i, j, pi);
    if e >= INF { INF64 } else { e as i64 }
}

fn hairpin_e(seq: &[u8], i: usize, j: usize, pi: usize) -> i32 {
    let size = j - i - 1;
    if size < MIN_HAIRPIN { return INF; }

    let mut energy = if size <= 30 {
        HAIRPIN_INIT[size]
    } else {
        HAIRPIN_INIT[30]
            + (LOOP_EXTRAPOLATION_COEFF * (size as f64 / 30.0).ln() * 100.0).round() as i32
    };

    if size == 3 {
        let w = &seq[i..=j];
        let bonus = triloop_bonus(w);
        if bonus != 0 { energy = bonus; }
        if is_au_gu(pi) { energy += TERMINAL_AU_PENALTY; }
        if (i + 1..j).all(|k| seq[k] == 1) { energy += HAIRPIN_C3; }
        return energy;
    }
    if size == 4 {
        let w = &seq[i..=j];
        let bonus = tetraloop_bonus(w);
        if bonus != 0 { energy = bonus; }
    }
    let b5 = seq[i + 1] as usize;
    let b3 = seq[j - 1] as usize;
    energy += HAIRPIN_MM[pi][b5][b3];
    let mm = (seq[i + 1], seq[j - 1]);
    if mm == (3, 3) || mm == (2, 0) { energy += HAIRPIN_UU_GA_BONUS; }
    if mm == (2, 2) { energy += HAIRPIN_GG_BONUS; }
    if pi == 4 && i >= 2 && seq[i - 1] == 2 && seq[i - 2] == 2 { energy += HAIRPIN_SPECIAL_GU; }
    if (i + 1..j).all(|k| seq[k] == 1) {
        energy += HAIRPIN_C_SLOPE * size as i32 + HAIRPIN_C_INTERCEPT;
    }
    energy
}

#[inline]
fn bulge64(pi: usize, pp: usize, size: usize) -> i64 {
    let mut e = if size <= 30 { BULGE_INIT[size] } else {
        BULGE_INIT[30] + (LOOP_EXTRAPOLATION_COEFF * (size as f64 / 30.0).ln() * 100.0).round() as i32
    };
    if size == 1 { e += STACK[pi][pp]; }
    if is_au_gu(pi) { e += TERMINAL_AU_PENALTY; }
    if is_au_gu(pp) { e += TERMINAL_AU_PENALTY; }
    e as i64
}

#[inline]
fn interior64(
    seq: &[u8],
    i: usize, j: usize,
    p: usize, q: usize,
    pi: usize, pp: usize,
    nl: usize, nr: usize,
) -> i64 {
    let total = nl + nr;

    if nl == 1 && nr == 1 {
        let mm5 = seq[i + 1] as usize;
        let mm3 = seq[j - 1] as usize;
        let val = INT11[pi][pp][mm5][mm3];
        if val < INF { return val as i64; }
    }

    let mut e = if total <= 30 { INTERIOR_INIT[total] } else {
        INTERIOR_INIT[30] + (LOOP_EXTRAPOLATION_COEFF * (total as f64 / 30.0).ln() * 100.0).round() as i32
    };

    let asym = (nl as i32 - nr as i32).unsigned_abs() as i32;
    e += (NINIO_M * asym).min(NINIO_MAX);

    if !(nl == 1 && nr == 1) {
        let b5o = seq[i + 1] as usize;
        let b3o = seq[j - 1] as usize;
        e += INTERIOR_MM[pi][b5o][b3o];
        let b3i = if p > 0 { seq[p - 1] as usize } else { 0 };
        let b5i = if q + 1 < seq.len() { seq[q + 1] as usize } else { 0 };
        e += INTERIOR_MM[pp][b3i][b5i];
    }

    if is_au_gu(pi) { e += TERMINAL_AU_PENALTY; }
    if is_au_gu(pp) { e += TERMINAL_AU_PENALTY; }

    e as i64
}
