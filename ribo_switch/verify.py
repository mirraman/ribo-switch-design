from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ribo_switch.types import Base, Sequence, Structure
from ribo_switch.structure import parse_dot_bracket
from ribo_switch.turner import TurnerParams

def bp_distance(a: Structure, b: Structure) -> int:

    set_a = set(a.pairs)
    set_b = set(b.pairs)
    return len(set_a.symmetric_difference(set_b))


def bp_confusion(pred: Structure, target: Structure) -> tuple[int, int, int]:

    pred_set   = set(pred.pairs)
    target_set = set(target.pairs)
    tp = len(pred_set & target_set)
    fp = len(pred_set - target_set)
    fn = len(target_set - pred_set)
    return tp, fp, fn


def bp_precision_recall_f1(
    pred: Structure,
    target: Structure,
) -> tuple[float, float, float]:

    tp, fp, fn = bp_confusion(pred, target)

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return precision, recall, f1


def position_match(a: Structure, b: Structure) -> tuple[int, int]:
    n = a.length
    assert b.length == n, "Structures must have the same length"
    matched = sum(
        1 for i in range(n) if a.pair_table[i] == b.pair_table[i]
    )
    return matched, n


def position_f1(a: Structure, b: Structure) -> float:
    n = a.length
    assert b.length == n, "Structures must have the same length"

    tp = fp = fn = 0
    for i in range(n):
        a_paired = a.pair_table[i] != -1
        b_paired = b.pair_table[i] != -1
        if a_paired and b_paired:
            tp += 1
        elif a_paired and not b_paired:
            fp += 1
        elif not a_paired and b_paired:
            fn += 1

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)

@dataclass
class VerificationReport:
    label:              str
    sequence:           str
    target_db:          str
    predicted_db:       str
    mfe_kcal:           float
    target_energy_kcal: float
    gap_kcal:           float
    bp_dist:            int
    bp_tp:              int
    bp_fp:              int
    bp_fn:              int
    bp_precision:       float
    bp_recall:          float
    bp_f1:              float
    pos_match:          int
    pos_total:          int
    pos_f1:             float
    matches_exactly:    bool

    def summary_lines(self) -> list[str]:
        verdict = "EXACT MATCH" if self.matches_exactly else f"MISMATCH (bpΔ={self.bp_dist})"
        return [
            f"Label       : {self.label}",
            f"Sequence    : {self.sequence}",
            f"Target      : {self.target_db}",
            f"Predicted   : {self.predicted_db}",
            f"MFE energy  : {self.mfe_kcal:.2f} kcal/mol",
            f"Target ΔG   : {self.target_energy_kcal:.2f} kcal/mol",
            f"Gap (ΔG−MFE): {self.gap_kcal:.2f} kcal/mol",
            f"bp-distance : {self.bp_dist}  "
            f"(TP={self.bp_tp}, FP={self.bp_fp}, FN={self.bp_fn})",
            f"bp-F1       : {self.bp_f1:.3f}  "
            f"(precision={self.bp_precision:.3f}, recall={self.bp_recall:.3f})",
            f"pos-match   : {self.pos_match}/{self.pos_total}  "
            f"(pos-F1={self.pos_f1:.3f})",
            f"Verdict     : {verdict}",
        ]

def _seq_from_str(seq_str: str) -> Sequence:
    """Convert a plain ACGU string to a Sequence."""
    _map = {'A': Base.A, 'C': Base.C, 'G': Base.G, 'U': Base.U}
    bases = [_map[ch.upper()] for ch in seq_str]
    return Sequence(bases=bases)


def verify_sequence(
    seq_str: str,
    target_db: str,
    params: Optional[TurnerParams] = None,
    label: str = "ON",
) -> VerificationReport:

    from ribo_switch.rust_bridge import fold_mfe as _fold_mfe
    from ribo_switch.energy import eval_energy as _eval_energy

    if params is None:
        params = TurnerParams.turner2004()

    seq = _seq_from_str(seq_str)
    target_struct = parse_dot_bracket(target_db)

    fold_result = _fold_mfe(seq, params)
    pred_struct  = parse_dot_bracket(fold_result.mfe_structure)
    mfe_kcal     = fold_result.mfe_energy / 100.0

    target_energy_int = _eval_energy(seq, target_struct, params)
    target_energy_kcal = target_energy_int / 100.0
    gap_kcal = target_energy_kcal - mfe_kcal

    dist                        = bp_distance(pred_struct, target_struct)
    tp, fp, fn                  = bp_confusion(pred_struct, target_struct)
    precision, recall, bp_f1   = bp_precision_recall_f1(pred_struct, target_struct)
    pos_match, pos_total        = position_match(pred_struct, target_struct)
    pos_f1_val                  = position_f1(pred_struct, target_struct)

    return VerificationReport(
        label              = label,
        sequence           = seq_str,
        target_db          = target_db,
        predicted_db       = fold_result.mfe_structure,
        mfe_kcal           = mfe_kcal,
        target_energy_kcal = target_energy_kcal,
        gap_kcal           = gap_kcal,
        bp_dist            = dist,
        bp_tp              = tp,
        bp_fp              = fp,
        bp_fn              = fn,
        bp_precision       = precision,
        bp_recall          = recall,
        bp_f1              = bp_f1,
        pos_match          = pos_match,
        pos_total          = pos_total,
        pos_f1             = pos_f1_val,
        matches_exactly    = (dist == 0),
    )


def verify_against_both(
    seq_str: str,
    s_on_db: str,
    s_off_db: str,
    params: Optional[TurnerParams] = None,
) -> tuple[VerificationReport, VerificationReport]:

    from ribo_switch.rust_bridge import fold_mfe as _fold_mfe
    from ribo_switch.energy import eval_energy as _eval_energy

    if params is None:
        params = TurnerParams.turner2004()

    seq = _seq_from_str(seq_str)
    s_on  = parse_dot_bracket(s_on_db)
    s_off = parse_dot_bracket(s_off_db)

    fold_result = _fold_mfe(seq, params)
    pred_struct  = parse_dot_bracket(fold_result.mfe_structure)
    mfe_kcal     = fold_result.mfe_energy / 100.0

    def _report(target: Structure, target_db: str, label: str) -> VerificationReport:
        e_int  = _eval_energy(seq, target, params)
        e_kcal = e_int / 100.0
        dist                       = bp_distance(pred_struct, target)
        tp, fp, fn                 = bp_confusion(pred_struct, target)
        precision, recall, bp_f1   = bp_precision_recall_f1(pred_struct, target)
        pos_match, pos_total       = position_match(pred_struct, target)
        pos_f1_val                 = position_f1(pred_struct, target)
        return VerificationReport(
            label              = label,
            sequence           = seq_str,
            target_db          = target_db,
            predicted_db       = fold_result.mfe_structure,
            mfe_kcal           = mfe_kcal,
            target_energy_kcal = e_kcal,
            gap_kcal           = e_kcal - mfe_kcal,
            bp_dist            = dist,
            bp_tp              = tp,
            bp_fp              = fp,
            bp_fn              = fn,
            bp_precision       = precision,
            bp_recall          = recall,
            bp_f1              = bp_f1,
            pos_match          = pos_match,
            pos_total          = pos_total,
            pos_f1             = pos_f1_val,
            matches_exactly    = (dist == 0),
        )

    report_on  = _report(s_on,  s_on_db,  "ON")
    report_off = _report(s_off, s_off_db, "OFF")
    return report_on, report_off

def bp_distance_from_tables(
    pt_pred: list[int],
    pt_target: list[int],
) -> int:

    n = len(pt_pred)
    assert len(pt_target) == n, "pair tables must have the same length"
    dist = 0
    for i in range(n):
        p = pt_pred[i]
        t = pt_target[i]
        if i < p and i < t:
            if p != t:
                dist += 2  
        elif i < p:          
            dist += 1
        elif i < t:          
            dist += 1
    return dist


def bp_f1_from_tables(
    pt_pred: list[int],
    pt_target: list[int],
) -> float:
    n = len(pt_pred)
    tp = fp = fn = 0
    for i in range(n):
        p = pt_pred[i]
        t = pt_target[i]
        if i < p or i < t:     
            pred_pair   = i < p
            target_pair = i < t
            if pred_pair and target_pair:
                if p == t:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            elif pred_pair:
                fp += 1
            else:
                fn += 1

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
