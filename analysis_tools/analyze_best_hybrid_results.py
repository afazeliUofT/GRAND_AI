#!/usr/bin/env python3
"""Concise analysis for the latest hybrid result set.

Usage:
  python analysis_tools/analyze_best_hybrid_results.py
  python analysis_tools/analyze_best_hybrid_results.py results/run_<jobname>_<jobid>

It auto-detects the newest result directory when no path is given.
It compares ldpc15 against the strongest available hybrid in this order.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

DecoderRow = Dict[str, str]
HYBRID_PRIORITY = ["hybairtgsub15", "hybairtg15", "hybairpmix15", "hybairpfix15", "hybairpwin15", "hybairprobe15", "hybairprobefix15", "hybairwroi15", "hybairdtbwin15", "hybairroi15", "hybairdtbroi15", "hybairdtroi15", "hybairdtb15", "hybairdt15", "hybair15", "hybmeta15", "hybahr15", "hybosd15", "hybbgr15", "hyb15"]


def _to_float(value: object) -> float:
    if value is None:
        return float("nan")
    value = str(value).strip()
    if value == "" or value.lower() == "nan":
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _read_csv(path: str) -> List[DecoderRow]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _latest_result_dir(repo_root: str) -> str:
    patterns = [
        os.path.join(repo_root, "results", "run_*_*"),
        os.path.join(repo_root, "results", "run_*"),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend([p for p in glob.glob(pattern) if os.path.isdir(p)])
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(f"No result directories found under {os.path.join(repo_root, 'results')}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _find_one(pattern: str) -> str:
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {pattern}, found {len(matches)}")
    return matches[0]


def _latest_log_file(repo_root: str) -> Optional[str]:
    matches = glob.glob(os.path.join(repo_root, 'run_*.out'))
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def _latest_sbatch_file(repo_root: str, log_path: Optional[str] = None) -> Optional[str]:
    matches = glob.glob(os.path.join(repo_root, 'run_*.sbatch'))
    if not matches:
        return None
    if log_path:
        base = os.path.basename(log_path)
        prefix = re.sub(r'-\d+\.out$', '', base)
        for p in matches:
            if os.path.basename(p).replace('.sbatch', '') == prefix:
                return p
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def _extract_hybrid_labels_from_log(path: str) -> List[str]:
    out: List[str] = []
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    except Exception:
        return out
    for m in re.finditer(r'Decoder label\s*:\s*([^\s]+)', text):
        label = m.group(1).strip()
        if label.startswith('hyb') and label not in out:
            out.append(label)
    return out


def _extract_verify_triplets(path: str) -> List[str]:
    vals: List[str] = []
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if '[VERIFY] Receiver9 decoder=' in line:
                    vals.append(line.strip())
    except Exception:
        return vals
    return vals


def _extract_expected_triplet_from_sbatch(path: str) -> Tuple[str, str, str]:
    dec = pol = sel = ""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    except Exception:
        return dec, pol, sel
    m = re.search(r'EXPECTED_HYBRID_DECODER=([^\s]+)', text)
    if m:
        dec = m.group(1).strip()
    m = re.search(r'EXPECTED_HYBRID_POLICY=([^\s]+)', text)
    if m:
        pol = m.group(1).strip()
    m = re.search(r'EXPECTED_HYBRID_SELECTION=([^\s]+)', text)
    if m:
        sel = m.group(1).strip()
    return dec, pol, sel


def _load_result_set(base_dir: str) -> Tuple[str, List[DecoderRow], List[DecoderRow], List[DecoderRow]]:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Result directory not found: {base_dir}")
    summary = _find_one(os.path.join(base_dir, "*_summary.csv"))
    diag = _find_one(os.path.join(base_dir, "*_summary_diagnostics.csv"))
    tails = _find_one(os.path.join(base_dir, "*_summary_tails.csv"))
    return base_dir, _read_csv(summary), _read_csv(diag), _read_csv(tails)


def _index(rows: Iterable[DecoderRow]) -> Dict[Tuple[float, str], DecoderRow]:
    out: Dict[Tuple[float, str], DecoderRow] = {}
    for row in rows:
        out[(_to_float(row.get("snr_db")), str(row.get("decoder", "")))] = row
    return out


def _fmt_num(x: float, digits: int = 3) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


def _fmt_pct(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{100.0 * x:.2f}%"


def _print_header(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def _pick_hybrid(summary_idx: Dict[Tuple[float, str], DecoderRow]) -> str:
    present = {dec for (_snr, dec) in summary_idx.keys()}
    for dec in HYBRID_PRIORITY:
        if dec in present:
            return dec
    raise KeyError("No supported hybrid decoder found in summary CSV")


def _safe_mean(xs: List[float]) -> float:
    vals = [x for x in xs if not math.isnan(x)]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> int:
    repo_root = os.getcwd()
    if len(sys.argv) > 2:
        print("Usage: python analysis_tools/analyze_best_hybrid_results.py [results/run_<jobname>_<jobid>]", file=sys.stderr)
        return 2

    try:
        result_dir = sys.argv[1] if len(sys.argv) == 2 else _latest_result_dir(repo_root)
        result_dir, summary_rows, diag_rows, tail_rows = _load_result_set(result_dir)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    summary_idx = _index(summary_rows)
    diag_idx = _index(diag_rows)
    tail_idx = _index(tail_rows)
    snrs = sorted({snr for (snr, _dec) in summary_idx.keys()})

    try:
        hybrid_name = _pick_hybrid(summary_idx)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    _print_header("Using result set")
    print(result_dir)

    _print_header("Comparing decoders")
    print(f"Legacy baseline : ldpc15")
    print(f"Hybrid analyzed : {hybrid_name}")
    if hybrid_name == "hybair15":
        print("WARNING: active result is still heuristic AIR, not the distilled-tree run.")
    if hybrid_name == "hybairdtbroi15":
        print("INFO: active result is the verified distilled-tree-bandit + ROI rank run.")
    if hybrid_name == "hybairroi15":
        print("INFO: active result is the forced-explore ROI controller run.")
    if hybrid_name == "hybairprobe15":
        print("INFO: active result is the probe-and-escalate ROI run.")
    if hybrid_name == "hybairpwin15":
        print("INFO: active result is the windowed probe-and-escalate ROI run.")
    if hybrid_name == "hybairtg15":
        print("INFO: active result is the Tanner-graph distilled ROI run.")
    if hybrid_name == "hybairtgsub15":
        print("INFO: active result is the Tanner-subgraph distilled ROI run.")

    log_path = _latest_log_file(repo_root)
    if log_path is not None:
        labels = _extract_hybrid_labels_from_log(log_path)
        verify_lines = _extract_verify_triplets(log_path)
        print(f"Latest log       : {os.path.basename(log_path)}")
        print(f"Hybrid labels in log: {', '.join(labels) if labels else 'none'}")
        if verify_lines:
            print(f"Latest verify line : {verify_lines[-1]}")
        if labels and hybrid_name not in labels:
            print("WARNING: summary/log decoder mismatch. Results may be stale or renamed.")

    sbatch_path = _latest_sbatch_file(repo_root, log_path)
    if sbatch_path and os.path.isfile(sbatch_path):
        exp_dec, exp_pol, exp_sel = _extract_expected_triplet_from_sbatch(sbatch_path)
        if exp_dec or exp_pol or exp_sel:
            print(f"Current sbatch expects : decoder={exp_dec or 'n/a'} policy={exp_pol or 'n/a'} selection={exp_sel or 'n/a'}")
        if log_path is not None and verify_lines and (exp_dec or exp_pol or exp_sel):
            latest_verify = verify_lines[-1]
            mismatch = (exp_dec and (f'decoder={exp_dec}' not in latest_verify)) or (exp_pol and (f'policy={exp_pol}' not in latest_verify)) or (exp_sel and (f'selection={exp_sel}' not in latest_verify))
            if mismatch:
                print('WARNING: current code modules do not match the latest pushed run/log. Results are stale relative to the code.')


    _print_header("Hybrid vs legacy LDPC")
    print(
        "SNR(dB) | BER gain | FER gain | latency x | LDPC us | Hybrid us\n"
        "--------|----------|----------|-----------|---------|----------"
    )

    ber_gains: List[float] = []
    fer_gains: List[float] = []
    lat_factors: List[float] = []

    for snr in snrs:
        ld = summary_idx.get((snr, "ldpc15"))
        hy = summary_idx.get((snr, hybrid_name))
        if ld is None or hy is None:
            continue

        ld_ber = _to_float(ld.get("ber"))
        hy_ber = _to_float(hy.get("ber"))
        ld_fer = _to_float(ld.get("fer"))
        hy_fer = _to_float(hy.get("fer"))
        ld_t = _to_float(ld.get("avg_hw_time_us"))
        hy_t = _to_float(hy.get("avg_hw_time_us"))

        ber_gain = (ld_ber - hy_ber) / ld_ber if ld_ber > 0 else float("nan")
        fer_gain = (ld_fer - hy_fer) / ld_fer if ld_fer > 0 else float("nan")
        lat_factor = hy_t / ld_t if ld_t > 0 else float("nan")

        ber_gains.append(ber_gain)
        fer_gains.append(fer_gain)
        lat_factors.append(lat_factor)

        print(
            f"{snr:7.1f} | {100.0*ber_gain:8.2f}% | {100.0*fer_gain:8.2f}% |"
            f" {lat_factor:9.2f} | {ld_t:7.3f} | {hy_t:8.3f}"
        )

    print()
    print(f"Average BER gain : {100.0*_safe_mean(ber_gains):.2f}%")
    print(f"Average FER gain : {100.0*_safe_mean(fer_gains):.2f}%")
    print(f"Average latency x: {_safe_mean(lat_factors):.2f}")

    hyb_diag_present = any((snr, hybrid_name) in diag_idx for snr in snrs)
    if hyb_diag_present:
        _print_header("Stage-2 efficiency")
        print(
            "SNR(dB) | invoke | true-fix if invoked | avg tries | avg success snap\n"
            "--------|--------|----------------------|-----------|-----------------"
        )
        zero_invocation = True
        for snr in snrs:
            hy = diag_idx.get((snr, hybrid_name))
            if hy is None:
                continue
            inv = _to_float(hy.get('stage2_invocation_rate'))
            if (not math.isnan(inv)) and inv > 0.0:
                zero_invocation = False
            print(
                f"{snr:7.1f} | {_fmt_pct(inv):>6} |"
                f" {_fmt_pct(_to_float(hy.get('stage2_true_fix_rate_if_invoked'))):>20} |"
                f" {_fmt_num(_to_float(hy.get('avg_snapshot_attempts_if_invoked')), 3):>9} |"
                f" {_fmt_num(_to_float(hy.get('avg_snapshot_success_iter_if_fixed')), 3):>15}"
            )

        if zero_invocation:
            print("WARNING: stage-2 invocation is zero at all SNR points, so the hybrid collapsed to LDPC-only behavior.")

        if any((snr, hybrid_name) in diag_idx for snr in snrs):
            print("Tip: if snapshot_success_at_15 stays zero and fallback success stays zero, remove them in the next run.")

        sample = diag_idx.get((snrs[0], hybrid_name)) if snrs else None
        if sample and ("probe_invocation_rate" in sample):
            _print_header("Probe-and-escalate behavior")
            print(
                "SNR(dB) | probe invoke | probe success | probe escalate | probe syndrome drop\n"
                "--------|--------------|---------------|----------------|-------------------"
            )
            all_zero_drop = True
            all_zero_escal = True
            for snr in snrs:
                hy = diag_idx.get((snr, hybrid_name))
                if hy is None:
                    continue
                drop_v = _to_float(hy.get('probe_syndrome_drop_mean_if_invoked'))
                esc_v = _to_float(hy.get('probe_escalation_rate_if_invoked'))
                if (not math.isnan(drop_v)) and drop_v > 0.0:
                    all_zero_drop = False
                if (not math.isnan(esc_v)) and esc_v > 0.0:
                    all_zero_escal = False
                print(
                    f"{snr:7.1f} | {_fmt_pct(_to_float(hy.get('probe_invocation_rate'))):>12} |"
                    f" {_fmt_pct(_to_float(hy.get('probe_success_rate_if_invoked'))):>13} |"
                    f" {_fmt_pct(_to_float(hy.get('probe_escalation_rate_if_invoked'))):>14} |"
                    f" {_fmt_num(_to_float(hy.get('probe_syndrome_drop_mean_if_invoked')), 3):>17}"
                )
            if all_zero_drop:
                print("WARNING: probe saw zero syndrome drop on every invocation. This usually means the progress signal is not being captured, or the tiny probe window is too weak.")
            if all_zero_escal:
                print("WARNING: probe never escalated. If invoke>0 but escalate=0 everywhere, the controller is stuck in tiny-only mode.")

        sample = diag_idx.get((snrs[0], hybrid_name)) if snrs else None
        if sample and ("ai_gate_skip_rate" in sample or "ai_gate_skip_rate_if_failed" in sample):
            _print_header("AI gate behavior")
            sample_policy = str(sample.get("ai_gate_policy_mode", "")).strip() if sample else ""
            if sample_policy:
                print(f"Policy mode: {sample_policy}")
            use_failed = "ai_gate_skip_rate_if_failed" in sample
            suffix = "_if_failed" if use_failed else ""
            scope = "failed stage-1 frames" if use_failed else "invoked stage-2 frames"
            print(f"Scope: {scope}")
            print(
                "SNR(dB) | skip | first skip | tiny | full | meta | decisions | escal. | gate conf | gate promise\n"
                "--------|------|------------|------|------|------|-----------|--------|-----------|-------------"
            )
            for snr in snrs:
                hy = diag_idx.get((snr, hybrid_name))
                if hy is None:
                    continue
                print(
                    f"{snr:7.1f} | {_fmt_pct(_to_float(hy.get('ai_gate_skip_rate' + suffix))):>4} |"
                    f" {_fmt_pct(_to_float(hy.get('ai_gate_first_skip_rate' + suffix))):>10} |"
                    f" {_fmt_pct(_to_float(hy.get('ai_gate_tiny_rate' + suffix))):>4} |"
                    f" {_fmt_pct(_to_float(hy.get('ai_gate_full_rate' + suffix))):>4} |"
                    f" {_fmt_pct(_to_float(hy.get('ai_gate_meta_rate' + suffix))):>4} |"
                    f" {_fmt_num(_to_float(hy.get('ai_gate_decision_count_mean' + ('_if_failed' if use_failed else '_if_invoked'))), 3):>9} |"
                    f" {_fmt_pct(_to_float(hy.get('ai_gate_escalation_rate' + ('_if_failed' if use_failed else '_if_invoked')))):>6} |"
                    f" {_fmt_num(_to_float(hy.get('ai_gate_confidence_mean' + ('_if_failed' if use_failed else '_if_invoked'))), 3):>9} |"
                    f" {_fmt_num(_to_float(hy.get('ai_gate_promise_mean' + ('_if_failed' if use_failed else '_if_invoked'))), 3):>11}"
                )

        _print_header("Residual structure on invoked frames")
        print(
            "SNR(dB) | avg bit errs | avg syndrome wt | avg span | avg runs | avg block conc\n"
            "--------|--------------|-----------------|----------|----------|----------------"
        )
        for snr in snrs:
            hy = diag_idx.get((snr, hybrid_name))
            if hy is None:
                continue
            print(
                f"{snr:7.1f} | {_fmt_num(_to_float(hy.get('avg_stage1_bit_errors_if_invoked')), 1):>12} |"
                f" {_fmt_num(_to_float(hy.get('avg_stage1_syndrome_weight_if_invoked')), 1):>15} |"
                f" {_fmt_num(_to_float(hy.get('avg_stage1_error_span_if_invoked')), 1):>8} |"
                f" {_fmt_num(_to_float(hy.get('avg_stage1_error_runs_if_invoked')), 1):>8} |"
                f" {_fmt_num(_to_float(hy.get('avg_stage1_block_concentration_if_invoked')), 3):>14}"
            )

    hyb_tail_present = any((snr, hybrid_name) in tail_idx for snr in snrs)
    if hyb_tail_present:
        _print_header("Tail latency / search effort")
        print(
            "SNR(dB) | hybrid p99 us | grand p99 us | patterns mean if grand | patterns p95 if grand\n"
            "--------|---------------|--------------|------------------------|----------------------"
        )
        for snr in snrs:
            hy = tail_idx.get((snr, hybrid_name))
            if hy is None:
                continue
            grand_p99_cycles = _to_float(hy.get("grand_cycles_p99"))
            grand_p99_us = grand_p99_cycles / 800.0 if not math.isnan(grand_p99_cycles) else float("nan")
            print(
                f"{snr:7.1f} | {_fmt_num(_to_float(hy.get('hw_time_us_p99')), 3):>13} |"
                f" {_fmt_num(grand_p99_us, 3):>12} |"
                f" {_fmt_num(_to_float(hy.get('patterns_tested_mean_if_grand')), 1):>22} |"
                f" {_fmt_num(_to_float(hy.get('patterns_tested_p95_if_grand')), 1):>20}"
            )

    _print_header("Main conclusion")
    print("1) The target outcome is higher FER gain than hybahr15 with a clearly lower latency factor.")
    print("2) If stage-2 invocation collapses to zero, the gate is too conservative and the run is effectively plain LDPC.")
    print("3) The next target is meaningful FER gain with stage-2 active on a controlled minority of failed frames, not on all frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
