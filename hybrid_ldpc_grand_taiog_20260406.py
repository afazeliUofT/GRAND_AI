#!/usr/bin/env python3
import os
import sys
import math
import json
import time
import pickle
import datetime
import itertools
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# ------------------------- Optional acceleration -------------------------
_DISABLE_NUMBA = os.getenv("LDPC_GRAND_DISABLE_NUMBA", "0").strip().lower() in ("1", "true", "yes")
if not _DISABLE_NUMBA:
    try:
        from numba import njit, prange, set_num_threads, get_num_threads
        NUMBA_AVAILABLE = True
    except Exception:
        njit = None
        prange = range
        set_num_threads = None
        get_num_threads = None
        NUMBA_AVAILABLE = False
else:
    njit = None
    prange = range
    set_num_threads = None
    get_num_threads = None
    NUMBA_AVAILABLE = False


def _detect_num_threads() -> int:
    for key in ("LDPC_GRAND_NUM_THREADS", "SLURM_CPUS_PER_TASK", "NUMBA_NUM_THREADS"):
        val = os.environ.get(key, "").strip()
        if val:
            try:
                n = int(val)
                if n > 0:
                    return n
            except Exception:
                pass
    try:
        import multiprocessing
        n = multiprocessing.cpu_count()
        if n > 0:
            return int(n)
    except Exception:
        pass
    val = os.environ.get("OMP_NUM_THREADS", "").strip()
    if val:
        try:
            n = int(val)
            if n > 0:
                return n
        except Exception:
            pass
    return 1


NUMBA_THREADS = _detect_num_threads()
if NUMBA_AVAILABLE and set_num_threads is not None:
    try:
        set_num_threads(NUMBA_THREADS)
    except Exception:
        pass

# ------------------------- Optional TensorFlow runtime control -------------------------
if os.getenv("USE_GPU", "0").lower() not in ("1", "true", "yes"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.getenv("TF_CPP_MIN_LOG_LEVEL", "2"))

tf = None
if os.getenv("ENABLE_TF_RUNTIME_CONFIG", "0").lower() in ("1", "true", "yes"):
    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        tf = None

if tf is not None:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(int(os.getenv("TF_INTRA_OP", "1")))
        tf.config.threading.set_inter_op_parallelism_threads(int(os.getenv("TF_INTER_OP", "1")))
    except Exception:
        pass
    if os.getenv("USE_GPU", "0").lower() not in ("1", "true", "yes"):
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

# ------------------------- Sionna compatibility -------------------------
def _import_symbol_candidates(symbol_name: str, candidate_paths: List[str]):
    errors = []
    for module_path in candidate_paths:
        try:
            module = __import__(module_path, fromlist=[symbol_name])
            symbol = getattr(module, symbol_name)
            return symbol, f"{module_path}.{symbol_name}"
        except Exception as e:
            errors.append(f"{module_path}.{symbol_name}: {repr(e)}")
    return None, " | ".join(errors)


LDPC5GEncoder, _LDPC5G_IMPORT_DETAIL = _import_symbol_candidates(
    "LDPC5GEncoder",
    [
        "sionna.phy.fec.ldpc",
        "sionna.phy.fec.ldpc.encoding",
        "sionna.fec.ldpc",
        "sionna.fec.ldpc.encoding",
    ],
)
TDL, _TDL_IMPORT_DETAIL = _import_symbol_candidates(
    "TDL",
    [
        "sionna.phy.channel.tr38901",
        "sionna.phy.channel.tr38901.tdl",
        "sionna.channel.tr38901",
        "sionna.channel.tr38901.tdl",
    ],
)
SIONNA_LDPC_AVAILABLE = LDPC5GEncoder is not None
SIONNA_TDL_AVAILABLE = TDL is not None
SIONNA_AVAILABLE = SIONNA_LDPC_AVAILABLE and SIONNA_TDL_AVAILABLE
_sionna_missing = []
if not SIONNA_LDPC_AVAILABLE:
    _sionna_missing.append(f"LDPC5GEncoder import failed: {_LDPC5G_IMPORT_DETAIL}")
if not SIONNA_TDL_AVAILABLE:
    _sionna_missing.append(f"TDL import failed: {_TDL_IMPORT_DETAIL}")
_SIONNA_IMPORT_ERROR = None if not _sionna_missing else " ; ".join(_sionna_missing)


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    mod = getattr(type(x), "__module__", "") or ""
    if hasattr(x, "numpy"):
        try:
            return x.numpy()
        except Exception:
            pass
    if hasattr(x, "cpu") and hasattr(x, "numpy") and not mod.startswith("tensorflow"):
        try:
            return x.cpu().numpy()
        except Exception:
            pass
    return np.array(x)


# ------------------------- Config dataclasses -------------------------
@dataclass
class CodeConfig:
    code_name: str
    N: int
    K: int
    rate: float
    M: int = 0
    checks_to_vars: Optional[List[np.ndarray]] = field(default=None, repr=False)
    vars_to_checks: Optional[List[np.ndarray]] = field(default=None, repr=False)
    var_to_checks_edge_pos: Optional[List[np.ndarray]] = field(default=None, repr=False)


@dataclass
class DecoderConfig:
    max_iters: int
    alpha: float = 0.8
    early_stop: bool = True


@dataclass
class MCConfig:
    target_frame_errors: int = 150
    max_frames: int = 120000
    min_frames: int = 0


@dataclass
class CalibrationConfig:
    target_failed_frames: int = 384
    max_frames: int = 150000
    neg_ratio: float = 6.0
    hard_negative_cap: int = 128
    teacher_hidden: int = 64
    teacher_epochs: int = 24
    student_epochs: int = 20
    batch_size: int = 4096
    teacher_lr: float = 0.01
    student_lr: float = 0.022
    temperature: float = 2.5
    calib_tx_info_pool: int = 26
    calib_punctured_info_pool: int = 14
    calib_tx_parity_pool: int = 12
    calib_weight_cap: int = 10
    calib_free_dim_cap: int = 12
    calib_max_candidates: int = 4096

@dataclass
class HybridConfig:
    tx_info_pool: int = 24
    punctured_info_pool: int = 16
    tx_parity_pool: int = 12
    round2_extra_info: int = 10
    round2_extra_punctured: int = 8
    round2_extra_parity: int = 6
    max_weight: int = 18
    free_dim_cap: int = 18
    max_patterns: int = 200000
    frame_gate_threshold: float = 0.0
    gate_max_synd: int = 999
    gate_max_hard_ones: int = 999
    ai_weight: float = 1.50
    gain_weight: float = 0.24
    osc_weight: float = 0.18
    llr_weight: float = 1.0
    sw_weight: float = 1.20
    info_bonus: float = 0.70
    punctured_info_bonus: float = 0.40
    tx_bonus: float = 0.16
    parity_penalty: float = 0.15
    block_prob_weight: float = 0.95
    block_mass_weight: float = 0.55
    block_combo_max: int = 3
    top_blocks: int = 8
    block_beam_width: int = 48
    prefix_keep: int = 6
    block_refine_bits: int = 12
    global_top_bits: int = 20
    direct_top_bits: int = 6
    exact_pool_cap: int = 56
    block_mask_variants: int = 6
    traj_depth: int = 5
    traj_try_top: int = 2
    traj_info_weight: float = 2.20
    traj_punctured_weight: float = 1.35
    traj_parity_weight: float = 0.45
    traj_synd_weight: float = 0.10
    traj_best_scale: float = 1.80
    traj_second_scale: float = 1.25
    single_pool: int = 56
    pair_top_blocks: int = 6
    queue_cap: int = 4096
    expand_top_k: int = 24
    max_atoms_per_pattern: int = 4
    max_support_bits: int = 128
    block_prefix_sizes: List[int] = field(default_factory=lambda: [2, 4, 8, 12, 16, 24, 32])
    pair_prefix_sizes: List[int] = field(default_factory=lambda: [4, 8, 12, 16])
    syndrome_weight: float = 1.60
    atom_bonus_weight: float = 0.95
    pair_bonus_weight: float = 0.80
    size_prior_weight: float = 0.45
    support_penalty: float = 0.035
    atom_count_penalty: float = 0.12
    overlap_penalty: float = 0.18

@dataclass
class RunConfig:
    results_dir: str
    stage1_iters: int
    k_info: int
    n_tx: int
    qm: int
    alpha: float
    eval_snr_db: List[float]
    calib_snr_db: List[float]
    mc: MCConfig
    calib: CalibrationConfig
    hybrid: HybridConfig
    base_seed: int


# ------------------------- Small env helpers -------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _env_csv_floats(name: str, default: str) -> List[float]:
    raw = os.environ.get(name, default)
    out: List[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return out


def _stable_seed(*parts: Any) -> int:
    s = "|".join(str(x) for x in parts)
    h = 2166136261
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _now_tag() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")


# ------------------------- Sionna helpers -------------------------
def _pcm_to_tanner_neighborhoods(pcm) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    try:
        import scipy.sparse as sp  # type: ignore
    except Exception:
        sp = None  # type: ignore

    if sp is not None and sp.issparse(pcm):
        pcm_csr = pcm.tocsr()
        M, N = pcm_csr.shape
        checks_to_vars: List[np.ndarray] = []
        for c in range(M):
            s = pcm_csr.indptr[c]
            e = pcm_csr.indptr[c + 1]
            checks_to_vars.append(pcm_csr.indices[s:e].astype(np.int32, copy=False))
    else:
        pcm_dense = np.asarray(_as_numpy(pcm))
        if pcm_dense.ndim != 2:
            raise ValueError(f"PCM must be 2D, got shape {pcm_dense.shape}")
        M, N = pcm_dense.shape
        checks_to_vars = [np.flatnonzero(pcm_dense[c]).astype(np.int32) for c in range(M)]

    vars_to_checks_lists: List[List[int]] = [[] for _ in range(N)]
    for c, v_arr in enumerate(checks_to_vars):
        for v in v_arr:
            vars_to_checks_lists[int(v)].append(int(c))
    vars_to_checks: List[np.ndarray] = [np.asarray(lst, dtype=np.int32) for lst in vars_to_checks_lists]

    var_to_checks_edge_pos: List[np.ndarray] = []
    for v in range(N):
        cn_list = vars_to_checks[v]
        pos = np.empty(cn_list.shape[0], dtype=np.int32)
        for i, c in enumerate(cn_list):
            loc = np.where(checks_to_vars[int(c)] == v)[0]
            pos[i] = int(loc[0])
        var_to_checks_edge_pos.append(pos)
    return checks_to_vars, vars_to_checks, var_to_checks_edge_pos


def _sionna5g_internal_tx_positions(code_cfg: CodeConfig) -> np.ndarray:
    if not hasattr(code_cfg, "sionna"):
        raise ValueError("code_cfg has no .sionna metadata")
    s = code_cfg.sionna
    z = int(s.get("z", 0))
    n_tx = int(s.get("n_tx"))
    k_filler = int(s.get("k_filler", 0))
    k_info = int(code_cfg.K)
    N_int = int(code_cfg.N)
    L_pre = N_int - k_filler
    start = 2 * z
    stop = start + n_tx
    if stop > L_pre:
        raise ValueError(
            f"Invalid 5G mapping: need stop={stop} <= L_pre={L_pre}. "
            f"(N_int={N_int}, k_filler={k_filler}, z={z}, n_tx={n_tx})"
        )
    pos = np.arange(start, stop, dtype=np.int32)
    if k_filler > 0:
        pos = pos + (pos >= k_info).astype(np.int32) * k_filler
    return pos


def _sionna5g_tx_llr_to_internal_llr(llr_tx: np.ndarray, code_cfg: CodeConfig, llr_max: float = 50.0) -> np.ndarray:
    if not hasattr(code_cfg, "sionna"):
        raise ValueError("code_cfg has no .sionna metadata")
    s = code_cfg.sionna
    z = int(s.get("z", 0))
    n_tx = int(s.get("n_tx"))
    k_filler = int(s.get("k_filler", 0))
    out_int_inv = s.get("out_int_inv", None)
    k_info = int(code_cfg.K)
    N_int = int(code_cfg.N)

    llr_tx = np.asarray(llr_tx, dtype=np.float32).reshape(-1)
    if llr_tx.size != n_tx:
        raise ValueError(f"llr_tx length mismatch: got {llr_tx.size}, expected {n_tx}")

    if out_int_inv is not None:
        llr_tx = llr_tx[np.asarray(out_int_inv, dtype=np.int32)]

    L_pre = N_int - k_filler
    tail_len = L_pre - (2 * z + n_tx)
    if tail_len < 0:
        raise ValueError(
            f"Invalid tail_len={tail_len}. (N_int={N_int}, k_filler={k_filler}, z={z}, n_tx={n_tx})"
        )

    llr_pre = np.concatenate(
        [
            np.zeros(2 * z, dtype=np.float32),
            llr_tx,
            np.zeros(tail_len, dtype=np.float32),
        ],
        axis=0,
    )
    if k_filler > 0:
        filler = np.full(k_filler, float(llr_max), dtype=np.float32)
        llr_int = np.concatenate([llr_pre[:k_info], filler, llr_pre[k_info:]], axis=0)
    else:
        llr_int = llr_pre
    if llr_int.size != N_int:
        raise RuntimeError(f"Internal LLR length mismatch: got {llr_int.size}, expected {N_int}")
    return llr_int


_TDL_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _get_cached_tdl_model() -> Any:
    if not SIONNA_TDL_AVAILABLE:
        raise RuntimeError(
            "Sionna TDL channel model not available. "
            f"Import detail: {_SIONNA_IMPORT_ERROR}"
        )
    model = os.getenv("SIONNA_TDL_MODEL", "C")
    delay_spread_s = float(os.getenv("SIONNA_TDL_DELAY_SPREAD_S", "3e-7"))
    carrier_frequency_hz = float(os.getenv("SIONNA_TDL_CARRIER_FREQUENCY_HZ", "3.5e9"))
    min_speed = float(os.getenv("SIONNA_TDL_MIN_SPEED", "5.0"))
    max_speed = float(os.getenv("SIONNA_TDL_MAX_SPEED", "20.0"))
    key = ("TDL", model, delay_spread_s, carrier_frequency_hz, min_speed, max_speed)
    if key in _TDL_CACHE:
        return _TDL_CACHE[key]
    tdl = TDL(
        model=model,
        delay_spread=delay_spread_s,
        carrier_frequency=carrier_frequency_hz,
        min_speed=min_speed,
        max_speed=max_speed,
    )
    _TDL_CACHE[key] = tdl
    return tdl


def sionna_tdl_ofdm_siso_bpsk(n_bits: int, snr_db: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float]:
    tdl = _get_cached_tdl_model()
    fft_size = int(os.getenv("SIONNA_OFDM_FFT_SIZE", "512"))
    scs_hz = float(os.getenv("SIONNA_OFDM_SUBCARRIER_SPACING_HZ", "15000"))
    if fft_size <= 0:
        raise ValueError("SIONNA_OFDM_FFT_SIZE must be > 0")
    n_ofdm = int(math.ceil(n_bits / fft_size))
    pad = n_ofdm * fft_size - n_bits
    x = np.ones(n_bits + pad, dtype=np.complex64).reshape(n_ofdm, fft_size)
    sampling_frequency = float(fft_size) * scs_hz
    try:
        from sionna.phy import config as sionna_config  # type: ignore
        sionna_config.seed = int(rng.integers(0, 2**31 - 1))
    except Exception:
        pass
    a, tau = tdl(batch_size=1, num_time_steps=n_ofdm, sampling_frequency=sampling_frequency)
    a = _as_numpy(a)
    tau = _as_numpy(tau)
    a_siso = a[0, 0, 0, 0, 0, :, :]
    a_siso = np.transpose(a_siso, (1, 0))
    tau_siso = tau[0, 0, 0, :]
    f = (np.arange(fft_size, dtype=np.float32) * scs_hz)[None, :]
    phase = np.exp(-1j * 2.0 * np.pi * tau_siso[:, None] * f)
    h = (a_siso @ phase).astype(np.complex64)
    snr_lin = 10.0 ** (snr_db / 10.0)
    no = 1.0 / snr_lin
    w = (
        rng.standard_normal((n_ofdm, fft_size)).astype(np.float32)
        + 1j * rng.standard_normal((n_ofdm, fft_size)).astype(np.float32)
    ) * np.sqrt(no / 2.0)
    y = h * x + w
    cfo_hz = float(os.getenv("SIONNA_CFO_HZ", "0.0"))
    if cfo_hz != 0.0:
        t_sym = 1.0 / scs_hz
        rot = np.exp(1j * 2.0 * np.pi * cfo_hz * t_sym * np.arange(n_ofdm, dtype=np.float32))
        y = (rot[:, None] * y).astype(np.complex64)
    y_vec = y.reshape(-1)[:n_bits]
    h_vec = h.reshape(-1)[:n_bits]
    return y_vec, h_vec, float(no)


def _llr_bpsk_known_h(y: np.ndarray, h: np.ndarray, no: float) -> np.ndarray:
    return (4.0 / float(no)) * np.real(np.conj(h) * y).astype(np.float32)


def _estimate_h_for_llr(y: np.ndarray, h_true: np.ndarray, no: float) -> Tuple[np.ndarray, str]:
    mode = str(os.getenv("SIONNA_CSI_MODE", "nr_imperfect") or "nr_imperfect").strip().lower()
    h_true = np.asarray(h_true, dtype=np.complex64)
    y = np.asarray(y, dtype=np.complex64)
    if mode in ("", "perfect", "ideal", "known_h", "true"):
        return h_true, "perfect"
    n = int(y.size)
    if n <= 0:
        return h_true, "perfect"
    stride = max(1, int(float(os.getenv("SIONNA_CSI_PILOT_STRIDE", os.getenv("SIONNA_CSI_BLOCK_SC", "12")))))
    smooth = max(1, int(float(os.getenv("SIONNA_CSI_SMOOTH_PILOTS", "1"))))
    pilot_idx = np.arange(0, n, stride, dtype=np.int32)
    if pilot_idx.size == 0:
        pilot_idx = np.array([0], dtype=np.int32)
    h_p = y[pilot_idx].astype(np.complex64, copy=True)
    est_snr_db_str = str(os.getenv("SIONNA_CSI_EST_SNR_DB", "")).strip()
    if est_snr_db_str:
        try:
            est_snr_db = float(est_snr_db_str)
            est_var = 10.0 ** (-est_snr_db / 10.0)
            rng = np.random.default_rng(int(np.abs(np.sum(np.real(y[:min(32, n)]))) * 1e6) % (2**32 - 1))
            z = (
                rng.standard_normal(h_p.shape).astype(np.float32)
                + 1j * rng.standard_normal(h_p.shape).astype(np.float32)
            ) * np.sqrt(est_var / 2.0)
            h_p = (h_p + z.astype(np.complex64)).astype(np.complex64)
        except Exception:
            pass
    if smooth > 1 and h_p.size > 1:
        win = min(int(smooth), int(h_p.size))
        kernel = np.ones(win, dtype=np.float32) / float(win)
        h_p = (
            np.convolve(h_p.real, kernel, mode="same")
            + 1j * np.convolve(h_p.imag, kernel, mode="same")
        ).astype(np.complex64)
    xi = np.arange(n, dtype=np.float32)
    h_hat = (
        np.interp(xi, pilot_idx.astype(np.float32), h_p.real.astype(np.float32))
        + 1j * np.interp(xi, pilot_idx.astype(np.float32), h_p.imag.astype(np.float32))
    ).astype(np.complex64)
    if mode in ("nr_imperfect", "block_ls_drift", "imperfect_nr", "block_ls"):
        try:
            fft_size = max(1, int(os.getenv("SIONNA_OFDM_FFT_SIZE", "512")))
            scs_hz = float(os.getenv("SIONNA_OFDM_SUBCARRIER_SPACING_HZ", "15000"))
            n_sym = int(math.ceil(n / fft_size))
            h_grid = np.zeros((n_sym * fft_size,), dtype=np.complex64)
            h_grid[:n] = h_hat
            h_grid = h_grid.reshape(n_sym, fft_size)
            phase_step_std = np.deg2rad(float(os.getenv("SIONNA_CSI_PHASE_DRIFT_STD_DEG", "1.5")))
            amp_ripple_db = float(os.getenv("SIONNA_CSI_AMP_RIPPLE_DB", "0.6"))
            delay_bias_ns = float(os.getenv("SIONNA_CSI_DELAY_BIAS_NS", "8.0"))
            block_sc = max(1, int(os.getenv("SIONNA_CSI_BLOCK_SC", str(stride))))
            amp_blocks = int(math.ceil(fft_size / block_sc))
            sub_idx = np.arange(fft_size, dtype=np.float32)
            seed_real = float(np.abs(np.sum(np.real(y[:min(64, n)]))))
            seed_imag = float(np.abs(np.sum(np.imag(y[:min(64, n)]))))
            seed = int((seed_real * 1e6 + 17.0 * seed_imag * 1e6)) % (2**32 - 1)
            rng = np.random.default_rng(seed)
            phase_state = 0.0
            for t in range(n_sym):
                phase_state += float(rng.normal(0.0, phase_step_std))
                delay_bias = float(delay_bias_ns) * 1e-9 * float(rng.normal(1.0, 0.20))
                slope = (-2.0 * np.pi * delay_bias * scs_hz * sub_idx).astype(np.float32, copy=False)
                amp_db = rng.normal(0.0, amp_ripple_db, size=amp_blocks).astype(np.float32)
                amp = np.repeat((10.0 ** (amp_db / 20.0)).astype(np.float32), block_sc)[:fft_size]
                rot = np.exp(1j * (phase_state + slope)).astype(np.complex64)
                h_grid[t, :] = (h_grid[t, :] * amp.astype(np.complex64) * rot).astype(np.complex64)
            h_hat = h_grid.reshape(-1)[:n].astype(np.complex64, copy=False)
        except Exception:
            pass
        return h_hat, "nr_imperfect"
    return h_true, "perfect"


def build_sionna_5g_nr_code_cfg(k_info: int, n_tx: int, num_bits_per_symbol: int = 1) -> CodeConfig:
    if not SIONNA_LDPC_AVAILABLE:
        raise RuntimeError(
            "Sionna 5G LDPC encoder not available. Install a compatible Sionna runtime. "
            f"Import detail: {_SIONNA_IMPORT_ERROR}"
        )
    qm = int(num_bits_per_symbol)
    enc = LDPC5GEncoder(k=int(k_info), n=int(n_tx), num_bits_per_symbol=qm)
    pcm = enc.pcm
    checks_to_vars, vars_to_checks, var_to_checks_edge_pos = _pcm_to_tanner_neighborhoods(pcm)
    M, N = pcm.shape
    rate_eff = float(k_info) / float(n_tx)
    code_name = f"sionna5g_k{k_info}_n{n_tx}_qm{qm}"
    code_cfg = CodeConfig(code_name=code_name, N=int(N), K=int(k_info), rate=rate_eff)
    code_cfg.M = int(M)
    code_cfg.checks_to_vars = checks_to_vars
    code_cfg.vars_to_checks = vars_to_checks
    code_cfg.var_to_checks_edge_pos = var_to_checks_edge_pos
    out_int_inv = None
    if getattr(enc, "out_int_inv", None) is not None:
        try:
            out_int_inv = np.array(_as_numpy(enc.out_int_inv), dtype=np.int32)
        except Exception:
            out_int_inv = np.array(enc.out_int_inv, dtype=np.int32)
    code_cfg.sionna = {
        "k_info": int(k_info),
        "n_tx": int(n_tx),
        "qm": qm,
        "z": int(getattr(enc, "z", 0)),
        "k_filler": int(getattr(enc, "k_filler", 0)),
        "out_int_inv": out_int_inv,
    }
    code_cfg.sionna["tx_pos"] = _sionna5g_internal_tx_positions(code_cfg)
    tx_mask = np.zeros(code_cfg.N, dtype=np.uint8)
    tx_mask[np.asarray(code_cfg.sionna["tx_pos"], dtype=np.int32)] = 1
    info_mask = np.zeros(code_cfg.N, dtype=np.uint8)
    info_mask[: code_cfg.K] = 1
    filler_mask = np.zeros(code_cfg.N, dtype=np.uint8)
    k_filler = int(code_cfg.sionna.get("k_filler", 0))
    if k_filler > 0:
        filler_mask[code_cfg.K : code_cfg.K + k_filler] = 1
    tx_info_mask = (tx_mask & info_mask).astype(np.uint8)
    tx_parity_mask = (tx_mask & (1 - info_mask) & (1 - filler_mask)).astype(np.uint8)
    latent_mask = ((1 - tx_mask) & (1 - filler_mask)).astype(np.uint8)
    punctured_info_mask = ((1 - tx_mask) & info_mask).astype(np.uint8)
    code_cfg._tx_mask = tx_mask
    code_cfg._info_mask = info_mask
    code_cfg._tx_info_mask = tx_info_mask
    code_cfg._tx_parity_mask = tx_parity_mask
    code_cfg._latent_mask = latent_mask
    code_cfg._punctured_info_mask = punctured_info_mask
    code_cfg._filler_mask = filler_mask
    z = max(1, int(code_cfg.sionna.get("z", 1)))
    code_cfg._qc_z = int(z)
    code_cfg._block_ids = (np.arange(code_cfg.N, dtype=np.int32) // int(z)).astype(np.int32)
    code_cfg._num_blocks = int(code_cfg._block_ids.max()) + 1 if code_cfg.N > 0 else 0
    prepare_code_for_fast_decoding(code_cfg)
    _prepare_static_graph_features(code_cfg)
    return code_cfg


# ------------------------- Fast Tanner structures -------------------------
def prepare_flat_adjacency(code_cfg: CodeConfig):
    M = code_cfg.M
    checks_to_vars = code_cfg.checks_to_vars
    total_edges = sum(len(cv) for cv in checks_to_vars)
    check_ptrs = np.zeros(M + 1, dtype=np.int64)
    check_indices = np.zeros(total_edges, dtype=np.int64)
    ptr = 0
    for j in range(M):
        check_ptrs[j] = ptr
        cv = checks_to_vars[j]
        check_indices[ptr:ptr + len(cv)] = cv
        ptr += len(cv)
    check_ptrs[M] = ptr
    return check_ptrs, check_indices


def prepare_code_for_fast_decoding(code_cfg: CodeConfig) -> None:
    M = code_cfg.M
    N = code_cfg.N
    checks_to_vars = code_cfg.checks_to_vars
    vars_to_checks = code_cfg.vars_to_checks
    var_to_checks_edge_pos = code_cfg.var_to_checks_edge_pos

    check_ptrs, check_indices = prepare_flat_adjacency(code_cfg)
    code_cfg._check_ptrs = check_ptrs
    code_cfg._check_indices = check_indices

    total_c2v = sum(len(cv) for cv in checks_to_vars)
    c2v_ptrs = np.zeros(M + 1, dtype=np.int32)
    c2v_indices = np.zeros(total_c2v, dtype=np.int32)
    ptr = 0
    for j in range(M):
        c2v_ptrs[j] = ptr
        cv = checks_to_vars[j]
        c2v_indices[ptr:ptr + len(cv)] = cv
        ptr += len(cv)
    c2v_ptrs[M] = ptr

    total_v2c = sum(len(vc) for vc in vars_to_checks)
    v2c_ptrs = np.zeros(N + 1, dtype=np.int32)
    v2c_checks = np.zeros(total_v2c, dtype=np.int32)
    v2c_edge_pos = np.zeros(total_v2c, dtype=np.int32)
    ptr = 0
    for v in range(N):
        v2c_ptrs[v] = ptr
        vc = vars_to_checks[v]
        ep = var_to_checks_edge_pos[v]
        v2c_checks[ptr:ptr + len(vc)] = vc
        v2c_edge_pos[ptr:ptr + len(ep)] = ep
        ptr += len(vc)
    v2c_ptrs[N] = ptr

    check_degree = np.array([len(cv) for cv in checks_to_vars], dtype=np.int32)
    bit_degree = np.array([len(vc) for vc in vars_to_checks], dtype=np.int32)

    code_cfg._c2v_ptrs = c2v_ptrs
    code_cfg._c2v_indices = c2v_indices
    code_cfg._v2c_ptrs = v2c_ptrs
    code_cfg._v2c_checks = v2c_checks
    code_cfg._v2c_edge_pos = v2c_edge_pos
    code_cfg._check_degree = check_degree
    code_cfg._bit_degree = bit_degree
    code_cfg._mean_check_degree = float(check_degree.mean()) if check_degree.size else 0.0


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=False)
    def _compute_syndrome_numba(bits, check_ptrs, check_indices, M):
        s = np.zeros(M, dtype=np.uint8)
        for j in prange(M):
            start = check_ptrs[j]
            end = check_ptrs[j + 1]
            parity = 0
            for idx in range(start, end):
                parity ^= bits[check_indices[idx]]
            s[j] = parity
        return s

    @njit(parallel=True, cache=False)
    def _check_node_update_numba(msg_v2c_flat, msg_c2v_flat, c2v_ptrs, M, alpha):
        for j in prange(M):
            start = c2v_ptrs[j]
            end = c2v_ptrs[j + 1]
            d = end - start
            if d == 0:
                continue
            sign_all = 1.0
            min1 = 1e30
            min2 = 1e30
            idx_min1 = 0
            for e in range(d):
                msg = msg_v2c_flat[start + e]
                if msg < 0.0:
                    sign_all *= -1.0
                    abs_val = -msg
                else:
                    abs_val = msg if msg > 0.0 else 0.0
                if abs_val < min1:
                    min2 = min1
                    min1 = abs_val
                    idx_min1 = e
                elif abs_val < min2:
                    min2 = abs_val
            if d == 1:
                min2 = min1
            for e in range(d):
                msg = msg_v2c_flat[start + e]
                sign_e = -sign_all if msg < 0.0 else sign_all
                mag_e = min2 if e == idx_min1 else min1
                msg_c2v_flat[start + e] = alpha * sign_e * mag_e

    @njit(parallel=True, cache=False)
    def _variable_node_update_numba(llr_channel, msg_c2v_flat, v2c_ptrs, v2c_checks, v2c_edge_pos, c2v_ptrs, N):
        llr_posterior = np.empty_like(llr_channel)
        for v in prange(N):
            start = v2c_ptrs[v]
            end = v2c_ptrs[v + 1]
            total = 0.0
            for idx in range(start, end):
                j = v2c_checks[idx]
                e = v2c_edge_pos[idx]
                total += msg_c2v_flat[c2v_ptrs[j] + e]
            llr_posterior[v] = llr_channel[v] + total
        return llr_posterior

    @njit(parallel=True, cache=False)
    def _vn_to_cn_update_numba(llr_posterior, msg_c2v_flat, msg_v2c_flat, v2c_ptrs, v2c_checks, v2c_edge_pos, c2v_ptrs, N):
        for v in prange(N):
            start = v2c_ptrs[v]
            end = v2c_ptrs[v + 1]
            L_v = llr_posterior[v]
            for idx in range(start, end):
                j = v2c_checks[idx]
                e = v2c_edge_pos[idx]
                base = c2v_ptrs[j] + e
                msg_v2c_flat[base] = L_v - msg_c2v_flat[base]
else:
    def _compute_syndrome_numba(bits, check_ptrs, check_indices, M):
        s = np.zeros(int(M), dtype=np.uint8)
        for j in range(int(M)):
            start = int(check_ptrs[j])
            end = int(check_ptrs[j + 1])
            parity = 0
            for idx in range(start, end):
                parity ^= int(bits[int(check_indices[idx])])
            s[j] = parity
        return s

    def _check_node_update_numba(msg_v2c_flat, msg_c2v_flat, c2v_ptrs, M, alpha):
        for j in range(int(M)):
            start = int(c2v_ptrs[j])
            end = int(c2v_ptrs[j + 1])
            d = end - start
            if d <= 0:
                continue
            sign_all = 1.0
            min1 = 1e30
            min2 = 1e30
            idx_min1 = 0
            for e in range(d):
                msg = float(msg_v2c_flat[start + e])
                if msg < 0.0:
                    sign_all *= -1.0
                    abs_val = -msg
                else:
                    abs_val = msg if msg > 0.0 else 0.0
                if abs_val < min1:
                    min2 = min1
                    min1 = abs_val
                    idx_min1 = e
                elif abs_val < min2:
                    min2 = abs_val
            if d == 1:
                min2 = min1
            for e in range(d):
                msg = float(msg_v2c_flat[start + e])
                sign_e = -sign_all if msg < 0.0 else sign_all
                mag_e = min2 if e == idx_min1 else min1
                msg_c2v_flat[start + e] = float(alpha) * sign_e * mag_e

    def _variable_node_update_numba(llr_channel, msg_c2v_flat, v2c_ptrs, v2c_checks, v2c_edge_pos, c2v_ptrs, N):
        llr_posterior = np.empty_like(llr_channel)
        for v in range(int(N)):
            start = int(v2c_ptrs[v])
            end = int(v2c_ptrs[v + 1])
            total = 0.0
            for idx in range(start, end):
                j = int(v2c_checks[idx])
                e = int(v2c_edge_pos[idx])
                total += float(msg_c2v_flat[int(c2v_ptrs[j]) + e])
            llr_posterior[v] = float(llr_channel[v]) + total
        return llr_posterior

    def _vn_to_cn_update_numba(llr_posterior, msg_c2v_flat, msg_v2c_flat, v2c_ptrs, v2c_checks, v2c_edge_pos, c2v_ptrs, N):
        for v in range(int(N)):
            start = int(v2c_ptrs[v])
            end = int(v2c_ptrs[v + 1])
            L_v = float(llr_posterior[v])
            for idx in range(start, end):
                j = int(v2c_checks[idx])
                e = int(v2c_edge_pos[idx])
                base = int(c2v_ptrs[j]) + e
                msg_v2c_flat[base] = L_v - float(msg_c2v_flat[base])


def compute_syndrome(bits: np.ndarray, code_cfg: CodeConfig) -> np.ndarray:
    if NUMBA_AVAILABLE:
        return _compute_syndrome_numba(bits.astype(np.uint8), code_cfg._check_ptrs, code_cfg._check_indices, code_cfg.M)
    s = np.zeros(code_cfg.M, dtype=np.uint8)
    for j, vs in enumerate(code_cfg.checks_to_vars):
        parity = 0
        for v in vs:
            parity ^= int(bits[int(v)])
        s[j] = parity
    return s


def ldpc_min_sum_decode(llr_channel: np.ndarray, code_cfg: CodeConfig, dec_cfg: DecoderConfig, snapshot_iters: Optional[List[int]] = None):
    N = code_cfg.N
    M = code_cfg.M
    c2v_ptrs = code_cfg._c2v_ptrs
    c2v_indices = code_cfg._c2v_indices
    v2c_ptrs = code_cfg._v2c_ptrs
    v2c_checks = code_cfg._v2c_checks
    v2c_edge_pos = code_cfg._v2c_edge_pos
    total_edges = c2v_ptrs[M]
    msg_v2c_flat = np.zeros(total_edges, dtype=np.float64)
    msg_c2v_flat = np.zeros(total_edges, dtype=np.float64)
    for j in range(M):
        start = c2v_ptrs[j]
        end = c2v_ptrs[j + 1]
        for idx in range(start, end):
            v = c2v_indices[idx]
            msg_v2c_flat[idx] = llr_channel[v]
    iter_used = dec_cfg.max_iters
    snapshot_set = set(snapshot_iters) if snapshot_iters else set()
    snapshots: Dict[str, Dict[int, np.ndarray]] = {"llr": {}, "hard_bits": {}, "syndrome": {}}
    llr_posterior = llr_channel.copy()
    for it in range(1, dec_cfg.max_iters + 1):
        _check_node_update_numba(msg_v2c_flat, msg_c2v_flat, c2v_ptrs, M, dec_cfg.alpha)
        llr_posterior = _variable_node_update_numba(
            llr_channel, msg_c2v_flat, v2c_ptrs, v2c_checks, v2c_edge_pos, c2v_ptrs, N
        )
        hard_bits = (llr_posterior < 0.0).astype(np.uint8)
        syndrome = _compute_syndrome_numba(hard_bits, code_cfg._check_ptrs, code_cfg._check_indices, M)
        if it in snapshot_set:
            snapshots["llr"][it] = llr_posterior.copy()
            snapshots["hard_bits"][it] = hard_bits.copy()
            snapshots["syndrome"][it] = syndrome.copy()
        if dec_cfg.early_stop and int(syndrome.sum()) == 0:
            iter_used = it
            break
        _vn_to_cn_update_numba(
            llr_posterior, msg_c2v_flat, msg_v2c_flat, v2c_ptrs, v2c_checks, v2c_edge_pos, c2v_ptrs, N
        )
        iter_used = it
    hard_bits = (llr_posterior < 0.0).astype(np.uint8)
    syndrome = _compute_syndrome_numba(hard_bits, code_cfg._check_ptrs, code_cfg._check_indices, M)
    return hard_bits, llr_posterior.astype(np.float32), syndrome.astype(np.uint8), int(iter_used), snapshots


# ------------------------- Static graph features -------------------------
def _prepare_static_graph_features(code_cfg: CodeConfig) -> None:
    N = code_cfg.N
    checks_to_vars = code_cfg.checks_to_vars
    vars_to_checks = code_cfg.vars_to_checks
    twohop = np.zeros(N, dtype=np.float32)
    cycle4 = np.zeros(N, dtype=np.float32)
    for v in range(N):
        seen: Dict[int, int] = {}
        for c in vars_to_checks[v]:
            for u in checks_to_vars[int(c)]:
                u = int(u)
                if u == v:
                    continue
                seen[u] = seen.get(u, 0) + 1
        twohop[v] = float(len(seen))
        cyc = 0.0
        for cnt in seen.values():
            if cnt > 1:
                cyc += float(cnt - 1)
        cycle4[v] = cyc
    twohop_norm = twohop / max(1.0, float(twohop.max()))
    cycle4_norm = cycle4 / max(1.0, float(cycle4.max()))
    code_cfg._twohop_norm = twohop_norm.astype(np.float32)
    code_cfg._cycle4_norm = cycle4_norm.astype(np.float32)


# ------------------------- Frame generation -------------------------
def generate_frame_llr(code_cfg: CodeConfig, snr_db: float, seed: int) -> np.ndarray:
    n_tx = int(code_cfg.sionna["n_tx"])
    rng = np.random.default_rng(int(seed))
    y_c, h_c, no = sionna_tdl_ofdm_siso_bpsk(n_tx, snr_db, rng)
    h_llr, _ = _estimate_h_for_llr(y_c, h_c, no)
    llr_tx = _llr_bpsk_known_h(y_c, h_llr, no)
    llr_max = float(os.getenv("SIONNA_LLR_MAX", "50.0"))
    llr_int = _sionna5g_tx_llr_to_internal_llr(llr_tx, code_cfg, llr_max=llr_max)
    return llr_int.astype(np.float32)


# ------------------------- AI model -------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


class TinyTeacherMLP:
    def __init__(self, d_in: int, hidden: int, seed: int):
        rng = np.random.default_rng(int(seed))
        self.W1 = (0.10 * rng.standard_normal((d_in, hidden))).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (0.10 * rng.standard_normal((hidden, 1))).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

    def logits(self, X: np.ndarray) -> np.ndarray:
        H = np.maximum(X @ self.W1 + self.b1[None, :], 0.0)
        Z = H @ self.W2 + self.b2[None, :]
        return Z[:, 0].astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, epochs: int, batch_size: int, lr: float, seed: int):
        rng = np.random.default_rng(int(seed))
        n = X.shape[0]
        if n == 0:
            return
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mW1 = np.zeros_like(self.W1)
        mb1 = np.zeros_like(self.b1)
        mW2 = np.zeros_like(self.W2)
        mb2 = np.zeros_like(self.b2)
        vW1 = np.zeros_like(self.W1)
        vb1 = np.zeros_like(self.b1)
        vW2 = np.zeros_like(self.W2)
        vb2 = np.zeros_like(self.b2)
        t = 0
        for _ in range(int(epochs)):
            order = rng.permutation(n)
            for start in range(0, n, int(batch_size)):
                idx = order[start:start + int(batch_size)]
                xb = X[idx]
                yb = y[idx]
                wb = sample_weight[idx]
                z1 = xb @ self.W1 + self.b1[None, :]
                h = np.maximum(z1, 0.0)
                logits = (h @ self.W2 + self.b2[None, :])[:, 0]
                p = _sigmoid(logits)
                grad_logits = (p - yb) * wb
                grad_logits /= max(1.0, float(wb.sum()))
                gW2 = h.T @ grad_logits[:, None]
                gb2 = np.array([grad_logits.sum()], dtype=np.float32)
                gh = grad_logits[:, None] @ self.W2.T
                gz1 = gh * (z1 > 0.0)
                gW1 = xb.T @ gz1
                gb1 = gz1.sum(axis=0)
                t += 1
                for P, G, M, V in (
                    (self.W1, gW1, mW1, vW1),
                    (self.b1, gb1, mb1, vb1),
                    (self.W2, gW2, mW2, vW2),
                    (self.b2, gb2, mb2, vb2),
                ):
                    M *= beta1
                    M += (1.0 - beta1) * G
                    V *= beta2
                    V += (1.0 - beta2) * (G * G)
                    mhat = M / (1.0 - beta1 ** t)
                    vhat = V / (1.0 - beta2 ** t)
                    P -= float(lr) * mhat / (np.sqrt(vhat) + eps)


class DistilledLinearStudent:
    def __init__(self, d_in: int):
        self.w = np.zeros(d_in, dtype=np.float32)
        self.b = 0.0

    def logits(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.w + self.b).astype(np.float32)

    def prob(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.logits(X))

    def fit(self, X: np.ndarray, soft_target: np.ndarray, hard_target: np.ndarray, sample_weight: np.ndarray, epochs: int, batch_size: int, lr: float, seed: int):
        rng = np.random.default_rng(int(seed))
        n = X.shape[0]
        if n == 0:
            return
        target = 0.65 * soft_target + 0.35 * hard_target
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mw = np.zeros_like(self.w)
        vw = np.zeros_like(self.w)
        mb = 0.0
        vb = 0.0
        t = 0
        for _ in range(int(epochs)):
            order = rng.permutation(n)
            for start in range(0, n, int(batch_size)):
                idx = order[start:start + int(batch_size)]
                xb = X[idx]
                yb = target[idx]
                wb = sample_weight[idx]
                logits = xb @ self.w + self.b
                p = _sigmoid(logits)
                grad_logits = (p - yb) * wb
                grad_logits /= max(1.0, float(wb.sum()))
                gw = xb.T @ grad_logits
                gb = float(grad_logits.sum())
                t += 1
                mw = beta1 * mw + (1.0 - beta1) * gw
                vw = beta2 * vw + (1.0 - beta2) * (gw * gw)
                mwhat = mw / (1.0 - beta1 ** t)
                vwhat = vw / (1.0 - beta2 ** t)
                self.w -= float(lr) * mwhat / (np.sqrt(vwhat) + eps)
                mb = beta1 * mb + (1.0 - beta1) * gb
                vb = beta2 * vb + (1.0 - beta2) * (gb * gb)
                mbhat = mb / (1.0 - beta1 ** t)
                vbhat = vb / (1.0 - beta2 ** t)
                self.b -= float(lr) * mbhat / (math.sqrt(vbhat) + eps)

    def save(self, path: str, feature_names: List[str]):
        np.savez(path, w=self.w, b=np.array([self.b], dtype=np.float32), feature_names=np.array(feature_names, dtype=object))

    @staticmethod
    def load(path: str):
        payload = np.load(path, allow_pickle=False)
        obj = DistilledLinearStudent(int(payload["w"].shape[0]))
        obj.w = payload["w"].astype(np.float32)
        obj.b = float(payload["b"][0])
        feature_names = [str(x) for x in payload["feature_names"].tolist()]
        return obj, feature_names


FEATURE_NAMES = [
    "inv_abs_llr",
    "abs_llr_norm",
    "delta1_norm",
    "delta2_norm",
    "flip_rate",
    "hard_one",
    "unsat_frac",
    "gain_norm",
    "weighted_unsat",
    "local_weak_frac",
    "cycle4_norm",
    "twohop_norm",
    "global_synd_ratio",
    "global_mean_abs",
    "global_flip_ratio",
    "is_tx",
    "is_info",
    "is_tx_info",
    "is_punctured_info",
    "is_tx_parity",
    "is_latent",
]
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

def _snapshot_pick(snapshots: Dict[str, Dict[int, np.ndarray]], wanted: int, fallback: np.ndarray) -> np.ndarray:
    if wanted in snapshots["llr"]:
        return snapshots["llr"][wanted]
    keys = sorted(snapshots["llr"].keys())
    if keys:
        return snapshots["llr"][keys[-1]]
    return fallback


def _feature_pack(code_cfg: CodeConfig, llr_final: np.ndarray, hard_final: np.ndarray, syndrome_final: np.ndarray, snapshots: Dict[str, Dict[int, np.ndarray]], stage1_iters: int) -> np.ndarray:
    it_a = max(1, stage1_iters - 2)
    it_b = max(1, stage1_iters - 1)
    llr_a = snapshots["llr"].get(it_a, llr_final)
    llr_b = snapshots["llr"].get(it_b, llr_final)
    hb_a = snapshots["hard_bits"].get(it_a, hard_final)
    hb_b = snapshots["hard_bits"].get(it_b, hard_final)
    abs_final = np.abs(llr_final).astype(np.float32)
    mean_abs = float(abs_final.mean() + 1e-6)
    delta1 = np.abs(llr_final - llr_b).astype(np.float32)
    delta2 = np.abs(llr_b - llr_a).astype(np.float32)
    flip_rate = ((hb_a != hb_b).astype(np.float32) + (hb_b != hard_final).astype(np.float32)) * 0.5
    weak_thr = float(np.quantile(abs_final, 0.25)) if abs_final.size > 0 else 0.0
    weak_mask = abs_final <= weak_thr
    unsat_mask = syndrome_final.astype(bool)
    check_degree = code_cfg._check_degree
    bit_degree = np.maximum(1, code_cfg._bit_degree).astype(np.float32)
    weak_per_check = np.zeros(code_cfg.M, dtype=np.float32)
    for c in range(code_cfg.M):
        vs = code_cfg.checks_to_vars[c]
        weak_per_check[c] = float(np.sum(weak_mask[vs]))
    unsat_count = np.zeros(code_cfg.N, dtype=np.float32)
    weighted_unsat = np.zeros(code_cfg.N, dtype=np.float32)
    local_weak_frac = np.zeros(code_cfg.N, dtype=np.float32)
    for v in range(code_cfg.N):
        cn = code_cfg.vars_to_checks[v]
        if cn.size == 0:
            continue
        uc = 0.0
        wu = 0.0
        weak_num = 0.0
        weak_den = 0.0
        for c in cn:
            c = int(c)
            if unsat_mask[c]:
                uc += 1.0
                wu += 1.0 / float(max(1, check_degree[c]))
                weak_num += float(weak_per_check[c])
                weak_den += float(check_degree[c])
        unsat_count[v] = uc
        weighted_unsat[v] = wu
        if weak_den > 0.0:
            local_weak_frac[v] = weak_num / weak_den
    gain_norm = (2.0 * unsat_count - bit_degree) / bit_degree
    global_synd_ratio = float(np.mean(unsat_mask.astype(np.float32)))
    global_flip_ratio = float(np.mean(flip_rate))
    global_mean_abs = float(mean_abs / (1.0 + mean_abs))
    feat = np.stack(
        [
            1.0 / (1.0 + abs_final),
            abs_final / mean_abs,
            delta1 / mean_abs,
            delta2 / mean_abs,
            flip_rate,
            hard_final.astype(np.float32),
            unsat_count / bit_degree,
            gain_norm,
            weighted_unsat,
            local_weak_frac,
            code_cfg._cycle4_norm,
            code_cfg._twohop_norm,
            np.full(code_cfg.N, global_synd_ratio, dtype=np.float32),
            np.full(code_cfg.N, global_mean_abs, dtype=np.float32),
            np.full(code_cfg.N, global_flip_ratio, dtype=np.float32),
            code_cfg._tx_mask.astype(np.float32),
            code_cfg._info_mask.astype(np.float32),
            code_cfg._tx_info_mask.astype(np.float32),
            code_cfg._punctured_info_mask.astype(np.float32),
            code_cfg._tx_parity_mask.astype(np.float32),
            code_cfg._latent_mask.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return feat


def _snapshot_iter_list(stage1_iters: int, depth: int) -> List[int]:
    depth = max(1, int(depth))
    start = max(1, int(stage1_iters) - depth + 1)
    return list(range(start, int(stage1_iters) + 1))


def _snapshot_state(
    snapshots: Dict[str, Dict[int, np.ndarray]],
    target_it: int,
    fallback_llr: np.ndarray,
    fallback_hard: np.ndarray,
    fallback_syndrome: np.ndarray,
):
    llr_t = snapshots["llr"].get(int(target_it), fallback_llr)
    hard_t = snapshots["hard_bits"].get(int(target_it), fallback_hard)
    synd_t = snapshots["syndrome"].get(int(target_it), fallback_syndrome)
    return llr_t, hard_t, synd_t


def _feature_pack_for_target(
    code_cfg: CodeConfig,
    snapshots: Dict[str, Dict[int, np.ndarray]],
    target_it: int,
    fallback_llr: np.ndarray,
    fallback_hard: np.ndarray,
    fallback_syndrome: np.ndarray,
) -> np.ndarray:
    llr_t, hard_t, synd_t = _snapshot_state(snapshots, int(target_it), fallback_llr, fallback_hard, fallback_syndrome)
    prev1 = max(1, int(target_it) - 1)
    prev2 = max(1, int(target_it) - 2)
    llr_a = snapshots["llr"].get(prev2, llr_t)
    llr_b = snapshots["llr"].get(prev1, llr_t)
    hb_a = snapshots["hard_bits"].get(prev2, hard_t)
    hb_b = snapshots["hard_bits"].get(prev1, hard_t)
    abs_t = np.abs(llr_t).astype(np.float32)
    mean_abs = float(abs_t.mean() + 1e-6)
    delta1 = np.abs(llr_t - llr_b).astype(np.float32)
    delta2 = np.abs(llr_b - llr_a).astype(np.float32)
    flip_rate = ((hb_a != hb_b).astype(np.float32) + (hb_b != hard_t).astype(np.float32)) * 0.5
    weak_thr = float(np.quantile(abs_t, 0.25)) if abs_t.size > 0 else 0.0
    weak_mask = abs_t <= weak_thr
    unsat_mask = synd_t.astype(bool)
    check_degree = code_cfg._check_degree
    bit_degree = np.maximum(1, code_cfg._bit_degree).astype(np.float32)
    weak_per_check = np.zeros(code_cfg.M, dtype=np.float32)
    for c in range(code_cfg.M):
        vs = code_cfg.checks_to_vars[c]
        weak_per_check[c] = float(np.sum(weak_mask[vs]))
    unsat_count = np.zeros(code_cfg.N, dtype=np.float32)
    weighted_unsat = np.zeros(code_cfg.N, dtype=np.float32)
    local_weak_frac = np.zeros(code_cfg.N, dtype=np.float32)
    for v in range(code_cfg.N):
        cn = code_cfg.vars_to_checks[v]
        if cn.size == 0:
            continue
        uc = 0.0
        wu = 0.0
        weak_num = 0.0
        weak_den = 0.0
        for c in cn:
            c = int(c)
            if unsat_mask[c]:
                uc += 1.0
                wu += 1.0 / float(max(1, check_degree[c]))
                weak_num += float(weak_per_check[c])
                weak_den += float(check_degree[c])
        unsat_count[v] = uc
        weighted_unsat[v] = wu
        if weak_den > 0.0:
            local_weak_frac[v] = weak_num / weak_den
    gain_norm = (2.0 * unsat_count - bit_degree) / bit_degree
    global_synd_ratio = float(np.mean(unsat_mask.astype(np.float32)))
    global_flip_ratio = float(np.mean(flip_rate))
    global_mean_abs = float(mean_abs / (1.0 + mean_abs))
    feat = np.stack(
        [
            1.0 / (1.0 + abs_t),
            abs_t / mean_abs,
            delta1 / mean_abs,
            delta2 / mean_abs,
            flip_rate,
            hard_t.astype(np.float32),
            unsat_count / bit_degree,
            gain_norm,
            weighted_unsat,
            local_weak_frac,
            code_cfg._cycle4_norm,
            code_cfg._twohop_norm,
            np.full(code_cfg.N, global_synd_ratio, dtype=np.float32),
            np.full(code_cfg.N, global_mean_abs, dtype=np.float32),
            np.full(code_cfg.N, global_flip_ratio, dtype=np.float32),
            code_cfg._tx_mask.astype(np.float32),
            code_cfg._info_mask.astype(np.float32),
            code_cfg._tx_info_mask.astype(np.float32),
            code_cfg._punctured_info_mask.astype(np.float32),
            code_cfg._tx_parity_mask.astype(np.float32),
            code_cfg._latent_mask.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return feat


def _trajectory_oracle_cost(code_cfg: CodeConfig, hard_bits: np.ndarray, syndrome_bits: np.ndarray, cfg: HybridConfig) -> float:
    mask = hard_bits.astype(bool)
    info_cnt = float(np.sum(mask & code_cfg._tx_info_mask.astype(bool)))
    punct_cnt = float(np.sum(mask & code_cfg._punctured_info_mask.astype(bool)))
    parity_cnt = float(np.sum(mask & code_cfg._tx_parity_mask.astype(bool)))
    synd_cnt = float(np.sum(syndrome_bits))
    return (
        float(cfg.traj_info_weight) * info_cnt
        + float(cfg.traj_punctured_weight) * punct_cnt
        + float(cfg.traj_parity_weight) * parity_cnt
        + float(cfg.traj_synd_weight) * synd_cnt
    )


def _trajectory_proxy_score(frame_feat: np.ndarray, target_it: int, stage1_iters: int) -> float:
    iter_norm = float(target_it) / float(max(1, stage1_iters))
    score = (
        -3.0 * float(frame_feat[0])
        -4.2 * float(frame_feat[2])
        -2.0 * float(frame_feat[3])
        -1.0 * float(frame_feat[4])
        -0.5 * float(frame_feat[9])
        +1.2 * float(frame_feat[10])
        +0.8 * float(frame_feat[11])
        +0.5 * float(frame_feat[12])
        -0.3 * float(frame_feat[13])
        +0.10 * (1.0 - iter_norm)
    )
    return float(_sigmoid(np.array([score], dtype=np.float32))[0])


def _collect_training_rows(feat: np.ndarray, labels: np.ndarray, llr_final: np.ndarray, syndrome_final: np.ndarray, cfg: CalibrationConfig, seed: int):
    rng = np.random.default_rng(int(seed))
    pos = np.flatnonzero(labels > 0)
    if pos.size == 0:
        return None
    abs_llr = np.abs(llr_final)
    tx_info = feat[:, FEATURE_INDEX["is_tx_info"]]
    punctured_info = feat[:, FEATURE_INDEX["is_punctured_info"]]
    tx_parity = feat[:, FEATURE_INDEX["is_tx_parity"]]
    score = (
        (1.0 / (1.0 + abs_llr))
        + 0.80 * feat[:, FEATURE_INDEX["unsat_frac"]]
        + 0.60 * feat[:, FEATURE_INDEX["flip_rate"]]
        + 0.42 * tx_info
        + 0.30 * punctured_info
        - 0.08 * tx_parity
    )
    cand = np.argsort(-score)
    zero_mask = labels == 0
    hard_neg = cand[zero_mask[cand]] if cand.size else np.array([], dtype=np.int64)
    hard_cap = int(max(8, cfg.hard_negative_cap))
    target_neg = int(max(len(pos), round(cfg.neg_ratio * len(pos))))
    hard_take = min(hard_cap, target_neg, hard_neg.size)
    picked_hard = hard_neg[:hard_take]
    remaining_neg = target_neg - picked_hard.size
    zeros = np.flatnonzero(zero_mask)
    if remaining_neg > 0 and zeros.size > 0:
        picked_rand = rng.choice(zeros, size=min(remaining_neg, zeros.size), replace=False)
    else:
        picked_rand = np.array([], dtype=np.int64)
    keep = np.concatenate([pos, picked_hard.astype(np.int64), picked_rand.astype(np.int64)])
    keep = np.unique(keep)
    X = feat[keep].astype(np.float32)
    y = labels[keep].astype(np.float32)
    tx_info_keep = tx_info[keep].astype(np.float32)
    punctured_info_keep = punctured_info[keep].astype(np.float32)
    tx_parity_keep = tx_parity[keep].astype(np.float32)
    w = np.where(
        y > 0.5,
        1.0 + 0.60 * tx_info_keep + 0.34 * punctured_info_keep,
        0.32 + 0.04 * (1.0 - tx_parity_keep),
    ).astype(np.float32)
    return X, y, w

def train_ai_ranker(rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], cfg: CalibrationConfig, seed: int) -> Optional[DistilledLinearStudent]:
    if not rows:
        return None
    X = np.concatenate([r[0] for r in rows], axis=0).astype(np.float32)
    y = np.concatenate([r[1] for r in rows], axis=0).astype(np.float32)
    w = np.concatenate([r[2] for r in rows], axis=0).astype(np.float32)
    if X.shape[0] == 0:
        return None
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mu) / sigma
    teacher = TinyTeacherMLP(d_in=Xn.shape[1], hidden=int(cfg.teacher_hidden), seed=int(seed + 11))
    teacher.fit(Xn, y, w, epochs=int(cfg.teacher_epochs), batch_size=int(cfg.batch_size), lr=float(cfg.teacher_lr), seed=int(seed + 13))
    teacher_soft = _sigmoid(teacher.logits(Xn) / float(cfg.temperature)).astype(np.float32)
    student = DistilledLinearStudent(d_in=Xn.shape[1])
    student.fit(
        Xn,
        teacher_soft,
        y,
        w,
        epochs=int(cfg.student_epochs),
        batch_size=int(cfg.batch_size),
        lr=float(cfg.student_lr),
        seed=int(seed + 17),
    )
    student.norm_mu = mu.astype(np.float32)
    student.norm_sigma = sigma.astype(np.float32)
    return student


def save_student(path: str, student: DistilledLinearStudent):
    np.savez(
        path,
        w=student.w.astype(np.float32),
        b=np.array([student.b], dtype=np.float32),
        mu=student.norm_mu.astype(np.float32),
        sigma=student.norm_sigma.astype(np.float32),
        feature_names=np.asarray(FEATURE_NAMES, dtype=np.str_),
    )


def load_student(path: str) -> DistilledLinearStudent:
    payload = np.load(path, allow_pickle=True)
    student = DistilledLinearStudent(int(payload["w"].shape[0]))
    student.w = payload["w"].astype(np.float32)
    student.b = float(payload["b"][0])
    student.norm_mu = payload["mu"].astype(np.float32)
    student.norm_sigma = payload["sigma"].astype(np.float32)
    return student


def student_prob(student: Optional[DistilledLinearStudent], feat: np.ndarray) -> np.ndarray:
    if student is None:
        # heuristic fallback, still AI-like feature fusion but without learned weights
        score = 1.25 * feat[:, 0] + 0.90 * feat[:, 6] + 0.55 * feat[:, 4] + 0.35 * feat[:, 8] + 0.20 * feat[:, 10]
        return _sigmoid(score.astype(np.float32))
    Xn = (feat - student.norm_mu) / student.norm_sigma
    return student.prob(Xn).astype(np.float32)


BLOCK_FEATURE_NAMES = [
    "inv_abs_mean",
    "inv_abs_q25",
    "hard_one_ratio",
    "unsat_mean",
    "weighted_unsat_mean",
    "flip_rate_mean",
    "gain_mean",
    "score_mass",
    "score_top4_mass",
    "score_top8_mass",
    "tx_info_share",
    "punctured_info_share",
    "tx_parity_share",
    "block_size_norm",
]


def _block_feature_pack(
    code_cfg: CodeConfig,
    feat: np.ndarray,
    llr_final: np.ndarray,
    hard_final: np.ndarray,
    base_score: np.ndarray,
) -> np.ndarray:
    B = int(getattr(code_cfg, "_num_blocks", 0))
    if B <= 0:
        return np.zeros((0, len(BLOCK_FEATURE_NAMES)), dtype=np.float32)
    out = np.zeros((B, len(BLOCK_FEATURE_NAMES)), dtype=np.float32)
    filler = getattr(code_cfg, "_filler_mask", np.zeros(code_cfg.N, dtype=np.uint8)).astype(bool)
    valid = ~filler
    abs_llr = np.abs(llr_final).astype(np.float32)
    inv_abs = 1.0 / (1.0 + abs_llr)
    score_pos = np.maximum(base_score.astype(np.float32), 0.0)
    mean_abs = float(np.mean(abs_llr[valid])) if np.any(valid) else 1.0
    max_block = max(1, int(getattr(code_cfg, "_qc_z", 1)))
    for b in range(B):
        idx = np.flatnonzero((code_cfg._block_ids == b) & valid)
        if idx.size == 0:
            continue
        block_abs = abs_llr[idx]
        block_inv = inv_abs[idx]
        block_score = score_pos[idx]
        order = np.argsort(-block_score)
        top4 = block_score[order[: min(4, order.size)]] if order.size > 0 else np.zeros((0,), dtype=np.float32)
        top8 = block_score[order[: min(8, order.size)]] if order.size > 0 else np.zeros((0,), dtype=np.float32)
        out[b, 0] = float(np.mean(block_inv))
        out[b, 1] = float(np.quantile(block_inv, 0.25)) if block_inv.size > 0 else 0.0
        out[b, 2] = float(np.mean(hard_final[idx].astype(np.float32)))
        out[b, 3] = float(np.mean(feat[idx, FEATURE_INDEX["unsat_frac"]]))
        out[b, 4] = float(np.mean(feat[idx, FEATURE_INDEX["weighted_unsat"]]))
        out[b, 5] = float(np.mean(feat[idx, FEATURE_INDEX["flip_rate"]]))
        out[b, 6] = float(np.mean(feat[idx, FEATURE_INDEX["gain_norm"]]))
        out[b, 7] = float(np.sum(block_score) / max(1.0, float(idx.size)))
        out[b, 8] = float(np.sum(top4) / max(1.0, float(top4.size))) if top4.size > 0 else 0.0
        out[b, 9] = float(np.sum(top8) / max(1.0, float(top8.size))) if top8.size > 0 else 0.0
        out[b, 10] = float(np.mean(code_cfg._tx_info_mask[idx].astype(np.float32)))
        out[b, 11] = float(np.mean(code_cfg._punctured_info_mask[idx].astype(np.float32)))
        out[b, 12] = float(np.mean(code_cfg._tx_parity_mask[idx].astype(np.float32)))
        out[b, 13] = float(idx.size / max_block)
    return out.astype(np.float32)


def _collect_block_training_rows(block_feat: np.ndarray, block_labels: np.ndarray, cfg: CalibrationConfig, seed: int):
    pos = np.flatnonzero(block_labels > 0)
    neg = np.flatnonzero(block_labels <= 0)
    if pos.size == 0:
        return None
    rng = np.random.default_rng(int(seed))
    target_neg = int(max(pos.size, round(cfg.neg_ratio * pos.size)))
    take_neg = min(target_neg, neg.size)
    if take_neg > 0:
        picked_neg = rng.choice(neg, size=take_neg, replace=False)
    else:
        picked_neg = np.array([], dtype=np.int64)
    keep = np.unique(np.concatenate([pos.astype(np.int64), picked_neg.astype(np.int64)]))
    X = block_feat[keep].astype(np.float32)
    y = block_labels[keep].astype(np.float32)
    w = np.where(y > 0.5, 1.0 + float(len(picked_neg)) / max(1.0, float(pos.size)), 1.0).astype(np.float32)
    return X, y, w


def train_block_ranker(rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], cfg: CalibrationConfig, seed: int) -> Optional[DistilledLinearStudent]:
    if not rows:
        return None
    X = np.concatenate([r[0] for r in rows], axis=0).astype(np.float32)
    y = np.concatenate([r[1] for r in rows], axis=0).astype(np.float32)
    w = np.concatenate([r[2] for r in rows], axis=0).astype(np.float32)
    if X.shape[0] == 0 or np.all(y == y[0]):
        return None
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mu) / sigma
    teacher = TinyTeacherMLP(d_in=Xn.shape[1], hidden=int(cfg.teacher_hidden), seed=int(seed + 31))
    teacher.fit(Xn, y, w, epochs=int(cfg.teacher_epochs), batch_size=int(cfg.batch_size), lr=float(cfg.teacher_lr), seed=int(seed + 37))
    teacher_soft = _sigmoid(teacher.logits(Xn) / float(cfg.temperature)).astype(np.float32)
    student = DistilledLinearStudent(d_in=Xn.shape[1])
    student.fit(Xn, teacher_soft, y, w, epochs=int(cfg.student_epochs), batch_size=int(cfg.batch_size), lr=float(cfg.student_lr), seed=int(seed + 41))
    student.norm_mu = mu.astype(np.float32)
    student.norm_sigma = sigma.astype(np.float32)
    return student


def save_block_student(path: str, student: DistilledLinearStudent):
    np.savez(
        path,
        w=student.w.astype(np.float32),
        b=np.array([student.b], dtype=np.float32),
        mu=student.norm_mu.astype(np.float32),
        sigma=student.norm_sigma.astype(np.float32),
        feature_names=np.asarray(BLOCK_FEATURE_NAMES, dtype=np.str_),
    )


def load_block_student(path: str) -> DistilledLinearStudent:
    payload = np.load(path, allow_pickle=False)
    student = DistilledLinearStudent(int(payload["w"].shape[0]))
    student.w = payload["w"].astype(np.float32)
    student.b = float(payload["b"][0])
    student.norm_mu = payload["mu"].astype(np.float32)
    student.norm_sigma = payload["sigma"].astype(np.float32)
    return student


def block_prob(student: Optional[DistilledLinearStudent], block_feat: np.ndarray) -> np.ndarray:
    if block_feat.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if student is None:
        score = (
            1.20 * block_feat[:, 0]
            + 0.85 * block_feat[:, 3]
            + 0.70 * block_feat[:, 4]
            + 0.55 * block_feat[:, 5]
            + 0.45 * block_feat[:, 7]
            + 0.30 * block_feat[:, 10]
            + 0.18 * block_feat[:, 11]
            - 0.10 * block_feat[:, 12]
        )
        return _sigmoid(score.astype(np.float32))
    Xn = (block_feat - student.norm_mu) / student.norm_sigma
    return student.prob(Xn).astype(np.float32)


# ------------------------- AI-guided GRAND -------------------------
FRAME_FEATURE_NAMES = [
    "synd_ratio",
    "hard_one_ratio",
    "tx_info_one_ratio",
    "punctured_info_one_ratio",
    "tx_parity_one_ratio",
    "weak_hard_min",
    "weak_hard_mean",
    "weak_hard_q25",
    "global_mean_abs",
    "global_flip_ratio",
    "top12_score_mass",
    "top12_tx_info_mass",
    "top12_punctured_mass",
    "top12_tx_parity_mass",
    "top4_score_conc",
]

def save_gate_student(path: str, student: DistilledLinearStudent):
    np.savez(
        path,
        w=student.w.astype(np.float32),
        b=np.array([student.b], dtype=np.float32),
        mu=student.norm_mu.astype(np.float32),
        sigma=student.norm_sigma.astype(np.float32),
        feature_names=np.array(FRAME_FEATURE_NAMES, dtype=object),
    )


def load_gate_student(path: str) -> DistilledLinearStudent:
    payload = np.load(path, allow_pickle=True)
    student = DistilledLinearStudent(int(payload["w"].shape[0]))
    student.w = payload["w"].astype(np.float32)
    student.b = float(payload["b"][0])
    student.norm_mu = payload["mu"].astype(np.float32)
    student.norm_sigma = payload["sigma"].astype(np.float32)
    return student


def _frame_feature_vec(
    code_cfg: CodeConfig,
    hard_final: np.ndarray,
    llr_final: np.ndarray,
    syndrome_final: np.ndarray,
    feat: np.ndarray,
    base_score: np.ndarray,
) -> np.ndarray:
    hard_mask = hard_final.astype(bool)
    abs_llr = np.abs(llr_final).astype(np.float32)
    hard_abs = abs_llr[hard_mask]
    if hard_abs.size == 0:
        hard_abs = abs_llr
    top_take = min(12, int(base_score.size))
    top_idx = np.argsort(-base_score)[:top_take]
    top_score = base_score[top_idx] if top_idx.size > 0 else np.zeros((1,), dtype=np.float32)
    top_score_mass = float(np.sum(np.maximum(top_score, 0.0)))
    denom_score = float(np.sum(np.maximum(base_score, 0.0)) + 1e-6)
    q25 = float(np.quantile(hard_abs, 0.25)) if hard_abs.size > 0 else 0.0
    top4 = np.sort(top_score)[-min(4, top_idx.size):] if top_idx.size > 0 else np.zeros((1,), dtype=np.float32)
    return np.array(
        [
            float(np.sum(syndrome_final)) / float(max(1, code_cfg.M)),
            float(np.sum(hard_mask)) / float(max(1, code_cfg.N)),
            float(np.sum(hard_mask & code_cfg._tx_info_mask.astype(bool))) / float(max(1, int(np.sum(code_cfg._tx_info_mask)))),
            float(np.sum(hard_mask & code_cfg._punctured_info_mask.astype(bool))) / float(max(1, int(np.sum(code_cfg._punctured_info_mask)))),
            float(np.sum(hard_mask & code_cfg._tx_parity_mask.astype(bool))) / float(max(1, int(np.sum(code_cfg._tx_parity_mask)))),
            float(np.min(hard_abs)) / float(1.0 + np.mean(abs_llr)),
            float(np.mean(hard_abs)) / float(1.0 + np.mean(abs_llr)),
            float(q25) / float(1.0 + np.mean(abs_llr)),
            float(np.mean(abs_llr)) / float(1.0 + np.mean(abs_llr)),
            float(np.mean(feat[:, FEATURE_INDEX["flip_rate"]])),
            float(top_score_mass / denom_score),
            float(np.sum(code_cfg._tx_info_mask[top_idx])) / float(max(1, top_idx.size)),
            float(np.sum(code_cfg._punctured_info_mask[top_idx])) / float(max(1, top_idx.size)),
            float(np.sum(code_cfg._tx_parity_mask[top_idx])) / float(max(1, top_idx.size)),
            float(np.sum(np.maximum(top4, 0.0)) / denom_score),
        ],
        dtype=np.float32,
    )


def train_frame_gate(rows: List[Tuple[np.ndarray, int]], cfg: CalibrationConfig, seed: int) -> Optional[DistilledLinearStudent]:
    if not rows:
        return None
    X = np.stack([r[0] for r in rows], axis=0).astype(np.float32)
    y = np.asarray([r[1] for r in rows], dtype=np.float32)
    if X.shape[0] == 0 or np.all(y == y[0]):
        return None
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mu) / sigma
    pos = float(np.sum(y > 0.5))
    neg = float(max(1, np.sum(y <= 0.5)))
    w = np.where(y > 0.5, 1.0 + neg / max(1.0, pos), 1.0).astype(np.float32)
    teacher = TinyTeacherMLP(d_in=Xn.shape[1], hidden=max(24, int(cfg.teacher_hidden) // 2), seed=int(seed + 23))
    teacher.fit(Xn, y, w, epochs=max(10, int(cfg.teacher_epochs) // 2), batch_size=int(cfg.batch_size), lr=float(cfg.teacher_lr), seed=int(seed + 29))
    soft = _sigmoid(teacher.logits(Xn) / float(cfg.temperature)).astype(np.float32)
    student = DistilledLinearStudent(d_in=Xn.shape[1])
    student.fit(
        Xn,
        soft,
        y,
        w,
        epochs=max(8, int(cfg.student_epochs) // 2),
        batch_size=int(cfg.batch_size),
        lr=float(cfg.student_lr),
        seed=int(seed + 31),
    )
    student.norm_mu = mu.astype(np.float32)
    student.norm_sigma = sigma.astype(np.float32)
    return student


def frame_gate_prob(student: Optional[DistilledLinearStudent], frame_feat: np.ndarray) -> float:
    x = frame_feat.reshape(1, -1).astype(np.float32)
    if student is None:
        score = (
            -3.2 * x[0, 0]
            -2.0 * x[0, 1]
            -1.2 * x[0, 4]
            +1.6 * x[0, 10]
            +1.2 * x[0, 11]
            +0.8 * x[0, 12]
            -0.9 * x[0, 13]
            +0.6 * x[0, 14]
        )
        return float(_sigmoid(np.array([score], dtype=np.float32))[0])
    xn = (x - student.norm_mu) / student.norm_sigma
    return float(student.prob(xn)[0])


def _bit_search_base_score(
    code_cfg: CodeConfig,
    llr_final: np.ndarray,
    ai_prob_vec: np.ndarray,
    feat: np.ndarray,
    cfg: HybridConfig,
) -> np.ndarray:
    score = (
        ai_prob_vec
        + float(cfg.gain_weight) * feat[:, FEATURE_INDEX["gain_norm"]]
        + float(cfg.osc_weight) * feat[:, FEATURE_INDEX["flip_rate"]]
        + 0.28 * feat[:, FEATURE_INDEX["inv_abs_llr"]]
        + 0.08 * feat[:, FEATURE_INDEX["hard_one"]]
        + float(cfg.tx_bonus) * feat[:, FEATURE_INDEX["is_tx"]]
        + float(cfg.info_bonus) * feat[:, FEATURE_INDEX["is_tx_info"]]
        + float(cfg.punctured_info_bonus) * feat[:, FEATURE_INDEX["is_punctured_info"]]
        - float(cfg.parity_penalty) * feat[:, FEATURE_INDEX["is_tx_parity"]]
    )
    return score.astype(np.float32)


def _sorted_ranked(indices: np.ndarray, base_score: np.ndarray, llr_final: np.ndarray) -> List[int]:
    if indices.size == 0:
        return []
    order = sorted(indices.tolist(), key=lambda v: (-float(base_score[int(v)]), float(np.abs(llr_final[int(v)])), int(v)))
    return [int(v) for v in order]


def _domain_ranked_lists(
    code_cfg: CodeConfig,
    hard_final: np.ndarray,
    llr_final: np.ndarray,
    syndrome_final: np.ndarray,
    ai_prob_vec: np.ndarray,
    feat: np.ndarray,
    cfg: HybridConfig,
):
    del hard_final, syndrome_final
    searchable = (~code_cfg._filler_mask.astype(bool))
    base_score = _bit_search_base_score(code_cfg, llr_final, ai_prob_vec, feat, cfg)
    info_idx = np.flatnonzero(searchable & code_cfg._tx_info_mask.astype(bool))
    punctured_info_idx = np.flatnonzero(searchable & code_cfg._punctured_info_mask.astype(bool))
    tx_par_idx = np.flatnonzero(searchable & code_cfg._tx_parity_mask.astype(bool))
    info_ranked = _sorted_ranked(info_idx, base_score, llr_final)
    punct_ranked = _sorted_ranked(punctured_info_idx, base_score, llr_final)
    tx_par_ranked = _sorted_ranked(tx_par_idx, base_score, llr_final)
    return info_ranked, punct_ranked, tx_par_ranked, base_score


def _pool_unique(*lists: List[int]) -> List[int]:
    out = []
    seen = set()
    for lst in lists:
        for v in lst:
            if int(v) not in seen:
                out.append(int(v))
                seen.add(int(v))
    return out


def _build_pool_system(code_cfg: CodeConfig, syndrome_final: np.ndarray, pool: List[int]):
    row_set = set(int(x) for x in np.flatnonzero(syndrome_final))
    for v in pool:
        for c in code_cfg.vars_to_checks[int(v)]:
            row_set.add(int(c))
    rows = sorted(row_set)
    if not rows:
        return rows, np.zeros((0, len(pool)), dtype=np.uint8), np.zeros((0,), dtype=np.uint8)
    row_pos = {r: i for i, r in enumerate(rows)}
    A = np.zeros((len(rows), len(pool)), dtype=np.uint8)
    for j, v in enumerate(pool):
        for c in code_cfg.vars_to_checks[int(v)]:
            A[row_pos[int(c)], j] ^= 1
    b = syndrome_final[np.asarray(rows, dtype=np.int32)].astype(np.uint8)
    return rows, A, b


def _gf2_rref_family(A: np.ndarray, b: np.ndarray):
    A = A.astype(np.uint8, copy=True)
    b = b.astype(np.uint8, copy=True)
    rows, cols = A.shape
    pivot_cols: List[int] = []
    r = 0
    for c in range(cols):
        pivot = -1
        for i in range(r, rows):
            if int(A[i, c]) == 1:
                pivot = i
                break
        if pivot < 0:
            continue
        if pivot != r:
            A[[r, pivot], :] = A[[pivot, r], :]
            b[[r, pivot]] = b[[pivot, r]]
        for i in range(rows):
            if i != r and int(A[i, c]) == 1:
                A[i, :] ^= A[r, :]
                b[i] ^= b[r]
        pivot_cols.append(int(c))
        r += 1
        if r >= rows:
            break
    for i in range(r, rows):
        if int(b[i]) != 0:
            return None
    free_cols = [c for c in range(cols) if c not in set(pivot_cols)]
    x0 = np.zeros(cols, dtype=np.uint8)
    for row_idx, pcol in enumerate(pivot_cols):
        x0[pcol] = b[row_idx]
    null_basis: List[np.ndarray] = []
    for fcol in free_cols:
        z = np.zeros(cols, dtype=np.uint8)
        z[int(fcol)] = 1
        for row_idx, pcol in enumerate(pivot_cols):
            if int(A[row_idx, int(fcol)]) == 1:
                z[int(pcol)] = 1
        null_basis.append(z)
    return x0, null_basis, pivot_cols, free_cols, int(r)


def _column_metric(v: int, llr_final: np.ndarray, ai_prob_vec: np.ndarray, code_cfg: CodeConfig, cfg: HybridConfig) -> float:
    return (
        float(np.abs(llr_final[int(v)]))
        - float(cfg.ai_weight) * float(ai_prob_vec[int(v)])
        - float(cfg.info_bonus) * float(code_cfg._tx_info_mask[int(v)])
        - float(cfg.punctured_info_bonus) * float(code_cfg._punctured_info_mask[int(v)])
        - float(cfg.tx_bonus) * float(code_cfg._tx_mask[int(v)])
        + float(cfg.parity_penalty) * float(code_cfg._tx_parity_mask[int(v)])
    )


def _combined_ai_prob(bit_prob_vec: np.ndarray, block_prob_vec: np.ndarray, code_cfg: CodeConfig) -> np.ndarray:
    if block_prob_vec.size == 0:
        return np.clip(bit_prob_vec.astype(np.float32), 0.0, 1.0)
    blk = block_prob_vec[np.asarray(code_cfg._block_ids, dtype=np.int32)]
    combo = 0.65 * bit_prob_vec.astype(np.float32) + 0.35 * blk.astype(np.float32)
    return np.clip(combo, 0.0, 1.0).astype(np.float32)


def _search_score_all_bits(
    code_cfg: CodeConfig,
    llr_final: np.ndarray,
    feat: np.ndarray,
    bit_prob_vec: np.ndarray,
    block_prob_vec: np.ndarray,
    cfg: HybridConfig,
) -> np.ndarray:
    base = _bit_search_base_score(code_cfg, llr_final, bit_prob_vec, feat, cfg)
    if block_prob_vec.size == 0:
        blk = np.zeros(code_cfg.N, dtype=np.float32)
    else:
        blk = block_prob_vec[np.asarray(code_cfg._block_ids, dtype=np.int32)].astype(np.float32)
    score = (
        base
        + float(cfg.block_prob_weight) * blk
        + float(cfg.block_mass_weight) * feat[:, FEATURE_INDEX["weighted_unsat"]]
        + 0.06 * feat[:, FEATURE_INDEX["hard_one"]]
    )
    score[code_cfg._filler_mask.astype(bool)] = -1e9
    return score.astype(np.float32)


def _pattern_eval(
    code_cfg: CodeConfig,
    hard_final: np.ndarray,
    llr_final: np.ndarray,
    ai_prob_vec: np.ndarray,
    pattern: Tuple[int, ...],
    cfg: HybridConfig,
) -> Dict[str, Any]:
    if len(pattern) == 0:
        bits = hard_final.copy()
    else:
        idx = np.asarray(pattern, dtype=np.int32)
        bits = hard_final.copy()
        bits[idx] ^= 1
    synd = compute_syndrome(bits.astype(np.uint8), code_cfg)
    if len(pattern) > 0:
        idx = np.asarray(pattern, dtype=np.int32)
        info_count = int(np.sum(code_cfg._tx_info_mask[idx]))
        punctured_info_count = int(np.sum(code_cfg._punctured_info_mask[idx]))
        parity_count = int(np.sum(code_cfg._tx_parity_mask[idx]))
        tx_count = int(np.sum(code_cfg._tx_mask[idx]))
        llr_cost = float(np.sum(np.abs(llr_final[idx])))
        ai_bonus = float(np.sum(ai_prob_vec[idx]))
    else:
        info_count = punctured_info_count = parity_count = tx_count = 0
        llr_cost = 0.0
        ai_bonus = 0.0
    metric = _state_metric(int(np.sum(synd)), llr_cost, ai_bonus, info_count, punctured_info_count, tx_count, parity_count, cfg) + 0.02 * len(pattern)
    return {
        "bits": bits,
        "syndrome": synd,
        "metric": float(metric),
        "llr_cost": float(llr_cost),
        "ai_bonus": float(ai_bonus),
        "info_count": int(info_count),
        "punctured_info_count": int(punctured_info_count),
        "parity_count": int(parity_count),
        "tx_count": int(tx_count),
    }


def _top_ranked_blocks(code_cfg: CodeConfig, block_feat: np.ndarray, block_prob_vec: np.ndarray, search_score: np.ndarray, cfg: HybridConfig) -> List[int]:
    if block_feat.size == 0:
        return []
    blk_mass = np.zeros(block_prob_vec.shape[0], dtype=np.float32)
    for b in range(block_prob_vec.shape[0]):
        idx = np.flatnonzero((code_cfg._block_ids == b) & (~code_cfg._filler_mask.astype(bool)))
        if idx.size == 0:
            continue
        blk_mass[b] = float(np.mean(np.maximum(search_score[idx], 0.0)))
    score = 1.1 * block_prob_vec + 0.55 * blk_mass
    order = np.argsort(-score)
    take = int(min(max(1, cfg.top_blocks), order.size))
    return [int(x) for x in order[:take].tolist()]


def _block_mask_candidates(
    code_cfg: CodeConfig,
    llr_final: np.ndarray,
    hard_final: np.ndarray,
    search_score: np.ndarray,
    block_prob_vec: np.ndarray,
    blocks_ranked: List[int],
    cfg: HybridConfig,
) -> Dict[int, List[Tuple[int, ...]]]:
    out: Dict[int, List[Tuple[int, ...]]] = {}
    valid = ~code_cfg._filler_mask.astype(bool)
    for b in blocks_ranked:
        idx = np.flatnonzero((code_cfg._block_ids == int(b)) & valid)
        if idx.size == 0:
            out[int(b)] = []
            continue
        order = sorted(idx.tolist(), key=lambda v: (-float(search_score[int(v)]), float(np.abs(llr_final[int(v)])), -float(hard_final[int(v)]), int(v)))
        n = len(order)
        q_list = [1, 2, 4, max(1, n // 4), max(1, n // 2), n]
        pred = int(max(1, round(float(block_prob_vec[int(b)]) * float(n)))) if block_prob_vec.size > int(b) else max(1, n // 3)
        q_list.extend([pred, min(n, pred + 2), min(n, pred + max(2, n // 8))])
        uniq = []
        seen = set()
        for q in q_list:
            q = int(max(1, min(n, q)))
            if q not in seen:
                seen.add(q)
                uniq.append(q)
        patterns = []
        for q in uniq[: max(2, int(cfg.block_mask_variants))]:
            pat = tuple(sorted(int(v) for v in order[:q]))
            patterns.append(pat)
        out[int(b)] = patterns
    return out


def _beam_block_prefixes(
    code_cfg: CodeConfig,
    hard_final: np.ndarray,
    llr_final: np.ndarray,
    ai_prob_vec: np.ndarray,
    blocks_ranked: List[int],
    masks_by_block: Dict[int, List[Tuple[int, ...]]],
    cfg: HybridConfig,
):
    beam = [{"pattern": tuple(), "used": tuple(), "metric": float("inf"), "bits": hard_final, "syndrome": compute_syndrome(hard_final.astype(np.uint8), code_cfg), "selected_blocks": 0}]
    all_states = []
    direct_success = None
    for depth in range(1, int(cfg.block_combo_max) + 1):
        cand = []
        for st in beam:
            used = set(int(x) for x in st["used"])
            for b in blocks_ranked:
                b = int(b)
                if b in used:
                    continue
                for mask in masks_by_block.get(b, []):
                    merged = tuple(sorted(set(st["pattern"]).union(mask)))
                    ev = _pattern_eval(code_cfg, hard_final, llr_final, ai_prob_vec, merged, cfg)
                    row = {
                        "pattern": merged,
                        "used": tuple(sorted(tuple(used.union({b})))),
                        "metric": float(ev["metric"]),
                        "bits": ev["bits"],
                        "syndrome": ev["syndrome"],
                        "selected_blocks": int(len(used) + 1),
                    }
                    cand.append(row)
                    if int(np.sum(ev["syndrome"])) == 0:
                        if direct_success is None or row["metric"] < direct_success["metric"]:
                            direct_success = row
        if not cand:
            break
        # deduplicate by pattern
        best_map = {}
        for row in cand:
            key = row["pattern"]
            if (key not in best_map) or (row["metric"] < best_map[key]["metric"]):
                best_map[key] = row
        ordered = sorted(best_map.values(), key=lambda r: (r["metric"], len(r["pattern"]), r["pattern"]))
        all_states.extend(ordered)
        beam = ordered[: max(1, int(cfg.block_beam_width))]
    uniq_map = {}
    for row in all_states:
        key = row["pattern"]
        if (key not in uniq_map) or (row["metric"] < uniq_map[key]["metric"]):
            uniq_map[key] = row
    prefixes = sorted(uniq_map.values(), key=lambda r: (r["metric"], len(r["pattern"]), r["pattern"]))[: max(1, int(cfg.prefix_keep))]
    return direct_success, prefixes


def _build_exact_pool_from_prefix(
    code_cfg: CodeConfig,
    llr_final: np.ndarray,
    search_score: np.ndarray,
    prefix_pattern: Tuple[int, ...],
    residual_syndrome: np.ndarray,
    cfg: HybridConfig,
) -> List[int]:
    valid = ~code_cfg._filler_mask.astype(bool)
    prefix_set = set(int(v) for v in prefix_pattern)
    pool = list(prefix_pattern)
    # top bits inside selected blocks
    sel_blocks = sorted(set(int(code_cfg._block_ids[int(v)]) for v in prefix_pattern)) if prefix_pattern else []
    for b in sel_blocks:
        idx = np.flatnonzero((code_cfg._block_ids == int(b)) & valid)
        order = sorted(idx.tolist(), key=lambda v: (-float(search_score[int(v)]), float(np.abs(llr_final[int(v)])), int(v)))
        pool.extend(int(v) for v in order[: max(2, int(cfg.block_refine_bits))])
    # global bits touching current unsatisfied checks
    near = set()
    for c in np.flatnonzero(residual_syndrome):
        for v in code_cfg.checks_to_vars[int(c)]:
            if not bool(code_cfg._filler_mask[int(v)]):
                near.add(int(v))
    near_list = sorted(list(near), key=lambda v: (-float(search_score[int(v)]), float(np.abs(llr_final[int(v)])), int(v)))
    pool.extend(near_list[: max(4, int(cfg.global_top_bits) * 2)])
    # a few global highest-scored bits
    global_idx = np.flatnonzero(valid)
    global_order = sorted(global_idx.tolist(), key=lambda v: (-float(search_score[int(v)]), float(np.abs(llr_final[int(v)])), int(v)))
    pool.extend(global_order[: max(4, int(cfg.global_top_bits))])
    out = []
    seen = set()
    for v in pool:
        v = int(v)
        if v not in seen:
            out.append(v)
            seen.add(v)
        if len(out) >= int(cfg.exact_pool_cap):
            break
    return out


def _direct_global_bits_pattern(code_cfg: CodeConfig, llr_final: np.ndarray, search_score: np.ndarray, top_k: int) -> Tuple[int, ...]:
    valid = np.flatnonzero(~code_cfg._filler_mask.astype(bool))
    order = sorted(valid.tolist(), key=lambda v: (-float(search_score[int(v)]), float(np.abs(llr_final[int(v)])), int(v)))
    k = int(max(1, min(len(order), top_k))) if order else 0
    return tuple(sorted(int(v) for v in order[:k]))


def _state_metric(
    sw_after: int,
    llr_cost: float,
    ai_bonus: float,
    info_count: int,
    punctured_info_count: int,
    tx_count: int,
    parity_count: int,
    cfg: HybridConfig,
) -> float:
    return (
        float(cfg.sw_weight) * float(sw_after)
        + float(cfg.llr_weight) * float(llr_cost)
        - float(cfg.ai_weight) * float(ai_bonus)
        - float(cfg.info_bonus) * float(info_count)
        - float(cfg.punctured_info_bonus) * float(punctured_info_count)
        - float(cfg.tx_bonus) * float(tx_count)
        + float(cfg.parity_penalty) * float(parity_count)
    )


def _solve_exact_on_pool(
    round_name: str,
    code_cfg: CodeConfig,
    hard_final: np.ndarray,
    llr_final: np.ndarray,
    syndrome_final: np.ndarray,
    ai_prob_vec: np.ndarray,
    pool: List[int],
    cfg: HybridConfig,
    require_info: bool,
    max_weight: int,
    free_dim_cap: int,
    max_candidates: int,
):
    if not pool:
        return {
            "success": False,
            "bits": hard_final,
            "pattern": tuple(),
            "round_name": round_name,
            "pool_size": 0,
            "rank": 0,
            "free_dim": 0,
            "solutions_tested": 0,
        }
    _, A, b = _build_pool_system(code_cfg, syndrome_final, pool)
    fam = _gf2_rref_family(A, b)
    if fam is None:
        return {
            "success": False,
            "bits": hard_final,
            "pattern": tuple(),
            "round_name": round_name,
            "pool_size": int(len(pool)),
            "rank": 0,
            "free_dim": 0,
            "solutions_tested": 0,
        }
    x0, null_basis, pivot_cols, free_cols, rank = fam
    free_dim = int(len(free_cols))
    if free_dim > int(free_dim_cap):
        return {
            "success": False,
            "bits": hard_final,
            "pattern": tuple(),
            "round_name": round_name,
            "pool_size": int(len(pool)),
            "rank": int(rank),
            "free_dim": int(free_dim),
            "solutions_tested": 0,
        }

    col_cost = np.array([_column_metric(int(v), llr_final, ai_prob_vec, code_cfg, cfg) for v in pool], dtype=np.float32)
    free_order = sorted(range(free_dim), key=lambda j: (float(col_cost[int(free_cols[j])]), int(j)))
    basis = [null_basis[j] for j in free_order]
    tested = 0
    best = None
    best_metric = float("inf")

    total_assignments = 1 << free_dim
    for mask in range(total_assignments):
        if tested >= int(max_candidates):
            break
        x = x0.copy()
        for j in range(free_dim):
            if (mask >> j) & 1:
                x ^= basis[j]
        wt = int(np.sum(x))
        if wt <= 0 or wt > int(max_weight):
            continue
        pat_cols = np.flatnonzero(x)
        if pat_cols.size == 0:
            continue
        pattern = tuple(sorted(int(pool[int(c)]) for c in pat_cols.tolist()))
        info_count = int(np.sum(code_cfg._tx_info_mask[np.asarray(pattern, dtype=np.int32)]))
        punctured_info_count = int(np.sum(code_cfg._punctured_info_mask[np.asarray(pattern, dtype=np.int32)]))
        if require_info and (info_count + punctured_info_count) <= 0:
            continue
        parity_count = int(np.sum(code_cfg._tx_parity_mask[np.asarray(pattern, dtype=np.int32)]))
        tx_count = int(np.sum(code_cfg._tx_mask[np.asarray(pattern, dtype=np.int32)]))
        llr_cost = float(np.sum(np.abs(llr_final[np.asarray(pattern, dtype=np.int32)])))
        ai_bonus = float(np.sum(ai_prob_vec[np.asarray(pattern, dtype=np.int32)]))
        metric = _state_metric(0, llr_cost, ai_bonus, info_count, punctured_info_count, tx_count, parity_count, cfg) + 0.02 * wt
        tested += 1
        if metric < best_metric:
            bits = hard_final.copy()
            bits[np.asarray(pattern, dtype=np.int32)] ^= 1
            if int(np.sum(compute_syndrome(bits.astype(np.uint8), code_cfg))) == 0:
                best_metric = metric
                best = {
                    "success": True,
                    "bits": bits,
                    "pattern": pattern,
                    "round_name": round_name,
                    "pool_size": int(len(pool)),
                    "rank": int(rank),
                    "free_dim": int(free_dim),
                    "solutions_tested": int(tested),
                    "info_count": int(info_count),
                    "punctured_info_count": int(punctured_info_count),
                    "parity_count": int(parity_count),
                    "tx_count": int(tx_count),
                }
    if best is not None:
        return best
    return {
        "success": False,
        "bits": hard_final,
        "pattern": tuple(),
        "round_name": round_name,
        "pool_size": int(len(pool)),
        "rank": int(rank),
        "free_dim": int(free_dim),
        "solutions_tested": int(tested),
    }




# ------------------------- AI Posterior-Ordered GRAND -------------------------

def _default_pattern_stats(code_cfg: CodeConfig, cfg: HybridConfig) -> Dict[str, Any]:
    size_choices = np.asarray(sorted(set(int(x) for x in cfg.block_prefix_sizes if int(x) > 0)), dtype=np.int32)
    if size_choices.size == 0:
        size_choices = np.asarray([2, 4, 8, 12, 16, 24, 32], dtype=np.int32)
    size_prior = np.ones((size_choices.size,), dtype=np.float32)
    size_prior /= float(size_prior.sum())
    B = int(getattr(code_cfg, "_num_blocks", 0))
    pair_pmi = np.zeros((B, B), dtype=np.float32)
    block_active = np.zeros((B,), dtype=np.float32)
    return {
        "size_choices": size_choices,
        "size_prior": size_prior,
        "pair_pmi": pair_pmi,
        "block_active": block_active,
        "mean_candidate_coverage": np.array([0.0], dtype=np.float32),
    }


def save_pattern_stats(path: str, stats: Dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        size_choices=np.asarray(stats.get("size_choices", []), dtype=np.int32),
        size_prior=np.asarray(stats.get("size_prior", []), dtype=np.float32),
        pair_pmi=np.asarray(stats.get("pair_pmi", []), dtype=np.float32),
        block_active=np.asarray(stats.get("block_active", []), dtype=np.float32),
        mean_candidate_coverage=np.asarray(stats.get("mean_candidate_coverage", [0.0]), dtype=np.float32),
    )


def load_pattern_stats(path: str) -> Dict[str, Any]:
    payload = np.load(path, allow_pickle=False)
    out = {
        "size_choices": payload["size_choices"].astype(np.int32),
        "size_prior": payload["size_prior"].astype(np.float32),
        "pair_pmi": payload["pair_pmi"].astype(np.float32),
        "block_active": payload["block_active"].astype(np.float32),
    }
    if "mean_candidate_coverage" in payload:
        out["mean_candidate_coverage"] = payload["mean_candidate_coverage"].astype(np.float32)
    else:
        out["mean_candidate_coverage"] = np.array([0.0], dtype=np.float32)
    return out


def _prob_clip_bounds() -> Tuple[float, float]:
    pmin = float(os.getenv("GRAND_PROB_MIN", "1.0e-4"))
    pmax = float(os.getenv("GRAND_PROB_MAX", "0.49"))
    pmin = min(max(pmin, 1.0e-6), 0.10)
    pmax = min(max(pmax, 0.05), 0.499999)
    if pmax <= pmin:
        pmax = min(0.49, pmin + 0.10)
    return float(pmin), float(pmax)


def _nearest_anchor_idx(val: int, anchors: np.ndarray) -> int:
    arr = np.asarray(anchors, dtype=np.int32).reshape(-1)
    if arr.size == 0:
        return 0
    v = int(val)
    diffs = np.abs(arr.astype(np.int64) - np.int64(v))
    return int(np.argmin(diffs))


def _safe_logit_prob(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float32), 1.0e-6, 1.0 - 1.0e-6)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def _current_block_scores(
    code_cfg: CodeConfig,
    feat: np.ndarray,
    llr_final: np.ndarray,
    hard_final: np.ndarray,
    bit_prob_vec: np.ndarray,
    cfg: HybridConfig,
    block_student: Optional[DistilledLinearStudent],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_score = _bit_search_base_score(code_cfg, llr_final, bit_prob_vec, feat, cfg)
    blk_feat = _block_feature_pack(code_cfg, feat, llr_final, hard_final, base_score)
    B = int(getattr(code_cfg, "_num_blocks", 0))
    if blk_feat.shape[0] == 0 or B <= 0:
        return blk_feat, np.zeros((B,), dtype=np.float32), np.zeros((B,), dtype=np.float32)
    blk_prob = block_prob(block_student, blk_feat).astype(np.float32)
    mass = blk_feat[:, 7].astype(np.float32)
    mass_norm = mass / float(max(1e-6, float(np.max(mass))))
    blk_score = (
        blk_prob
        + 0.22 * mass_norm
        + 0.08 * blk_feat[:, 10]
        + 0.05 * blk_feat[:, 11]
        - 0.04 * blk_feat[:, 12]
    ).astype(np.float32)
    return blk_feat, blk_prob, blk_score


def _rank_domain_bits(
    code_cfg: CodeConfig,
    llr_final: np.ndarray,
    search_score: np.ndarray,
    limit_info: int,
    limit_punct: int,
    limit_parity: int,
    limit_total: int,
) -> List[int]:
    valid = ~code_cfg._filler_mask.astype(bool)

    def _rank(mask: np.ndarray, lim: int) -> List[int]:
        idx = np.flatnonzero(valid & mask.astype(bool))
        if idx.size == 0 or lim <= 0:
            return []
        order = sorted(idx.tolist(), key=lambda v: (-float(search_score[int(v)]), float(np.abs(llr_final[int(v)])), int(v)))
        return [int(v) for v in order[: int(lim)]]

    ranked: List[int] = []
    seen = set()
    for lst in (
        _rank(code_cfg._tx_info_mask, limit_info),
        _rank(code_cfg._punctured_info_mask, limit_punct),
        _rank(code_cfg._tx_parity_mask, limit_parity),
    ):
        for v in lst:
            if int(v) not in seen:
                ranked.append(int(v))
                seen.add(int(v))

    if len(ranked) < int(limit_total):
        idx = np.flatnonzero(valid)
        order = sorted(idx.tolist(), key=lambda v: (-float(search_score[int(v)]), float(np.abs(llr_final[int(v)])), int(v)))
        for v in order:
            if int(v) not in seen:
                ranked.append(int(v))
                seen.add(int(v))
            if len(ranked) >= int(limit_total):
                break
    return ranked[: int(limit_total)]


def _ai_posterior_prob(
    code_cfg: CodeConfig,
    llr_final: np.ndarray,
    feat: np.ndarray,
    bit_prob_vec: np.ndarray,
    block_student: Optional[DistilledLinearStudent],
    hard_final: np.ndarray,
    cfg: HybridConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    blk_feat, blk_prob_vec, blk_score = _current_block_scores(code_cfg, feat, llr_final, hard_final, bit_prob_vec, cfg, block_student)
    del blk_feat
    valid = ~code_cfg._filler_mask.astype(bool)
    bit_prob_vec = np.clip(np.asarray(bit_prob_vec, dtype=np.float32), 1.0e-6, 1.0 - 1.0e-6)
    bit_logit = _safe_logit_prob(bit_prob_vec)
    if blk_prob_vec.size > 0:
        blk_per_bit = np.clip(blk_prob_vec[np.asarray(code_cfg._block_ids, dtype=np.int32)], 1.0e-6, 1.0 - 1.0e-6)
        blk_logit = _safe_logit_prob(blk_per_bit)
    else:
        blk_per_bit = np.zeros(code_cfg.N, dtype=np.float32)
        blk_logit = np.zeros(code_cfg.N, dtype=np.float32)
    abs_llr = np.abs(llr_final).astype(np.float32)
    mean_abs = float(np.mean(abs_llr[valid])) if np.any(valid) else 1.0
    mean_abs = max(mean_abs, 1.0e-6)
    logit = (
        0.92 * bit_logit
        + 0.34 * blk_logit
        + 0.26 * feat[:, FEATURE_INDEX["weighted_unsat"]]
        + 0.18 * feat[:, FEATURE_INDEX["flip_rate"]]
        + 0.10 * feat[:, FEATURE_INDEX["gain_norm"]]
        + 0.12 * feat[:, FEATURE_INDEX["local_weak_frac"]]
        + 0.08 * feat[:, FEATURE_INDEX["cycle4_norm"]]
        + 0.06 * feat[:, FEATURE_INDEX["twohop_norm"]]
        - 0.12 * (abs_llr / mean_abs)
        + 0.48 * float(cfg.info_bonus) * feat[:, FEATURE_INDEX["is_tx_info"]]
        + 0.42 * float(cfg.punctured_info_bonus) * feat[:, FEATURE_INDEX["is_punctured_info"]]
        - 0.36 * float(cfg.parity_penalty) * feat[:, FEATURE_INDEX["is_tx_parity"]]
        - 0.10 * feat[:, FEATURE_INDEX["is_latent"]]
    ).astype(np.float32)
    pmin, pmax = _prob_clip_bounds()
    prob = _sigmoid(logit).astype(np.float32)
    prob = np.clip(prob, pmin, pmax).astype(np.float32)
    prob[~valid] = pmin
    search_score = _search_score_all_bits(code_cfg, llr_final, feat, prob, blk_prob_vec, cfg)
    return prob.astype(np.float32), search_score.astype(np.float32), blk_score.astype(np.float32)


def _ordered_candidate_pool(
    code_cfg: CodeConfig,
    llr_final: np.ndarray,
    search_score: np.ndarray,
    cfg: HybridConfig,
) -> List[int]:
    total_pool = int(max(int(cfg.single_pool), int(cfg.tx_info_pool) + int(cfg.punctured_info_pool) + int(cfg.tx_parity_pool)))
    pool = _rank_domain_bits(
        code_cfg,
        llr_final,
        search_score,
        int(cfg.tx_info_pool),
        int(cfg.punctured_info_pool),
        int(cfg.tx_parity_pool),
        int(total_pool),
    )
    if len(pool) > total_pool:
        pool = pool[:total_pool]
    return [int(v) for v in pool]


def _syndrome_array_to_int(syndrome_bits: np.ndarray) -> int:
    out = 0
    idx = np.flatnonzero(np.asarray(syndrome_bits, dtype=np.uint8))
    for c in idx.tolist():
        out |= (1 << int(c))
    return int(out)


def _popcount_int(x: int) -> int:
    x = int(x)
    try:
        return int(x.bit_count())
    except AttributeError:
        return int(bin(x).count("1"))


def _ensure_bit_check_mask_int(code_cfg: CodeConfig) -> None:
    if hasattr(code_cfg, "_bit_check_mask_int"):
        return
    masks: List[int] = []
    for v in range(code_cfg.N):
        m = 0
        for c in code_cfg.vars_to_checks[v]:
            m |= (1 << int(c))
        masks.append(int(m))
    code_cfg._bit_check_mask_int = masks


def ordered_pattern_grand(
    code_cfg: CodeConfig,
    hard_final: np.ndarray,
    llr_final: np.ndarray,
    syndrome_final: np.ndarray,
    feat: np.ndarray,
    bit_prob_vec: np.ndarray,
    cfg: HybridConfig,
    block_student: Optional[DistilledLinearStudent],
    pattern_stats: Dict[str, Any],
) -> Dict[str, Any]:
    del pattern_stats
    base_synd = _syndrome_array_to_int(syndrome_final)
    if base_synd == 0:
        return {
            "success": True,
            "bits": hard_final.copy(),
            "pattern": tuple(),
            "patterns_tested": 0,
            "atoms_total": 0,
            "queue_max": 0,
            "zero_synd_candidates": 1,
            "support_weight": 0,
            "atom_count": 0,
            "candidate_pool": 0,
            "predicted_error_mass": 0.0,
        }

    _ensure_bit_check_mask_int(code_cfg)
    prob_vec, search_score, blk_score = _ai_posterior_prob(code_cfg, llr_final, feat, bit_prob_vec, block_student, hard_final, cfg)
    pool = _ordered_candidate_pool(code_cfg, llr_final, search_score, cfg)
    if not pool:
        return {
            "success": False,
            "bits": hard_final,
            "pattern": tuple(),
            "patterns_tested": 0,
            "atoms_total": 0,
            "queue_max": 0,
            "zero_synd_candidates": 0,
            "support_weight": 0,
            "atom_count": 0,
            "candidate_pool": 0,
            "predicted_error_mass": 0.0,
        }

    p_pool = prob_vec[np.asarray(pool, dtype=np.int32)].astype(np.float32)
    costs = np.log((1.0 - p_pool) / p_pool).astype(np.float32)
    order = sorted(range(len(pool)), key=lambda i: (float(costs[i]), -float(search_score[pool[i]]), float(np.abs(llr_final[pool[i]])), int(pool[i])))
    bits_sorted = [int(pool[i]) for i in order]
    cost_sorted = np.asarray([float(costs[i]) for i in order], dtype=np.float32)
    prob_sorted = np.asarray([float(p_pool[i]) for i in order], dtype=np.float32)
    col_masks = [int(code_cfg._bit_check_mask_int[int(v)]) for v in bits_sorted]

    L = len(bits_sorted)
    if L <= 0:
        return {
            "success": False,
            "bits": hard_final,
            "pattern": tuple(),
            "patterns_tested": 0,
            "atoms_total": 0,
            "queue_max": 0,
            "zero_synd_candidates": 0,
            "support_weight": 0,
            "atom_count": 0,
            "candidate_pool": 0,
            "predicted_error_mass": 0.0,
        }

    heap: List[Tuple[float, int, int, Tuple[int, ...], int]] = []
    root_support = (0,)
    root_synd = int(base_synd ^ col_masks[0])
    heapq.heappush(heap, (float(cost_sorted[0]), _popcount_int(root_synd), 1, root_support, root_synd))
    queue_max = 1
    patterns_tested = 0
    zero_synd_candidates = 0
    max_support = int(max(1, cfg.max_support_bits))
    max_patterns = int(max(1, cfg.max_patterns))

    while heap and patterns_tested < max_patterns:
        cost, _, _, support, synd_mask = heapq.heappop(heap)
        patterns_tested += 1
        if synd_mask == 0:
            zero_synd_candidates += 1
            final_bits = hard_final.copy()
            flip_bits = [bits_sorted[idx] for idx in support]
            if flip_bits:
                final_bits[np.asarray(flip_bits, dtype=np.int32)] ^= 1
            return {
                "success": True,
                "bits": final_bits,
                "pattern": tuple(int(v) for v in flip_bits),
                "patterns_tested": int(patterns_tested),
                "atoms_total": int(L),
                "queue_max": int(queue_max),
                "zero_synd_candidates": int(zero_synd_candidates),
                "support_weight": int(len(flip_bits)),
                "atom_count": int(len(flip_bits)),
                "candidate_pool": int(L),
                "predicted_error_mass": float(np.sum(prob_sorted)),
                "first_success_cost": float(cost),
                "block_score_peak": float(np.max(blk_score)) if blk_score.size > 0 else 0.0,
            }

        last = int(support[-1])
        nxt = last + 1
        if nxt >= L:
            continue

        # Child A: replace last candidate by the next one
        if len(support) == 1 or nxt > int(support[-2]):
            child_a = support[:-1] + (nxt,)
            cost_a = float(cost - float(cost_sorted[last]) + float(cost_sorted[nxt]))
            synd_a = int(synd_mask ^ col_masks[last] ^ col_masks[nxt])
            heapq.heappush(heap, (cost_a, _popcount_int(synd_a), len(child_a), child_a, synd_a))

        # Child B: append the next candidate
        if len(support) < max_support:
            child_b = support + (nxt,)
            cost_b = float(cost + float(cost_sorted[nxt]))
            synd_b = int(synd_mask ^ col_masks[nxt])
            heapq.heappush(heap, (cost_b, _popcount_int(synd_b), len(child_b), child_b, synd_b))

        if len(heap) > queue_max:
            queue_max = len(heap)

    return {
        "success": False,
        "bits": hard_final,
        "pattern": tuple(),
        "patterns_tested": int(patterns_tested),
        "atoms_total": int(L),
        "queue_max": int(queue_max),
        "zero_synd_candidates": int(zero_synd_candidates),
        "support_weight": 0,
        "atom_count": 0,
        "candidate_pool": int(L),
        "predicted_error_mass": float(np.sum(prob_sorted)),
        "first_success_cost": 0.0,
        "block_score_peak": float(np.max(blk_score)) if blk_score.size > 0 else 0.0,
    }


# ------------------------- Calibration -------------------------
def _warmup(code_cfg: CodeConfig, stage1_iters: int) -> None:
    llr = np.ones(code_cfg.N, dtype=np.float32)
    llr[: min(16, code_cfg.N)] = -0.5
    dec_cfg = DecoderConfig(max_iters=int(stage1_iters), alpha=float(_env_float("LDPC_ALPHA", 0.80)), early_stop=True)
    snapshot_iters = _snapshot_iter_list(stage1_iters, 3)
    hard, llr_post, synd, _, snaps = ldpc_min_sum_decode(llr, code_cfg, dec_cfg, snapshot_iters=snapshot_iters)
    feat = _feature_pack(code_cfg, llr_post, hard, synd, snaps, int(stage1_iters))
    heur_prob = student_prob(None, feat)
    stats = _default_pattern_stats(code_cfg, HybridConfig())
    _ = ordered_pattern_grand(code_cfg, hard, llr_post, synd, feat, heur_prob, HybridConfig(), None, stats)


def run_calibration(
    run_cfg: RunConfig,
    code_cfg: CodeConfig,
    bit_model_path: str,
    block_model_path: str,
    pattern_stats_path: str,
) -> Dict[str, Any]:
    bit_rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    block_rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    B = int(getattr(code_cfg, "_num_blocks", 0))
    block_active = np.zeros((B,), dtype=np.float32)
    pair_counts = np.zeros((B, B), dtype=np.float32)
    size_choices = np.asarray(sorted(set(int(x) for x in run_cfg.hybrid.block_prefix_sizes if int(x) > 0)), dtype=np.int32)
    if size_choices.size == 0:
        size_choices = np.asarray([2, 4, 8, 12, 16, 24, 32], dtype=np.int32)
    size_counts = np.zeros((size_choices.size,), dtype=np.float32)

    fail_frames = 0
    total_frames = 0
    dec_cfg = DecoderConfig(max_iters=int(run_cfg.stage1_iters), alpha=float(run_cfg.alpha), early_stop=True)
    snapshot_iters = _snapshot_iter_list(run_cfg.stage1_iters, 3)

    for snr_db in run_cfg.calib_snr_db:
        frame = 0
        while total_frames < int(run_cfg.calib.max_frames) and fail_frames < int(run_cfg.calib.target_failed_frames):
            seed = _stable_seed(run_cfg.base_seed, "calib", run_cfg.stage1_iters, snr_db, frame)
            llr = generate_frame_llr(code_cfg, float(snr_db), seed)
            hard, llr_post, synd, _, snaps = ldpc_min_sum_decode(llr, code_cfg, dec_cfg, snapshot_iters=snapshot_iters)
            total_frames += 1
            if int(np.sum(synd)) != 0:
                feat = _feature_pack(code_cfg, llr_post, hard, synd, snaps, int(run_cfg.stage1_iters))
                labels = hard.astype(np.float32).copy()
                labels[code_cfg._filler_mask.astype(bool)] = 0.0
                packed = _collect_training_rows(feat, labels, llr_post, synd, run_cfg.calib, seed + 7)
                if packed is not None:
                    bit_rows.append(packed)

                heur_prob = student_prob(None, feat)
                base_score = _bit_search_base_score(code_cfg, llr_post, heur_prob, feat, run_cfg.hybrid)
                blk_feat = _block_feature_pack(code_cfg, feat, llr_post, hard, base_score)
                if blk_feat.shape[0] > 0:
                    blk_labels = np.zeros((blk_feat.shape[0],), dtype=np.float32)
                    err_counts: List[Tuple[int, int]] = []
                    for b in range(blk_feat.shape[0]):
                        idxs = np.flatnonzero((code_cfg._block_ids == b) & (~code_cfg._filler_mask.astype(bool)))
                        if idxs.size == 0:
                            continue
                        cnt = int(np.sum(labels[idxs]))
                        if cnt > 0:
                            blk_labels[b] = 1.0
                            block_active[b] += 1.0
                            size_idx = _nearest_anchor_idx(cnt, size_choices)
                            size_counts[size_idx] += 1.0
                            err_counts.append((int(b), cnt))
                    blk_packed = _collect_block_training_rows(blk_feat, blk_labels, run_cfg.calib, seed + 19)
                    if blk_packed is not None:
                        block_rows.append(blk_packed)
                    err_counts.sort(key=lambda t: (-int(t[1]), int(t[0])))
                    sel = [int(t[0]) for t in err_counts[: min(6, len(err_counts))]]
                    for i in range(len(sel)):
                        for j in range(i + 1, len(sel)):
                            bi = int(sel[i])
                            bj = int(sel[j])
                            pair_counts[bi, bj] += 1.0
                            pair_counts[bj, bi] += 1.0
                fail_frames += 1
            frame += 1
            if fail_frames >= int(run_cfg.calib.target_failed_frames):
                break
        if fail_frames >= int(run_cfg.calib.target_failed_frames):
            break

    bit_student = train_ai_ranker(bit_rows, run_cfg.calib, seed=run_cfg.base_seed + 701)
    block_student_obj = train_block_ranker(block_rows, run_cfg.calib, seed=run_cfg.base_seed + 1701)

    if bit_student is not None:
        save_student(bit_model_path, bit_student)
    if block_student_obj is not None:
        save_block_student(block_model_path, block_student_obj)

    if fail_frames > 0:
        size_prior = (size_counts + 1.0).astype(np.float32)
        size_prior /= float(size_prior.sum())
        p_i = (block_active + 1.0) / float(fail_frames + 2.0)
        pair_pmi = np.zeros((B, B), dtype=np.float32)
        for i in range(B):
            for j in range(B):
                if i == j:
                    continue
                p_ij = (pair_counts[i, j] + 1.0) / float(fail_frames + 2.0)
                denom = float(p_i[i] * p_i[j]) + 1e-6
                score = math.log(float(p_ij) / denom)
                pair_pmi[i, j] = float(max(0.0, score))
    else:
        size_prior = np.ones((size_choices.size,), dtype=np.float32)
        size_prior /= float(size_prior.sum())
        pair_pmi = np.zeros((B, B), dtype=np.float32)

    stats = {
        "size_choices": size_choices.astype(np.int32),
        "size_prior": size_prior.astype(np.float32),
        "pair_pmi": pair_pmi.astype(np.float32),
        "block_active": block_active.astype(np.float32),
    }
    save_pattern_stats(pattern_stats_path, stats)

    return {
        "calib_total_frames": int(total_frames),
        "calib_failed_frames": int(fail_frames),
        "calib_bit_rows": int(sum(r[0].shape[0] for r in bit_rows)) if bit_rows else 0,
        "calib_block_rows": int(sum(r[0].shape[0] for r in block_rows)) if block_rows else 0,
        "bit_model_ready": bool(bit_student is not None),
        "block_model_ready": bool(block_student_obj is not None),
        "bit_model_path": bit_model_path,
        "block_model_path": block_model_path,
        "pattern_stats_path": pattern_stats_path,
        "size_choices": [int(x) for x in size_choices.tolist()],
    }

# ------------------------- Evaluation -------------------------
def _info_errors(bits: np.ndarray, code_cfg: CodeConfig) -> int:
    return int(np.sum(bits[: code_cfg.K]))


def evaluate_one_snr(
    run_cfg: RunConfig,
    code_cfg: CodeConfig,
    snr_db: float,
    bit_student: Optional[DistilledLinearStudent],
    block_student_obj: Optional[DistilledLinearStudent],
    pattern_stats: Dict[str, Any],
) -> Dict[str, Any]:
    dec_cfg = DecoderConfig(max_iters=int(run_cfg.stage1_iters), alpha=float(run_cfg.alpha), early_stop=True)
    snapshot_iters = _snapshot_iter_list(run_cfg.stage1_iters, 3)

    frames = 0
    ldpc_bit_errors = 0
    hyb_bit_errors = 0
    ldpc_frame_errors = 0
    hyb_frame_errors = 0
    stage1_iter_sum = 0
    failed_frames = 0
    grand_invocations = 0
    grand_rescues = 0
    grand_info_rescues = 0
    grand_parity_only_rescues = 0
    stage1_time = 0.0
    grand_time = 0.0
    patterns_tested_total = 0
    candidate_pool_total = 0
    queue_max_total = 0
    zero_synd_candidates_total = 0
    success_weight_total = 0
    success_atom_count_total = 0
    predicted_error_mass_total = 0.0
    success_cost_total = 0.0
    block_score_peak_total = 0.0
    cap_hits_total = 0

    while frames < int(run_cfg.mc.max_frames):
        seed = _stable_seed(run_cfg.base_seed, "eval", run_cfg.stage1_iters, snr_db, frames)
        llr = generate_frame_llr(code_cfg, float(snr_db), seed)

        t0 = time.perf_counter()
        hard, llr_post, synd, iter_used, snaps = ldpc_min_sum_decode(llr, code_cfg, dec_cfg, snapshot_iters=snapshot_iters)
        stage1_time += time.perf_counter() - t0

        stage1_iter_sum += int(iter_used)
        e_ldpc = _info_errors(hard, code_cfg)
        ldpc_bit_errors += e_ldpc
        ldpc_frame_errors += int(e_ldpc > 0 or int(np.sum(synd)) != 0)

        final_bits = hard
        if int(np.sum(synd)) != 0:
            failed_frames += 1
            grand_invocations += 1
            feat = _feature_pack(code_cfg, llr_post, hard, synd, snaps, int(run_cfg.stage1_iters))
            bit_prob = student_prob(bit_student, feat)
            t1 = time.perf_counter()
            res = ordered_pattern_grand(code_cfg, hard, llr_post, synd, feat, bit_prob, run_cfg.hybrid, block_student_obj, pattern_stats)
            grand_time += time.perf_counter() - t1
            patterns_tested_total += int(res.get("patterns_tested", 0))
            candidate_pool_total += int(res.get("candidate_pool", res.get("atoms_total", 0)))
            queue_max_total += int(res.get("queue_max", 0))
            zero_synd_candidates_total += int(res.get("zero_synd_candidates", 0))
            predicted_error_mass_total += float(res.get("predicted_error_mass", 0.0))
            block_score_peak_total += float(res.get("block_score_peak", 0.0))
            if int(res.get("patterns_tested", 0)) >= int(run_cfg.hybrid.max_patterns):
                cap_hits_total += 1
            if res.get("success", False):
                final_bits = res["bits"]
                grand_rescues += 1
                success_weight_total += int(res.get("support_weight", 0))
                success_atom_count_total += int(res.get("atom_count", 0))
                success_cost_total += float(res.get("first_success_cost", 0.0))

        e_hyb = _info_errors(final_bits, code_cfg)
        hyb_bit_errors += e_hyb
        hyb_synd = compute_syndrome(final_bits.astype(np.uint8), code_cfg)
        hyb_frame_errors += int(e_hyb > 0 or int(np.sum(hyb_synd)) != 0)

        if int(np.sum(synd)) != 0 and int(np.sum(hyb_synd)) == 0:
            if e_hyb < e_ldpc:
                grand_info_rescues += 1
            elif e_ldpc == 0 and e_hyb == 0:
                grand_parity_only_rescues += 1

        frames += 1
        if frames >= int(run_cfg.mc.min_frames):
            if ldpc_frame_errors >= int(run_cfg.mc.target_frame_errors) and hyb_frame_errors >= int(run_cfg.mc.target_frame_errors):
                break

    denom_bits = float(frames * code_cfg.K)
    return {
        "snr_db": float(snr_db),
        "frames": int(frames),
        "ldpc_ber": float(ldpc_bit_errors / denom_bits) if denom_bits > 0 else 0.0,
        "ldpc_fer": float(ldpc_frame_errors / frames) if frames > 0 else 0.0,
        "hyb_ber": float(hyb_bit_errors / denom_bits) if denom_bits > 0 else 0.0,
        "hyb_fer": float(hyb_frame_errors / frames) if frames > 0 else 0.0,
        "avg_stage1_iters": float(stage1_iter_sum / frames) if frames > 0 else 0.0,
        "failed_frame_rate": float(failed_frames / frames) if frames > 0 else 0.0,
        "grand_invocation_rate": float(grand_invocations / frames) if frames > 0 else 0.0,
        "grand_rescue_rate_given_invoked": float(grand_rescues / grand_invocations) if grand_invocations > 0 else 0.0,
        "grand_info_rescue_rate_given_invoked": float(grand_info_rescues / grand_invocations) if grand_invocations > 0 else 0.0,
        "grand_parity_only_rescue_rate_given_invoked": float(grand_parity_only_rescues / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_patterns_tested_per_invoked": float(patterns_tested_total / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_candidate_pool_per_invoked": float(candidate_pool_total / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_atoms_total_per_invoked": float(candidate_pool_total / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_queue_max_per_invoked": float(queue_max_total / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_zero_synd_candidates_per_invoked": float(zero_synd_candidates_total / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_predicted_error_mass_per_invoked": float(predicted_error_mass_total / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_block_score_peak_per_invoked": float(block_score_peak_total / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_success_pattern_weight_given_rescue": float(success_weight_total / grand_rescues) if grand_rescues > 0 else 0.0,
        "avg_success_atom_count_given_rescue": float(success_atom_count_total / grand_rescues) if grand_rescues > 0 else 0.0,
        "avg_first_success_cost_given_rescue": float(success_cost_total / grand_rescues) if grand_rescues > 0 else 0.0,
        "grand_cap_hit_rate_given_invoked": float(cap_hits_total / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_stage1_decoder_us": float(stage1_time * 1e6 / frames) if frames > 0 else 0.0,
        "avg_grand_decoder_us": float(grand_time * 1e6 / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_total_hybrid_decoder_us": float((stage1_time + grand_time) * 1e6 / frames) if frames > 0 else 0.0,
    }


def save_summary_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            vals = []
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, float):
                    vals.append(f"{v:.12g}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")


# ------------------------- Main -------------------------
def build_run_config(results_dir: str) -> RunConfig:
    stage1_iters = _env_int("STAGE1_ITERS", 15)
    k_info = _env_int("SIONNA_5G_K", 1024)
    n_tx = _env_int("SIONNA_5G_N", 2048)
    qm = _env_int("SIONNA_5G_QM", 1)
    alpha = _env_float("LDPC_ALPHA", 0.80)
    eval_snr_db = _env_csv_floats("EVAL_SNR_SWEEP", "6.0,6.5,7.0,7.5,8.0")
    calib_snr_db = _env_csv_floats("CALIB_SNR_SWEEP", ",".join(str(x) for x in eval_snr_db[: max(4, min(5, len(eval_snr_db)))]))
    mc = MCConfig(
        target_frame_errors=_env_int("TARGET_FRAME_ERRORS", 100),
        max_frames=_env_int("MAX_FRAMES", 150000),
        min_frames=_env_int("MIN_FRAMES", 0),
    )
    calib = CalibrationConfig(
        target_failed_frames=_env_int("CALIB_TARGET_FAILED_FRAMES", 700),
        max_frames=_env_int("CALIB_MAX_FRAMES", 260000),
        neg_ratio=_env_float("CALIB_NEG_RATIO", 6.0),
        hard_negative_cap=_env_int("CALIB_HARD_NEG_CAP", 128),
        teacher_hidden=_env_int("AI_TEACHER_HIDDEN", 96),
        teacher_epochs=_env_int("AI_TEACHER_EPOCHS", 28),
        student_epochs=_env_int("AI_STUDENT_EPOCHS", 22),
        batch_size=_env_int("AI_BATCH_SIZE", 4096),
        teacher_lr=_env_float("AI_TEACHER_LR", 0.01),
        student_lr=_env_float("AI_STUDENT_LR", 0.020),
        temperature=_env_float("AI_DISTILL_TEMP", 2.5),
        calib_tx_info_pool=_env_int("CALIB_TX_INFO_POOL", 56),
        calib_punctured_info_pool=_env_int("CALIB_PUNCTURED_INFO_POOL", 24),
        calib_tx_parity_pool=_env_int("CALIB_TX_PARITY_POOL", 16),
        calib_weight_cap=_env_int("CALIB_WEIGHT_CAP", 12),
        calib_free_dim_cap=_env_int("CALIB_FREE_DIM_CAP", 14),
        calib_max_candidates=_env_int("CALIB_MAX_CANDIDATES", 8192),
    )
    hybrid = HybridConfig(
        tx_info_pool=_env_int("GRAND_TX_INFO_POOL", 56),
        tx_parity_pool=_env_int("GRAND_TX_PARITY_POOL", 16),
        punctured_info_pool=_env_int("GRAND_PUNCTURED_INFO_POOL", 24),
        round2_extra_info=_env_int("GRAND_ROUND2_EXTRA_INFO", 12),
        round2_extra_punctured=_env_int("GRAND_ROUND2_EXTRA_PUNCTURED", 8),
        round2_extra_parity=_env_int("GRAND_ROUND2_EXTRA_PARITY", 6),
        max_weight=_env_int("GRAND_MAX_WEIGHT", 160),
        free_dim_cap=_env_int("GRAND_FREE_DIM_CAP", 20),
        max_patterns=_env_int("GRAND_MAX_PATTERNS", 200000),
        frame_gate_threshold=_env_float("GRAND_FRAME_GATE_THRESHOLD", 0.0),
        gate_max_synd=_env_int("GRAND_GATE_MAX_SYND", 999),
        gate_max_hard_ones=_env_int("GRAND_GATE_MAX_HARD_ONES", 999),
        ai_weight=_env_float("GRAND_AI_WEIGHT", 1.60),
        gain_weight=_env_float("GRAND_GAIN_WEIGHT", 0.24),
        osc_weight=_env_float("GRAND_OSC_WEIGHT", 0.18),
        llr_weight=_env_float("GRAND_LLR_WEIGHT", 1.00),
        sw_weight=_env_float("GRAND_SW_WEIGHT", 1.20),
        info_bonus=_env_float("GRAND_INFO_BONUS", 0.82),
        tx_bonus=_env_float("GRAND_TX_BONUS", 0.16),
        punctured_info_bonus=_env_float("GRAND_PUNCTURED_INFO_BONUS", 0.46),
        parity_penalty=_env_float("GRAND_PARITY_PENALTY", 0.20),
        block_prob_weight=_env_float("GRAND_BLOCK_PROB_WEIGHT", 0.95),
        block_mass_weight=_env_float("GRAND_BLOCK_MASS_WEIGHT", 0.55),
        block_combo_max=_env_int("GRAND_BLOCK_COMBO_MAX", 3),
        top_blocks=_env_int("GRAND_TOP_BLOCKS", 12),
        block_beam_width=_env_int("GRAND_BLOCK_BEAM_WIDTH", 64),
        prefix_keep=_env_int("GRAND_PREFIX_KEEP", 8),
        block_refine_bits=_env_int("GRAND_BLOCK_REFINE_BITS", 16),
        global_top_bits=_env_int("GRAND_GLOBAL_TOP_BITS", 28),
        direct_top_bits=_env_int("GRAND_DIRECT_TOP_BITS", 8),
        exact_pool_cap=_env_int("GRAND_EXACT_POOL_CAP", 72),
        block_mask_variants=_env_int("GRAND_BLOCK_MASK_VARIANTS", 4),
        traj_depth=_env_int("GRAND_TRAJ_DEPTH", 5),
        traj_try_top=_env_int("GRAND_TRAJ_TRY_TOP", 2),
        traj_info_weight=_env_float("GRAND_TRAJ_INFO_WEIGHT", 2.20),
        traj_punctured_weight=_env_float("GRAND_TRAJ_PUNCTURED_WEIGHT", 1.35),
        traj_parity_weight=_env_float("GRAND_TRAJ_PARITY_WEIGHT", 0.45),
        traj_synd_weight=_env_float("GRAND_TRAJ_SYND_WEIGHT", 0.10),
        traj_best_scale=_env_float("GRAND_TRAJ_BEST_SCALE", 1.80),
        traj_second_scale=_env_float("GRAND_TRAJ_SECOND_SCALE", 1.25),
        single_pool=_env_int("GRAND_SINGLE_POOL", 96),
        pair_top_blocks=_env_int("GRAND_PAIR_TOP_BLOCKS", 6),
        queue_cap=_env_int("GRAND_QUEUE_CAP", 65536),
        expand_top_k=_env_int("GRAND_EXPAND_TOP_K", 24),
        max_atoms_per_pattern=_env_int("GRAND_MAX_ATOMS_PER_PATTERN", 4),
        max_support_bits=_env_int("GRAND_MAX_SUPPORT_BITS", 160),
        block_prefix_sizes=[int(round(x)) for x in _env_csv_floats("GRAND_BLOCK_PREFIX_SIZES", "2,4,8,12,16,24,32")],
        pair_prefix_sizes=[int(round(x)) for x in _env_csv_floats("GRAND_PAIR_PREFIX_SIZES", "4,8,12,16")],
        syndrome_weight=_env_float("GRAND_SYNDROME_WEIGHT", 1.60),
        atom_bonus_weight=_env_float("GRAND_ATOM_BONUS_WEIGHT", 0.95),
        pair_bonus_weight=_env_float("GRAND_PAIR_BONUS_WEIGHT", 0.80),
        size_prior_weight=_env_float("GRAND_SIZE_PRIOR_WEIGHT", 0.45),
        support_penalty=_env_float("GRAND_SUPPORT_PENALTY", 0.035),
        atom_count_penalty=_env_float("GRAND_ATOM_COUNT_PENALTY", 0.12),
        overlap_penalty=_env_float("GRAND_OVERLAP_PENALTY", 0.18),
    )
    base_seed = _env_int("RNG_SEED_GLOBAL", 20260406)
    return RunConfig(
        results_dir=results_dir,
        stage1_iters=stage1_iters,
        k_info=k_info,
        n_tx=n_tx,
        qm=qm,
        alpha=alpha,
        eval_snr_db=eval_snr_db,
        calib_snr_db=calib_snr_db,
        mc=mc,
        calib=calib,
        hybrid=hybrid,
        base_seed=base_seed,
    )


def main():
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        results_dir = sys.argv[1].strip()
    else:
        results_dir = os.environ.get("RESULTS_DIR", "./results")
    os.makedirs(results_dir, exist_ok=True)

    if not SIONNA_AVAILABLE:
        raise RuntimeError(
            "Sionna LDPC/TDL support is required. "
            f"Import detail: {_SIONNA_IMPORT_ERROR}"
        )

    run_cfg = build_run_config(results_dir)
    print(f"[RUNTIME] python={sys.executable}")
    print(f"[RUNTIME] numba_available={NUMBA_AVAILABLE} numba_threads={NUMBA_THREADS} disable_numba={_DISABLE_NUMBA}")
    print(f"[RUNTIME] results_dir={results_dir}")
    code_cfg = build_sionna_5g_nr_code_cfg(run_cfg.k_info, run_cfg.n_tx, run_cfg.qm)
    _warmup(code_cfg, run_cfg.stage1_iters)

    bit_model_path = os.path.join(results_dir, f"aipog_bit_student_it{run_cfg.stage1_iters:02d}.npz")
    block_model_path = os.path.join(results_dir, f"aipog_block_student_it{run_cfg.stage1_iters:02d}.npz")
    pattern_stats_path = os.path.join(results_dir, f"aipog_pattern_stats_it{run_cfg.stage1_iters:02d}.npz")

    calib_meta = run_calibration(run_cfg, code_cfg, bit_model_path, block_model_path, pattern_stats_path)
    try:
        bit_student = load_student(bit_model_path) if calib_meta["bit_model_ready"] and os.path.exists(bit_model_path) else None
    except Exception as e:
        print(f"[WARN] failed to load bit student: {e!r}")
        bit_student = None
    try:
        block_student_obj = load_block_student(block_model_path) if calib_meta["block_model_ready"] and os.path.exists(block_model_path) else None
    except Exception as e:
        print(f"[WARN] failed to load block student: {e!r}")
        block_student_obj = None
    try:
        pattern_stats = load_pattern_stats(pattern_stats_path) if os.path.exists(pattern_stats_path) else _default_pattern_stats(code_cfg, run_cfg.hybrid)
    except Exception as e:
        print(f"[WARN] failed to load pattern stats: {e!r}")
        pattern_stats = _default_pattern_stats(code_cfg, run_cfg.hybrid)

    rows: List[Dict[str, Any]] = []
    print(f"[RUN] code={code_cfg.code_name} it={run_cfg.stage1_iters} eval_snr={run_cfg.eval_snr_db}")
    print(f"[RUN] calib={calib_meta}")
    print(f"[RUN] aipog max_patterns={run_cfg.hybrid.max_patterns} pool={run_cfg.hybrid.single_pool} tx_info_pool={run_cfg.hybrid.tx_info_pool} punct_pool={run_cfg.hybrid.punctured_info_pool} tx_par_pool={run_cfg.hybrid.tx_parity_pool}")

    for snr_db in run_cfg.eval_snr_db:
        res = evaluate_one_snr(run_cfg, code_cfg, float(snr_db), bit_student, block_student_obj, pattern_stats)
        rows.append(res)
        print("\n=== AIPOG HYBRID EVAL ===")
        print(f"SNR (dB)                         : {res['snr_db']:.2f}")
        print(f"Frames simulated                 : {res['frames']}")
        print(f"LDPC FER / BER                   : {res['ldpc_fer']:.6e} / {res['ldpc_ber']:.6e}")
        print(f"Hybrid FER / BER                 : {res['hyb_fer']:.6e} / {res['hyb_ber']:.6e}")
        print(f"Avg stage-1 iters/frame          : {res['avg_stage1_iters']:.3f}")
        print(f"Failed frame rate                : {res['failed_frame_rate']:.6f}")
        print(f"GRAND invocation rate            : {res['grand_invocation_rate']:.6f}")
        print(f"GRAND rescue rate | invoked      : {res['grand_rescue_rate_given_invoked']:.6f}")
        print(f"Info rescue rate | invoked       : {res['grand_info_rescue_rate_given_invoked']:.6f}")
        print(f"Parity-only rescue | invoked     : {res['grand_parity_only_rescue_rate_given_invoked']:.6f}")
        print(f"Avg patterns tested | invoked    : {res['avg_patterns_tested_per_invoked']:.3f}")
        print(f"Avg candidate pool | invoked     : {res['avg_candidate_pool_per_invoked']:.3f}")
        print(f"Avg predicted mass | invoked     : {res['avg_predicted_error_mass_per_invoked']:.3f}")
        print(f"Avg block-peak | invoked         : {res['avg_block_score_peak_per_invoked']:.3f}")
        print(f"Avg queue max | invoked          : {res['avg_queue_max_per_invoked']:.3f}")
        print(f"Avg zero-synd candidates | inv   : {res['avg_zero_synd_candidates_per_invoked']:.3f}")
        print(f"Success pattern weight           : {res['avg_success_pattern_weight_given_rescue']:.3f}")
        print(f"Success first-cost               : {res['avg_first_success_cost_given_rescue']:.3f}")
        print(f"Avg stage-1 decoder us           : {res['avg_stage1_decoder_us']:.3f}")
        print(f"Avg GRAND decoder us             : {res['avg_grand_decoder_us']:.3f}")
        print(f"Avg total hybrid decoder us      : {res['avg_total_hybrid_decoder_us']:.3f}")

    prefix = f"aipog_it{run_cfg.stage1_iters:02d}_{_now_tag()}"
    summary_path = os.path.join(results_dir, f"{prefix}_summary.csv")
    raw_path = os.path.join(results_dir, f"{prefix}_raw.pkl")
    cfg_path = os.path.join(results_dir, f"{prefix}_config.json")
    save_summary_csv(rows, summary_path)
    with open(raw_path, "wb") as f:
        pickle.dump({"rows": rows, "calibration": calib_meta}, f)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage1_iters": run_cfg.stage1_iters,
                "code": code_cfg.code_name,
                "k_info": run_cfg.k_info,
                "n_tx": run_cfg.n_tx,
                "qm": run_cfg.qm,
                "alpha": run_cfg.alpha,
                "eval_snr_db": run_cfg.eval_snr_db,
                "calib_snr_db": run_cfg.calib_snr_db,
                "mc": run_cfg.mc.__dict__,
                "calib": run_cfg.calib.__dict__,
                "hybrid": run_cfg.hybrid.__dict__,
                "calibration": calib_meta,
                "channel": {
                    "CHANNEL_NAME": os.getenv("CHANNEL_NAME", "SIONNA_TDL"),
                    "SIONNA_TDL_MODEL": os.getenv("SIONNA_TDL_MODEL", "B"),
                    "SIONNA_TDL_DELAY_SPREAD_S": os.getenv("SIONNA_TDL_DELAY_SPREAD_S", "6e-8"),
                    "SIONNA_TDL_MIN_SPEED": os.getenv("SIONNA_TDL_MIN_SPEED", "0.2"),
                    "SIONNA_TDL_MAX_SPEED": os.getenv("SIONNA_TDL_MAX_SPEED", "1.0"),
                    "SIONNA_CFO_HZ": os.getenv("SIONNA_CFO_HZ", "4.0"),
                    "SIONNA_CSI_MODE": os.getenv("SIONNA_CSI_MODE", "nr_imperfect"),
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"[SAVE] summary={summary_path}")
    print(f"[SAVE] raw={raw_path}")
    print(f"[SAVE] config={cfg_path}")
    print(f"[SAVE] bit_model={bit_model_path if calib_meta['bit_model_ready'] else 'not-created'}")
    print(f"[SAVE] block_model={block_model_path if calib_meta['block_model_ready'] else 'not-created'}")
    print(f"[SAVE] pattern_stats={pattern_stats_path}")



if __name__ == "__main__":
    main()
