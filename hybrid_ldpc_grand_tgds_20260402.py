#!/usr/bin/env python3
import os
import sys
import math
import json
import time
import pickle
import datetime
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# ------------------------- Optional acceleration -------------------------
try:
    from numba import njit, prange, set_num_threads, get_num_threads
    NUMBA_AVAILABLE = True
except Exception:
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
    target_failed_frames: int = 96
    max_frames: int = 50000
    neg_ratio: float = 6.0
    hard_negative_cap: int = 64
    teacher_hidden: int = 32
    teacher_epochs: int = 18
    student_epochs: int = 16
    batch_size: int = 4096
    teacher_lr: float = 0.01
    student_lr: float = 0.025
    temperature: float = 2.5


@dataclass
class HybridConfig:
    global_pool: int = 14
    llr_pool: int = 14
    gain_pool: int = 14
    osc_pool: int = 10
    union_cap: int = 24
    roi_local_cap: int = 10
    max_rois: int = 4
    tiny_budget: int = 32
    mid_budget: int = 128
    full_budget: int = 384
    tiny_wmax: int = 2
    mid_wmax: int = 3
    full_wmax: int = 4
    ai_weight: float = 1.35
    gain_weight: float = 0.22
    osc_weight: float = 0.16
    compact_weight: float = 0.10
    llr_weight: float = 1.0


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
    @njit(parallel=True, cache=True)
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

    @njit(parallel=True, cache=True)
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

    @njit(parallel=True, cache=True)
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

    @njit(parallel=True, cache=True)
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
        payload = np.load(path, allow_pickle=True)
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
]


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
        ],
        axis=1,
    ).astype(np.float32)
    return feat


def _collect_training_rows(feat: np.ndarray, labels: np.ndarray, llr_final: np.ndarray, syndrome_final: np.ndarray, cfg: CalibrationConfig, seed: int):
    rng = np.random.default_rng(int(seed))
    pos = np.flatnonzero(labels > 0)
    if pos.size == 0:
        return None
    abs_llr = np.abs(llr_final)
    bit_degree = np.maximum(1.0, feat[:, 6] * 0 + 1.0)  # placeholder to keep shapes simple
    # Hard negatives: wrong-looking but not true errors
    score = (1.0 / (1.0 + abs_llr)) + 0.8 * feat[:, 6] + 0.6 * feat[:, 4]
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
    w = np.where(y > 0.5, np.float32(1.0), np.float32(0.35)).astype(np.float32)
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
        feature_names=np.array(FEATURE_NAMES, dtype=object),
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


# ------------------------- AI-guided GRAND -------------------------
def _candidate_union(code_cfg: CodeConfig, hard_final: np.ndarray, llr_final: np.ndarray, syndrome_final: np.ndarray, ai_prob_vec: np.ndarray, feat: np.ndarray, cfg: HybridConfig) -> List[int]:
    hard_idx = np.flatnonzero(hard_final > 0)
    if hard_idx.size == 0:
        return []
    abs_llr = np.abs(llr_final)
    score = ai_prob_vec + float(cfg.gain_weight) * feat[:, 7] + float(cfg.osc_weight) * feat[:, 4]
    gain_rank = np.argsort(-(feat[hard_idx, 7] + 0.25 * ai_prob_vec[hard_idx]))
    ai_rank = np.argsort(-score[hard_idx])
    llr_rank = np.argsort(abs_llr[hard_idx])
    osc_rank = np.argsort(-(feat[hard_idx, 4] + 0.25 * ai_prob_vec[hard_idx]))
    out: List[int] = []
    seen = set()
    for arr, take in (
        (ai_rank, cfg.global_pool),
        (llr_rank, cfg.llr_pool),
        (gain_rank, cfg.gain_pool),
        (osc_rank, cfg.osc_pool),
    ):
        for idx in arr[:int(take)]:
            v = int(hard_idx[int(idx)])
            if v not in seen:
                seen.add(v)
                out.append(v)
            if len(out) >= int(cfg.union_cap):
                return out
    return out


def _shared_unsat_counts(code_cfg: CodeConfig, union_bits: List[int], syndrome_final: np.ndarray) -> np.ndarray:
    m = len(union_bits)
    shared = np.zeros((m, m), dtype=np.int32)
    unsat = syndrome_final.astype(bool)
    bit_to_pos = {v: i for i, v in enumerate(union_bits)}
    for c in np.flatnonzero(unsat):
        loc = [bit_to_pos[int(v)] for v in code_cfg.checks_to_vars[int(c)] if int(v) in bit_to_pos]
        for i in range(len(loc)):
            for j in range(i + 1, len(loc)):
                shared[loc[i], loc[j]] += 1
                shared[loc[j], loc[i]] += 1
    return shared


def _roi_components(shared: np.ndarray) -> List[List[int]]:
    m = shared.shape[0]
    seen = np.zeros(m, dtype=np.uint8)
    comps: List[List[int]] = []
    for i in range(m):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = 1
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            nbrs = np.flatnonzero(shared[u] > 0)
            for v in nbrs:
                if not seen[v]:
                    seen[v] = 1
                    stack.append(int(v))
        comps.append(comp)
    return comps


def _apply_pattern_syndrome(syndrome: np.ndarray, pattern: Tuple[int, ...], code_cfg: CodeConfig) -> np.ndarray:
    s = syndrome.copy()
    for v in pattern:
        for c in code_cfg.vars_to_checks[int(v)]:
            s[int(c)] ^= 1
    return s


def _pattern_metric(pattern: Tuple[int, ...], llr_final: np.ndarray, ai_prob_vec: np.ndarray, shared_lookup: Dict[Tuple[int, int], int], cfg: HybridConfig) -> float:
    llr_cost = float(np.sum(np.abs(llr_final[list(pattern)])))
    ai_bonus = float(np.sum(ai_prob_vec[list(pattern)]))
    compact = 0.0
    if len(pattern) >= 2:
        for a, b in itertools.combinations(pattern, 2):
            compact += float(shared_lookup.get((min(a, b), max(a, b)), 0))
    return float(cfg.llr_weight) * llr_cost - float(cfg.ai_weight) * ai_bonus - float(cfg.compact_weight) * compact


def ai_guided_grand(code_cfg: CodeConfig, hard_final: np.ndarray, llr_final: np.ndarray, syndrome_final: np.ndarray, ai_prob_vec: np.ndarray, feat: np.ndarray, cfg: HybridConfig):
    union_bits = _candidate_union(code_cfg, hard_final, llr_final, syndrome_final, ai_prob_vec, feat, cfg)
    if not union_bits:
        return {
            "success": False,
            "bits": hard_final,
            "patterns_tested": 0,
            "union_bits": [],
            "budget": 0,
            "wmax": 0,
            "trigger_mode": "empty",
        }
    shared = _shared_unsat_counts(code_cfg, union_bits, syndrome_final)
    comps = _roi_components(shared)
    score = ai_prob_vec + float(cfg.gain_weight) * feat[:, 7] + float(cfg.osc_weight) * feat[:, 4]
    comp_scored = []
    for comp in comps:
        bits = [union_bits[i] for i in comp]
        bits = sorted(bits, key=lambda v: (-score[v], np.abs(llr_final[v])))
        comp_score = float(np.mean(score[bits])) + 0.05 * float(len(bits) <= int(cfg.roi_local_cap))
        comp_scored.append((comp_score, bits))
    comp_scored.sort(key=lambda x: -x[0])
    topk = min(4, len(union_bits))
    top_conf = float(np.mean(np.sort(score[union_bits])[-topk:])) if topk > 0 else 0.0
    synd_w = int(np.sum(syndrome_final))
    if synd_w <= 8 and top_conf >= 0.42:
        budget = int(cfg.full_budget)
        wmax = int(cfg.full_wmax)
        trigger_mode = "full"
    elif synd_w <= 18 and top_conf >= 0.28:
        budget = int(cfg.mid_budget)
        wmax = int(cfg.mid_wmax)
        trigger_mode = "mid"
    else:
        budget = int(cfg.tiny_budget)
        wmax = int(cfg.tiny_wmax)
        trigger_mode = "tiny"

    shared_lookup: Dict[Tuple[int, int], int] = {}
    for i, a in enumerate(union_bits):
        for j in range(i + 1, len(union_bits)):
            b = union_bits[j]
            val = int(shared[i, j])
            if val > 0:
                shared_lookup[(min(a, b), max(a, b))] = val

    patterns = []
    seen = set()

    # ROI-first enumeration
    for _, bits in comp_scored[: int(cfg.max_rois)]:
        local = bits[: int(cfg.roi_local_cap)]
        for w in range(1, min(wmax, len(local)) + 1):
            for comb in itertools.combinations(local, w):
                if comb in seen:
                    continue
                seen.add(comb)
                s_after = _apply_pattern_syndrome(syndrome_final, comb, code_cfg)
                sw_after = int(np.sum(s_after))
                metric = _pattern_metric(comb, llr_final, ai_prob_vec, shared_lookup, cfg)
                patterns.append((sw_after, metric, len(comb), comb))

    # Global complementary lists
    global_sorted = sorted(union_bits, key=lambda v: (-score[v], np.abs(llr_final[v])))
    for w in range(1, min(wmax, len(global_sorted)) + 1):
        for comb in itertools.combinations(global_sorted, w):
            if comb in seen:
                continue
            seen.add(comb)
            s_after = _apply_pattern_syndrome(syndrome_final, comb, code_cfg)
            sw_after = int(np.sum(s_after))
            metric = _pattern_metric(comb, llr_final, ai_prob_vec, shared_lookup, cfg)
            patterns.append((sw_after, metric, len(comb), comb))

    patterns.sort(key=lambda x: (x[0], x[1], x[2]))
    tested = 0
    for sw_after, _, _, comb in patterns[:budget]:
        tested += 1
        if sw_after == 0:
            bits = hard_final.copy()
            bits[list(comb)] ^= 1
            return {
                "success": True,
                "bits": bits,
                "patterns_tested": tested,
                "union_bits": union_bits,
                "budget": budget,
                "wmax": wmax,
                "trigger_mode": trigger_mode,
            }
    return {
        "success": False,
        "bits": hard_final,
        "patterns_tested": tested,
        "union_bits": union_bits,
        "budget": budget,
        "wmax": wmax,
        "trigger_mode": trigger_mode,
    }


# ------------------------- Calibration -------------------------
def _warmup(code_cfg: CodeConfig, stage1_iters: int):
    llr = np.ones(code_cfg.N, dtype=np.float32)
    llr[: min(16, code_cfg.N)] = -0.5
    dec_cfg = DecoderConfig(max_iters=int(stage1_iters), alpha=float(_env_float("LDPC_ALPHA", 0.80)), early_stop=True)
    snapshot_iters = sorted(set([max(1, stage1_iters - 2), max(1, stage1_iters - 1), stage1_iters]))
    hard, llr_post, synd, _, snaps = ldpc_min_sum_decode(llr, code_cfg, dec_cfg, snapshot_iters=snapshot_iters)
    feat = _feature_pack(code_cfg, llr_post, hard, synd, snaps, int(stage1_iters))
    _ = ai_guided_grand(code_cfg, hard, llr_post, synd, student_prob(None, feat), feat, HybridConfig())


def run_calibration(run_cfg: RunConfig, code_cfg: CodeConfig, model_path: str) -> Dict[str, Any]:
    rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    fail_frames = 0
    total_frames = 0
    dec_cfg = DecoderConfig(max_iters=int(run_cfg.stage1_iters), alpha=float(run_cfg.alpha), early_stop=True)
    snapshot_iters = sorted(set([max(1, run_cfg.stage1_iters - 2), max(1, run_cfg.stage1_iters - 1), run_cfg.stage1_iters]))
    info_pos = np.arange(code_cfg.K, dtype=np.int32)
    for snr_db in run_cfg.calib_snr_db:
        frame = 0
        while total_frames < int(run_cfg.calib.max_frames) and fail_frames < int(run_cfg.calib.target_failed_frames):
            seed = _stable_seed(run_cfg.base_seed, "calib", run_cfg.stage1_iters, snr_db, frame)
            llr = generate_frame_llr(code_cfg, float(snr_db), seed)
            hard, llr_post, synd, _, snaps = ldpc_min_sum_decode(llr, code_cfg, dec_cfg, snapshot_iters=snapshot_iters)
            total_frames += 1
            if int(np.sum(synd)) != 0:
                feat = _feature_pack(code_cfg, llr_post, hard, synd, snaps, run_cfg.stage1_iters)
                labels = hard.copy().astype(np.uint8)
                packed = _collect_training_rows(feat, labels, llr_post, synd, run_cfg.calib, seed + 7)
                if packed is not None:
                    rows.append(packed)
                    fail_frames += 1
            frame += 1
            if fail_frames >= int(run_cfg.calib.target_failed_frames):
                break
        if fail_frames >= int(run_cfg.calib.target_failed_frames):
            break
    student = train_ai_ranker(rows, run_cfg.calib, seed=run_cfg.base_seed + 701)
    if student is not None:
        save_student(model_path, student)
    return {
        "calib_total_frames": int(total_frames),
        "calib_failed_frames": int(fail_frames),
        "model_ready": bool(student is not None),
        "model_path": model_path,
    }


# ------------------------- Evaluation -------------------------
def _info_errors(bits: np.ndarray, code_cfg: CodeConfig) -> int:
    return int(np.sum(bits[: code_cfg.K]))


def evaluate_one_snr(run_cfg: RunConfig, code_cfg: CodeConfig, snr_db: float, student: Optional[DistilledLinearStudent]) -> Dict[str, Any]:
    dec_cfg = DecoderConfig(max_iters=int(run_cfg.stage1_iters), alpha=float(run_cfg.alpha), early_stop=True)
    snapshot_iters = sorted(set([max(1, run_cfg.stage1_iters - 2), max(1, run_cfg.stage1_iters - 1), run_cfg.stage1_iters]))

    frames = 0
    ldpc_bit_errors = 0
    hyb_bit_errors = 0
    ldpc_frame_errors = 0
    hyb_frame_errors = 0
    stage1_iter_sum = 0
    grand_invocations = 0
    grand_rescues = 0
    grand_patterns_total = 0
    stage1_time = 0.0
    grand_time = 0.0

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
            grand_invocations += 1
            feat = _feature_pack(code_cfg, llr_post, hard, synd, snaps, run_cfg.stage1_iters)
            prob = student_prob(student, feat)
            t1 = time.perf_counter()
            res = ai_guided_grand(code_cfg, hard, llr_post, synd, prob, feat, run_cfg.hybrid)
            grand_time += time.perf_counter() - t1
            grand_patterns_total += int(res["patterns_tested"])
            if res["success"]:
                final_bits = res["bits"]
                grand_rescues += 1
        e_hyb = _info_errors(final_bits, code_cfg)
        hyb_bit_errors += e_hyb
        hyb_synd = compute_syndrome(final_bits.astype(np.uint8), code_cfg)
        hyb_frame_errors += int(e_hyb > 0 or int(np.sum(hyb_synd)) != 0)

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
        "grand_invocation_rate": float(grand_invocations / frames) if frames > 0 else 0.0,
        "grand_rescue_rate_given_invoked": float(grand_rescues / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_grand_patterns_per_failed_frame": float(grand_patterns_total / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_grand_patterns_per_frame": float(grand_patterns_total / frames) if frames > 0 else 0.0,
        "avg_stage1_decoder_us": float(1e6 * stage1_time / frames) if frames > 0 else 0.0,
        "avg_grand_decoder_us": float(1e6 * grand_time / frames) if frames > 0 else 0.0,
        "avg_total_hybrid_decoder_us": float(1e6 * (stage1_time + grand_time) / frames) if frames > 0 else 0.0,
    }


# ------------------------- Save helpers -------------------------
def save_summary_csv(rows: List[Dict[str, Any]], path: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            vals = []
            for k in keys:
                v = row[k]
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
    eval_snr_db = _env_csv_floats("EVAL_SNR_SWEEP", "10.5,11.0,11.5,12.0")
    calib_snr_db = _env_csv_floats("CALIB_SNR_SWEEP", ",".join(str(x) for x in eval_snr_db[: max(2, min(4, len(eval_snr_db)))]))
    mc = MCConfig(
        target_frame_errors=_env_int("TARGET_FRAME_ERRORS", 150),
        max_frames=_env_int("MAX_FRAMES", 120000),
        min_frames=_env_int("MIN_FRAMES", 0),
    )
    calib = CalibrationConfig(
        target_failed_frames=_env_int("CALIB_TARGET_FAILED_FRAMES", 96),
        max_frames=_env_int("CALIB_MAX_FRAMES", 50000),
        neg_ratio=_env_float("CALIB_NEG_RATIO", 6.0),
        hard_negative_cap=_env_int("CALIB_HARD_NEG_CAP", 64),
        teacher_hidden=_env_int("AI_TEACHER_HIDDEN", 32),
        teacher_epochs=_env_int("AI_TEACHER_EPOCHS", 18),
        student_epochs=_env_int("AI_STUDENT_EPOCHS", 16),
        batch_size=_env_int("AI_BATCH_SIZE", 4096),
        teacher_lr=_env_float("AI_TEACHER_LR", 0.01),
        student_lr=_env_float("AI_STUDENT_LR", 0.025),
        temperature=_env_float("AI_DISTILL_TEMP", 2.5),
    )
    hybrid = HybridConfig(
        global_pool=_env_int("GRAND_GLOBAL_POOL", 14),
        llr_pool=_env_int("GRAND_LLR_POOL", 14),
        gain_pool=_env_int("GRAND_GAIN_POOL", 14),
        osc_pool=_env_int("GRAND_OSC_POOL", 10),
        union_cap=_env_int("GRAND_UNION_CAP", 24),
        roi_local_cap=_env_int("GRAND_ROI_LOCAL_CAP", 10),
        max_rois=_env_int("GRAND_MAX_ROIS", 4),
        tiny_budget=_env_int("GRAND_TINY_BUDGET", 32),
        mid_budget=_env_int("GRAND_MID_BUDGET", 128),
        full_budget=_env_int("GRAND_FULL_BUDGET", 384),
        tiny_wmax=_env_int("GRAND_TINY_WMAX", 2),
        mid_wmax=_env_int("GRAND_MID_WMAX", 3),
        full_wmax=_env_int("GRAND_FULL_WMAX", 4),
        ai_weight=_env_float("GRAND_AI_WEIGHT", 1.35),
        gain_weight=_env_float("GRAND_GAIN_WEIGHT", 0.22),
        osc_weight=_env_float("GRAND_OSC_WEIGHT", 0.16),
        compact_weight=_env_float("GRAND_COMPACT_WEIGHT", 0.10),
        llr_weight=_env_float("GRAND_LLR_WEIGHT", 1.0),
    )
    base_seed = _env_int("RNG_SEED_GLOBAL", 20260402)
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
    code_cfg = build_sionna_5g_nr_code_cfg(run_cfg.k_info, run_cfg.n_tx, run_cfg.qm)
    _warmup(code_cfg, run_cfg.stage1_iters)

    model_path = os.path.join(results_dir, f"tgds_student_it{run_cfg.stage1_iters:02d}.npz")
    calib_meta = run_calibration(run_cfg, code_cfg, model_path)
    student = load_student(model_path) if calib_meta["model_ready"] else None

    rows = []
    print(f"[RUN] code={code_cfg.code_name} it={run_cfg.stage1_iters} eval_snr={run_cfg.eval_snr_db}")
    print(f"[RUN] calib={calib_meta}")
    for snr_db in run_cfg.eval_snr_db:
        res = evaluate_one_snr(run_cfg, code_cfg, float(snr_db), student)
        rows.append(res)
        print("\n=== TGDS HYBRID EVAL ===")
        print(f"SNR (dB)                    : {res['snr_db']:.2f}")
        print(f"Frames simulated            : {res['frames']}")
        print(f"LDPC FER / BER              : {res['ldpc_fer']:.6e} / {res['ldpc_ber']:.6e}")
        print(f"Hybrid FER / BER            : {res['hyb_fer']:.6e} / {res['hyb_ber']:.6e}")
        print(f"Avg stage-1 iters/frame     : {res['avg_stage1_iters']:.3f}")
        print(f"GRAND invocation rate       : {res['grand_invocation_rate']:.6f}")
        print(f"GRAND rescue rate | invoked : {res['grand_rescue_rate_given_invoked']:.6f}")
        print(f"Avg GRAND patterns/failure  : {res['avg_grand_patterns_per_failed_frame']:.3f}")
        print(f"Avg stage-1 decoder us      : {res['avg_stage1_decoder_us']:.3f}")
        print(f"Avg GRAND decoder us        : {res['avg_grand_decoder_us']:.3f}")
        print(f"Avg total hybrid decoder us : {res['avg_total_hybrid_decoder_us']:.3f}")

    prefix = f"tgds_it{run_cfg.stage1_iters:02d}_{_now_tag()}"
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
                    "SIONNA_TDL_MODEL": os.getenv("SIONNA_TDL_MODEL", "C"),
                    "SIONNA_TDL_DELAY_SPREAD_S": os.getenv("SIONNA_TDL_DELAY_SPREAD_S", "3e-7"),
                    "SIONNA_TDL_MIN_SPEED": os.getenv("SIONNA_TDL_MIN_SPEED", "5.0"),
                    "SIONNA_TDL_MAX_SPEED": os.getenv("SIONNA_TDL_MAX_SPEED", "20.0"),
                    "SIONNA_CFO_HZ": os.getenv("SIONNA_CFO_HZ", "0.0"),
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
    print(f"[SAVE] model={model_path if calib_meta['model_ready'] else 'not-created'}")


if __name__ == "__main__":
    main()
