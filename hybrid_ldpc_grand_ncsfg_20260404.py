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
    tx_info_pool: int = 18
    punctured_info_pool: int = 10
    tx_parity_pool: int = 8
    round2_extra_info: int = 6
    round2_extra_punctured: int = 4
    round2_extra_parity: int = 4
    max_weight: int = 8
    free_dim_cap: int = 10
    max_patterns: int = 1024
    frame_gate_threshold: float = 0.28
    gate_max_synd: int = 18
    gate_max_hard_ones: int = 28
    ai_weight: float = 1.50
    gain_weight: float = 0.24
    osc_weight: float = 0.18
    llr_weight: float = 1.0
    sw_weight: float = 1.20
    info_bonus: float = 0.70
    punctured_info_bonus: float = 0.40
    tx_bonus: float = 0.16
    parity_penalty: float = 0.15

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
    hard_mask = hard_final.astype(bool)
    searchable = hard_mask & (~code_cfg._filler_mask.astype(bool))
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


def ai_guided_grand(
    code_cfg: CodeConfig,
    hard_final: np.ndarray,
    llr_final: np.ndarray,
    syndrome_final: np.ndarray,
    ai_prob_vec: np.ndarray,
    feat: np.ndarray,
    cfg: HybridConfig,
):
    info_ranked, punct_ranked, tx_par_ranked, base_score = _domain_ranked_lists(
        code_cfg, hard_final, llr_final, syndrome_final, ai_prob_vec, feat, cfg
    )
    info_1 = info_ranked[: int(cfg.tx_info_pool)]
    punc_1 = punct_ranked[: int(cfg.punctured_info_pool)]
    par_1 = tx_par_ranked[: int(cfg.tx_parity_pool)]
    info_2 = info_ranked[: int(cfg.tx_info_pool + cfg.round2_extra_info)]
    punc_2 = punct_ranked[: int(cfg.punctured_info_pool + cfg.round2_extra_punctured)]
    par_2 = tx_par_ranked[: int(cfg.tx_parity_pool + cfg.round2_extra_parity)]

    rounds = [
        ("info_exact", _pool_unique(info_1, punc_1), True, int(cfg.max_weight)),
        ("mixed_exact", _pool_unique(info_2, punc_2, par_1), True, int(cfg.max_weight)),
        ("parity_cleanup", _pool_unique(info_2, punc_2, par_2), False, int(cfg.max_weight)),
    ]
    last = {
        "success": False,
        "bits": hard_final,
        "pattern": tuple(),
        "round_name": "none",
        "pool_size": 0,
        "rank": 0,
        "free_dim": 0,
        "solutions_tested": 0,
    }
    total_tested = 0
    total_pool = 0
    max_free_dim_seen = 0
    success_round = "none"
    for round_name, pool, require_info, max_weight in rounds:
        res = _solve_exact_on_pool(
            round_name,
            code_cfg,
            hard_final,
            llr_final,
            syndrome_final,
            ai_prob_vec,
            pool,
            cfg,
            require_info=require_info,
            max_weight=max_weight,
            free_dim_cap=int(cfg.free_dim_cap),
            max_candidates=int(cfg.max_patterns),
        )
        total_tested += int(res.get("solutions_tested", 0))
        total_pool += int(res.get("pool_size", 0))
        max_free_dim_seen = max(max_free_dim_seen, int(res.get("free_dim", 0)))
        last = res
        if res.get("success", False):
            success_round = str(round_name)
            break
    last.update(
        {
            "success_round": success_round,
            "candidates_tested": int(total_tested),
            "aggregate_pool_size": int(total_pool),
            "max_free_dim_seen": int(max_free_dim_seen),
            "pool_info": int(len(info_2)),
            "pool_punctured_info": int(len(punc_2)),
            "pool_tx_parity": int(len(par_2)),
            "base_score": base_score,
        }
    )
    return last
# ------------------------- Calibration -------------------------
def _warmup(code_cfg: CodeConfig, stage1_iters: int):
    llr = np.ones(code_cfg.N, dtype=np.float32)
    llr[: min(16, code_cfg.N)] = -0.5
    dec_cfg = DecoderConfig(max_iters=int(stage1_iters), alpha=float(_env_float("LDPC_ALPHA", 0.80)), early_stop=True)
    snapshot_iters = sorted(set([max(1, stage1_iters - 2), max(1, stage1_iters - 1), stage1_iters]))
    hard, llr_post, synd, _, snaps = ldpc_min_sum_decode(llr, code_cfg, dec_cfg, snapshot_iters=snapshot_iters)
    feat = _feature_pack(code_cfg, llr_post, hard, synd, snaps, int(stage1_iters))
    heur_prob = student_prob(None, feat)
    _ = ai_guided_grand(code_cfg, hard, llr_post, synd, heur_prob, feat, HybridConfig())


def _collect_bit_training_rows_from_pattern(
    feat: np.ndarray,
    pattern: Tuple[int, ...],
    llr_final: np.ndarray,
    cfg: CalibrationConfig,
    seed: int,
):
    rng = np.random.default_rng(int(seed))
    labels = np.zeros(feat.shape[0], dtype=np.float32)
    if len(pattern) == 0:
        return None
    labels[np.asarray(pattern, dtype=np.int32)] = 1.0
    return _collect_training_rows(feat, labels, llr_final, np.zeros((1,), dtype=np.uint8), cfg, seed)


def _heuristic_base_score(code_cfg: CodeConfig, llr_final: np.ndarray, feat: np.ndarray) -> np.ndarray:
    return _bit_search_base_score(code_cfg, llr_final, student_prob(None, feat), feat, HybridConfig())


def run_calibration(run_cfg: RunConfig, code_cfg: CodeConfig, bit_model_path: str, gate_model_path: str) -> Dict[str, Any]:
    bit_rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    frame_rows: List[Tuple[np.ndarray, int]] = []
    fail_frames = 0
    pos_frames = 0
    total_frames = 0
    dec_cfg = DecoderConfig(max_iters=int(run_cfg.stage1_iters), alpha=float(run_cfg.alpha), early_stop=True)
    snapshot_iters = sorted(set([max(1, run_cfg.stage1_iters - 2), max(1, run_cfg.stage1_iters - 1), run_cfg.stage1_iters]))
    search_cfg = HybridConfig(
        tx_info_pool=int(run_cfg.calib.calib_tx_info_pool),
        punctured_info_pool=int(run_cfg.calib.calib_punctured_info_pool),
        tx_parity_pool=int(run_cfg.calib.calib_tx_parity_pool),
        round2_extra_info=0,
        round2_extra_punctured=0,
        round2_extra_parity=0,
        max_weight=int(run_cfg.calib.calib_weight_cap),
        free_dim_cap=int(run_cfg.calib.calib_free_dim_cap),
        max_patterns=int(run_cfg.calib.calib_max_candidates),
        frame_gate_threshold=0.0,
        gate_max_synd=999,
        gate_max_hard_ones=999,
        ai_weight=float(run_cfg.hybrid.ai_weight),
        gain_weight=float(run_cfg.hybrid.gain_weight),
        osc_weight=float(run_cfg.hybrid.osc_weight),
        llr_weight=float(run_cfg.hybrid.llr_weight),
        sw_weight=float(run_cfg.hybrid.sw_weight),
        info_bonus=float(run_cfg.hybrid.info_bonus),
        punctured_info_bonus=float(run_cfg.hybrid.punctured_info_bonus),
        tx_bonus=float(run_cfg.hybrid.tx_bonus),
        parity_penalty=float(run_cfg.hybrid.parity_penalty),
    )
    for snr_db in run_cfg.calib_snr_db:
        frame = 0
        while total_frames < int(run_cfg.calib.max_frames) and fail_frames < int(run_cfg.calib.target_failed_frames):
            seed = _stable_seed(run_cfg.base_seed, "calib", run_cfg.stage1_iters, snr_db, frame)
            llr = generate_frame_llr(code_cfg, float(snr_db), seed)
            hard, llr_post, synd, _, snaps = ldpc_min_sum_decode(llr, code_cfg, dec_cfg, snapshot_iters=snapshot_iters)
            total_frames += 1
            if int(np.sum(synd)) != 0:
                feat = _feature_pack(code_cfg, llr_post, hard, synd, snaps, run_cfg.stage1_iters)
                heur_prob = student_prob(None, feat)
                res = ai_guided_grand(code_cfg, hard, llr_post, synd, heur_prob, feat, search_cfg)
                frame_feat = _frame_feature_vec(code_cfg, hard, llr_post, synd, feat, res["base_score"])
                label = int(bool(res.get("success", False)))
                frame_rows.append((frame_feat, label))
                fail_frames += 1
                if label:
                    packed = _collect_bit_training_rows_from_pattern(feat, tuple(res["pattern"]), llr_post, run_cfg.calib, seed + 7)
                    if packed is not None:
                        bit_rows.append(packed)
                    pos_frames += 1
            frame += 1
            if fail_frames >= int(run_cfg.calib.target_failed_frames):
                break
        if fail_frames >= int(run_cfg.calib.target_failed_frames):
            break
    bit_student = train_ai_ranker(bit_rows, run_cfg.calib, seed=run_cfg.base_seed + 701)
    gate_student = train_frame_gate(frame_rows, run_cfg.calib, seed=run_cfg.base_seed + 1701)
    if bit_student is not None:
        save_student(bit_model_path, bit_student)
    if gate_student is not None:
        save_gate_student(gate_model_path, gate_student)
    return {
        "calib_total_frames": int(total_frames),
        "calib_failed_frames": int(fail_frames),
        "calib_positive_frames": int(pos_frames),
        "bit_model_ready": bool(bit_student is not None),
        "gate_model_ready": bool(gate_student is not None),
        "bit_model_path": bit_model_path,
        "gate_model_path": gate_model_path,
    }


# ------------------------- Evaluation -------------------------
def _info_errors(bits: np.ndarray, code_cfg: CodeConfig) -> int:
    return int(np.sum(bits[: code_cfg.K]))


def evaluate_one_snr(
    run_cfg: RunConfig,
    code_cfg: CodeConfig,
    snr_db: float,
    bit_student: Optional[DistilledLinearStudent],
    gate_student: Optional[DistilledLinearStudent],
) -> Dict[str, Any]:
    dec_cfg = DecoderConfig(max_iters=int(run_cfg.stage1_iters), alpha=float(run_cfg.alpha), early_stop=True)
    snapshot_iters = sorted(set([max(1, run_cfg.stage1_iters - 2), max(1, run_cfg.stage1_iters - 1), run_cfg.stage1_iters]))

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
    info_round_successes = 0
    parity_round_successes = 0
    gate_positive = 0
    gate_prob_sum = 0.0
    candidate_pool_total = 0
    max_free_dim_total = 0
    solver_candidates_total = 0
    pool_info_total = 0
    pool_tx_parity_total = 0
    pool_punctured_info_total = 0
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
            failed_frames += 1
            feat = _feature_pack(code_cfg, llr_post, hard, synd, snaps, run_cfg.stage1_iters)
            bit_prob = student_prob(bit_student, feat)
            _, _, _, base_score = _domain_ranked_lists(code_cfg, hard, llr_post, synd, bit_prob, feat, run_cfg.hybrid)
            frame_feat = _frame_feature_vec(code_cfg, hard, llr_post, synd, feat, base_score)
            gate_p = frame_gate_prob(gate_student, frame_feat)
            gate_prob_sum += float(gate_p)

            hard_ones = int(np.sum(hard))
            near_codeword_ok = (int(np.sum(synd)) <= int(run_cfg.hybrid.gate_max_synd)) and (hard_ones <= int(run_cfg.hybrid.gate_max_hard_ones))
            if gate_p >= float(run_cfg.hybrid.frame_gate_threshold) and near_codeword_ok:
                gate_positive += 1
                grand_invocations += 1
                t1 = time.perf_counter()
                res = ai_guided_grand(code_cfg, hard, llr_post, synd, bit_prob, feat, run_cfg.hybrid)
                grand_time += time.perf_counter() - t1
                solver_candidates_total += int(res.get("candidates_tested", 0))
                candidate_pool_total += int(res.get("aggregate_pool_size", 0))
                max_free_dim_total += int(res.get("max_free_dim_seen", 0))
                pool_info_total += int(res.get("pool_info", 0))
                pool_tx_parity_total += int(res.get("pool_tx_parity", 0))
                pool_punctured_info_total += int(res.get("pool_punctured_info", 0))
                if res.get("success", False):
                    final_bits = res["bits"]
                    grand_rescues += 1
                    if str(res.get("success_round", "")) in ("info_exact", "mixed_exact"):
                        info_round_successes += 1
                    elif str(res.get("success_round", "")) == "parity_cleanup":
                        parity_round_successes += 1

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
        "gate_positive_rate_given_failed": float(gate_positive / failed_frames) if failed_frames > 0 else 0.0,
        "avg_gate_prob_given_failed": float(gate_prob_sum / failed_frames) if failed_frames > 0 else 0.0,
        "grand_rescue_rate_given_invoked": float(grand_rescues / grand_invocations) if grand_invocations > 0 else 0.0,
        "grand_info_rescue_rate_given_invoked": float(grand_info_rescues / grand_invocations) if grand_invocations > 0 else 0.0,
        "grand_parity_only_rescue_rate_given_invoked": float(grand_parity_only_rescues / grand_invocations) if grand_invocations > 0 else 0.0,
        "grand_info_round_success_rate_given_invoked": float(info_round_successes / grand_invocations) if grand_invocations > 0 else 0.0,
        "grand_parity_round_success_rate_given_invoked": float(parity_round_successes / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_solver_candidates_per_invoked": float(solver_candidates_total / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_candidate_pool_per_invoked": float(candidate_pool_total / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_max_free_dim_per_invoked": float(max_free_dim_total / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_info_pool_per_invoked": float(pool_info_total / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_tx_parity_pool_per_invoked": float(pool_tx_parity_total / grand_invocations) if grand_invocations > 0 else 0.0,
        "avg_punctured_info_pool_per_invoked": float(pool_punctured_info_total / grand_invocations) if grand_invocations > 0 else 0.0,
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
    eval_snr_db = _env_csv_floats("EVAL_SNR_SWEEP", "8.5,9.0,9.5,10.0")
    calib_snr_db = _env_csv_floats("CALIB_SNR_SWEEP", ",".join(str(x) for x in eval_snr_db[: max(3, min(5, len(eval_snr_db)))]))
    mc = MCConfig(
        target_frame_errors=_env_int("TARGET_FRAME_ERRORS", 120),
        max_frames=_env_int("MAX_FRAMES", 150000),
        min_frames=_env_int("MIN_FRAMES", 0),
    )
    calib = CalibrationConfig(
        target_failed_frames=_env_int("CALIB_TARGET_FAILED_FRAMES", 384),
        max_frames=_env_int("CALIB_MAX_FRAMES", 150000),
        neg_ratio=_env_float("CALIB_NEG_RATIO", 6.0),
        hard_negative_cap=_env_int("CALIB_HARD_NEG_CAP", 128),
        teacher_hidden=_env_int("AI_TEACHER_HIDDEN", 64),
        teacher_epochs=_env_int("AI_TEACHER_EPOCHS", 24),
        student_epochs=_env_int("AI_STUDENT_EPOCHS", 20),
        batch_size=_env_int("AI_BATCH_SIZE", 4096),
        teacher_lr=_env_float("AI_TEACHER_LR", 0.01),
        student_lr=_env_float("AI_STUDENT_LR", 0.022),
        temperature=_env_float("AI_DISTILL_TEMP", 2.5),
        calib_tx_info_pool=_env_int("CALIB_TX_INFO_POOL", 26),
        calib_punctured_info_pool=_env_int("CALIB_PUNCTURED_INFO_POOL", 14),
        calib_tx_parity_pool=_env_int("CALIB_TX_PARITY_POOL", 12),
        calib_weight_cap=_env_int("CALIB_WEIGHT_CAP", 10),
        calib_free_dim_cap=_env_int("CALIB_FREE_DIM_CAP", 12),
        calib_max_candidates=_env_int("CALIB_MAX_CANDIDATES", 4096),
    )
    hybrid = HybridConfig(
        tx_info_pool=_env_int("GRAND_TX_INFO_POOL", 18),
        tx_parity_pool=_env_int("GRAND_TX_PARITY_POOL", 8),
        punctured_info_pool=_env_int("GRAND_PUNCTURED_INFO_POOL", 10),
        round2_extra_info=_env_int("GRAND_ROUND2_EXTRA_INFO", 6),
        round2_extra_punctured=_env_int("GRAND_ROUND2_EXTRA_PUNCTURED", 4),
        round2_extra_parity=_env_int("GRAND_ROUND2_EXTRA_PARITY", 4),
        max_weight=_env_int("GRAND_MAX_WEIGHT", 8),
        free_dim_cap=_env_int("GRAND_FREE_DIM_CAP", 10),
        max_patterns=_env_int("GRAND_MAX_PATTERNS", 1024),
        frame_gate_threshold=_env_float("GRAND_FRAME_GATE_THRESHOLD", 0.28),
        gate_max_synd=_env_int("GRAND_GATE_MAX_SYND", 18),
        gate_max_hard_ones=_env_int("GRAND_GATE_MAX_HARD_ONES", 28),
        ai_weight=_env_float("GRAND_AI_WEIGHT", 1.50),
        gain_weight=_env_float("GRAND_GAIN_WEIGHT", 0.24),
        osc_weight=_env_float("GRAND_OSC_WEIGHT", 0.18),
        llr_weight=_env_float("GRAND_LLR_WEIGHT", 1.0),
        sw_weight=_env_float("GRAND_SW_WEIGHT", 1.20),
        info_bonus=_env_float("GRAND_INFO_BONUS", 0.70),
        tx_bonus=_env_float("GRAND_TX_BONUS", 0.16),
        punctured_info_bonus=_env_float("GRAND_PUNCTURED_INFO_BONUS", 0.40),
        parity_penalty=_env_float("GRAND_PARITY_PENALTY", 0.15),
    )
    base_seed = _env_int("RNG_SEED_GLOBAL", 20260404)
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

    bit_model_path = os.path.join(results_dir, f"ncsfg_bit_student_it{run_cfg.stage1_iters:02d}.npz")
    gate_model_path = os.path.join(results_dir, f"ncsfg_gate_student_it{run_cfg.stage1_iters:02d}.npz")
    calib_meta = run_calibration(run_cfg, code_cfg, bit_model_path, gate_model_path)
    bit_student = load_student(bit_model_path) if calib_meta["bit_model_ready"] else None
    gate_student = load_gate_student(gate_model_path) if calib_meta["gate_model_ready"] else None

    rows = []
    print(f"[RUN] code={code_cfg.code_name} it={run_cfg.stage1_iters} eval_snr={run_cfg.eval_snr_db}")
    print(f"[RUN] calib={calib_meta}")
    for snr_db in run_cfg.eval_snr_db:
        res = evaluate_one_snr(run_cfg, code_cfg, float(snr_db), bit_student, gate_student)
        rows.append(res)
        print("\n=== NCSFG HYBRID EVAL ===")
        print(f"SNR (dB)                         : {res['snr_db']:.2f}")
        print(f"Frames simulated                 : {res['frames']}")
        print(f"LDPC FER / BER                   : {res['ldpc_fer']:.6e} / {res['ldpc_ber']:.6e}")
        print(f"Hybrid FER / BER                 : {res['hyb_fer']:.6e} / {res['hyb_ber']:.6e}")
        print(f"Avg stage-1 iters/frame          : {res['avg_stage1_iters']:.3f}")
        print(f"Failed frame rate                : {res['failed_frame_rate']:.6f}")
        print(f"Gate positive | failed          : {res['gate_positive_rate_given_failed']:.6f}")
        print(f"Avg gate prob | failed          : {res['avg_gate_prob_given_failed']:.6f}")
        print(f"GRAND invocation rate            : {res['grand_invocation_rate']:.6f}")
        print(f"GRAND rescue rate | invoked      : {res['grand_rescue_rate_given_invoked']:.6f}")
        print(f"Info rescue rate | invoked       : {res['grand_info_rescue_rate_given_invoked']:.6f}")
        print(f"Parity-only rescue | invoked     : {res['grand_parity_only_rescue_rate_given_invoked']:.6f}")
        print(f"Info / parity round success      : {res['grand_info_round_success_rate_given_invoked']:.6f} / {res['grand_parity_round_success_rate_given_invoked']:.6f}")
        print(f"Pools info/punc/txp | invoked    : {res['avg_info_pool_per_invoked']:.2f} / {res['avg_punctured_info_pool_per_invoked']:.2f} / {res['avg_tx_parity_pool_per_invoked']:.2f}")
        print(f"Avg candidate pool | invoked     : {res['avg_candidate_pool_per_invoked']:.3f}")
        print(f"Avg solver candidates | invoked  : {res['avg_solver_candidates_per_invoked']:.3f}")
        print(f"Avg max free dim | invoked       : {res['avg_max_free_dim_per_invoked']:.3f}")
        print(f"Avg stage-1 decoder us           : {res['avg_stage1_decoder_us']:.3f}")
        print(f"Avg GRAND decoder us             : {res['avg_grand_decoder_us']:.3f}")
        print(f"Avg total hybrid decoder us      : {res['avg_total_hybrid_decoder_us']:.3f}")

    prefix = f"ncsfg_it{run_cfg.stage1_iters:02d}_{_now_tag()}"
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
                    "SIONNA_TDL_DELAY_SPREAD_S": os.getenv("SIONNA_TDL_DELAY_SPREAD_S", "1e-7"),
                    "SIONNA_TDL_MIN_SPEED": os.getenv("SIONNA_TDL_MIN_SPEED", "1.0"),
                    "SIONNA_TDL_MAX_SPEED": os.getenv("SIONNA_TDL_MAX_SPEED", "3.0"),
                    "SIONNA_CFO_HZ": os.getenv("SIONNA_CFO_HZ", "15.0"),
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
    print(f"[SAVE] gate_model={gate_model_path if calib_meta['gate_model_ready'] else 'not-created'}")


if __name__ == "__main__":
    main()
