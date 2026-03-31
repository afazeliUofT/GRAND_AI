##Update##
### CELL number 1 ###
import os

try:
    from numba import set_num_threads, get_num_threads
    _NUMBA_THREADING_AVAILABLE = True
except ImportError:
    set_num_threads = None
    get_num_threads = None
    _NUMBA_THREADING_AVAILABLE = False


def _detect_num_threads():
    """
    Detect a reasonable number of CPU threads to use on this node.

    Priority (from most to least authoritative):

      0) LDPC_GRAND_NUM_THREADS  (explicit override for this script)
      1) SLURM_CPUS_PER_TASK     (set by Slurm when you use --cpus-per-task)
      2) NUMBA_NUM_THREADS       (Numba's own override if you set it)
      3) multiprocessing.cpu_count()  (what the OS reports as available)
      4) OMP_NUM_THREADS         (used only as a last resort)

    Rationale:
      - On Narval your job stats show ~11× wall‑clock CPU usage over 3.25 h
        (≈17% of 64 cores), which strongly suggests that OMP_NUM_THREADS is
        limiting Numba to ~11 threads even though you requested 64 cores.
      - By making OMP_NUM_THREADS the *last* fallback, and preferring Slurm /
        Numba / cpu_count, we let the job actually use the full core allocation.
    """

    # 0) Explicit override for this script
    env_val = os.environ.get("LDPC_GRAND_NUM_THREADS")
    if env_val:
        try:
            n = int(env_val)
            if n > 0:
                return n
        except ValueError:
            pass  # fall through to the other heuristics

    # 1) Slurm hint: CPUs per task
    env_val = os.environ.get("SLURM_CPUS_PER_TASK")
    if env_val:
        try:
            n = int(env_val)
            if n > 0:
                return n
        except ValueError:
            pass

    # 2) Numba-specific override
    env_val = os.environ.get("NUMBA_NUM_THREADS")
    if env_val:
        try:
            n = int(env_val)
            if n > 0:
                return n
        except ValueError:
            pass

    # 3) What the OS / cgroup says is available to this process
    try:
        import multiprocessing
        n = multiprocessing.cpu_count()
        if n > 0:
            return n
    except Exception:
        pass

    # 4) As a *last resort*, honour OMP_NUM_THREADS
    env_val = os.environ.get("OMP_NUM_THREADS")
    if env_val:
        try:
            n = int(env_val)
            if n > 0:
                return n
        except ValueError:
            pass

    # Fallback if everything else fails
    return 1


# Global thread budget seen by the rest of the script
NUMBA_THREADS = _detect_num_threads()

if _NUMBA_THREADING_AVAILABLE:
    try:
        set_num_threads(NUMBA_THREADS)
    except Exception:
        pass

    try:
        current = get_num_threads()
    except Exception:
        current = NUMBA_THREADS

    print(f"Numba threads: {current}")
else:
    print("Numba not available; using default threading.")



# %%
### CELL number 2 ###
import sys
import platform
import datetime
import os
import time  # NEW: for latency measurements
import copy

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from functools import partial

import numpy as np

# Numba for JIT acceleration
try:
    import numba as nb
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    nb = None
    njit = None
    prange = range
    NUMBA_AVAILABLE = False

# Joblib for potential multiprocessing (kept for compatibility)
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    Parallel = None
    delayed = None
    JOBLIB_AVAILABLE = False

# TensorFlow / Sionna (optional; required for 5G NR LDPC + 3GPP channels)
# Keep this import logic tolerant to both legacy TensorFlow-based Sionna 1.x and
# current PyTorch-based Sionna 2.x. We intentionally do *not* gate Sionna support
# on a top-level tensorflow import because Sionna 2.x removed TensorFlow as a
# dependency.
if os.getenv("USE_GPU", "0").lower() not in ("1", "true", "yes"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.getenv("TF_CPP_MIN_LOG_LEVEL", "2"))

tf = None  # type: ignore
if os.getenv("ENABLE_TF_RUNTIME_CONFIG", "0").lower() in ("1", "true", "yes"):
    try:
        import tensorflow as tf  # optional; only used for thread/GPU config on legacy stacks
    except Exception:
        tf = None  # type: ignore


def _import_symbol_candidates(symbol_name: str, candidate_paths: List[str]):
    """Try importing ``symbol_name`` from a list of module paths.

    Returns ``(symbol_or_None, debug_message)`` where ``debug_message`` contains
    the exact failure chain if no candidate succeeded.
    """
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

# Avoid oversubscribing threads when joblib/numba is used for parallel sweeps
if tf is not None:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(int(os.getenv("TF_INTRA_OP", "1")))
        tf.config.threading.set_inter_op_parallelism_threads(int(os.getenv("TF_INTER_OP", "1")))
    except Exception:
        pass

    # Default: CPU-only unless explicitly enabled
    if os.getenv("USE_GPU", "0").lower() not in ("1", "true", "yes"):
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass


# %%
### CELL number 3 ###
@dataclass
class CodeConfig:
    """Static information about the code itself."""
    code_name: str
    N: int          # codeword length
    K: int          # information length
    rate: float
    H_path: Optional[str] = None  # where parity-check matrix will live

    # Tanner-graph neighbourhoods for checks and variables
    checks_to_vars: Optional[List[np.ndarray]] = field(default=None, repr=False)
    vars_to_checks: Optional[List[np.ndarray]] = field(default=None, repr=False)

    # NEW: for each variable, position of that variable inside each neighbouring check
    # var_to_checks_edge_pos[v][k] = local edge index e such that
    #   checks_to_vars[ vars_to_checks[v][k] ][ e ] == v
    var_to_checks_edge_pos: Optional[List[np.ndarray]] = field(default=None, repr=False)


@dataclass
class InterleaverConfig:
    """Bit-level interleaver/de-interleaver description."""
    name: str                   # e.g. "identity", "random"
    pattern: np.ndarray         # permutation of [0..N-1]
    inverse_pattern: np.ndarray # inverse permutation


@dataclass
class ChannelConfig:
    """Channel model configuration."""
    name: str     # "SIONNA_TDL"
    snr_db: float




# %%
### CELL number 4 ###
@dataclass
class SimulationConfig:
    """Global config for a simulation run."""
    code: CodeConfig
    channel: ChannelConfig
    interleaver: InterleaverConfig
    rng_seed_global: int
    # Minor suggestion applied: snapshot iterations prepared from day one
    snapshot_iters: List[int] = field(default_factory=lambda: [4, 8, 12, 40])


@dataclass
class FrameLog:
    """
    Per-frame log. 
    We define all fields we know we’ll need later so we don't have to change this structure.
    """
    frame_id: int
    rng_seed_frame: int

    # --- Encoder / input ---
    u_bits: np.ndarray                  # length K
    c_bits: np.ndarray                  # length N (pre-interleaver)
    interleaver_pattern: np.ndarray     # length N
    deinterleaver_pattern: np.ndarray   # length N

    # --- Channel ---
    s_symbols: np.ndarray               # BPSK symbols actually sent (interleaver order)
    y_channel: np.ndarray               # received samples (interleaver order)
    y_received: np.ndarray              # deinterleaved samples (decoder input)
    channel_realization: Dict[str, np.ndarray]

    # Precomputed channel LLRs aligned to the decoding graph (mother code length)
    llr_channel: Optional[np.ndarray] = None

    # --- Decoder outputs (to be filled later) ---

    # --- Decoder outputs (to be filled later) ---
    dec_success: Optional[bool] = None
    iter_used: Optional[int] = None
    hard_bits_final: Optional[np.ndarray] = None
    llr_final: Optional[np.ndarray] = None
    syndrome_final: Optional[np.ndarray] = None

    # Explicit error positions for analysis
    error_positions_final: Optional[np.ndarray] = None

    # NEW: per-iteration snapshots for GRAND experiments
    # snapshots["llr"][it]       -> LLRs at iteration it
    # snapshots["hard_bits"][it] -> hard bits at iteration it
    # snapshots["syndrome"][it]  -> syndrome at iteration it
    snapshots: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)




# %%
### CELL number 5 ###
def create_identity_interleaver(N: int) -> InterleaverConfig:
    """Identity interleaver: useful as a default, and keeps the API general."""
    pattern = np.arange(N, dtype=np.int64)
    inverse_pattern = np.argsort(pattern)  # for identity this is the same array
    return InterleaverConfig(name="identity", pattern=pattern, inverse_pattern=inverse_pattern)


def create_interleaver_from_pattern(pattern: np.ndarray,
                                    name: str = "custom") -> InterleaverConfig:
    """
    Create a general interleaver from an arbitrary permutation of [0..N-1].

    pattern: 1D array of length N containing a permutation of 0..N-1.
    """
    pattern = np.asarray(pattern, dtype=np.int64)
    if pattern.ndim != 1:
        raise ValueError("Interleaver pattern must be a 1D array")

    N = pattern.size
    # Basic sanity: pattern must be a permutation of 0..N-1
    if np.unique(pattern).size != N or pattern.min() != 0 or pattern.max() != N - 1:
        raise ValueError("Interleaver pattern must be a permutation of 0..N-1")

    # Build inverse permutation explicitly
    inverse_pattern = np.empty_like(pattern)
    inverse_pattern[pattern] = np.arange(N, dtype=np.int64)

    return InterleaverConfig(name=name, pattern=pattern, inverse_pattern=inverse_pattern)


def interleave(bits: np.ndarray, ilv: InterleaverConfig) -> np.ndarray:
    return bits[ilv.pattern]


def deinterleave(bits: np.ndarray, ilv: InterleaverConfig) -> np.ndarray:
    return bits[ilv.inverse_pattern]










# -------------------- Sionna 5G NR helpers (rate-matching + 3GPP channels) --------------------
_TDL_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _sionna5g_internal_tx_positions(code_cfg: CodeConfig) -> np.ndarray:
    """Return internal VN indices (mother graph) corresponding to transmitted bits (RV=0-style).

    This matches the simplified Sionna encoder behavior:
      - remove filler bits (shortening)
      - puncture first 2*Z bits
      - take next n_tx bits
      - apply output interleaver (if qm>1)

    We return indices in the *internal/mother* VN ordering (after re-inserting filler bits),
    aligned with the LLR vector fed to the LDPC decoder.
    """
    if not hasattr(code_cfg, "sionna"):
        raise ValueError("code_cfg has no .sionna metadata")
    s = code_cfg.sionna
    z = int(s.get("z", 0))
    n_tx = int(s.get("n_tx"))
    k_filler = int(s.get("k_filler", 0))
    k_info = int(code_cfg.K)

    N_int = int(code_cfg.N)
    L_pre = N_int - k_filler  # length before re-inserting filler bits

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


def _sionna5g_tx_llr_to_internal_llr(
    llr_tx: np.ndarray,
    code_cfg: CodeConfig,
    llr_max: float = 50.0,
) -> np.ndarray:
    """Rate-recover transmitted-bit LLRs into the mother LDPC graph LLR vector.

    llr_tx must be LLRs in the convention log p(x=0|y)/p(x=1|y).
    """
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

    # Undo output interleaver (if any)
    if out_int_inv is not None:
        llr_tx = llr_tx[np.asarray(out_int_inv, dtype=np.int32)]

    # Build the pre-filler vector: [0...(2Z) | llr_tx | 0...(tail)]
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

    # Re-insert filler bits with strong LLR towards 0
    if k_filler > 0:
        filler = np.full(k_filler, float(llr_max), dtype=np.float32)
        llr_int = np.concatenate([llr_pre[:k_info], filler, llr_pre[k_info:]], axis=0)
    else:
        llr_int = llr_pre

    if llr_int.size != N_int:
        raise RuntimeError(f"Internal LLR length mismatch: got {llr_int.size}, expected {N_int}")
    return llr_int


def _as_numpy(x: Any) -> np.ndarray:
    """Convert TensorFlow/PyTorch/NumPy-like objects to a NumPy array safely.

    Order matters here:
      * PyTorch tensors need detach().cpu().numpy().
      * TensorFlow eager tensors already expose .numpy(); calling .cpu() on them
        emits a deprecation warning in recent TensorFlow builds.
    """
    if isinstance(x, np.ndarray):
        return x

    # PyTorch tensors
    if hasattr(x, "detach"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass

    mod = getattr(type(x), "__module__", "") or ""

    # TensorFlow eager tensors and most NumPy-like wrappers
    if hasattr(x, "numpy"):
        try:
            return x.numpy()
        except Exception:
            pass

    # Fallback for other frameworks that require an explicit host transfer
    if hasattr(x, "cpu") and hasattr(x, "numpy") and not mod.startswith("tensorflow"):
        try:
            return x.cpu().numpy()
        except Exception:
            pass

    return np.array(x)


def _get_cached_tdl_model() -> Any:
    """Create/cache a Sionna TDL channel object (SISO) based on env vars."""
    if not SIONNA_TDL_AVAILABLE:
        raise RuntimeError(
            "Sionna TDL channel model not available. "
            f"Import detail: {_SIONNA_IMPORT_ERROR}"
        )

    model = os.getenv("SIONNA_TDL_MODEL", "A")
    delay_spread_s = float(os.getenv("SIONNA_TDL_DELAY_SPREAD_S", "3e-7"))  # 300 ns
    carrier_frequency_hz = float(os.getenv("SIONNA_TDL_CARRIER_FREQUENCY_HZ", "3.5e9"))
    min_speed = float(os.getenv("SIONNA_TDL_MIN_SPEED", "0.0"))
    max_speed = float(os.getenv("SIONNA_TDL_MAX_SPEED", "0.0"))

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


def sionna_tdl_ofdm_siso_bpsk(
    n_bits: int,
    snr_db: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate SISO BPSK over a 3GPP TDL channel (frequency-domain single-tap per RE).

    Returns:
      y_vec: complex received samples for the transmitted REs (length n_bits)
      h_vec: complex channel frequency response on those REs (length n_bits)
      no: noise variance per complex sample (E|w|^2)
    """
    tdl = _get_cached_tdl_model()

    fft_size = int(os.getenv("SIONNA_OFDM_FFT_SIZE", "256"))
    scs_hz = float(os.getenv("SIONNA_OFDM_SUBCARRIER_SPACING_HZ", "15000"))
    if fft_size <= 0:
        raise ValueError("SIONNA_OFDM_FFT_SIZE must be > 0")
    n_ofdm = int(math.ceil(n_bits / fft_size))
    pad = n_ofdm * fft_size - n_bits

    # all-zero CW -> all +1 symbols
    x = np.ones(n_bits + pad, dtype=np.complex64).reshape(n_ofdm, fft_size)

    sampling_frequency = float(fft_size) * scs_hz

    # Seed Sionna RNG from numpy rng for deterministic per-frame channel draws
    try:
        from sionna.phy import config as sionna_config  # Sionna v1.x
        sionna_config.seed = int(rng.integers(0, 2**31 - 1))
    except Exception:
        pass

    a, tau = tdl(batch_size=1, num_time_steps=n_ofdm, sampling_frequency=sampling_frequency)
    a = _as_numpy(a)
    tau = _as_numpy(tau)

    # SISO extraction (per Sionna docs):
    # a: [batch, rx=1, rx_ant=1, tx=1, tx_ant=1, num_paths, num_time_steps]
    a_siso = a[0, 0, 0, 0, 0, :, :]          # [n_paths, n_ofdm]
    a_siso = np.transpose(a_siso, (1, 0))    # [n_ofdm, n_paths]
    tau_siso = tau[0, 0, 0, :]               # [n_paths]

    # Frequency response on subcarriers (index 0..fft_size-1)
    f = (np.arange(fft_size, dtype=np.float32) * scs_hz)[None, :]        # [1, fft_size]
    phase = np.exp(-1j * 2.0 * np.pi * tau_siso[:, None] * f)            # [n_paths, fft_size]
    h = (a_siso @ phase).astype(np.complex64)                             # [n_ofdm, fft_size]

    snr_lin = 10.0 ** (snr_db / 10.0)
    no = 1.0 / snr_lin  # noise variance per complex sample
    w = (
        rng.standard_normal((n_ofdm, fft_size)).astype(np.float32)
        + 1j * rng.standard_normal((n_ofdm, fft_size)).astype(np.float32)
    ) * np.sqrt(no / 2.0)
    y = h * x + w

    # Optional CFO knob (simple per-OFDM-symbol phase rotation)
    cfo_hz = float(os.getenv("SIONNA_CFO_HZ", "0.0"))
    if cfo_hz != 0.0:
        t_sym = 1.0 / scs_hz  # rough OFDM symbol duration (no CP)
        rot = np.exp(1j * 2.0 * np.pi * cfo_hz * t_sym * np.arange(n_ofdm, dtype=np.float32))
        y = (rot[:, None] * y).astype(np.complex64)

    y_vec = y.reshape(-1)[:n_bits]
    h_vec = h.reshape(-1)[:n_bits]
    return y_vec, h_vec, float(no)


def _llr_bpsk_known_h(y: np.ndarray, h: np.ndarray, no: float) -> np.ndarray:
    """LLR for BPSK (0->+1,1->-1) over complex channel y=h*x+w, w~CN(0,no)."""
    # LLR = log p(y|x=+1)/p(y|x=-1) = 4*Re(h^* y)/no
    return (4.0 / float(no)) * np.real(np.conj(h) * y).astype(np.float32)


def _estimate_h_for_llr(y: np.ndarray,
                        h_true: np.ndarray,
                        no: float) -> Tuple[np.ndarray, str]:
    """Resolve the channel estimate used for LLR generation.

    Modes:
      - perfect      : use the exact per-RE channel
      - block_ls     : sparse-pilot / block-held LS approximation over the flattened
                       OFDM resource grid. This is a lightweight 5G-compatible proxy
                       for imperfect CSI without requiring a full DMRS resource-grid
                       implementation in this script.
      - nr_imperfect : block-LS plus structured interpolation / phase / amplitude
                       bias intended to emulate a harsher but still NR-like CSI
                       mismatch regime where BP can be misled by correlated LLR errors.
    """
    mode = str(os.getenv("SIONNA_CSI_MODE", "perfect") or "perfect").strip().lower()
    h_true = np.asarray(h_true, dtype=np.complex64)
    y = np.asarray(y, dtype=np.complex64)

    if mode in ("", "perfect", "ideal", "known_h", "true"):
        return h_true, "perfect"

    if mode in ("block_ls", "pilot_hold", "coarse", "coarse_ls", "nr_imperfect", "block_ls_drift", "imperfect_nr"):
        n = int(y.size)
        if n <= 0:
            return h_true, "perfect"

        stride = max(1, int(float(os.getenv("SIONNA_CSI_PILOT_STRIDE", os.getenv("SIONNA_CSI_BLOCK_SC", "12")))))
        smooth = max(1, int(float(os.getenv("SIONNA_CSI_SMOOTH_PILOTS", "1"))))
        pilot_idx = np.arange(0, n, stride, dtype=np.int32)
        if pilot_idx.size == 0:
            pilot_idx = np.array([0], dtype=np.int32)

        # LS on sparse pilot REs (x=+1 for all-zero CW under BPSK)
        h_p = y[pilot_idx].astype(np.complex64, copy=True)

        est_snr_db_str = str(os.getenv("SIONNA_CSI_EST_SNR_DB", "")).strip()
        if est_snr_db_str:
            try:
                est_snr_db = float(est_snr_db_str)
                est_var = 10.0 ** (-est_snr_db / 10.0)
                rng = np.random.default_rng(int(np.abs(np.sum(np.real(y[:min(32, n)]))) * 1e6) % (2**32 - 1))
                z = (rng.standard_normal(h_p.shape).astype(np.float32)
                     + 1j * rng.standard_normal(h_p.shape).astype(np.float32)) * np.sqrt(est_var / 2.0)
                h_p = (h_p + z.astype(np.complex64)).astype(np.complex64)
            except Exception:
                pass

        if smooth > 1 and h_p.size > 1:
            win = min(int(smooth), int(h_p.size))
            kernel = np.ones(win, dtype=np.float32) / float(win)
            h_p = (np.convolve(h_p.real, kernel, mode="same")
                   + 1j * np.convolve(h_p.imag, kernel, mode="same")).astype(np.complex64)

        xi = np.arange(n, dtype=np.float32)
        h_hat = (np.interp(xi, pilot_idx.astype(np.float32), h_p.real.astype(np.float32))
                 + 1j * np.interp(xi, pilot_idx.astype(np.float32), h_p.imag.astype(np.float32))).astype(np.complex64)

        if mode in ("nr_imperfect", "block_ls_drift", "imperfect_nr"):
            try:
                fft_size = max(1, int(os.getenv("SIONNA_OFDM_FFT_SIZE", "256")))
                scs_hz = float(os.getenv("SIONNA_OFDM_SUBCARRIER_SPACING_HZ", "15000"))
                n_sym = int(math.ceil(n / fft_size))
                h_grid = np.zeros((n_sym * fft_size,), dtype=np.complex64)
                h_grid[:n] = h_hat
                h_grid = h_grid.reshape(n_sym, fft_size)

                phase_step_std = np.deg2rad(float(os.getenv("SIONNA_CSI_PHASE_DRIFT_STD_DEG", "2.0")))
                amp_ripple_db = float(os.getenv("SIONNA_CSI_AMP_RIPPLE_DB", "1.25"))
                delay_bias_ns = float(os.getenv("SIONNA_CSI_DELAY_BIAS_NS", "15.0"))
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
                    delay_bias = float(delay_bias_ns) * 1e-9 * float(rng.normal(1.0, 0.25))
                    slope = (-2.0 * np.pi * delay_bias * scs_hz * sub_idx).astype(np.float32, copy=False)
                    amp_db = rng.normal(0.0, amp_ripple_db, size=amp_blocks).astype(np.float32)
                    amp = np.repeat((10.0 ** (amp_db / 20.0)).astype(np.float32), block_sc)[:fft_size]
                    rot = np.exp(1j * (phase_state + slope)).astype(np.complex64)
                    h_grid[t, :] = (h_grid[t, :] * amp.astype(np.complex64) * rot).astype(np.complex64)

                h_hat = h_grid.reshape(-1)[:n].astype(np.complex64, copy=False)
            except Exception:
                pass
            return h_hat, "nr_imperfect"

        return h_hat, "block_ls"

    return h_true, "perfect"


def run_single_frame_sionna5g(sim_cfg: SimulationConfig, frame_id: int, global_rng: np.random.Generator) -> FrameLog:
    """Single-frame runner for Sionna 5G NR LDPC codes.

    We transmit only the rate-matched `n_tx` bits, then rate-recover into the mother graph
    (pcm.shape[1]) before LDPC decoding / GRAND.
    """
    code_cfg = sim_cfg.code
    if not hasattr(code_cfg, "sionna"):
        raise ValueError("run_single_frame_sionna5g called but code_cfg has no .sionna")

    s = code_cfg.sionna
    n_tx = int(s["n_tx"])
    tx_pos_int = np.asarray(s["tx_pos"], dtype=np.int32)

    rng_seed_frame = int(global_rng.integers(0, 2**32 - 1))
    rng = np.random.default_rng(rng_seed_frame)

    # All-zero CW (symmetry)
    u_bits = np.zeros(code_cfg.K, dtype=np.uint8)
    c_bits = np.zeros(code_cfg.N, dtype=np.uint8)

    # Transmitted BPSK symbols (only n_tx positions). For logging, embed into length-N arrays.
    x_tx = np.ones(n_tx, dtype=np.float32)  # all +1

    ch_name = str(sim_cfg.channel.name).upper()
    channel_realization: Dict[str, Any] = {}

    if ch_name in ("SIONNA_TDL", "TDL"):
        y_c, h_c, no = sionna_tdl_ofdm_siso_bpsk(n_tx, sim_cfg.channel.snr_db, rng)
        h_llr, csi_mode_used = _estimate_h_for_llr(y_c, h_c, no)
        llr_tx = _llr_bpsk_known_h(y_c, h_llr, no)
        y_tx = y_c  # complex
        channel_realization.update({
            "no": np.array([no], dtype=np.float64),
            "tdl_model": np.array([str(os.getenv("SIONNA_TDL_MODEL", "A"))]),
            "csi_mode": np.array([str(csi_mode_used)]),
        })
    else:
        raise ValueError(
            f"Unsupported channel name '{sim_cfg.channel.name}'. "
            "This cleaned version supports only CHANNEL_NAME=SIONNA_TDL."
        )

    # Rate recovery into mother-graph LLRs
    llr_max = float(os.getenv("SIONNA_LLR_MAX", "50.0"))
    llr_int = _sionna5g_tx_llr_to_internal_llr(llr_tx, code_cfg, llr_max=llr_max)

    # Build logging arrays (length = N_internal)
    s_symbols = np.zeros(code_cfg.N, dtype=np.complex64)
    y_received = np.zeros(code_cfg.N, dtype=np.complex64)
    s_symbols[tx_pos_int] = x_tx.astype(np.complex64)
    y_received[tx_pos_int] = y_tx.astype(np.complex64)

    frame_log = FrameLog(
        frame_id=frame_id,
        rng_seed_frame=rng_seed_frame,
        u_bits=u_bits,
        c_bits=c_bits,
        interleaver_pattern=sim_cfg.interleaver.pattern,
        deinterleaver_pattern=sim_cfg.interleaver.inverse_pattern,
        s_symbols=s_symbols,
        y_channel=y_received,
        y_received=y_received,
        channel_realization=channel_realization,
    )

    # Decoder uses mother-graph channel LLRs
    frame_log.llr_channel = llr_int
    return frame_log


def run_single_frame(sim_cfg: SimulationConfig, frame_id: int, global_rng: np.random.Generator) -> FrameLog:
    """Dispatch per-frame simulation (cleaned: only Sionna 5G NR LDPC + SIONNA_TDL)."""
    if not hasattr(sim_cfg.code, "sionna"):
        raise ValueError("Only the Sionna 5G NR LDPC path ('sionna5g') is supported in this cleaned script.")
    return run_single_frame_sionna5g(sim_cfg, frame_id, global_rng)



### CELL number 8 ###
# (Removed) AWGN LLR helper.
# This cleaned script supports only CHANNEL_NAME=SIONNA_TDL and expects
# FrameLog.llr_channel to be set by run_single_frame_sionna5g().

# %%




# %%
### CELL number 9 ###
def prepare_flat_adjacency(code_cfg):
    """Convert checks_to_vars to flat CSR format for Numba."""
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


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _compute_syndrome_numba(bits, check_ptrs, check_indices, M):
        """
        Numba-accelerated syndrome computation, parallel over checks.

        For each check j:
            s[j] = XOR_{v in N(j)} bits[v]
        """
        s = np.zeros(M, dtype=np.uint8)
        for j in prange(M):
            start = check_ptrs[j]
            end = check_ptrs[j + 1]
            parity = 0
            for idx in range(start, end):
                parity ^= bits[check_indices[idx]]
            s[j] = parity
        return s


# %%
### CELL number 10 ###
def compute_syndrome_from_checks(bits: np.ndarray, code_cfg) -> np.ndarray:
    """
    Compute syndrome s = H * bits (mod 2).
    Uses Numba acceleration if available.
    """
    # Use fast path if Numba structures are prepared
    if NUMBA_AVAILABLE and hasattr(code_cfg, '_check_ptrs'):
        return _compute_syndrome_numba(
            bits,
            code_cfg._check_ptrs,
            code_cfg._check_indices,
            code_cfg.M
        )
    
    # Fallback to original
    M = code_cfg.M
    s = np.zeros(M, dtype=np.uint8)
    for j, var_indices in enumerate(code_cfg.checks_to_vars):
        if var_indices.size == 0:
            s[j] = 0
        else:
            s[j] = int(bits[var_indices].sum() % 2)
    return s

# %%
### CELL number 11 ###
import numpy as np
from typing import List

def build_systematic_ldpc_H(code_cfg: CodeConfig,
                             dv_info: int = 3,
                             dv_parity_extra: int = 1,
                             rng_seed: int = 2025) -> None:
    """
    Build a 'less easy' LDPC-style parity-check matrix for the current (N, K):

      - Let M = N - K.
      - H = [A | P], where:
          * A is M x K, sparse with dv_info ones per *information* column.
          * P is M x M, upper-triangular with:
              - 1s on the diagonal (always invertible over GF(2))
              - plus dv_parity_extra extra 1s ABOVE the diagonal in each column
                (so parity bits also have degree > 1).

    Properties:
      * rank(H) = M  => code dimension = K (with very high probability).
      * Systematic encoder: c = [u | p], with P p = A u (mod 2).
      * Variable node degrees:
          - info bits: ≈ dv_info
          - parity bits: ≥ 1 + dv_parity_extra (except maybe very first few cols)
      * Check node degrees: ≈ dv_info + (# of parity bits in each row).

    Also builds Tanner graph adjacency lists:
      - checks_to_vars[j] = array of variable indices connected to check j
      - vars_to_checks[v] = array of check indices connected to variable v
      - var_to_checks_edge_pos[v][k] = local edge index inside checks_to_vars[ vars_to_checks[v][k] ]

    NEW:
      - code_cfg.P_sys: the parity submatrix P
      - code_cfg.P_row_upper_indices: for each row i, the indices of columns k>i
        where P[i, k] = 1, used for fast back-substitution in encoding.
    """
    N, K = code_cfg.N, code_cfg.K
    M = N - K
    rng = np.random.default_rng(rng_seed)

    # ---- 1) Build sparse A for info bits ----
    A = np.zeros((M, K), dtype=np.uint8)
    for col in range(K):
        # dv_info distinct rows per info column
        rows = rng.choice(M, size=dv_info, replace=False)
        A[rows, col] = 1

    # ---- 2) Build upper-triangular P for parity bits ----
    # Start with identity (guarantees invertible, diag=1).
    P = np.eye(M, dtype=np.uint8)

    # Add extra 1s above the diagonal to increase parity column degrees.
    # For column 'col', we may add up to dv_parity_extra ones in rows < col.
    if dv_parity_extra > 0:
        for col in range(1, M):
            # how many extra ones we can place in this column (bounded by 'col')
            n_extra = min(col, dv_parity_extra)
            if n_extra == 0:
                continue
            rows = rng.choice(col, size=n_extra, replace=False)
            P[rows, col] ^= 1  # toggle bits (0->1 or 1->0, though diag never touched)

    # ---- 3) Assemble H = [A | P] ----
    H = np.concatenate([A, P], axis=1)  # shape (M, N)

    # ---- 4) Build Tanner-graph adjacency lists ----
    checks_to_vars: List[np.ndarray] = []
    vars_to_checks_lists: List[list] = [[] for _ in range(N)]
    edge_pos_lists: List[list] = [[] for _ in range(N)]

    for j in range(M):
        cols = np.flatnonzero(H[j])              # variable indices for check j
        cols_int = cols.astype(np.int32)
        checks_to_vars.append(cols_int)

        for local_e, v in enumerate(cols_int):
            vars_to_checks_lists[v].append(j)    # check index
            edge_pos_lists[v].append(local_e)    # position in checks_to_vars[j]

    vars_to_checks = [np.array(lst, dtype=np.int32) for lst in vars_to_checks_lists]
    var_to_checks_edge_pos = [np.array(lst, dtype=np.int32) for lst in edge_pos_lists]

    # ---- 5) Precompute row structure of P for back-substitution ----
    # For each row i, we want the list of columns k > i where P[i, k] = 1.
    P_row_upper_indices: List[np.ndarray] = []
    for i in range(M):
        nz = np.flatnonzero(P[i])
        nz_upper = nz[nz > i]  # strictly above diagonal
        P_row_upper_indices.append(nz_upper.astype(np.int32))

    # ---- 6) Attach to code_cfg ----
    code_cfg.M = M
    code_cfg.H = H
    code_cfg.A_sys = A                  # used by encoder
    code_cfg.P_sys = P                  # new: parity submatrix for encoder
    code_cfg.dv_info = dv_info
    code_cfg.checks_to_vars = checks_to_vars
    code_cfg.vars_to_checks = vars_to_checks
    code_cfg.var_to_checks_edge_pos = var_to_checks_edge_pos
    code_cfg.P_row_upper_indices = P_row_upper_indices

    # ---- 7) Quick stats ----
    var_degrees = np.array([len(v) for v in vars_to_checks])
    check_degrees = np.array([len(c) for c in checks_to_vars])
    parity_degrees = var_degrees[K:]  # last M columns are parity bits

    print(f"Built systematic LDPC-style H = [A | P] for code '{code_cfg.code_name}':")
    print(f" - N = {N}, K = {K}, M = {M}")
    print(f" - Info column degree target dv_info      = {dv_info}")
    print(f" - Variable degrees (all bits): min={var_degrees.min()}, "
          f"max={var_degrees.max()}, mean={var_degrees.mean():.2f}")
    print(f" - Parity column degrees:      min={parity_degrees.min()}, "
          f"max={parity_degrees.max()}, mean={parity_degrees.mean():.2f}")
    print(f" - Check degrees:              min={check_degrees.min()}, "
          f"max={check_degrees.max()}, mean={check_degrees.mean():.2f}")


### CELL number 12 ###
def prepare_code_for_fast_decoding(code_cfg) -> None:
    """
    Prepare flattened adjacency structures for Numba-accelerated decoding.

    MUST be called once after building code_cfg.checks_to_vars / vars_to_checks /
    var_to_checks_edge_pos (for any Tanner graph construction).
    """
    if not NUMBA_AVAILABLE:
        print("Numba not available - skipping fast preparation")
        return
    
    M = code_cfg.M
    N = code_cfg.N
    checks_to_vars = code_cfg.checks_to_vars
    vars_to_checks = code_cfg.vars_to_checks
    var_to_checks_edge_pos = code_cfg.var_to_checks_edge_pos
    
    # Syndrome computation structures
    check_ptrs, check_indices = prepare_flat_adjacency(code_cfg)
    code_cfg._check_ptrs = check_ptrs
    code_cfg._check_indices = check_indices
    
    # Min-sum decoder structures (CSR format)
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
    
    check_degrees = np.array([len(cv) for cv in checks_to_vars], dtype=np.int32)
    
    code_cfg._c2v_ptrs = c2v_ptrs
    code_cfg._c2v_indices = c2v_indices
    code_cfg._v2c_ptrs = v2c_ptrs
    code_cfg._v2c_checks = v2c_checks
    code_cfg._v2c_edge_pos = v2c_edge_pos
    code_cfg._check_degrees = check_degrees
    code_cfg._max_check_degree = int(check_degrees.max()) if M > 0 else 0
    
    print(f"Fast decoding structures prepared: {total_c2v} edges")


# NOTE: Legacy systematic-LDPC import-time build disabled/removed.
# It caused import-time side effects and a NameError after refactors (code_cfg is not module-global anymore).


# %%
### CELL number 13 ###
def encode_bits_ldpc_systematic(u_bits: np.ndarray, code_cfg: CodeConfig) -> np.ndarray:
    """
    Systematic LDPC encoder for H = [A | P]:

      Given info bits u (length K) in positions [0..K-1], produce codeword
          c = [u | p]
      where
          A u + P p = 0 (mod 2)   =>   P p = A u  (mod 2),

      with:
        - A = code_cfg.A_sys, shape (M, K)
        - P = code_cfg.P_sys, shape (M, M), upper triangular with 1s on the diagonal.

    We solve P p = t over GF(2) by back-substitution, where t = A u (mod 2).
    """
    A = code_cfg.A_sys          # shape (M, K)
    P = getattr(code_cfg, "P_sys", None)
    u = u_bits.astype(np.uint8)

    # Right-hand side: t = A * u (mod 2)
    # Use int32 for dot, then reduce mod 2.
    t = (A.dot(u.astype(np.int32)) & 1).astype(np.uint8)

    # If no P_sys is present (backward compatibility), treat P as identity
    if P is None:
        p = t
    else:
        M = P.shape[0]
        p = np.zeros(M, dtype=np.uint8)

        # Precomputed from build_systematic_ldpc_H: list of columns > i where P[i,k] = 1
        upper = code_cfg.P_row_upper_indices

        # Back substitution from bottom row up (since P is upper triangular with diag=1)
        for i in range(M - 1, -1, -1):
            acc = t[i]
            # subtract (XOR) contributions from already-solved p[k], k > i
            for k in upper[i]:
                acc ^= p[k]
            # P[i,i] = 1, so p[i] = acc (mod 2)
            p[i] = acc & 1

    # Assemble full codeword in [info | parity] layout
    c = np.zeros(code_cfg.N, dtype=np.uint8)
    c[:code_cfg.K] = u_bits
    c[code_cfg.K:] = p
    return c


def sanity_check_ldpc_encoder(code_cfg: CodeConfig,
                              num_tests: int = 3,
                              rng_seed: int = 999) -> None:
    """
    Verify that our encoder and H are consistent:
      For random u_bits, H @ c_bits % 2 == 0.
    """
    rng = np.random.default_rng(rng_seed)
    H = code_cfg.H
    M, N = H.shape

    print("=== LDPC encoder vs H consistency check ===")
    for t_idx in range(num_tests):
        u = rng.integers(0, 2, size=code_cfg.K, dtype=np.uint8)
        c = encode_bits_ldpc_systematic(u, code_cfg)

        if c.shape[0] != N:
            print(f"  [Test {t_idx}] ERROR: c_bits length {c.shape[0]} != N={N}")
            continue

        syn = (H.astype(np.int32) @ c.astype(np.int32)) & 1
        syn_weight = int(syn.sum())

        print(f"  Test {t_idx}: syndrome weight = {syn_weight}")
        if syn_weight == 0:
            print("    -> OK (valid codeword)")
        else:
            print("    -> ERROR (H c != 0), something is inconsistent!")

    print("Encoder/H consistency check done.")


### CELL number 14 ###
def encode_bits_simple(u_bits: np.ndarray, code_cfg: CodeConfig) -> np.ndarray:
    """
    Unified encoder used by the simulation pipeline.

    Supported modes:
      - encoder_mode == "all_zero": always return the all-zero codeword (valid for any linear code).
      - If code_cfg.A_sys exists (LDPC-style H = [A | I]), use systematic encoding: c=[u | A u].
      - Otherwise: fall back to the old placeholder encoder that appends zeros as parity bits.

    NOTE:
      For non-systematic codes (e.g., Gallager/5G Tanner graphs), use encoder_mode="all_zero"
      to avoid building a generator matrix.
    """
    enc_mode = str(getattr(code_cfg, "encoder_mode", "systematic")).lower()
    if enc_mode == "all_zero":
        return np.zeros(code_cfg.N, dtype=np.uint8)

    if hasattr(code_cfg, "A_sys"):
        # Use our proper LDPC-style encoder
        return encode_bits_ldpc_systematic(u_bits, code_cfg)

    # Fallback: old behaviour (u | zeros)
    N, K = code_cfg.N, code_cfg.K
    c_bits = np.zeros(N, dtype=np.uint8)
    c_bits[:K] = u_bits
    return c_bits




# %%
### CELL number 15 ###
@dataclass
class DecoderConfig:
    """Config for LDPC min-sum / normalized min-sum decoder."""
    max_iters: int = 40      # maximum number of iterations
    alpha: float = 0.8       # normalization factor (1.0 = pure min-sum)
    early_stop: bool = True  # stop when syndrome is 0




# %%
### CELL number 16 ###
dec_cfg_min_sum_4_early = DecoderConfig(
    max_iters=4,
    alpha=0.8,
    early_stop=True,   # baseline: allow early convergence
)

dec_cfg_min_sum_4_no_early = DecoderConfig(
    max_iters=4,
    alpha=0.8,
    early_stop=True,   # hybrid stage-1: early-stop; GRAND only if LDPC fails
)


# %%
### CELL number 17 ###
if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _check_node_update_numba(msg_v2c_flat, msg_c2v_flat, c2v_ptrs, M, alpha):
        """
        Numba-accelerated check node update, parallel over checks.

        For each check j, we compute outgoing CN->VN messages on all its edges
        using the normalized min-sum rule with factor 'alpha'.
        """
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

            # Scan incoming messages on this check
            for e in range(d):
                msg = msg_v2c_flat[start + e]

                if msg < 0.0:
                    sign_all *= -1.0
                    abs_val = -msg
                else:
                    # Treat very small positives as zero to match original behaviour
                    abs_val = msg if msg > 0.0 else 0.0

                if abs_val < min1:
                    min2 = min1
                    min1 = abs_val
                    idx_min1 = e
                elif abs_val < min2:
                    min2 = abs_val

            if d == 1:
                # For degree-1 checks, fall back to min1 for all edges (same as original)
                min2 = min1

            # Produce outgoing messages on each edge of this check
            for e in range(d):
                msg = msg_v2c_flat[start + e]
                sign_e = -sign_all if msg < 0.0 else sign_all
                mag_e = min2 if e == idx_min1 else min1
                msg_c2v_flat[start + e] = alpha * sign_e * mag_e

    @njit(parallel=True, cache=True)
    def _variable_node_update_numba(llr_channel,
                                    msg_c2v_flat,
                                    v2c_ptrs,
                                    v2c_checks,
                                    v2c_edge_pos,
                                    c2v_ptrs,
                                    N):
        """
        Numba-accelerated variable node update, parallel over variables.

        For each variable v:
            L_post[v] = L_ch[v] + sum_{j in N(v)} m_{j->v}
        """
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
    def _vn_to_cn_update_numba(llr_posterior,
                               msg_c2v_flat,
                               msg_v2c_flat,
                               v2c_ptrs,
                               v2c_checks,
                               v2c_edge_pos,
                               c2v_ptrs,
                               N):
        """
        Numba-accelerated VN->CN message update, parallel over variables.

        For each variable v and each neighbouring check j:
            m_{v->j} = L_post[v] - m_{j->v}
        """
        for v in prange(N):
            start = v2c_ptrs[v]
            end = v2c_ptrs[v + 1]
            L_v = llr_posterior[v]

            for idx in range(start, end):
                j = v2c_checks[idx]
                e = v2c_edge_pos[idx]
                base = c2v_ptrs[j] + e
                msg_v2c_flat[base] = L_v - msg_c2v_flat[base]


# %%
### CELL number 18 ###
def ldpc_min_sum_decode(llr_channel: np.ndarray,
                        code_cfg,
                        dec_cfg,
                        snapshot_iters: Optional[List[int]] = None,
                        snapshots: Optional[Dict[str, Dict[int, np.ndarray]]] = None):
    """
    Normalized min-sum LDPC decoding.
    Uses Numba acceleration if available.
    """
    N = code_cfg.N
    M = code_cfg.M
    
    # Use fast path if Numba structures are prepared
    use_fast = NUMBA_AVAILABLE and hasattr(code_cfg, '_c2v_ptrs')
    
    if use_fast:
        return _ldpc_min_sum_decode_fast(llr_channel, code_cfg, dec_cfg,
                                          snapshot_iters, snapshots)
    else:
        return _ldpc_min_sum_decode_original(llr_channel, code_cfg, dec_cfg,
                                              snapshot_iters, snapshots)


def _ldpc_min_sum_decode_fast(llr_channel, code_cfg, dec_cfg,
                               snapshot_iters, snapshots):
    """Numba-accelerated implementation."""
    N = code_cfg.N
    M = code_cfg.M
    
    c2v_ptrs = code_cfg._c2v_ptrs
    c2v_indices = code_cfg._c2v_indices
    v2c_ptrs = code_cfg._v2c_ptrs
    v2c_checks = code_cfg._v2c_checks
    v2c_edge_pos = code_cfg._v2c_edge_pos
    check_ptrs = code_cfg._check_ptrs
    check_indices = code_cfg._check_indices
    
    total_edges = c2v_ptrs[M]
    msg_v2c_flat = np.zeros(total_edges, dtype=np.float64)
    msg_c2v_flat = np.zeros(total_edges, dtype=np.float64)
    
    # Initialize VN->CN messages
    for j in range(M):
        start = c2v_ptrs[j]
        end = c2v_ptrs[j + 1]
        for idx in range(start, end):
            v = c2v_indices[idx]
            msg_v2c_flat[idx] = llr_channel[v]
    
    iter_used = dec_cfg.max_iters
    alpha = dec_cfg.alpha
    snapshot_set = set(snapshot_iters) if snapshot_iters else set()
    
    if snapshots is not None:
        snapshots.setdefault("llr", {})
        snapshots.setdefault("hard_bits", {})
        snapshots.setdefault("syndrome", {})
        snapshots.setdefault("unsat_checks", {})
    
    llr_posterior = llr_channel.copy()
    
    for it in range(1, dec_cfg.max_iters + 1):
        _check_node_update_numba(msg_v2c_flat, msg_c2v_flat, c2v_ptrs, M, alpha)
        
        llr_posterior = _variable_node_update_numba(
            llr_channel, msg_c2v_flat, v2c_ptrs, v2c_checks,
            v2c_edge_pos, c2v_ptrs, N
        )
        
        hard_bits = (llr_posterior < 0.0).astype(np.uint8)
        syndrome = _compute_syndrome_numba(hard_bits, check_ptrs, check_indices, M)
        
        if snapshots is not None and it in snapshot_set:
            snapshots["llr"][it] = llr_posterior.copy()
            snapshots["hard_bits"][it] = hard_bits.copy()
            snapshots["syndrome"][it] = syndrome.copy()
            snapshots["unsat_checks"][it] = np.flatnonzero(syndrome)
        
        if dec_cfg.early_stop and syndrome.sum() == 0:
            iter_used = it
            break
        
        _vn_to_cn_update_numba(
            llr_posterior, msg_c2v_flat, msg_v2c_flat,
            v2c_ptrs, v2c_checks, v2c_edge_pos, c2v_ptrs, N
        )
        iter_used = it
    
    hard_bits = (llr_posterior < 0.0).astype(np.uint8)
    syndrome = _compute_syndrome_numba(hard_bits, check_ptrs, check_indices, M)
    
    return hard_bits, llr_posterior, syndrome, iter_used


def _ldpc_min_sum_decode_original(llr_channel, code_cfg, dec_cfg,
                                   snapshot_iters, snapshots):
    """Original implementation (fallback)."""
    # [Keep your original implementation here as fallback]
    # Copy lines 813-928 from your original CELL 24
    N = code_cfg.N
    M = code_cfg.M
    checks_to_vars = code_cfg.checks_to_vars
    vars_to_checks = code_cfg.vars_to_checks
    var_to_checks_edge_pos = code_cfg.var_to_checks_edge_pos

    msg_v2c = [np.zeros(len(checks_to_vars[j]), dtype=np.float64) for j in range(M)]
    msg_c2v = [np.zeros(len(checks_to_vars[j]), dtype=np.float64) for j in range(M)]

    for j in range(M):
        neigh_vars = checks_to_vars[j]
        if neigh_vars.size == 0:
            continue
        msg_v2c[j][:] = llr_channel[neigh_vars]

    llr_posterior = llr_channel.copy()
    iter_used = dec_cfg.max_iters

    if snapshot_iters is not None:
        snapshot_set = set(snapshot_iters)
    else:
        snapshot_set = set()

    if snapshots is not None:
        snapshots.setdefault("llr", {})
        snapshots.setdefault("hard_bits", {})
        snapshots.setdefault("syndrome", {})
        snapshots.setdefault("unsat_checks", {})

    for it in range(1, dec_cfg.max_iters + 1):
        alpha = dec_cfg.alpha
        for j in range(M):
            msgs = msg_v2c[j]
            d = msgs.size
            if d == 0:
                continue

            signs = np.sign(msgs)
            signs[signs == 0] = 1.0
            abs_vals = np.abs(msgs)

            idx_min1 = int(np.argmin(abs_vals))
            min1 = abs_vals[idx_min1]
            if d > 1:
                tmp = abs_vals.copy()
                tmp[idx_min1] = np.inf
                min2 = float(tmp.min())
            else:
                min2 = min1

            sign_all = float(np.prod(signs))
            out = msg_c2v[j]

            for e in range(d):
                sign_e = sign_all * signs[e]
                mag_e = min2 if e == idx_min1 else min1
                out[e] = alpha * sign_e * mag_e

        llr_posterior = llr_channel.copy()

        for v in range(N):
            checks = vars_to_checks[v]
            if checks.size == 0:
                continue
            edge_pos = var_to_checks_edge_pos[v]
            total = 0.0
            for k, j in enumerate(checks):
                e = int(edge_pos[k])
                total += msg_c2v[j][e]
            llr_posterior[v] += total

        hard_bits = (llr_posterior < 0.0).astype(np.uint8)
        syndrome = compute_syndrome_from_checks(hard_bits, code_cfg)

        if snapshots is not None and it in snapshot_set:
            snapshots["llr"][it] = llr_posterior.copy()
            snapshots["hard_bits"][it] = hard_bits.copy()
            snapshots["syndrome"][it] = syndrome.copy()
            unsat_idx = np.flatnonzero(syndrome)
            snapshots["unsat_checks"][it] = unsat_idx

        if dec_cfg.early_stop and int(syndrome.sum()) == 0:
            iter_used = it
            break

        for v in range(N):
            checks = vars_to_checks[v]
            if checks.size == 0:
                continue
            edge_pos = var_to_checks_edge_pos[v]
            L_v = llr_posterior[v]
            for k, j in enumerate(checks):
                e = int(edge_pos[k])
                msg_v2c[j][e] = L_v - msg_c2v[j][e]

        iter_used = it

    hard_bits = (llr_posterior < 0.0).astype(np.uint8)
    syndrome = compute_syndrome_from_checks(hard_bits, code_cfg)

    return hard_bits, llr_posterior, syndrome, iter_used

# %%
def ldpc_min_sum_decoder_frame(frame: FrameLog,
                               sim_cfg: SimulationConfig,
                               dec_cfg: DecoderConfig) -> None:
    """
    Run LDPC normalized min-sum decoding on a single frame and populate the FrameLog
    with decoder outputs + per-iteration snapshots.
    """
    # Channel LLRs must be provided by run_single_frame_sionna5g() in this cleaned script.
    llr_ch = frame.llr_channel
    if llr_ch is None:
        raise RuntimeError(
            "FrameLog.llr_channel is None. "
            "This cleaned script supports only CHANNEL_NAME=SIONNA_TDL and requires "
            "run_single_frame() to populate FrameLog.llr_channel."
        )

   

    # Prepare snapshots dict on the frame
    frame.snapshots = {
        "llr": {},
        "hard_bits": {},
        "syndrome": {},
        "unsat_checks": {},
    }

    # Core LDPC decoding with snapshot support
    hard_bits, llr_post, syndrome, iter_used = ldpc_min_sum_decode(
        llr_ch,
        sim_cfg.code,
        dec_cfg,
        snapshot_iters=sim_cfg.snapshot_iters,
        snapshots=frame.snapshots,
    )

    # Fill standard FrameLog fields
    frame.hard_bits_final = hard_bits
    frame.llr_final = llr_post
    frame.syndrome_final = syndrome
    frame.iter_used = iter_used

    # Compare against transmitted codeword
    diff = (hard_bits != frame.c_bits)
    frame.error_positions_final = np.flatnonzero(diff)
    frame.dec_success = bool(frame.error_positions_final.size == 0)




### CELL number 20 ###
# Optional sanity check was removed (legacy AWGN path).
# Use the provided bash probes instead.
_run_sanity = str(os.environ.get("RUN_SANITY_CHECKS", "0")).strip().lower()
if _run_sanity not in ("0", "", "false", "no", "off"):
    raise RuntimeError(
        "RUN_SANITY_CHECKS is not supported in this cleaned Sionna-only script. "
        "Run the probe script instead."
    )




# %%
### CELL number 21 ###
def find_variable_clusters_from_syndrome(syndrome: np.ndarray,
                                         code_cfg: CodeConfig) -> List[np.ndarray]:
    """
    Given a syndrome vector (0/1) and the Tanner graph in code_cfg, return
    clusters of variable nodes that are connected via *unsatisfied* checks.

    Each cluster is a connected component in the graph where:
      - Nodes are variable indices v with degree > 0 in the unsatisfied-check subgraph.
      - There is an (undirected) edge between v1 and v2 if they share an unsatisfied check.
    """
    checks_to_vars = code_cfg.checks_to_vars
    N = code_cfg.N
    M = code_cfg.M

    # 1) Indices of unsatisfied checks
    unsat_checks = np.flatnonzero(syndrome)
    if unsat_checks.size == 0:
        return []

    # 2) Collect all variables touched by unsatisfied checks
    active_vars_set = set()
    for j in unsat_checks:
        for v in checks_to_vars[j]:
            active_vars_set.add(int(v))

    if not active_vars_set:
        return []

    active_vars = sorted(active_vars_set)
    L = len(active_vars)

    # Map variable index -> local index in [0..L-1]
    var_to_local = {v: i for i, v in enumerate(active_vars)}

    # 3) Build adjacency list for variable-variable graph (only active vars)
    adj = [[] for _ in range(L)]
    for j in unsat_checks:
        vars_j = [var_to_local[int(v)] for v in checks_to_vars[j]]
        d = len(vars_j)
        # Fully connect all vars in this unsatisfied check
        for a in range(d):
            va = vars_j[a]
            for b in range(a + 1, d):
                vb = vars_j[b]
                adj[va].append(vb)
                adj[vb].append(va)

    # 4) Find connected components via BFS/DFS
    clusters: List[np.ndarray] = []
    visited = np.zeros(L, dtype=bool)

    for start in range(L):
        if visited[start]:
            continue
        # BFS from 'start'
        queue = [start]
        visited[start] = True
        comp_local = []

        while queue:
            u = queue.pop()
            comp_local.append(u)
            for w in adj[u]:
                if not visited[w]:
                    visited[w] = True
                    queue.append(w)

        # Map local indices back to global variable indices
        comp_vars = np.array([active_vars[idx] for idx in comp_local], dtype=np.int32)
        clusters.append(comp_vars)

    return clusters




# %%
### CELL number 22 ###
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class ClusterGrandConfig:
    '''
    Config for local GRAND search on clusters / unions of clusters.

    Ordering rule used in this code:
      - enumerate flip patterns of weight 1..max_weight over a set of
        candidate bits;
      - assign each pattern a cost = sum(|LLR|) of the flipped bits;
      - test patterns in ascending cost (ties broken by lower weight).

    Key knobs:
      max_bits_from_cluster:
        - If an integer: use at most this many candidate bits (after sorting).
        - If None: AUTO-pick the number of candidate bits so that the total
          number of generated patterns stays close to max_patterns.
          (controlled by pattern_overgen_ratio).

      llr_source:
        - "posterior": use the LDPC posterior LLR snapshot for ordering/cost.
        - "channel"  : use the channel LLRs for ordering/cost (often better
                       when the LDPC posterior becomes over-confident on wrong bits).

      max_syndrome_weight_for_grand:
        - Optional guardrail. If set and the snapshot syndrome weight exceeds
          this threshold, skip GRAND (likely too many errors for a low-weight search).

    Notes:
      - max_patterns is BOTH a cap on *tested* patterns and (when AUTO bit-picking
        is used) an implicit cap on *generated/sorted* patterns, which is important
        for keeping the GRAND front-end cost reasonable.
    '''
    # Search space
    max_weight: int = 2
    max_patterns: int = 2000
    max_bits_from_cluster: Optional[int] = None

    # Logging / debug
    verbose: bool = True

    # Reliability / fade-aware gating (used by union-of-clusters code)
    low_llr_fraction: Optional[float] = None
    num_worst_blocks: Optional[int] = None

    # Chunked batching: number of patterns per Numba batch (if enabled)
    batch_size: int = 256

    # --- New knobs (hybrid performance + robustness) ---
    llr_source: str = "posterior"          # "posterior", "channel", or "mixed"
    pattern_overgen_ratio: float = 1.02    # used only when max_bits_from_cluster is None
    max_syndrome_weight_for_grand: Optional[int] = None

    # Receiver front-end selection
    selection_mode: str = "llr"            # "llr" (Receiver 1) or "syndrome_vote" (Receiver 2)
    sv_epsilon: float = 1e-3               # epsilon in eta_v = u_v / (rho_v + epsilon)
    sv_check_cover_k: int = 0              # k_cc ; 0 disables check-cover seeding

    # Receiver 3 / stronger pre-solver knobs
    pre_solver_mode: str = "none"          # "none", "peel_gf2", or "chase_list"
    peel_candidate_ratio: float = 1.50     # L_peel ~= ratio * L_search
    peel_max_bits: Optional[int] = None    # hard cap on peel candidate size
    peel_dense_max_vars: int = 32          # exact weighted GF(2) solve only if residual vars <= this
    peel_max_free_enum: int = 12           # enumerate weighted nullspace only if free dimension <= this
    peel_extra_llr_bits: int = 0           # add this many plain-LLR candidates to the peel set

    # Receiver 4 / Chase-list + short-LDPC post-processing knobs
    chase_candidate_ratio: float = 2.25    # L_chase ~= ratio * L_search
    chase_max_bits: Optional[int] = None   # hard cap on chase candidate size
    chase_core_max_bits: int = 12          # enumerate patterns only on the top-ranked core
    chase_max_weight: int = 3              # Chase pattern weight cap on that core
    chase_max_candidates: int = 96         # number of ranked candidates to try with short LDPC
    chase_ldpc_extra_iters: int = 6        # short polishing pass per Chase candidate
    chase_llr_gain: float = 2.5            # force candidate hypothesis with this |LLR| multiplier
    chase_llr_abs_floor: float = 4.0       # minimum |LLR| used when forcing candidate bits

    # Receiver 5 / Local OSD + anchored full-graph restarts
    osd_candidate_ratio: float = 2.75      # L_osd ~= ratio * L_search
    osd_max_bits: Optional[int] = None     # hard cap on OSD candidate size
    osd_order: int = 2                     # OSD order (enumerate low-order info-set flips)
    osd_enum_max_bits: int = 18            # enumerate only this many least-reliable free cols
    osd_max_candidates: int = 128          # keep at most this many ranked OSD candidates
    osd_disagreement_extra_bits: int = 0   # append channel-vs-posterior disagreement bits to pre-solver set
    restart_max_candidates: int = 24       # try at most this many OSD candidates globally
    restart_ldpc_iters: int = 14           # full-graph anchored restart iterations per candidate
    restart_alpha: float = 0.78            # NMS alpha for anchored restart
    restart_llr_gain: float = 4.5          # primary anchor gain
    restart_llr_abs_floor: float = 6.0     # minimum |LLR| on anchored bits
    restart_dual_gain: float = 6.5         # optional stronger second anchor gain
    restart_anchor_all_selected: int = 0   # 1 -> anchor all selected vars to candidate values, else support only

    # Receiver 6 / soft-hypothesis + anchored restart
    soft_candidate_ratio: float = 3.0      # L_soft ~= ratio * L_search
    soft_max_bits: Optional[int] = None    # hard cap on soft-hypothesis candidate size
    soft_core_max_bits: int = 14           # enumerate hypotheses on this many top-ranked bits
    soft_max_weight: int = 3               # max local flip weight for soft hypotheses
    soft_max_candidates: int = 128         # keep at most this many ranked soft hypotheses
    soft_sat_penalty: float = 0.35         # penalize variables mostly connected to satisfied checks
    soft_llr_weight: float = 0.10          # small extra penalty on large-|LLR| bits in ranking

    # Lightweight AI-ranker (Receiver 9 helper)
    ai_rank_vote_weight: float = 1.00      # weight on unsatisfied-check vote score
    ai_rank_llr_weight: float = 0.85       # weight on inverse-|LLR| reliability score
    ai_rank_disagreement_weight: float = 0.55  # weight on channel-vs-posterior sign disagreement
    ai_rank_density_weight: float = 0.35   # weight on local unsatisfied-check density
    ai_rank_roi_block_size: int = 64       # block size for compactness / concentration cues
    ai_rank_roi_weak_llr_quantile: float = 0.30
    ai_rank_roi_weak_llr_abs_cap: float = 2.50
    ai_rank_roi_diffuse_union_size: int = 208
    ai_rank_roi_diffuse_block_concentration: float = 0.08
    ai_rank_roi_compact_block_concentration: float = 0.11
    ai_rank_roi_diffuse_l_scale: float = 0.70
    ai_rank_roi_local_conflict_bonus: float = 0.30

    # Windowed AI ranker (block-local mixture-of-experts style)
    ai_window_block_size: int = 64
    ai_window_top_blocks: int = 2
    ai_window_neighbor_blocks: int = 0
    ai_window_local_seed_per_block: int = 4
    ai_window_diffuse_extra_blocks: int = 1
    ai_window_compact_single_threshold: float = 0.18
    ai_window_block_score_conflict_bonus: float = 0.35
    ai_window_block_score_density_bonus: float = 0.20

    # Receiver 7 / basis-GRAND + block-debias anchored restart
    basis_candidate_ratio: float = 3.0     # L_basis ~= ratio * L_search
    basis_max_bits: Optional[int] = None   # hard cap on basis candidate size
    basis_max_vectors: int = 18            # keep at most this many scored basis vectors
    basis_core_vectors: int = 12           # enumerate combinations only over this many best basis vectors
    basis_combo_max: int = 3               # GRAND over basis vectors up to this combination size
    basis_max_candidates: int = 128        # keep at most this many ranked basis combinations
    basis_group_max_bits: int = 10         # cap size of check/disagreement basis vectors
    basis_window_max: int = 6              # max width of ranked contiguous-window basis vectors
    basis_window_span: int = 18            # only generate windows inside this prefix of ranked vars
    basis_disagreement_groups: int = 4     # number of disagreement groups to seed
    basis_disagreement_chunk: int = 6      # bits per disagreement group
    basis_top_singletons: int = 8          # seed this many singleton basis vectors
    debias_blend: float = 0.65             # snapshot blend weight on debiased group vars
    debias_relax: float = 0.45             # magnitude relaxation on non-support group vars



### CELL number 23 ###
from dataclasses import dataclass
from typing import List, Tuple
import itertools
import numpy as np

@dataclass
class ClusterGrandResult:
    success: bool
    pattern_weight: int
    flipped_vars: np.ndarray
    patterns_tested: int
    initial_syndrome_weight: int
    final_syndrome_weight: int
    initial_bit_errors: int
    final_bit_errors: int

    # -------- NEW: op / complexity counters for hardware-like modeling --------
    # Total # of variable-to-check edges visited while testing patterns
    total_v2c_edge_visits: int = 0
    # Total # of UNIQUE checks encountered while testing patterns
    total_unique_checks_visited: int = 0
    # Total # of UNIQUE checks toggled odd times (i.e., weight-update touches)
    total_unique_checks_toggled: int = 0
    # Total patterns generated (before applying max_patterns cap)
    patterns_generated: int = 0


def _syndrome_weight_and_counts_after_flips_from_base(
    base_syndrome: np.ndarray,
    base_weight: int,
    flipped_vars: List[int],
    code_cfg: CodeConfig,
) -> Tuple[int, int, int, int]:
    """
    Exact incremental syndrome-weight update + op counters.

    Candidate = base_bits with flips at flipped_vars.
    Syndrome update:
        syn(cand) = syn(base) XOR (XOR of columns for flipped vars)

    Returns:
        syn_weight,
        v2c_edge_visits          : sum_{v in flipped} deg(v)
        unique_checks_visited    : | union of neighbouring checks |
        unique_checks_toggled    : | checks toggled odd times |
    """
    if not flipped_vars:
        return int(base_weight), 0, 0, 0

    toggled_checks = set()   # odd-toggle set
    visited_checks = set()   # union of checks seen at least once
    v2c = code_cfg.vars_to_checks

    edge_visits = 0
    for v in flipped_vars:
        for j in v2c[v]:
            edge_visits += 1
            j_int = int(j)
            visited_checks.add(j_int)
            if j_int in toggled_checks:
                toggled_checks.remove(j_int)
            else:
                toggled_checks.add(j_int)

    w = int(base_weight)
    for j in toggled_checks:
        if base_syndrome[j] == 0:
            w += 1
        else:
            w -= 1

    return w, edge_visits, len(visited_checks), len(toggled_checks)


def _bit_errors_after_flips_from_base(
    base_bits: np.ndarray,
    true_bits: np.ndarray,
    base_bit_errors: int,
    flipped_vars: List[int],
) -> int:
    """
    Exact incremental bit-error-count update (vs. true_bits).
    Only depends on whether each flipped bit was correct/incorrect in base_bits.
    """
    err = int(base_bit_errors)
    for v in flipped_vars:
        if base_bits[v] == true_bits[v]:
            err += 1
        else:
            err -= 1
    return err


# Batched GRAND pattern evaluation (per-batch, parallel over patterns)
# Optimized: incremental syndrome update + op counters
_GRAND_MAX_TOGGLES = 256  # must exceed (max_weight * max_variable_degree)


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _grand_eval_batch_numba_incremental(
        base_syndrome,
        base_syndrome_weight,
        base_bit_errors,
        base_bits,
        true_c_bits,
        search_vars,
        pattern_starts,
        pattern_lengths,
        pattern_positions,
        v2c_ptrs,
        v2c_checks,
    ):
        num_patterns = pattern_lengths.shape[0]
        M = base_syndrome.shape[0]

        syn_weights = np.empty(num_patterns, dtype=np.int32)
        bit_errors = np.full(num_patterns, -1, dtype=np.int32)

        # ---- NEW: per-pattern op counters ----
        edge_visits = np.zeros(num_patterns, dtype=np.int32)
        uniq_checks_visited = np.zeros(num_patterns, dtype=np.int32)
        uniq_checks_toggled = np.zeros(num_patterns, dtype=np.int32)

        for p in prange(num_patterns):
            start_p = pattern_starts[p]
            len_p = pattern_lengths[p]

            overflow = False

            # Track unique checks encountered; check_tog holds odd/even parity.
            check_ids = np.empty(_GRAND_MAX_TOGGLES, dtype=np.int32)
            check_tog = np.empty(_GRAND_MAX_TOGGLES, dtype=np.uint8)
            num_tog = 0

            err = base_bit_errors
            edges = 0

            # Build toggle set + error count
            for k in range(len_p):
                local_pos = pattern_positions[start_p + k]
                v = search_vars[local_pos]

                # bit-error update
                if base_bits[v] == true_c_bits[v]:
                    err += 1
                else:
                    err -= 1

                # syndrome toggles induced by flipping bit v
                vs = v2c_ptrs[v]
                ve = v2c_ptrs[v + 1]
                for idx in range(vs, ve):
                    edges += 1
                    j = v2c_checks[idx]

                    # find j in current unique list
                    found_idx = -1
                    for t in range(num_tog):
                        if check_ids[t] == j:
                            found_idx = t
                            break

                    if found_idx >= 0:
                        check_tog[found_idx] ^= np.uint8(1)
                    else:
                        if num_tog < _GRAND_MAX_TOGGLES:
                            check_ids[num_tog] = j
                            check_tog[num_tog] = np.uint8(1)
                            num_tog += 1
                        else:
                            overflow = True
                            break

                if overflow:
                    break

            if overflow:
                # Very rare for this project, but keep correctness.
                cand_syn = base_syndrome.copy()
                err2 = base_bit_errors
                edges2 = 0

                for k in range(len_p):
                    local_pos = pattern_positions[start_p + k]
                    v = search_vars[local_pos]

                    if base_bits[v] == true_c_bits[v]:
                        err2 += 1
                    else:
                        err2 -= 1

                    vs = v2c_ptrs[v]
                    ve = v2c_ptrs[v + 1]
                    for idx in range(vs, ve):
                        edges2 += 1
                        j = v2c_checks[idx]
                        cand_syn[j] ^= np.uint8(1)

                w = 0
                toggled = 0
                for j in range(M):
                    if cand_syn[j] != 0:
                        w += 1
                    if cand_syn[j] != base_syndrome[j]:
                        toggled += 1

                syn_weights[p] = w
                bit_errors[p] = err2 if w == 0 else -1

                edge_visits[p] = edges2
                uniq_checks_visited[p] = -1   # unknown in overflow path
                uniq_checks_toggled[p] = toggled
                continue

            # Store op counts
            edge_visits[p] = edges
            uniq_checks_visited[p] = num_tog

            toggled_cnt = 0
            for t in range(num_tog):
                if check_tog[t] != 0:
                    toggled_cnt += 1
            uniq_checks_toggled[p] = toggled_cnt

            # Fast weight update from base_syndrome_weight
            w = base_syndrome_weight
            for t in range(num_tog):
                if check_tog[t] != 0:
                    j = check_ids[t]
                    if base_syndrome[j] == 0:
                        w += 1
                    else:
                        w -= 1

            syn_weights[p] = w
            if w == 0:
                bit_errors[p] = err

        return syn_weights, bit_errors, edge_visits, uniq_checks_visited, uniq_checks_toggled


def run_local_grand_on_cluster(frame: FrameLog,
                               sim_cfg: SimulationConfig,
                               snapshot_iter: int,
                               cluster_index: int,
                               cfg: ClusterGrandConfig) -> ClusterGrandResult:
    """
    Local GRAND-style search on a single variable cluster at a given
    decoder iteration snapshot, with chunked batched pattern testing.

    Membership test:
      - incremental syndrome updates (no full syndrome recomputation per pattern)

    NEW:
      - accumulates per-frame op counters for hardware-like modeling.
    """
    code_cfg = sim_cfg.code

    # ---- Extract snapshot data ----
    snaps = frame.snapshots
    syn_snaps = snaps.get("syndrome", {})
    hard_snaps = snaps.get("hard_bits", {})
    llr_snaps = snaps.get("llr", {})

    if (snapshot_iter not in syn_snaps or
        snapshot_iter not in hard_snaps or
        snapshot_iter not in llr_snaps):
        raise ValueError(
            f"Snapshot at iter {snapshot_iter} is not fully available "
            f"(keys: syndrome={list(syn_snaps.keys())}, "
            f"hard_bits={list(hard_snaps.keys())}, "
            f"llr={list(llr_snaps.keys())})"
        )

    syndrome = syn_snaps[snapshot_iter]
    hard_bits_snapshot = hard_snaps[snapshot_iter].copy()
    llr_snapshot = llr_snaps[snapshot_iter]

    initial_syndrome_weight = int(syndrome.sum())

    # If already a codeword, nothing to do
    if initial_syndrome_weight == 0:
        diff_init = (hard_bits_snapshot != frame.c_bits)
        initial_bit_errors = int(diff_init.sum())
        return ClusterGrandResult(
            success=True,
            pattern_weight=0,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )

    # ---- Build clusters from current syndrome ----
    clusters = find_variable_clusters_from_syndrome(syndrome, code_cfg)
    if not clusters:
        diff_init = (hard_bits_snapshot != frame.c_bits)
        initial_bit_errors = int(diff_init.sum())
        return ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )

    if cluster_index < 0 or cluster_index >= len(clusters):
        raise IndexError(f"cluster_index {cluster_index} out of range [0, {len(clusters)-1}]")

    cluster_vars = clusters[cluster_index]
    cluster_size = cluster_vars.size

    # Initial bit errors at snapshot vs ground truth
    diff_init = (hard_bits_snapshot != frame.c_bits)
    initial_bit_errors = int(diff_init.sum())

    # ---- Order cluster bits by |LLR| (least reliable first) ----
    abs_llr_cluster = np.abs(llr_snapshot[cluster_vars])
    order = np.argsort(abs_llr_cluster)   # ascending |LLR|
    cluster_vars_sorted = cluster_vars[order]

    if cfg.max_bits_from_cluster is not None and cfg.max_bits_from_cluster < cluster_size:
        search_vars = cluster_vars_sorted[:cfg.max_bits_from_cluster]
    else:
        search_vars = cluster_vars_sorted
    L = search_vars.size

    patterns_tested = 0
    found = False
    found_weight = -1
    found_flipped = np.array([], dtype=np.int32)
    final_syn_weight = initial_syndrome_weight
    final_bit_errors = initial_bit_errors

    # ---- NEW totals ----
    total_edge_visits = 0
    total_uniq_checks_visited = 0
    total_uniq_checks_toggled = 0

    if L == 0 or cfg.max_weight <= 0:
        return ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=final_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )

    max_w = min(cfg.max_weight, L)

    # ---- Build all patterns and order them by sum‑|LLR| ----
    pattern_items: List[tuple] = []
    abs_llr_local = np.abs(llr_snapshot[search_vars])

    for w in range(1, max_w + 1):
        for comb in itertools.combinations(range(L), w):
            cost = float(abs_llr_local[list(comb)].sum())
            pattern_items.append((cost, w, comb))

    pattern_items.sort(key=lambda t: (t[0], t[1]))
    patterns_generated = len(pattern_items)

    if not pattern_items:
        return ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )

    use_batch = (
        NUMBA_AVAILABLE
        and hasattr(code_cfg, "_v2c_ptrs")
        and hasattr(code_cfg, "_v2c_checks")
        and getattr(cfg, "batch_size", 0) > 0
    )

    if use_batch:
        base_bits = hard_bits_snapshot.astype(np.uint8)
        true_c_bits = frame.c_bits.astype(np.uint8)
        search_vars_int = search_vars.astype(np.int64)

        base_syn = syndrome.astype(np.uint8)
        base_syn_w = np.int32(initial_syndrome_weight)
        base_bit_err = np.int32(initial_bit_errors)

        total_patterns = len(pattern_items)
        max_patterns = int(cfg.max_patterns)
        limit = min(total_patterns, max_patterns)
        batch_size = int(getattr(cfg, "batch_size", 256))
        if batch_size <= 0:
            batch_size = limit

        for start_idx in range(0, limit, batch_size):
            end_idx = min(start_idx + batch_size, limit)
            num_batch = end_idx - start_idx

            # Pack this batch
            total_positions = 0
            for i in range(start_idx, end_idx):
                total_positions += pattern_items[i][1]

            pattern_starts = np.zeros(num_batch, dtype=np.int32)
            pattern_lengths = np.zeros(num_batch, dtype=np.int32)
            pattern_positions = np.zeros(total_positions, dtype=np.int32)

            pos_ptr = 0
            for b in range(num_batch):
                _, w, comb = pattern_items[start_idx + b]
                pattern_starts[b] = pos_ptr
                pattern_lengths[b] = w
                for lp in comb:
                    pattern_positions[pos_ptr] = int(lp)
                    pos_ptr += 1

            syn_w_arr, bit_err_arr, edge_arr, uniq_arr, tog_arr = _grand_eval_batch_numba_incremental(
                base_syn,
                base_syn_w,
                base_bit_err,
                base_bits,
                true_c_bits,
                search_vars_int,
                pattern_starts,
                pattern_lengths,
                pattern_positions,
                code_cfg._v2c_ptrs,
                code_cfg._v2c_checks,
            )

            # Find first success in this batch
            success_rel = -1
            for b in range(num_batch):
                if syn_w_arr[b] == 0:
                    success_rel = b
                    break

            if success_rel >= 0:
                # Accumulate counters up to and including success_rel
                total_edge_visits += int(edge_arr[:success_rel + 1].sum())
                total_uniq_checks_visited += int(np.maximum(uniq_arr[:success_rel + 1], 0).sum())
                total_uniq_checks_toggled += int(tog_arr[:success_rel + 1].sum())

                global_idx = start_idx + success_rel
                patterns_tested = global_idx + 1
                _, w, comb = pattern_items[global_idx]
                flipped = [int(search_vars[pos]) for pos in comb]
                found_flipped = np.array(flipped, dtype=np.int32)
                found_weight = w
                final_syn_weight = 0
                be = int(bit_err_arr[success_rel])
                final_bit_errors = be if be >= 0 else initial_bit_errors
                found = True
                break
            else:
                # Accumulate full batch counters
                total_edge_visits += int(edge_arr.sum())
                total_uniq_checks_visited += int(np.maximum(uniq_arr, 0).sum())
                total_uniq_checks_toggled += int(tog_arr.sum())
                patterns_tested = end_idx

    else:
        # Sequential one‑by‑one testing (incremental membership + counters)
        for _, w, comb in pattern_items:
            patterns_tested += 1
            if patterns_tested > cfg.max_patterns:
                break

            flipped = [int(search_vars[pos]) for pos in comb]

            syn_w, e_cnt, uq_cnt, tg_cnt = _syndrome_weight_and_counts_after_flips_from_base(
                base_syndrome=syndrome,
                base_weight=initial_syndrome_weight,
                flipped_vars=flipped,
                code_cfg=code_cfg,
            )

            total_edge_visits += e_cnt
            total_uniq_checks_visited += uq_cnt
            total_uniq_checks_toggled += tg_cnt

            if syn_w == 0:
                bit_err_cand = _bit_errors_after_flips_from_base(
                    base_bits=hard_bits_snapshot,
                    true_bits=frame.c_bits,
                    base_bit_errors=initial_bit_errors,
                    flipped_vars=flipped,
                )

                found = True
                found_weight = w
                found_flipped = np.array(flipped, dtype=np.int32)
                final_syn_weight = 0
                final_bit_errors = bit_err_cand
                break

    return ClusterGrandResult(
        success=found,
        pattern_weight=found_weight if found else -1,
        flipped_vars=found_flipped,
        patterns_tested=patterns_tested,
        initial_syndrome_weight=initial_syndrome_weight,
        final_syndrome_weight=final_syn_weight,
        initial_bit_errors=initial_bit_errors,
        final_bit_errors=final_bit_errors,
        total_v2c_edge_visits=int(total_edge_visits),
        total_unique_checks_visited=int(total_uniq_checks_visited),
        total_unique_checks_toggled=int(total_uniq_checks_toggled),
        patterns_generated=int(patterns_generated),
    )







# %%
### CELL number 26 ###
def build_allowed_mask_from_config(frame: FrameLog,
                                   sim_cfg: SimulationConfig,
                                   snapshot_iter: int,
                                   cfg: ClusterGrandConfig) -> np.ndarray:
    """
    Build a boolean mask allowed_mask[v] telling GRAND which variable
    nodes it is allowed to flip at the given snapshot.

    Combines:
      - global low-|LLR| gating via cfg.low_llr_fraction in (0, 1),
      - block-fading gating via cfg.num_worst_blocks (integer >= 1).

    If those attributes are missing or None, we fall back to
    'all bits allowed'.
    """
    N = sim_cfg.code.N

    # Start with everything allowed
    allowed = np.ones(N, dtype=bool)

    # Pull LLR snapshot
    llr_snaps = frame.snapshots.get("llr", {})
    if snapshot_iter not in llr_snaps:
        # No LLR at this iteration; don't gate anything.
        return allowed

    llr_snapshot = llr_snaps[snapshot_iter]
    if llr_snapshot.shape[0] != N:
        raise ValueError(f"LLR snapshot length {llr_snapshot.shape[0]} != N={N}")

    # ---- 1) low-|LLR| gating (reliability gating) ----
    low_frac = getattr(cfg, "low_llr_fraction", None)
    if isinstance(low_frac, (int, float)) and 0.0 < low_frac < 1.0:
        abs_llr = np.abs(llr_snapshot)
        K = max(1, int(np.round(low_frac * N)))
        if K < N:
            # Threshold tau: K smallest |LLR|
            tau = np.partition(abs_llr, K - 1)[K - 1]
            mask_low = abs_llr <= tau
        else:
            mask_low = np.ones_like(allowed)
        allowed &= mask_low

    # ---- 2) block-fading gating (worst B blocks) ----
    num_worst_blocks = getattr(cfg, "num_worst_blocks", None)
    if isinstance(num_worst_blocks, int) and num_worst_blocks > 0:
        ch = frame.channel_realization
        block_idx = ch.get("block_index_per_bit", None)
        num_blocks_arr = ch.get("num_blocks", None)

        if block_idx is not None and num_blocks_arr is not None:
            block_idx = block_idx.astype(np.int64)
            F = int(num_blocks_arr[0])
            abs_llr = np.abs(llr_snapshot)

            # Per-block mean |LLR|
            block_scores = np.full(F, np.inf, dtype=np.float64)
            for b in range(F):
                mask_b = (block_idx == b)
                if not mask_b.any():
                    continue
                # Score = avg |LLR| in the block
                block_scores[b] = abs_llr[mask_b].mean()

            B = min(num_worst_blocks, F)
            worst_ids = np.argsort(block_scores)[:B]
            in_worst = np.isin(block_idx, worst_ids)
            allowed &= in_worst

    return allowed




def _auto_pick_grand_search_size(L_full: int, cfg: ClusterGrandConfig) -> int:
    """Choose the GRAND search size L.

    If max_bits_from_cluster is None, pick the largest L whose generated pattern
    count stays close to cfg.max_patterns (using cfg.pattern_overgen_ratio).
    Otherwise, clamp to max_bits_from_cluster.
    """
    L_full = int(max(L_full, 0))
    if L_full <= 0:
        return 0

    if cfg.max_bits_from_cluster is None:
        over = float(getattr(cfg, "pattern_overgen_ratio", 1.02) or 1.02)
        target = max(1, int(round(over * int(cfg.max_patterns))))
        max_w = max(1, int(cfg.max_weight))

        L = 1
        while True:
            L_try = L + 1
            total = 0
            for w in range(1, max_w + 1):
                total += math.comb(L_try, w)
                if total > target:
                    break
            if total > target:
                break
            if L_try >= L_full:
                L = L_try
                break
            L = L_try
        return int(min(max(L, 1), L_full))

    return int(min(max(int(cfg.max_bits_from_cluster), 0), L_full))



def _select_search_vars_llr(union_vars: np.ndarray,
                            llr_for_sort: np.ndarray,
                            L: int) -> Tuple[np.ndarray, Dict[str, int]]:
    """Receiver 1 front-end: pick the L least reliable bits in the union."""
    L = int(max(L, 0))
    union_vars = np.asarray(union_vars, dtype=np.int32)
    if L <= 0 or union_vars.size == 0:
        return np.array([], dtype=np.int32), {
            "selection_mode_used": "llr",
            "sv_seeded_count": 0,
            "sv_neighbor_visits": 0,
            "sv_score_len": int(union_vars.size),
        }

    abs_llr_union = np.abs(llr_for_sort[union_vars])
    order = np.argsort(abs_llr_union, kind="stable")
    search_vars = union_vars[order[:L]].astype(np.int32, copy=False)
    return search_vars, {
        "selection_mode_used": "llr",
        "sv_seeded_count": 0,
        "sv_neighbor_visits": 0,
        "sv_score_len": int(union_vars.size),
    }



def _select_search_vars_syndrome_vote(union_vars: np.ndarray,
                                      unsat_checks: np.ndarray,
                                      code_cfg: CodeConfig,
                                      llr_for_sort: np.ndarray,
                                      L: int,
                                      cfg: ClusterGrandConfig) -> Tuple[np.ndarray, Dict[str, int]]:
    """Receiver 2 front-end: syndrome-vote ranking with optional check-cover seeding.

    Score:
        eta_v = u_v / (rho_v + epsilon)
    where u_v counts the number of unsatisfied checks touching v and
    rho_v = |LLR_v| from the selected LLR source.

    Check-cover seeding:
        For each unsatisfied check j, include up to k_cc of its least reliable
        neighbours before filling the remaining budget from the global score rank.
    """
    union_vars = np.asarray(union_vars, dtype=np.int32)
    unsat_checks = np.asarray(unsat_checks, dtype=np.int32)
    L = int(max(L, 0))

    if L <= 0 or union_vars.size == 0:
        return np.array([], dtype=np.int32), {
            "selection_mode_used": "syndrome_vote",
            "sv_seeded_count": 0,
            "sv_neighbor_visits": 0,
            "sv_score_len": int(union_vars.size),
        }

    abs_llr_union = np.abs(llr_for_sort[union_vars]).astype(np.float64, copy=False)
    eps = float(getattr(cfg, "sv_epsilon", 1e-3) or 1e-3)
    k_cc = max(0, int(getattr(cfg, "sv_check_cover_k", 0) or 0))

    vote_counts = np.zeros(union_vars.size, dtype=np.int32)
    var_to_local = {int(v): i for i, v in enumerate(union_vars.tolist())}

    seed_list: List[int] = []
    seeded = set()
    sv_neighbor_visits = 0

    for j in unsat_checks:
        local_neighbors: List[int] = []
        for v in code_cfg.checks_to_vars[int(j)]:
            v_int = int(v)
            loc = var_to_local.get(v_int, None)
            if loc is None:
                continue
            vote_counts[loc] += 1
            local_neighbors.append(loc)
            sv_neighbor_visits += 1

        if k_cc > 0 and local_neighbors:
            # Least reliable neighbours first; deterministic tie-break by vote then index.
            local_neighbors_sorted = sorted(
                local_neighbors,
                key=lambda loc: (float(abs_llr_union[loc]), -int(vote_counts[loc]), int(union_vars[loc]))
            )
            take = min(k_cc, len(local_neighbors_sorted))
            for loc in local_neighbors_sorted[:take]:
                v_int = int(union_vars[loc])
                if v_int not in seeded:
                    seed_list.append(v_int)
                    seeded.add(v_int)

    # Global score rank for Receiver 2
    scores = vote_counts.astype(np.float64) / (abs_llr_union + eps)
    global_order = sorted(
        range(union_vars.size),
        key=lambda loc: (-float(scores[loc]), float(abs_llr_union[loc]), -int(vote_counts[loc]), int(union_vars[loc]))
    )

    selected: List[int] = []
    seen = set()

    for v_int in seed_list:
        if len(selected) >= L:
            break
        if v_int not in seen:
            selected.append(int(v_int))
            seen.add(int(v_int))

    for loc in global_order:
        if len(selected) >= L:
            break
        v_int = int(union_vars[loc])
        if v_int not in seen:
            selected.append(v_int)
            seen.add(v_int)

    return np.asarray(selected[:L], dtype=np.int32), {
        "selection_mode_used": "syndrome_vote",
        "sv_seeded_count": int(min(len(seed_list), L)),
        "sv_neighbor_visits": int(sv_neighbor_visits),
        "sv_score_len": int(union_vars.size),
    }



def _select_search_vars_ai_rank(union_vars: np.ndarray,
                                 unsat_checks: np.ndarray,
                                 code_cfg: CodeConfig,
                                 llr_for_sort: np.ndarray,
                                 L: int,
                                 cfg: ClusterGrandConfig,
                                 llr_snapshot: Optional[np.ndarray] = None,
                                 llr_channel: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """Lightweight AI-style search-var ranker.

    This is intentionally tiny and hardware-friendly. It blends:
      - unsatisfied-check vote strength,
      - inverse |LLR| reliability,
      - local unsatisfied-check density,
      - posterior-vs-channel sign disagreement.

    The score is a distilled one-layer model so it can later be replaced by
    trained INT8 weights without changing the decoder interface.
    """
    union_vars = np.asarray(union_vars, dtype=np.int32)
    unsat_checks = np.asarray(unsat_checks, dtype=np.int32)
    L = int(max(L, 0))

    if L <= 0 or union_vars.size == 0:
        return np.array([], dtype=np.int32), {
            "selection_mode_used": "ai_rank",
            "sv_seeded_count": 0,
            "sv_neighbor_visits": 0,
            "sv_score_len": int(union_vars.size),
        }

    abs_llr_union = np.abs(llr_for_sort[union_vars]).astype(np.float64, copy=False)
    eps = float(getattr(cfg, "sv_epsilon", 1e-3) or 1e-3)
    k_cc = max(0, int(getattr(cfg, "sv_check_cover_k", 0) or 0))

    vote_counts = np.zeros(union_vars.size, dtype=np.int32)
    var_to_local = {int(v): i for i, v in enumerate(union_vars.tolist())}
    sv_neighbor_visits = 0

    seed_list: List[int] = []
    seeded = set()
    for j in unsat_checks:
        local_neighbors: List[int] = []
        for v in code_cfg.checks_to_vars[int(j)]:
            v_int = int(v)
            loc = var_to_local.get(v_int, None)
            if loc is None:
                continue
            vote_counts[loc] += 1
            local_neighbors.append(loc)
            sv_neighbor_visits += 1

        if k_cc > 0 and local_neighbors:
            local_neighbors_sorted = sorted(
                local_neighbors,
                key=lambda loc: (float(abs_llr_union[loc]), -int(vote_counts[loc]), int(union_vars[loc]))
            )
            take = min(k_cc, len(local_neighbors_sorted))
            for loc in local_neighbors_sorted[:take]:
                v_int = int(union_vars[loc])
                if v_int not in seeded:
                    seed_list.append(v_int)
                    seeded.add(v_int)

    max_vote = max(1, int(vote_counts.max()) if vote_counts.size else 0)
    inv_llr = 1.0 / (abs_llr_union + eps)
    max_inv_llr = float(inv_llr.max()) if inv_llr.size else 1.0
    if max_inv_llr <= 0.0:
        max_inv_llr = 1.0

    vote_norm = vote_counts.astype(np.float64) / float(max_vote)
    inv_llr_norm = inv_llr / float(max_inv_llr)

    degs = np.asarray([max(1, len(code_cfg.vars_to_checks[int(v)])) for v in union_vars.tolist()], dtype=np.float64)
    density = np.clip(vote_counts.astype(np.float64) / degs, 0.0, 1.0)

    disagree = np.zeros(union_vars.size, dtype=np.float64)
    if llr_snapshot is not None and llr_channel is not None:
        llr_snapshot = np.asarray(llr_snapshot, dtype=np.float32)
        llr_channel = np.asarray(llr_channel, dtype=np.float32)
        snap_sign = np.sign(llr_snapshot[union_vars]).astype(np.int8, copy=False)
        chan_sign = np.sign(llr_channel[union_vars]).astype(np.int8, copy=False)
        disagree = ((snap_sign * chan_sign) < 0).astype(np.float64, copy=False)

    w_vote = float(getattr(cfg, "ai_rank_vote_weight", 1.0) or 1.0)
    w_llr = float(getattr(cfg, "ai_rank_llr_weight", 0.85) or 0.85)
    w_dis = float(getattr(cfg, "ai_rank_disagreement_weight", 0.55) or 0.55)
    w_den = float(getattr(cfg, "ai_rank_density_weight", 0.35) or 0.35)

    # Distilled one-layer score. Conservative tie-breaks preserve the old behaviour.
    ai_score = (
        w_vote * vote_norm
        + w_llr * inv_llr_norm
        + w_dis * disagree * inv_llr_norm
        + w_den * density
    )

    global_order = sorted(
        range(union_vars.size),
        key=lambda loc: (-float(ai_score[loc]), float(abs_llr_union[loc]), -int(vote_counts[loc]), int(union_vars[loc]))
    )

    selected: List[int] = []
    seen = set()
    for v_int in seed_list:
        if len(selected) >= L:
            break
        if v_int not in seen:
            selected.append(int(v_int))
            seen.add(int(v_int))

    for loc in global_order:
        if len(selected) >= L:
            break
        v_int = int(union_vars[loc])
        if v_int not in seen:
            selected.append(v_int)
            seen.add(v_int)

    return np.asarray(selected[:L], dtype=np.int32), {
        "selection_mode_used": "ai_rank",
        "sv_seeded_count": int(min(len(seed_list), L)),
        "sv_neighbor_visits": int(sv_neighbor_visits),
        "sv_score_len": int(union_vars.size),
    }


def _ai_rank_roi_block_concentration(union_vars: np.ndarray, block_size: int) -> float:
    union_vars = np.asarray(union_vars, dtype=np.int32).reshape(-1)
    if union_vars.size == 0:
        return 0.0
    bs = max(1, int(block_size))
    block_ids = (union_vars.astype(np.int64) // bs).astype(np.int64, copy=False)
    counts = np.bincount(block_ids, minlength=int(block_ids.max()) + 1)
    if counts.size == 0:
        return 0.0
    return float(counts.max()) / float(max(1, union_vars.size))


def _select_search_vars_ai_rank_roi(union_vars: np.ndarray,
                                     unsat_checks: np.ndarray,
                                     code_cfg: CodeConfig,
                                     llr_for_sort: np.ndarray,
                                     L: int,
                                     cfg: ClusterGrandConfig,
                                     llr_snapshot: Optional[np.ndarray] = None,
                                     llr_channel: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """ROI-aware lightweight AI ranker.

    This is a tiny distilled controller on top of ``ai_rank``. It uses cheap
    context cues from the current residual to choose one of a few weight/budget
    profiles. The goal is to preserve rescue power on compact conflict-rich
    cases, while shrinking the search set on diffuse cases that are unlikely to
    be fixed by local GRAND.
    """
    union_vars = np.asarray(union_vars, dtype=np.int32)
    unsat_checks = np.asarray(unsat_checks, dtype=np.int32)
    L = int(max(L, 0))

    if L <= 0 or union_vars.size == 0:
        return np.array([], dtype=np.int32), {
            "selection_mode_used": "ai_rank_roi",
            "sv_seeded_count": 0,
            "sv_neighbor_visits": 0,
            "sv_score_len": int(union_vars.size),
            "ai_rank_roi_profile": "empty",
            "ai_rank_roi_budget": 0,
        }

    abs_llr_union = np.abs(llr_for_sort[union_vars]).astype(np.float64, copy=False)
    eps = float(getattr(cfg, "sv_epsilon", 1e-3) or 1e-3)
    k_cc = max(0, int(getattr(cfg, "sv_check_cover_k", 0) or 0))

    vote_counts = np.zeros(union_vars.size, dtype=np.int32)
    var_to_local = {int(v): i for i, v in enumerate(union_vars.tolist())}
    sv_neighbor_visits = 0

    seed_list: List[int] = []
    seeded = set()
    for j in unsat_checks:
        local_neighbors: List[int] = []
        for v in code_cfg.checks_to_vars[int(j)]:
            v_int = int(v)
            loc = var_to_local.get(v_int, None)
            if loc is None:
                continue
            vote_counts[loc] += 1
            local_neighbors.append(loc)
            sv_neighbor_visits += 1

        if k_cc > 0 and local_neighbors:
            local_neighbors_sorted = sorted(
                local_neighbors,
                key=lambda loc: (float(abs_llr_union[loc]), -int(vote_counts[loc]), int(union_vars[loc]))
            )
            take = min(k_cc, len(local_neighbors_sorted))
            for loc in local_neighbors_sorted[:take]:
                v_int = int(union_vars[loc])
                if v_int not in seeded:
                    seed_list.append(v_int)
                    seeded.add(v_int)

    max_vote = max(1, int(vote_counts.max()) if vote_counts.size else 0)
    inv_llr = 1.0 / (abs_llr_union + eps)
    max_inv_llr = float(inv_llr.max()) if inv_llr.size else 1.0
    if max_inv_llr <= 0.0:
        max_inv_llr = 1.0

    vote_norm = vote_counts.astype(np.float64) / float(max_vote)
    inv_llr_norm = inv_llr / float(max_inv_llr)
    degs = np.asarray([max(1, len(code_cfg.vars_to_checks[int(v)])) for v in union_vars.tolist()], dtype=np.float64)
    density = np.clip(vote_counts.astype(np.float64) / degs, 0.0, 1.0)

    disagree = np.zeros(union_vars.size, dtype=np.float64)
    weak_disagree = np.zeros(union_vars.size, dtype=np.float64)
    weak_thr = float(getattr(cfg, "ai_rank_roi_weak_llr_abs_cap", 2.5) or 2.5)
    if abs_llr_union.size > 0:
        try:
            qthr = float(np.quantile(abs_llr_union, float(getattr(cfg, "ai_rank_roi_weak_llr_quantile", 0.30) or 0.30)))
        except Exception:
            qthr = float(np.median(abs_llr_union)) if abs_llr_union.size else weak_thr
        weak_thr = min(weak_thr, max(0.5, qthr))
    weak_mask = (abs_llr_union <= weak_thr)

    if llr_snapshot is not None and llr_channel is not None:
        llr_snapshot = np.asarray(llr_snapshot, dtype=np.float32)
        llr_channel = np.asarray(llr_channel, dtype=np.float32)
        snap_sign = np.sign(llr_snapshot[union_vars]).astype(np.int8, copy=False)
        chan_sign = np.sign(llr_channel[union_vars]).astype(np.int8, copy=False)
        disagree = ((snap_sign * chan_sign) < 0).astype(np.float64, copy=False)
        weak_disagree = (disagree > 0.0) & weak_mask
        weak_disagree = weak_disagree.astype(np.float64, copy=False)

    block_conc = _ai_rank_roi_block_concentration(union_vars, int(getattr(cfg, "ai_rank_roi_block_size", 64) or 64))
    low_llr_frac = float(np.mean(weak_mask)) if weak_mask.size else 0.0
    disagreement_rate = float(np.mean(disagree)) if disagree.size else 0.0
    conflict = float(np.clip(0.65 * disagreement_rate + 0.35 * (float(np.mean(weak_disagree)) if weak_disagree.size else 0.0), 0.0, 1.0))

    union_size = int(union_vars.size)
    diffuse = (
        float(union_size >= int(getattr(cfg, "ai_rank_roi_diffuse_union_size", 208) or 208))
        and block_conc <= float(getattr(cfg, "ai_rank_roi_diffuse_block_concentration", 0.08) or 0.08)
    )
    compact = block_conc >= float(getattr(cfg, "ai_rank_roi_compact_block_concentration", 0.11) or 0.11)

    w_vote = float(getattr(cfg, "ai_rank_vote_weight", 1.0) or 1.0)
    w_llr = float(getattr(cfg, "ai_rank_llr_weight", 0.85) or 0.85)
    w_dis = float(getattr(cfg, "ai_rank_disagreement_weight", 0.55) or 0.55)
    w_den = float(getattr(cfg, "ai_rank_density_weight", 0.35) or 0.35)
    profile = "base"
    L_eff = int(L)

    if diffuse:
        profile = "diffuse_prune"
        L_eff = max(8, min(int(L), int(math.ceil(float(L) * float(getattr(cfg, "ai_rank_roi_diffuse_l_scale", 0.70) or 0.70)))))
        w_vote *= 1.05
        w_llr *= 0.75
        w_dis *= 0.60
        w_den *= 0.65
    elif compact and conflict >= 0.08:
        profile = "local_conflict"
        w_vote *= 0.95
        w_llr *= 0.80
        w_dis *= 1.35
        w_den *= 1.25
    elif compact and low_llr_frac >= 0.24:
        profile = "uncertain_local"
        w_vote *= 0.95
        w_llr *= 1.20
        w_dis *= 1.05
        w_den *= 1.10
    else:
        profile = "balanced"
        w_vote *= 1.00
        w_llr *= 0.95
        w_dis *= 1.05
        w_den *= 0.95

    ai_score = (
        w_vote * vote_norm
        + w_llr * inv_llr_norm
        + w_dis * disagree * inv_llr_norm
        + w_den * density
        + float(getattr(cfg, "ai_rank_roi_local_conflict_bonus", 0.30) or 0.30) * block_conc * weak_disagree * inv_llr_norm
    )

    global_order = sorted(
        range(union_vars.size),
        key=lambda loc: (-float(ai_score[loc]), float(abs_llr_union[loc]), -int(vote_counts[loc]), int(union_vars[loc]))
    )

    if profile == "local_conflict":
        conflict_order = sorted(
            range(union_vars.size),
            key=lambda loc: (-float(weak_disagree[loc]), -float(disagree[loc]), float(abs_llr_union[loc]), int(union_vars[loc]))
        )
        for loc in conflict_order[: min(8, union_vars.size)]:
            v_int = int(union_vars[loc])
            if v_int not in seeded:
                seed_list.append(v_int)
                seeded.add(v_int)

    selected: List[int] = []
    seen = set()
    for v_int in seed_list:
        if len(selected) >= L_eff:
            break
        if v_int not in seen:
            selected.append(int(v_int))
            seen.add(int(v_int))

    for loc in global_order:
        if len(selected) >= L_eff:
            break
        v_int = int(union_vars[loc])
        if v_int not in seen:
            selected.append(v_int)
            seen.add(v_int)

    return np.asarray(selected[:L_eff], dtype=np.int32), {
        "selection_mode_used": "ai_rank_roi",
        "sv_seeded_count": int(min(len(seed_list), L_eff)),
        "sv_neighbor_visits": int(sv_neighbor_visits),
        "sv_score_len": int(union_vars.size),
        "ai_rank_roi_profile": str(profile),
        "ai_rank_roi_budget": int(L_eff),
    }


def _select_search_vars_ai_window_roi(union_vars: np.ndarray,
                                      unsat_checks: np.ndarray,
                                      code_cfg: CodeConfig,
                                      llr_for_sort: np.ndarray,
                                      L: int,
                                      cfg: ClusterGrandConfig,
                                      llr_snapshot: Optional[np.ndarray] = None,
                                      llr_channel: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """Window-aware lightweight AI ranker.

    It first scores variables using the same cheap cues as ``ai_rank_roi``.
    Then it chooses a tiny number of high-value windows (blocks) and restricts
    GRAND to those windows only. This is intended for the diffuse residuals seen
    in the current data, where the global union is wide but the true rescue ROI
    may still be concentrated in a few sub-blocks.
    """
    union_vars = np.asarray(union_vars, dtype=np.int32)
    unsat_checks = np.asarray(unsat_checks, dtype=np.int32)
    L = int(max(L, 0))

    if L <= 0 or union_vars.size == 0:
        return np.array([], dtype=np.int32), {
            "selection_mode_used": "ai_window_roi",
            "sv_seeded_count": 0,
            "sv_neighbor_visits": 0,
            "sv_score_len": int(union_vars.size),
            "ai_window_profile": "empty",
            "ai_window_blocks_used": 0,
            "ai_window_budget": 0,
        }

    abs_llr_union = np.abs(llr_for_sort[union_vars]).astype(np.float64, copy=False)
    eps = float(getattr(cfg, "sv_epsilon", 1e-3) or 1e-3)
    vote_counts = np.zeros(union_vars.size, dtype=np.int32)
    var_to_local = {int(v): i for i, v in enumerate(union_vars.tolist())}
    sv_neighbor_visits = 0

    for j in unsat_checks:
        for v in code_cfg.checks_to_vars[int(j)]:
            loc = var_to_local.get(int(v), None)
            if loc is None:
                continue
            vote_counts[loc] += 1
            sv_neighbor_visits += 1

    max_vote = max(1, int(vote_counts.max()) if vote_counts.size else 0)
    inv_llr = 1.0 / (abs_llr_union + eps)
    max_inv_llr = float(inv_llr.max()) if inv_llr.size else 1.0
    if max_inv_llr <= 0.0:
        max_inv_llr = 1.0

    vote_norm = vote_counts.astype(np.float64) / float(max_vote)
    inv_llr_norm = inv_llr / float(max_inv_llr)
    degs = np.asarray([max(1, len(code_cfg.vars_to_checks[int(v)])) for v in union_vars.tolist()], dtype=np.float64)
    density = np.clip(vote_counts.astype(np.float64) / degs, 0.0, 1.0)

    disagree = np.zeros(union_vars.size, dtype=np.float64)
    weak_disagree = np.zeros(union_vars.size, dtype=np.float64)
    weak_thr = float(getattr(cfg, "ai_rank_roi_weak_llr_abs_cap", 2.5) or 2.5)
    if abs_llr_union.size > 0:
        try:
            qthr = float(np.quantile(abs_llr_union, float(getattr(cfg, "ai_rank_roi_weak_llr_quantile", 0.30) or 0.30)))
        except Exception:
            qthr = float(np.median(abs_llr_union)) if abs_llr_union.size else weak_thr
        weak_thr = min(weak_thr, max(0.5, qthr))
    weak_mask = (abs_llr_union <= weak_thr)

    if llr_snapshot is not None and llr_channel is not None:
        llr_snapshot = np.asarray(llr_snapshot, dtype=np.float32)
        llr_channel = np.asarray(llr_channel, dtype=np.float32)
        snap_sign = np.sign(llr_snapshot[union_vars]).astype(np.int8, copy=False)
        chan_sign = np.sign(llr_channel[union_vars]).astype(np.int8, copy=False)
        disagree = ((snap_sign * chan_sign) < 0).astype(np.float64, copy=False)
        weak_disagree = ((disagree > 0.0) & weak_mask).astype(np.float64, copy=False)

    block_size = max(1, int(getattr(cfg, "ai_window_block_size", getattr(cfg, "ai_rank_roi_block_size", 64)) or 64))
    block_ids = (union_vars.astype(np.int64) // block_size).astype(np.int64, copy=False)
    block_conc = _ai_rank_roi_block_concentration(union_vars, block_size)

    ai_score = (
        float(getattr(cfg, "ai_rank_vote_weight", 1.0) or 1.0) * vote_norm
        + float(getattr(cfg, "ai_rank_llr_weight", 0.85) or 0.85) * inv_llr_norm
        + float(getattr(cfg, "ai_rank_disagreement_weight", 0.55) or 0.55) * disagree * inv_llr_norm
        + float(getattr(cfg, "ai_rank_density_weight", 0.35) or 0.35) * density
        + float(getattr(cfg, "ai_rank_roi_local_conflict_bonus", 0.30) or 0.30) * block_conc * weak_disagree * inv_llr_norm
    )

    unique_blocks = np.unique(block_ids)
    block_score_map: Dict[int, float] = {}
    block_conf_map: Dict[int, float] = {}
    block_density_map: Dict[int, float] = {}
    for b in unique_blocks.tolist():
        mask = (block_ids == int(b))
        locs = np.flatnonzero(mask)
        if locs.size == 0:
            continue
        loc_scores = ai_score[locs]
        loc_conf = float(np.mean(weak_disagree[locs])) if locs.size else 0.0
        loc_den = float(np.mean(density[locs])) if locs.size else 0.0
        top_take = min(8, int(locs.size))
        top_sum = float(np.sort(loc_scores)[-top_take:].sum()) if top_take > 0 else 0.0
        score = top_sum + float(getattr(cfg, "ai_window_block_score_conflict_bonus", 0.35) or 0.35) * loc_conf + float(getattr(cfg, "ai_window_block_score_density_bonus", 0.20) or 0.20) * loc_den
        block_score_map[int(b)] = float(score)
        block_conf_map[int(b)] = float(loc_conf)
        block_density_map[int(b)] = float(loc_den)

    union_size = int(union_vars.size)
    diffuse = (
        float(union_size >= int(getattr(cfg, "ai_rank_roi_diffuse_union_size", 208) or 208))
        and block_conc <= float(getattr(cfg, "ai_rank_roi_diffuse_block_concentration", 0.08) or 0.08)
    )
    compact = block_conc >= float(getattr(cfg, "ai_window_compact_single_threshold", 0.18) or 0.18)

    top_blocks = max(1, int(getattr(cfg, "ai_window_top_blocks", 2) or 2))
    profile = "pair"
    if compact:
        top_blocks = 1
        profile = "single_compact"
    elif diffuse:
        top_blocks = min(max(2, top_blocks + int(getattr(cfg, "ai_window_diffuse_extra_blocks", 1) or 1)), 4)
        profile = "diffuse_multi"
    else:
        profile = "pair_balanced" if top_blocks > 1 else "single"

    block_order = sorted(unique_blocks.tolist(), key=lambda b: (-float(block_score_map.get(int(b), 0.0)), -float(block_conf_map.get(int(b), 0.0)), int(b)))
    chosen_blocks: List[int] = []
    seen_blocks = set()
    for b in block_order:
        if len(chosen_blocks) >= top_blocks:
            break
        b_int = int(b)
        if b_int not in seen_blocks:
            chosen_blocks.append(b_int)
            seen_blocks.add(b_int)

    nb = max(0, int(getattr(cfg, "ai_window_neighbor_blocks", 0) or 0))
    if nb > 0:
        for b in list(chosen_blocks):
            for delta in range(1, nb + 1):
                for nbid in (b - delta, b + delta):
                    if nbid in block_score_map and nbid not in seen_blocks:
                        chosen_blocks.append(int(nbid))
                        seen_blocks.add(int(nbid))

    chosen_mask = np.isin(block_ids, np.asarray(chosen_blocks, dtype=np.int64))
    chosen_locs = np.flatnonzero(chosen_mask)
    if chosen_locs.size == 0:
        chosen_locs = np.arange(union_vars.size, dtype=np.int32)
        profile = "fallback_all"

    target_budget = min(int(L), int(chosen_locs.size))
    min_target = max(8, min(int(L), int(max(1, len(chosen_blocks)) * 8)))
    if target_budget < min_target and len(chosen_blocks) < len(block_order):
        for b in block_order[len(chosen_blocks):]:
            add_mask = (block_ids == int(b))
            add_locs = np.flatnonzero(add_mask)
            if add_locs.size == 0:
                continue
            chosen_locs = np.unique(np.concatenate([chosen_locs.astype(np.int32, copy=False), add_locs.astype(np.int32, copy=False)])).astype(np.int32, copy=False)
            chosen_blocks.append(int(b))
            target_budget = min(int(L), int(chosen_locs.size))
            if target_budget >= min_target:
                break

    seed_per_block = max(0, int(getattr(cfg, "ai_window_local_seed_per_block", 4) or 4))
    seeds: List[int] = []
    seed_seen = set()
    for b in chosen_blocks:
        locs = [int(loc) for loc in chosen_locs.tolist() if int(block_ids[int(loc)]) == int(b)]
        locs_sorted = sorted(locs, key=lambda loc: (-float(weak_disagree[loc]), -float(ai_score[loc]), float(abs_llr_union[loc]), int(union_vars[loc])))
        for loc in locs_sorted[: min(seed_per_block, len(locs_sorted))]:
            v_int = int(union_vars[loc])
            if v_int not in seed_seen:
                seeds.append(v_int)
                seed_seen.add(v_int)

    chosen_order = sorted(chosen_locs.tolist(), key=lambda loc: (-float(weak_disagree[int(loc)]), -float(ai_score[int(loc)]), float(abs_llr_union[int(loc)]), -int(vote_counts[int(loc)]), int(union_vars[int(loc)])))
    selected: List[int] = []
    seen = set()
    for v_int in seeds:
        if len(selected) >= target_budget:
            break
        if v_int not in seen:
            selected.append(int(v_int))
            seen.add(int(v_int))
    for loc in chosen_order:
        if len(selected) >= target_budget:
            break
        v_int = int(union_vars[int(loc)])
        if v_int not in seen:
            selected.append(v_int)
            seen.add(v_int)

    return np.asarray(selected[:target_budget], dtype=np.int32), {
        "selection_mode_used": "ai_window_roi",
        "sv_seeded_count": int(min(len(seeds), target_budget)),
        "sv_neighbor_visits": int(sv_neighbor_visits),
        "sv_score_len": int(union_vars.size),
        "ai_window_profile": str(profile),
        "ai_window_blocks_used": int(len(chosen_blocks)),
        "ai_window_budget": int(target_budget),
    }


def _select_search_vars_ai_mix_roi(union_vars: np.ndarray,
                                   unsat_checks: np.ndarray,
                                   code_cfg: CodeConfig,
                                   llr_for_sort: np.ndarray,
                                   L: int,
                                   cfg: ClusterGrandConfig,
                                   llr_snapshot: Optional[np.ndarray] = None,
                                   llr_channel: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """Mixture-of-experts ROI selector.

    Combines a local window expert with a global ROI rank expert.
    This is aimed at the measured hybrid failures in the current data: the
    residuals are usually diffuse, so a pure local window often misses the real
    rescue region, while a pure global list spends too much budget on scattered
    bits. We therefore allocate part of the search budget to each expert and
    interleave their strongest candidates.
    """
    union_vars = np.asarray(union_vars, dtype=np.int32)
    L = int(max(L, 0))
    if L <= 0 or union_vars.size == 0:
        return np.array([], dtype=np.int32), {
            "selection_mode_used": "ai_mix_roi",
            "sv_seeded_count": 0,
            "sv_neighbor_visits": 0,
            "sv_score_len": int(union_vars.size),
            "ai_mix_profile": "empty",
            "ai_mix_local_budget": 0,
            "ai_mix_global_budget": 0,
        }

    block_size = max(1, int(getattr(cfg, "ai_window_block_size", getattr(cfg, "ai_rank_roi_block_size", 64)) or 64))
    block_conc = _ai_rank_roi_block_concentration(union_vars, block_size)
    union_size = int(union_vars.size)
    diffuse = (
        float(union_size >= int(getattr(cfg, "ai_rank_roi_diffuse_union_size", 208) or 208))
        and block_conc <= float(getattr(cfg, "ai_rank_roi_diffuse_block_concentration", 0.08) or 0.08)
    )
    compact = block_conc >= float(getattr(cfg, "ai_window_compact_single_threshold", 0.18) or 0.18)

    if compact:
        local_share = float(getattr(cfg, "ai_mix_local_share_compact", 0.72) or 0.72)
        profile = "compact_local"
    elif diffuse:
        local_share = float(getattr(cfg, "ai_mix_local_share_diffuse", 0.38) or 0.38)
        profile = "diffuse_global"
    else:
        local_share = float(getattr(cfg, "ai_mix_local_share_balanced", 0.55) or 0.55)
        profile = "balanced"

    local_budget = int(max(8, round(L * np.clip(local_share, 0.20, 0.85))))
    local_budget = min(local_budget, L)
    global_budget = int(max(8, L - local_budget))
    if local_budget + global_budget < L:
        global_budget += int(L - (local_budget + global_budget))

    local_vars, local_meta = _select_search_vars_ai_window_roi(
        union_vars=union_vars,
        unsat_checks=unsat_checks,
        code_cfg=code_cfg,
        llr_for_sort=llr_for_sort,
        L=local_budget,
        cfg=cfg,
        llr_snapshot=llr_snapshot,
        llr_channel=llr_channel,
    )
    global_vars, global_meta = _select_search_vars_ai_rank_roi(
        union_vars=union_vars,
        unsat_checks=unsat_checks,
        code_cfg=code_cfg,
        llr_for_sort=llr_for_sort,
        L=global_budget,
        cfg=cfg,
        llr_snapshot=llr_snapshot,
        llr_channel=llr_channel,
    )

    local_list = [int(v) for v in np.asarray(local_vars, dtype=np.int32).tolist()]
    global_list = [int(v) for v in np.asarray(global_vars, dtype=np.int32).tolist()]
    selected: List[int] = []
    seen = set()

    # Interleave the two experts so the search set remains diverse on diffuse frames.
    max_len = max(len(local_list), len(global_list))
    if diffuse:
        order = (global_list, local_list)
    else:
        order = (local_list, global_list)
    for i in range(max_len):
        for seq in order:
            if i >= len(seq):
                continue
            v_int = int(seq[i])
            if v_int not in seen:
                selected.append(v_int)
                seen.add(v_int)
                if len(selected) >= L:
                    break
        if len(selected) >= L:
            break

    if len(selected) < L:
        for seq in (local_list, global_list):
            for v_int in seq:
                v_int = int(v_int)
                if v_int not in seen:
                    selected.append(v_int)
                    seen.add(v_int)
                    if len(selected) >= L:
                        break
            if len(selected) >= L:
                break

    return np.asarray(selected[:L], dtype=np.int32), {
        "selection_mode_used": "ai_mix_roi",
        "sv_seeded_count": int(max(local_meta.get("sv_seeded_count", 0), global_meta.get("sv_seeded_count", 0))),
        "sv_neighbor_visits": int(max(local_meta.get("sv_neighbor_visits", 0), global_meta.get("sv_neighbor_visits", 0))),
        "sv_score_len": int(union_vars.size),
        "ai_mix_profile": str(profile),
        "ai_mix_local_budget": int(local_budget),
        "ai_mix_global_budget": int(global_budget),
        "ai_window_blocks_used": int(local_meta.get("ai_window_blocks_used", 0)),
    }


def _get_ai_tanner_static_prior(code_cfg: CodeConfig) -> np.ndarray:
    """Return a cached static Tanner-graph prior per variable node.

    This is a very light proxy for an offline teacher trained on the LDPC
    Tanner graph: variables that sit in denser check-overlap / short-cycle
    neighborhoods get a slightly higher prior score. Runtime cost is zero after
    the first build because the vector is cached on ``code_cfg``.
    """
    cached = getattr(code_cfg, "_ai_tanner_static_prior_cache", None)
    if isinstance(cached, np.ndarray) and cached.size == int(code_cfg.N):
        return np.asarray(cached, dtype=np.float32)

    N = int(code_cfg.N)
    pair_overlap: Dict[Tuple[int, int], int] = {}
    check_deg = np.asarray([max(1, len(ch)) for ch in code_cfg.checks_to_vars], dtype=np.float64)

    for v in range(N):
        checks = [int(c) for c in code_cfg.vars_to_checks[v]]
        for i in range(len(checks)):
            ci = int(checks[i])
            for j in range(i + 1, len(checks)):
                cj = int(checks[j])
                key = (ci, cj) if ci < cj else (cj, ci)
                pair_overlap[key] = int(pair_overlap.get(key, 0) + 1)

    prior = np.zeros(N, dtype=np.float64)
    for v in range(N):
        checks = [int(c) for c in code_cfg.vars_to_checks[v]]
        deg_v = float(len(checks))
        cycle_score = 0.0
        overlap_score = 0.0
        inv_check_score = 0.0
        for c in checks:
            inv_check_score += 1.0 / float(max(1.0, check_deg[int(c)]))
        for i in range(len(checks)):
            ci = int(checks[i])
            for j in range(i + 1, len(checks)):
                cj = int(checks[j])
                key = (ci, cj) if ci < cj else (cj, ci)
                ov = float(pair_overlap.get(key, 0))
                overlap_score += ov
                if ov > 1.0:
                    cycle_score += (ov - 1.0)
        prior[v] = 0.55 * cycle_score + 0.20 * overlap_score + 0.15 * deg_v + 0.10 * inv_check_score

    pmin = float(prior.min()) if prior.size else 0.0
    pmax = float(prior.max()) if prior.size else 1.0
    if pmax > pmin:
        prior = (prior - pmin) / float(pmax - pmin)
    else:
        prior[:] = 0.0
    prior = prior.astype(np.float32, copy=False)
    setattr(code_cfg, "_ai_tanner_static_prior_cache", prior)
    return prior


def _select_search_vars_ai_tanner_roi(union_vars: np.ndarray,
                                      unsat_checks: np.ndarray,
                                      code_cfg: CodeConfig,
                                      llr_for_sort: np.ndarray,
                                      L: int,
                                      cfg: ClusterGrandConfig,
                                      llr_snapshot: Optional[np.ndarray] = None,
                                      llr_channel: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """Graph-aware ROI selector with a distilled Tanner prior.

    Runtime behavior:
      1) cheap static Tanner prior (precomputed once);
      2) one/two tiny rounds of message passing on the current unsatisfied-check
         subgraph;
      3) local/global mixture so diffuse residuals still get a global scout.

    This is written to approximate a future offline-trained Tanner-GNN teacher,
    but its live inference path stays lightweight.
    """
    union_vars = np.asarray(union_vars, dtype=np.int32)
    unsat_checks = np.asarray(unsat_checks, dtype=np.int32)
    L = int(max(L, 0))
    if L <= 0 or union_vars.size == 0:
        return np.array([], dtype=np.int32), {
            "selection_mode_used": "ai_tanner_roi",
            "sv_seeded_count": 0,
            "sv_neighbor_visits": 0,
            "sv_score_len": int(union_vars.size),
            "ai_tanner_profile": "empty",
            "ai_tanner_local_budget": 0,
            "ai_tanner_global_budget": 0,
            "ai_tanner_blocks_used": 0,
        }

    abs_llr_union = np.abs(llr_for_sort[union_vars]).astype(np.float64, copy=False)
    eps = float(getattr(cfg, "sv_epsilon", 1e-3) or 1e-3)
    vote_counts = np.zeros(union_vars.size, dtype=np.int32)
    var_to_local = {int(v): i for i, v in enumerate(union_vars.tolist())}
    sv_neighbor_visits = 0

    for j in unsat_checks:
        for v in code_cfg.checks_to_vars[int(j)]:
            loc = var_to_local.get(int(v), None)
            if loc is None:
                continue
            vote_counts[loc] += 1
            sv_neighbor_visits += 1

    max_vote = max(1, int(vote_counts.max()) if vote_counts.size else 0)
    inv_llr = 1.0 / (abs_llr_union + eps)
    max_inv_llr = float(inv_llr.max()) if inv_llr.size else 1.0
    if max_inv_llr <= 0.0:
        max_inv_llr = 1.0

    vote_norm = vote_counts.astype(np.float64) / float(max_vote)
    inv_llr_norm = inv_llr / float(max_inv_llr)
    var_deg = np.asarray([max(1, len(code_cfg.vars_to_checks[int(v)])) for v in union_vars.tolist()], dtype=np.float64)
    density = np.clip(vote_counts.astype(np.float64) / var_deg, 0.0, 1.0)

    disagree = np.zeros(union_vars.size, dtype=np.float64)
    weak_disagree = np.zeros(union_vars.size, dtype=np.float64)
    weak_thr = float(getattr(cfg, "ai_rank_roi_weak_llr_abs_cap", 2.5) or 2.5)
    if abs_llr_union.size > 0:
        try:
            qthr = float(np.quantile(abs_llr_union, float(getattr(cfg, "ai_rank_roi_weak_llr_quantile", 0.30) or 0.30)))
        except Exception:
            qthr = float(np.median(abs_llr_union)) if abs_llr_union.size else weak_thr
        weak_thr = min(weak_thr, max(0.5, qthr))
    weak_mask = (abs_llr_union <= weak_thr)

    if llr_snapshot is not None and llr_channel is not None:
        llr_snapshot = np.asarray(llr_snapshot, dtype=np.float32)
        llr_channel = np.asarray(llr_channel, dtype=np.float32)
        snap_sign = np.sign(llr_snapshot[union_vars]).astype(np.int8, copy=False)
        chan_sign = np.sign(llr_channel[union_vars]).astype(np.int8, copy=False)
        disagree = ((snap_sign * chan_sign) < 0).astype(np.float64, copy=False)
        weak_disagree = ((disagree > 0.0) & weak_mask).astype(np.float64, copy=False)

    block_size = max(1, int(getattr(cfg, "ai_tanner_block_size", getattr(cfg, "ai_window_block_size", 64)) or 64))
    block_conc = _ai_rank_roi_block_concentration(union_vars, block_size)
    union_size = int(union_vars.size)
    diffuse = (
        float(union_size >= int(getattr(cfg, "ai_tanner_diffuse_union_size", 192) or 192))
        and block_conc <= float(getattr(cfg, "ai_tanner_diffuse_block_concentration", 0.09) or 0.09)
    )
    compact = block_conc >= float(getattr(cfg, "ai_tanner_compact_block_concentration", 0.12) or 0.12)

    static_prior_full = _get_ai_tanner_static_prior(code_cfg)
    static_prior = np.asarray(static_prior_full[union_vars], dtype=np.float64)
    if static_prior.size and float(static_prior.max()) > float(static_prior.min()):
        static_prior = (static_prior - float(static_prior.min())) / float(max(1e-9, float(static_prior.max() - static_prior.min())))
    else:
        static_prior = np.zeros_like(inv_llr_norm)

    # Tiny Tanner-subgraph message passing on the current residual.
    check_core: Dict[int, float] = {}
    check_locals: Dict[int, List[int]] = {}
    for j in unsat_checks.tolist():
        neigh_locs: List[int] = []
        for v in code_cfg.checks_to_vars[int(j)]:
            loc = var_to_local.get(int(v), None)
            if loc is not None:
                neigh_locs.append(int(loc))
        if not neigh_locs:
            continue
        weak_rate = float(np.mean(weak_mask[neigh_locs])) if neigh_locs else 0.0
        dis_rate = float(np.mean(disagree[neigh_locs])) if neigh_locs else 0.0
        dens_rate = float(np.mean(density[neigh_locs])) if neigh_locs else 0.0
        stat_rate = float(np.mean(static_prior[neigh_locs])) if neigh_locs else 0.0
        check_core[int(j)] = float(0.40 + 0.32 * weak_rate + 0.26 * dis_rate + 0.18 * dens_rate + 0.14 * stat_rate)
        check_locals[int(j)] = neigh_locs

    msg1 = np.zeros(union_vars.size, dtype=np.float64)
    for loc, v_int in enumerate(union_vars.tolist()):
        s = 0.0
        for c in code_cfg.vars_to_checks[int(v_int)]:
            ci = int(c)
            if ci not in check_core:
                continue
            s += float(check_core[ci]) / math.sqrt(float(max(1, len(code_cfg.checks_to_vars[ci]))))
        msg1[loc] = s / math.sqrt(float(max(1, len(code_cfg.vars_to_checks[int(v_int)]))))
    if msg1.size and float(msg1.max()) > float(msg1.min()):
        msg1 = (msg1 - float(msg1.min())) / float(max(1e-9, float(msg1.max() - msg1.min())))

    check_core_2 = {}
    for ci, neigh_locs in check_locals.items():
        avg1 = float(np.mean(msg1[neigh_locs])) if neigh_locs else 0.0
        check_core_2[int(ci)] = float(check_core[ci] * (0.65 + 0.35 * avg1))

    msg2 = np.zeros(union_vars.size, dtype=np.float64)
    for loc, v_int in enumerate(union_vars.tolist()):
        s = 0.0
        for c in code_cfg.vars_to_checks[int(v_int)]:
            ci = int(c)
            if ci not in check_core_2:
                continue
            s += float(check_core_2[ci]) / math.sqrt(float(max(1, len(code_cfg.checks_to_vars[ci]))))
        msg2[loc] = s / math.sqrt(float(max(1, len(code_cfg.vars_to_checks[int(v_int)]))))
    if msg2.size and float(msg2.max()) > float(msg2.min()):
        msg2 = (msg2 - float(msg2.min())) / float(max(1e-9, float(msg2.max() - msg2.min())))

    w_vote = float(getattr(cfg, "ai_rank_vote_weight", 1.00) or 1.00)
    w_llr = float(getattr(cfg, "ai_rank_llr_weight", 0.85) or 0.85)
    w_dis = float(getattr(cfg, "ai_rank_disagreement_weight", 0.55) or 0.55)
    w_den = float(getattr(cfg, "ai_rank_density_weight", 0.35) or 0.35)
    w_msg = float(getattr(cfg, "ai_tanner_message_weight", 1.00) or 1.00)
    w_cycle = float(getattr(cfg, "ai_tanner_cycle_weight", 0.40) or 0.40)
    w_static = float(getattr(cfg, "ai_tanner_static_prior_weight", 0.30) or 0.30)
    conflict_bonus = float(getattr(cfg, "ai_rank_roi_local_conflict_bonus", 0.30) or 0.30)

    graph_score = (
        w_vote * vote_norm
        + w_llr * inv_llr_norm
        + w_dis * disagree * inv_llr_norm
        + w_den * density
        + w_msg * msg2
        + w_cycle * msg1
        + w_static * static_prior
        + conflict_bonus * weak_disagree * (0.55 * msg2 + 0.45 * inv_llr_norm)
    )

    if compact:
        local_share = float(getattr(cfg, "ai_tanner_local_share_compact", 0.66) or 0.66)
        profile = "compact_tanner"
    elif diffuse:
        local_share = float(getattr(cfg, "ai_tanner_local_share_diffuse", 0.30) or 0.30)
        profile = "diffuse_tanner"
    else:
        local_share = float(getattr(cfg, "ai_tanner_local_share_balanced", 0.48) or 0.48)
        profile = "balanced_tanner"

    local_budget = int(max(8, round(float(L) * np.clip(local_share, 0.18, 0.82))))
    local_budget = min(local_budget, L)
    global_budget = max(1, int(L - local_budget))
    if local_budget + global_budget < L:
        global_budget += int(L - (local_budget + global_budget))

    blocks = (union_vars // block_size).astype(np.int64)
    block_to_locs: Dict[int, List[int]] = {}
    for loc, blk in enumerate(blocks.tolist()):
        block_to_locs.setdefault(int(blk), []).append(int(loc))

    block_scores: List[Tuple[float, int]] = []
    for blk, locs in block_to_locs.items():
        locs_arr = np.asarray(locs, dtype=np.int32)
        top_local = np.sort(graph_score[locs_arr])[-min(4, locs_arr.size):]
        top_mean = float(np.mean(top_local)) if top_local.size else 0.0
        dis_mean = float(np.mean(weak_disagree[locs_arr])) if locs_arr.size else 0.0
        stat_mean = float(np.mean(static_prior[locs_arr])) if locs_arr.size else 0.0
        blk_score = top_mean + 0.25 * dis_mean + 0.18 * stat_mean
        block_scores.append((float(blk_score), int(blk)))
    block_scores.sort(key=lambda t: (-float(t[0]), int(t[1])))

    top_blocks = int(getattr(cfg, "ai_tanner_top_blocks", getattr(cfg, "ai_window_top_blocks", 2)) or 2)
    if diffuse:
        top_blocks += int(getattr(cfg, "ai_tanner_diffuse_extra_blocks", getattr(cfg, "ai_window_diffuse_extra_blocks", 1)) or 1)
    if compact:
        top_blocks = max(1, min(2, top_blocks))
    neighbor_blocks = max(0, int(getattr(cfg, "ai_tanner_neighbor_blocks", getattr(cfg, "ai_window_neighbor_blocks", 1)) or 1))

    chosen_blocks: List[int] = []
    chosen_seen = set()
    for _score, blk in block_scores[:max(1, top_blocks)]:
        for b in range(int(blk) - neighbor_blocks, int(blk) + neighbor_blocks + 1):
            if b in block_to_locs and b not in chosen_seen:
                chosen_blocks.append(int(b))
                chosen_seen.add(int(b))

    local_order: List[int] = []
    for blk in chosen_blocks:
        locs = block_to_locs.get(int(blk), [])
        ranked = sorted(locs, key=lambda loc: (-float(graph_score[int(loc)]), float(abs_llr_union[int(loc)]), int(union_vars[int(loc)])))
        local_order.extend(ranked)

    global_order = sorted(range(union_vars.size), key=lambda loc: (-float(graph_score[int(loc)]), float(abs_llr_union[int(loc)]), int(union_vars[int(loc)])))

    seed_count = max(4, int(getattr(cfg, "ai_tanner_top_global_extra", 10) or 10))
    seeds = sorted(range(union_vars.size), key=lambda loc: (-float(weak_disagree[int(loc)]), -float(graph_score[int(loc)]), float(abs_llr_union[int(loc)]), int(union_vars[int(loc)])))[:min(seed_count, union_vars.size)]

    selected: List[int] = []
    seen = set()
    for loc in seeds:
        v_int = int(union_vars[int(loc)])
        if v_int not in seen:
            selected.append(v_int)
            seen.add(v_int)
            if len(selected) >= min(L, seed_count):
                break

    local_taken = 0
    global_taken = 0
    local_seq = local_order if profile != "diffuse_tanner" else global_order
    global_seq = global_order if profile != "diffuse_tanner" else local_order

    li = gi = 0
    while len(selected) < L and (li < len(local_seq) or gi < len(global_seq)):
        if local_taken < local_budget and li < len(local_seq):
            v_int = int(union_vars[int(local_seq[li])])
            li += 1
            if v_int not in seen:
                selected.append(v_int)
                seen.add(v_int)
                local_taken += 1
                if len(selected) >= L:
                    break
        if global_taken < global_budget and gi < len(global_seq):
            v_int = int(union_vars[int(global_seq[gi])])
            gi += 1
            if v_int not in seen:
                selected.append(v_int)
                seen.add(v_int)
                global_taken += 1
                if len(selected) >= L:
                    break
        if (local_taken >= local_budget or li >= len(local_seq)) and (global_taken >= global_budget or gi >= len(global_seq)):
            break

    if len(selected) < L:
        for loc in global_order:
            v_int = int(union_vars[int(loc)])
            if v_int not in seen:
                selected.append(v_int)
                seen.add(v_int)
                if len(selected) >= L:
                    break

    return np.asarray(selected[:L], dtype=np.int32), {
        "selection_mode_used": "ai_tanner_roi",
        "sv_seeded_count": int(min(len(seeds), L)),
        "sv_neighbor_visits": int(sv_neighbor_visits),
        "sv_score_len": int(union_vars.size),
        "ai_tanner_profile": str(profile),
        "ai_tanner_local_budget": int(local_budget),
        "ai_tanner_global_budget": int(global_budget),
        "ai_tanner_blocks_used": int(len(chosen_blocks)),
    }


def _select_search_vars_ai_tanner_subgraph_roi(union_vars: np.ndarray,
                                               unsat_checks: np.ndarray,
                                               code_cfg: CodeConfig,
                                               llr_for_sort: np.ndarray,
                                               L: int,
                                               cfg: ClusterGrandConfig,
                                               llr_snapshot: Optional[np.ndarray] = None,
                                               llr_channel: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """Tanner-subgraph selector: connected core + a small scout budget.

    This approximates a future Tanner-GNN teacher with a very cheap runtime path:
      1) reuse the Tanner-ROI scorer as a teacher-style ordering;
      2) carve out a compact connected subgraph around the best unsatisfied checks
         and variable seeds;
      3) keep a small block-diverse scout budget outside the subgraph.

    The goal is to preserve the useful FER gain of Tanner-aware rescue while
    shrinking search diffuseness and tail latency.
    """
    union_vars = np.asarray(union_vars, dtype=np.int32)
    unsat_checks = np.asarray(unsat_checks, dtype=np.int32)
    L = int(max(L, 0))
    if L <= 0 or union_vars.size == 0:
        return np.array([], dtype=np.int32), {
            "selection_mode_used": "ai_tanner_subgraph_roi",
            "sv_seeded_count": 0,
            "sv_neighbor_visits": 0,
            "sv_score_len": int(union_vars.size),
            "ai_tg2_profile": "empty",
            "ai_tg2_prefilter": 0,
            "ai_tg2_local_budget": 0,
            "ai_tg2_global_budget": 0,
            "ai_tg2_seed_vars": 0,
            "ai_tg2_seed_checks": 0,
            "ai_tg2_local_size": 0,
            "ai_tg2_blocks_used": 0,
        }

    pre_scale = float(getattr(cfg, "ai_tg2_prefilter_scale", 2.4) or 2.4)
    pre_extra = max(8, int(getattr(cfg, "ai_tg2_prefilter_extra", max(16, L)) or max(16, L)))
    pre_cap = int(min(union_vars.size, max(L, int(round(pre_scale * max(L, 1))), int(L + pre_extra))))

    ordered_vars, meta0 = _select_search_vars_ai_tanner_roi(
        union_vars=union_vars,
        unsat_checks=unsat_checks,
        code_cfg=code_cfg,
        llr_for_sort=llr_for_sort,
        L=pre_cap,
        cfg=cfg,
        llr_snapshot=llr_snapshot,
        llr_channel=llr_channel,
    )
    ordered_vars = np.asarray(ordered_vars, dtype=np.int32)
    if ordered_vars.size <= L:
        meta = dict(meta0)
        meta.update({
            "selection_mode_used": "ai_tanner_subgraph_roi",
            "ai_tg2_profile": str(meta0.get("ai_tanner_profile", "teacher_only")),
            "ai_tg2_prefilter": int(ordered_vars.size),
            "ai_tg2_local_budget": int(min(L, ordered_vars.size)),
            "ai_tg2_global_budget": 0,
            "ai_tg2_seed_vars": int(min(ordered_vars.size, L)),
            "ai_tg2_seed_checks": 0,
            "ai_tg2_local_size": int(ordered_vars.size),
            "ai_tg2_blocks_used": 0,
        })
        return ordered_vars[:L], meta

    block_size = max(1, int(getattr(cfg, "ai_tg2_block_size", getattr(cfg, "ai_tanner_block_size", 64)) or 64))
    union_size = int(union_vars.size)
    block_conc = _ai_rank_roi_block_concentration(union_vars, block_size)
    diffuse = (
        float(union_size >= int(getattr(cfg, "ai_tg2_diffuse_union_size", getattr(cfg, "ai_tanner_diffuse_union_size", 192)) or getattr(cfg, "ai_tanner_diffuse_union_size", 192)))
        and block_conc <= float(getattr(cfg, "ai_tg2_diffuse_block_concentration", getattr(cfg, "ai_tanner_diffuse_block_concentration", 0.09)) or getattr(cfg, "ai_tanner_diffuse_block_concentration", 0.09))
    )
    compact = block_conc >= float(getattr(cfg, "ai_tg2_compact_block_concentration", getattr(cfg, "ai_tanner_compact_block_concentration", 0.12)) or getattr(cfg, "ai_tanner_compact_block_concentration", 0.12))

    if compact:
        local_share = float(getattr(cfg, "ai_tg2_local_share_compact", 0.78) or 0.78)
        seed_vars_k = int(getattr(cfg, "ai_tg2_seed_vars_compact", 6) or 6)
        top_checks_k = int(getattr(cfg, "ai_tg2_top_checks_compact", 5) or 5)
        profile = "compact_subgraph"
    elif diffuse:
        local_share = float(getattr(cfg, "ai_tg2_local_share_diffuse", 0.58) or 0.58)
        seed_vars_k = int(getattr(cfg, "ai_tg2_seed_vars_diffuse", 10) or 10)
        top_checks_k = int(getattr(cfg, "ai_tg2_top_checks_diffuse", 8) or 8)
        profile = "diffuse_subgraph"
    else:
        local_share = float(getattr(cfg, "ai_tg2_local_share_balanced", 0.66) or 0.66)
        seed_vars_k = int(getattr(cfg, "ai_tg2_seed_vars_balanced", 8) or 8)
        top_checks_k = int(getattr(cfg, "ai_tg2_top_checks_balanced", 6) or 6)
        profile = "balanced_subgraph"

    ordered_list = [int(v) for v in ordered_vars.tolist()]
    order_rank = {int(v): i for i, v in enumerate(ordered_list)}
    seed_vars = ordered_list[:min(seed_vars_k, len(ordered_list))]
    unsat_set = {int(c) for c in unsat_checks.tolist()}

    check_scores: List[Tuple[float, int]] = []
    for c in unsat_checks.tolist():
        ranked: List[int] = []
        for v in code_cfg.checks_to_vars[int(c)]:
            r = order_rank.get(int(v), None)
            if r is not None:
                ranked.append(int(r))
        if not ranked:
            continue
        ranked.sort()
        top_r = ranked[:min(4, len(ranked))]
        score = float(sum(float(pre_cap - r) for r in top_r)) / float(max(1, len(top_r)))
        score += 0.20 * float(len(top_r))
        if any((ordered_list[r] in seed_vars) for r in top_r):
            score += 0.35
        check_scores.append((float(score), int(c)))
    check_scores.sort(key=lambda t: (-float(t[0]), int(t[1])))

    chosen_checks = {int(c) for _score, c in check_scores[:max(1, top_checks_k)]}
    for v in seed_vars:
        for c in code_cfg.vars_to_checks[int(v)]:
            ci = int(c)
            if ci in unsat_set:
                chosen_checks.add(ci)

    radius = max(1, int(getattr(cfg, "ai_tg2_radius", 1) or 1))
    local_vars = set(int(v) for v in seed_vars)
    frontier_vars = set(int(v) for v in seed_vars)
    frontier_checks = set(int(c) for c in chosen_checks)

    for _hop in range(radius):
        next_vars = set()
        for c in list(frontier_checks):
            for v in code_cfg.checks_to_vars[int(c)]:
                vi = int(v)
                if vi in order_rank:
                    local_vars.add(vi)
                    next_vars.add(vi)
        next_checks = set()
        for v in list(frontier_vars | next_vars):
            for c in code_cfg.vars_to_checks[int(v)]:
                ci = int(c)
                if ci in unsat_set:
                    next_checks.add(ci)
        frontier_vars = next_vars
        frontier_checks = next_checks
        if not frontier_vars and not frontier_checks:
            break

    neighbor_blocks = max(0, int(getattr(cfg, "ai_tg2_neighbor_blocks", 1) or 1))
    chosen_blocks = set()
    for v in seed_vars:
        blk = int(v) // block_size
        for b in range(int(blk) - neighbor_blocks, int(blk) + neighbor_blocks + 1):
            chosen_blocks.add(int(b))
    for v in ordered_list:
        if (int(v) // block_size) in chosen_blocks:
            local_vars.add(int(v))

    local_order = [int(v) for v in ordered_list if int(v) in local_vars]
    global_order = [int(v) for v in ordered_list if int(v) not in local_vars]

    min_local = max(4, int(getattr(cfg, "ai_tg2_min_local_budget", 6) or 6))
    local_budget = int(min(L, max(min_local, round(float(L) * np.clip(local_share, 0.30, 0.90)))))
    local_budget = min(local_budget, len(local_order))
    global_budget = max(0, int(L - local_budget))

    selected: List[int] = []
    seen = set()
    for v in local_order[:local_budget]:
        if v not in seen:
            selected.append(int(v))
            seen.add(int(v))

    used_blocks = {int(v) // block_size for v in selected}
    scout_diverse: List[int] = []
    for v in global_order:
        blk = int(v) // block_size
        if blk not in used_blocks:
            scout_diverse.append(int(v))
            used_blocks.add(blk)
        if len(scout_diverse) >= global_budget:
            break
    if len(scout_diverse) < global_budget:
        for v in global_order:
            vi = int(v)
            if vi not in scout_diverse:
                scout_diverse.append(vi)
            if len(scout_diverse) >= global_budget:
                break

    for v in scout_diverse[:global_budget]:
        if v not in seen:
            selected.append(int(v))
            seen.add(int(v))

    if len(selected) < L:
        for v in ordered_list:
            if int(v) not in seen:
                selected.append(int(v))
                seen.add(int(v))
                if len(selected) >= L:
                    break

    meta = dict(meta0)
    meta.update({
        "selection_mode_used": "ai_tanner_subgraph_roi",
        "ai_tg2_profile": str(profile),
        "ai_tg2_prefilter": int(pre_cap),
        "ai_tg2_local_budget": int(local_budget),
        "ai_tg2_global_budget": int(global_budget),
        "ai_tg2_seed_vars": int(len(seed_vars)),
        "ai_tg2_seed_checks": int(len(chosen_checks)),
        "ai_tg2_local_size": int(len(local_order)),
        "ai_tg2_blocks_used": int(len(chosen_blocks)),
    })
    return np.asarray(selected[:L], dtype=np.int32), meta


def _resolve_sort_llr_vector(llr_snapshot: np.ndarray,
                             llr_channel: Optional[np.ndarray],
                             cfg: ClusterGrandConfig) -> Tuple[np.ndarray, str]:
    """Resolve the LLR vector used for ranking/costing.

    Supported sources:
      - "posterior": use the stage-1 snapshot posterior LLRs
      - "channel"  : use the channel LLRs when available
      - "mixed"    : conservative magnitude = min(|posterior|, |channel|)
                     with posterior sign carried for deterministic ordering
    """
    llr_snapshot = np.asarray(llr_snapshot, dtype=np.float32)
    llr_source = str(getattr(cfg, "llr_source", "posterior") or "posterior").strip().lower()

    if llr_source == "channel":
        if llr_channel is not None:
            return np.asarray(llr_channel, dtype=np.float32), "channel"
        return llr_snapshot, "posterior"

    if llr_source in ("mixed", "hybrid", "minabs"):
        if llr_channel is not None:
            llr_channel = np.asarray(llr_channel, dtype=np.float32)
            abs_mix = np.minimum(np.abs(llr_snapshot), np.abs(llr_channel)).astype(np.float32, copy=False)
            sign_ref = np.sign(llr_snapshot).astype(np.float32, copy=False)
            zero_mask = (sign_ref == 0)
            if np.any(zero_mask):
                sign_ref = sign_ref.copy()
                if llr_channel is not None:
                    sign_ref[zero_mask] = np.sign(llr_channel[zero_mask]).astype(np.float32, copy=False)
                    zero_mask = (sign_ref == 0)
                if np.any(zero_mask):
                    sign_ref[zero_mask] = 1.0
            return (sign_ref * abs_mix).astype(np.float32, copy=False), "mixed"
        return llr_snapshot, "posterior"

    return llr_snapshot, "posterior"


def _auto_pick_peel_candidate_size(L_full: int,
                                   L_search: int,
                                   cfg: ClusterGrandConfig) -> int:
    """Choose L_peel >= L_search, capped by peel_max_bits / L_full."""
    L_full = int(max(L_full, 0))
    L_search = int(max(L_search, 0))
    if L_full <= 0:
        return 0

    ratio = float(getattr(cfg, "peel_candidate_ratio", 1.0) or 1.0)
    ratio = max(1.0, ratio)
    L_peel = int(np.ceil(ratio * max(L_search, 1)))

    peel_max_bits = getattr(cfg, "peel_max_bits", None)
    if isinstance(peel_max_bits, int) and peel_max_bits > 0:
        L_peel = min(L_peel, int(peel_max_bits))

    L_peel = max(L_peel, max(L_search, 1))
    return int(min(L_peel, L_full))


def _select_presolver_vars(union_vars: np.ndarray,
                           unsat_checks: np.ndarray,
                           code_cfg: CodeConfig,
                           llr_for_sort: np.ndarray,
                           L_peel: int,
                           cfg: ClusterGrandConfig,
                           llr_snapshot: Optional[np.ndarray] = None,
                           llr_channel: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """Receiver-3/5 candidate-set construction.

    Base list:
      - uses the configured front-end selection mode (LLR or syndrome-vote)

    Optional strengthening:
      - append a small number of extra plain-LLR candidates so the pre-solver
        is less brittle if the syndrome-vote list misses a true error bit.
      - append a few posterior-vs-channel sign-disagreement bits, which are a
        strong clue for trapping-set / pseudo-codeword failures.
    """
    union_vars = np.asarray(union_vars, dtype=np.int32)
    L_peel = int(max(L_peel, 0))
    if L_peel <= 0 or union_vars.size == 0:
        return np.array([], dtype=np.int32), {
            "selection_mode_used": str(getattr(cfg, "selection_mode", "llr")),
            "sv_seeded_count": 0,
            "sv_neighbor_visits": 0,
            "sv_score_len": int(union_vars.size),
            "peel_extra_llr_added": 0,
            "disagreement_added": 0,
        }

    selection_mode = str(getattr(cfg, "selection_mode", "llr") or "llr").strip().lower()
    if selection_mode in ("ai_tanner_subgraph_roi", "aitg2", "tanner_subgraph_roi", "receiver9_tg2"):
        base_vars, meta = _select_search_vars_ai_tanner_subgraph_roi(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=code_cfg,
            llr_for_sort=llr_for_sort,
            L=L_peel,
            cfg=cfg,
            llr_snapshot=llr_snapshot,
            llr_channel=llr_channel,
        )
    elif selection_mode in ("ai_tanner_roi", "aitg", "tanner_roi", "receiver9_tg"):
        base_vars, meta = _select_search_vars_ai_tanner_roi(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=code_cfg,
            llr_for_sort=llr_for_sort,
            L=L_peel,
            cfg=cfg,
            llr_snapshot=llr_snapshot,
            llr_channel=llr_channel,
        )
    elif selection_mode in ("ai_mix_roi", "aimix", "mix_roi", "receiver9_mix"):
        base_vars, meta = _select_search_vars_ai_mix_roi(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=code_cfg,
            llr_for_sort=llr_for_sort,
            L=L_peel,
            cfg=cfg,
            llr_snapshot=llr_snapshot,
            llr_channel=llr_channel,
        )
    elif selection_mode in ("ai_window_roi", "aiwindow", "window_roi", "receiver9_window"):
        base_vars, meta = _select_search_vars_ai_window_roi(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=code_cfg,
            llr_for_sort=llr_for_sort,
            L=L_peel,
            cfg=cfg,
            llr_snapshot=llr_snapshot,
            llr_channel=llr_channel,
        )
    elif selection_mode in ("ai_rank_roi", "airoi", "roi_rank", "receiver9_roi"):
        base_vars, meta = _select_search_vars_ai_rank_roi(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=code_cfg,
            llr_for_sort=llr_for_sort,
            L=L_peel,
            cfg=cfg,
            llr_snapshot=llr_snapshot,
            llr_channel=llr_channel,
        )
    elif selection_mode in ("ai_rank", "ai", "airank", "receiver9"):
        base_vars, meta = _select_search_vars_ai_rank(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=code_cfg,
            llr_for_sort=llr_for_sort,
            L=L_peel,
            cfg=cfg,
            llr_snapshot=llr_snapshot,
            llr_channel=llr_channel,
        )
    elif selection_mode in ("syndrome_vote", "sv", "receiver2"):
        base_vars, meta = _select_search_vars_syndrome_vote(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=code_cfg,
            llr_for_sort=llr_for_sort,
            L=L_peel,
            cfg=cfg,
        )
    else:
        base_vars, meta = _select_search_vars_llr(
            union_vars=union_vars,
            llr_for_sort=llr_for_sort,
            L=L_peel,
        )

    selected = [int(v) for v in np.asarray(base_vars, dtype=np.int32).tolist()]
    seen = set(selected)

    extra_llr = max(0, int(getattr(cfg, "peel_extra_llr_bits", 0) or 0))
    extra_added = 0
    if extra_llr > 0 and len(selected) < L_peel:
        llr_vars, _ = _select_search_vars_llr(
            union_vars=union_vars,
            llr_for_sort=llr_for_sort,
            L=min(int(union_vars.size), int(L_peel + extra_llr)),
        )
        for v in np.asarray(llr_vars, dtype=np.int32).tolist():
            v_int = int(v)
            if v_int not in seen:
                selected.append(v_int)
                seen.add(v_int)
                extra_added += 1
            if len(selected) >= L_peel:
                break

    disagreement_added = 0
    disagree_budget = max(0, int(getattr(cfg, "osd_disagreement_extra_bits", 0) or 0))
    if disagree_budget > 0 and len(selected) < L_peel and llr_snapshot is not None and llr_channel is not None:
        llr_snapshot = np.asarray(llr_snapshot, dtype=np.float32)
        llr_channel = np.asarray(llr_channel, dtype=np.float32)
        snap_sign = np.sign(llr_snapshot[union_vars]).astype(np.int8, copy=False)
        chan_sign = np.sign(llr_channel[union_vars]).astype(np.int8, copy=False)
        disagree_mask = (snap_sign * chan_sign) < 0
        if np.any(disagree_mask):
            disagree_vars = union_vars[disagree_mask]
            disagree_vars_sorted = sorted(
                disagree_vars.tolist(),
                key=lambda v: (
                    float(min(abs(float(llr_snapshot[int(v)])), abs(float(llr_channel[int(v)])))),
                    float(abs(float(llr_for_sort[int(v)]))),
                    int(v),
                ),
            )
            for v in disagree_vars_sorted:
                v_int = int(v)
                if v_int not in seen:
                    selected.append(v_int)
                    seen.add(v_int)
                    disagreement_added += 1
                if len(selected) >= L_peel or disagreement_added >= disagree_budget:
                    break

    out = np.asarray(selected[:L_peel], dtype=np.int32)
    meta = dict(meta)
    meta["peel_extra_llr_added"] = int(extra_added)
    meta["disagreement_added"] = int(disagreement_added)
    return out, meta


def _build_local_subsystem_for_candidate(candidate_vars: np.ndarray,
                                         unsat_checks: np.ndarray,
                                         code_cfg: CodeConfig,
                                         syndrome: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build A, b for H_sub e = syndrome on a localized row set.

    Row set = all checks touched by candidate_vars UNION all unsatisfied checks.
    If a selected candidate set cannot touch some unsatisfied check, that row
    becomes all-zero with RHS=1, and the solver correctly declares failure.
    """
    candidate_vars = np.asarray(candidate_vars, dtype=np.int32)
    unsat_checks = np.asarray(unsat_checks, dtype=np.int32)

    row_set = set(int(j) for j in unsat_checks.tolist())
    for v in candidate_vars.tolist():
        for j in code_cfg.vars_to_checks[int(v)]:
            row_set.add(int(j))

    rows = np.array(sorted(row_set), dtype=np.int32)
    m = int(rows.size)
    n = int(candidate_vars.size)
    if m == 0 or n == 0:
        return np.zeros((0, n), dtype=np.uint8), np.zeros((0,), dtype=np.uint8)

    row_to_idx = {int(j): i for i, j in enumerate(rows.tolist())}
    A = np.zeros((m, n), dtype=np.uint8)
    for c_idx, v in enumerate(candidate_vars.tolist()):
        for j in code_cfg.vars_to_checks[int(v)]:
            r_idx = row_to_idx[int(j)]
            A[r_idx, c_idx] ^= np.uint8(1)

    b = syndrome[rows].astype(np.uint8, copy=True)
    return A, b


def _peel_reduce_system(A: np.ndarray,
                        b: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, int]:
    """Peel degree-1 equations in a binary linear system.

    Returns:
      ok,
      fixed_assignments (len n; -1 = unresolved, else 0/1),
      unresolved_col_idx,
      unresolved_row_idx,
      peel_edge_work
    """
    A = np.asarray(A, dtype=np.uint8).copy()
    b = np.asarray(b, dtype=np.uint8).copy()

    m, n = A.shape
    fixed = np.full(n, -1, dtype=np.int8)
    active_rows = np.ones(m, dtype=np.bool_)
    active_cols = np.ones(n, dtype=np.bool_)

    row_deg = A.sum(axis=1).astype(np.int32, copy=False)
    peel_edge_work = 0

    changed = True
    while changed:
        changed = False

        # Contradiction rows (0 = 1)
        bad_rows = np.flatnonzero(active_rows & (row_deg == 0) & (b != 0))
        if bad_rows.size > 0:
            return False, fixed, np.flatnonzero(active_cols), np.flatnonzero(active_rows), int(peel_edge_work)

        deg1_rows = np.flatnonzero(active_rows & (row_deg == 1))
        if deg1_rows.size == 0:
            break

        changed = True
        for r in deg1_rows.tolist():
            if not bool(active_rows[r]) or int(row_deg[r]) != 1:
                continue

            cols = np.flatnonzero(A[r] & active_cols)
            if cols.size != 1:
                continue
            c = int(cols[0])

            x = int(b[r] & 1)
            fixed[c] = np.int8(x)

            touched_rows = np.flatnonzero(A[:, c] & active_rows)
            peel_edge_work += int(touched_rows.size)

            if x != 0:
                b[touched_rows] ^= np.uint8(1)

            A[touched_rows, c] = np.uint8(0)
            row_deg[touched_rows] -= 1
            active_cols[c] = False
            active_rows[r] = False
            row_deg[r] = 0

    # Final contradiction check
    bad_rows = np.flatnonzero(active_rows & (row_deg == 0) & (b != 0))
    if bad_rows.size > 0:
        return False, fixed, np.flatnonzero(active_cols), np.flatnonzero(active_rows), int(peel_edge_work)

    unresolved_cols = np.flatnonzero(active_cols)
    unresolved_rows = np.flatnonzero(active_rows & (row_deg > 0))
    return True, fixed, unresolved_cols.astype(np.int32), unresolved_rows.astype(np.int32), int(peel_edge_work)


def _gf2_weighted_solve(A: np.ndarray,
                        b: np.ndarray,
                        weights: np.ndarray,
                        max_free_enum: int = 12) -> Tuple[bool, np.ndarray, int, int]:
    """Solve A x = b over GF(2) and choose a low-cost solution.

    If the solution space has free dimension <= max_free_enum, enumerate the
    affine nullspace and pick the minimum weighted cost solution.
    Otherwise return failure so the caller can fall back to GRAND.
    """
    A = np.asarray(A, dtype=np.uint8).copy()
    b = np.asarray(b, dtype=np.uint8).copy()
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)

    m, n = A.shape
    if n == 0:
        ok = bool(np.all((b & 1) == 0))
        return ok, np.zeros((0,), dtype=np.uint8), 0, 0

    xor_ops = 0
    row = 0
    pivot_cols: List[int] = []
    pivot_rows: List[int] = []

    for col in range(n):
        pivot = None
        for r in range(row, m):
            if int(A[r, col]) != 0:
                pivot = r
                break
        if pivot is None:
            continue

        if pivot != row:
            A[[row, pivot], :] = A[[pivot, row], :]
            b[[row, pivot]] = b[[pivot, row]]

        # Full elimination to RREF-style pivot columns
        for r in range(m):
            if r != row and int(A[r, col]) != 0:
                A[r, :] ^= A[row, :]
                b[r] ^= b[row]
                xor_ops += int(n + 1)

        pivot_cols.append(int(col))
        pivot_rows.append(int(row))
        row += 1
        if row >= m:
            break

    # Inconsistency: 0 = 1
    for r in range(m):
        if int(A[r].sum()) == 0 and int(b[r]) != 0:
            return False, np.zeros((n,), dtype=np.uint8), 0, int(xor_ops)

    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n) if c not in pivot_set]
    free_dim = int(len(free_cols))

    # One particular solution (free vars = 0)
    x0 = np.zeros((n,), dtype=np.uint8)
    for prow, pcol in zip(pivot_rows, pivot_cols):
        x0[pcol] = np.uint8(b[prow] & 1)

    if free_dim == 0:
        return True, x0, 0, int(xor_ops)

    if free_dim > int(max_free_enum):
        return False, np.zeros((n,), dtype=np.uint8), free_dim, int(xor_ops)

    pivot_row_for_col = {int(c): int(r) for r, c in zip(pivot_rows, pivot_cols)}
    basis = []
    for fcol in free_cols:
        vec = np.zeros((n,), dtype=np.uint8)
        vec[int(fcol)] = np.uint8(1)
        for pcol in pivot_cols:
            prow = pivot_row_for_col[int(pcol)]
            if int(A[prow, int(fcol)]) != 0:
                vec[int(pcol)] = np.uint8(1)
        basis.append(vec)

    best_x = x0.copy()
    best_cost = float(np.dot(weights, best_x.astype(np.float64)))
    best_weight = int(best_x.sum())

    for mask in range(1, 1 << free_dim):
        x = x0.copy()
        for i in range(free_dim):
            if (mask >> i) & 1:
                x ^= basis[i]
        cost = float(np.dot(weights, x.astype(np.float64)))
        hwt = int(x.sum())
        if (cost < best_cost - 1e-12) or (abs(cost - best_cost) <= 1e-12 and hwt < best_weight):
            best_cost = cost
            best_weight = hwt
            best_x = x

    return True, best_x, free_dim, int(xor_ops)


def _auto_pick_chase_candidate_size(L_full: int,
                                    L_search: int,
                                    cfg: ClusterGrandConfig) -> int:
    """Choose L_chase >= L_search, capped by chase_max_bits / L_full."""
    L_full = int(max(L_full, 0))
    L_search = int(max(L_search, 0))
    if L_full <= 0:
        return 0

    ratio = float(getattr(cfg, "chase_candidate_ratio", 1.0) or 1.0)
    ratio = max(1.0, ratio)
    L_chase = int(np.ceil(ratio * max(L_search, 1)))

    chase_max_bits = getattr(cfg, "chase_max_bits", None)
    if isinstance(chase_max_bits, int) and chase_max_bits > 0:
        L_chase = min(L_chase, int(chase_max_bits))

    L_chase = max(L_chase, max(L_search, 1))
    return int(min(L_chase, L_full))


def _enumerate_ranked_local_patterns(core_vars: np.ndarray,
                                     llr_for_sort: np.ndarray,
                                     base_syndrome: np.ndarray,
                                     base_weight: int,
                                     code_cfg: CodeConfig,
                                     max_weight: int,
                                     max_candidates: int) -> Tuple[List[Tuple[int, float, int, Tuple[int, ...], int, int, int]], Dict[str, int]]:
    """Enumerate Chase-style local patterns and rank them by post-flip syndrome."""
    core_vars = np.asarray(core_vars, dtype=np.int32)
    L = int(core_vars.size)
    max_weight = int(max(1, max_weight))
    max_candidates = int(max(1, max_candidates))

    if L <= 0:
        return [], {
            "chase_patterns_considered": 0,
            "chase_score_edge_visits": 0,
            "chase_score_checks_toggled": 0,
            "chase_score_sum_pattern_weights": 0,
        }

    abs_llr = np.abs(llr_for_sort[core_vars]).astype(np.float64, copy=False)
    patterns: List[Tuple[int, float, int, Tuple[int, ...], int, int, int]] = []
    score_edge_visits = 0
    score_checks_toggled = 0
    score_sumw = 0

    for w in range(1, min(max_weight, L) + 1):
        for comb in itertools.combinations(range(L), w):
            flip_vars = tuple(int(core_vars[i]) for i in comb)
            syn_w, e_cnt, uq_cnt, tg_cnt = _syndrome_weight_and_counts_after_flips_from_base(
                base_syndrome=base_syndrome,
                base_weight=int(base_weight),
                flipped_vars=list(flip_vars),
                code_cfg=code_cfg,
            )
            cost = float(sum(float(abs_llr[i]) for i in comb))
            patterns.append((int(syn_w), float(cost), int(w), flip_vars, int(e_cnt), int(uq_cnt), int(tg_cnt)))
            score_edge_visits += int(e_cnt)
            score_checks_toggled += int(tg_cnt)
            score_sumw += int(w)

    patterns.sort(key=lambda t: (int(t[0]), float(t[1]), int(t[2]), tuple(t[3])))
    if len(patterns) > max_candidates:
        patterns = patterns[:max_candidates]

    return patterns, {
        "chase_patterns_considered": int(len(patterns)),
        "chase_score_edge_visits": int(score_edge_visits),
        "chase_score_checks_toggled": int(score_checks_toggled),
        "chase_score_sum_pattern_weights": int(score_sumw),
    }


def _make_chase_biased_llr(base_llr_channel: np.ndarray,
                           llr_snapshot: np.ndarray,
                           hard_bits_snapshot: np.ndarray,
                           flipped_vars: np.ndarray,
                           gain: float,
                           abs_floor: float) -> np.ndarray:
    """Force a local Chase hypothesis into the channel LLR vector."""
    llr_base = np.asarray(base_llr_channel, dtype=np.float32)
    llr_snapshot = np.asarray(llr_snapshot, dtype=np.float32)
    out = llr_base.astype(np.float32, copy=True)

    flipped_vars = np.asarray(flipped_vars, dtype=np.int32)
    if flipped_vars.size == 0:
        return out

    mags = np.maximum(np.abs(llr_base[flipped_vars]), np.abs(llr_snapshot[flipped_vars])).astype(np.float32, copy=False)
    mags = np.maximum(mags, float(abs_floor)).astype(np.float32, copy=False)
    target_bits = np.asarray(hard_bits_snapshot[flipped_vars] ^ np.uint8(1), dtype=np.uint8)
    signs = np.where(target_bits == 0, 1.0, -1.0).astype(np.float32)
    out[flipped_vars] = (float(gain) * mags * signs).astype(np.float32, copy=False)
    return out


def _run_short_ldpc_finish_pass(llr_channel: np.ndarray,
                                true_bits: np.ndarray,
                                code_cfg: CodeConfig,
                                max_iters: int,
                                alpha: float) -> Dict[str, Any]:
    """Short re-decoding pass used by Receiver 4 candidates."""
    dec_cfg = DecoderConfig(max_iters=int(max_iters), alpha=float(alpha), early_stop=True)
    hard_bits, llr_post, syndrome, iter_used = ldpc_min_sum_decode(
        llr_channel=np.asarray(llr_channel, dtype=np.float32),
        code_cfg=code_cfg,
        dec_cfg=dec_cfg,
        snapshot_iters=[],
        snapshots=None,
    )
    final_bit_errors = int(np.count_nonzero(hard_bits != np.asarray(true_bits, dtype=np.uint8)))
    success = (int(np.asarray(syndrome, dtype=np.uint8).sum()) == 0)
    return {
        "success": bool(success),
        "final_bit_errors": int(final_bit_errors),
        "iter_used": int(iter_used),
        "final_syndrome_weight": int(np.asarray(syndrome, dtype=np.uint8).sum()),
    }


def _run_presolver_chase_list(frame: FrameLog,
                              sim_cfg: SimulationConfig,
                              snapshot_iter: int,
                              cfg: ClusterGrandConfig) -> Optional[ClusterGrandResult]:
    """Receiver 4 pre-solver: Chase-style local list + short LDPC polish."""
    code_cfg = sim_cfg.code

    snaps = frame.snapshots
    syn_snaps = snaps.get("syndrome", {})
    hard_snaps = snaps.get("hard_bits", {})
    llr_snaps = snaps.get("llr", {})

    if (snapshot_iter not in syn_snaps or
        snapshot_iter not in hard_snaps or
        snapshot_iter not in llr_snaps):
        raise ValueError(f"Snapshot at iter {snapshot_iter} is not fully available for Chase pre-solver.")

    syndrome = syn_snaps[snapshot_iter]
    hard_bits_snapshot = hard_snaps[snapshot_iter].copy()
    llr_snapshot = llr_snaps[snapshot_iter]

    initial_syndrome_weight = int(np.asarray(syndrome, dtype=np.uint8).sum())
    if initial_syndrome_weight == 0:
        return None

    diff_init = (hard_bits_snapshot != frame.c_bits)
    initial_bit_errors = int(diff_init.sum())
    unsat_checks = np.flatnonzero(syndrome).astype(np.int32)

    cluster_unsat_edges = 0
    cluster_pair_edges = 0
    if unsat_checks.size > 0:
        for j in unsat_checks:
            neigh = code_cfg.checks_to_vars[int(j)]
            d = int(neigh.size)
            cluster_unsat_edges += d
            if d >= 2:
                cluster_pair_edges += int(d * (d - 1) // 2)

    llr_for_sort, llr_source_used = _resolve_sort_llr_vector(
        llr_snapshot=llr_snapshot,
        llr_channel=getattr(frame, "llr_channel", None),
        cfg=cfg,
    )

    allowed_mask = build_allowed_mask_from_config(frame, sim_cfg, snapshot_iter, cfg)
    clusters = find_variable_clusters_from_syndrome(syndrome, code_cfg)
    if not clusters:
        return None

    union_vars = np.unique(np.concatenate(clusters)).astype(np.int32)
    union_vars = union_vars[allowed_mask[union_vars]]
    L_full = int(union_vars.size)
    if L_full == 0:
        return None

    L_search = _auto_pick_grand_search_size(L_full, cfg)
    L_chase = _auto_pick_chase_candidate_size(L_full, L_search, cfg)
    chase_vars, front_end_meta = _select_presolver_vars(
        union_vars=union_vars,
        unsat_checks=unsat_checks,
        code_cfg=code_cfg,
        llr_for_sort=llr_for_sort,
        L_peel=L_chase,
        cfg=cfg,
        llr_snapshot=llr_snapshot,
        llr_channel=getattr(frame, "llr_channel", None),
    )

    if chase_vars.size == 0:
        return None

    core_max_bits = max(1, int(getattr(cfg, "chase_core_max_bits", 12) or 12))
    core_vars = np.asarray(chase_vars[:min(core_max_bits, chase_vars.size)], dtype=np.int32)
    max_weight = max(1, int(getattr(cfg, "chase_max_weight", 3) or 3))
    max_candidates = max(1, int(getattr(cfg, "chase_max_candidates", 96) or 96))

    patterns, enum_meta = _enumerate_ranked_local_patterns(
        core_vars=core_vars,
        llr_for_sort=llr_for_sort,
        base_syndrome=syndrome,
        base_weight=int(initial_syndrome_weight),
        code_cfg=code_cfg,
        max_weight=max_weight,
        max_candidates=max_candidates,
    )

    gain = float(getattr(cfg, "chase_llr_gain", 2.5) or 2.5)
    abs_floor = float(getattr(cfg, "chase_llr_abs_floor", 4.0) or 4.0)
    extra_iters = max(1, int(getattr(cfg, "chase_ldpc_extra_iters", 6) or 6))
    alpha = float(os.getenv("CHASE_LDPC_ALPHA", os.getenv("GRAND_CTG_LDPC_ALPHA", "0.8")))

    llr_base = np.asarray(getattr(frame, "llr_channel", None), dtype=np.float32) if getattr(frame, "llr_channel", None) is not None else np.asarray(llr_snapshot, dtype=np.float32)

    def _make_attempt_result(success: bool,
                             flipped_vars: Optional[np.ndarray] = None,
                             final_bit_errors: Optional[int] = None,
                             chase_candidates_tested: int = 0,
                             chase_ldpc_total_iters: int = 0,
                             chase_ldpc_num_runs: int = 0,
                             chase_ldpc_num_nonconverged: int = 0,
                             chase_best_syndrome_weight: Optional[int] = None,
                             e_cnt: int = 0,
                             uq_cnt: int = 0,
                             tg_cnt: int = 0) -> ClusterGrandResult:
        if flipped_vars is None:
            flipped_vars = np.array([], dtype=np.int32)
        if final_bit_errors is None:
            final_bit_errors = int(initial_bit_errors)
        if chase_best_syndrome_weight is None:
            chase_best_syndrome_weight = int(initial_syndrome_weight)

        res_local = ClusterGrandResult(
            success=bool(success),
            pattern_weight=int(flipped_vars.size) if bool(success) else -1,
            flipped_vars=np.asarray(flipped_vars, dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=int(initial_syndrome_weight),
            final_syndrome_weight=0 if bool(success) else int(initial_syndrome_weight),
            initial_bit_errors=int(initial_bit_errors),
            final_bit_errors=int(final_bit_errors),
            total_v2c_edge_visits=int(e_cnt),
            total_unique_checks_visited=int(uq_cnt),
            total_unique_checks_toggled=int(tg_cnt),
            patterns_generated=0,
        )
        setattr(res_local, "patterns_evaluated", 0)
        setattr(res_local, "total_v2c_edge_visits_evaluated", 0)
        setattr(res_local, "total_unique_checks_visited_evaluated", 0)
        setattr(res_local, "total_unique_checks_toggled_evaluated", 0)
        setattr(res_local, "union_size", int(L_full))
        setattr(res_local, "search_size", int(L_search))
        setattr(res_local, "llr_sort_len", int(L_full))
        setattr(res_local, "sum_pattern_weights_generated", 0)
        setattr(res_local, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res_local, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res_local, "num_batches_evaluated", 0)
        setattr(res_local, "positions_packed_evaluated", 0)
        setattr(res_local, "batch_size_used", 0)
        setattr(res_local, "llr_source_used", str(llr_source_used))
        setattr(res_local, "selection_mode_used", str(front_end_meta.get("selection_mode_used", getattr(cfg, "selection_mode", "llr"))))
        setattr(res_local, "sv_seeded_count", int(front_end_meta.get("sv_seeded_count", 0)))
        setattr(res_local, "sv_neighbor_visits", int(front_end_meta.get("sv_neighbor_visits", 0)))
        setattr(res_local, "sv_score_len", int(front_end_meta.get("sv_score_len", L_full)))

        setattr(res_local, "pre_solver_mode_used", "chase_list")
        setattr(res_local, "pre_solver_attempted", 1)
        setattr(res_local, "pre_solver_success", 1 if bool(success) else 0)
        setattr(res_local, "peel_candidate_size", int(chase_vars.size))
        setattr(res_local, "peel_residual_vars", int(chase_best_syndrome_weight))
        setattr(res_local, "peel_residual_rows", 0)
        setattr(res_local, "peel_edge_work", 0)
        setattr(res_local, "peel_dense_xor_ops", 0)
        setattr(res_local, "peel_free_dim", 0)
        setattr(res_local, "peel_extra_llr_added", int(front_end_meta.get("peel_extra_llr_added", 0)))
        setattr(res_local, "disagreement_added", int(front_end_meta.get("disagreement_added", 0)))

        setattr(res_local, "chase_candidate_size", int(chase_vars.size))
        setattr(res_local, "chase_core_size", int(core_vars.size))
        setattr(res_local, "chase_patterns_considered", int(enum_meta.get("chase_patterns_considered", 0)))
        setattr(res_local, "chase_candidates_tested", int(chase_candidates_tested))
        setattr(res_local, "chase_score_edge_visits", int(enum_meta.get("chase_score_edge_visits", 0)))
        setattr(res_local, "chase_score_checks_toggled", int(enum_meta.get("chase_score_checks_toggled", 0)))
        setattr(res_local, "chase_score_sum_pattern_weights", int(enum_meta.get("chase_score_sum_pattern_weights", 0)))
        setattr(res_local, "chase_ldpc_total_iters", int(chase_ldpc_total_iters))
        setattr(res_local, "chase_ldpc_num_runs", int(chase_ldpc_num_runs))
        setattr(res_local, "chase_ldpc_num_nonconverged", int(chase_ldpc_num_nonconverged))
        setattr(res_local, "chase_best_syndrome_weight", int(chase_best_syndrome_weight))
        return res_local

    if not patterns:
        return _make_attempt_result(success=False)

    chase_candidates_tested = 0
    chase_ldpc_total_iters = 0
    chase_ldpc_num_runs = 0
    chase_ldpc_num_nonconverged = 0
    best_syn = int(initial_syndrome_weight)

    for syn_w, cost, w, flip_tuple, e_cnt, uq_cnt, tg_cnt in patterns:
        best_syn = min(best_syn, int(syn_w))
        flip_arr = np.asarray(flip_tuple, dtype=np.int32)

        if int(syn_w) == 0:
            final_bit_errors = _bit_errors_after_flips_from_base(
                base_bits=hard_bits_snapshot,
                true_bits=frame.c_bits,
                base_bit_errors=int(initial_bit_errors),
                flipped_vars=list(flip_tuple),
            )
            return _make_attempt_result(
                success=True,
                flipped_vars=flip_arr,
                final_bit_errors=int(final_bit_errors),
                chase_candidates_tested=int(chase_candidates_tested),
                chase_ldpc_total_iters=int(chase_ldpc_total_iters),
                chase_ldpc_num_runs=int(chase_ldpc_num_runs),
                chase_ldpc_num_nonconverged=int(chase_ldpc_num_nonconverged),
                chase_best_syndrome_weight=int(best_syn),
                e_cnt=int(e_cnt),
                uq_cnt=int(uq_cnt),
                tg_cnt=int(tg_cnt),
            )

        llr_biased = _make_chase_biased_llr(
            base_llr_channel=llr_base,
            llr_snapshot=llr_snapshot,
            hard_bits_snapshot=hard_bits_snapshot,
            flipped_vars=flip_arr,
            gain=float(gain),
            abs_floor=float(abs_floor),
        )
        cand = _run_short_ldpc_finish_pass(
            llr_channel=llr_biased,
            true_bits=frame.c_bits,
            code_cfg=code_cfg,
            max_iters=int(extra_iters),
            alpha=float(alpha),
        )
        chase_candidates_tested += 1
        chase_ldpc_num_runs += 1
        chase_ldpc_total_iters += int(cand.get("iter_used", 0))
        if not bool(cand.get("success", False)):
            chase_ldpc_num_nonconverged += 1
        best_syn = min(best_syn, int(cand.get("final_syndrome_weight", best_syn)))

        if bool(cand.get("success", False)):
            return _make_attempt_result(
                success=True,
                flipped_vars=flip_arr,
                final_bit_errors=int(cand.get("final_bit_errors", initial_bit_errors)),
                chase_candidates_tested=int(chase_candidates_tested),
                chase_ldpc_total_iters=int(chase_ldpc_total_iters),
                chase_ldpc_num_runs=int(chase_ldpc_num_runs),
                chase_ldpc_num_nonconverged=int(chase_ldpc_num_nonconverged),
                chase_best_syndrome_weight=int(best_syn),
                e_cnt=int(e_cnt),
                uq_cnt=int(uq_cnt),
                tg_cnt=int(tg_cnt),
            )

    return _make_attempt_result(
        success=False,
        chase_candidates_tested=int(chase_candidates_tested),
        chase_ldpc_total_iters=int(chase_ldpc_total_iters),
        chase_ldpc_num_runs=int(chase_ldpc_num_runs),
        chase_ldpc_num_nonconverged=int(chase_ldpc_num_nonconverged),
        chase_best_syndrome_weight=int(best_syn),
    )




def _auto_pick_osd_candidate_size(L_full: int,
                                  L_search: int,
                                  cfg: ClusterGrandConfig) -> int:
    """Choose L_osd >= L_search, capped by osd_max_bits / L_full."""
    L_full = int(max(L_full, 0))
    L_search = int(max(L_search, 0))
    if L_full <= 0:
        return 0

    ratio = float(getattr(cfg, "osd_candidate_ratio", 1.0) or 1.0)
    ratio = max(1.0, ratio)
    L_osd = int(np.ceil(ratio * max(L_search, 1)))

    osd_max_bits = getattr(cfg, "osd_max_bits", None)
    if isinstance(osd_max_bits, int) and osd_max_bits > 0:
        L_osd = min(L_osd, int(osd_max_bits))

    L_osd = max(L_osd, max(L_search, 1))
    return int(min(L_osd, L_full))



def _gf2_osd_ranked_candidates(A: np.ndarray,
                               b: np.ndarray,
                               reliabilities: np.ndarray,
                               order: int,
                               max_enum_bits: int,
                               max_candidates: int) -> Tuple[List[Tuple[float, int, int, np.ndarray]], Dict[str, int]]:
    """Build a local OSD/MRB candidate list for A x = b over GF(2).

    Columns are reliability-ordered from most to least reliable. We choose a
    most-reliable basis via elimination and enumerate low-order patterns on the
    least-reliable free columns. Returned candidates are in the *original* local
    column order of A.
    """
    A = np.asarray(A, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8).reshape(-1)
    rel = np.asarray(reliabilities, dtype=np.float64).reshape(-1)

    m, n = A.shape
    order = int(max(0, order))
    max_enum_bits = int(max(0, max_enum_bits))
    max_candidates = int(max(1, max_candidates))

    if n == 0:
        return [], {
            "osd_basis_xor_ops": 0,
            "osd_free_dim": 0,
            "osd_enum_bits_used": 0,
            "osd_candidates_considered": 0,
            "osd_sum_candidate_weights": 0,
        }

    perm = np.argsort(-rel, kind="mergesort").astype(np.int32, copy=False)
    A_r = A[:, perm].copy()
    b_r = b.copy()
    rel_r = rel[perm]

    xor_ops = 0
    row = 0
    pivot_cols: List[int] = []
    pivot_rows: List[int] = []

    for col in range(n):
        pivot = None
        for r in range(row, m):
            if int(A_r[r, col]) != 0:
                pivot = r
                break
        if pivot is None:
            continue

        if pivot != row:
            A_r[[row, pivot], :] = A_r[[pivot, row], :]
            b_r[[row, pivot]] = b_r[[pivot, row]]

        for r in range(m):
            if r != row and int(A_r[r, col]) != 0:
                A_r[r, :] ^= A_r[row, :]
                b_r[r] ^= b_r[row]
                xor_ops += int(n + 1)

        pivot_cols.append(int(col))
        pivot_rows.append(int(row))
        row += 1
        if row >= m:
            break

    for r in range(m):
        if int(A_r[r].sum()) == 0 and int(b_r[r]) != 0:
            return [], {
                "osd_basis_xor_ops": int(xor_ops),
                "osd_free_dim": 0,
                "osd_enum_bits_used": 0,
                "osd_candidates_considered": 0,
                "osd_sum_candidate_weights": 0,
            }

    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n) if c not in pivot_set]
    free_dim = int(len(free_cols))

    x0 = np.zeros((n,), dtype=np.uint8)
    for prow, pcol in zip(pivot_rows, pivot_cols):
        x0[pcol] = np.uint8(b_r[prow] & 1)

    pivot_row_for_col = {int(c): int(r) for r, c in zip(pivot_rows, pivot_cols)}
    basis: Dict[int, np.ndarray] = {}
    for fcol in free_cols:
        vec = np.zeros((n,), dtype=np.uint8)
        vec[int(fcol)] = np.uint8(1)
        for pcol in pivot_cols:
            prow = pivot_row_for_col[int(pcol)]
            if int(A_r[prow, int(fcol)]) != 0:
                vec[int(pcol)] = np.uint8(1)
        basis[int(fcol)] = vec

    enum_cols = sorted(free_cols, key=lambda c: (float(rel_r[int(c)]), int(c)))
    if max_enum_bits > 0:
        enum_cols = enum_cols[:max_enum_bits]
    else:
        enum_cols = []

    candidates: List[Tuple[float, int, int, np.ndarray, Tuple[int, ...]]] = []
    seen = set()
    sumw = 0

    def _emit_candidate(x_r: np.ndarray, free_weight: int) -> None:
        nonlocal sumw
        x_r = np.asarray(x_r, dtype=np.uint8)
        key = x_r.tobytes()
        if key in seen:
            return
        seen.add(key)
        x_orig = np.zeros((n,), dtype=np.uint8)
        x_orig[perm] = x_r
        hwt = int(x_orig.sum())
        sumw += int(hwt)
        cost = float(np.dot(rel, x_orig.astype(np.float64)))
        support_tuple = tuple(np.flatnonzero(x_orig).astype(np.int32).tolist())
        candidates.append((cost, hwt, int(free_weight), x_orig, support_tuple))

    _emit_candidate(x0, 0)

    if order > 0 and enum_cols:
        for w in range(1, min(order, len(enum_cols)) + 1):
            for comb in itertools.combinations(enum_cols, w):
                x = x0.copy()
                for fcol in comb:
                    x ^= basis[int(fcol)]
                _emit_candidate(x, int(w))

    candidates.sort(key=lambda t: (float(t[0]), int(t[1]), int(t[2]), t[4]))
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    out = [(float(cost), int(hwt), int(free_weight), np.asarray(x_orig, dtype=np.uint8))
           for cost, hwt, free_weight, x_orig, _support in candidates]
    return out, {
        "osd_basis_xor_ops": int(xor_ops),
        "osd_free_dim": int(free_dim),
        "osd_enum_bits_used": int(len(enum_cols)),
        "osd_candidates_considered": int(len(out)),
        "osd_sum_candidate_weights": int(sumw),
    }



def _make_anchor_restart_llr(base_llr_channel: np.ndarray,
                             llr_snapshot: np.ndarray,
                             hard_bits_snapshot: np.ndarray,
                             candidate_vars: np.ndarray,
                             x_local: np.ndarray,
                             gain: float,
                             abs_floor: float,
                             anchor_all_selected: bool = False) -> Tuple[np.ndarray, int]:
    """Create a full-graph anchored-restart LLR vector from a local hypothesis."""
    llr_base = np.asarray(base_llr_channel, dtype=np.float32)
    llr_snapshot = np.asarray(llr_snapshot, dtype=np.float32)
    hard_bits_snapshot = np.asarray(hard_bits_snapshot, dtype=np.uint8)
    candidate_vars = np.asarray(candidate_vars, dtype=np.int32)
    x_local = np.asarray(x_local, dtype=np.uint8).reshape(-1)

    out = llr_base.astype(np.float32, copy=True)
    if candidate_vars.size == 0 or x_local.size == 0:
        return out, 0

    if bool(anchor_all_selected):
        anchor_vars = candidate_vars
        target_bits = (hard_bits_snapshot[anchor_vars] ^ x_local[:anchor_vars.size]).astype(np.uint8, copy=False)
    else:
        support = np.flatnonzero(x_local).astype(np.int32)
        if support.size == 0:
            return out, 0
        anchor_vars = candidate_vars[support]
        target_bits = (hard_bits_snapshot[anchor_vars] ^ np.uint8(1)).astype(np.uint8, copy=False)

    mags = np.maximum(np.abs(llr_base[anchor_vars]), np.abs(llr_snapshot[anchor_vars])).astype(np.float32, copy=False)
    mags = np.maximum(mags, float(abs_floor)).astype(np.float32, copy=False)
    signs = np.where(target_bits == 0, 1.0, -1.0).astype(np.float32, copy=False)
    out[anchor_vars] = (float(gain) * mags * signs).astype(np.float32, copy=False)
    return out, int(anchor_vars.size)






def _make_group_debias_restart_llr(base_llr_channel: np.ndarray,
                                   llr_snapshot: np.ndarray,
                                   hard_bits_snapshot: np.ndarray,
                                   group_vars: np.ndarray,
                                   flip_vars: np.ndarray,
                                   gain: float,
                                   abs_floor: float,
                                   blend: float = 0.65,
                                   relax: float = 0.45) -> Tuple[np.ndarray, int]:
    """Create a full-graph restart LLR with block/group debiasing plus hard anchors.

    This targets structured CSI-induced reliability bias: the support bits are
    anchored strongly, while the surrounding group is re-initialized with a
    softer channel/posterior blend instead of the original channel LLR alone.
    """
    llr_base = np.asarray(base_llr_channel, dtype=np.float32)
    llr_snapshot = np.asarray(llr_snapshot, dtype=np.float32)
    hard_bits_snapshot = np.asarray(hard_bits_snapshot, dtype=np.uint8)
    group_vars = np.asarray(group_vars, dtype=np.int32)
    flip_vars = np.asarray(flip_vars, dtype=np.int32)

    out = llr_base.astype(np.float32, copy=True)
    if group_vars.size == 0:
        return out, 0

    group_vars = np.unique(group_vars).astype(np.int32, copy=False)
    blend = float(min(max(blend, 0.0), 1.0))
    relax = float(min(max(relax, 0.05), 1.0))

    base_group = ((1.0 - blend) * llr_base[group_vars] + blend * llr_snapshot[group_vars]).astype(np.float32, copy=False)
    if relax < 0.999:
        base_group = (relax * base_group).astype(np.float32, copy=False)

    chan_sign = np.sign(llr_base[group_vars]).astype(np.float32, copy=False)
    snap_sign = np.sign(llr_snapshot[group_vars]).astype(np.float32, copy=False)
    dis = (chan_sign * snap_sign) < 0
    if np.any(dis):
        base_group = base_group.copy()
        mags = np.maximum(np.abs(base_group[dis]), float(abs_floor) * 0.35).astype(np.float32, copy=False)
        pref_sign = np.where(snap_sign[dis] >= 0.0, 1.0, -1.0).astype(np.float32, copy=False)
        base_group[dis] = (pref_sign * mags).astype(np.float32, copy=False)

    out[group_vars] = base_group

    if flip_vars.size > 0:
        flip_vars = np.unique(flip_vars).astype(np.int32, copy=False)
        target_bits = (hard_bits_snapshot[flip_vars] ^ np.uint8(1)).astype(np.uint8, copy=False)
        mags = np.maximum(np.abs(llr_base[flip_vars]), np.abs(llr_snapshot[flip_vars])).astype(np.float32, copy=False)
        mags = np.maximum(mags, float(abs_floor)).astype(np.float32, copy=False)
        signs = np.where(target_bits == 0, 1.0, -1.0).astype(np.float32, copy=False)
        out[flip_vars] = (float(gain) * mags * signs).astype(np.float32, copy=False)

    return out, int(group_vars.size)


def _build_basis_vectors(base_syndrome: np.ndarray,
                         candidate_vars: np.ndarray,
                         unsat_checks: np.ndarray,
                         code_cfg: CodeConfig,
                         llr_for_sort: np.ndarray,
                         cfg: ClusterGrandConfig,
                         llr_snapshot: Optional[np.ndarray] = None,
                         llr_channel: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], Dict[str, int]]:
    """Build a small library of structured basis patterns for Receiver-7.

    The basis patterns are meant to capture *structured* post-BP failures:
      - unsatisfied-check neighbourhoods
      - local ranked windows
      - channel-vs-posterior disagreement groups
      - top singleton suspects

    Receiver-7 then runs GRAND over *combinations* of these basis vectors.
    """
    candidate_vars = np.asarray(candidate_vars, dtype=np.int32)
    if candidate_vars.size == 0:
        return [], {
            "basis_candidate_size": 0,
            "basis_vectors_kept": 0,
            "basis_vectors_considered": 0,
            "basis_score_edge_visits": 0,
            "basis_score_checks_toggled": 0,
            "basis_score_sum_pattern_weights": 0,
        }

    ranked_vars = _soft_rank_candidate_vars(
        candidate_vars=candidate_vars,
        unsat_checks=np.asarray(unsat_checks, dtype=np.int32),
        code_cfg=code_cfg,
        llr_for_sort=llr_for_sort,
        cfg=cfg,
        llr_snapshot=llr_snapshot,
        llr_channel=llr_channel,
    )
    candidate_set = set(int(v) for v in candidate_vars.tolist())
    rank_index = {int(v): i for i, v in enumerate(ranked_vars.tolist())}
    unsat_set = set(int(j) for j in np.asarray(unsat_checks, dtype=np.int32).tolist())

    max_vectors = max(4, int(getattr(cfg, "basis_max_vectors", 18) or 18))
    group_max_bits = max(1, int(getattr(cfg, "basis_group_max_bits", 10) or 10))
    window_max = max(2, int(getattr(cfg, "basis_window_max", 6) or 6))
    window_span = max(2, int(getattr(cfg, "basis_window_span", 18) or 18))
    top_singletons = max(1, int(getattr(cfg, "basis_top_singletons", 8) or 8))
    disagree_groups = max(0, int(getattr(cfg, "basis_disagreement_groups", 4) or 4))
    disagree_chunk = max(1, int(getattr(cfg, "basis_disagreement_chunk", 6) or 6))

    kind_rank = {"singleton": 0, "check": 1, "disagree": 2, "window": 3}
    scored: List[Tuple[int, float, int, int, Tuple[int, ...]]] = []
    seen = set()
    total_edge_visits = 0
    total_checks_toggled = 0
    total_pattern_weights = 0

    def _add_pattern(vars_list: List[int], kind: str) -> None:
        nonlocal total_edge_visits, total_checks_toggled, total_pattern_weights
        uniq = sorted(set(int(v) for v in vars_list if int(v) in candidate_set))
        if not uniq:
            return
        if len(uniq) > group_max_bits and kind != "window":
            uniq = uniq[:group_max_bits]
        tup = tuple(int(v) for v in uniq)
        if (not tup) or (tup in seen):
            return
        seen.add(tup)

        arr = np.asarray(tup, dtype=np.int32)
        syn_w, e_cnt, _uq_cnt, tg_cnt = _syndrome_weight_and_counts_after_flips_from_base(
            base_syndrome=np.asarray(base_syndrome, dtype=np.uint8),
            base_weight=int(np.asarray(base_syndrome, dtype=np.uint8).sum()),
            flipped_vars=[int(v) for v in arr.tolist()],
            code_cfg=code_cfg,
        )
        llr_cost = float(np.sum(np.abs(llr_for_sort[arr])))
        touched_unsat = set()
        for v in arr.tolist():
            for j in code_cfg.vars_to_checks[int(v)]:
                if int(j) in unsat_set:
                    touched_unsat.add(int(j))
        scored.append((
            int(syn_w),
            float(llr_cost / max(1, int(arr.size))),
            int(arr.size),
            int(kind_rank.get(kind, 9) - len(touched_unsat)),
            tup,
        ))
        total_edge_visits += int(e_cnt)
        total_checks_toggled += int(tg_cnt)
        total_pattern_weights += int(arr.size)

    # Top singleton suspects
    for v in ranked_vars[:min(int(ranked_vars.size), top_singletons)].tolist():
        _add_pattern([int(v)], "singleton")

    # Unsatisfied-check neighbourhood basis
    for j in np.asarray(unsat_checks, dtype=np.int32).tolist():
        neigh = [int(v) for v in code_cfg.checks_to_vars[int(j)].tolist() if int(v) in candidate_set]
        if not neigh:
            continue
        neigh.sort(key=lambda v: (rank_index.get(int(v), 10**9), abs(float(llr_for_sort[int(v)])), int(v)))
        _add_pattern(neigh[:group_max_bits], "check")

    # Ranked contiguous windows
    span = min(int(ranked_vars.size), window_span)
    for start in range(span):
        stop_max = min(int(ranked_vars.size), start + window_max)
        for stop in range(start + 2, stop_max + 1):
            _add_pattern([int(v) for v in ranked_vars[start:stop].tolist()], "window")

    # Channel-vs-posterior disagreement groups
    if disagree_groups > 0 and llr_snapshot is not None and llr_channel is not None:
        llr_snapshot_arr = np.asarray(llr_snapshot, dtype=np.float32)
        llr_channel_arr = np.asarray(llr_channel, dtype=np.float32)
        prod = np.sign(llr_snapshot_arr[candidate_vars]) * np.sign(llr_channel_arr[candidate_vars])
        disagree_vars = candidate_vars[prod < 0]
        if disagree_vars.size > 0:
            dis_sorted = sorted(
                disagree_vars.tolist(),
                key=lambda v: (
                    rank_index.get(int(v), 10**9),
                    float(min(abs(float(llr_snapshot_arr[int(v)])), abs(float(llr_channel_arr[int(v)])))),
                    int(v),
                )
            )
            take = dis_sorted[:int(disagree_groups * disagree_chunk)]
            for g in range(disagree_groups):
                chunk = take[g * disagree_chunk : (g + 1) * disagree_chunk]
                if chunk:
                    _add_pattern([int(v) for v in chunk], "disagree")

    vectors_considered = int(len(scored))
    scored.sort(key=lambda t: (int(t[0]), float(t[1]), int(t[2]), int(t[3]), t[4]))
    kept = scored[:max_vectors]
    vectors = [np.asarray(list(tup), dtype=np.int32) for _syn_w, _cost, _sz, _prio, tup in kept]

    return vectors, {
        "basis_candidate_size": int(candidate_vars.size),
        "basis_vectors_kept": int(len(vectors)),
        "basis_vectors_considered": int(vectors_considered),
        "basis_score_edge_visits": int(total_edge_visits),
        "basis_score_checks_toggled": int(total_checks_toggled),
        "basis_score_sum_pattern_weights": int(total_pattern_weights),
    }


def _enumerate_basis_hypotheses(base_syndrome: np.ndarray,
                                base_syndrome_weight: int,
                                basis_vectors: List[np.ndarray],
                                llr_for_sort: np.ndarray,
                                code_cfg: CodeConfig,
                                cfg: ClusterGrandConfig) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, int]]:
    """GRAND over a library of structured basis patterns (Receiver-7)."""
    if not basis_vectors:
        return [], {
            "basis_core_vectors_used": 0,
            "basis_candidates_considered": 0,
            "basis_score_edge_visits": 0,
            "basis_score_checks_toggled": 0,
            "basis_score_sum_pattern_weights": 0,
        }

    max_candidates = max(1, int(getattr(cfg, "basis_max_candidates", 128) or 128))
    core_vectors = max(1, int(getattr(cfg, "basis_core_vectors", min(12, len(basis_vectors))) or 12))
    combo_max = max(1, int(getattr(cfg, "basis_combo_max", 3) or 3))
    core = [np.asarray(v, dtype=np.int32) for v in basis_vectors[:min(len(basis_vectors), core_vectors)]]

    ranked: List[Tuple[int, float, int, Tuple[int, ...], Tuple[int, ...]]] = []
    seen = set()
    total_edge_visits = 0
    total_checks_toggled = 0
    total_pattern_weights = 0

    for r in range(1, min(combo_max, len(core)) + 1):
        for comb in itertools.combinations(range(len(core)), r):
            parity = {}
            context = set()
            for idx in comb:
                vec = core[int(idx)]
                for v in vec.tolist():
                    v_int = int(v)
                    context.add(v_int)
                    parity[v_int] = int(parity.get(v_int, 0)) ^ 1
            flip = tuple(sorted(v for v, bit in parity.items() if int(bit) & 1))
            if (not flip) or (flip in seen):
                continue
            seen.add(flip)
            arr = np.asarray(flip, dtype=np.int32)
            syn_w, e_cnt, _uq_cnt, tg_cnt = _syndrome_weight_and_counts_after_flips_from_base(
                base_syndrome=np.asarray(base_syndrome, dtype=np.uint8),
                base_weight=int(base_syndrome_weight),
                flipped_vars=[int(v) for v in arr.tolist()],
                code_cfg=code_cfg,
            )
            llr_cost = float(np.sum(np.abs(llr_for_sort[arr])))
            ranked.append((
                int(syn_w),
                float(llr_cost),
                int(arr.size),
                tuple(int(v) for v in flip),
                tuple(sorted(int(v) for v in context)),
            ))
            total_edge_visits += int(e_cnt)
            total_checks_toggled += int(tg_cnt)
            total_pattern_weights += int(arr.size)

    total_considered = int(len(ranked))
    ranked.sort(key=lambda t: (int(t[0]), float(t[1]), int(t[2]), t[3], t[4]))
    ranked = ranked[:max_candidates]

    out = [{
        "flip_vars": np.asarray(list(flip), dtype=np.int32),
        "group_vars": np.asarray(list(context), dtype=np.int32),
    } for _syn_w, _cost, _sz, flip, context in ranked]

    return out, {
        "basis_core_vectors_used": int(len(core)),
        "basis_candidates_considered": int(total_considered),
        "basis_score_edge_visits": int(total_edge_visits),
        "basis_score_checks_toggled": int(total_checks_toggled),
        "basis_score_sum_pattern_weights": int(total_pattern_weights),
    }


def _run_presolver_basis_anchor(frame: FrameLog,
                                sim_cfg: SimulationConfig,
                                snapshot_iter: int,
                                cfg: ClusterGrandConfig) -> Optional[ClusterGrandResult]:
    """Receiver-7: GRAND over structured basis patterns + block-debias restarts."""
    code_cfg = sim_cfg.code

    snaps = frame.snapshots
    syn_snaps = snaps.get("syndrome", {})
    hard_snaps = snaps.get("hard_bits", {})
    llr_snaps = snaps.get("llr", {})

    if (snapshot_iter not in syn_snaps or
        snapshot_iter not in hard_snaps or
        snapshot_iter not in llr_snaps):
        raise ValueError(f"Snapshot at iter {snapshot_iter} is not fully available for Receiver-7 pre-solver.")

    syndrome = np.asarray(syn_snaps[snapshot_iter], dtype=np.uint8)
    hard_bits_snapshot = np.asarray(hard_snaps[snapshot_iter], dtype=np.uint8).copy()
    llr_snapshot = np.asarray(llr_snaps[snapshot_iter], dtype=np.float32)

    initial_syndrome_weight = int(syndrome.sum())
    if initial_syndrome_weight == 0:
        return None

    diff_init = (hard_bits_snapshot != np.asarray(frame.c_bits, dtype=np.uint8))
    initial_bit_errors = int(diff_init.sum())
    unsat_checks = np.flatnonzero(syndrome).astype(np.int32)

    cluster_unsat_edges = 0
    cluster_pair_edges = 0
    if unsat_checks.size > 0:
        for j in unsat_checks.tolist():
            neigh = code_cfg.checks_to_vars[int(j)]
            d = int(neigh.size)
            cluster_unsat_edges += d
            if d >= 2:
                cluster_pair_edges += int(d * (d - 1) // 2)

    llr_for_sort, llr_source_used = _resolve_sort_llr_vector(
        llr_snapshot=llr_snapshot,
        llr_channel=getattr(frame, "llr_channel", None),
        cfg=cfg,
    )

    allowed_mask = build_allowed_mask_from_config(frame, sim_cfg, snapshot_iter, cfg)
    clusters = find_variable_clusters_from_syndrome(syndrome, code_cfg)
    if not clusters:
        return None

    union_vars = np.unique(np.concatenate(clusters)).astype(np.int32)
    union_vars = union_vars[allowed_mask[union_vars]]
    L_full = int(union_vars.size)
    if L_full == 0:
        return None

    L_search = _auto_pick_grand_search_size(L_full, cfg)
    basis_ratio = float(getattr(cfg, "basis_candidate_ratio", 3.0) or 3.0)
    L_basis = max(int(L_search), int(np.ceil(max(1.0, basis_ratio) * max(1, int(L_search)))))
    basis_cap = getattr(cfg, "basis_max_bits", None)
    if basis_cap is not None:
        try:
            L_basis = min(L_basis, int(basis_cap))
        except Exception:
            pass
    L_basis = min(L_basis, L_full)

    basis_vars, front_end_meta = _select_presolver_vars(
        union_vars=union_vars,
        unsat_checks=unsat_checks,
        code_cfg=code_cfg,
        llr_for_sort=llr_for_sort,
        L_peel=L_basis,
        cfg=cfg,
        llr_snapshot=llr_snapshot,
        llr_channel=getattr(frame, "llr_channel", None),
    )
    if basis_vars.size == 0:
        return None

    basis_vectors, basis_vec_meta = _build_basis_vectors(
        base_syndrome=syndrome,
        candidate_vars=basis_vars,
        unsat_checks=unsat_checks,
        code_cfg=code_cfg,
        llr_for_sort=llr_for_sort,
        cfg=cfg,
        llr_snapshot=llr_snapshot,
        llr_channel=getattr(frame, "llr_channel", None),
    )

    hypotheses, basis_hyp_meta = _enumerate_basis_hypotheses(
        base_syndrome=syndrome,
        base_syndrome_weight=int(initial_syndrome_weight),
        basis_vectors=basis_vectors,
        llr_for_sort=llr_for_sort,
        code_cfg=code_cfg,
        cfg=cfg,
    )

    llr_base = np.asarray(getattr(frame, "llr_channel", None), dtype=np.float32) if getattr(frame, "llr_channel", None) is not None else np.asarray(llr_snapshot, dtype=np.float32)
    restart_max = max(1, int(getattr(cfg, "restart_max_candidates", 24) or 24))
    restart_iters = max(1, int(getattr(cfg, "restart_ldpc_iters", 14) or 14))
    restart_alpha = float(getattr(cfg, "restart_alpha", 0.78) or 0.78)
    gain1 = float(getattr(cfg, "restart_llr_gain", 4.5) or 4.5)
    gain2 = float(getattr(cfg, "restart_dual_gain", gain1) or gain1)
    abs_floor = float(getattr(cfg, "restart_llr_abs_floor", 6.0) or 6.0)
    debias_blend = float(getattr(cfg, "debias_blend", 0.65) or 0.65)
    debias_relax = float(getattr(cfg, "debias_relax", 0.45) or 0.45)
    second_pass_cap = max(0, min(restart_max, 8))

    def _make_attempt_result(success: bool,
                             flipped_vars: Optional[np.ndarray] = None,
                             final_bit_errors: Optional[int] = None,
                             candidates_tested: int = 0,
                             restart_num_runs: int = 0,
                             restart_total_ldpc_iters: int = 0,
                             restart_num_nonconverged: int = 0,
                             restart_anchor_bits_total: int = 0,
                             restart_best_syndrome_weight: Optional[int] = None) -> ClusterGrandResult:
        if flipped_vars is None:
            flipped_vars = np.array([], dtype=np.int32)
        if final_bit_errors is None:
            final_bit_errors = int(initial_bit_errors)
        if restart_best_syndrome_weight is None:
            restart_best_syndrome_weight = int(initial_syndrome_weight)

        res_local = ClusterGrandResult(
            success=bool(success),
            pattern_weight=int(np.asarray(flipped_vars, dtype=np.int32).size) if bool(success) else -1,
            flipped_vars=np.asarray(flipped_vars, dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=int(initial_syndrome_weight),
            final_syndrome_weight=0 if bool(success) else int(restart_best_syndrome_weight),
            initial_bit_errors=int(initial_bit_errors),
            final_bit_errors=int(final_bit_errors),
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        setattr(res_local, "patterns_evaluated", 0)
        setattr(res_local, "total_v2c_edge_visits_evaluated", 0)
        setattr(res_local, "total_unique_checks_visited_evaluated", 0)
        setattr(res_local, "total_unique_checks_toggled_evaluated", 0)
        setattr(res_local, "union_size", int(L_full))
        setattr(res_local, "search_size", int(L_search))
        setattr(res_local, "llr_sort_len", int(L_full))
        setattr(res_local, "sum_pattern_weights_generated", 0)
        setattr(res_local, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res_local, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res_local, "num_batches_evaluated", 0)
        setattr(res_local, "positions_packed_evaluated", 0)
        setattr(res_local, "batch_size_used", 0)
        setattr(res_local, "llr_source_used", str(llr_source_used))
        setattr(res_local, "selection_mode_used", str(front_end_meta.get("selection_mode_used", getattr(cfg, "selection_mode", "llr"))))
        setattr(res_local, "sv_seeded_count", int(front_end_meta.get("sv_seeded_count", 0)))
        setattr(res_local, "sv_neighbor_visits", int(front_end_meta.get("sv_neighbor_visits", 0)))
        setattr(res_local, "sv_score_len", int(front_end_meta.get("sv_score_len", L_full)))
        setattr(res_local, "pre_solver_mode_used", "basis_anchor")
        setattr(res_local, "pre_solver_attempted", 1)
        setattr(res_local, "pre_solver_success", 1 if bool(success) else 0)
        setattr(res_local, "peel_extra_llr_added", int(front_end_meta.get("peel_extra_llr_added", 0)))
        setattr(res_local, "disagreement_added", int(front_end_meta.get("disagreement_added", 0)))

        setattr(res_local, "chase_candidate_size", int(basis_vec_meta.get("basis_candidate_size", int(basis_vars.size))))
        setattr(res_local, "chase_core_size", int(basis_hyp_meta.get("basis_core_vectors_used", 0)))
        setattr(res_local, "chase_patterns_considered", int(basis_hyp_meta.get("basis_candidates_considered", 0)))
        setattr(res_local, "chase_candidates_tested", int(candidates_tested))
        setattr(res_local, "chase_score_edge_visits", int(basis_vec_meta.get("basis_score_edge_visits", 0)) + int(basis_hyp_meta.get("basis_score_edge_visits", 0)))
        setattr(res_local, "chase_score_checks_toggled", int(basis_vec_meta.get("basis_score_checks_toggled", 0)) + int(basis_hyp_meta.get("basis_score_checks_toggled", 0)))
        setattr(res_local, "chase_score_sum_pattern_weights", int(basis_vec_meta.get("basis_score_sum_pattern_weights", 0)) + int(basis_hyp_meta.get("basis_score_sum_pattern_weights", 0)))
        setattr(res_local, "chase_ldpc_total_iters", 0)
        setattr(res_local, "chase_ldpc_num_runs", 0)
        setattr(res_local, "chase_ldpc_num_nonconverged", 0)
        setattr(res_local, "chase_best_syndrome_weight", int(restart_best_syndrome_weight))

        setattr(res_local, "restart_num_runs", int(restart_num_runs))
        setattr(res_local, "restart_total_ldpc_iters", int(restart_total_ldpc_iters))
        setattr(res_local, "restart_num_nonconverged", int(restart_num_nonconverged))
        setattr(res_local, "restart_anchor_bits_total", int(restart_anchor_bits_total))
        setattr(res_local, "restart_best_syndrome_weight", int(restart_best_syndrome_weight))
        return res_local

    if not hypotheses:
        return _make_attempt_result(success=False)

    candidates_tested = 0
    restart_num_runs = 0
    restart_total_ldpc_iters = 0
    restart_num_nonconverged = 0
    restart_anchor_bits_total = 0
    best_syn = int(initial_syndrome_weight)

    for idx, hyp in enumerate(hypotheses[:restart_max]):
        flip_vars = np.asarray(hyp.get("flip_vars", np.array([], dtype=np.int32)), dtype=np.int32)
        group_vars = np.asarray(hyp.get("group_vars", flip_vars), dtype=np.int32)
        candidates_tested += 1

        syn_w_full, _e_cnt, _uq_cnt, _tg_cnt = _syndrome_weight_and_counts_after_flips_from_base(
            base_syndrome=syndrome,
            base_weight=int(initial_syndrome_weight),
            flipped_vars=[int(v) for v in flip_vars.tolist()],
            code_cfg=code_cfg,
        )
        best_syn = min(best_syn, int(syn_w_full))
        if int(syn_w_full) == 0:
            final_bit_errors = _bit_errors_after_flips_from_base(
                base_bits=hard_bits_snapshot,
                true_bits=np.asarray(frame.c_bits, dtype=np.uint8),
                base_bit_errors=int(initial_bit_errors),
                flipped_vars=[int(v) for v in flip_vars.tolist()],
            )
            return _make_attempt_result(
                success=True,
                flipped_vars=flip_vars,
                final_bit_errors=int(final_bit_errors),
                candidates_tested=int(candidates_tested),
                restart_num_runs=int(restart_num_runs),
                restart_total_ldpc_iters=int(restart_total_ldpc_iters),
                restart_num_nonconverged=int(restart_num_nonconverged),
                restart_anchor_bits_total=int(restart_anchor_bits_total),
                restart_best_syndrome_weight=int(best_syn),
            )

        llr_restart, anchor_bits = _make_group_debias_restart_llr(
            base_llr_channel=llr_base,
            llr_snapshot=llr_snapshot,
            hard_bits_snapshot=hard_bits_snapshot,
            group_vars=group_vars,
            flip_vars=flip_vars,
            gain=float(gain1),
            abs_floor=float(abs_floor),
            blend=float(debias_blend),
            relax=float(debias_relax),
        )
        if anchor_bits > 0:
            cand = _run_short_ldpc_finish_pass(
                llr_channel=llr_restart,
                true_bits=frame.c_bits,
                code_cfg=code_cfg,
                max_iters=int(restart_iters),
                alpha=float(restart_alpha),
            )
            restart_num_runs += 1
            restart_total_ldpc_iters += int(cand.get("iter_used", 0))
            restart_anchor_bits_total += int(anchor_bits)
            if not bool(cand.get("success", False)):
                restart_num_nonconverged += 1
            best_syn = min(best_syn, int(cand.get("final_syndrome_weight", best_syn)))
            if bool(cand.get("success", False)):
                return _make_attempt_result(
                    success=True,
                    flipped_vars=flip_vars,
                    final_bit_errors=int(cand.get("final_bit_errors", initial_bit_errors)),
                    candidates_tested=int(candidates_tested),
                    restart_num_runs=int(restart_num_runs),
                    restart_total_ldpc_iters=int(restart_total_ldpc_iters),
                    restart_num_nonconverged=int(restart_num_nonconverged),
                    restart_anchor_bits_total=int(restart_anchor_bits_total),
                    restart_best_syndrome_weight=int(best_syn),
                )

        if idx < second_pass_cap and gain2 > gain1 + 1e-6:
            llr_restart2, anchor_bits2 = _make_group_debias_restart_llr(
                base_llr_channel=llr_base,
                llr_snapshot=llr_snapshot,
                hard_bits_snapshot=hard_bits_snapshot,
                group_vars=group_vars,
                flip_vars=flip_vars,
                gain=float(gain2),
                abs_floor=float(abs_floor),
                blend=float(min(1.0, debias_blend + 0.15)),
                relax=float(max(0.15, 0.80 * debias_relax)),
            )
            if anchor_bits2 > 0:
                cand2 = _run_short_ldpc_finish_pass(
                    llr_channel=llr_restart2,
                    true_bits=frame.c_bits,
                    code_cfg=code_cfg,
                    max_iters=int(restart_iters),
                    alpha=float(restart_alpha),
                )
                restart_num_runs += 1
                restart_total_ldpc_iters += int(cand2.get("iter_used", 0))
                restart_anchor_bits_total += int(anchor_bits2)
                if not bool(cand2.get("success", False)):
                    restart_num_nonconverged += 1
                best_syn = min(best_syn, int(cand2.get("final_syndrome_weight", best_syn)))
                if bool(cand2.get("success", False)):
                    return _make_attempt_result(
                        success=True,
                        flipped_vars=flip_vars,
                        final_bit_errors=int(cand2.get("final_bit_errors", initial_bit_errors)),
                        candidates_tested=int(candidates_tested),
                        restart_num_runs=int(restart_num_runs),
                        restart_total_ldpc_iters=int(restart_total_ldpc_iters),
                        restart_num_nonconverged=int(restart_num_nonconverged),
                        restart_anchor_bits_total=int(restart_anchor_bits_total),
                        restart_best_syndrome_weight=int(best_syn),
                    )

    return _make_attempt_result(
        success=False,
        candidates_tested=int(candidates_tested),
        restart_num_runs=int(restart_num_runs),
        restart_total_ldpc_iters=int(restart_total_ldpc_iters),
        restart_num_nonconverged=int(restart_num_nonconverged),
        restart_anchor_bits_total=int(restart_anchor_bits_total),
        restart_best_syndrome_weight=int(best_syn),
    )





def _soft_rank_candidate_vars(candidate_vars: np.ndarray,
                              unsat_checks: np.ndarray,
                              code_cfg: CodeConfig,
                              llr_for_sort: np.ndarray,
                              cfg: ClusterGrandConfig,
                              llr_snapshot: Optional[np.ndarray] = None,
                              llr_channel: Optional[np.ndarray] = None) -> np.ndarray:
    """Rank candidate variables for Receiver-6 soft hypotheses."""
    candidate_vars = np.asarray(candidate_vars, dtype=np.int32)
    if candidate_vars.size == 0:
        return candidate_vars

    unsat_set = set(int(j) for j in np.asarray(unsat_checks, dtype=np.int32).tolist())
    eps = float(max(1e-9, float(getattr(cfg, "sv_epsilon", 1e-3) or 1e-3)))
    sat_penalty = float(getattr(cfg, "soft_sat_penalty", 0.35) or 0.35)
    llr_penalty = float(getattr(cfg, "soft_llr_weight", 0.10) or 0.10)

    llr_snapshot_arr = None if llr_snapshot is None else np.asarray(llr_snapshot, dtype=np.float32)
    llr_channel_arr = None if llr_channel is None else np.asarray(llr_channel, dtype=np.float32)

    ranked = []
    for v in candidate_vars.tolist():
        checks = code_cfg.vars_to_checks[int(v)]
        deg = int(len(checks))
        unsat_deg = 0
        for j in checks:
            if int(j) in unsat_set:
                unsat_deg += 1
        sat_deg = max(0, deg - unsat_deg)

        rel = float(abs(float(llr_for_sort[int(v)])))
        disagree_bonus = 0.0
        if llr_snapshot_arr is not None and llr_channel_arr is not None:
            a = float(llr_snapshot_arr[int(v)])
            b = float(llr_channel_arr[int(v)])
            if np.sign(a) * np.sign(b) < 0:
                disagree_bonus = 0.75

        score = (float(unsat_deg) - sat_penalty * float(sat_deg) + disagree_bonus) / (rel + eps)
        score -= llr_penalty * rel
        ranked.append((-float(score), rel, int(v)))

    ranked.sort()
    return np.asarray([v for _neg_s, _rel, v in ranked], dtype=np.int32)


def _enumerate_soft_hypotheses(base_syndrome: np.ndarray,
                               base_syndrome_weight: int,
                               candidate_vars: np.ndarray,
                               code_cfg: CodeConfig,
                               llr_for_sort: np.ndarray,
                               cfg: ClusterGrandConfig,
                               llr_snapshot: Optional[np.ndarray] = None,
                               llr_channel: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], Dict[str, int]]:
    """Generate a ranked list of soft local hypotheses for Receiver-6.

    Unlike Receiver-5 OSD, this does not require the induced local subsystem to
    be exactly consistent. We rank low-weight local flips by full-syndrome
    reduction plus LLR cost, which is better matched to trapping-set escape.
    """
    candidate_vars = np.asarray(candidate_vars, dtype=np.int32)
    if candidate_vars.size == 0:
        return [], {
            "soft_candidate_size": 0,
            "soft_core_size": 0,
            "soft_patterns_considered": 0,
            "soft_score_edge_visits": 0,
            "soft_score_checks_toggled": 0,
            "soft_score_sum_pattern_weights": 0,
        }

    ranked_vars = _soft_rank_candidate_vars(
        candidate_vars=candidate_vars,
        unsat_checks=np.flatnonzero(np.asarray(base_syndrome, dtype=np.uint8)).astype(np.int32),
        code_cfg=code_cfg,
        llr_for_sort=llr_for_sort,
        cfg=cfg,
        llr_snapshot=llr_snapshot,
        llr_channel=llr_channel,
    )

    core_max_bits = max(1, int(getattr(cfg, "soft_core_max_bits", 14) or 14))
    max_weight = max(1, int(getattr(cfg, "soft_max_weight", 3) or 3))
    max_candidates = max(1, int(getattr(cfg, "soft_max_candidates", 128) or 128))

    core = ranked_vars[:min(int(ranked_vars.size), core_max_bits)]
    if core.size == 0:
        return [], {
            "soft_candidate_size": int(candidate_vars.size),
            "soft_core_size": 0,
            "soft_patterns_considered": 0,
            "soft_score_edge_visits": 0,
            "soft_score_checks_toggled": 0,
            "soft_score_sum_pattern_weights": 0,
        }

    ranked = []
    total_edge_visits = 0
    total_checks_toggled = 0
    total_pattern_weights = 0
    seen = set()
    core_list = [int(v) for v in core.tolist()]

    for w in range(1, min(max_weight, len(core_list)) + 1):
        for comb in itertools.combinations(core_list, w):
            if comb in seen:
                continue
            seen.add(comb)
            syn_w, e_cnt, _uq_cnt, tg_cnt = _syndrome_weight_and_counts_after_flips_from_base(
                base_syndrome=base_syndrome,
                base_weight=int(base_syndrome_weight),
                flipped_vars=list(comb),
                code_cfg=code_cfg,
            )
            llr_cost = float(np.sum(np.abs(llr_for_sort[np.asarray(comb, dtype=np.int32)])))
            ranked.append((int(syn_w), float(llr_cost), int(w), tuple(int(x) for x in comb)))
            total_edge_visits += int(e_cnt)
            total_checks_toggled += int(tg_cnt)
            total_pattern_weights += int(w)

    ranked.sort(key=lambda t: (int(t[0]), float(t[1]), int(t[2]), t[3]))
    if len(ranked) > max_candidates:
        ranked = ranked[:max_candidates]

    out = [np.asarray(list(comb), dtype=np.int32) for _syn_w, _cost, _w, comb in ranked]
    return out, {
        "soft_candidate_size": int(candidate_vars.size),
        "soft_core_size": int(core.size),
        "soft_patterns_considered": int(len(ranked)),
        "soft_score_edge_visits": int(total_edge_visits),
        "soft_score_checks_toggled": int(total_checks_toggled),
        "soft_score_sum_pattern_weights": int(total_pattern_weights),
    }


def _run_presolver_soft_anchor(frame: FrameLog,
                               sim_cfg: SimulationConfig,
                               snapshot_iter: int,
                               cfg: ClusterGrandConfig) -> Optional[ClusterGrandResult]:
    """Receiver-6 pre-solver: soft local hypotheses + anchored full-graph restarts."""
    code_cfg = sim_cfg.code

    snaps = frame.snapshots
    syn_snaps = snaps.get("syndrome", {})
    hard_snaps = snaps.get("hard_bits", {})
    llr_snaps = snaps.get("llr", {})

    if (snapshot_iter not in syn_snaps or
        snapshot_iter not in hard_snaps or
        snapshot_iter not in llr_snaps):
        raise ValueError(f"Snapshot at iter {snapshot_iter} is not fully available for Receiver-6 pre-solver.")

    syndrome = np.asarray(syn_snaps[snapshot_iter], dtype=np.uint8)
    hard_bits_snapshot = np.asarray(hard_snaps[snapshot_iter], dtype=np.uint8).copy()
    llr_snapshot = np.asarray(llr_snaps[snapshot_iter], dtype=np.float32)

    initial_syndrome_weight = int(syndrome.sum())
    if initial_syndrome_weight == 0:
        return None

    diff_init = (hard_bits_snapshot != np.asarray(frame.c_bits, dtype=np.uint8))
    initial_bit_errors = int(diff_init.sum())
    unsat_checks = np.flatnonzero(syndrome).astype(np.int32)

    cluster_unsat_edges = 0
    cluster_pair_edges = 0
    if unsat_checks.size > 0:
        for j in unsat_checks.tolist():
            neigh = code_cfg.checks_to_vars[int(j)]
            d = int(neigh.size)
            cluster_unsat_edges += d
            if d >= 2:
                cluster_pair_edges += int(d * (d - 1) // 2)

    llr_for_sort, llr_source_used = _resolve_sort_llr_vector(
        llr_snapshot=llr_snapshot,
        llr_channel=getattr(frame, "llr_channel", None),
        cfg=cfg,
    )

    allowed_mask = build_allowed_mask_from_config(frame, sim_cfg, snapshot_iter, cfg)
    clusters = find_variable_clusters_from_syndrome(syndrome, code_cfg)
    if not clusters:
        return None

    union_vars = np.unique(np.concatenate(clusters)).astype(np.int32)
    union_vars = union_vars[allowed_mask[union_vars]]
    L_full = int(union_vars.size)
    if L_full == 0:
        return None

    L_search = _auto_pick_grand_search_size(L_full, cfg)
    soft_ratio = float(getattr(cfg, "soft_candidate_ratio", 3.0) or 3.0)
    L_soft = max(int(L_search), int(np.ceil(max(1.0, soft_ratio) * max(1, int(L_search)))))
    soft_cap = getattr(cfg, "soft_max_bits", None)
    if soft_cap is not None:
        try:
            L_soft = min(L_soft, int(soft_cap))
        except Exception:
            pass
    L_soft = min(L_soft, L_full)

    soft_vars, front_end_meta = _select_presolver_vars(
        union_vars=union_vars,
        unsat_checks=unsat_checks,
        code_cfg=code_cfg,
        llr_for_sort=llr_for_sort,
        L_peel=L_soft,
        cfg=cfg,
        llr_snapshot=llr_snapshot,
        llr_channel=getattr(frame, "llr_channel", None),
    )
    if soft_vars.size == 0:
        return None

    hypotheses, soft_meta = _enumerate_soft_hypotheses(
        base_syndrome=syndrome,
        base_syndrome_weight=int(initial_syndrome_weight),
        candidate_vars=soft_vars,
        code_cfg=code_cfg,
        llr_for_sort=llr_for_sort,
        cfg=cfg,
        llr_snapshot=llr_snapshot,
        llr_channel=getattr(frame, "llr_channel", None),
    )

    llr_base = np.asarray(getattr(frame, "llr_channel", None), dtype=np.float32) if getattr(frame, "llr_channel", None) is not None else np.asarray(llr_snapshot, dtype=np.float32)
    restart_max = max(1, int(getattr(cfg, "restart_max_candidates", 24) or 24))
    restart_iters = max(1, int(getattr(cfg, "restart_ldpc_iters", 14) or 14))
    restart_alpha = float(getattr(cfg, "restart_alpha", 0.78) or 0.78)
    gain1 = float(getattr(cfg, "restart_llr_gain", 4.5) or 4.5)
    gain2 = float(getattr(cfg, "restart_dual_gain", gain1) or gain1)
    abs_floor = float(getattr(cfg, "restart_llr_abs_floor", 6.0) or 6.0)
    anchor_all_first = bool(int(getattr(cfg, "restart_anchor_all_selected", 0) or 0))
    second_pass_cap = max(0, min(restart_max, 8))
    soft_idx = {int(v): i for i, v in enumerate(np.asarray(soft_vars, dtype=np.int32).tolist())}

    def _make_attempt_result(success: bool,
                             flipped_vars: Optional[np.ndarray] = None,
                             final_bit_errors: Optional[int] = None,
                             candidates_tested: int = 0,
                             restart_num_runs: int = 0,
                             restart_total_ldpc_iters: int = 0,
                             restart_num_nonconverged: int = 0,
                             restart_anchor_bits_total: int = 0,
                             restart_best_syndrome_weight: Optional[int] = None) -> ClusterGrandResult:
        if flipped_vars is None:
            flipped_vars = np.array([], dtype=np.int32)
        if final_bit_errors is None:
            final_bit_errors = int(initial_bit_errors)
        if restart_best_syndrome_weight is None:
            restart_best_syndrome_weight = int(initial_syndrome_weight)

        res_local = ClusterGrandResult(
            success=bool(success),
            pattern_weight=int(np.asarray(flipped_vars, dtype=np.int32).size) if bool(success) else -1,
            flipped_vars=np.asarray(flipped_vars, dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=int(initial_syndrome_weight),
            final_syndrome_weight=0 if bool(success) else int(restart_best_syndrome_weight),
            initial_bit_errors=int(initial_bit_errors),
            final_bit_errors=int(final_bit_errors),
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        setattr(res_local, "patterns_evaluated", 0)
        setattr(res_local, "total_v2c_edge_visits_evaluated", 0)
        setattr(res_local, "total_unique_checks_visited_evaluated", 0)
        setattr(res_local, "total_unique_checks_toggled_evaluated", 0)
        setattr(res_local, "union_size", int(L_full))
        setattr(res_local, "search_size", int(L_search))
        setattr(res_local, "llr_sort_len", int(L_full))
        setattr(res_local, "sum_pattern_weights_generated", 0)
        setattr(res_local, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res_local, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res_local, "num_batches_evaluated", 0)
        setattr(res_local, "positions_packed_evaluated", 0)
        setattr(res_local, "batch_size_used", 0)
        setattr(res_local, "llr_source_used", str(llr_source_used))
        setattr(res_local, "selection_mode_used", str(front_end_meta.get("selection_mode_used", getattr(cfg, "selection_mode", "llr"))))
        setattr(res_local, "sv_seeded_count", int(front_end_meta.get("sv_seeded_count", 0)))
        setattr(res_local, "sv_neighbor_visits", int(front_end_meta.get("sv_neighbor_visits", 0)))
        setattr(res_local, "sv_score_len", int(front_end_meta.get("sv_score_len", L_full)))
        setattr(res_local, "pre_solver_mode_used", "soft_anchor")
        setattr(res_local, "pre_solver_attempted", 1)
        setattr(res_local, "pre_solver_success", 1 if bool(success) else 0)
        setattr(res_local, "peel_extra_llr_added", int(front_end_meta.get("peel_extra_llr_added", 0)))
        setattr(res_local, "disagreement_added", int(front_end_meta.get("disagreement_added", 0)))

        setattr(res_local, "chase_candidate_size", int(soft_meta.get("soft_candidate_size", int(soft_vars.size))))
        setattr(res_local, "chase_core_size", int(soft_meta.get("soft_core_size", 0)))
        setattr(res_local, "chase_patterns_considered", int(soft_meta.get("soft_patterns_considered", 0)))
        setattr(res_local, "chase_candidates_tested", int(candidates_tested))
        setattr(res_local, "chase_score_edge_visits", int(soft_meta.get("soft_score_edge_visits", 0)))
        setattr(res_local, "chase_score_checks_toggled", int(soft_meta.get("soft_score_checks_toggled", 0)))
        setattr(res_local, "chase_score_sum_pattern_weights", int(soft_meta.get("soft_score_sum_pattern_weights", 0)))
        setattr(res_local, "chase_ldpc_total_iters", 0)
        setattr(res_local, "chase_ldpc_num_runs", 0)
        setattr(res_local, "chase_ldpc_num_nonconverged", 0)
        setattr(res_local, "chase_best_syndrome_weight", int(restart_best_syndrome_weight))

        setattr(res_local, "restart_num_runs", int(restart_num_runs))
        setattr(res_local, "restart_total_ldpc_iters", int(restart_total_ldpc_iters))
        setattr(res_local, "restart_num_nonconverged", int(restart_num_nonconverged))
        setattr(res_local, "restart_anchor_bits_total", int(restart_anchor_bits_total))
        setattr(res_local, "restart_best_syndrome_weight", int(restart_best_syndrome_weight))
        return res_local

    if not hypotheses:
        return _make_attempt_result(success=False)

    candidates_tested = 0
    restart_num_runs = 0
    restart_total_ldpc_iters = 0
    restart_num_nonconverged = 0
    restart_anchor_bits_total = 0
    best_syn = int(initial_syndrome_weight)

    for idx, flip_vars in enumerate(hypotheses[:restart_max]):
        flip_vars = np.asarray(flip_vars, dtype=np.int32)
        candidates_tested += 1

        syn_w_full, _e_cnt, _uq_cnt, _tg_cnt = _syndrome_weight_and_counts_after_flips_from_base(
            base_syndrome=syndrome,
            base_weight=int(initial_syndrome_weight),
            flipped_vars=[int(v) for v in flip_vars.tolist()],
            code_cfg=code_cfg,
        )
        best_syn = min(best_syn, int(syn_w_full))
        if int(syn_w_full) == 0:
            final_bit_errors = _bit_errors_after_flips_from_base(
                base_bits=hard_bits_snapshot,
                true_bits=np.asarray(frame.c_bits, dtype=np.uint8),
                base_bit_errors=int(initial_bit_errors),
                flipped_vars=[int(v) for v in flip_vars.tolist()],
            )
            return _make_attempt_result(
                success=True,
                flipped_vars=flip_vars,
                final_bit_errors=int(final_bit_errors),
                candidates_tested=int(candidates_tested),
                restart_num_runs=int(restart_num_runs),
                restart_total_ldpc_iters=int(restart_total_ldpc_iters),
                restart_num_nonconverged=int(restart_num_nonconverged),
                restart_anchor_bits_total=int(restart_anchor_bits_total),
                restart_best_syndrome_weight=int(best_syn),
            )

        x_local = np.zeros((soft_vars.size,), dtype=np.uint8)
        for v in flip_vars.tolist():
            pos = soft_idx.get(int(v), None)
            if pos is not None:
                x_local[int(pos)] = np.uint8(1)

        llr_restart, anchor_bits = _make_anchor_restart_llr(
            base_llr_channel=llr_base,
            llr_snapshot=llr_snapshot,
            hard_bits_snapshot=hard_bits_snapshot,
            candidate_vars=soft_vars,
            x_local=x_local,
            gain=float(gain1),
            abs_floor=float(abs_floor),
            anchor_all_selected=bool(anchor_all_first),
        )
        if anchor_bits > 0:
            cand = _run_short_ldpc_finish_pass(
                llr_channel=llr_restart,
                true_bits=frame.c_bits,
                code_cfg=code_cfg,
                max_iters=int(restart_iters),
                alpha=float(restart_alpha),
            )
            restart_num_runs += 1
            restart_total_ldpc_iters += int(cand.get("iter_used", 0))
            restart_anchor_bits_total += int(anchor_bits)
            if not bool(cand.get("success", False)):
                restart_num_nonconverged += 1
            best_syn = min(best_syn, int(cand.get("final_syndrome_weight", best_syn)))
            if bool(cand.get("success", False)):
                return _make_attempt_result(
                    success=True,
                    flipped_vars=flip_vars,
                    final_bit_errors=int(cand.get("final_bit_errors", initial_bit_errors)),
                    candidates_tested=int(candidates_tested),
                    restart_num_runs=int(restart_num_runs),
                    restart_total_ldpc_iters=int(restart_total_ldpc_iters),
                    restart_num_nonconverged=int(restart_num_nonconverged),
                    restart_anchor_bits_total=int(restart_anchor_bits_total),
                    restart_best_syndrome_weight=int(best_syn),
                )

        if idx < second_pass_cap and gain2 > gain1 + 1e-6:
            llr_restart2, anchor_bits2 = _make_anchor_restart_llr(
                base_llr_channel=llr_base,
                llr_snapshot=llr_snapshot,
                hard_bits_snapshot=hard_bits_snapshot,
                candidate_vars=soft_vars,
                x_local=x_local,
                gain=float(gain2),
                abs_floor=float(abs_floor),
                anchor_all_selected=True,
            )
            if anchor_bits2 > 0:
                cand2 = _run_short_ldpc_finish_pass(
                    llr_channel=llr_restart2,
                    true_bits=frame.c_bits,
                    code_cfg=code_cfg,
                    max_iters=int(restart_iters),
                    alpha=float(restart_alpha),
                )
                restart_num_runs += 1
                restart_total_ldpc_iters += int(cand2.get("iter_used", 0))
                restart_anchor_bits_total += int(anchor_bits2)
                if not bool(cand2.get("success", False)):
                    restart_num_nonconverged += 1
                best_syn = min(best_syn, int(cand2.get("final_syndrome_weight", best_syn)))
                if bool(cand2.get("success", False)):
                    return _make_attempt_result(
                        success=True,
                        flipped_vars=flip_vars,
                        final_bit_errors=int(cand2.get("final_bit_errors", initial_bit_errors)),
                        candidates_tested=int(candidates_tested),
                        restart_num_runs=int(restart_num_runs),
                        restart_total_ldpc_iters=int(restart_total_ldpc_iters),
                        restart_num_nonconverged=int(restart_num_nonconverged),
                        restart_anchor_bits_total=int(restart_anchor_bits_total),
                        restart_best_syndrome_weight=int(best_syn),
                    )

    return _make_attempt_result(
        success=False,
        candidates_tested=int(candidates_tested),
        restart_num_runs=int(restart_num_runs),
        restart_total_ldpc_iters=int(restart_total_ldpc_iters),
        restart_num_nonconverged=int(restart_num_nonconverged),
        restart_anchor_bits_total=int(restart_anchor_bits_total),
        restart_best_syndrome_weight=int(best_syn),
    )


def _run_presolver_osd_anchor(frame: FrameLog,
                              sim_cfg: SimulationConfig,
                              snapshot_iter: int,
                              cfg: ClusterGrandConfig) -> Optional[ClusterGrandResult]:
    """Receiver 5 pre-solver: local OSD/MRB list + anchored full-graph restarts."""
    code_cfg = sim_cfg.code

    snaps = frame.snapshots
    syn_snaps = snaps.get("syndrome", {})
    hard_snaps = snaps.get("hard_bits", {})
    llr_snaps = snaps.get("llr", {})

    if (snapshot_iter not in syn_snaps or
        snapshot_iter not in hard_snaps or
        snapshot_iter not in llr_snaps):
        raise ValueError(f"Snapshot at iter {snapshot_iter} is not fully available for OSD pre-solver.")

    syndrome = syn_snaps[snapshot_iter]
    hard_bits_snapshot = hard_snaps[snapshot_iter].copy()
    llr_snapshot = llr_snaps[snapshot_iter]

    initial_syndrome_weight = int(np.asarray(syndrome, dtype=np.uint8).sum())
    if initial_syndrome_weight == 0:
        return None

    diff_init = (hard_bits_snapshot != frame.c_bits)
    initial_bit_errors = int(diff_init.sum())
    unsat_checks = np.flatnonzero(syndrome).astype(np.int32)

    cluster_unsat_edges = 0
    cluster_pair_edges = 0
    if unsat_checks.size > 0:
        for j in unsat_checks:
            neigh = code_cfg.checks_to_vars[int(j)]
            d = int(neigh.size)
            cluster_unsat_edges += d
            if d >= 2:
                cluster_pair_edges += int(d * (d - 1) // 2)

    llr_for_sort, llr_source_used = _resolve_sort_llr_vector(
        llr_snapshot=llr_snapshot,
        llr_channel=getattr(frame, "llr_channel", None),
        cfg=cfg,
    )

    allowed_mask = build_allowed_mask_from_config(frame, sim_cfg, snapshot_iter, cfg)
    clusters = find_variable_clusters_from_syndrome(syndrome, code_cfg)
    if not clusters:
        return None

    union_vars = np.unique(np.concatenate(clusters)).astype(np.int32)
    union_vars = union_vars[allowed_mask[union_vars]]
    L_full = int(union_vars.size)
    if L_full == 0:
        return None

    L_search = _auto_pick_grand_search_size(L_full, cfg)
    L_osd = _auto_pick_osd_candidate_size(L_full, L_search, cfg)
    osd_vars, front_end_meta = _select_presolver_vars(
        union_vars=union_vars,
        unsat_checks=unsat_checks,
        code_cfg=code_cfg,
        llr_for_sort=llr_for_sort,
        L_peel=L_osd,
        cfg=cfg,
        llr_snapshot=llr_snapshot,
        llr_channel=getattr(frame, "llr_channel", None),
    )

    if osd_vars.size == 0:
        return None

    A, b = _build_local_subsystem_for_candidate(
        candidate_vars=osd_vars,
        unsat_checks=unsat_checks,
        code_cfg=code_cfg,
        syndrome=syndrome,
    )
    if A.size == 0 or A.shape[1] == 0:
        return None

    order = max(0, int(getattr(cfg, "osd_order", 2) or 2))
    enum_bits = max(0, int(getattr(cfg, "osd_enum_max_bits", 18) or 18))
    max_candidates = max(1, int(getattr(cfg, "osd_max_candidates", 128) or 128))

    candidates, osd_meta = _gf2_osd_ranked_candidates(
        A=A,
        b=b,
        reliabilities=np.abs(llr_for_sort[osd_vars]).astype(np.float64, copy=False),
        order=int(order),
        max_enum_bits=int(enum_bits),
        max_candidates=int(max_candidates),
    )

    llr_base = np.asarray(getattr(frame, "llr_channel", None), dtype=np.float32) if getattr(frame, "llr_channel", None) is not None else np.asarray(llr_snapshot, dtype=np.float32)
    restart_max = max(1, int(getattr(cfg, "restart_max_candidates", 24) or 24))
    restart_iters = max(1, int(getattr(cfg, "restart_ldpc_iters", 14) or 14))
    restart_alpha = float(getattr(cfg, "restart_alpha", 0.78) or 0.78)
    gain1 = float(getattr(cfg, "restart_llr_gain", 4.5) or 4.5)
    gain2 = float(getattr(cfg, "restart_dual_gain", gain1) or gain1)
    abs_floor = float(getattr(cfg, "restart_llr_abs_floor", 6.0) or 6.0)
    anchor_all = bool(int(getattr(cfg, "restart_anchor_all_selected", 0) or 0))
    second_pass_cap = max(0, min(restart_max, 6))

    def _make_attempt_result(success: bool,
                             flipped_vars: Optional[np.ndarray] = None,
                             final_bit_errors: Optional[int] = None,
                             osd_candidates_tested: int = 0,
                             restart_num_runs: int = 0,
                             restart_total_ldpc_iters: int = 0,
                             restart_num_nonconverged: int = 0,
                             restart_anchor_bits_total: int = 0,
                             restart_best_syndrome_weight: Optional[int] = None) -> ClusterGrandResult:
        if flipped_vars is None:
            flipped_vars = np.array([], dtype=np.int32)
        if final_bit_errors is None:
            final_bit_errors = int(initial_bit_errors)
        if restart_best_syndrome_weight is None:
            restart_best_syndrome_weight = int(initial_syndrome_weight)

        res_local = ClusterGrandResult(
            success=bool(success),
            pattern_weight=int(flipped_vars.size) if bool(success) else -1,
            flipped_vars=np.asarray(flipped_vars, dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=int(initial_syndrome_weight),
            final_syndrome_weight=0 if bool(success) else int(initial_syndrome_weight),
            initial_bit_errors=int(initial_bit_errors),
            final_bit_errors=int(final_bit_errors),
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        setattr(res_local, "patterns_evaluated", 0)
        setattr(res_local, "total_v2c_edge_visits_evaluated", 0)
        setattr(res_local, "total_unique_checks_visited_evaluated", 0)
        setattr(res_local, "total_unique_checks_toggled_evaluated", 0)
        setattr(res_local, "union_size", int(L_full))
        setattr(res_local, "search_size", int(L_search))
        setattr(res_local, "llr_sort_len", int(L_full))
        setattr(res_local, "sum_pattern_weights_generated", 0)
        setattr(res_local, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res_local, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res_local, "num_batches_evaluated", 0)
        setattr(res_local, "positions_packed_evaluated", 0)
        setattr(res_local, "batch_size_used", 0)
        setattr(res_local, "llr_source_used", str(llr_source_used))
        setattr(res_local, "selection_mode_used", str(front_end_meta.get("selection_mode_used", getattr(cfg, "selection_mode", "llr"))))
        setattr(res_local, "sv_seeded_count", int(front_end_meta.get("sv_seeded_count", 0)))
        setattr(res_local, "sv_neighbor_visits", int(front_end_meta.get("sv_neighbor_visits", 0)))
        setattr(res_local, "sv_score_len", int(front_end_meta.get("sv_score_len", L_full)))
        setattr(res_local, "pre_solver_mode_used", "osd_anchor")
        setattr(res_local, "pre_solver_attempted", 1)
        setattr(res_local, "pre_solver_success", 1 if bool(success) else 0)
        setattr(res_local, "peel_extra_llr_added", int(front_end_meta.get("peel_extra_llr_added", 0)))
        setattr(res_local, "disagreement_added", int(front_end_meta.get("disagreement_added", 0)))
        setattr(res_local, "osd_candidate_size", int(osd_vars.size))
        setattr(res_local, "osd_matrix_rows", int(A.shape[0]))
        setattr(res_local, "osd_free_dim", int(osd_meta.get("osd_free_dim", 0)))
        setattr(res_local, "osd_enum_bits_used", int(osd_meta.get("osd_enum_bits_used", 0)))
        setattr(res_local, "osd_basis_xor_ops", int(osd_meta.get("osd_basis_xor_ops", 0)))
        setattr(res_local, "osd_candidates_considered", int(osd_meta.get("osd_candidates_considered", 0)))
        setattr(res_local, "osd_candidates_tested", int(osd_candidates_tested))
        setattr(res_local, "osd_sum_candidate_weights", int(osd_meta.get("osd_sum_candidate_weights", 0)))
        setattr(res_local, "restart_num_runs", int(restart_num_runs))
        setattr(res_local, "restart_total_ldpc_iters", int(restart_total_ldpc_iters))
        setattr(res_local, "restart_num_nonconverged", int(restart_num_nonconverged))
        setattr(res_local, "restart_anchor_bits_total", int(restart_anchor_bits_total))
        setattr(res_local, "restart_best_syndrome_weight", int(restart_best_syndrome_weight))
        return res_local

    if not candidates:
        return _make_attempt_result(success=False)

    osd_candidates_tested = 0
    restart_num_runs = 0
    restart_total_ldpc_iters = 0
    restart_num_nonconverged = 0
    restart_anchor_bits_total = 0
    best_syn = int(initial_syndrome_weight)

    for idx, (cost, hwt, free_weight, x_local) in enumerate(candidates[:restart_max]):
        flip_vars = osd_vars[np.flatnonzero(x_local).astype(np.int32)]
        osd_candidates_tested += 1

        cand_bits = hard_bits_snapshot.copy()
        if flip_vars.size > 0:
            cand_bits[flip_vars] ^= np.uint8(1)
        syn_full = compute_syndrome_from_checks(cand_bits, code_cfg)
        syn_w_full = int(np.asarray(syn_full, dtype=np.uint8).sum())
        best_syn = min(best_syn, syn_w_full)
        if syn_w_full == 0:
            final_bit_errors = int(np.count_nonzero(cand_bits != np.asarray(frame.c_bits, dtype=np.uint8)))
            return _make_attempt_result(
                success=True,
                flipped_vars=np.asarray(flip_vars, dtype=np.int32),
                final_bit_errors=int(final_bit_errors),
                osd_candidates_tested=int(osd_candidates_tested),
                restart_num_runs=int(restart_num_runs),
                restart_total_ldpc_iters=int(restart_total_ldpc_iters),
                restart_num_nonconverged=int(restart_num_nonconverged),
                restart_anchor_bits_total=int(restart_anchor_bits_total),
                restart_best_syndrome_weight=int(best_syn),
            )

        llr_restart, anchor_bits = _make_anchor_restart_llr(
            base_llr_channel=llr_base,
            llr_snapshot=llr_snapshot,
            hard_bits_snapshot=hard_bits_snapshot,
            candidate_vars=osd_vars,
            x_local=x_local,
            gain=float(gain1),
            abs_floor=float(abs_floor),
            anchor_all_selected=bool(anchor_all),
        )
        if anchor_bits > 0:
            cand = _run_short_ldpc_finish_pass(
                llr_channel=llr_restart,
                true_bits=frame.c_bits,
                code_cfg=code_cfg,
                max_iters=int(restart_iters),
                alpha=float(restart_alpha),
            )
            restart_num_runs += 1
            restart_total_ldpc_iters += int(cand.get("iter_used", 0))
            restart_anchor_bits_total += int(anchor_bits)
            if not bool(cand.get("success", False)):
                restart_num_nonconverged += 1
            best_syn = min(best_syn, int(cand.get("final_syndrome_weight", best_syn)))
            if bool(cand.get("success", False)):
                return _make_attempt_result(
                    success=True,
                    flipped_vars=np.asarray(flip_vars, dtype=np.int32),
                    final_bit_errors=int(cand.get("final_bit_errors", initial_bit_errors)),
                    osd_candidates_tested=int(osd_candidates_tested),
                    restart_num_runs=int(restart_num_runs),
                    restart_total_ldpc_iters=int(restart_total_ldpc_iters),
                    restart_num_nonconverged=int(restart_num_nonconverged),
                    restart_anchor_bits_total=int(restart_anchor_bits_total),
                    restart_best_syndrome_weight=int(best_syn),
                )

        if idx < second_pass_cap and gain2 > gain1 + 1e-6:
            llr_restart2, anchor_bits2 = _make_anchor_restart_llr(
                base_llr_channel=llr_base,
                llr_snapshot=llr_snapshot,
                hard_bits_snapshot=hard_bits_snapshot,
                candidate_vars=osd_vars,
                x_local=x_local,
                gain=float(gain2),
                abs_floor=float(abs_floor),
                anchor_all_selected=True,
            )
            if anchor_bits2 > 0:
                cand2 = _run_short_ldpc_finish_pass(
                    llr_channel=llr_restart2,
                    true_bits=frame.c_bits,
                    code_cfg=code_cfg,
                    max_iters=int(restart_iters),
                    alpha=float(restart_alpha),
                )
                restart_num_runs += 1
                restart_total_ldpc_iters += int(cand2.get("iter_used", 0))
                restart_anchor_bits_total += int(anchor_bits2)
                if not bool(cand2.get("success", False)):
                    restart_num_nonconverged += 1
                best_syn = min(best_syn, int(cand2.get("final_syndrome_weight", best_syn)))
                if bool(cand2.get("success", False)):
                    return _make_attempt_result(
                        success=True,
                        flipped_vars=np.asarray(flip_vars, dtype=np.int32),
                        final_bit_errors=int(cand2.get("final_bit_errors", initial_bit_errors)),
                        osd_candidates_tested=int(osd_candidates_tested),
                        restart_num_runs=int(restart_num_runs),
                        restart_total_ldpc_iters=int(restart_total_ldpc_iters),
                        restart_num_nonconverged=int(restart_num_nonconverged),
                        restart_anchor_bits_total=int(restart_anchor_bits_total),
                        restart_best_syndrome_weight=int(best_syn),
                    )

    return _make_attempt_result(
        success=False,
        osd_candidates_tested=int(osd_candidates_tested),
        restart_num_runs=int(restart_num_runs),
        restart_total_ldpc_iters=int(restart_total_ldpc_iters),
        restart_num_nonconverged=int(restart_num_nonconverged),
        restart_anchor_bits_total=int(restart_anchor_bits_total),
        restart_best_syndrome_weight=int(best_syn),
    )


def _run_presolver_peel_gf2(frame: FrameLog,
                            sim_cfg: SimulationConfig,
                            snapshot_iter: int,
                            cfg: ClusterGrandConfig) -> Optional[ClusterGrandResult]:
    """Receiver 3+ pre-solver: peel + weighted small GF(2) solve on a larger local set.

    Returns:
      - successful ClusterGrandResult if the pre-solver alone fixes the frame
      - failed ClusterGrandResult with pre-solver metadata if it *attempted* but
        could not certify a correction (so the caller can still charge its cost)
      - None if the pre-solver was not applicable / not meaningfully attempted
    """
    code_cfg = sim_cfg.code

    snaps = frame.snapshots
    syn_snaps = snaps.get("syndrome", {})
    hard_snaps = snaps.get("hard_bits", {})
    llr_snaps = snaps.get("llr", {})

    if (snapshot_iter not in syn_snaps or
        snapshot_iter not in hard_snaps or
        snapshot_iter not in llr_snaps):
        raise ValueError(f"Snapshot at iter {snapshot_iter} is not fully available for pre-solver.")

    syndrome = syn_snaps[snapshot_iter]
    hard_bits_snapshot = hard_snaps[snapshot_iter].copy()
    llr_snapshot = llr_snaps[snapshot_iter]

    initial_syndrome_weight = int(syndrome.sum())
    if initial_syndrome_weight == 0:
        return None

    diff_init = (hard_bits_snapshot != frame.c_bits)
    initial_bit_errors = int(diff_init.sum())
    unsat_checks = np.flatnonzero(syndrome).astype(np.int32)

    cluster_unsat_edges = 0
    cluster_pair_edges = 0
    if unsat_checks.size > 0:
        for j in unsat_checks:
            neigh = code_cfg.checks_to_vars[int(j)]
            d = int(neigh.size)
            cluster_unsat_edges += d
            if d >= 2:
                cluster_pair_edges += int(d * (d - 1) // 2)

    llr_for_sort, llr_source_used = _resolve_sort_llr_vector(
        llr_snapshot=llr_snapshot,
        llr_channel=getattr(frame, "llr_channel", None),
        cfg=cfg,
    )

    allowed_mask = build_allowed_mask_from_config(frame, sim_cfg, snapshot_iter, cfg)
    clusters = find_variable_clusters_from_syndrome(syndrome, code_cfg)
    if not clusters:
        return None

    union_vars = np.unique(np.concatenate(clusters)).astype(np.int32)
    union_vars = union_vars[allowed_mask[union_vars]]
    L_full = int(union_vars.size)
    if L_full == 0:
        return None

    L_search = _auto_pick_grand_search_size(L_full, cfg)
    L_peel = _auto_pick_peel_candidate_size(L_full, L_search, cfg)
    peel_vars, front_end_meta = _select_presolver_vars(
        union_vars=union_vars,
        unsat_checks=unsat_checks,
        code_cfg=code_cfg,
        llr_for_sort=llr_for_sort,
        L_peel=L_peel,
        cfg=cfg,
        llr_snapshot=llr_snapshot,
        llr_channel=getattr(frame, "llr_channel", None),
    )

    if peel_vars.size == 0:
        return None

    # Common metadata carried on both success and attempted-failure returns
    def _make_attempt_result(success: bool,
                             flipped_vars: Optional[np.ndarray] = None,
                             final_bit_errors: Optional[int] = None,
                             peel_edge_work: int = 0,
                             dense_xor_ops: int = 0,
                             free_dim: int = 0,
                             residual_vars: int = 0,
                             residual_rows: int = 0,
                             e_cnt: int = 0,
                             uq_cnt: int = 0,
                             tg_cnt: int = 0) -> ClusterGrandResult:
        if flipped_vars is None:
            flipped_vars = np.array([], dtype=np.int32)
        if final_bit_errors is None:
            final_bit_errors = int(initial_bit_errors)
        res_local = ClusterGrandResult(
            success=bool(success),
            pattern_weight=int(flipped_vars.size) if bool(success) else -1,
            flipped_vars=np.asarray(flipped_vars, dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=int(initial_syndrome_weight),
            final_syndrome_weight=0 if bool(success) else int(initial_syndrome_weight),
            initial_bit_errors=int(initial_bit_errors),
            final_bit_errors=int(final_bit_errors),
            total_v2c_edge_visits=int(e_cnt),
            total_unique_checks_visited=int(uq_cnt),
            total_unique_checks_toggled=int(tg_cnt),
            patterns_generated=0,
        )
        setattr(res_local, "patterns_evaluated", 0)
        setattr(res_local, "total_v2c_edge_visits_evaluated", 0)
        setattr(res_local, "total_unique_checks_visited_evaluated", 0)
        setattr(res_local, "total_unique_checks_toggled_evaluated", 0)
        setattr(res_local, "union_size", int(L_full))
        setattr(res_local, "search_size", int(L_search))
        setattr(res_local, "llr_sort_len", int(L_full))
        setattr(res_local, "sum_pattern_weights_generated", 0)
        setattr(res_local, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res_local, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res_local, "num_batches_evaluated", 0)
        setattr(res_local, "positions_packed_evaluated", 0)
        setattr(res_local, "batch_size_used", 0)
        setattr(res_local, "llr_source_used", str(llr_source_used))
        setattr(res_local, "selection_mode_used", str(front_end_meta.get("selection_mode_used", getattr(cfg, "selection_mode", "llr"))))
        setattr(res_local, "sv_seeded_count", int(front_end_meta.get("sv_seeded_count", 0)))
        setattr(res_local, "sv_neighbor_visits", int(front_end_meta.get("sv_neighbor_visits", 0)))
        setattr(res_local, "sv_score_len", int(front_end_meta.get("sv_score_len", L_full)))
        setattr(res_local, "pre_solver_mode_used", "peel_gf2")
        setattr(res_local, "pre_solver_attempted", 1)
        setattr(res_local, "pre_solver_success", 1 if bool(success) else 0)
        setattr(res_local, "peel_candidate_size", int(peel_vars.size))
        setattr(res_local, "peel_residual_vars", int(residual_vars))
        setattr(res_local, "peel_residual_rows", int(residual_rows))
        setattr(res_local, "peel_edge_work", int(peel_edge_work))
        setattr(res_local, "peel_dense_xor_ops", int(dense_xor_ops))
        setattr(res_local, "peel_free_dim", int(free_dim))
        setattr(res_local, "peel_extra_llr_added", int(front_end_meta.get("peel_extra_llr_added", 0)))
        setattr(res_local, "disagreement_added", int(front_end_meta.get("disagreement_added", 0)))
        return res_local

    A, b = _build_local_subsystem_for_candidate(
        candidate_vars=peel_vars,
        unsat_checks=unsat_checks,
        code_cfg=code_cfg,
        syndrome=syndrome,
    )

    ok_peel, fixed, unresolved_cols, unresolved_rows, peel_edge_work = _peel_reduce_system(A, b)
    if not ok_peel:
        return _make_attempt_result(
            success=False,
            peel_edge_work=int(peel_edge_work),
            residual_vars=int(unresolved_cols.size),
            residual_rows=int(unresolved_rows.size),
        )

    solution = np.zeros((peel_vars.size,), dtype=np.uint8)
    fixed_mask = (fixed >= 0)
    if np.any(fixed_mask):
        solution[fixed_mask] = fixed[fixed_mask].astype(np.uint8, copy=False)

    dense_xor_ops = 0
    free_dim = 0
    residual_vars = int(unresolved_cols.size)
    residual_rows = int(unresolved_rows.size)

    if residual_vars > 0:
        dense_limit = max(0, int(getattr(cfg, "peel_dense_max_vars", 0) or 0))
        if residual_vars > dense_limit:
            return _make_attempt_result(
                success=False,
                peel_edge_work=int(peel_edge_work),
                residual_vars=int(residual_vars),
                residual_rows=int(residual_rows),
            )

        A_red = A[np.asarray(unresolved_rows, dtype=np.int32)][:, np.asarray(unresolved_cols, dtype=np.int32)]
        b_red = b[np.asarray(unresolved_rows, dtype=np.int32)]
        w_red = np.abs(llr_for_sort[peel_vars[np.asarray(unresolved_cols, dtype=np.int32)]]).astype(np.float64, copy=False)

        ok_dense, x_red, free_dim, dense_xor_ops = _gf2_weighted_solve(
            A=A_red,
            b=b_red,
            weights=w_red,
            max_free_enum=int(getattr(cfg, "peel_max_free_enum", 12) or 12),
        )
        if not ok_dense:
            return _make_attempt_result(
                success=False,
                peel_edge_work=int(peel_edge_work),
                dense_xor_ops=int(dense_xor_ops),
                free_dim=int(free_dim),
                residual_vars=int(residual_vars),
                residual_rows=int(residual_rows),
            )

        solution[np.asarray(unresolved_cols, dtype=np.int32)] = x_red.astype(np.uint8, copy=False)

    flipped_vars = peel_vars[solution.astype(bool)]
    if flipped_vars.size == 0:
        return _make_attempt_result(
            success=False,
            peel_edge_work=int(peel_edge_work),
            dense_xor_ops=int(dense_xor_ops),
            free_dim=int(free_dim),
            residual_vars=int(residual_vars),
            residual_rows=int(residual_rows),
        )

    syn_w, e_cnt, uq_cnt, tg_cnt = _syndrome_weight_and_counts_after_flips_from_base(
        base_syndrome=syndrome,
        base_weight=int(initial_syndrome_weight),
        flipped_vars=[int(v) for v in flipped_vars.tolist()],
        code_cfg=code_cfg,
    )
    if int(syn_w) != 0:
        return _make_attempt_result(
            success=False,
            peel_edge_work=int(peel_edge_work),
            dense_xor_ops=int(dense_xor_ops),
            free_dim=int(free_dim),
            residual_vars=int(residual_vars),
            residual_rows=int(residual_rows),
        )

    final_bit_errors = _bit_errors_after_flips_from_base(
        base_bits=hard_bits_snapshot,
        true_bits=frame.c_bits,
        base_bit_errors=int(initial_bit_errors),
        flipped_vars=[int(v) for v in flipped_vars.tolist()],
    )

    return _make_attempt_result(
        success=True,
        flipped_vars=flipped_vars.astype(np.int32, copy=False),
        final_bit_errors=int(final_bit_errors),
        peel_edge_work=int(peel_edge_work),
        dense_xor_ops=int(dense_xor_ops),
        free_dim=int(free_dim),
        residual_vars=int(residual_vars),
        residual_rows=int(residual_rows),
        e_cnt=int(e_cnt),
        uq_cnt=int(uq_cnt),
        tg_cnt=int(tg_cnt),
    )



def run_local_rescue_with_optional_presolver(frame: FrameLog,
                                             sim_cfg: SimulationConfig,
                                             snapshot_iter: int,
                                             cfg: ClusterGrandConfig) -> ClusterGrandResult:
    """Stage-2 wrapper.

    Supported stronger front-ends:
      - Receiver 3 : peel + weighted GF(2) pre-solver
      - Receiver 4 : Chase-list + short-LDPC polish, then peel, then GRAND
      - Receiver 5 : local OSD + anchored full-graph restarts, then peel, then GRAND
      - Receiver 6 : soft local hypotheses + anchored full-graph restarts, then peel, then GRAND
      - Receiver 7 : basis-GRAND + block-debias anchored restarts, then peel, then GRAND
    """
    mode = str(getattr(cfg, "pre_solver_mode", "none") or "none").strip().lower()

    def _copy_attrs(dst: ClusterGrandResult,
                    src: Optional[ClusterGrandResult],
                    attrs: List[str]) -> None:
        if src is None:
            return
        for attr in attrs:
            if not hasattr(src, attr):
                continue
            src_val = getattr(src, attr)
            cur = getattr(dst, attr, None)
            should_copy = (cur is None)
            if not should_copy:
                if isinstance(cur, (int, np.integer)) and int(cur) == 0 and isinstance(src_val, (int, np.integer)) and int(src_val) != 0:
                    should_copy = True
                elif isinstance(cur, str) and cur in ("", "none") and isinstance(src_val, str) and src_val not in ("", "none"):
                    should_copy = True
                elif isinstance(cur, np.ndarray) and isinstance(src_val, np.ndarray) and cur.size == 0 and src_val.size > 0:
                    should_copy = True
            if should_copy:
                setattr(dst, attr, src_val)

    def _merge_failure_progress(dst: ClusterGrandResult,
                                *sources: Optional[ClusterGrandResult]) -> None:
        if dst is None or bool(getattr(dst, "success", False)):
            return
        best_syn = int(getattr(dst, "final_syndrome_weight", getattr(dst, "initial_syndrome_weight", 0)) or 0)
        best_be = int(getattr(dst, "final_bit_errors", getattr(dst, "initial_bit_errors", 0)) or 0)
        best_src = str(getattr(dst, "stage2_profile_name", "grand") or "grand")

        def _consider(src_obj: Optional[ClusterGrandResult], label: str) -> None:
            nonlocal best_syn, best_be, best_src
            if src_obj is None:
                return
            candidates = []
            for attr in ("best_progress_syndrome_weight", "final_syndrome_weight", "restart_best_syndrome_weight", "chase_best_syndrome_weight"):
                try:
                    val = getattr(src_obj, attr)
                except Exception:
                    continue
                if val is None:
                    continue
                try:
                    candidates.append(int(val))
                except Exception:
                    pass
            if not candidates:
                return
            src_best_syn = min(candidates)
            try:
                src_best_be = int(getattr(src_obj, "best_progress_bit_errors", getattr(src_obj, "final_bit_errors", best_be)) or best_be)
            except Exception:
                src_best_be = best_be
            if (src_best_syn < best_syn) or (src_best_syn == best_syn and src_best_be < best_be):
                best_syn = int(src_best_syn)
                best_be = int(src_best_be)
                best_src = str(label)

        for idx, src in enumerate(sources, start=1):
            _consider(src, f"presolver{idx}")

        setattr(dst, "final_syndrome_weight", int(best_syn))
        setattr(dst, "best_progress_syndrome_weight", int(best_syn))
        setattr(dst, "best_progress_bit_errors", int(best_be))
        setattr(dst, "best_progress_found", int(best_syn < int(getattr(dst, "initial_syndrome_weight", best_syn) or best_syn)))
        setattr(dst, "best_progress_source", str(best_src))

    peel_attrs = [
        "pre_solver_mode_used",
        "pre_solver_attempted",
        "pre_solver_success",
        "peel_candidate_size",
        "peel_residual_vars",
        "peel_residual_rows",
        "peel_edge_work",
        "peel_dense_xor_ops",
        "peel_free_dim",
        "peel_extra_llr_added",
        "disagreement_added",
    ]
    chase_attrs = [
        "chase_candidate_size",
        "chase_core_size",
        "chase_patterns_considered",
        "chase_candidates_tested",
        "chase_score_edge_visits",
        "chase_score_checks_toggled",
        "chase_score_sum_pattern_weights",
        "chase_ldpc_total_iters",
        "chase_ldpc_num_runs",
        "chase_ldpc_num_nonconverged",
        "chase_best_syndrome_weight",
        "llr_source_used",
        "selection_mode_used",
        "sv_seeded_count",
        "sv_neighbor_visits",
        "sv_score_len",
        "peel_extra_llr_added",
        "disagreement_added",
    ]
    osd_attrs = [
        "osd_candidate_size",
        "osd_matrix_rows",
        "osd_free_dim",
        "osd_enum_bits_used",
        "osd_basis_xor_ops",
        "osd_candidates_considered",
        "osd_candidates_tested",
        "osd_sum_candidate_weights",
        "restart_num_runs",
        "restart_total_ldpc_iters",
        "restart_num_nonconverged",
        "restart_anchor_bits_total",
        "restart_best_syndrome_weight",
        "llr_source_used",
        "selection_mode_used",
        "sv_seeded_count",
        "sv_neighbor_visits",
        "sv_score_len",
        "peel_extra_llr_added",
        "disagreement_added",
    ]
    soft_attrs = [
        "chase_candidate_size",
        "chase_core_size",
        "chase_patterns_considered",
        "chase_candidates_tested",
        "chase_score_edge_visits",
        "chase_score_checks_toggled",
        "chase_score_sum_pattern_weights",
        "chase_best_syndrome_weight",
        "restart_num_runs",
        "restart_total_ldpc_iters",
        "restart_num_nonconverged",
        "restart_anchor_bits_total",
        "restart_best_syndrome_weight",
        "llr_source_used",
        "selection_mode_used",
        "sv_seeded_count",
        "sv_neighbor_visits",
        "sv_score_len",
        "peel_extra_llr_added",
        "disagreement_added",
    ]
    basis_attrs = list(soft_attrs)

    if mode in ("basis_anchor", "receiver7", "bgr", "hybbgr", "basis"):
        res_basis = _run_presolver_basis_anchor(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=snapshot_iter,
            cfg=cfg,
        )
        if (res_basis is not None) and bool(res_basis.success):
            return res_basis

        res_peel = _run_presolver_peel_gf2(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=snapshot_iter,
            cfg=cfg,
        )
        if (res_peel is not None) and bool(res_peel.success):
            _copy_attrs(res_peel, res_basis, basis_attrs)
            setattr(res_peel, "pre_solver_mode_used", "basis_anchor+peel_gf2")
            setattr(res_peel, "pre_solver_attempted", 1)
            setattr(res_peel, "pre_solver_success", 1)
            return res_peel

        res = run_local_grand_on_union_of_clusters(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=snapshot_iter,
            cfg=cfg,
        )
        _copy_attrs(res, res_basis, basis_attrs)
        _copy_attrs(res, res_peel, peel_attrs)
        _merge_failure_progress(res, res_basis, res_peel)
        if (res_basis is not None) or (res_peel is not None):
            setattr(res, "pre_solver_attempted", 1)
            setattr(res, "pre_solver_success", 0)
            setattr(res, "pre_solver_mode_used", "basis_anchor+peel_gf2")
        return res

    if mode in ("soft_anchor", "receiver6", "ahr", "hybahr", "soft"):
        res_soft = _run_presolver_soft_anchor(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=snapshot_iter,
            cfg=cfg,
        )
        if (res_soft is not None) and bool(res_soft.success):
            return res_soft

        res_peel = _run_presolver_peel_gf2(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=snapshot_iter,
            cfg=cfg,
        )
        if (res_peel is not None) and bool(res_peel.success):
            _copy_attrs(res_peel, res_soft, soft_attrs)
            setattr(res_peel, "pre_solver_mode_used", "soft_anchor+peel_gf2")
            setattr(res_peel, "pre_solver_attempted", 1)
            setattr(res_peel, "pre_solver_success", 1)
            return res_peel

        res = run_local_grand_on_union_of_clusters(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=snapshot_iter,
            cfg=cfg,
        )
        _copy_attrs(res, res_soft, soft_attrs)
        _copy_attrs(res, res_peel, peel_attrs)
        _merge_failure_progress(res, res_soft, res_peel)
        if (res_soft is not None) or (res_peel is not None):
            setattr(res, "pre_solver_attempted", 1)
            setattr(res, "pre_solver_success", 0)
            setattr(res, "pre_solver_mode_used", "soft_anchor+peel_gf2")
        return res

    if mode in ("osd_anchor", "receiver5", "osd", "hybosd"):
        res_osd = _run_presolver_osd_anchor(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=snapshot_iter,
            cfg=cfg,
        )
        if (res_osd is not None) and bool(res_osd.success):
            return res_osd

        res_peel = _run_presolver_peel_gf2(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=snapshot_iter,
            cfg=cfg,
        )
        if (res_peel is not None) and bool(res_peel.success):
            _copy_attrs(res_peel, res_osd, osd_attrs)
            setattr(res_peel, "pre_solver_mode_used", "osd_anchor+peel_gf2")
            setattr(res_peel, "pre_solver_attempted", 1)
            setattr(res_peel, "pre_solver_success", 1)
            return res_peel

        res = run_local_grand_on_union_of_clusters(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=snapshot_iter,
            cfg=cfg,
        )
        _copy_attrs(res, res_osd, osd_attrs)
        _copy_attrs(res, res_peel, peel_attrs)
        _merge_failure_progress(res, res_osd, res_peel)
        if (res_osd is not None) or (res_peel is not None):
            setattr(res, "pre_solver_attempted", 1)
            setattr(res, "pre_solver_success", 0)
            setattr(res, "pre_solver_mode_used", "osd_anchor+peel_gf2")
        return res

    if mode in ("chase_list", "receiver4", "ctg"):
        res_chase = _run_presolver_chase_list(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=snapshot_iter,
            cfg=cfg,
        )
        if (res_chase is not None) and bool(res_chase.success):
            return res_chase

        res_peel = _run_presolver_peel_gf2(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=snapshot_iter,
            cfg=cfg,
        )
        if (res_peel is not None) and bool(res_peel.success):
            _copy_attrs(res_peel, res_chase, chase_attrs)
            setattr(res_peel, "pre_solver_mode_used", "chase_list+peel_gf2")
            setattr(res_peel, "pre_solver_attempted", 1)
            setattr(res_peel, "pre_solver_success", 1)
            return res_peel

        res = run_local_grand_on_union_of_clusters(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=snapshot_iter,
            cfg=cfg,
        )
        _copy_attrs(res, res_chase, chase_attrs)
        _copy_attrs(res, res_peel, peel_attrs)
        _merge_failure_progress(res, res_chase, res_peel)
        if (res_chase is not None) or (res_peel is not None):
            setattr(res, "pre_solver_attempted", 1)
            setattr(res, "pre_solver_success", 0)
            setattr(res, "pre_solver_mode_used", "chase_list+peel_gf2")
        return res

    if mode in ("peel_gf2", "receiver3", "ptg"):
        res_pre = _run_presolver_peel_gf2(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=snapshot_iter,
            cfg=cfg,
        )
        if (res_pre is not None) and bool(res_pre.success):
            return res_pre

        res = run_local_grand_on_union_of_clusters(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=snapshot_iter,
            cfg=cfg,
        )
        if res_pre is not None:
            _copy_attrs(res, res_pre, peel_attrs)
            setattr(res, "pre_solver_attempted", 1)
            setattr(res, "pre_solver_success", 0)
        else:
            setattr(res, "pre_solver_mode_used", mode)
            setattr(res, "pre_solver_attempted", 1)
            setattr(res, "pre_solver_success", 0)
        return res

    return run_local_grand_on_union_of_clusters(
        frame=frame,
        sim_cfg=sim_cfg,
        snapshot_iter=snapshot_iter,
        cfg=cfg,
    )


def run_local_grand_on_union_of_clusters(frame: FrameLog,
                                         sim_cfg: SimulationConfig,
                                         snapshot_iter: int,
                                         cfg: ClusterGrandConfig) -> ClusterGrandResult:
    """
    Local GRAND-style search over the *union* of all variable clusters
    induced by the snapshot syndrome at iteration `snapshot_iter`.

    Membership test:
      - incremental syndrome updates (no full syndrome recomputation per pattern)

    IMPORTANT for hardware-time modeling:
      - We distinguish "tested" work (up to first success) from "evaluated" work
        (the actual chunk/batch work performed by the batch-parallel engine).
      - We also expose front-end GRAND work: union sorting length, pattern generation,
        and pattern ordering complexity proxies.

    Notes:
      - The returned ClusterGrandResult keeps the original fields for backward
        compatibility (tested counters).
      - Extra hardware-relevant fields are attached dynamically as attributes:
            patterns_evaluated
            total_v2c_edge_visits_evaluated
            total_unique_checks_visited_evaluated
            total_unique_checks_toggled_evaluated
            union_size, search_size, llr_sort_len
            sum_pattern_weights_generated
            cluster_unsat_edges, cluster_pair_edges
            num_batches_evaluated
            positions_packed_evaluated
            batch_size_used
            llr_source_used
    """
    code_cfg = sim_cfg.code

    # ---- Extract snapshot data ----
    snaps = frame.snapshots
    syn_snaps = snaps.get("syndrome", {})
    hard_snaps = snaps.get("hard_bits", {})
    llr_snaps = snaps.get("llr", {})

    if (snapshot_iter not in syn_snaps or
        snapshot_iter not in hard_snaps or
        snapshot_iter not in llr_snaps):
        raise ValueError(
            f"Snapshot at iter {snapshot_iter} is not fully available "
            f"(keys: syndrome={list(syn_snaps.keys())}, "
            f"hard_bits={list(hard_snaps.keys())}, "
            f"llr={list(llr_snaps.keys())})"
        )

    syndrome = syn_snaps[snapshot_iter]
    hard_bits_snapshot = hard_snaps[snapshot_iter].copy()
    llr_snapshot = llr_snaps[snapshot_iter]

    initial_syndrome_weight = int(syndrome.sum())
    diff_init = (hard_bits_snapshot != frame.c_bits)
    initial_bit_errors = int(diff_init.sum())

    # ---- LLR source selection (posterior vs channel vs mixed) ----
    llr_for_sort, llr_source_used = _resolve_sort_llr_vector(
        llr_snapshot=llr_snapshot,
        llr_channel=getattr(frame, "llr_channel", None),
        cfg=cfg,
    )

    # ---- Cluster-complexity proxy counters from the snapshot syndrome ----
    #   cluster_unsat_edges: total degree sum over unsatisfied checks
    #   cluster_pair_edges : sum over unsat checks of (deg choose 2)
    unsat_checks = np.flatnonzero(syndrome).astype(np.int32)

    cluster_unsat_edges = 0
    cluster_pair_edges = 0
    if unsat_checks.size > 0:
        for j in unsat_checks:
            # FIX: CodeConfig uses checks_to_vars (plural), not check_to_vars
            neigh = code_cfg.checks_to_vars[int(j)]
            d = int(neigh.size)
            cluster_unsat_edges += d
            if d >= 2:
                cluster_pair_edges += int(d * (d - 1) // 2)

    # If already a codeword, nothing to do
    if initial_syndrome_weight == 0:
        res = ClusterGrandResult(
            success=True,
            pattern_weight=0,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        # Attach meta (mostly zeros)
        setattr(res, "patterns_evaluated", 0)
        setattr(res, "total_v2c_edge_visits_evaluated", 0)
        setattr(res, "total_unique_checks_visited_evaluated", 0)
        setattr(res, "total_unique_checks_toggled_evaluated", 0)
        setattr(res, "union_size", 0)
        setattr(res, "search_size", 0)
        setattr(res, "llr_sort_len", 0)
        setattr(res, "sum_pattern_weights_generated", 0)
        setattr(res, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res, "num_batches_evaluated", 0)
        setattr(res, "positions_packed_evaluated", 0)
        setattr(res, "batch_size_used", 0)
        setattr(res, "llr_source_used", str(llr_source_used))
        return res

    # Optional guardrail: skip GRAND when syndrome is huge (likely too many errors)
    max_syn = getattr(cfg, "max_syndrome_weight_for_grand", None)
    if isinstance(max_syn, int) and max_syn > 0 and initial_syndrome_weight > int(max_syn):
        res = ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        setattr(res, "patterns_evaluated", 0)
        setattr(res, "total_v2c_edge_visits_evaluated", 0)
        setattr(res, "total_unique_checks_visited_evaluated", 0)
        setattr(res, "total_unique_checks_toggled_evaluated", 0)
        setattr(res, "union_size", 0)
        setattr(res, "search_size", 0)
        setattr(res, "llr_sort_len", 0)
        setattr(res, "sum_pattern_weights_generated", 0)
        setattr(res, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res, "num_batches_evaluated", 0)
        setattr(res, "positions_packed_evaluated", 0)
        setattr(res, "batch_size_used", 0)
        setattr(res, "llr_source_used", "skipped")
        return res

    # Reliability + fade gating
    allowed_mask = build_allowed_mask_from_config(frame, sim_cfg, snapshot_iter, cfg)

    # Build unsatisfied-check clusters (structure only)
    clusters = find_variable_clusters_from_syndrome(syndrome, code_cfg)
    if not clusters:
        res = ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        setattr(res, "patterns_evaluated", 0)
        setattr(res, "total_v2c_edge_visits_evaluated", 0)
        setattr(res, "total_unique_checks_visited_evaluated", 0)
        setattr(res, "total_unique_checks_toggled_evaluated", 0)
        setattr(res, "union_size", 0)
        setattr(res, "search_size", 0)
        setattr(res, "llr_sort_len", 0)
        setattr(res, "sum_pattern_weights_generated", 0)
        setattr(res, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res, "num_batches_evaluated", 0)
        setattr(res, "positions_packed_evaluated", 0)
        setattr(res, "batch_size_used", 0)
        setattr(res, "llr_source_used", str(llr_source_used))
        return res

    # Union of all cluster variables, then intersect with allowed_mask
    union_vars = np.unique(np.concatenate(clusters)).astype(np.int32)
    union_vars = union_vars[allowed_mask[union_vars]]
    L_full = int(union_vars.size)

    if L_full == 0:
        res = ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        setattr(res, "patterns_evaluated", 0)
        setattr(res, "total_v2c_edge_visits_evaluated", 0)
        setattr(res, "total_unique_checks_visited_evaluated", 0)
        setattr(res, "total_unique_checks_toggled_evaluated", 0)
        setattr(res, "union_size", int(L_full))
        setattr(res, "search_size", 0)
        setattr(res, "llr_sort_len", int(L_full))
        setattr(res, "sum_pattern_weights_generated", 0)
        setattr(res, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res, "num_batches_evaluated", 0)
        setattr(res, "positions_packed_evaluated", 0)
        setattr(res, "batch_size_used", 0)
        setattr(res, "llr_source_used", str(llr_source_used))
        return res

    # Determine the search-space budget L, then choose the front-end ranking.
    L = _auto_pick_grand_search_size(L_full, cfg)
    selection_mode = str(getattr(cfg, "selection_mode", "llr") or "llr").strip().lower()

    if selection_mode in ("ai_tanner_subgraph_roi", "aitg2", "tanner_subgraph_roi", "receiver9_tg2"):
        search_vars, front_end_meta = _select_search_vars_ai_tanner_subgraph_roi(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=code_cfg,
            llr_for_sort=llr_for_sort,
            L=L,
            cfg=cfg,
            llr_snapshot=llr_snapshot,
            llr_channel=getattr(frame, "llr_channel", None),
        )
    elif selection_mode in ("ai_tanner_roi", "aitg", "tanner_roi", "receiver9_tg"):
        search_vars, front_end_meta = _select_search_vars_ai_tanner_roi(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=code_cfg,
            llr_for_sort=llr_for_sort,
            L=L,
            cfg=cfg,
            llr_snapshot=llr_snapshot,
            llr_channel=getattr(frame, "llr_channel", None),
        )
    elif selection_mode in ("ai_mix_roi", "aimix", "mix_roi", "receiver9_mix"):
        search_vars, front_end_meta = _select_search_vars_ai_mix_roi(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=code_cfg,
            llr_for_sort=llr_for_sort,
            L=L,
            cfg=cfg,
            llr_snapshot=llr_snapshot,
            llr_channel=getattr(frame, "llr_channel", None),
        )
    elif selection_mode in ("ai_window_roi", "aiwindow", "window_roi", "receiver9_window"):
        search_vars, front_end_meta = _select_search_vars_ai_window_roi(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=code_cfg,
            llr_for_sort=llr_for_sort,
            L=L,
            cfg=cfg,
            llr_snapshot=llr_snapshot,
            llr_channel=getattr(frame, "llr_channel", None),
        )
    elif selection_mode in ("ai_rank_roi", "airoi", "roi_rank", "receiver9_roi"):
        search_vars, front_end_meta = _select_search_vars_ai_rank_roi(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=code_cfg,
            llr_for_sort=llr_for_sort,
            L=L,
            cfg=cfg,
            llr_snapshot=llr_snapshot,
            llr_channel=getattr(frame, "llr_channel", None),
        )
    elif selection_mode in ("ai_rank", "ai", "airank", "receiver9"):
        search_vars, front_end_meta = _select_search_vars_ai_rank(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=code_cfg,
            llr_for_sort=llr_for_sort,
            L=L,
            cfg=cfg,
            llr_snapshot=llr_snapshot,
            llr_channel=getattr(frame, "llr_channel", None),
        )
    elif selection_mode in ("syndrome_vote", "sv", "receiver2"):
        search_vars, front_end_meta = _select_search_vars_syndrome_vote(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=code_cfg,
            llr_for_sort=llr_for_sort,
            L=L,
            cfg=cfg,
        )
    else:
        search_vars, front_end_meta = _select_search_vars_llr(
            union_vars=union_vars,
            llr_for_sort=llr_for_sort,
            L=L,
        )

    selection_mode_used = str(front_end_meta.get("selection_mode_used", "llr"))
    sv_seeded_count = int(front_end_meta.get("sv_seeded_count", 0))
    sv_neighbor_visits = int(front_end_meta.get("sv_neighbor_visits", 0))
    sv_score_len = int(front_end_meta.get("sv_score_len", L_full))

    L = int(search_vars.size)

    if L == 0:
        res = ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        setattr(res, "patterns_evaluated", 0)
        setattr(res, "total_v2c_edge_visits_evaluated", 0)
        setattr(res, "total_unique_checks_visited_evaluated", 0)
        setattr(res, "total_unique_checks_toggled_evaluated", 0)
        setattr(res, "union_size", int(L_full))
        setattr(res, "search_size", int(L))
        setattr(res, "llr_sort_len", int(L_full))
        setattr(res, "sum_pattern_weights_generated", 0)
        setattr(res, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res, "num_batches_evaluated", 0)
        setattr(res, "positions_packed_evaluated", 0)
        setattr(res, "batch_size_used", 0)
        setattr(res, "llr_source_used", str(llr_source_used))
        setattr(res, "selection_mode_used", str(selection_mode_used))
        setattr(res, "sv_seeded_count", int(sv_seeded_count))
        setattr(res, "sv_neighbor_visits", int(sv_neighbor_visits))
        setattr(res, "sv_score_len", int(sv_score_len))
        return res

    patterns_tested = 0
    patterns_evaluated = 0
    found = False
    found_weight = -1
    found_flipped = np.array([], dtype=np.int32)
    final_syn_weight = initial_syndrome_weight
    final_bit_errors = initial_bit_errors
    best_progress_syn_weight = int(initial_syndrome_weight)
    best_progress_bit_errors = int(initial_bit_errors)
    best_progress_weight = -1
    best_progress_flipped = np.array([], dtype=np.int32)

    def _update_best_progress(candidate_syn: int,
                              candidate_bit_errors: int,
                              candidate_weight: int,
                              candidate_flipped: Sequence[int]) -> None:
        nonlocal best_progress_syn_weight, best_progress_bit_errors, best_progress_weight, best_progress_flipped
        syn_i = int(candidate_syn)
        be_i = int(candidate_bit_errors)
        w_i = int(candidate_weight)
        better = (
            syn_i < best_progress_syn_weight
            or (syn_i == best_progress_syn_weight and be_i < best_progress_bit_errors)
            or (syn_i == best_progress_syn_weight and be_i == best_progress_bit_errors and (best_progress_weight < 0 or (w_i >= 0 and w_i < best_progress_weight)))
        )
        if better:
            best_progress_syn_weight = int(syn_i)
            best_progress_bit_errors = int(be_i)
            best_progress_weight = int(w_i)
            best_progress_flipped = np.asarray(list(candidate_flipped), dtype=np.int32)

    # ---- Counters (tested vs evaluated) ----
    total_edge_visits_tested = 0
    total_uniq_checks_visited_tested = 0
    total_uniq_checks_toggled_tested = 0

    total_edge_visits_eval = 0
    total_uniq_checks_visited_eval = 0
    total_uniq_checks_toggled_eval = 0

    # Batch/packing overhead proxy: total pattern positions packed (sum of weights)
    positions_packed_eval = 0
    num_batches_evaluated = 0
    batch_size_used = 0

    if cfg.max_weight <= 0:
        res = ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        setattr(res, "patterns_evaluated", 0)
        setattr(res, "total_v2c_edge_visits_evaluated", 0)
        setattr(res, "total_unique_checks_visited_evaluated", 0)
        setattr(res, "total_unique_checks_toggled_evaluated", 0)
        setattr(res, "union_size", int(L_full))
        setattr(res, "search_size", int(L))
        setattr(res, "llr_sort_len", int(L_full))
        setattr(res, "sum_pattern_weights_generated", 0)
        setattr(res, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res, "num_batches_evaluated", 0)
        setattr(res, "positions_packed_evaluated", 0)
        setattr(res, "batch_size_used", 0)
        setattr(res, "llr_source_used", str(llr_source_used))
        return res

    max_w = min(int(cfg.max_weight), int(L))

    # ---- Build all patterns and order them by sum‑|LLR| ----
    pattern_items: List[tuple] = []
    abs_llr_local = np.abs(llr_for_sort[search_vars])

    # This counts the total number of abs-LLR terms summed across all generated patterns.
    # It is a direct proxy for "pattern cost computation" complexity.
    sum_pattern_weights_generated = 0

    for w in range(1, max_w + 1):
        for comb in itertools.combinations(range(L), w):
            # cost = sum_{i in comb} |LLR_i|
            # Hardware ops proxy: w reads + (w-1) adds (we store w; model can convert)
            sum_pattern_weights_generated += int(w)
            cost = float(abs_llr_local[list(comb)].sum())
            pattern_items.append((cost, w, comb))

    pattern_items.sort(key=lambda t: (t[0], t[1]))
    patterns_generated = int(len(pattern_items))

    if not pattern_items:
        res = ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        setattr(res, "patterns_evaluated", 0)
        setattr(res, "total_v2c_edge_visits_evaluated", 0)
        setattr(res, "total_unique_checks_visited_evaluated", 0)
        setattr(res, "total_unique_checks_toggled_evaluated", 0)
        setattr(res, "union_size", int(L_full))
        setattr(res, "search_size", int(L))
        setattr(res, "llr_sort_len", int(L_full))
        setattr(res, "sum_pattern_weights_generated", int(sum_pattern_weights_generated))
        setattr(res, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res, "num_batches_evaluated", 0)
        setattr(res, "positions_packed_evaluated", 0)
        setattr(res, "batch_size_used", 0)
        setattr(res, "llr_source_used", str(llr_source_used))
        return res

    # ---- Pattern testing: batch-parallel (evaluated counters) or sequential (tested counters) ----
    use_batch = (
        NUMBA_AVAILABLE
        and hasattr(code_cfg, "_v2c_ptrs")
        and hasattr(code_cfg, "_v2c_checks")
        and getattr(cfg, "batch_size", 0) > 0
    )

    if use_batch:
        base_bits = hard_bits_snapshot.astype(np.uint8)
        true_c_bits = frame.c_bits.astype(np.uint8)
        search_vars_int = search_vars.astype(np.int64)

        base_syn = syndrome.astype(np.uint8)
        base_syn_w = np.int32(initial_syndrome_weight)
        base_bit_err = np.int32(initial_bit_errors)

        total_patterns = int(len(pattern_items))
        max_patterns = int(cfg.max_patterns)
        limit = int(min(total_patterns, max_patterns))
        batch_size = int(getattr(cfg, "batch_size", 256))
        if batch_size <= 0:
            batch_size = limit
        batch_size_used = int(batch_size)

        for start_idx in range(0, limit, batch_size):
            end_idx = min(start_idx + batch_size, limit)
            num_batch = end_idx - start_idx
            if num_batch <= 0:
                continue

            num_batches_evaluated += 1
            patterns_evaluated += int(num_batch)

            # Pack this batch (proxy cost: positions_packed_eval += sum(weights))
            total_positions = 0
            for i in range(start_idx, end_idx):
                total_positions += int(pattern_items[i][1])
            positions_packed_eval += int(total_positions)

            pattern_starts = np.zeros(num_batch, dtype=np.int32)
            pattern_lengths = np.zeros(num_batch, dtype=np.int32)
            pattern_positions = np.zeros(total_positions, dtype=np.int32)

            pos_ptr = 0
            for b in range(num_batch):
                _, w, comb = pattern_items[start_idx + b]
                pattern_starts[b] = pos_ptr
                pattern_lengths[b] = int(w)
                for lp in comb:
                    pattern_positions[pos_ptr] = int(lp)
                    pos_ptr += 1

            syn_w_arr, bit_err_arr, edge_arr, uniq_arr, tog_arr = _grand_eval_batch_numba_incremental(
                base_syn,
                base_syn_w,
                base_bit_err,
                base_bits,
                true_c_bits,
                search_vars_int,
                pattern_starts,
                pattern_lengths,
                pattern_positions,
                code_cfg._v2c_ptrs,
                code_cfg._v2c_checks,
            )

            # Hardware-evaluated counters: full batch always evaluated
            total_edge_visits_eval += int(edge_arr.sum())
            total_uniq_checks_visited_eval += int(np.maximum(uniq_arr, 0).sum())
            total_uniq_checks_toggled_eval += int(tog_arr.sum())

            # Track the best partial syndrome improvement in this batch as well.
            if num_batch > 0:
                try:
                    syn_w_batch = np.asarray(syn_w_arr, dtype=np.int64)
                    bit_err_batch = np.asarray(bit_err_arr, dtype=np.int64)
                    best_rel = min(
                        range(num_batch),
                        key=lambda b: (int(syn_w_batch[b]), int(bit_err_batch[b]), int(pattern_items[start_idx + b][1]))
                    )
                    best_global_idx = start_idx + int(best_rel)
                    _, best_w, best_comb = pattern_items[best_global_idx]
                    best_flipped_batch = [int(search_vars[pos]) for pos in best_comb]
                    _update_best_progress(
                        candidate_syn=int(syn_w_batch[best_rel]),
                        candidate_bit_errors=int(bit_err_batch[best_rel]) if int(bit_err_batch[best_rel]) >= 0 else int(initial_bit_errors),
                        candidate_weight=int(best_w),
                        candidate_flipped=best_flipped_batch,
                    )
                except Exception:
                    pass

            # Find first success in this batch
            success_rel = -1
            for b in range(num_batch):
                if syn_w_arr[b] == 0:
                    success_rel = b
                    break

            if success_rel >= 0:
                # Tested counters: only up to success
                total_edge_visits_tested += int(edge_arr[:success_rel + 1].sum())
                total_uniq_checks_visited_tested += int(np.maximum(uniq_arr[:success_rel + 1], 0).sum())
                total_uniq_checks_toggled_tested += int(tog_arr[:success_rel + 1].sum())

                global_idx = start_idx + success_rel
                patterns_tested = int(global_idx + 1)

                _, w, comb = pattern_items[global_idx]
                flipped = [int(search_vars[pos]) for pos in comb]

                found = True
                found_weight = int(w)
                found_flipped = np.array(flipped, dtype=np.int32)
                final_syn_weight = 0
                be = int(bit_err_arr[success_rel])
                final_bit_errors = be if be >= 0 else int(initial_bit_errors)
                break
            else:
                # Tested counters: entire batch tested
                total_edge_visits_tested += int(edge_arr.sum())
                total_uniq_checks_visited_tested += int(np.maximum(uniq_arr, 0).sum())
                total_uniq_checks_toggled_tested += int(tog_arr.sum())
                patterns_tested = int(end_idx)

    else:
        # Sequential one‑by‑one testing (incremental membership + counters)
        for _, w, comb in pattern_items:
            patterns_tested += 1
            if patterns_tested > int(cfg.max_patterns):
                break

            flipped = [int(search_vars[pos]) for pos in comb]

            syn_w, e_cnt, uq_cnt, tg_cnt = _syndrome_weight_and_counts_after_flips_from_base(
                base_syndrome=syndrome,
                base_weight=int(initial_syndrome_weight),
                flipped_vars=flipped,
                code_cfg=code_cfg,
            )

            total_edge_visits_tested += int(e_cnt)
            total_uniq_checks_visited_tested += int(uq_cnt)
            total_uniq_checks_toggled_tested += int(tg_cnt)

            # In sequential mode, evaluated == tested
            patterns_evaluated = int(patterns_tested)
            total_edge_visits_eval = int(total_edge_visits_tested)
            total_uniq_checks_visited_eval = int(total_uniq_checks_visited_tested)
            total_uniq_checks_toggled_eval = int(total_uniq_checks_toggled_tested)

            if int(syn_w) < int(best_progress_syn_weight):
                bit_err_progress = _bit_errors_after_flips_from_base(
                    base_bits=hard_bits_snapshot,
                    true_bits=frame.c_bits,
                    base_bit_errors=int(initial_bit_errors),
                    flipped_vars=flipped,
                )
                _update_best_progress(
                    candidate_syn=int(syn_w),
                    candidate_bit_errors=int(bit_err_progress),
                    candidate_weight=int(w),
                    candidate_flipped=flipped,
                )

            if syn_w == 0:
                bit_err_cand = _bit_errors_after_flips_from_base(
                    base_bits=hard_bits_snapshot,
                    true_bits=frame.c_bits,
                    base_bit_errors=int(initial_bit_errors),
                    flipped_vars=flipped,
                )

                found = True
                found_weight = int(w)
                found_flipped = np.array(flipped, dtype=np.int32)
                final_syn_weight = 0
                final_bit_errors = int(bit_err_cand)
                break

    if not found:
        final_syn_weight = int(best_progress_syn_weight)
        final_bit_errors = int(best_progress_bit_errors)

    # Build result (keep legacy semantics: these are TESTED counters)
    res = ClusterGrandResult(
        success=bool(found),
        pattern_weight=int(found_weight) if found else -1,
        flipped_vars=found_flipped,
        patterns_tested=int(patterns_tested),
        initial_syndrome_weight=int(initial_syndrome_weight),
        final_syndrome_weight=int(final_syn_weight),
        initial_bit_errors=int(initial_bit_errors),
        final_bit_errors=int(final_bit_errors),
        total_v2c_edge_visits=int(total_edge_visits_tested),
        total_unique_checks_visited=int(total_uniq_checks_visited_tested),
        total_unique_checks_toggled=int(total_uniq_checks_toggled_tested),
        patterns_generated=int(patterns_generated),
    )

    setattr(res, "best_progress_syndrome_weight", int(best_progress_syn_weight))
    setattr(res, "best_progress_bit_errors", int(best_progress_bit_errors))
    setattr(res, "best_progress_pattern_weight", int(best_progress_weight))
    setattr(res, "best_progress_found", int(best_progress_syn_weight < int(initial_syndrome_weight)))
    setattr(res, "best_progress_flipped_vars", np.asarray(best_progress_flipped, dtype=np.int32))

    # Attach evaluated counters + front-end meta for hardware-time model
    setattr(res, "patterns_evaluated", int(patterns_evaluated))
    setattr(res, "total_v2c_edge_visits_evaluated", int(total_edge_visits_eval))
    setattr(res, "total_unique_checks_visited_evaluated", int(total_uniq_checks_visited_eval))
    setattr(res, "total_unique_checks_toggled_evaluated", int(total_uniq_checks_toggled_eval))

    setattr(res, "union_size", int(L_full))
    setattr(res, "search_size", int(L))
    setattr(res, "llr_sort_len", int(L_full))

    setattr(res, "sum_pattern_weights_generated", int(sum_pattern_weights_generated))

    setattr(res, "cluster_unsat_edges", int(cluster_unsat_edges))
    setattr(res, "cluster_pair_edges", int(cluster_pair_edges))

    setattr(res, "num_batches_evaluated", int(num_batches_evaluated))
    setattr(res, "positions_packed_evaluated", int(positions_packed_eval))
    setattr(res, "batch_size_used", int(batch_size_used))

    setattr(res, "llr_source_used", str(llr_source_used))
    setattr(res, "selection_mode_used", str(selection_mode_used))
    setattr(res, "sv_seeded_count", int(sv_seeded_count))
    setattr(res, "sv_neighbor_visits", int(sv_neighbor_visits))
    setattr(res, "sv_score_len", int(sv_score_len))

    return res







# ==================== CELL 28 (DROP-IN REPLACEMENT) ====================
import os

def _get_int_env(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

def _get_float_env(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default

# ---- Baseline GRAND (keep your current behavior) ----
grand_cfg_awgn = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_MAX_WEIGHT", 5),
    max_patterns=_get_int_env("GRAND_MAX_PATTERNS", 5000),
    max_bits_from_cluster=None,     # keep auto-pick L
    verbose=False,
    # IMPORTANT: channel-LLR ordering is usually more robust than posterior-LLR ordering
    llr_source=os.environ.get("GRAND_LLR_SOURCE", "channel").strip().lower(),
    pattern_overgen_ratio=_get_float_env("GRAND_OVERGEN", 1.02),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_BATCH_SIZE", 256),
)

# ---- Boost GRAND (only for baseline-exhausted hard frames) ----
GRAND_USE_BOOST = bool(_get_int_env("GRAND_USE_BOOST", 1))

grand_cfg_awgn_boost = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_BOOST_MAX_WEIGHT", 5),
    max_patterns=_get_int_env("GRAND_BOOST_MAX_PATTERNS", 15000),
    max_bits_from_cluster=None,     # keep auto-pick L
    verbose=False,
    llr_source=os.environ.get("GRAND_LLR_SOURCE", "channel").strip().lower(),
    pattern_overgen_ratio=_get_float_env("GRAND_BOOST_OVERGEN", 1.02),
    # Optional: set to e.g. 0 or leave None; using None keeps boost always eligible
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_BATCH_SIZE", 256),
)

# ---- Receiver 2: syndrome-vote + check-cover front-end ----
RUN_RECEIVER2 = bool(_get_int_env("RUN_RECEIVER2", 0))
GRAND_SV_USE_BOOST = bool(_get_int_env("GRAND_SV_USE_BOOST", _get_int_env("GRAND_USE_BOOST", 1)))

grand_cfg_awgn_sv = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_SV_MAX_WEIGHT", _get_int_env("GRAND_MAX_WEIGHT", 5)),
    max_patterns=_get_int_env("GRAND_SV_MAX_PATTERNS", _get_int_env("GRAND_MAX_PATTERNS", 5000)),
    max_bits_from_cluster=None,     # keep auto-pick L
    verbose=False,
    llr_source=os.environ.get(
        "GRAND_SV_LLR_SOURCE",
        os.environ.get("GRAND_LLR_SOURCE", "channel"),
    ).strip().lower(),
    pattern_overgen_ratio=_get_float_env(
        "GRAND_SV_OVERGEN",
        _get_float_env("GRAND_OVERGEN", 1.02),
    ),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_SV_BATCH_SIZE", _get_int_env("GRAND_BATCH_SIZE", 256)),
    selection_mode="syndrome_vote",
    sv_epsilon=_get_float_env("GRAND_SV_EPSILON", 1e-3),
    sv_check_cover_k=_get_int_env("GRAND_SV_CHECK_COVER_K", 1),
)

grand_cfg_awgn_sv_boost = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_SV_BOOST_MAX_WEIGHT", _get_int_env("GRAND_BOOST_MAX_WEIGHT", 5)),
    max_patterns=_get_int_env("GRAND_SV_BOOST_MAX_PATTERNS", _get_int_env("GRAND_BOOST_MAX_PATTERNS", 15000)),
    max_bits_from_cluster=None,     # keep auto-pick L
    verbose=False,
    llr_source=os.environ.get(
        "GRAND_SV_LLR_SOURCE",
        os.environ.get("GRAND_LLR_SOURCE", "channel"),
    ).strip().lower(),
    pattern_overgen_ratio=_get_float_env(
        "GRAND_SV_BOOST_OVERGEN",
        _get_float_env("GRAND_BOOST_OVERGEN", 1.02),
    ),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_SV_BATCH_SIZE", _get_int_env("GRAND_BATCH_SIZE", 256)),
    selection_mode="syndrome_vote",
    sv_epsilon=_get_float_env("GRAND_SV_EPSILON", 1e-3),
    sv_check_cover_k=_get_int_env("GRAND_SV_CHECK_COVER_K", 1),
)
# ---- Receiver 3+: syndrome-vote front-end + peel/weighted-GF(2) pre-solver ----
RUN_RECEIVER3 = bool(_get_int_env("RUN_RECEIVER3", 0))
GRAND_PTG_USE_BOOST = bool(_get_int_env("GRAND_PTG_USE_BOOST", _get_int_env("GRAND_SV_USE_BOOST", _get_int_env("GRAND_USE_BOOST", 1))))

grand_cfg_awgn_ptg = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_PTG_MAX_WEIGHT", _get_int_env("GRAND_SV_MAX_WEIGHT", _get_int_env("GRAND_MAX_WEIGHT", 5))),
    max_patterns=_get_int_env("GRAND_PTG_MAX_PATTERNS", _get_int_env("GRAND_SV_MAX_PATTERNS", _get_int_env("GRAND_MAX_PATTERNS", 5000))),
    max_bits_from_cluster=None,
    verbose=False,
    llr_source=os.environ.get(
        "GRAND_PTG_LLR_SOURCE",
        os.environ.get("GRAND_SV_LLR_SOURCE", os.environ.get("GRAND_LLR_SOURCE", "mixed")),
    ).strip().lower(),
    pattern_overgen_ratio=_get_float_env(
        "GRAND_PTG_OVERGEN",
        _get_float_env("GRAND_SV_OVERGEN", _get_float_env("GRAND_OVERGEN", 1.02)),
    ),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_PTG_BATCH_SIZE", _get_int_env("GRAND_SV_BATCH_SIZE", _get_int_env("GRAND_BATCH_SIZE", 256))),
    selection_mode="syndrome_vote",
    sv_epsilon=_get_float_env("GRAND_PTG_EPSILON", _get_float_env("GRAND_SV_EPSILON", 1e-3)),
    sv_check_cover_k=_get_int_env("GRAND_PTG_CHECK_COVER_K", _get_int_env("GRAND_SV_CHECK_COVER_K", 1)),
    pre_solver_mode="peel_gf2",
    peel_candidate_ratio=_get_float_env("GRAND_PTG_PEEL_RATIO", 1.75),
    peel_max_bits=_get_int_env("GRAND_PTG_PEEL_MAX_BITS", 48),
    peel_dense_max_vars=_get_int_env("GRAND_PTG_PEEL_DENSE_MAX_VARS", 28),
    peel_max_free_enum=_get_int_env("GRAND_PTG_PEEL_MAX_FREE_ENUM", 12),
    peel_extra_llr_bits=_get_int_env("GRAND_PTG_PEEL_EXTRA_LLR_BITS", 8),
)

grand_cfg_awgn_ptg_boost = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_PTG_BOOST_MAX_WEIGHT", _get_int_env("GRAND_SV_BOOST_MAX_WEIGHT", _get_int_env("GRAND_BOOST_MAX_WEIGHT", 5))),
    max_patterns=_get_int_env("GRAND_PTG_BOOST_MAX_PATTERNS", _get_int_env("GRAND_SV_BOOST_MAX_PATTERNS", _get_int_env("GRAND_BOOST_MAX_PATTERNS", 15000))),
    max_bits_from_cluster=None,
    verbose=False,
    llr_source=os.environ.get(
        "GRAND_PTG_LLR_SOURCE",
        os.environ.get("GRAND_SV_LLR_SOURCE", os.environ.get("GRAND_LLR_SOURCE", "mixed")),
    ).strip().lower(),
    pattern_overgen_ratio=_get_float_env(
        "GRAND_PTG_BOOST_OVERGEN",
        _get_float_env("GRAND_SV_BOOST_OVERGEN", _get_float_env("GRAND_BOOST_OVERGEN", 1.02)),
    ),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_PTG_BATCH_SIZE", _get_int_env("GRAND_SV_BATCH_SIZE", _get_int_env("GRAND_BATCH_SIZE", 256))),
    selection_mode="syndrome_vote",
    sv_epsilon=_get_float_env("GRAND_PTG_EPSILON", _get_float_env("GRAND_SV_EPSILON", 1e-3)),
    sv_check_cover_k=_get_int_env("GRAND_PTG_CHECK_COVER_K", _get_int_env("GRAND_SV_CHECK_COVER_K", 1)),
    pre_solver_mode="none",  # avoid repeating the same pre-solver on the boost path
    peel_candidate_ratio=_get_float_env("GRAND_PTG_PEEL_RATIO", 1.75),
    peel_max_bits=_get_int_env("GRAND_PTG_PEEL_MAX_BITS", 48),
    peel_dense_max_vars=_get_int_env("GRAND_PTG_PEEL_DENSE_MAX_VARS", 28),
    peel_max_free_enum=_get_int_env("GRAND_PTG_PEEL_MAX_FREE_ENUM", 12),
    peel_extra_llr_bits=_get_int_env("GRAND_PTG_PEEL_EXTRA_LLR_BITS", 8),
)

# ---- Receiver 4: Chase-list + short-LDPC polish + peel + GRAND fallback ----
RUN_RECEIVER4 = bool(_get_int_env("RUN_RECEIVER4", 0))
GRAND_CTG_USE_BOOST = bool(_get_int_env("GRAND_CTG_USE_BOOST", _get_int_env("GRAND_PTG_USE_BOOST", _get_int_env("GRAND_USE_BOOST", 1))))

grand_cfg_awgn_ctg = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_CTG_MAX_WEIGHT", _get_int_env("GRAND_PTG_MAX_WEIGHT", _get_int_env("GRAND_MAX_WEIGHT", 5))),
    max_patterns=_get_int_env("GRAND_CTG_MAX_PATTERNS", max(_get_int_env("GRAND_PTG_MAX_PATTERNS", _get_int_env("GRAND_MAX_PATTERNS", 5000)), 20000)),
    max_bits_from_cluster=None,
    verbose=False,
    llr_source=os.environ.get(
        "GRAND_CTG_LLR_SOURCE",
        os.environ.get("GRAND_PTG_LLR_SOURCE", os.environ.get("GRAND_LLR_SOURCE", "mixed")),
    ).strip().lower(),
    pattern_overgen_ratio=_get_float_env(
        "GRAND_CTG_OVERGEN",
        _get_float_env("GRAND_PTG_OVERGEN", _get_float_env("GRAND_OVERGEN", 1.02)),
    ),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_CTG_BATCH_SIZE", _get_int_env("GRAND_PTG_BATCH_SIZE", _get_int_env("GRAND_BATCH_SIZE", 256))),
    selection_mode="syndrome_vote",
    sv_epsilon=_get_float_env("GRAND_CTG_EPSILON", _get_float_env("GRAND_PTG_EPSILON", _get_float_env("GRAND_SV_EPSILON", 1e-3))),
    sv_check_cover_k=_get_int_env("GRAND_CTG_CHECK_COVER_K", _get_int_env("GRAND_PTG_CHECK_COVER_K", _get_int_env("GRAND_SV_CHECK_COVER_K", 1))),
    pre_solver_mode="chase_list",
    peel_candidate_ratio=_get_float_env("GRAND_CTG_PEEL_RATIO", _get_float_env("GRAND_PTG_PEEL_RATIO", 1.75)),
    peel_max_bits=_get_int_env("GRAND_CTG_PEEL_MAX_BITS", _get_int_env("GRAND_PTG_PEEL_MAX_BITS", 48)),
    peel_dense_max_vars=_get_int_env("GRAND_CTG_PEEL_DENSE_MAX_VARS", _get_int_env("GRAND_PTG_PEEL_DENSE_MAX_VARS", 28)),
    peel_max_free_enum=_get_int_env("GRAND_CTG_PEEL_MAX_FREE_ENUM", _get_int_env("GRAND_PTG_PEEL_MAX_FREE_ENUM", 12)),
    peel_extra_llr_bits=_get_int_env("GRAND_CTG_PEEL_EXTRA_LLR_BITS", _get_int_env("GRAND_PTG_PEEL_EXTRA_LLR_BITS", 8)),
    chase_candidate_ratio=_get_float_env("GRAND_CTG_CHASE_RATIO", 2.25),
    chase_max_bits=_get_int_env("GRAND_CTG_CHASE_MAX_BITS", 64),
    chase_core_max_bits=_get_int_env("GRAND_CTG_CORE_MAX_BITS", 14),
    chase_max_weight=_get_int_env("GRAND_CTG_CORE_MAX_WEIGHT", 3),
    chase_max_candidates=_get_int_env("GRAND_CTG_MAX_CANDIDATES", 96),
    chase_ldpc_extra_iters=_get_int_env("GRAND_CTG_LDPC_EXTRA_ITERS", 6),
    chase_llr_gain=_get_float_env("GRAND_CTG_LLR_GAIN", 3.0),
    chase_llr_abs_floor=_get_float_env("GRAND_CTG_LLR_ABS_FLOOR", 5.0),
)

grand_cfg_awgn_ctg_boost = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_CTG_BOOST_MAX_WEIGHT", _get_int_env("GRAND_PTG_BOOST_MAX_WEIGHT", _get_int_env("GRAND_BOOST_MAX_WEIGHT", 5))),
    max_patterns=_get_int_env("GRAND_CTG_BOOST_MAX_PATTERNS", max(_get_int_env("GRAND_PTG_BOOST_MAX_PATTERNS", _get_int_env("GRAND_BOOST_MAX_PATTERNS", 15000)), 60000)),
    max_bits_from_cluster=None,
    verbose=False,
    llr_source=os.environ.get(
        "GRAND_CTG_LLR_SOURCE",
        os.environ.get("GRAND_PTG_LLR_SOURCE", os.environ.get("GRAND_LLR_SOURCE", "mixed")),
    ).strip().lower(),
    pattern_overgen_ratio=_get_float_env(
        "GRAND_CTG_BOOST_OVERGEN",
        _get_float_env("GRAND_PTG_BOOST_OVERGEN", _get_float_env("GRAND_BOOST_OVERGEN", 1.02)),
    ),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_CTG_BATCH_SIZE", _get_int_env("GRAND_PTG_BATCH_SIZE", _get_int_env("GRAND_BATCH_SIZE", 256))),
    selection_mode="syndrome_vote",
    sv_epsilon=_get_float_env("GRAND_CTG_EPSILON", _get_float_env("GRAND_PTG_EPSILON", _get_float_env("GRAND_SV_EPSILON", 1e-3))),
    sv_check_cover_k=_get_int_env("GRAND_CTG_CHECK_COVER_K", _get_int_env("GRAND_PTG_CHECK_COVER_K", _get_int_env("GRAND_SV_CHECK_COVER_K", 1))),
    pre_solver_mode="none",  # do not repeat Chase/peel on the boost path
    peel_candidate_ratio=_get_float_env("GRAND_CTG_PEEL_RATIO", _get_float_env("GRAND_PTG_PEEL_RATIO", 1.75)),
    peel_max_bits=_get_int_env("GRAND_CTG_PEEL_MAX_BITS", _get_int_env("GRAND_PTG_PEEL_MAX_BITS", 48)),
    peel_dense_max_vars=_get_int_env("GRAND_CTG_PEEL_DENSE_MAX_VARS", _get_int_env("GRAND_PTG_PEEL_DENSE_MAX_VARS", 28)),
    peel_max_free_enum=_get_int_env("GRAND_CTG_PEEL_MAX_FREE_ENUM", _get_int_env("GRAND_PTG_PEEL_MAX_FREE_ENUM", 12)),
    peel_extra_llr_bits=_get_int_env("GRAND_CTG_PEEL_EXTRA_LLR_BITS", _get_int_env("GRAND_PTG_PEEL_EXTRA_LLR_BITS", 8)),
    chase_candidate_ratio=_get_float_env("GRAND_CTG_CHASE_RATIO", 2.25),
    chase_max_bits=_get_int_env("GRAND_CTG_CHASE_MAX_BITS", 64),
    chase_core_max_bits=_get_int_env("GRAND_CTG_CORE_MAX_BITS", 14),
    chase_max_weight=_get_int_env("GRAND_CTG_CORE_MAX_WEIGHT", 3),
    chase_max_candidates=_get_int_env("GRAND_CTG_MAX_CANDIDATES", 96),
    chase_ldpc_extra_iters=_get_int_env("GRAND_CTG_LDPC_EXTRA_ITERS", 6),
    chase_llr_gain=_get_float_env("GRAND_CTG_LLR_GAIN", 3.0),
    chase_llr_abs_floor=_get_float_env("GRAND_CTG_LLR_ABS_FLOOR", 5.0),
)


# ---- Receiver 5: local OSD + anchored full-graph restarts + peel + GRAND fallback ----
RUN_RECEIVER5 = bool(_get_int_env("RUN_RECEIVER5", 0))
GRAND_OSD_USE_BOOST = bool(_get_int_env("GRAND_OSD_USE_BOOST", _get_int_env("GRAND_CTG_USE_BOOST", _get_int_env("GRAND_USE_BOOST", 1))))

grand_cfg_awgn_osd = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_OSD_MAX_WEIGHT", _get_int_env("GRAND_CTG_MAX_WEIGHT", _get_int_env("GRAND_MAX_WEIGHT", 5))),
    max_patterns=_get_int_env("GRAND_OSD_MAX_PATTERNS", max(_get_int_env("GRAND_CTG_MAX_PATTERNS", _get_int_env("GRAND_MAX_PATTERNS", 5000)), 80000)),
    max_bits_from_cluster=None,
    verbose=False,
    llr_source=os.environ.get(
        "GRAND_OSD_LLR_SOURCE",
        os.environ.get("GRAND_CTG_LLR_SOURCE", os.environ.get("GRAND_LLR_SOURCE", "mixed")),
    ).strip().lower(),
    pattern_overgen_ratio=_get_float_env(
        "GRAND_OSD_OVERGEN",
        _get_float_env("GRAND_CTG_OVERGEN", _get_float_env("GRAND_OVERGEN", 1.02)),
    ),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_OSD_BATCH_SIZE", _get_int_env("GRAND_CTG_BATCH_SIZE", _get_int_env("GRAND_BATCH_SIZE", 256))),
    selection_mode="syndrome_vote",
    sv_epsilon=_get_float_env("GRAND_OSD_EPSILON", _get_float_env("GRAND_CTG_EPSILON", _get_float_env("GRAND_SV_EPSILON", 1e-3))),
    sv_check_cover_k=_get_int_env("GRAND_OSD_CHECK_COVER_K", _get_int_env("GRAND_CTG_CHECK_COVER_K", _get_int_env("GRAND_SV_CHECK_COVER_K", 2))),
    pre_solver_mode="osd_anchor",
    peel_candidate_ratio=_get_float_env("GRAND_OSD_PEEL_RATIO", _get_float_env("GRAND_CTG_PEEL_RATIO", 2.0)),
    peel_max_bits=_get_int_env("GRAND_OSD_PEEL_MAX_BITS", _get_int_env("GRAND_CTG_PEEL_MAX_BITS", 64)),
    peel_dense_max_vars=_get_int_env("GRAND_OSD_PEEL_DENSE_MAX_VARS", _get_int_env("GRAND_CTG_PEEL_DENSE_MAX_VARS", 32)),
    peel_max_free_enum=_get_int_env("GRAND_OSD_PEEL_MAX_FREE_ENUM", _get_int_env("GRAND_CTG_PEEL_MAX_FREE_ENUM", 12)),
    peel_extra_llr_bits=_get_int_env("GRAND_OSD_PEEL_EXTRA_LLR_BITS", _get_int_env("GRAND_CTG_PEEL_EXTRA_LLR_BITS", 10)),
    osd_candidate_ratio=_get_float_env("GRAND_OSD_RATIO", 2.9),
    osd_max_bits=_get_int_env("GRAND_OSD_MAX_BITS", 80),
    osd_order=_get_int_env("GRAND_OSD_ORDER", 2),
    osd_enum_max_bits=_get_int_env("GRAND_OSD_ENUM_MAX_BITS", 20),
    osd_max_candidates=_get_int_env("GRAND_OSD_MAX_CANDIDATES", 160),
    osd_disagreement_extra_bits=_get_int_env("GRAND_OSD_DISAGREEMENT_BITS", 10),
    restart_max_candidates=_get_int_env("GRAND_OSD_RESTART_MAX_CANDIDATES", 24),
    restart_ldpc_iters=_get_int_env("GRAND_OSD_RESTART_ITERS", 16),
    restart_alpha=_get_float_env("GRAND_OSD_RESTART_ALPHA", 0.78),
    restart_llr_gain=_get_float_env("GRAND_OSD_RESTART_GAIN", 4.5),
    restart_llr_abs_floor=_get_float_env("GRAND_OSD_RESTART_ABS_FLOOR", 6.0),
    restart_dual_gain=_get_float_env("GRAND_OSD_RESTART_DUAL_GAIN", 6.5),
    restart_anchor_all_selected=_get_int_env("GRAND_OSD_RESTART_ANCHOR_ALL", 0),
)

grand_cfg_awgn_osd_boost = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_OSD_BOOST_MAX_WEIGHT", _get_int_env("GRAND_CTG_BOOST_MAX_WEIGHT", _get_int_env("GRAND_BOOST_MAX_WEIGHT", 5))),
    max_patterns=_get_int_env("GRAND_OSD_BOOST_MAX_PATTERNS", max(_get_int_env("GRAND_CTG_BOOST_MAX_PATTERNS", _get_int_env("GRAND_BOOST_MAX_PATTERNS", 15000)), 500000)),
    max_bits_from_cluster=None,
    verbose=False,
    llr_source=os.environ.get(
        "GRAND_OSD_LLR_SOURCE",
        os.environ.get("GRAND_CTG_LLR_SOURCE", os.environ.get("GRAND_LLR_SOURCE", "mixed")),
    ).strip().lower(),
    pattern_overgen_ratio=_get_float_env(
        "GRAND_OSD_BOOST_OVERGEN",
        _get_float_env("GRAND_CTG_BOOST_OVERGEN", _get_float_env("GRAND_BOOST_OVERGEN", 1.02)),
    ),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_OSD_BATCH_SIZE", _get_int_env("GRAND_CTG_BATCH_SIZE", _get_int_env("GRAND_BATCH_SIZE", 256))),
    selection_mode="syndrome_vote",
    sv_epsilon=_get_float_env("GRAND_OSD_EPSILON", _get_float_env("GRAND_CTG_EPSILON", _get_float_env("GRAND_SV_EPSILON", 1e-3))),
    sv_check_cover_k=_get_int_env("GRAND_OSD_CHECK_COVER_K", _get_int_env("GRAND_CTG_CHECK_COVER_K", _get_int_env("GRAND_SV_CHECK_COVER_K", 2))),
    pre_solver_mode="none",  # do not repeat OSD / peel on the boost path
    peel_candidate_ratio=_get_float_env("GRAND_OSD_PEEL_RATIO", _get_float_env("GRAND_CTG_PEEL_RATIO", 2.0)),
    peel_max_bits=_get_int_env("GRAND_OSD_PEEL_MAX_BITS", _get_int_env("GRAND_CTG_PEEL_MAX_BITS", 64)),
    peel_dense_max_vars=_get_int_env("GRAND_OSD_PEEL_DENSE_MAX_VARS", _get_int_env("GRAND_CTG_PEEL_DENSE_MAX_VARS", 32)),
    peel_max_free_enum=_get_int_env("GRAND_OSD_PEEL_MAX_FREE_ENUM", _get_int_env("GRAND_CTG_PEEL_MAX_FREE_ENUM", 12)),
    peel_extra_llr_bits=_get_int_env("GRAND_OSD_PEEL_EXTRA_LLR_BITS", _get_int_env("GRAND_CTG_PEEL_EXTRA_LLR_BITS", 10)),
    osd_candidate_ratio=_get_float_env("GRAND_OSD_RATIO", 2.9),
    osd_max_bits=_get_int_env("GRAND_OSD_MAX_BITS", 80),
    osd_order=_get_int_env("GRAND_OSD_ORDER", 2),
    osd_enum_max_bits=_get_int_env("GRAND_OSD_ENUM_MAX_BITS", 20),
    osd_max_candidates=_get_int_env("GRAND_OSD_MAX_CANDIDATES", 160),
    osd_disagreement_extra_bits=_get_int_env("GRAND_OSD_DISAGREEMENT_BITS", 10),
    restart_max_candidates=_get_int_env("GRAND_OSD_RESTART_MAX_CANDIDATES", 24),
    restart_ldpc_iters=_get_int_env("GRAND_OSD_RESTART_ITERS", 16),
    restart_alpha=_get_float_env("GRAND_OSD_RESTART_ALPHA", 0.78),
    restart_llr_gain=_get_float_env("GRAND_OSD_RESTART_GAIN", 4.5),
    restart_llr_abs_floor=_get_float_env("GRAND_OSD_RESTART_ABS_FLOOR", 6.0),
    restart_dual_gain=_get_float_env("GRAND_OSD_RESTART_DUAL_GAIN", 6.5),
    restart_anchor_all_selected=_get_int_env("GRAND_OSD_RESTART_ANCHOR_ALL", 0),
)

# ---- Receiver 6: soft local hypotheses + anchored restarts + peel + GRAND fallback ----
RUN_RECEIVER6 = bool(_get_int_env("RUN_RECEIVER6", 0))
GRAND_AHR_USE_BOOST = bool(_get_int_env("GRAND_AHR_USE_BOOST", _get_int_env("GRAND_OSD_USE_BOOST", _get_int_env("GRAND_USE_BOOST", 1))))

grand_cfg_awgn_ahr = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_AHR_MAX_WEIGHT", _get_int_env("GRAND_OSD_MAX_WEIGHT", _get_int_env("GRAND_MAX_WEIGHT", 5))),
    max_patterns=_get_int_env("GRAND_AHR_MAX_PATTERNS", max(_get_int_env("GRAND_OSD_MAX_PATTERNS", _get_int_env("GRAND_MAX_PATTERNS", 5000)), 100000)),
    max_bits_from_cluster=None,
    verbose=False,
    llr_source=os.environ.get(
        "GRAND_AHR_LLR_SOURCE",
        os.environ.get("GRAND_OSD_LLR_SOURCE", os.environ.get("GRAND_LLR_SOURCE", "mixed")),
    ).strip().lower(),
    pattern_overgen_ratio=_get_float_env(
        "GRAND_AHR_OVERGEN",
        _get_float_env("GRAND_OSD_OVERGEN", _get_float_env("GRAND_OVERGEN", 1.02)),
    ),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_AHR_BATCH_SIZE", _get_int_env("GRAND_OSD_BATCH_SIZE", _get_int_env("GRAND_BATCH_SIZE", 256))),
    selection_mode="syndrome_vote",
    sv_epsilon=_get_float_env("GRAND_AHR_EPSILON", _get_float_env("GRAND_OSD_EPSILON", _get_float_env("GRAND_SV_EPSILON", 1e-3))),
    sv_check_cover_k=_get_int_env("GRAND_AHR_CHECK_COVER_K", _get_int_env("GRAND_OSD_CHECK_COVER_K", _get_int_env("GRAND_SV_CHECK_COVER_K", 2))),
    pre_solver_mode="soft_anchor",
    peel_candidate_ratio=_get_float_env("GRAND_AHR_PEEL_RATIO", _get_float_env("GRAND_OSD_PEEL_RATIO", 2.0)),
    peel_max_bits=_get_int_env("GRAND_AHR_PEEL_MAX_BITS", _get_int_env("GRAND_OSD_PEEL_MAX_BITS", 72)),
    peel_dense_max_vars=_get_int_env("GRAND_AHR_PEEL_DENSE_MAX_VARS", _get_int_env("GRAND_OSD_PEEL_DENSE_MAX_VARS", 32)),
    peel_max_free_enum=_get_int_env("GRAND_AHR_PEEL_MAX_FREE_ENUM", _get_int_env("GRAND_OSD_PEEL_MAX_FREE_ENUM", 12)),
    peel_extra_llr_bits=_get_int_env("GRAND_AHR_PEEL_EXTRA_LLR_BITS", _get_int_env("GRAND_OSD_PEEL_EXTRA_LLR_BITS", 12)),
    osd_disagreement_extra_bits=_get_int_env("GRAND_AHR_DISAGREEMENT_BITS", _get_int_env("GRAND_OSD_DISAGREEMENT_BITS", 12)),
    restart_max_candidates=_get_int_env("GRAND_AHR_RESTART_MAX_CANDIDATES", 24),
    restart_ldpc_iters=_get_int_env("GRAND_AHR_RESTART_ITERS", 18),
    restart_alpha=_get_float_env("GRAND_AHR_RESTART_ALPHA", 0.78),
    restart_llr_gain=_get_float_env("GRAND_AHR_RESTART_GAIN", 4.8),
    restart_llr_abs_floor=_get_float_env("GRAND_AHR_RESTART_ABS_FLOOR", 6.5),
    restart_dual_gain=_get_float_env("GRAND_AHR_RESTART_DUAL_GAIN", 7.0),
    restart_anchor_all_selected=_get_int_env("GRAND_AHR_RESTART_ANCHOR_ALL", 0),
    soft_candidate_ratio=_get_float_env("GRAND_AHR_RATIO", 3.2),
    soft_max_bits=_get_int_env("GRAND_AHR_MAX_BITS", 96),
    soft_core_max_bits=_get_int_env("GRAND_AHR_CORE_MAX_BITS", 14),
    soft_max_weight=_get_int_env("GRAND_AHR_CORE_MAX_WEIGHT", 3),
    soft_max_candidates=_get_int_env("GRAND_AHR_MAX_CANDIDATES", 128),
    soft_sat_penalty=_get_float_env("GRAND_AHR_SAT_PENALTY", 0.35),
    soft_llr_weight=_get_float_env("GRAND_AHR_LLR_WEIGHT", 0.10),
)

grand_cfg_awgn_ahr_boost = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_AHR_BOOST_MAX_WEIGHT", _get_int_env("GRAND_OSD_BOOST_MAX_WEIGHT", _get_int_env("GRAND_BOOST_MAX_WEIGHT", 5))),
    max_patterns=_get_int_env("GRAND_AHR_BOOST_MAX_PATTERNS", max(_get_int_env("GRAND_OSD_BOOST_MAX_PATTERNS", _get_int_env("GRAND_BOOST_MAX_PATTERNS", 15000)), 600000)),
    max_bits_from_cluster=None,
    verbose=False,
    llr_source=os.environ.get(
        "GRAND_AHR_LLR_SOURCE",
        os.environ.get("GRAND_OSD_LLR_SOURCE", os.environ.get("GRAND_LLR_SOURCE", "mixed")),
    ).strip().lower(),
    pattern_overgen_ratio=_get_float_env(
        "GRAND_AHR_BOOST_OVERGEN",
        _get_float_env("GRAND_OSD_BOOST_OVERGEN", _get_float_env("GRAND_BOOST_OVERGEN", 1.02)),
    ),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_AHR_BATCH_SIZE", _get_int_env("GRAND_OSD_BATCH_SIZE", _get_int_env("GRAND_BATCH_SIZE", 256))),
    selection_mode="syndrome_vote",
    sv_epsilon=_get_float_env("GRAND_AHR_EPSILON", _get_float_env("GRAND_OSD_EPSILON", _get_float_env("GRAND_SV_EPSILON", 1e-3))),
    sv_check_cover_k=_get_int_env("GRAND_AHR_CHECK_COVER_K", _get_int_env("GRAND_OSD_CHECK_COVER_K", _get_int_env("GRAND_SV_CHECK_COVER_K", 2))),
    pre_solver_mode="none",  # do not repeat soft hypotheses / peel on the boost path
    peel_candidate_ratio=_get_float_env("GRAND_AHR_PEEL_RATIO", _get_float_env("GRAND_OSD_PEEL_RATIO", 2.0)),
    peel_max_bits=_get_int_env("GRAND_AHR_PEEL_MAX_BITS", _get_int_env("GRAND_OSD_PEEL_MAX_BITS", 72)),
    peel_dense_max_vars=_get_int_env("GRAND_AHR_PEEL_DENSE_MAX_VARS", _get_int_env("GRAND_OSD_PEEL_DENSE_MAX_VARS", 32)),
    peel_max_free_enum=_get_int_env("GRAND_AHR_PEEL_MAX_FREE_ENUM", _get_int_env("GRAND_OSD_PEEL_MAX_FREE_ENUM", 12)),
    peel_extra_llr_bits=_get_int_env("GRAND_AHR_PEEL_EXTRA_LLR_BITS", _get_int_env("GRAND_OSD_PEEL_EXTRA_LLR_BITS", 12)),
    osd_disagreement_extra_bits=_get_int_env("GRAND_AHR_DISAGREEMENT_BITS", _get_int_env("GRAND_OSD_DISAGREEMENT_BITS", 12)),
    restart_max_candidates=_get_int_env("GRAND_AHR_RESTART_MAX_CANDIDATES", 24),
    restart_ldpc_iters=_get_int_env("GRAND_AHR_RESTART_ITERS", 18),
    restart_alpha=_get_float_env("GRAND_AHR_RESTART_ALPHA", 0.78),
    restart_llr_gain=_get_float_env("GRAND_AHR_RESTART_GAIN", 4.8),
    restart_llr_abs_floor=_get_float_env("GRAND_AHR_RESTART_ABS_FLOOR", 6.5),
    restart_dual_gain=_get_float_env("GRAND_AHR_RESTART_DUAL_GAIN", 7.0),
    restart_anchor_all_selected=_get_int_env("GRAND_AHR_RESTART_ANCHOR_ALL", 0),
    soft_candidate_ratio=_get_float_env("GRAND_AHR_RATIO", 3.2),
    soft_max_bits=_get_int_env("GRAND_AHR_MAX_BITS", 96),
    soft_core_max_bits=_get_int_env("GRAND_AHR_CORE_MAX_BITS", 14),
    soft_max_weight=_get_int_env("GRAND_AHR_CORE_MAX_WEIGHT", 3),
    soft_max_candidates=_get_int_env("GRAND_AHR_MAX_CANDIDATES", 128),
    soft_sat_penalty=_get_float_env("GRAND_AHR_SAT_PENALTY", 0.35),
    soft_llr_weight=_get_float_env("GRAND_AHR_LLR_WEIGHT", 0.10),
)


# ---- Receiver 7: basis-GRAND + block-debias anchored restarts + peel + GRAND fallback ----
RUN_RECEIVER7 = bool(_get_int_env("RUN_RECEIVER7", 0))
GRAND_BGR_USE_BOOST = bool(_get_int_env("GRAND_BGR_USE_BOOST", _get_int_env("GRAND_AHR_USE_BOOST", _get_int_env("GRAND_USE_BOOST", 1))))

grand_cfg_awgn_bgr = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_BGR_MAX_WEIGHT", _get_int_env("GRAND_AHR_MAX_WEIGHT", _get_int_env("GRAND_MAX_WEIGHT", 5))),
    max_patterns=_get_int_env("GRAND_BGR_MAX_PATTERNS", max(_get_int_env("GRAND_AHR_MAX_PATTERNS", _get_int_env("GRAND_MAX_PATTERNS", 5000)), 80000)),
    max_bits_from_cluster=None,
    verbose=False,
    llr_source=os.environ.get(
        "GRAND_BGR_LLR_SOURCE",
        os.environ.get("GRAND_AHR_LLR_SOURCE", os.environ.get("GRAND_LLR_SOURCE", "mixed")),
    ).strip().lower(),
    pattern_overgen_ratio=_get_float_env(
        "GRAND_BGR_OVERGEN",
        _get_float_env("GRAND_AHR_OVERGEN", _get_float_env("GRAND_OVERGEN", 1.02)),
    ),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_BGR_BATCH_SIZE", _get_int_env("GRAND_AHR_BATCH_SIZE", _get_int_env("GRAND_BATCH_SIZE", 256))),
    selection_mode="syndrome_vote",
    sv_epsilon=_get_float_env("GRAND_BGR_EPSILON", _get_float_env("GRAND_AHR_EPSILON", _get_float_env("GRAND_SV_EPSILON", 1e-3))),
    sv_check_cover_k=_get_int_env("GRAND_BGR_CHECK_COVER_K", _get_int_env("GRAND_AHR_CHECK_COVER_K", _get_int_env("GRAND_SV_CHECK_COVER_K", 2))),
    pre_solver_mode="basis_anchor",
    peel_candidate_ratio=_get_float_env("GRAND_BGR_PEEL_RATIO", _get_float_env("GRAND_AHR_PEEL_RATIO", 2.0)),
    peel_max_bits=_get_int_env("GRAND_BGR_PEEL_MAX_BITS", _get_int_env("GRAND_AHR_PEEL_MAX_BITS", 72)),
    peel_dense_max_vars=_get_int_env("GRAND_BGR_PEEL_DENSE_MAX_VARS", _get_int_env("GRAND_AHR_PEEL_DENSE_MAX_VARS", 32)),
    peel_max_free_enum=_get_int_env("GRAND_BGR_PEEL_MAX_FREE_ENUM", _get_int_env("GRAND_AHR_PEEL_MAX_FREE_ENUM", 12)),
    peel_extra_llr_bits=_get_int_env("GRAND_BGR_PEEL_EXTRA_LLR_BITS", _get_int_env("GRAND_AHR_PEEL_EXTRA_LLR_BITS", 12)),
    osd_disagreement_extra_bits=_get_int_env("GRAND_BGR_DISAGREEMENT_BITS", _get_int_env("GRAND_AHR_DISAGREEMENT_BITS", 12)),
    restart_max_candidates=_get_int_env("GRAND_BGR_RESTART_MAX_CANDIDATES", 18),
    restart_ldpc_iters=_get_int_env("GRAND_BGR_RESTART_ITERS", 18),
    restart_alpha=_get_float_env("GRAND_BGR_RESTART_ALPHA", 0.78),
    restart_llr_gain=_get_float_env("GRAND_BGR_RESTART_GAIN", 5.2),
    restart_llr_abs_floor=_get_float_env("GRAND_BGR_RESTART_ABS_FLOOR", 6.0),
    restart_dual_gain=_get_float_env("GRAND_BGR_RESTART_DUAL_GAIN", 7.4),
    restart_anchor_all_selected=0,
    basis_candidate_ratio=_get_float_env("GRAND_BGR_RATIO", 3.0),
    basis_max_bits=_get_int_env("GRAND_BGR_MAX_BITS", 96),
    basis_max_vectors=_get_int_env("GRAND_BGR_MAX_VECTORS", 18),
    basis_core_vectors=_get_int_env("GRAND_BGR_CORE_VECTORS", 12),
    basis_combo_max=_get_int_env("GRAND_BGR_COMBO_MAX", 3),
    basis_max_candidates=_get_int_env("GRAND_BGR_MAX_CANDIDATES", 128),
    basis_group_max_bits=_get_int_env("GRAND_BGR_GROUP_MAX_BITS", 10),
    basis_window_max=_get_int_env("GRAND_BGR_WINDOW_MAX", 6),
    basis_window_span=_get_int_env("GRAND_BGR_WINDOW_SPAN", 18),
    basis_disagreement_groups=_get_int_env("GRAND_BGR_DISAGREE_GROUPS", 4),
    basis_disagreement_chunk=_get_int_env("GRAND_BGR_DISAGREE_CHUNK", 6),
    basis_top_singletons=_get_int_env("GRAND_BGR_TOP_SINGLETONS", 8),
    debias_blend=_get_float_env("GRAND_BGR_DEBIAS_BLEND", 0.65),
    debias_relax=_get_float_env("GRAND_BGR_DEBIAS_RELAX", 0.45),
)

grand_cfg_awgn_bgr_boost = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_BGR_BOOST_MAX_WEIGHT", _get_int_env("GRAND_AHR_BOOST_MAX_WEIGHT", _get_int_env("GRAND_BOOST_MAX_WEIGHT", 5))),
    max_patterns=_get_int_env("GRAND_BGR_BOOST_MAX_PATTERNS", max(_get_int_env("GRAND_AHR_BOOST_MAX_PATTERNS", _get_int_env("GRAND_BOOST_MAX_PATTERNS", 15000)), 250000)),
    max_bits_from_cluster=None,
    verbose=False,
    llr_source=os.environ.get(
        "GRAND_BGR_LLR_SOURCE",
        os.environ.get("GRAND_AHR_LLR_SOURCE", os.environ.get("GRAND_LLR_SOURCE", "mixed")),
    ).strip().lower(),
    pattern_overgen_ratio=_get_float_env(
        "GRAND_BGR_BOOST_OVERGEN",
        _get_float_env("GRAND_AHR_BOOST_OVERGEN", _get_float_env("GRAND_BOOST_OVERGEN", 1.02)),
    ),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_BGR_BATCH_SIZE", _get_int_env("GRAND_AHR_BATCH_SIZE", _get_int_env("GRAND_BATCH_SIZE", 256))),
    selection_mode="syndrome_vote",
    sv_epsilon=_get_float_env("GRAND_BGR_EPSILON", _get_float_env("GRAND_AHR_EPSILON", _get_float_env("GRAND_SV_EPSILON", 1e-3))),
    sv_check_cover_k=_get_int_env("GRAND_BGR_CHECK_COVER_K", _get_int_env("GRAND_AHR_CHECK_COVER_K", _get_int_env("GRAND_SV_CHECK_COVER_K", 2))),
    pre_solver_mode="none",
    peel_candidate_ratio=_get_float_env("GRAND_BGR_PEEL_RATIO", _get_float_env("GRAND_AHR_PEEL_RATIO", 2.0)),
    peel_max_bits=_get_int_env("GRAND_BGR_PEEL_MAX_BITS", _get_int_env("GRAND_AHR_PEEL_MAX_BITS", 72)),
    peel_dense_max_vars=_get_int_env("GRAND_BGR_PEEL_DENSE_MAX_VARS", _get_int_env("GRAND_AHR_PEEL_DENSE_MAX_VARS", 32)),
    peel_max_free_enum=_get_int_env("GRAND_BGR_PEEL_MAX_FREE_ENUM", _get_int_env("GRAND_AHR_PEEL_MAX_FREE_ENUM", 12)),
    peel_extra_llr_bits=_get_int_env("GRAND_BGR_PEEL_EXTRA_LLR_BITS", _get_int_env("GRAND_AHR_PEEL_EXTRA_LLR_BITS", 12)),
    osd_disagreement_extra_bits=_get_int_env("GRAND_BGR_DISAGREEMENT_BITS", _get_int_env("GRAND_AHR_DISAGREEMENT_BITS", 12)),
    restart_max_candidates=_get_int_env("GRAND_BGR_RESTART_MAX_CANDIDATES", 18),
    restart_ldpc_iters=_get_int_env("GRAND_BGR_RESTART_ITERS", 18),
    restart_alpha=_get_float_env("GRAND_BGR_RESTART_ALPHA", 0.78),
    restart_llr_gain=_get_float_env("GRAND_BGR_RESTART_GAIN", 5.2),
    restart_llr_abs_floor=_get_float_env("GRAND_BGR_RESTART_ABS_FLOOR", 6.0),
    restart_dual_gain=_get_float_env("GRAND_BGR_RESTART_DUAL_GAIN", 7.4),
    restart_anchor_all_selected=0,
    basis_candidate_ratio=_get_float_env("GRAND_BGR_RATIO", 3.0),
    basis_max_bits=_get_int_env("GRAND_BGR_MAX_BITS", 96),
    basis_max_vectors=_get_int_env("GRAND_BGR_MAX_VECTORS", 18),
    basis_core_vectors=_get_int_env("GRAND_BGR_CORE_VECTORS", 12),
    basis_combo_max=_get_int_env("GRAND_BGR_COMBO_MAX", 3),
    basis_max_candidates=_get_int_env("GRAND_BGR_MAX_CANDIDATES", 128),
    basis_group_max_bits=_get_int_env("GRAND_BGR_GROUP_MAX_BITS", 10),
    basis_window_max=_get_int_env("GRAND_BGR_WINDOW_MAX", 6),
    basis_window_span=_get_int_env("GRAND_BGR_WINDOW_SPAN", 18),
    basis_disagreement_groups=_get_int_env("GRAND_BGR_DISAGREE_GROUPS", 4),
    basis_disagreement_chunk=_get_int_env("GRAND_BGR_DISAGREE_CHUNK", 6),
    basis_top_singletons=_get_int_env("GRAND_BGR_TOP_SINGLETONS", 8),
    debias_blend=_get_float_env("GRAND_BGR_DEBIAS_BLEND", 0.65),
    debias_relax=_get_float_env("GRAND_BGR_DEBIAS_RELAX", 0.45),
)
# ======================================================================






# ======================================================================







# ---- Receiver 8: cascade AHR -> basis-GRAND fallback on the same LDPC snapshots ----
RUN_RECEIVER8 = bool(_get_int_env("RUN_RECEIVER8", 0))
GRAND_META_USE_FALLBACK = bool(_get_int_env("GRAND_META_USE_FALLBACK", 1))

# Stronger primary profile built from Receiver-6 because the committed localbias results
# show the anchored-restart / soft-hypothesis path is consistently better than Receiver-7.
grand_cfg_awgn_meta = copy.deepcopy(grand_cfg_awgn_ahr)
grand_cfg_awgn_meta.max_patterns = _get_int_env("GRAND_META_MAX_PATTERNS", max(int(getattr(grand_cfg_awgn_ahr, "max_patterns", 0) or 0), 140000))
grand_cfg_awgn_meta.restart_max_candidates = _get_int_env("GRAND_META_RESTART_MAX_CANDIDATES", max(int(getattr(grand_cfg_awgn_ahr, "restart_max_candidates", 0) or 0), 28))
grand_cfg_awgn_meta.restart_ldpc_iters = _get_int_env("GRAND_META_RESTART_ITERS", max(int(getattr(grand_cfg_awgn_ahr, "restart_ldpc_iters", 0) or 0), 22))
grand_cfg_awgn_meta.restart_llr_gain = _get_float_env("GRAND_META_RESTART_GAIN", max(float(getattr(grand_cfg_awgn_ahr, "restart_llr_gain", 0.0) or 0.0), 5.4))
grand_cfg_awgn_meta.restart_dual_gain = _get_float_env("GRAND_META_RESTART_DUAL_GAIN", max(float(getattr(grand_cfg_awgn_ahr, "restart_dual_gain", 0.0) or 0.0), 7.8))
grand_cfg_awgn_meta.soft_candidate_ratio = _get_float_env("GRAND_META_RATIO", max(float(getattr(grand_cfg_awgn_ahr, "soft_candidate_ratio", 0.0) or 0.0), 3.6))
grand_cfg_awgn_meta.soft_max_candidates = _get_int_env("GRAND_META_MAX_CANDIDATES", max(int(getattr(grand_cfg_awgn_ahr, "soft_max_candidates", 0) or 0), 160))
grand_cfg_awgn_meta.soft_core_max_bits = _get_int_env("GRAND_META_CORE_MAX_BITS", max(int(getattr(grand_cfg_awgn_ahr, "soft_core_max_bits", 0) or 0), 16))
grand_cfg_awgn_meta.peel_candidate_ratio = _get_float_env("GRAND_META_PEEL_RATIO", max(float(getattr(grand_cfg_awgn_ahr, "peel_candidate_ratio", 0.0) or 0.0), 2.3))

grand_cfg_awgn_meta_boost = copy.deepcopy(grand_cfg_awgn_ahr_boost)
grand_cfg_awgn_meta_boost.max_patterns = _get_int_env("GRAND_META_BOOST_MAX_PATTERNS", max(int(getattr(grand_cfg_awgn_ahr_boost, "max_patterns", 0) or 0), 800000))
grand_cfg_awgn_meta_boost.restart_max_candidates = _get_int_env("GRAND_META_RESTART_MAX_CANDIDATES", max(int(getattr(grand_cfg_awgn_ahr_boost, "restart_max_candidates", 0) or 0), 28))
grand_cfg_awgn_meta_boost.restart_ldpc_iters = _get_int_env("GRAND_META_RESTART_ITERS", max(int(getattr(grand_cfg_awgn_ahr_boost, "restart_ldpc_iters", 0) or 0), 22))
grand_cfg_awgn_meta_boost.restart_llr_gain = _get_float_env("GRAND_META_RESTART_GAIN", max(float(getattr(grand_cfg_awgn_ahr_boost, "restart_llr_gain", 0.0) or 0.0), 5.4))
grand_cfg_awgn_meta_boost.restart_dual_gain = _get_float_env("GRAND_META_RESTART_DUAL_GAIN", max(float(getattr(grand_cfg_awgn_ahr_boost, "restart_dual_gain", 0.0) or 0.0), 7.8))
grand_cfg_awgn_meta_boost.soft_candidate_ratio = _get_float_env("GRAND_META_RATIO", max(float(getattr(grand_cfg_awgn_ahr_boost, "soft_candidate_ratio", 0.0) or 0.0), 3.6))
grand_cfg_awgn_meta_boost.soft_max_candidates = _get_int_env("GRAND_META_MAX_CANDIDATES", max(int(getattr(grand_cfg_awgn_ahr_boost, "soft_max_candidates", 0) or 0), 160))
grand_cfg_awgn_meta_boost.soft_core_max_bits = _get_int_env("GRAND_META_CORE_MAX_BITS", max(int(getattr(grand_cfg_awgn_ahr_boost, "soft_core_max_bits", 0) or 0), 16))
### CELL number 28-B ###
import os
import math
from dataclasses import dataclass, asdict
from typing import Sequence

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)

@dataclass
class HardwareTimingModel:
    # Global clock (MHz)
    fclk_mhz: int = 800

    # LDPC throughputs
    ldpc_edges_per_cycle: int = 64
    ldpc_bits_per_cycle_hd: int = 64
    ldpc_iter_overhead_cycles: int = 20

    # GRAND throughputs (membership test)
    grand_edges_per_cycle: int = 64
    grand_checks_per_cycle: int = 64
    grand_batch_overhead_cycles: int = 200

    # GRAND front-end (LLR sorting + pattern formation)
    grand_abs_per_cycle: int = 64
    grand_sort_elem_log2_per_cycle: int = 64
    grand_cost_add_per_cycle: int = 64
    grand_patsort_patlog2_per_cycle: int = 64

    # Optional: cluster clique-building proxy
    cluster_pair_edges_per_cycle: int = 0

def load_hw_model_from_env() -> HardwareTimingModel:
    return HardwareTimingModel(
        fclk_mhz=_env_int("HW_FCLK_MHZ", 800),

        ldpc_edges_per_cycle=_env_int("HW_LDPC_EDGES_PER_CYCLE", 64),
        ldpc_bits_per_cycle_hd=_env_int("HW_LDPC_BITS_PER_CYCLE_HD", 64),
        ldpc_iter_overhead_cycles=_env_int("HW_LDPC_ITER_OVERHEAD_CYCLES", 20),

        grand_edges_per_cycle=_env_int("HW_GRAND_EDGES_PER_CYCLE", 64),
        grand_checks_per_cycle=_env_int("HW_GRAND_CHECKS_PER_CYCLE", 64),
        grand_batch_overhead_cycles=_env_int("HW_GRAND_BATCH_OVERHEAD_CYCLES", 200),

        grand_abs_per_cycle=_env_int("HW_GRAND_ABS_PER_CYCLE", 64),
        grand_sort_elem_log2_per_cycle=_env_int("HW_GRAND_SORT_ELEMLOG2_PER_CYCLE", 64),
        grand_cost_add_per_cycle=_env_int("HW_GRAND_COST_ADD_PER_CYCLE", 64),
        grand_patsort_patlog2_per_cycle=_env_int("HW_GRAND_PATSORT_PATLOG2_PER_CYCLE", 64),

        cluster_pair_edges_per_cycle=_env_int("HW_CLUSTER_PAIR_EDGES_PER_CYCLE", 0),
    )

def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("Throughput parameter must be > 0")
    if a <= 0:
        return 0
    return int((a + b - 1) // b)

def _safe_log2_int(n: int) -> int:
    """Return ceil(log2(n)) for n>=2, else 0."""
    if n <= 1:
        return 0
    return int(math.ceil(math.log2(float(n))))

def cycles_to_us(cycles: int, hw: HardwareTimingModel) -> float:
    """Convert cycles to microseconds (fclk_mhz cycles per microsecond)."""
    return float(cycles) / float(hw.fclk_mhz)

def _ldpc_total_edges(code_cfg: CodeConfig) -> int:
    """Total Tanner-graph edges E = |{(check,var): H[check,var]=1}|."""
    if hasattr(code_cfg, "_c2v_ptrs"):
        return int(code_cfg._c2v_ptrs[int(code_cfg.M)])
    # Fallback (slower)
    return int(sum(len(cv) for cv in code_cfg.checks_to_vars))

def ldpc_hw_cycles_frame(
    iter_used: int,
    code_cfg: CodeConfig,
    hw: HardwareTimingModel,
    final_vn2cn_executed: bool = False,
) -> int:
    """
    Cycle model for ONE LDPC decoding invocation on ONE frame.

    IMPORTANT:
      - This is an ALGORITHMIC hardware-intent model,
        NOT a model of Python overhead.

    Counted operations:
      (A) Message initialization (VN->CN): write E messages.
      (B) For each executed iteration:
            - check node update        : process E edges
            - variable node update     : process E edges
            - hard decision            : process N bits
            - syndrome computation     : process E edges
            - iteration control overhead (constant)
      (C) VN->CN update pass:
            - for iterations 1..(iter_used-1): ALWAYS executed
            - on the LAST iteration: executed only if the software actually ran it,
              controlled by final_vn2cn_executed.

    This makes the HW-cycle accounting match the *actual control flow* of the
    current software implementation:
      - if early-stop triggers, the last VN->CN update is skipped
      - if early-stop does NOT trigger (e.g. decode fails at max_iters),
        the last VN->CN update *is* executed by the current code.
    """
    it = int(iter_used) if iter_used is not None else 0
    if it <= 0:
        return 0

    N = int(code_cfg.N)
    E = _ldpc_total_edges(code_cfg)

    c_edge = _ceil_div(E, int(hw.ldpc_edges_per_cycle))
    c_hd = _ceil_div(N, int(hw.ldpc_bits_per_cycle_hd))

    # Init: VN->CN message init across all edges
    c_init = c_edge

    # Iteration core (excluding VN->CN)
    c_iter_core = (
        c_edge               # CN update
        + c_edge             # VN update
        + c_hd               # hard decision
        + c_edge             # syndrome compute
        + int(hw.ldpc_iter_overhead_cycles)
    )

    # VN->CN:
    # - Always for the first (it-1) iterations
    # - Plus the last one iff the software executed it (final_vn2cn_executed)
    c_vn2cn = c_edge * max(0, it - 1)
    if bool(final_vn2cn_executed):
        c_vn2cn += c_edge

    return int(c_init + it * c_iter_core + c_vn2cn)

def grand_hw_cycles_from_result(
    result: ClusterGrandResult,
    sim_cfg: SimulationConfig,
    hw: HardwareTimingModel
) -> int:
    """
    Cycle model for ONE GRAND (union-of-clusters) decoding invocation.

    IMPORTANT:
      - Uses the "evaluated" counters, which represent the workload actually
        executed by the chunked/batch-parallel engine. This is the correct
        quantity for serial-parallel (chunked) hardware timing.
      - Front-end costs (LLR abs+sort, pattern cost computation, pattern sort)
        are ALSO charged using metadata recorded by run_local_grand_on_union_of_clusters().
    """
    code_cfg = sim_cfg.code
    M = int(code_cfg.M)

    # --- Evaluated counters (batch-true workload) ---
    edge_visits = int(getattr(result, "total_v2c_edge_visits_evaluated",
                              getattr(result, "total_v2c_edge_visits", 0)))
    checks_toggled = int(getattr(result, "total_unique_checks_toggled_evaluated",
                                 getattr(result, "total_unique_checks_toggled", 0)))
    num_batches = int(getattr(result, "num_batches_evaluated", 0))
    positions_packed = int(getattr(result, "positions_packed_evaluated", 0))

    # --- Front-end meta (always generated, even if early success) ---
    llr_sort_len = int(getattr(result, "llr_sort_len", 0))
    search_sz = int(getattr(result, "search_size", 0))
    patterns_gen = int(getattr(result, "patterns_generated", 0))
    sumw_gen = int(getattr(result, "sum_pattern_weights_generated", 0))
    selection_mode_used = str(getattr(result, "selection_mode_used", "llr") or "llr").strip().lower()
    sv_score_len = int(getattr(result, "sv_score_len", llr_sort_len))

    # --- Cluster extraction proxies ---
    cluster_unsat_edges = int(getattr(result, "cluster_unsat_edges", 0))
    cluster_pair_edges = int(getattr(result, "cluster_pair_edges", 0))

    # --- Throughputs ---
    epc = int(hw.grand_edges_per_cycle)
    cpc = int(hw.grand_checks_per_cycle)
    abs_pc = int(hw.grand_abs_per_cycle)
    sort_pc = int(hw.grand_sort_elem_log2_per_cycle)
    add_pc = int(hw.grand_cost_add_per_cycle)
    psort_pc = int(hw.grand_patsort_patlog2_per_cycle)

    cycles = 0

    # (0) Scan syndrome to identify unsatisfied checks (proxy = M checks)
    cycles += _ceil_div(M, cpc)

    # (1) Cluster extraction proxies
    cycles += _ceil_div(cluster_unsat_edges, epc)
    if int(hw.cluster_pair_edges_per_cycle) > 0:
        cycles += _ceil_div(cluster_pair_edges, int(hw.cluster_pair_edges_per_cycle))

    # (2) LLR abs computations:
    #     - abs for union bits (length llr_sort_len)
    #     - abs AGAIN for the truncated search set (length search_sz)
    cycles += _ceil_div(llr_sort_len, abs_pc)
    cycles += _ceil_div(search_sz, abs_pc)

    # (3) Sort union bits by the front-end key:
    #     - Receiver 1 : |LLR|
    #     - Receiver 2 : eta_v = u_v / (rho_v + epsilon)
    cycles += _ceil_div(llr_sort_len * _safe_log2_int(llr_sort_len), sort_pc)

    # Small extra arithmetic for advanced front-end score formation. The unsatisfied-check
    # neighbour scan itself is already covered by cluster_unsat_edges above.
    if selection_mode_used in ("syndrome_vote", "sv", "receiver2"):
        cycles += _ceil_div(sv_score_len, add_pc)
    elif selection_mode_used in ("ai_tanner_subgraph_roi", "aitg2", "tanner_subgraph_roi", "receiver9_tg2", "ai_tanner_roi", "aitg", "tanner_roi", "receiver9_tg", "ai_mix_roi", "aimix", "mix_roi", "receiver9_mix", "ai_window_roi", "aiwindow", "window_roi", "receiver9_window", "ai_rank_roi", "airoi", "roi_rank", "receiver9_roi", "ai_rank", "ai", "airank", "receiver9"):
        # Weighted blend of vote, inverse-|LLR|, disagreement, density, plus lightweight block scoring.
        cycles += _ceil_div(4 * sv_score_len, add_pc)

    # (4) Receiver-3-style pre-solver cost (if enabled)
    peel_candidate_size = int(getattr(result, "peel_candidate_size", 0))
    peel_edge_work = int(getattr(result, "peel_edge_work", 0))
    peel_dense_xor_ops = int(getattr(result, "peel_dense_xor_ops", 0))
    if peel_candidate_size > 0:
        cycles += _ceil_div(peel_candidate_size, abs_pc)   # extra reliability work on V_peel
    if peel_edge_work > 0:
        cycles += _ceil_div(peel_edge_work, epc)
    if peel_dense_xor_ops > 0:
        cycles += _ceil_div(peel_dense_xor_ops, add_pc)

    # (4b) Receiver-4-style Chase-list work
    chase_candidate_size = int(getattr(result, "chase_candidate_size", 0))
    chase_core_size = int(getattr(result, "chase_core_size", 0))
    chase_patterns_considered = int(getattr(result, "chase_patterns_considered", 0))
    chase_score_edge_visits = int(getattr(result, "chase_score_edge_visits", 0))
    chase_score_checks_toggled = int(getattr(result, "chase_score_checks_toggled", 0))
    chase_score_sumw = int(getattr(result, "chase_score_sum_pattern_weights", 0))
    chase_ldpc_num_runs = int(getattr(result, "chase_ldpc_num_runs", 0))
    chase_ldpc_total_iters = int(getattr(result, "chase_ldpc_total_iters", 0))
    chase_ldpc_num_nonconverged = int(getattr(result, "chase_ldpc_num_nonconverged", 0))

    if chase_candidate_size > 0:
        cycles += _ceil_div(chase_candidate_size, abs_pc)
    if chase_core_size > 0:
        cycles += _ceil_div(chase_core_size * _safe_log2_int(chase_core_size), sort_pc)
    if chase_patterns_considered > 0:
        cycles += _ceil_div(chase_score_sumw, add_pc)
        cycles += _ceil_div(chase_patterns_considered * _safe_log2_int(chase_patterns_considered), psort_pc)
    if chase_score_edge_visits > 0:
        cycles += _ceil_div(chase_score_edge_visits, epc)
    if chase_score_checks_toggled > 0:
        cycles += _ceil_div(chase_score_checks_toggled, cpc)
    if chase_ldpc_num_runs > 0 and chase_ldpc_total_iters > 0:
        E = _ldpc_total_edges(code_cfg)
        N = int(code_cfg.N)
        c_edge = _ceil_div(E, int(hw.ldpc_edges_per_cycle))
        c_hd = _ceil_div(N, int(hw.ldpc_bits_per_cycle_hd))
        c_init = c_edge
        c_iter_core = (c_edge + c_edge + c_hd + c_edge + int(hw.ldpc_iter_overhead_cycles))
        cycles += int(chase_ldpc_num_runs) * int(c_init)
        cycles += int(chase_ldpc_total_iters) * int(c_iter_core)
        cycles += int(c_edge) * max(0, int(chase_ldpc_total_iters) - int(chase_ldpc_num_runs) + int(chase_ldpc_num_nonconverged))

    # (4c) Receiver-5-style local OSD + anchored restart work
    osd_candidate_size = int(getattr(result, "osd_candidate_size", 0))
    osd_matrix_rows = int(getattr(result, "osd_matrix_rows", 0))
    osd_free_dim = int(getattr(result, "osd_free_dim", 0))
    osd_enum_bits_used = int(getattr(result, "osd_enum_bits_used", 0))
    osd_basis_xor_ops = int(getattr(result, "osd_basis_xor_ops", 0))
    osd_candidates_considered = int(getattr(result, "osd_candidates_considered", 0))
    osd_sum_candidate_weights = int(getattr(result, "osd_sum_candidate_weights", 0))
    restart_num_runs = int(getattr(result, "restart_num_runs", 0))
    restart_total_ldpc_iters = int(getattr(result, "restart_total_ldpc_iters", 0))
    restart_num_nonconverged = int(getattr(result, "restart_num_nonconverged", 0))
    restart_anchor_bits_total = int(getattr(result, "restart_anchor_bits_total", 0))

    if osd_candidate_size > 0:
        cycles += _ceil_div(osd_candidate_size, abs_pc)
        cycles += _ceil_div(osd_candidate_size * _safe_log2_int(osd_candidate_size), sort_pc)
    if osd_matrix_rows > 0 and osd_candidate_size > 0:
        cycles += _ceil_div(osd_matrix_rows * osd_candidate_size, cpc)
    if osd_basis_xor_ops > 0:
        cycles += _ceil_div(osd_basis_xor_ops, add_pc)
    if osd_candidates_considered > 0:
        cycles += _ceil_div(osd_sum_candidate_weights, add_pc)
        cycles += _ceil_div(osd_candidates_considered * _safe_log2_int(osd_candidates_considered), psort_pc)
    if restart_anchor_bits_total > 0:
        cycles += _ceil_div(restart_anchor_bits_total, abs_pc)
    if restart_num_runs > 0 and restart_total_ldpc_iters > 0:
        E = _ldpc_total_edges(code_cfg)
        N = int(code_cfg.N)
        c_edge = _ceil_div(E, int(hw.ldpc_edges_per_cycle))
        c_hd = _ceil_div(N, int(hw.ldpc_bits_per_cycle_hd))
        c_init = c_edge
        c_iter_core = (c_edge + c_edge + c_hd + c_edge + int(hw.ldpc_iter_overhead_cycles))
        cycles += int(restart_num_runs) * int(c_init)
        cycles += int(restart_total_ldpc_iters) * int(c_iter_core)
        cycles += int(c_edge) * max(0, int(restart_total_ldpc_iters) - int(restart_num_runs) + int(restart_num_nonconverged))

    # (5) Pattern cost computation: total number of |LLR| terms summed
    cycles += _ceil_div(sumw_gen, add_pc)

    # (6) Pattern ordering sort: O(P log2 P)
    cycles += _ceil_div(patterns_gen * _safe_log2_int(patterns_gen), psort_pc)

    # (6) Packing overhead proxy: sum of weights of evaluated patterns
    cycles += _ceil_div(positions_packed, add_pc)

    # (7) Membership test evaluation workload (evaluated counters)
    cycles += _ceil_div(edge_visits, epc)
    cycles += _ceil_div(checks_toggled, cpc)

    # (8) Chunk/batch overhead (serial-parallel engine granularity)
    cycles += int(num_batches) * int(hw.grand_batch_overhead_cycles)

    return int(cycles)

# A single global instance (each Loky process will have its own copy)
HW_MODEL = load_hw_model_from_env()
print("[HW model] ", asdict(HW_MODEL))


@dataclass
class AIGatedHybridConfig:
    """Lightweight AI controller for stage-2 rescue budgeting.

    Modes:
      - ``linear_ucb``: original compact contextual-bandit gate.
      - ``distilled_tree``: a tiny hardware-friendly policy tree.
      - ``distilled_tree_bandit``: distilled tree prior + LinUCB correction.
      - ``distilled_tree_roi``: distilled tree + forced exploration + empirical ROI governor.
      - ``probe_moe_roi``: always-on tiny probe + adaptive full-rescue escalation.

    The distilled-tree modes are meant to approximate a richer teacher policy
    with only a handful of comparisons, which is attractive for GRAND hardware
    control.
    """
    gate_snapshot_policy: str = "first"   # "first", "final", or integer-like token
    policy_mode: str = "distilled_tree_roi"   # preferred low-latency controller
    dynamic_per_snapshot: bool = True      # re-evaluate the gate at each saved LDPC snapshot
    tiny_snapshot_cap: int = 2
    full_snapshot_cap: int = 3
    meta_snapshot_cap: int = 4
    weak_llr_quantile: float = 0.30
    weak_llr_abs_cap: float = 2.50
    block_size: int = 64
    ucb_alpha: float = 0.18
    ridge: float = 1.00
    cost_lambda: float = 0.30
    cost_scale_cycles: float = 240000.0
    improvement_reward: float = 0.35
    true_fix_reward: float = 1.15
    skip_failure_penalty: float = 0.26
    tiny_cost_prior: float = 0.06
    full_cost_prior: float = 0.18
    meta_cost_prior: float = 0.28
    meta_min_compactness: float = 0.34
    meta_min_conflict: float = 0.08
    force_skip_diffuse: float = 0.95
    force_skip_promise: float = -0.22
    warmup_min_trials_per_action: int = 8
    warmup_snapshot_cap: int = 1
    suppress_skip_during_warmup: bool = True
    cold_start_bonus: float = 0.55
    roi_score_weight: float = 0.75
    roi_disable_min_trials: int = 6
    roi_disable_threshold: float = -0.03
    roi_disable_penalty: float = 0.40
    roi_promote_min_trials: int = 4
    roi_promote_threshold: float = 0.08
    roi_promote_weight: float = 0.30
    # Distilled-tree thresholds. These are seeded from the current committed
    # diagnostics, where diffuse wide residuals dominate the expensive failures.
    tree_diffuse_skip: float = 0.92
    tree_skip_union_size: int = 448
    tree_skip_promise: float = -0.12
    tree_skip_block_concentration: float = 0.06
    tree_tiny_compactness: float = 0.16
    tree_tiny_uncertainty: float = 0.20
    tree_tiny_block_concentration: float = 0.05
    tree_full_compactness: float = 0.24
    tree_full_uncertainty: float = 0.20
    tree_full_promise: float = -0.05
    tree_full_block_concentration: float = 0.07
    tree_full_conflict: float = 0.06
    tree_meta_compactness: float = 0.38
    tree_meta_conflict: float = 0.09
    tree_meta_promise: float = 0.02
    tree_meta_block_concentration: float = 0.10
    tree_tiny_rescue_compactness: float = 0.12
    tree_tiny_rescue_uncertainty: float = 0.22
    tree_promote_margin: float = 0.05
    tree_drop_margin: float = 0.06
    tree_late_snapshot_penalty: float = 0.18
    tree_late_stop_progress: float = 0.01
    tree_late_stop_compactness: float = 0.30
    tree_late_stop_block_concentration: float = 0.08
    # Probe-and-escalate controller (adaptive-compute MoE style).
    probe_min_syndrome_drop_full: float = 0.08
    probe_min_syndrome_drop_continue: float = 0.04
    probe_min_syndrome_drop_hard: float = 0.12
    probe_next_snapshot_min_drop: float = 0.03
    probe_local_compactness: float = 0.16
    probe_local_block_concentration: float = 0.07
    probe_local_conflict: float = 0.06
    probe_local_promise: float = -0.02
    probe_strong_compactness: float = 0.26
    probe_strong_block_concentration: float = 0.10
    probe_strong_conflict: float = 0.09
    probe_meta_drop: float = 0.16
    probe_failrate_easy: float = 0.018
    probe_failrate_hard: float = 0.080
    probe_failrate_min_frames: int = 192
    probe_force_tiny_all_failed: bool = True
    probe_late_snapshot_cap: int = 2


class AIGatedHybridState:
    def __init__(self, n_features: int, cfg: AIGatedHybridConfig):
        self.action_names = ["skip", "tiny", "full", "meta"]
        self._name_to_idx = {name: i for i, name in enumerate(self.action_names)}
        self.A = [np.eye(int(n_features), dtype=np.float64) * float(cfg.ridge) for _ in self.action_names]
        self.b = [np.zeros(int(n_features), dtype=np.float64) for _ in self.action_names]
        self.action_counts = np.zeros(len(self.action_names), dtype=np.int32)
        self.last_reward = np.zeros(len(self.action_names), dtype=np.float64)
        self.reward_sums = np.zeros(len(self.action_names), dtype=np.float64)
        self.cost_sums = np.zeros(len(self.action_names), dtype=np.float64)
        self.improve_counts = np.zeros(len(self.action_names), dtype=np.int32)
        self.fix_counts = np.zeros(len(self.action_names), dtype=np.int32)

    def idx(self, name: str) -> int:
        return int(self._name_to_idx[str(name)])

    def mean_reward(self, name: str) -> float:
        idx = self.idx(name)
        n = int(self.action_counts[idx])
        return float(self.reward_sums[idx] / n) if n > 0 else float("nan")

    def choose(self,
               x: np.ndarray,
               base_scores: Dict[str, float],
               allowed: Sequence[str],
               cfg: AIGatedHybridConfig,
               snapshot_pos: int = 1) -> Tuple[str, float, Dict[str, float]]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        allowed_set = set(str(a) for a in allowed)
        policy_mode = str(getattr(cfg, "policy_mode", "linear_ucb") or "linear_ucb").strip().lower()
        use_bandit = policy_mode in ("linear_ucb", "distilled_tree_bandit", "tree_bandit", "dt_bandit", "distilled_tree_roi", "tree_roi", "dt_roi")
        use_tree_only = policy_mode in ("distilled_tree", "tree", "dt")
        scores: Dict[str, float] = {}
        cost_prior = {
            "skip": 0.0,
            "tiny": float(cfg.tiny_cost_prior),
            "full": float(cfg.full_cost_prior),
            "meta": float(cfg.meta_cost_prior),
        }

        cold_candidates = []
        if int(snapshot_pos) <= max(1, int(getattr(cfg, "warmup_snapshot_cap", 1) or 1)):
            for name in ("tiny", "full", "meta"):
                if name in allowed_set and int(self.action_counts[self.idx(name)]) < max(0, int(getattr(cfg, "warmup_min_trials_per_action", 0) or 0)):
                    cold_candidates.append(name)

        if cold_candidates and bool(getattr(cfg, "suppress_skip_during_warmup", True)):
            allowed_set = set(cold_candidates)

        if cold_candidates:
            forced_name = max(
                cold_candidates,
                key=lambda n: (
                    float(base_scores.get(n, -1e9)),
                    -int(self.action_counts[self.idx(n)]),
                    -((0 if n == "tiny" else 1) if n == "full" else 2),
                ),
            )
            for name in self.action_names:
                scores[name] = -1e9
            scores[forced_name] = float(base_scores.get(forced_name, 0.0)) + float(getattr(cfg, "cold_start_bonus", 0.55) or 0.55)
            return str(forced_name), 0.60, scores

        for name in self.action_names:
            if name not in allowed_set:
                scores[name] = -1e9
                continue
            score = float(base_scores.get(name, 0.0)) - float(cost_prior.get(name, 0.0))
            idx = self.idx(name)
            n = int(self.action_counts[idx])
            mean_reward = float(self.reward_sums[idx] / n) if n > 0 else float("nan")
            if name != "skip" and not math.isnan(mean_reward):
                score += float(getattr(cfg, "roi_score_weight", 0.0) or 0.0) * mean_reward
                if n >= max(1, int(getattr(cfg, "roi_disable_min_trials", 1) or 1)) and mean_reward <= float(getattr(cfg, "roi_disable_threshold", -0.03) or -0.03):
                    score -= float(getattr(cfg, "roi_disable_penalty", 0.40) or 0.40)
                if n >= max(1, int(getattr(cfg, "roi_promote_min_trials", 1) or 1)) and mean_reward >= float(getattr(cfg, "roi_promote_threshold", 0.08) or 0.08):
                    score += float(getattr(cfg, "roi_promote_weight", 0.30) or 0.30) * mean_reward
            if use_bandit:
                A = self.A[idx]
                b = self.b[idx]
                try:
                    A_inv = np.linalg.inv(A)
                except Exception:
                    A_inv = np.linalg.pinv(A)
                theta = A_inv @ b
                explore = float(cfg.ucb_alpha) * math.sqrt(max(float(x @ A_inv @ x), 0.0))
                score += float(theta @ x) + explore
            elif (not use_tree_only) and policy_mode not in ("linear",):
                # Unknown mode -> fail safe to the deterministic prior.
                pass
            scores[name] = score

        best_name = max(scores.keys(), key=lambda k: float(scores[k]))
        logits = np.asarray([scores[n] for n in self.action_names], dtype=np.float64)
        logits = logits - np.nanmax(logits)
        probs = np.exp(np.clip(logits, -40.0, 40.0))
        denom = float(probs.sum())
        conf = float(probs[self.idx(best_name)] / denom) if denom > 0.0 else 1.0
        return str(best_name), float(conf), scores

    def update(self,
               action_name: str,
               x: np.ndarray,
               reward: float,
               hw_cycles: float = 0.0,
               improved: bool = False,
               true_fix: bool = False) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        idx = self.idx(action_name)
        self.A[idx] = self.A[idx] + np.outer(x, x)
        self.b[idx] = self.b[idx] + float(reward) * x
        self.action_counts[idx] += 1
        self.last_reward[idx] = float(reward)
        self.reward_sums[idx] += float(reward)
        self.cost_sums[idx] += float(hw_cycles)
        self.improve_counts[idx] += int(bool(improved))
        self.fix_counts[idx] += int(bool(true_fix))


def _ai_gate_select_snapshot(snapshot_schedule: Sequence[int], cfg: AIGatedHybridConfig) -> int:
    vals = [int(v) for v in snapshot_schedule if int(v) > 0]
    if not vals:
        return 0
    pol = str(getattr(cfg, "gate_snapshot_policy", "first") or "first").strip().lower()
    if pol in ("final", "last", "stage1"):
        return int(vals[-1])
    if pol in ("first", "early", "min"):
        return int(vals[0])
    try:
        target = int(float(pol))
        valid = [v for v in vals if v <= target]
        return int(valid[-1]) if valid else int(vals[0])
    except Exception:
        return int(vals[0])


def _ai_gate_block_concentration(union_vars: np.ndarray, block_size: int) -> float:
    union_vars = np.asarray(union_vars, dtype=np.int64).reshape(-1)
    if union_vars.size == 0:
        return 0.0
    bs = max(1, int(block_size))
    block_ids = (union_vars // bs).astype(np.int64, copy=False)
    max_block = int(block_ids.max()) if block_ids.size else 0
    counts = np.bincount(block_ids, minlength=max_block + 1)
    if counts.size == 0:
        return 0.0
    return float(counts.max()) / float(max(1, union_vars.size))


def _extract_ai_gate_context(frame: FrameLog,
                             sim_cfg: SimulationConfig,
                             snapshot_iter: int,
                             cfg: AIGatedHybridConfig) -> Tuple[np.ndarray, Dict[str, float]]:
    code_cfg = sim_cfg.code
    snaps = frame.snapshots
    syn_snaps = snaps.get("syndrome", {})
    llr_snaps = snaps.get("llr", {})

    syndrome = np.asarray(syn_snaps.get(int(snapshot_iter), np.array([], dtype=np.uint8)), dtype=np.uint8).reshape(-1)
    llr_snapshot = np.asarray(llr_snaps.get(int(snapshot_iter), np.array([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    syn_w = int(syndrome.sum()) if syndrome.size else 0
    unsat_checks = np.flatnonzero(syndrome).astype(np.int32) if syndrome.size else np.array([], dtype=np.int32)

    clusters = find_variable_clusters_from_syndrome(syndrome, code_cfg) if syndrome.size else []
    if clusters:
        union_vars = np.unique(np.concatenate(clusters)).astype(np.int32)
        cluster_sizes = sorted([int(np.asarray(c, dtype=np.int32).size) for c in clusters], reverse=True)
    else:
        union_vars = np.array([], dtype=np.int32)
        cluster_sizes = []

    union_size = int(union_vars.size)
    num_clusters = int(len(cluster_sizes))
    largest_cluster = int(cluster_sizes[0]) if cluster_sizes else 0
    max_cluster_ratio = float(largest_cluster) / float(max(1, union_size))
    block_conc = _ai_gate_block_concentration(union_vars, int(getattr(cfg, "block_size", 64) or 64))

    if llr_snapshot.size > 0 and union_size > 0:
        abs_all = np.abs(llr_snapshot).astype(np.float64, copy=False)
        abs_union = np.abs(llr_snapshot[union_vars]).astype(np.float64, copy=False)
        try:
            weak_thr = float(np.quantile(abs_all, float(getattr(cfg, "weak_llr_quantile", 0.30) or 0.30)))
        except Exception:
            weak_thr = float(np.median(abs_all)) if abs_all.size else 0.0
        weak_thr = min(float(getattr(cfg, "weak_llr_abs_cap", 2.5) or 2.5), max(0.5, weak_thr))
        low_llr_frac = float(np.mean(abs_union <= weak_thr)) if abs_union.size else 0.0
        mean_abs_norm = min(float(abs_union.mean()) / 8.0, 1.0) if abs_union.size else 1.0
    else:
        weak_thr = 0.0
        low_llr_frac = 0.0
        mean_abs_norm = 1.0

    disagree_rate = 0.0
    weak_disagree_rate = 0.0
    llr_channel = getattr(frame, "llr_channel", None)
    if llr_channel is not None and llr_snapshot.size > 0 and union_size > 0:
        llr_channel = np.asarray(llr_channel, dtype=np.float32).reshape(-1)
        snap_sign = np.sign(llr_snapshot[union_vars]).astype(np.int8, copy=False)
        chan_sign = np.sign(llr_channel[union_vars]).astype(np.int8, copy=False)
        disagree_mask = (snap_sign * chan_sign) < 0
        disagree_rate = float(np.mean(disagree_mask)) if disagree_mask.size else 0.0
        weak_mask = np.abs(llr_snapshot[union_vars]) <= float(weak_thr)
        weak_disagree_rate = float(np.mean(disagree_mask & weak_mask)) if disagree_mask.size else 0.0

    compactness = float(np.clip(0.55 * max_cluster_ratio + 0.45 * block_conc, 0.0, 1.0))
    uncertainty = float(np.clip(0.60 * low_llr_frac + 0.40 * (1.0 - mean_abs_norm), 0.0, 1.0))
    conflict = float(np.clip(0.65 * disagree_rate + 0.35 * weak_disagree_rate, 0.0, 1.0))
    diffuse = float(np.clip(
        0.45 * min(float(union_size) / 192.0, 1.0)
        + 0.30 * min(float(num_clusters) / 8.0, 1.0)
        + 0.25 * min(float(syn_w) / 96.0, 1.0),
        0.0,
        1.0,
    ))

    stage1_iter = int(frame.iter_used if frame.iter_used is not None else max(1, int(snapshot_iter)))
    earlyness = 1.0 - (float(max(1, int(snapshot_iter))) - 1.0) / float(max(1, stage1_iter))
    earlyness = float(np.clip(earlyness, 0.0, 1.0))

    promise = float(np.clip(0.50 * compactness + 0.25 * uncertainty + 0.25 * conflict - 0.55 * diffuse, -1.0, 1.0))

    x = np.asarray([
        1.0,
        compactness,
        uncertainty,
        conflict,
        1.0 - diffuse,
        earlyness,
    ], dtype=np.float64)

    meta = {
        "snapshot_iter": int(snapshot_iter),
        "syndrome_weight": int(syn_w),
        "union_size": int(union_size),
        "num_clusters": int(num_clusters),
        "largest_cluster_ratio": float(max_cluster_ratio),
        "block_concentration": float(block_conc),
        "low_llr_frac": float(low_llr_frac),
        "disagreement_rate": float(disagree_rate),
        "weak_disagreement_rate": float(weak_disagree_rate),
        "compactness": float(compactness),
        "uncertainty": float(uncertainty),
        "conflict": float(conflict),
        "diffuse": float(diffuse),
        "promise": float(promise),
        "earlyness": float(earlyness),
    }
    return x, meta


def _ai_gate_allowed_actions(meta: Dict[str, float],
                             cfg: AIGatedHybridConfig,
                             snapshot_pos: int = 1) -> List[str]:
    compactness = float(meta.get("compactness", 0.0))
    conflict = float(meta.get("conflict", 0.0))
    diffuse = float(meta.get("diffuse", 1.0))
    promise = float(meta.get("promise", -1.0))
    block_conc = float(meta.get("block_concentration", 0.0))

    if diffuse >= float(getattr(cfg, "force_skip_diffuse", 0.95) or 0.95) or promise <= float(getattr(cfg, "force_skip_promise", -0.22) or -0.22):
        allowed = ["skip", "tiny"]
        if block_conc >= float(getattr(cfg, "tree_full_block_concentration", 0.07) or 0.07) and compactness >= float(getattr(cfg, "tree_full_compactness", 0.24) or 0.24):
            allowed.append("full")
    else:
        allowed = ["skip", "tiny", "full"]
        if compactness >= float(getattr(cfg, "meta_min_compactness", 0.34) or 0.34) and conflict >= float(getattr(cfg, "meta_min_conflict", 0.08) or 0.08):
            allowed.append("meta")

    pos = max(1, int(snapshot_pos))
    if pos > max(1, int(getattr(cfg, "tiny_snapshot_cap", 2) or 2)):
        allowed = [a for a in allowed if a != "tiny"]
    if pos > max(1, int(getattr(cfg, "full_snapshot_cap", 3) or 3)):
        allowed = [a for a in allowed if a != "full"]
    if pos > max(1, int(getattr(cfg, "meta_snapshot_cap", 4) or 4)):
        allowed = [a for a in allowed if a != "meta"]
    if not allowed:
        allowed = ["skip"]
    if "skip" not in allowed:
        allowed = ["skip"] + list(allowed)
    return allowed


def _ai_gate_linear_scores(meta: Dict[str, float]) -> Dict[str, float]:
    compactness = float(meta.get("compactness", 0.0))
    uncertainty = float(meta.get("uncertainty", 0.0))
    conflict = float(meta.get("conflict", 0.0))
    diffuse = float(meta.get("diffuse", 1.0))
    earlyness = float(meta.get("earlyness", 0.0))
    return {
        "skip": 1.10 * diffuse - 0.70 * compactness - 0.35 * uncertainty - 0.25 * conflict,
        "tiny": 0.25 * compactness + 0.55 * uncertainty + 0.15 * conflict + 0.05 * earlyness - 0.10 * diffuse,
        "full": 0.80 * compactness + 0.55 * uncertainty + 0.25 * conflict + 0.10 * earlyness - 0.15 * diffuse,
        "meta": 1.00 * compactness + 0.55 * uncertainty + 0.45 * conflict + 0.10 * earlyness - 0.10 * diffuse,
    }


def _ai_gate_distilled_tree_scores(meta: Dict[str, float],
                                   cfg: AIGatedHybridConfig,
                                   snapshot_pos: int = 1,
                                   prev_meta: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, float], str]:
    compactness = float(meta.get("compactness", 0.0))
    uncertainty = float(meta.get("uncertainty", 0.0))
    conflict = float(meta.get("conflict", 0.0))
    diffuse = float(meta.get("diffuse", 1.0))
    promise = float(meta.get("promise", -1.0))
    earlyness = float(meta.get("earlyness", 0.0))
    union_size = int(meta.get("union_size", 0) or 0)
    num_clusters = int(meta.get("num_clusters", 0) or 0)
    block_conc = float(meta.get("block_concentration", 0.0))
    weak_disagree = float(meta.get("weak_disagreement_rate", 0.0))
    promise_prev = float(prev_meta.get("promise", promise)) if isinstance(prev_meta, dict) else float(promise)
    promise_delta = float(promise - promise_prev)

    leaf = "fallback_linear"
    scores = {"skip": 0.0, "tiny": 0.0, "full": 0.0, "meta": 0.0}

    late_stop_progress = float(getattr(cfg, "tree_late_stop_progress", 0.01) or 0.01)
    late_stop_compactness = float(getattr(cfg, "tree_late_stop_compactness", 0.30) or 0.30)
    late_stop_block = float(getattr(cfg, "tree_late_stop_block_concentration", 0.08) or 0.08)

    if int(snapshot_pos) > 1 and promise_delta <= late_stop_progress and compactness <= late_stop_compactness and block_conc <= late_stop_block:
        leaf = "late_stop"
        scores = {"skip": 1.18, "tiny": 0.04, "full": -0.78, "meta": -1.05}
    elif (((diffuse >= float(getattr(cfg, "tree_diffuse_skip", 0.92) or 0.92)) and block_conc <= float(getattr(cfg, "tree_skip_block_concentration", 0.06) or 0.06))
          or ((union_size >= int(getattr(cfg, "tree_skip_union_size", 448) or 448)) and block_conc <= float(getattr(cfg, "tree_skip_block_concentration", 0.06) or 0.06) and num_clusters >= 3)
          or (promise <= float(getattr(cfg, "tree_skip_promise", -0.12) or -0.12) and block_conc <= float(getattr(cfg, "tree_skip_block_concentration", 0.06) or 0.06) and num_clusters >= 3)):
        if block_conc >= float(getattr(cfg, "tree_tiny_block_concentration", 0.05) or 0.05) and compactness >= float(getattr(cfg, "tree_tiny_rescue_compactness", 0.12) or 0.12) and uncertainty >= float(getattr(cfg, "tree_tiny_rescue_uncertainty", 0.22) or 0.22):
            leaf = "diffuse_tiny_salvage"
            scores = {"skip": 0.46, "tiny": 0.94, "full": -0.34, "meta": -0.72}
        else:
            leaf = "diffuse_skip"
            scores = {"skip": 1.02, "tiny": 0.34, "full": -0.54, "meta": -0.86}
    elif (block_conc >= float(getattr(cfg, "tree_meta_block_concentration", 0.11) or 0.11)
          and compactness >= float(getattr(cfg, "tree_meta_compactness", 0.38) or 0.38)
          and conflict >= float(getattr(cfg, "tree_meta_conflict", 0.09) or 0.09)
          and promise >= float(getattr(cfg, "tree_meta_promise", 0.02) or 0.02)):
        leaf = "meta_ready"
        scores = {"skip": -0.85, "tiny": 0.10, "full": 0.74, "meta": 1.08}
    elif (block_conc >= float(getattr(cfg, "tree_full_block_concentration", 0.09) or 0.09)
          and compactness >= float(getattr(cfg, "tree_full_compactness", 0.24) or 0.24)
          and uncertainty >= float(getattr(cfg, "tree_full_uncertainty", 0.20) or 0.20)
          and conflict >= float(getattr(cfg, "tree_full_conflict", 0.06) or 0.06)):
        if promise >= float(getattr(cfg, "tree_full_promise", -0.05) or -0.05) or promise_delta >= float(getattr(cfg, "tree_promote_margin", 0.05) or 0.05):
            leaf = "full_promote"
            scores = {"skip": -0.72, "tiny": 0.28, "full": 1.02, "meta": 0.36}
        else:
            leaf = "tiny_then_full"
            scores = {"skip": -0.30, "tiny": 0.96, "full": 0.42, "meta": 0.00}
    elif (block_conc >= float(getattr(cfg, "tree_tiny_block_concentration", 0.07) or 0.07)
          and compactness >= float(getattr(cfg, "tree_tiny_compactness", 0.16) or 0.16)
          and uncertainty >= float(getattr(cfg, "tree_tiny_uncertainty", 0.20) or 0.20)):
        leaf = "tiny_local"
        scores = {"skip": -0.12, "tiny": 1.02, "full": 0.06, "meta": -0.10}
    else:
        leaf = "weak_skip"
        scores = {"skip": 0.82, "tiny": 0.30, "full": -0.38, "meta": -0.62}

    late_penalty = max(0.0, float(int(snapshot_pos) - 1)) * float(getattr(cfg, "tree_late_snapshot_penalty", 0.18) or 0.18)
    scores["tiny"] -= 0.35 * late_penalty
    scores["full"] -= 1.00 * late_penalty
    scores["meta"] -= 1.25 * late_penalty

    if promise_delta >= float(getattr(cfg, "tree_promote_margin", 0.05) or 0.05):
        scores["full"] += 0.18
        scores["meta"] += 0.10
        leaf = f"{leaf}+promote"
    elif promise_delta <= -float(getattr(cfg, "tree_drop_margin", 0.06) or 0.06):
        scores["skip"] += 0.24
        scores["tiny"] -= 0.10
        scores["full"] -= 0.24
        scores["meta"] -= 0.30
        leaf = f"{leaf}+drop"

    # Mild preference for acting earlier when the snapshot still looks local.
    scores["tiny"] += 0.10 * earlyness
    scores["full"] += 0.05 * earlyness
    scores["meta"] += 0.03 * earlyness

    # Weak disagreement is a useful low-cost confidence cue for rescue quality.
    scores["tiny"] += 0.10 * weak_disagree
    scores["full"] += 0.16 * weak_disagree
    scores["meta"] += 0.08 * weak_disagree
    return scores, leaf


def _ai_gate_base_scores(meta: Dict[str, float],
                         cfg: Optional[AIGatedHybridConfig] = None,
                         snapshot_pos: int = 1,
                         prev_meta: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, float], str]:
    policy_mode = str(getattr(cfg, "policy_mode", "linear_ucb") or "linear_ucb").strip().lower() if cfg is not None else "linear_ucb"
    if policy_mode in ("distilled_tree", "distilled_tree_bandit", "distilled_tree_roi", "tree", "tree_bandit", "tree_roi", "dt", "dt_bandit", "dt_roi") and cfg is not None:
        scores, leaf = _ai_gate_distilled_tree_scores(meta, cfg, snapshot_pos=snapshot_pos, prev_meta=prev_meta)
        return scores, leaf
    return _ai_gate_linear_scores(meta), "linear"


def _ai_gate_reward(action_name: str,
                    bit_errors_before: int,
                    bit_errors_after: int,
                    hw_cycles_grand: int,
                    cfg: AIGatedHybridConfig) -> float:
    be_before = max(0, int(bit_errors_before))
    be_after = max(0, int(bit_errors_after))
    reward = 0.0
    if be_after < be_before:
        frac = float(be_before - be_after) / float(max(1, be_before))
        reward += float(getattr(cfg, "improvement_reward", 0.35) or 0.35) * max(frac, 0.20)
    if be_after == 0 and be_before > 0:
        reward += float(getattr(cfg, "true_fix_reward", 1.15) or 1.15)
    if str(action_name) == "skip" and be_before > 0 and be_after >= be_before:
        reward -= float(getattr(cfg, "skip_failure_penalty", 0.26) or 0.26)
    cost_scale = max(1.0, float(getattr(cfg, "cost_scale_cycles", 240000.0) or 240000.0))
    reward -= float(getattr(cfg, "cost_lambda", 0.30) or 0.30) * min(float(hw_cycles_grand) / cost_scale, 2.0)
    return float(reward)


def _stage2_syndrome_drop_ratio(res: Optional[ClusterGrandResult]) -> float:
    if res is None:
        return 0.0
    sw0 = max(0, int(getattr(res, "initial_syndrome_weight", 0) or 0))
    sw1 = max(0, int(getattr(res, "final_syndrome_weight", sw0) or sw0))
    best_syn = getattr(res, "best_progress_syndrome_weight", None)
    if best_syn is not None:
        try:
            sw1 = min(sw1, max(0, int(best_syn)))
        except Exception:
            pass
    if sw0 <= 0:
        return 0.0
    return float(max(0, sw0 - sw1)) / float(max(1, sw0))


def _ai_probe_regime(stage1_fail_rate_est: float,
                     frames_seen: int,
                     cfg: AIGatedHybridConfig) -> str:
    if int(frames_seen) < max(1, int(getattr(cfg, "probe_failrate_min_frames", 192) or 192)):
        return "warmup"
    hard_thr = float(getattr(cfg, "probe_failrate_hard", 0.080) or 0.080)
    easy_thr = float(getattr(cfg, "probe_failrate_easy", 0.018) or 0.018)
    if float(stage1_fail_rate_est) >= hard_thr:
        return "hard"
    if float(stage1_fail_rate_est) <= easy_thr:
        return "easy"
    return "rescue"


def _ai_probe_plan(meta: Dict[str, float],
                   stage1_fail_rate_est: float,
                   frames_seen: int,
                   snapshot_pos: int,
                   probe_drop: float,
                   cfg: AIGatedHybridConfig) -> Tuple[str, bool, bool, bool, str]:
    compactness = float(meta.get("compactness", 0.0))
    block_conc = float(meta.get("block_concentration", 0.0))
    conflict = float(meta.get("conflict", 0.0))
    promise = float(meta.get("promise", -1.0))

    local = (
        compactness >= float(getattr(cfg, "probe_local_compactness", 0.16) or 0.16)
        and block_conc >= float(getattr(cfg, "probe_local_block_concentration", 0.07) or 0.07)
        and conflict >= float(getattr(cfg, "probe_local_conflict", 0.06) or 0.06)
    )
    strong = (
        compactness >= float(getattr(cfg, "probe_strong_compactness", 0.26) or 0.26)
        and block_conc >= float(getattr(cfg, "probe_strong_block_concentration", 0.10) or 0.10)
        and conflict >= float(getattr(cfg, "probe_strong_conflict", 0.09) or 0.09)
    )

    regime = _ai_probe_regime(stage1_fail_rate_est, frames_seen, cfg)
    allow_full = False
    allow_meta = False
    allow_next = False
    reason = "stop"

    full_drop = float(getattr(cfg, "probe_min_syndrome_drop_full", 0.08) or 0.08)
    cont_drop = float(getattr(cfg, "probe_min_syndrome_drop_continue", 0.04) or 0.04)
    hard_drop = float(getattr(cfg, "probe_min_syndrome_drop_hard", 0.12) or 0.12)
    next_drop = float(getattr(cfg, "probe_next_snapshot_min_drop", 0.03) or 0.03)
    local_promise = float(getattr(cfg, "probe_local_promise", -0.02) or -0.02)
    meta_drop = float(getattr(cfg, "probe_meta_drop", 0.16) or 0.16)
    late_cap = max(1, int(getattr(cfg, "probe_late_snapshot_cap", 2) or 2))

    if regime == "hard":
        allow_full = strong and (probe_drop >= hard_drop or promise >= max(0.02, local_promise))
        allow_meta = strong and probe_drop >= max(meta_drop, hard_drop)
        allow_next = (allow_full or allow_meta) and snapshot_pos < late_cap and probe_drop >= next_drop
        reason = "hard_gate"
    elif regime == "rescue":
        allow_full = (probe_drop >= full_drop) or (local and promise >= local_promise and probe_drop >= cont_drop)
        allow_meta = strong and promise >= max(0.0, local_promise) and probe_drop >= meta_drop
        allow_next = (local and probe_drop >= next_drop) and snapshot_pos < late_cap
        reason = "rescue_band"
    elif regime == "easy":
        allow_full = local and probe_drop >= cont_drop
        allow_meta = strong and promise >= 0.04 and probe_drop >= meta_drop
        allow_next = False
        reason = "easy_band"
    else:
        allow_full = strong or (local and probe_drop >= cont_drop)
        allow_meta = strong and probe_drop >= max(cont_drop, 0.08)
        allow_next = snapshot_pos < late_cap and probe_drop >= next_drop
        reason = "warmup"

    if probe_drop <= 1e-12 and (not strong):
        allow_meta = False
        if regime != "warmup":
            # Zero measured drop used to force-stop too aggressively. Keep a narrow
            # escape hatch when the residual still looks locally promising.
            if local and promise >= (local_promise - 0.04):
                allow_full = bool(allow_full or (regime in ("rescue", "hard")))
                allow_next = bool(allow_next or (snapshot_pos < late_cap))
                reason = f"{reason}_flat_but_local"
            else:
                allow_full = False
                allow_next = False
                reason = f"{reason}_no_progress"

    return str(regime), bool(allow_full), bool(allow_meta), bool(allow_next), str(reason)


# ---- Receiver 9: AI-gated AHR/BGR controller with lightweight AI ranking ----
RUN_RECEIVER9 = bool(_get_int_env("RUN_RECEIVER9", 0))
GRAND_AIR_USE_BOOST = bool(_get_int_env("GRAND_AIR_USE_BOOST", _get_int_env("GRAND_AHR_USE_BOOST", _get_int_env("GRAND_USE_BOOST", 1))))
GRAND_AIR_USE_FALLBACK = bool(_get_int_env("GRAND_AIR_USE_FALLBACK", 1))


def _apply_air_tanner_knobs(cfg: ClusterGrandConfig, top_blocks_env: str, default_top_blocks: int) -> None:
    cfg.ai_tanner_block_size = _get_int_env("GRAND_AIR_AI_TG_BLOCK_SIZE", _get_int_env("GRAND_AIR_AI_WINDOW_BLOCK_SIZE", 64))
    cfg.ai_tanner_static_prior_weight = _get_float_env("GRAND_AIR_AI_TG_STATIC_WEIGHT", 0.30)
    cfg.ai_tanner_message_weight = _get_float_env("GRAND_AIR_AI_TG_MESSAGE_WEIGHT", 1.00)
    cfg.ai_tanner_cycle_weight = _get_float_env("GRAND_AIR_AI_TG_CYCLE_WEIGHT", 0.42)
    cfg.ai_tanner_local_share_diffuse = _get_float_env("GRAND_AIR_AI_TG_LOCAL_SHARE_DIFFUSE", 0.30)
    cfg.ai_tanner_local_share_balanced = _get_float_env("GRAND_AIR_AI_TG_LOCAL_SHARE_BALANCED", 0.48)
    cfg.ai_tanner_local_share_compact = _get_float_env("GRAND_AIR_AI_TG_LOCAL_SHARE_COMPACT", 0.66)
    cfg.ai_tanner_diffuse_union_size = _get_int_env("GRAND_AIR_AI_TG_DIFFUSE_UNION", 192)
    cfg.ai_tanner_diffuse_block_concentration = _get_float_env("GRAND_AIR_AI_TG_DIFFUSE_BLOCK", 0.09)
    cfg.ai_tanner_compact_block_concentration = _get_float_env("GRAND_AIR_AI_TG_COMPACT_BLOCK", 0.12)
    cfg.ai_tanner_top_blocks = _get_int_env(top_blocks_env, default_top_blocks)
    cfg.ai_tanner_neighbor_blocks = _get_int_env("GRAND_AIR_AI_TG_NEIGHBOR_BLOCKS", _get_int_env("GRAND_AIR_AI_WINDOW_NEIGHBOR_BLOCKS", 1))
    cfg.ai_tanner_diffuse_extra_blocks = _get_int_env("GRAND_AIR_AI_TG_DIFFUSE_EXTRA_BLOCKS", 1)
    cfg.ai_tanner_top_global_extra = _get_int_env("GRAND_AIR_AI_TG_TOP_GLOBAL_EXTRA", 10)


def _apply_air_tg2_knobs(cfg: ClusterGrandConfig) -> None:
    cfg.ai_tg2_block_size = _get_int_env("GRAND_AIR_AI_TG2_BLOCK_SIZE", _get_int_env("GRAND_AIR_AI_TG_BLOCK_SIZE", 64))
    cfg.ai_tg2_prefilter_scale = _get_float_env("GRAND_AIR_AI_TG2_PREFILTER_SCALE", 2.4)
    cfg.ai_tg2_prefilter_extra = _get_int_env("GRAND_AIR_AI_TG2_PREFILTER_EXTRA", 24)
    cfg.ai_tg2_diffuse_union_size = _get_int_env("GRAND_AIR_AI_TG2_DIFFUSE_UNION", _get_int_env("GRAND_AIR_AI_TG_DIFFUSE_UNION", 192))
    cfg.ai_tg2_diffuse_block_concentration = _get_float_env("GRAND_AIR_AI_TG2_DIFFUSE_BLOCK", _get_float_env("GRAND_AIR_AI_TG_DIFFUSE_BLOCK", 0.09))
    cfg.ai_tg2_compact_block_concentration = _get_float_env("GRAND_AIR_AI_TG2_COMPACT_BLOCK", _get_float_env("GRAND_AIR_AI_TG_COMPACT_BLOCK", 0.12))
    cfg.ai_tg2_local_share_diffuse = _get_float_env("GRAND_AIR_AI_TG2_LOCAL_SHARE_DIFFUSE", 0.58)
    cfg.ai_tg2_local_share_balanced = _get_float_env("GRAND_AIR_AI_TG2_LOCAL_SHARE_BALANCED", 0.66)
    cfg.ai_tg2_local_share_compact = _get_float_env("GRAND_AIR_AI_TG2_LOCAL_SHARE_COMPACT", 0.78)
    cfg.ai_tg2_seed_vars_diffuse = _get_int_env("GRAND_AIR_AI_TG2_SEED_VARS_DIFFUSE", 10)
    cfg.ai_tg2_seed_vars_balanced = _get_int_env("GRAND_AIR_AI_TG2_SEED_VARS_BALANCED", 8)
    cfg.ai_tg2_seed_vars_compact = _get_int_env("GRAND_AIR_AI_TG2_SEED_VARS_COMPACT", 6)
    cfg.ai_tg2_top_checks_diffuse = _get_int_env("GRAND_AIR_AI_TG2_TOP_CHECKS_DIFFUSE", 8)
    cfg.ai_tg2_top_checks_balanced = _get_int_env("GRAND_AIR_AI_TG2_TOP_CHECKS_BALANCED", 6)
    cfg.ai_tg2_top_checks_compact = _get_int_env("GRAND_AIR_AI_TG2_TOP_CHECKS_COMPACT", 5)
    cfg.ai_tg2_neighbor_blocks = _get_int_env("GRAND_AIR_AI_TG2_NEIGHBOR_BLOCKS", 1)
    cfg.ai_tg2_radius = _get_int_env("GRAND_AIR_AI_TG2_RADIUS", 1)
    cfg.ai_tg2_min_local_budget = _get_int_env("GRAND_AIR_AI_TG2_MIN_LOCAL_BUDGET", 6)

# Tiny low-latency rescue arm: cheap shot on compact early residuals.
grand_cfg_awgn_air_tiny = copy.deepcopy(grand_cfg_awgn_ahr)
grand_cfg_awgn_air_tiny.selection_mode = os.environ.get("GRAND_AIR_SELECTION_MODE", "ai_rank").strip().lower() or "ai_rank"
grand_cfg_awgn_air_tiny.llr_source = os.environ.get("GRAND_AIR_LLR_SOURCE", getattr(grand_cfg_awgn_ahr, "llr_source", "mixed")).strip().lower() or "mixed"
grand_cfg_awgn_air_tiny.max_patterns = _get_int_env("GRAND_AIR_TINY_MAX_PATTERNS", 18000)
grand_cfg_awgn_air_tiny.restart_max_candidates = _get_int_env("GRAND_AIR_TINY_RESTART_MAX_CANDIDATES", 10)
grand_cfg_awgn_air_tiny.restart_ldpc_iters = _get_int_env("GRAND_AIR_TINY_RESTART_ITERS", 10)
grand_cfg_awgn_air_tiny.restart_llr_gain = _get_float_env("GRAND_AIR_TINY_RESTART_GAIN", 4.2)
grand_cfg_awgn_air_tiny.restart_dual_gain = _get_float_env("GRAND_AIR_TINY_RESTART_DUAL_GAIN", 0.0)
grand_cfg_awgn_air_tiny.soft_candidate_ratio = _get_float_env("GRAND_AIR_TINY_RATIO", 2.6)
grand_cfg_awgn_air_tiny.soft_max_bits = _get_int_env("GRAND_AIR_TINY_MAX_BITS", 72)
grand_cfg_awgn_air_tiny.soft_core_max_bits = _get_int_env("GRAND_AIR_TINY_CORE_MAX_BITS", 10)
grand_cfg_awgn_air_tiny.soft_max_weight = _get_int_env("GRAND_AIR_TINY_CORE_MAX_WEIGHT", 2)
grand_cfg_awgn_air_tiny.soft_max_candidates = _get_int_env("GRAND_AIR_TINY_MAX_CANDIDATES", 48)
grand_cfg_awgn_air_tiny.peel_candidate_ratio = _get_float_env("GRAND_AIR_TINY_PEEL_RATIO", 1.7)
grand_cfg_awgn_air_tiny.peel_extra_llr_bits = _get_int_env("GRAND_AIR_TINY_PEEL_EXTRA_LLR_BITS", 8)
grand_cfg_awgn_air_tiny.osd_disagreement_extra_bits = _get_int_env("GRAND_AIR_TINY_DISAGREEMENT_BITS", 8)
grand_cfg_awgn_air_tiny.ai_rank_vote_weight = _get_float_env("GRAND_AIR_AI_VOTE_WEIGHT", 1.00)
grand_cfg_awgn_air_tiny.ai_rank_llr_weight = _get_float_env("GRAND_AIR_AI_LLR_WEIGHT", 0.85)
grand_cfg_awgn_air_tiny.ai_rank_disagreement_weight = _get_float_env("GRAND_AIR_AI_DISAGREEMENT_WEIGHT", 0.55)
grand_cfg_awgn_air_tiny.ai_rank_density_weight = _get_float_env("GRAND_AIR_AI_DENSITY_WEIGHT", 0.35)
grand_cfg_awgn_air_tiny.ai_rank_roi_block_size = _get_int_env("GRAND_AIR_AI_ROI_BLOCK_SIZE", 64)
grand_cfg_awgn_air_tiny.ai_rank_roi_weak_llr_quantile = _get_float_env("GRAND_AIR_AI_ROI_WEAK_Q", 0.30)
grand_cfg_awgn_air_tiny.ai_rank_roi_weak_llr_abs_cap = _get_float_env("GRAND_AIR_AI_ROI_WEAK_ABS", 2.50)
grand_cfg_awgn_air_tiny.ai_rank_roi_diffuse_union_size = _get_int_env("GRAND_AIR_AI_ROI_DIFFUSE_UNION", 208)
grand_cfg_awgn_air_tiny.ai_rank_roi_diffuse_block_concentration = _get_float_env("GRAND_AIR_AI_ROI_DIFFUSE_BLOCK", 0.08)
grand_cfg_awgn_air_tiny.ai_rank_roi_compact_block_concentration = _get_float_env("GRAND_AIR_AI_ROI_COMPACT_BLOCK", 0.11)
grand_cfg_awgn_air_tiny.ai_rank_roi_diffuse_l_scale = _get_float_env("GRAND_AIR_AI_ROI_DIFFUSE_L_SCALE", 0.70)
grand_cfg_awgn_air_tiny.ai_rank_roi_local_conflict_bonus = _get_float_env("GRAND_AIR_AI_ROI_CONFLICT_BONUS", 0.30)
grand_cfg_awgn_air_tiny.ai_window_block_size = _get_int_env("GRAND_AIR_AI_WINDOW_BLOCK_SIZE", 64)
grand_cfg_awgn_air_tiny.ai_window_top_blocks = _get_int_env("GRAND_AIR_AI_WINDOW_TINY_TOP_BLOCKS", 1)
grand_cfg_awgn_air_tiny.ai_window_neighbor_blocks = _get_int_env("GRAND_AIR_AI_WINDOW_NEIGHBOR_BLOCKS", 0)
grand_cfg_awgn_air_tiny.ai_window_local_seed_per_block = _get_int_env("GRAND_AIR_AI_WINDOW_LOCAL_SEEDS", 4)
grand_cfg_awgn_air_tiny.ai_window_diffuse_extra_blocks = _get_int_env("GRAND_AIR_AI_WINDOW_DIFFUSE_EXTRA_BLOCKS", 1)
grand_cfg_awgn_air_tiny.ai_window_compact_single_threshold = _get_float_env("GRAND_AIR_AI_WINDOW_COMPACT_SINGLE", 0.18)
grand_cfg_awgn_air_tiny.ai_window_block_score_conflict_bonus = _get_float_env("GRAND_AIR_AI_WINDOW_CONFLICT_BONUS", 0.35)
grand_cfg_awgn_air_tiny.ai_window_block_score_density_bonus = _get_float_env("GRAND_AIR_AI_WINDOW_DENSITY_BONUS", 0.20)
_apply_air_tanner_knobs(grand_cfg_awgn_air_tiny, "GRAND_AIR_AI_TG_TINY_TOP_BLOCKS", 1)
_apply_air_tg2_knobs(grand_cfg_awgn_air_tiny)

grand_cfg_awgn_air_full = copy.deepcopy(grand_cfg_awgn_meta)
grand_cfg_awgn_air_full.selection_mode = os.environ.get("GRAND_AIR_SELECTION_MODE", "ai_rank").strip().lower() or "ai_rank"
grand_cfg_awgn_air_full.llr_source = os.environ.get("GRAND_AIR_LLR_SOURCE", getattr(grand_cfg_awgn_meta, "llr_source", "mixed")).strip().lower() or "mixed"
grand_cfg_awgn_air_full.max_patterns = _get_int_env("GRAND_AIR_MAX_PATTERNS", max(int(getattr(grand_cfg_awgn_meta, "max_patterns", 0) or 0), 160000))
grand_cfg_awgn_air_full.restart_max_candidates = _get_int_env("GRAND_AIR_RESTART_MAX_CANDIDATES", max(int(getattr(grand_cfg_awgn_meta, "restart_max_candidates", 0) or 0), 28))
grand_cfg_awgn_air_full.restart_ldpc_iters = _get_int_env("GRAND_AIR_RESTART_ITERS", max(int(getattr(grand_cfg_awgn_meta, "restart_ldpc_iters", 0) or 0), 22))
grand_cfg_awgn_air_full.restart_llr_gain = _get_float_env("GRAND_AIR_RESTART_GAIN", max(float(getattr(grand_cfg_awgn_meta, "restart_llr_gain", 0.0) or 0.0), 5.2))
grand_cfg_awgn_air_full.restart_dual_gain = _get_float_env("GRAND_AIR_RESTART_DUAL_GAIN", max(float(getattr(grand_cfg_awgn_meta, "restart_dual_gain", 0.0) or 0.0), 7.4))
grand_cfg_awgn_air_full.soft_candidate_ratio = _get_float_env("GRAND_AIR_RATIO", max(float(getattr(grand_cfg_awgn_meta, "soft_candidate_ratio", 0.0) or 0.0), 3.5))
grand_cfg_awgn_air_full.soft_max_candidates = _get_int_env("GRAND_AIR_MAX_CANDIDATES", max(int(getattr(grand_cfg_awgn_meta, "soft_max_candidates", 0) or 0), 160))
grand_cfg_awgn_air_full.soft_core_max_bits = _get_int_env("GRAND_AIR_CORE_MAX_BITS", max(int(getattr(grand_cfg_awgn_meta, "soft_core_max_bits", 0) or 0), 16))
grand_cfg_awgn_air_full.ai_rank_vote_weight = _get_float_env("GRAND_AIR_AI_VOTE_WEIGHT", 1.00)
grand_cfg_awgn_air_full.ai_rank_llr_weight = _get_float_env("GRAND_AIR_AI_LLR_WEIGHT", 0.85)
grand_cfg_awgn_air_full.ai_rank_disagreement_weight = _get_float_env("GRAND_AIR_AI_DISAGREEMENT_WEIGHT", 0.55)
grand_cfg_awgn_air_full.ai_rank_density_weight = _get_float_env("GRAND_AIR_AI_DENSITY_WEIGHT", 0.35)
grand_cfg_awgn_air_full.ai_rank_roi_block_size = _get_int_env("GRAND_AIR_AI_ROI_BLOCK_SIZE", 64)
grand_cfg_awgn_air_full.ai_rank_roi_weak_llr_quantile = _get_float_env("GRAND_AIR_AI_ROI_WEAK_Q", 0.30)
grand_cfg_awgn_air_full.ai_rank_roi_weak_llr_abs_cap = _get_float_env("GRAND_AIR_AI_ROI_WEAK_ABS", 2.50)
grand_cfg_awgn_air_full.ai_rank_roi_diffuse_union_size = _get_int_env("GRAND_AIR_AI_ROI_DIFFUSE_UNION", 208)
grand_cfg_awgn_air_full.ai_rank_roi_diffuse_block_concentration = _get_float_env("GRAND_AIR_AI_ROI_DIFFUSE_BLOCK", 0.08)
grand_cfg_awgn_air_full.ai_rank_roi_compact_block_concentration = _get_float_env("GRAND_AIR_AI_ROI_COMPACT_BLOCK", 0.11)
grand_cfg_awgn_air_full.ai_rank_roi_diffuse_l_scale = _get_float_env("GRAND_AIR_AI_ROI_DIFFUSE_L_SCALE", 0.70)
grand_cfg_awgn_air_full.ai_rank_roi_local_conflict_bonus = _get_float_env("GRAND_AIR_AI_ROI_CONFLICT_BONUS", 0.30)
grand_cfg_awgn_air_full.ai_window_block_size = _get_int_env("GRAND_AIR_AI_WINDOW_BLOCK_SIZE", 64)
grand_cfg_awgn_air_full.ai_window_top_blocks = _get_int_env("GRAND_AIR_AI_WINDOW_FULL_TOP_BLOCKS", 2)
grand_cfg_awgn_air_full.ai_window_neighbor_blocks = _get_int_env("GRAND_AIR_AI_WINDOW_NEIGHBOR_BLOCKS", 0)
grand_cfg_awgn_air_full.ai_window_local_seed_per_block = _get_int_env("GRAND_AIR_AI_WINDOW_LOCAL_SEEDS", 4)
grand_cfg_awgn_air_full.ai_window_diffuse_extra_blocks = _get_int_env("GRAND_AIR_AI_WINDOW_DIFFUSE_EXTRA_BLOCKS", 1)
grand_cfg_awgn_air_full.ai_window_compact_single_threshold = _get_float_env("GRAND_AIR_AI_WINDOW_COMPACT_SINGLE", 0.18)
grand_cfg_awgn_air_full.ai_window_block_score_conflict_bonus = _get_float_env("GRAND_AIR_AI_WINDOW_CONFLICT_BONUS", 0.35)
grand_cfg_awgn_air_full.ai_window_block_score_density_bonus = _get_float_env("GRAND_AIR_AI_WINDOW_DENSITY_BONUS", 0.20)
_apply_air_tanner_knobs(grand_cfg_awgn_air_full, "GRAND_AIR_AI_TG_FULL_TOP_BLOCKS", 2)
_apply_air_tg2_knobs(grand_cfg_awgn_air_full)

grand_cfg_awgn_air_full_boost = copy.deepcopy(grand_cfg_awgn_meta_boost)
grand_cfg_awgn_air_full_boost.selection_mode = os.environ.get("GRAND_AIR_SELECTION_MODE", "ai_rank").strip().lower() or "ai_rank"
grand_cfg_awgn_air_full_boost.llr_source = os.environ.get("GRAND_AIR_LLR_SOURCE", getattr(grand_cfg_awgn_meta_boost, "llr_source", "mixed")).strip().lower() or "mixed"
grand_cfg_awgn_air_full_boost.max_patterns = _get_int_env("GRAND_AIR_BOOST_MAX_PATTERNS", 420000)
grand_cfg_awgn_air_full_boost.ai_rank_vote_weight = _get_float_env("GRAND_AIR_AI_VOTE_WEIGHT", 1.00)
grand_cfg_awgn_air_full_boost.ai_rank_llr_weight = _get_float_env("GRAND_AIR_AI_LLR_WEIGHT", 0.85)
grand_cfg_awgn_air_full_boost.ai_rank_disagreement_weight = _get_float_env("GRAND_AIR_AI_DISAGREEMENT_WEIGHT", 0.55)
grand_cfg_awgn_air_full_boost.ai_rank_density_weight = _get_float_env("GRAND_AIR_AI_DENSITY_WEIGHT", 0.35)
grand_cfg_awgn_air_full_boost.ai_rank_roi_block_size = _get_int_env("GRAND_AIR_AI_ROI_BLOCK_SIZE", 64)
grand_cfg_awgn_air_full_boost.ai_rank_roi_weak_llr_quantile = _get_float_env("GRAND_AIR_AI_ROI_WEAK_Q", 0.30)
grand_cfg_awgn_air_full_boost.ai_rank_roi_weak_llr_abs_cap = _get_float_env("GRAND_AIR_AI_ROI_WEAK_ABS", 2.50)
grand_cfg_awgn_air_full_boost.ai_rank_roi_diffuse_union_size = _get_int_env("GRAND_AIR_AI_ROI_DIFFUSE_UNION", 208)
grand_cfg_awgn_air_full_boost.ai_rank_roi_diffuse_block_concentration = _get_float_env("GRAND_AIR_AI_ROI_DIFFUSE_BLOCK", 0.08)
grand_cfg_awgn_air_full_boost.ai_rank_roi_compact_block_concentration = _get_float_env("GRAND_AIR_AI_ROI_COMPACT_BLOCK", 0.11)
grand_cfg_awgn_air_full_boost.ai_rank_roi_diffuse_l_scale = _get_float_env("GRAND_AIR_AI_ROI_DIFFUSE_L_SCALE", 0.70)
grand_cfg_awgn_air_full_boost.ai_rank_roi_local_conflict_bonus = _get_float_env("GRAND_AIR_AI_ROI_CONFLICT_BONUS", 0.30)
grand_cfg_awgn_air_full_boost.ai_window_block_size = _get_int_env("GRAND_AIR_AI_WINDOW_BLOCK_SIZE", 64)
grand_cfg_awgn_air_full_boost.ai_window_top_blocks = _get_int_env("GRAND_AIR_AI_WINDOW_FULL_TOP_BLOCKS", 2)
grand_cfg_awgn_air_full_boost.ai_window_neighbor_blocks = _get_int_env("GRAND_AIR_AI_WINDOW_NEIGHBOR_BLOCKS", 0)
grand_cfg_awgn_air_full_boost.ai_window_local_seed_per_block = _get_int_env("GRAND_AIR_AI_WINDOW_LOCAL_SEEDS", 4)
grand_cfg_awgn_air_full_boost.ai_window_diffuse_extra_blocks = _get_int_env("GRAND_AIR_AI_WINDOW_DIFFUSE_EXTRA_BLOCKS", 1)
grand_cfg_awgn_air_full_boost.ai_window_compact_single_threshold = _get_float_env("GRAND_AIR_AI_WINDOW_COMPACT_SINGLE", 0.18)
grand_cfg_awgn_air_full_boost.ai_window_block_score_conflict_bonus = _get_float_env("GRAND_AIR_AI_WINDOW_CONFLICT_BONUS", 0.35)
grand_cfg_awgn_air_full_boost.ai_window_block_score_density_bonus = _get_float_env("GRAND_AIR_AI_WINDOW_DENSITY_BONUS", 0.20)
_apply_air_tanner_knobs(grand_cfg_awgn_air_full_boost, "GRAND_AIR_AI_TG_BOOST_TOP_BLOCKS", 3)
_apply_air_tg2_knobs(grand_cfg_awgn_air_full_boost)

grand_cfg_awgn_air_fallback = copy.deepcopy(grand_cfg_awgn_bgr)
grand_cfg_awgn_air_fallback.selection_mode = os.environ.get("GRAND_AIR_SELECTION_MODE", "ai_rank").strip().lower() or "ai_rank"
grand_cfg_awgn_air_fallback.llr_source = os.environ.get("GRAND_AIR_LLR_SOURCE", getattr(grand_cfg_awgn_bgr, "llr_source", "mixed")).strip().lower() or "mixed"
grand_cfg_awgn_air_fallback.max_patterns = _get_int_env("GRAND_AIR_FALLBACK_MAX_PATTERNS", 180000)
grand_cfg_awgn_air_fallback.ai_rank_vote_weight = _get_float_env("GRAND_AIR_AI_VOTE_WEIGHT", 1.00)
grand_cfg_awgn_air_fallback.ai_rank_llr_weight = _get_float_env("GRAND_AIR_AI_LLR_WEIGHT", 0.85)
grand_cfg_awgn_air_fallback.ai_rank_disagreement_weight = _get_float_env("GRAND_AIR_AI_DISAGREEMENT_WEIGHT", 0.55)
grand_cfg_awgn_air_fallback.ai_rank_density_weight = _get_float_env("GRAND_AIR_AI_DENSITY_WEIGHT", 0.35)
grand_cfg_awgn_air_fallback.ai_rank_roi_block_size = _get_int_env("GRAND_AIR_AI_ROI_BLOCK_SIZE", 64)
grand_cfg_awgn_air_fallback.ai_rank_roi_weak_llr_quantile = _get_float_env("GRAND_AIR_AI_ROI_WEAK_Q", 0.30)
grand_cfg_awgn_air_fallback.ai_rank_roi_weak_llr_abs_cap = _get_float_env("GRAND_AIR_AI_ROI_WEAK_ABS", 2.50)
grand_cfg_awgn_air_fallback.ai_rank_roi_diffuse_union_size = _get_int_env("GRAND_AIR_AI_ROI_DIFFUSE_UNION", 208)
grand_cfg_awgn_air_fallback.ai_rank_roi_diffuse_block_concentration = _get_float_env("GRAND_AIR_AI_ROI_DIFFUSE_BLOCK", 0.08)
grand_cfg_awgn_air_fallback.ai_rank_roi_compact_block_concentration = _get_float_env("GRAND_AIR_AI_ROI_COMPACT_BLOCK", 0.11)
grand_cfg_awgn_air_fallback.ai_rank_roi_diffuse_l_scale = _get_float_env("GRAND_AIR_AI_ROI_DIFFUSE_L_SCALE", 0.70)
grand_cfg_awgn_air_fallback.ai_rank_roi_local_conflict_bonus = _get_float_env("GRAND_AIR_AI_ROI_CONFLICT_BONUS", 0.30)
grand_cfg_awgn_air_fallback.ai_window_block_size = _get_int_env("GRAND_AIR_AI_WINDOW_BLOCK_SIZE", 64)
grand_cfg_awgn_air_fallback.ai_window_top_blocks = _get_int_env("GRAND_AIR_AI_WINDOW_META_TOP_BLOCKS", 3)
grand_cfg_awgn_air_fallback.ai_window_neighbor_blocks = _get_int_env("GRAND_AIR_AI_WINDOW_NEIGHBOR_BLOCKS", 0)
grand_cfg_awgn_air_fallback.ai_window_local_seed_per_block = _get_int_env("GRAND_AIR_AI_WINDOW_LOCAL_SEEDS", 4)
grand_cfg_awgn_air_fallback.ai_window_diffuse_extra_blocks = _get_int_env("GRAND_AIR_AI_WINDOW_DIFFUSE_EXTRA_BLOCKS", 1)
grand_cfg_awgn_air_fallback.ai_window_compact_single_threshold = _get_float_env("GRAND_AIR_AI_WINDOW_COMPACT_SINGLE", 0.18)
grand_cfg_awgn_air_fallback.ai_window_block_score_conflict_bonus = _get_float_env("GRAND_AIR_AI_WINDOW_CONFLICT_BONUS", 0.35)
grand_cfg_awgn_air_fallback.ai_window_block_score_density_bonus = _get_float_env("GRAND_AIR_AI_WINDOW_DENSITY_BONUS", 0.20)
_apply_air_tanner_knobs(grand_cfg_awgn_air_fallback, "GRAND_AIR_AI_TG_META_TOP_BLOCKS", 3)
_apply_air_tg2_knobs(grand_cfg_awgn_air_fallback)

grand_cfg_awgn_air_fallback_boost = copy.deepcopy(grand_cfg_awgn_bgr_boost)
grand_cfg_awgn_air_fallback_boost.selection_mode = os.environ.get("GRAND_AIR_SELECTION_MODE", "ai_rank").strip().lower() or "ai_rank"
grand_cfg_awgn_air_fallback_boost.llr_source = os.environ.get("GRAND_AIR_LLR_SOURCE", getattr(grand_cfg_awgn_bgr_boost, "llr_source", "mixed")).strip().lower() or "mixed"
grand_cfg_awgn_air_fallback_boost.max_patterns = _get_int_env("GRAND_AIR_FALLBACK_BOOST_MAX_PATTERNS", 260000)
grand_cfg_awgn_air_fallback_boost.ai_rank_vote_weight = _get_float_env("GRAND_AIR_AI_VOTE_WEIGHT", 1.00)
grand_cfg_awgn_air_fallback_boost.ai_rank_llr_weight = _get_float_env("GRAND_AIR_AI_LLR_WEIGHT", 0.85)
grand_cfg_awgn_air_fallback_boost.ai_rank_disagreement_weight = _get_float_env("GRAND_AIR_AI_DISAGREEMENT_WEIGHT", 0.55)
grand_cfg_awgn_air_fallback_boost.ai_rank_density_weight = _get_float_env("GRAND_AIR_AI_DENSITY_WEIGHT", 0.35)
grand_cfg_awgn_air_fallback_boost.ai_rank_roi_block_size = _get_int_env("GRAND_AIR_AI_ROI_BLOCK_SIZE", 64)
grand_cfg_awgn_air_fallback_boost.ai_rank_roi_weak_llr_quantile = _get_float_env("GRAND_AIR_AI_ROI_WEAK_Q", 0.30)
grand_cfg_awgn_air_fallback_boost.ai_rank_roi_weak_llr_abs_cap = _get_float_env("GRAND_AIR_AI_ROI_WEAK_ABS", 2.50)
grand_cfg_awgn_air_fallback_boost.ai_rank_roi_diffuse_union_size = _get_int_env("GRAND_AIR_AI_ROI_DIFFUSE_UNION", 208)
grand_cfg_awgn_air_fallback_boost.ai_rank_roi_diffuse_block_concentration = _get_float_env("GRAND_AIR_AI_ROI_DIFFUSE_BLOCK", 0.08)
grand_cfg_awgn_air_fallback_boost.ai_rank_roi_compact_block_concentration = _get_float_env("GRAND_AIR_AI_ROI_COMPACT_BLOCK", 0.11)
grand_cfg_awgn_air_fallback_boost.ai_rank_roi_diffuse_l_scale = _get_float_env("GRAND_AIR_AI_ROI_DIFFUSE_L_SCALE", 0.70)
grand_cfg_awgn_air_fallback_boost.ai_rank_roi_local_conflict_bonus = _get_float_env("GRAND_AIR_AI_ROI_CONFLICT_BONUS", 0.30)
grand_cfg_awgn_air_fallback_boost.ai_window_block_size = _get_int_env("GRAND_AIR_AI_WINDOW_BLOCK_SIZE", 64)
grand_cfg_awgn_air_fallback_boost.ai_window_top_blocks = _get_int_env("GRAND_AIR_AI_WINDOW_META_TOP_BLOCKS", 3)
grand_cfg_awgn_air_fallback_boost.ai_window_neighbor_blocks = _get_int_env("GRAND_AIR_AI_WINDOW_NEIGHBOR_BLOCKS", 0)
grand_cfg_awgn_air_fallback_boost.ai_window_local_seed_per_block = _get_int_env("GRAND_AIR_AI_WINDOW_LOCAL_SEEDS", 4)
grand_cfg_awgn_air_fallback_boost.ai_window_diffuse_extra_blocks = _get_int_env("GRAND_AIR_AI_WINDOW_DIFFUSE_EXTRA_BLOCKS", 1)
grand_cfg_awgn_air_fallback_boost.ai_window_compact_single_threshold = _get_float_env("GRAND_AIR_AI_WINDOW_COMPACT_SINGLE", 0.18)
grand_cfg_awgn_air_fallback_boost.ai_window_block_score_conflict_bonus = _get_float_env("GRAND_AIR_AI_WINDOW_CONFLICT_BONUS", 0.35)
grand_cfg_awgn_air_fallback_boost.ai_window_block_score_density_bonus = _get_float_env("GRAND_AIR_AI_WINDOW_DENSITY_BONUS", 0.20)
_apply_air_tanner_knobs(grand_cfg_awgn_air_fallback_boost, "GRAND_AIR_AI_TG_META_TOP_BLOCKS", 3)
_apply_air_tg2_knobs(grand_cfg_awgn_air_fallback_boost)

ai_gate_cfg_awgn_air = AIGatedHybridConfig(
    gate_snapshot_policy=os.environ.get("GRAND_AIR_GATE_SNAPSHOT", "4").strip() or "4",
    policy_mode=os.environ.get("GRAND_AIR_POLICY_MODE", "distilled_tree_roi").strip().lower() or "distilled_tree_roi",
    dynamic_per_snapshot=bool(_get_int_env("GRAND_AIR_DYNAMIC_PER_SNAPSHOT", 1)),
    tiny_snapshot_cap=_get_int_env("GRAND_AIR_TINY_SNAPSHOT_CAP", 2),
    full_snapshot_cap=_get_int_env("GRAND_AIR_FULL_SNAPSHOT_CAP", 3),
    meta_snapshot_cap=_get_int_env("GRAND_AIR_META_SNAPSHOT_CAP", 4),
    weak_llr_quantile=_get_float_env("GRAND_AIR_WEAK_LLR_QUANTILE", 0.30),
    weak_llr_abs_cap=_get_float_env("GRAND_AIR_WEAK_LLR_ABS_CAP", 2.50),
    block_size=_get_int_env("GRAND_AIR_BLOCK_SIZE", 64),
    ucb_alpha=_get_float_env("GRAND_AIR_UCB_ALPHA", 0.18),
    ridge=_get_float_env("GRAND_AIR_RIDGE", 1.00),
    cost_lambda=_get_float_env("GRAND_AIR_COST_LAMBDA", 0.30),
    cost_scale_cycles=_get_float_env("GRAND_AIR_COST_SCALE_CYCLES", 240000.0),
    improvement_reward=_get_float_env("GRAND_AIR_IMPROVEMENT_REWARD", 0.35),
    true_fix_reward=_get_float_env("GRAND_AIR_TRUE_FIX_REWARD", 1.15),
    skip_failure_penalty=_get_float_env("GRAND_AIR_SKIP_FAILURE_PENALTY", 0.26),
    tiny_cost_prior=_get_float_env("GRAND_AIR_TINY_COST_PRIOR", 0.06),
    full_cost_prior=_get_float_env("GRAND_AIR_FULL_COST_PRIOR", 0.18),
    meta_cost_prior=_get_float_env("GRAND_AIR_META_COST_PRIOR", 0.28),
    meta_min_compactness=_get_float_env("GRAND_AIR_META_MIN_COMPACTNESS", 0.34),
    meta_min_conflict=_get_float_env("GRAND_AIR_META_MIN_CONFLICT", 0.08),
    force_skip_diffuse=_get_float_env("GRAND_AIR_FORCE_SKIP_DIFFUSE", 0.95),
    force_skip_promise=_get_float_env("GRAND_AIR_FORCE_SKIP_PROMISE", -0.22),
    warmup_min_trials_per_action=_get_int_env("GRAND_AIR_WARMUP_MIN_TRIALS_PER_ACTION", 8),
    warmup_snapshot_cap=_get_int_env("GRAND_AIR_WARMUP_SNAPSHOT_CAP", 1),
    suppress_skip_during_warmup=bool(_get_int_env("GRAND_AIR_SUPPRESS_SKIP_DURING_WARMUP", 1)),
    cold_start_bonus=_get_float_env("GRAND_AIR_COLD_START_BONUS", 0.55),
    roi_score_weight=_get_float_env("GRAND_AIR_ROI_SCORE_WEIGHT", 0.75),
    roi_disable_min_trials=_get_int_env("GRAND_AIR_ROI_DISABLE_MIN_TRIALS", 6),
    roi_disable_threshold=_get_float_env("GRAND_AIR_ROI_DISABLE_THRESHOLD", -0.03),
    roi_disable_penalty=_get_float_env("GRAND_AIR_ROI_DISABLE_PENALTY", 0.40),
    roi_promote_min_trials=_get_int_env("GRAND_AIR_ROI_PROMOTE_MIN_TRIALS", 4),
    roi_promote_threshold=_get_float_env("GRAND_AIR_ROI_PROMOTE_THRESHOLD", 0.08),
    roi_promote_weight=_get_float_env("GRAND_AIR_ROI_PROMOTE_WEIGHT", 0.30),
    tree_diffuse_skip=_get_float_env("GRAND_AIR_TREE_DIFFUSE_SKIP", 0.92),
    tree_skip_union_size=_get_int_env("GRAND_AIR_TREE_SKIP_UNION_SIZE", 448),
    tree_skip_promise=_get_float_env("GRAND_AIR_TREE_SKIP_PROMISE", -0.12),
    tree_skip_block_concentration=_get_float_env("GRAND_AIR_TREE_SKIP_BLOCK_CONCENTRATION", 0.06),
    tree_tiny_compactness=_get_float_env("GRAND_AIR_TREE_TINY_COMPACTNESS", 0.16),
    tree_tiny_uncertainty=_get_float_env("GRAND_AIR_TREE_TINY_UNCERTAINTY", 0.20),
    tree_tiny_block_concentration=_get_float_env("GRAND_AIR_TREE_TINY_BLOCK_CONCENTRATION", 0.05),
    tree_full_compactness=_get_float_env("GRAND_AIR_TREE_FULL_COMPACTNESS", 0.24),
    tree_full_uncertainty=_get_float_env("GRAND_AIR_TREE_FULL_UNCERTAINTY", 0.20),
    tree_full_promise=_get_float_env("GRAND_AIR_TREE_FULL_PROMISE", -0.05),
    tree_full_block_concentration=_get_float_env("GRAND_AIR_TREE_FULL_BLOCK_CONCENTRATION", 0.07),
    tree_full_conflict=_get_float_env("GRAND_AIR_TREE_FULL_CONFLICT", 0.06),
    tree_meta_compactness=_get_float_env("GRAND_AIR_TREE_META_COMPACTNESS", 0.38),
    tree_meta_conflict=_get_float_env("GRAND_AIR_TREE_META_CONFLICT", 0.09),
    tree_meta_promise=_get_float_env("GRAND_AIR_TREE_META_PROMISE", 0.02),
    tree_meta_block_concentration=_get_float_env("GRAND_AIR_TREE_META_BLOCK_CONCENTRATION", 0.10),
    tree_tiny_rescue_compactness=_get_float_env("GRAND_AIR_TREE_TINY_RESCUE_COMPACTNESS", 0.12),
    tree_tiny_rescue_uncertainty=_get_float_env("GRAND_AIR_TREE_TINY_RESCUE_UNCERTAINTY", 0.22),
    tree_promote_margin=_get_float_env("GRAND_AIR_TREE_PROMOTE_MARGIN", 0.05),
    tree_drop_margin=_get_float_env("GRAND_AIR_TREE_DROP_MARGIN", 0.06),
    tree_late_snapshot_penalty=_get_float_env("GRAND_AIR_TREE_LATE_SNAPSHOT_PENALTY", 0.18),
    tree_late_stop_progress=_get_float_env("GRAND_AIR_TREE_LATE_STOP_PROGRESS", 0.01),
    tree_late_stop_compactness=_get_float_env("GRAND_AIR_TREE_LATE_STOP_COMPACTNESS", 0.30),
    tree_late_stop_block_concentration=_get_float_env("GRAND_AIR_TREE_LATE_STOP_BLOCK_CONCENTRATION", 0.08),
    probe_min_syndrome_drop_full=_get_float_env("GRAND_AIR_PROBE_MIN_DROP_FULL", 0.08),
    probe_min_syndrome_drop_continue=_get_float_env("GRAND_AIR_PROBE_MIN_DROP_CONTINUE", 0.04),
    probe_min_syndrome_drop_hard=_get_float_env("GRAND_AIR_PROBE_MIN_DROP_HARD", 0.12),
    probe_next_snapshot_min_drop=_get_float_env("GRAND_AIR_PROBE_NEXT_MIN_DROP", 0.03),
    probe_local_compactness=_get_float_env("GRAND_AIR_PROBE_LOCAL_COMPACTNESS", 0.16),
    probe_local_block_concentration=_get_float_env("GRAND_AIR_PROBE_LOCAL_BLOCK_CONCENTRATION", 0.07),
    probe_local_conflict=_get_float_env("GRAND_AIR_PROBE_LOCAL_CONFLICT", 0.06),
    probe_local_promise=_get_float_env("GRAND_AIR_PROBE_LOCAL_PROMISE", -0.02),
    probe_strong_compactness=_get_float_env("GRAND_AIR_PROBE_STRONG_COMPACTNESS", 0.26),
    probe_strong_block_concentration=_get_float_env("GRAND_AIR_PROBE_STRONG_BLOCK_CONCENTRATION", 0.10),
    probe_strong_conflict=_get_float_env("GRAND_AIR_PROBE_STRONG_CONFLICT", 0.09),
    probe_meta_drop=_get_float_env("GRAND_AIR_PROBE_META_DROP", 0.16),
    probe_failrate_easy=_get_float_env("GRAND_AIR_PROBE_FAILRATE_EASY", 0.018),
    probe_failrate_hard=_get_float_env("GRAND_AIR_PROBE_FAILRATE_HARD", 0.080),
    probe_failrate_min_frames=_get_int_env("GRAND_AIR_PROBE_FAILRATE_MIN_FRAMES", 192),
    probe_force_tiny_all_failed=bool(_get_int_env("GRAND_AIR_PROBE_FORCE_TINY_ALL_FAILED", 1)),
    probe_late_snapshot_cap=_get_int_env("GRAND_AIR_PROBE_LATE_SNAPSHOT_CAP", 2),
)


### CELL number 29 ###
from dataclasses import dataclass
from typing import Dict, Any, Optional

MAX_FRAMES_CAP = 1600

@dataclass
class AdaptiveMCConfig:
    """
    Configuration for adaptive Monte-Carlo simulations.

    target_frame_errors:
        Stop when this many frame errors have been observed
        (unless max_frames is hit first).

    min_frames:
        Reserved for future use (no minimum-frame requirement in current runs).

    max_frames:
        Hard cap on number of frames simulated.
    """
    target_frame_errors: int = 200
    min_frames: int = 0
    max_frames: int = MAX_FRAMES_CAP


def run_ldpc_min_sum_adaptive(
    sim_cfg: SimulationConfig,
    dec_cfg: DecoderConfig,
    mc_cfg: AdaptiveMCConfig,
    rng_seed: Optional[int] = None,
    label: Optional[str] = None,
    hw_model: Optional[HardwareTimingModel] = None,
) -> Dict[str, Any]:
    """
    Adaptive LDPC-only Monte-Carlo (channel = sim_cfg.channel.name).

    Hardware timing model:
      - NO CPU wall-time is used.
      - Per-frame LDPC cycles are computed from:
            * iter_used (early stopping / max-iters)
            * code Tanner-graph edge count E
            * throughput parameters in hw_model / environment variables
      - IMPORTANT detail:
            The last VN->CN update pass is charged iff the current *software*
            actually executed it. This occurs when early-stop did not trigger
            (e.g., failure at max_iters, or early_stop=False).
    """
    if hw_model is None:
        hw_model = HW_MODEL

    if rng_seed is None:
        rng_seed = sim_cfg.rng_seed_global + 1234

    global_rng = np.random.default_rng(rng_seed)

    N = sim_cfg.code.N
    max_frames = int(mc_cfg.max_frames)
    target_fe = int(mc_cfg.target_frame_errors)

    per_frame_errs = []
    per_frame_iters = []
    per_frame_hw_cycles = []
    per_frame_hw_time_us = []

    total_bits = 0
    total_bit_errs = 0
    frame_errors = 0
    total_unsat_checks = 0
    total_iters = 0

    total_hw_cycles = 0

    frame_id = 0
    while True:
        if frame_id >= max_frames:
            break

        frame = run_single_frame(sim_cfg, frame_id, global_rng)

        # ---- LDPC decode ----
        ldpc_min_sum_decoder_frame(frame, sim_cfg, dec_cfg)

        num_err = int(frame.error_positions_final.size)
        unsat = int(frame.syndrome_final.sum())
        it_used = int(frame.iter_used if frame.iter_used is not None else 0)

        # ---- Hardware cycles / time ----
        # If early-stop did NOT trigger, the software executed the last VN->CN pass.
        final_vn2cn_executed = bool((not dec_cfg.early_stop) or (unsat != 0))

        hw_cycles = ldpc_hw_cycles_frame(
            it_used,
            sim_cfg.code,
            hw_model,
            final_vn2cn_executed=final_vn2cn_executed,
        )
        hw_time_us = cycles_to_us(hw_cycles, hw_model)

        # Accumulate
        per_frame_errs.append(num_err)
        per_frame_iters.append(it_used)
        per_frame_hw_cycles.append(hw_cycles)
        per_frame_hw_time_us.append(hw_time_us)

        total_bits += int(N)
        total_bit_errs += num_err
        total_unsat_checks += unsat
        total_iters += it_used
        total_hw_cycles += hw_cycles

        if num_err > 0:
            frame_errors += 1

        frame_id += 1

        # Stopping condition
        if frame_errors >= target_fe:
            break

    num_frames = frame_id
    if num_frames == 0:
        return {
            "ber": 0.0,
            "fer": 0.0,
            "avg_unsat_checks": 0.0,
            "avg_iters": 0.0,
            "num_frames": 0,
            "frame_errors": 0,
            "per_frame_errs": np.array([], dtype=np.int32),
            "per_frame_iters": np.array([], dtype=np.int32),
            "per_frame_hw_cycles": np.array([], dtype=np.int64),
            "per_frame_hw_time_us": np.array([], dtype=np.float64),
            "avg_hw_cycles_per_frame": 0.0,
            "avg_hw_time_us_per_frame": 0.0,
        }

    ber = total_bit_errs / total_bits
    fer = frame_errors / num_frames
    avg_unsat_checks = total_unsat_checks / num_frames
    avg_iters = total_iters / num_frames

    avg_hw_cycles_per_frame = total_hw_cycles / num_frames
    avg_hw_time_us_per_frame = cycles_to_us(avg_hw_cycles_per_frame, hw_model)

    if label is None:
        label = (
            f"LDPC-only, max_iters={dec_cfg.max_iters}, "
            f"early_stop={dec_cfg.early_stop}"
        )

    print(f"\n=== Adaptive LDPC-only Monte-Carlo (channel={sim_cfg.channel.name}) ===")
    print(f"Decoder label                 : {label}")
    print(f"SNR (dB)                      : {sim_cfg.channel.snr_db:.2f}")
    print(f"Frames simulated              : {num_frames}")
    print(f"Frame errors                  : {frame_errors} (target={target_fe})")
    print(f"Bit error rate (BER)          : {ber:.3e}")
    print(f"Frame error rate (FER)        : {fer:.3e}")
    print(f"Avg LDPC iterations/frame     : {avg_iters:.2f}")
    print(f"Avg HW cycles/frame           : {avg_hw_cycles_per_frame:.2f}")
    print(f"Avg HW decode time/frame      : {avg_hw_time_us_per_frame:.2f} µs")
    print(f"HW model (fclk_mhz)            : {hw_model.fclk_mhz:.1f} MHz")

    return {
        "ber": float(ber),
        "fer": float(fer),
        "avg_unsat_checks": float(avg_unsat_checks),
        "avg_iters": float(avg_iters),
        "num_frames": int(num_frames),
        "frame_errors": int(frame_errors),
        "per_frame_errs": np.array(per_frame_errs, dtype=np.int32),
        "per_frame_iters": np.array(per_frame_iters, dtype=np.int32),
        "per_frame_hw_cycles": np.array(per_frame_hw_cycles, dtype=np.int64),
        "per_frame_hw_time_us": np.array(per_frame_hw_time_us, dtype=np.float64),
        "avg_hw_cycles_per_frame": float(avg_hw_cycles_per_frame),
        "avg_hw_time_us_per_frame": float(avg_hw_time_us_per_frame),
        "hw_model": asdict(hw_model),
    }






### CELL number 30 ###
from typing import Dict, Any, Optional, Sequence

# Attributes copied from a failed primary stage-2 attempt into the optional boosted
# retry result. Keep this list strictly *metadata-only*.
#
# Rationale: the primary and boost attempts are both appended to the `attempts` list and
# later aggregated by `_aggregate_stage2_attempts()`. Copying per-attempt work counters
# (restart counts, peel/chase/OSD work, etc.) into the boosted result causes those costs
# to be counted twice whenever the boost path runs. That inflates diagnostics and hardware
# time accounting without changing the actual decoder behavior.
_STAGE2_BOOST_COPY_ATTRS = [
    "pre_solver_mode_used",
    "pre_solver_attempted",
    "pre_solver_success",
    "llr_source_used",
    "selection_mode_used",
]

# Per-attempt quantities that should accumulate across repeated snapshot rescues.
_STAGE2_SUM_ATTRS = [
    "patterns_tested",
    "patterns_evaluated",
    "total_v2c_edge_visits",
    "total_v2c_edge_visits_evaluated",
    "total_unique_checks_visited",
    "total_unique_checks_toggled",
    "total_unique_checks_toggled_evaluated",
    "patterns_generated",
    "num_batches_evaluated",
    "positions_packed_evaluated",
    "llr_sort_len",
    "search_size",
    "sum_pattern_weights_generated",
    "cluster_unsat_edges",
    "cluster_pair_edges",
    "peel_candidate_size",
    "peel_residual_vars",
    "peel_residual_rows",
    "peel_edge_work",
    "peel_dense_xor_ops",
    "peel_free_dim",
    "peel_extra_llr_bits",
    "peel_extra_llr_added",
    "chase_candidate_size",
    "chase_core_size",
    "chase_patterns_considered",
    "chase_candidates_tested",
    "chase_score_edge_visits",
    "chase_score_checks_toggled",
    "chase_score_sum_pattern_weights",
    "chase_ldpc_total_iters",
    "chase_ldpc_num_runs",
    "chase_ldpc_num_nonconverged",
    "osd_candidate_size",
    "osd_matrix_rows",
    "osd_free_dim",
    "osd_enum_bits_used",
    "osd_basis_xor_ops",
    "osd_candidates_considered",
    "osd_candidates_tested",
    "osd_sum_candidate_weights",
    "restart_num_runs",
    "restart_total_ldpc_iters",
    "restart_num_nonconverged",
    "restart_anchor_bits_total",
    "sv_seeded_count",
    "sv_neighbor_visits",
    "sv_score_len",
    "disagreement_added",
]

# Frame-level binary indicators; keep them binary even if several snapshots are attempted.
_STAGE2_MAX_ATTRS = [
    "pre_solver_attempted",
    "pre_solver_success",
]


def _normalize_snapshot_schedule(snapshot_iter: Any) -> List[int]:
    """Normalize a stage-2 snapshot specification into a sorted unique list."""
    if isinstance(snapshot_iter, np.ndarray):
        vals = [int(x) for x in snapshot_iter.reshape(-1).tolist()]
    elif isinstance(snapshot_iter, (list, tuple)):
        vals = [int(x) for x in snapshot_iter]
    else:
        vals = [int(snapshot_iter)]

    vals = sorted(set(int(v) for v in vals if int(v) > 0))
    if not vals:
        raise ValueError("snapshot_iter must contain at least one positive iteration")
    return vals



def _resolve_grand_snapshot_schedule(stage1_iter: int) -> List[int]:
    """Resolve which LDPC snapshots the hybrid stage-2 should probe.

    Environment variable:
      GRAND_RESCUE_SNAPSHOT_ITERS
        Comma-separated tokens chosen from integers and/or one of
        {stage1, final, last, max, it}. Any integers above stage1_iter are ignored.

    Default schedule intentionally probes early/mid snapshots before the final one,
    because many structured residuals are easier to rescue before LDPC hardens around
    a wrong basin. The final stage-1 snapshot is always appended.
    """
    stage1_iter = int(stage1_iter)
    raw = str(os.environ.get("GRAND_RESCUE_SNAPSHOT_ITERS", "")).strip()
    if not raw:
        raw = "4,8,12,15,20,40,60,80,stage1"

    vals: List[int] = []
    for tok in raw.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t in ("stage1", "final", "last", "max", "it", "iter", "stage1_iter"):
            vals.append(stage1_iter)
            continue
        try:
            v = int(float(t))
        except Exception:
            continue
        if 0 < v <= stage1_iter:
            vals.append(v)

    append_final = bool(_get_int_env("GRAND_RESCUE_APPEND_FINAL", 1))
    if append_final:
        vals.append(stage1_iter)
    vals = sorted(set(int(v) for v in vals if 0 < int(v) <= stage1_iter))
    if not vals:
        vals = [stage1_iter]
    return vals



def _copy_missing_result_attrs(dst: ClusterGrandResult,
                               src: Optional[ClusterGrandResult],
                               attrs: Sequence[str]) -> None:
    if src is None:
        return
    for attr in attrs:
        if not hasattr(src, attr):
            continue
        src_val = getattr(src, attr)
        cur = getattr(dst, attr, None)
        should_copy = (not hasattr(dst, attr)) or (cur is None)
        if not should_copy:
            if isinstance(cur, (int, np.integer, float, np.floating)) and float(cur) == 0.0:
                should_copy = True
            elif isinstance(cur, str) and cur in ("", "none"):
                should_copy = True
            elif isinstance(cur, np.ndarray) and isinstance(src_val, np.ndarray) and cur.size == 0 and src_val.size > 0:
                should_copy = True
        if should_copy:
            setattr(dst, attr, src_val)



def _grand_attempt_exhausted(res: ClusterGrandResult, cfg: ClusterGrandConfig) -> bool:
    pt = int(getattr(res, "patterns_tested", 0))
    pg = int(getattr(res, "patterns_generated", 0))
    cap = int(getattr(cfg, "max_patterns", pt))
    return (pg > 0) and (pt >= min(pg, cap))



def _run_stage2_single_snapshot(
    frame: FrameLog,
    sim_cfg: SimulationConfig,
    snapshot_iter: int,
    grand_cfg: ClusterGrandConfig,
    grand_cfg_boost: Optional[ClusterGrandConfig] = None,
) -> Tuple[Optional[ClusterGrandResult], List[ClusterGrandResult]]:
    """Run one snapshot rescue, optionally followed by a boosted retry."""
    attempts: List[ClusterGrandResult] = []
    try:
        res = run_local_rescue_with_optional_presolver(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=int(snapshot_iter),
            cfg=grand_cfg,
        )
    except Exception as e:
        print(f"[WARN] GRAND failed to run on frame {frame.frame_id} at snap={snapshot_iter}: {e}")
        return None, attempts

    if res is None:
        return None, attempts

    attempts.append(res)

    if (not bool(res.success)) and (grand_cfg_boost is not None) and _grand_attempt_exhausted(res, grand_cfg):
        try:
            res2 = run_local_rescue_with_optional_presolver(
                frame=frame,
                sim_cfg=sim_cfg,
                snapshot_iter=int(snapshot_iter),
                cfg=grand_cfg_boost,
            )
        except Exception as e:
            print(f"[WARN] BOOST GRAND failed on frame {frame.frame_id} at snap={snapshot_iter}: {e}")
            res2 = None

        if res2 is not None:
            _copy_missing_result_attrs(res2, res, _STAGE2_BOOST_COPY_ATTRS)
            attempts.append(res2)
            res = res2

    return res, attempts



def _aggregate_stage2_attempts(
    final_res: ClusterGrandResult,
    attempts: Sequence[ClusterGrandResult],
    snapshot_schedule: Sequence[int],
    snapshot_attempts_count: int,
    snapshot_success_iter: int,
    snapshot_last_iter: int,
) -> ClusterGrandResult:
    """Aggregate counters across repeated stage-2 invocations.

    The returned object keeps success/failure and final error counts from the *last*
    attempt that mattered, but accumulates stage-2 work counters across all attempts.
    """
    agg = copy.deepcopy(final_res)

    for attr in _STAGE2_SUM_ATTRS:
        total = 0
        for res in attempts:
            try:
                total += int(getattr(res, attr, 0) or 0)
            except Exception:
                pass
        setattr(agg, attr, total)

    for attr in _STAGE2_MAX_ATTRS:
        val = 0
        for res in attempts:
            try:
                val = max(val, int(getattr(res, attr, 0) or 0))
            except Exception:
                pass
        setattr(agg, attr, val)

    setattr(agg, "snapshot_attempts_count", int(snapshot_attempts_count))
    setattr(agg, "snapshot_success_iter", int(snapshot_success_iter))
    setattr(agg, "snapshot_last_iter", int(snapshot_last_iter))
    setattr(agg, "snapshot_schedule_used", np.asarray(list(snapshot_schedule), dtype=np.int32))
    return agg



def run_hybrid_ldpc_grand_adaptive(
    sim_cfg: SimulationConfig,
    dec_cfg_stage1: DecoderConfig,
    grand_cfg: ClusterGrandConfig,
    snapshot_iter: Any,
    mc_cfg: AdaptiveMCConfig,
    rng_seed: Optional[int] = None,
    label: Optional[str] = None,
    hw_model: Optional[HardwareTimingModel] = None,
    grand_cfg_boost: Optional[ClusterGrandConfig] = None,
    grand_cfg_fallback: Optional[ClusterGrandConfig] = None,
    grand_cfg_boost_fallback: Optional[ClusterGrandConfig] = None,
    fallback_label: Optional[str] = None,
    ai_gate_cfg: Optional[AIGatedHybridConfig] = None,
    grand_cfg_tiny: Optional[ClusterGrandConfig] = None,
    grand_cfg_boost_tiny: Optional[ClusterGrandConfig] = None,
) -> Dict[str, Any]:
    """
    Hybrid decoder (two-stage):

      Stage-1: LDPC (normalized min-sum) with early stopping, up to max_iters.
      Stage-2: If stage-1 does NOT converge (syndrome != 0),
               run GRAND over one or more LDPC snapshots. Earlier snapshots are
               often easier to rescue because the residual is still localized, so
               the schedule can probe several iterations before the final stage-1
               snapshot.

    Hardware timing model:
      - NO CPU wall-time is used.
      - Stage-1 cycles are computed from iter_used and the code edge count.
        The last VN->CN pass is charged iff the software executed it.
      - Stage-2 cycles are summed exactly across repeated snapshot tries and boost tries.
    """
    snapshot_schedule = _normalize_snapshot_schedule(snapshot_iter)
    primary_snapshot_iter = int(snapshot_schedule[-1])

    if hw_model is None:
        hw_model = HW_MODEL

    if grand_cfg_boost is None and ("GRAND_USE_BOOST" in globals()) and GRAND_USE_BOOST and ("grand_cfg_awgn_boost" in globals()):
        grand_cfg_boost = grand_cfg_awgn_boost

    if label is None:
        snap_desc = str(primary_snapshot_iter) if len(snapshot_schedule) == 1 else ",".join(str(x) for x in snapshot_schedule)
        label = f"hyb: LDPC({dec_cfg_stage1.max_iters})+GRAND (snap={snap_desc})"

    if fallback_label is None:
        fallback_label = str(getattr(grand_cfg_fallback, "pre_solver_mode", "fallback")) if grand_cfg_fallback is not None else ""

    if rng_seed is None:
        rng_seed = sim_cfg.rng_seed_global + 900 + primary_snapshot_iter

    rng = np.random.default_rng(rng_seed)
    ai_gate_state = AIGatedHybridState(n_features=6, cfg=ai_gate_cfg) if ai_gate_cfg is not None else None

    N = int(sim_cfg.code.N)
    max_frames = int(mc_cfg.max_frames)
    target_fe = int(mc_cfg.target_frame_errors)

    total_bit_errs_stage1 = 0
    frame_errs_stage1 = 0
    total_iters_stage1 = 0

    total_bit_errs_after = 0
    frame_errs_after = 0

    total_hw_cycles_stage1 = 0
    total_hw_cycles_grand = 0
    total_hw_cycles_total = 0

    per_frame_iters_stage1 = []
    per_frame_stage1_failed = []

    per_frame_hw_cycles_stage1 = []
    per_frame_hw_cycles_grand = []
    per_frame_hw_cycles_total = []

    per_frame_patterns_tested = []
    per_frame_patterns_evaluated = []
    per_frame_grand_edge_visits_eval = []
    per_frame_grand_checks_toggled_eval = []
    per_frame_grand_num_batches = []
    per_frame_grand_llr_sort_len = []
    per_frame_grand_search_size = []
    per_frame_grand_patterns_generated = []
    per_frame_grand_sumw_generated = []
    per_frame_grand_positions_packed = []
    per_frame_cluster_unsat_edges = []
    per_frame_cluster_pair_edges = []

    per_frame_pre_solver_attempted = []
    per_frame_pre_solver_success = []
    per_frame_peel_candidate_size = []
    per_frame_peel_residual_vars = []
    per_frame_peel_dense_xor_ops = []

    per_frame_chase_candidate_size = []
    per_frame_chase_candidates_tested = []
    per_frame_chase_total_ldpc_iters = []

    per_frame_osd_candidate_size = []
    per_frame_osd_candidates_tested = []
    per_frame_osd_free_dim = []
    per_frame_restart_num_runs = []
    per_frame_restart_total_ldpc_iters = []
    per_frame_restart_anchor_bits_total = []
    per_frame_disagreement_added = []

    per_frame_snapshot_attempts = []
    per_frame_snapshot_success_iter = []
    per_frame_snapshot_last_iter = []

    per_frame_bit_errors_stage1 = []
    per_frame_bit_errors_after = []
    per_frame_stage2_improved = []
    per_frame_stage2_true_fix = []
    per_frame_stage1_syndrome_weight = []
    per_frame_stage1_error_span = []
    per_frame_stage1_error_runs = []
    per_frame_stage1_block_concentration = []
    per_frame_stage2_success_profile = []
    per_frame_stage2_invoked = []
    per_frame_ai_gate_action = []
    per_frame_ai_gate_first_action = []
    per_frame_ai_gate_confidence = []
    per_frame_ai_gate_promise = []
    per_frame_ai_gate_snapshot = []
    per_frame_ai_gate_decision_count = []
    per_frame_ai_gate_escalated = []
    per_frame_ai_gate_leaf = []
    per_frame_probe_invoked = []
    per_frame_probe_success = []
    per_frame_probe_syndrome_drop = []
    per_frame_probe_escalated = []
    per_frame_probe_regime = []

    diag_block_size = max(1, int(float(os.getenv("SIONNA_CSI_BLOCK_SC", os.getenv("SIONNA_CSI_PILOT_STRIDE", "1")) or 1)))

    n_frames = 0
    frame_id = 0

    while True:
        if frame_id >= max_frames:
            break
        if frame_errs_after >= target_fe:
            break

        frame = run_single_frame(sim_cfg, frame_id, rng)
        ldpc_min_sum_decoder_frame(frame, sim_cfg, dec_cfg_stage1)

        it1 = int(frame.iter_used if frame.iter_used is not None else 0)
        total_iters_stage1 += it1
        per_frame_iters_stage1.append(it1)

        be1 = int(frame.error_positions_final.size)
        total_bit_errs_stage1 += be1
        if be1 > 0:
            frame_errs_stage1 += 1

        syn_w = int(frame.syndrome_final.sum())
        stage1_failed = (syn_w != 0)
        per_frame_stage1_failed.append(bool(stage1_failed))

        stage1_err_pos = np.asarray(frame.error_positions_final, dtype=np.int32).reshape(-1)
        stage1_err_span = _diag_error_span(stage1_err_pos)
        stage1_err_runs = _diag_error_runs(stage1_err_pos)
        stage1_block_conc = _diag_block_concentration(stage1_err_pos, diag_block_size)

        final_vn2cn_executed_stage1 = bool((not dec_cfg_stage1.early_stop) or (syn_w != 0))
        hw_c_stage1 = ldpc_hw_cycles_frame(
            it1,
            sim_cfg.code,
            hw_model,
            final_vn2cn_executed=final_vn2cn_executed_stage1,
        )

        hw_c_grand = 0
        be_after = be1

        pt = 0
        pe = 0
        evis = 0
        ctog = 0
        nb = 0
        llrs = 0
        ss = 0
        pg = 0
        sw = 0
        posp = 0
        cu_e = 0
        cu_p = 0

        ps_attempt = 0
        ps_success = 0
        peel_cand = 0
        peel_res_vars = 0
        peel_xor = 0

        chase_cand = 0
        chase_tested = 0
        chase_ldpc_iters = 0

        osd_cand = 0
        osd_tested = 0
        osd_free = 0
        restart_runs = 0
        restart_iters = 0
        restart_anchor_bits = 0
        disagree_added = 0

        snapshot_attempts = 0
        snapshot_success_iter = 0
        snapshot_last_iter = 0
        stage2_success_profile = "stage1"
        stage2_invoked = False
        ai_gate_action = "none"
        ai_gate_first_action = "none"
        ai_gate_confidence = np.nan
        ai_gate_promise = np.nan
        ai_gate_snapshot = 0
        ai_gate_decision_count = 0
        ai_gate_escalated = 0
        ai_gate_leaf = "none"

        if stage1_failed:
            attempt_results: List[ClusterGrandResult] = []
            res_final: Optional[ClusterGrandResult] = None
            x_gate = None
            gate_meta = None
            prev_gate_meta = None
            action_rank = {"none": -1, "skip": 0, "tiny": 1, "full": 2, "meta": 3}
            probe_invoked = 0
            probe_success = 0
            probe_drop_value = 0.0
            probe_escalated = 0
            probe_regime = "none"
            frames_seen_so_far = int(n_frames) + 1
            stage1_fail_rate_est = float(frame_errs_stage1) / float(max(1, frames_seen_so_far))

            if (ai_gate_cfg is not None) and (grand_cfg_tiny is not None):
                policy_mode = str(getattr(ai_gate_cfg, "policy_mode", "linear_ucb") or "linear_ucb").strip().lower()
                dynamic_gate = bool(getattr(ai_gate_cfg, "dynamic_per_snapshot", True))
                gate_schedule = list(snapshot_schedule)
                if not dynamic_gate:
                    gate_schedule = [int(_ai_gate_select_snapshot(snapshot_schedule, ai_gate_cfg))]

                if policy_mode in ("probe_moe_roi_fix", "probe_moe_roi", "probe_moe", "probe_fix", "probe"):
                    active_schedule_for_res = list(gate_schedule)
                    for snap_pos, snap in enumerate(gate_schedule, start=1):
                        snap_i = int(snap)
                        x_gate, gate_meta = _extract_ai_gate_context(frame, sim_cfg, snap_i, ai_gate_cfg)
                        ai_gate_snapshot = int(snap_i) if ai_gate_snapshot == 0 else ai_gate_snapshot
                        ai_gate_first_action = "tiny" if ai_gate_decision_count == 0 else ai_gate_first_action
                        ai_gate_decision_count += 1
                        ai_gate_promise = float(gate_meta.get("promise", np.nan))
                        ai_gate_confidence = float(np.clip(0.55 + 0.35 * max(0.0, float(gate_meta.get("compactness", 0.0))), 0.0, 1.0))

                        # Always run the tiny probe on failed frames for this policy.
                        stage2_invoked = True
                        probe_invoked = 1
                        snapshot_attempts += 1
                        snapshot_last_iter = int(snap_i)

                        res_probe, probe_attempts = _run_stage2_single_snapshot(
                            frame=frame,
                            sim_cfg=sim_cfg,
                            snapshot_iter=snap_i,
                            grand_cfg=grand_cfg_tiny,
                            grand_cfg_boost=grand_cfg_boost_tiny,
                        )
                        for res_try in probe_attempts:
                            setattr(res_try, "snapshot_iter_used", snap_i)
                            setattr(res_try, "stage2_profile_name", "probe_tiny")
                            hw_c_grand += grand_hw_cycles_from_result(res_try, sim_cfg, hw_model)
                        attempt_results.extend(probe_attempts)

                        if res_probe is not None:
                            res_final = res_probe
                            setattr(res_final, "stage2_profile_name", "probe_tiny")
                            probe_drop_value = _stage2_syndrome_drop_ratio(res_probe)
                            if bool(res_probe.success):
                                probe_success = 1
                                ai_gate_action = "tiny"
                                ai_gate_leaf = "probe_success"
                                snapshot_success_iter = snap_i
                                stage2_success_profile = "probe_tiny"
                                break

                        probe_regime, allow_full, allow_meta, allow_next, plan_reason = _ai_probe_plan(
                            meta=gate_meta,
                            stage1_fail_rate_est=stage1_fail_rate_est,
                            frames_seen=frames_seen_so_far,
                            snapshot_pos=snap_pos,
                            probe_drop=probe_drop_value,
                            cfg=ai_gate_cfg,
                        )
                        ai_gate_leaf = f"probe_{probe_regime}_{plan_reason}"
                        ai_gate_action = "tiny"

                        if allow_full or allow_meta:
                            probe_escalated = 1
                            ai_gate_escalated = 1
                            ai_gate_action = "full"
                            stage2_profiles = [("probe_full", grand_cfg, grand_cfg_boost)]
                            if allow_meta and grand_cfg_fallback is not None:
                                stage2_profiles.append((fallback_label or str(getattr(grand_cfg_fallback, "pre_solver_mode", "fallback")), grand_cfg_fallback, grand_cfg_boost_fallback))

                            for profile_name, profile_cfg, profile_boost in stage2_profiles:
                                res_snap, res_attempts = _run_stage2_single_snapshot(
                                    frame=frame,
                                    sim_cfg=sim_cfg,
                                    snapshot_iter=snap_i,
                                    grand_cfg=profile_cfg,
                                    grand_cfg_boost=profile_boost,
                                )
                                for res_try in res_attempts:
                                    setattr(res_try, "snapshot_iter_used", snap_i)
                                    setattr(res_try, "stage2_profile_name", profile_name)
                                    hw_c_grand += grand_hw_cycles_from_result(res_try, sim_cfg, hw_model)
                                attempt_results.extend(res_attempts)
                                if res_snap is not None:
                                    res_final = res_snap
                                    setattr(res_final, "stage2_profile_name", profile_name)
                                    if bool(res_snap.success):
                                        snapshot_success_iter = snap_i
                                        stage2_success_profile = profile_name
                                        break
                            if stage2_success_profile not in ("stage1", "skip"):
                                break

                        if not allow_next:
                            stage2_success_profile = "probe_stop" if stage2_success_profile == "stage1" else stage2_success_profile
                            break

                    # Use the actual schedule we attempted for aggregation.
                    active_schedule_for_res = np.asarray(list(gate_schedule), dtype=np.int32)
                else:
                    for snap_pos, snap in enumerate(gate_schedule, start=1):
                        snap_i = int(snap)
                        x_gate, gate_meta = _extract_ai_gate_context(frame, sim_cfg, snap_i, ai_gate_cfg)
                        allowed_actions = _ai_gate_allowed_actions(gate_meta, ai_gate_cfg, snapshot_pos=snap_pos)
                        base_scores, gate_leaf = _ai_gate_base_scores(gate_meta, ai_gate_cfg, snapshot_pos=snap_pos, prev_meta=prev_gate_meta)
                        if ai_gate_state is not None:
                            gate_action, gate_confidence, _ = ai_gate_state.choose(x_gate, base_scores, allowed_actions, ai_gate_cfg)
                        else:
                            gate_action = max(allowed_actions, key=lambda a: base_scores.get(a, -1e9)) if allowed_actions else "skip"
                            gate_confidence = 0.0

                        if ai_gate_decision_count == 0:
                            ai_gate_first_action = str(gate_action)
                            ai_gate_snapshot = int(snap_i)
                        elif action_rank.get(str(gate_action), -1) > action_rank.get(str(ai_gate_action), -1):
                            ai_gate_escalated = 1

                        if action_rank.get(str(gate_action), -1) >= action_rank.get(str(ai_gate_action), -1):
                            ai_gate_action = str(gate_action)
                            ai_gate_confidence = float(gate_confidence)
                            ai_gate_promise = float(gate_meta.get("promise", np.nan))
                            ai_gate_leaf = str(gate_leaf)

                        ai_gate_decision_count += 1
                        prev_gate_meta = dict(gate_meta)

                        if str(gate_action) == "skip":
                            stage2_success_profile = "skip"
                            snapshot_last_iter = int(snap_i)
                            continue

                        stage2_invoked = True
                        snapshot_attempts += 1
                        snapshot_last_iter = int(snap_i)
                        if str(gate_action) == "tiny":
                            stage2_profiles = [("ai_tiny", grand_cfg_tiny, grand_cfg_boost_tiny)]
                        elif str(gate_action) == "meta":
                            stage2_profiles = [("ai_full", grand_cfg, grand_cfg_boost)]
                            if grand_cfg_fallback is not None:
                                stage2_profiles.append((fallback_label or str(getattr(grand_cfg_fallback, "pre_solver_mode", "fallback")), grand_cfg_fallback, grand_cfg_boost_fallback))
                        else:
                            stage2_profiles = [("ai_full", grand_cfg, grand_cfg_boost)]

                        for profile_name, profile_cfg, profile_boost in stage2_profiles:
                            res_snap, res_attempts = _run_stage2_single_snapshot(
                                frame=frame,
                                sim_cfg=sim_cfg,
                                snapshot_iter=snap_i,
                                grand_cfg=profile_cfg,
                                grand_cfg_boost=profile_boost,
                            )
                            for res_try in res_attempts:
                                setattr(res_try, "snapshot_iter_used", snap_i)
                                setattr(res_try, "stage2_profile_name", profile_name)
                                hw_c_grand += grand_hw_cycles_from_result(res_try, sim_cfg, hw_model)
                            attempt_results.extend(res_attempts)
                            if res_snap is not None:
                                res_final = res_snap
                                setattr(res_final, "stage2_profile_name", profile_name)
                                if bool(res_snap.success):
                                    snapshot_success_iter = snap_i
                                    stage2_success_profile = profile_name
                                    break
                        if stage2_success_profile not in ("stage1", "skip"):
                            break

                    active_schedule_for_res = gate_schedule if gate_schedule else list(snapshot_schedule)
            else:
                stage2_profiles = [(str(getattr(grand_cfg, "pre_solver_mode", "primary")), grand_cfg, grand_cfg_boost)]
                active_schedule_for_res = list(snapshot_schedule)
                if grand_cfg_fallback is not None:
                    stage2_profiles.append((fallback_label or str(getattr(grand_cfg_fallback, "pre_solver_mode", "fallback")), grand_cfg_fallback, grand_cfg_boost_fallback))

                for profile_name, profile_cfg, profile_boost in stage2_profiles:
                    for snap in active_schedule_for_res:
                        snap_i = int(snap)
                        stage2_invoked = True
                        snapshot_attempts += 1
                        snapshot_last_iter = snap_i

                        res_snap, res_attempts = _run_stage2_single_snapshot(
                            frame=frame,
                            sim_cfg=sim_cfg,
                            snapshot_iter=snap_i,
                            grand_cfg=profile_cfg,
                            grand_cfg_boost=profile_boost,
                        )

                        for res_try in res_attempts:
                            setattr(res_try, "snapshot_iter_used", snap_i)
                            setattr(res_try, "stage2_profile_name", profile_name)
                            hw_c_grand += grand_hw_cycles_from_result(res_try, sim_cfg, hw_model)
                        attempt_results.extend(res_attempts)

                        if res_snap is not None:
                            res_final = res_snap
                            setattr(res_final, "stage2_profile_name", profile_name)
                            if bool(res_snap.success):
                                snapshot_success_iter = snap_i
                                stage2_success_profile = profile_name
                                break
                    if stage2_success_profile not in ("stage1", "skip"):
                        break

            res = None
            if (res_final is not None) and attempt_results:
                res = _aggregate_stage2_attempts(
                    final_res=res_final,
                    attempts=attempt_results,
                    snapshot_schedule=np.asarray(active_schedule_for_res if active_schedule_for_res is not None else snapshot_schedule, dtype=np.int32),
                    snapshot_attempts_count=snapshot_attempts,
                    snapshot_success_iter=snapshot_success_iter,
                    snapshot_last_iter=snapshot_last_iter,
                )
            elif res_final is not None:
                res = res_final

            if res is not None:
                setattr(res, "stage2_profile_name", stage2_success_profile if stage2_success_profile != "stage1" else str(getattr(res, "stage2_profile_name", getattr(grand_cfg, "pre_solver_mode", "primary"))))
                if hw_c_grand == 0:
                    hw_c_grand = grand_hw_cycles_from_result(res, sim_cfg, hw_model)

                if bool(res.success):
                    be_after = int(res.final_bit_errors)
                else:
                    be_after = be1

                pt = int(getattr(res, "patterns_tested", 0))
                pe = int(getattr(res, "patterns_evaluated", pt))
                evis = int(getattr(res, "total_v2c_edge_visits_evaluated", getattr(res, "total_v2c_edge_visits", 0)))
                ctog = int(getattr(res, "total_unique_checks_toggled_evaluated", getattr(res, "total_unique_checks_toggled", 0)))
                nb = int(getattr(res, "num_batches_evaluated", 0))
                llrs = int(getattr(res, "llr_sort_len", 0))
                ss = int(getattr(res, "search_size", 0))
                pg = int(getattr(res, "patterns_generated", 0))
                sw = int(getattr(res, "sum_pattern_weights_generated", 0))
                posp = int(getattr(res, "positions_packed_evaluated", 0))
                cu_e = int(getattr(res, "cluster_unsat_edges", 0))
                cu_p = int(getattr(res, "cluster_pair_edges", 0))
                ps_attempt = int(getattr(res, "pre_solver_attempted", 0))
                ps_success = int(getattr(res, "pre_solver_success", 0))
                peel_cand = int(getattr(res, "peel_candidate_size", 0))
                peel_res_vars = int(getattr(res, "peel_residual_vars", 0))
                peel_xor = int(getattr(res, "peel_dense_xor_ops", 0))
                chase_cand = int(getattr(res, "chase_candidate_size", 0))
                chase_tested = int(getattr(res, "chase_candidates_tested", 0))
                chase_ldpc_iters = int(getattr(res, "chase_ldpc_total_iters", 0))
                osd_cand = int(getattr(res, "osd_candidate_size", 0))
                osd_tested = int(getattr(res, "osd_candidates_tested", 0))
                osd_free = int(getattr(res, "osd_free_dim", 0))
                restart_runs = int(getattr(res, "restart_num_runs", 0))
                restart_iters = int(getattr(res, "restart_total_ldpc_iters", 0))
                restart_anchor_bits = int(getattr(res, "restart_anchor_bits_total", 0))
                disagree_added = int(getattr(res, "disagreement_added", 0))
                snapshot_attempts = int(getattr(res, "snapshot_attempts_count", snapshot_attempts))
                snapshot_success_iter = int(getattr(res, "snapshot_success_iter", snapshot_success_iter))
                snapshot_last_iter = int(getattr(res, "snapshot_last_iter", snapshot_last_iter))
                stage2_success_profile = str(getattr(res, "stage2_profile_name", stage2_success_profile))

            if (ai_gate_state is not None) and (ai_gate_cfg is not None) and (x_gate is not None):
                reward = _ai_gate_reward(
                    action_name=str(ai_gate_action),
                    bit_errors_before=int(be1),
                    bit_errors_after=int(be_after),
                    hw_cycles_grand=int(hw_c_grand),
                    cfg=ai_gate_cfg,
                )
                policy_mode = str(getattr(ai_gate_cfg, "policy_mode", "linear_ucb") or "linear_ucb").strip().lower()
                if policy_mode in ("linear_ucb", "distilled_tree_bandit", "tree_bandit", "dt_bandit", "distilled_tree_roi", "tree_roi", "dt_roi", "probe_moe_roi", "probe_moe", "probe"):
                    ai_gate_state.update(str(ai_gate_action), x_gate, reward)

            per_frame_probe_invoked.append(int(probe_invoked))
            per_frame_probe_success.append(int(probe_success))
            per_frame_probe_syndrome_drop.append(float(probe_drop_value))
            per_frame_probe_escalated.append(int(probe_escalated))
            per_frame_probe_regime.append(str(probe_regime))
        else:
            per_frame_probe_invoked.append(0)
            per_frame_probe_success.append(0)
            per_frame_probe_syndrome_drop.append(0.0)
            per_frame_probe_escalated.append(0)
            per_frame_probe_regime.append("none")
        total_bit_errs_after += be_after
        if be_after > 0:
            frame_errs_after += 1

        hw_c_total = int(hw_c_stage1) + int(hw_c_grand)

        total_hw_cycles_stage1 += int(hw_c_stage1)
        total_hw_cycles_grand += int(hw_c_grand)
        total_hw_cycles_total += int(hw_c_total)

        per_frame_hw_cycles_stage1.append(int(hw_c_stage1))
        per_frame_hw_cycles_grand.append(int(hw_c_grand))
        per_frame_hw_cycles_total.append(int(hw_c_total))

        per_frame_patterns_tested.append(pt)
        per_frame_patterns_evaluated.append(pe)
        per_frame_grand_edge_visits_eval.append(evis)
        per_frame_grand_checks_toggled_eval.append(ctog)
        per_frame_grand_num_batches.append(nb)
        per_frame_grand_llr_sort_len.append(llrs)
        per_frame_grand_search_size.append(ss)
        per_frame_grand_patterns_generated.append(pg)
        per_frame_grand_sumw_generated.append(sw)
        per_frame_grand_positions_packed.append(posp)
        per_frame_cluster_unsat_edges.append(cu_e)
        per_frame_cluster_pair_edges.append(cu_p)
        per_frame_pre_solver_attempted.append(ps_attempt)
        per_frame_pre_solver_success.append(ps_success)
        per_frame_peel_candidate_size.append(peel_cand)
        per_frame_peel_residual_vars.append(peel_res_vars)
        per_frame_peel_dense_xor_ops.append(peel_xor)
        per_frame_chase_candidate_size.append(chase_cand)
        per_frame_chase_candidates_tested.append(chase_tested)
        per_frame_chase_total_ldpc_iters.append(chase_ldpc_iters)
        per_frame_osd_candidate_size.append(osd_cand)
        per_frame_osd_candidates_tested.append(osd_tested)
        per_frame_osd_free_dim.append(osd_free)
        per_frame_restart_num_runs.append(restart_runs)
        per_frame_restart_total_ldpc_iters.append(restart_iters)
        per_frame_restart_anchor_bits_total.append(restart_anchor_bits)
        per_frame_disagreement_added.append(disagree_added)
        per_frame_snapshot_attempts.append(int(snapshot_attempts))
        per_frame_snapshot_success_iter.append(int(snapshot_success_iter))
        per_frame_snapshot_last_iter.append(int(snapshot_last_iter))
        per_frame_bit_errors_stage1.append(int(be1))
        per_frame_bit_errors_after.append(int(be_after))
        per_frame_stage2_improved.append(int(stage1_failed and (be_after < be1)))
        per_frame_stage2_true_fix.append(int(stage1_failed and (be_after == 0)))
        per_frame_stage1_syndrome_weight.append(int(syn_w))
        per_frame_stage1_error_span.append(int(stage1_err_span))
        per_frame_stage1_error_runs.append(int(stage1_err_runs))
        per_frame_stage1_block_concentration.append(float(stage1_block_conc))
        per_frame_stage2_success_profile.append(str(stage2_success_profile))
        per_frame_stage2_invoked.append(bool(stage2_invoked))
        per_frame_ai_gate_action.append(str(ai_gate_action))
        per_frame_ai_gate_first_action.append(str(ai_gate_first_action))
        per_frame_ai_gate_confidence.append(float(ai_gate_confidence) if np.isfinite(ai_gate_confidence) else np.nan)
        per_frame_ai_gate_promise.append(float(ai_gate_promise) if np.isfinite(ai_gate_promise) else np.nan)
        per_frame_ai_gate_snapshot.append(int(ai_gate_snapshot))
        per_frame_ai_gate_decision_count.append(int(ai_gate_decision_count))
        per_frame_ai_gate_escalated.append(int(ai_gate_escalated))
        per_frame_ai_gate_leaf.append(str(ai_gate_leaf))

        n_frames += 1
        frame_id += 1

    if n_frames <= 0:
        return {
            "label": label,
            "snr_db": float(sim_cfg.channel.snr_db),
            "n_frames": 0,
            "ber_ldpc": 0.0,
            "fer_ldpc": 0.0,
            "ber_after": 0.0,
            "fer_after": 0.0,
            "ldpc_iters_hybrid_avg": 0.0,
            "avg_hw_cycles_stage1_per_frame": 0.0,
            "avg_hw_cycles_grand_per_frame": 0.0,
            "avg_hw_cycles_total_per_frame": 0.0,
            "avg_hw_time_stage1_us_per_frame": 0.0,
            "avg_hw_time_grand_us_per_frame": 0.0,
            "avg_hw_time_total_us_per_frame": 0.0,
            "hw_model": asdict(hw_model),
            "grand_snapshot_schedule": np.asarray(snapshot_schedule, dtype=np.int32),
        }

    ber_ldpc = total_bit_errs_stage1 / (n_frames * N)
    fer_ldpc = frame_errs_stage1 / n_frames
    avg_iters_stage1 = total_iters_stage1 / n_frames

    ber_after = total_bit_errs_after / (n_frames * N)
    fer_after = frame_errs_after / n_frames

    avg_hw_cycles_stage1 = total_hw_cycles_stage1 / n_frames
    avg_hw_cycles_grand = total_hw_cycles_grand / n_frames
    avg_hw_cycles_total = total_hw_cycles_total / n_frames

    avg_hw_time_stage1_us = cycles_to_us(avg_hw_cycles_stage1, hw_model)
    avg_hw_time_grand_us = cycles_to_us(avg_hw_cycles_grand, hw_model)
    avg_hw_time_total_us = cycles_to_us(avg_hw_cycles_total, hw_model)

    print(f"\n=== Adaptive HYBRID Monte-Carlo (channel={sim_cfg.channel.name}) ===")
    print(f"Decoder label                 : {label}")
    print(f"SNR (dB)                      : {sim_cfg.channel.snr_db:.2f}")
    print(f"Frames simulated              : {n_frames}")
    print(f"Final frame errors            : {frame_errs_after} (target={target_fe})")
    print(f"Stage-1 BER                   : {ber_ldpc:.3e}")
    print(f"Stage-1 FER                   : {fer_ldpc:.3e}")
    print(f"Final BER (LDPC+GRAND)        : {ber_after:.3e}")
    print(f"Final FER (LDPC+GRAND)        : {fer_after:.3e}")
    print(f"Avg stage-1 iters/frame       : {avg_iters_stage1:.2f}")
    print(f"Avg HW time/frame (stage-1)   : {avg_hw_time_stage1_us:.2f} µs")
    print(f"Avg HW time/frame (GRAND)     : {avg_hw_time_grand_us:.2f} µs")
    print(f"Avg HW time/frame (total)     : {avg_hw_time_total_us:.2f} µs")

    return {
        "label": label,
        "snr_db": float(sim_cfg.channel.snr_db),
        "n_frames": int(n_frames),
        "ber_ldpc": float(ber_ldpc),
        "fer_ldpc": float(fer_ldpc),
        "ldpc_iters_hybrid_avg": float(avg_iters_stage1),
        "ber_after": float(ber_after),
        "fer_after": float(fer_after),
        "avg_hw_cycles_stage1_per_frame": float(avg_hw_cycles_stage1),
        "avg_hw_cycles_grand_per_frame": float(avg_hw_cycles_grand),
        "avg_hw_cycles_total_per_frame": float(avg_hw_cycles_total),
        "avg_hw_time_stage1_us_per_frame": float(avg_hw_time_stage1_us),
        "avg_hw_time_grand_us_per_frame": float(avg_hw_time_grand_us),
        "avg_hw_time_total_us_per_frame": float(avg_hw_time_total_us),
        "per_frame_hw_cycles_stage1": np.array(per_frame_hw_cycles_stage1, dtype=np.int64),
        "per_frame_hw_cycles_grand": np.array(per_frame_hw_cycles_grand, dtype=np.int64),
        "per_frame_hw_cycles_total": np.array(per_frame_hw_cycles_total, dtype=np.int64),
        "per_frame_iters_stage1": np.array(per_frame_iters_stage1, dtype=np.int32),
        "per_frame_stage1_failed": np.array(per_frame_stage1_failed, dtype=np.bool_),
        "per_frame_patterns_tested": np.array(per_frame_patterns_tested, dtype=np.int32),
        "per_frame_patterns_evaluated": np.array(per_frame_patterns_evaluated, dtype=np.int32),
        "per_frame_grand_edge_visits_evaluated": np.array(per_frame_grand_edge_visits_eval, dtype=np.int64),
        "per_frame_grand_checks_toggled_evaluated": np.array(per_frame_grand_checks_toggled_eval, dtype=np.int64),
        "per_frame_grand_num_batches_evaluated": np.array(per_frame_grand_num_batches, dtype=np.int32),
        "per_frame_grand_llr_sort_len": np.array(per_frame_grand_llr_sort_len, dtype=np.int32),
        "per_frame_grand_search_size": np.array(per_frame_grand_search_size, dtype=np.int32),
        "per_frame_grand_patterns_generated": np.array(per_frame_grand_patterns_generated, dtype=np.int64),
        "per_frame_grand_sum_pattern_weights_generated": np.array(per_frame_grand_sumw_generated, dtype=np.int64),
        "per_frame_grand_positions_packed_evaluated": np.array(per_frame_grand_positions_packed, dtype=np.int64),
        "per_frame_cluster_unsat_edges": np.array(per_frame_cluster_unsat_edges, dtype=np.int64),
        "per_frame_cluster_pair_edges": np.array(per_frame_cluster_pair_edges, dtype=np.int64),
        "per_frame_pre_solver_attempted": np.array(per_frame_pre_solver_attempted, dtype=np.int8),
        "per_frame_pre_solver_success": np.array(per_frame_pre_solver_success, dtype=np.int8),
        "per_frame_peel_candidate_size": np.array(per_frame_peel_candidate_size, dtype=np.int32),
        "per_frame_peel_residual_vars": np.array(per_frame_peel_residual_vars, dtype=np.int32),
        "per_frame_peel_dense_xor_ops": np.array(per_frame_peel_dense_xor_ops, dtype=np.int64),
        "per_frame_chase_candidate_size": np.array(per_frame_chase_candidate_size, dtype=np.int32),
        "per_frame_chase_candidates_tested": np.array(per_frame_chase_candidates_tested, dtype=np.int32),
        "per_frame_chase_total_ldpc_iters": np.array(per_frame_chase_total_ldpc_iters, dtype=np.int32),
        "per_frame_osd_candidate_size": np.array(per_frame_osd_candidate_size, dtype=np.int32),
        "per_frame_osd_candidates_tested": np.array(per_frame_osd_candidates_tested, dtype=np.int32),
        "per_frame_osd_free_dim": np.array(per_frame_osd_free_dim, dtype=np.int32),
        "per_frame_restart_num_runs": np.array(per_frame_restart_num_runs, dtype=np.int32),
        "per_frame_restart_total_ldpc_iters": np.array(per_frame_restart_total_ldpc_iters, dtype=np.int32),
        "per_frame_restart_anchor_bits_total": np.array(per_frame_restart_anchor_bits_total, dtype=np.int32),
        "per_frame_disagreement_added": np.array(per_frame_disagreement_added, dtype=np.int32),
        "per_frame_snapshot_attempts": np.array(per_frame_snapshot_attempts, dtype=np.int32),
        "per_frame_snapshot_success_iter": np.array(per_frame_snapshot_success_iter, dtype=np.int32),
        "per_frame_snapshot_last_iter": np.array(per_frame_snapshot_last_iter, dtype=np.int32),
        "per_frame_bit_errors_stage1": np.array(per_frame_bit_errors_stage1, dtype=np.int32),
        "per_frame_bit_errors_after": np.array(per_frame_bit_errors_after, dtype=np.int32),
        "per_frame_stage2_improved": np.array(per_frame_stage2_improved, dtype=np.int8),
        "per_frame_stage2_true_fix": np.array(per_frame_stage2_true_fix, dtype=np.int8),
        "per_frame_stage1_syndrome_weight": np.array(per_frame_stage1_syndrome_weight, dtype=np.int32),
        "per_frame_stage1_error_span": np.array(per_frame_stage1_error_span, dtype=np.int32),
        "per_frame_stage1_error_runs": np.array(per_frame_stage1_error_runs, dtype=np.int32),
        "per_frame_stage1_block_concentration": np.array(per_frame_stage1_block_concentration, dtype=np.float32),
        "per_frame_stage2_success_profile": np.array(per_frame_stage2_success_profile, dtype=object),
        "per_frame_stage2_invoked": np.array(per_frame_stage2_invoked, dtype=np.bool_),
        "per_frame_ai_gate_action": np.array(per_frame_ai_gate_action, dtype=object),
        "per_frame_ai_gate_first_action": np.array(per_frame_ai_gate_first_action, dtype=object),
        "per_frame_ai_gate_confidence": np.array(per_frame_ai_gate_confidence, dtype=np.float32),
        "per_frame_ai_gate_promise": np.array(per_frame_ai_gate_promise, dtype=np.float32),
        "per_frame_ai_gate_snapshot": np.array(per_frame_ai_gate_snapshot, dtype=np.int32),
        "per_frame_ai_gate_decision_count": np.array(per_frame_ai_gate_decision_count, dtype=np.int32),
        "per_frame_ai_gate_escalated": np.array(per_frame_ai_gate_escalated, dtype=np.int8),
        "per_frame_ai_gate_leaf": np.array(per_frame_ai_gate_leaf, dtype=object),
        "per_frame_probe_invoked": np.array(per_frame_probe_invoked, dtype=np.int8),
        "per_frame_probe_success": np.array(per_frame_probe_success, dtype=np.int8),
        "per_frame_probe_syndrome_drop": np.array(per_frame_probe_syndrome_drop, dtype=np.float32),
        "per_frame_probe_escalated": np.array(per_frame_probe_escalated, dtype=np.int8),
        "per_frame_probe_regime": np.array(per_frame_probe_regime, dtype=object),
        "grand_ai_gate_enabled": bool(ai_gate_cfg is not None),
        "grand_ai_gate_policy_mode": str(getattr(ai_gate_cfg, "policy_mode", "none")) if ai_gate_cfg is not None else "none",
        "grand_selection_mode": str(getattr(grand_cfg, "selection_mode", "llr")),
        "grand_llr_source": str(getattr(grand_cfg, "llr_source", "posterior")),
        "grand_sv_check_cover_k": int(getattr(grand_cfg, "sv_check_cover_k", 0)),
        "grand_sv_epsilon": float(getattr(grand_cfg, "sv_epsilon", 0.0)),
        "grand_pre_solver_mode": str(getattr(grand_cfg, "pre_solver_mode", "none")),
        "grand_fallback_profile": str(fallback_label or ""),
        "grand_peel_candidate_ratio": float(getattr(grand_cfg, "peel_candidate_ratio", 1.0)),
        "grand_peel_max_bits": int(getattr(grand_cfg, "peel_max_bits", 0) or 0),
        "grand_snapshot_schedule": np.asarray(snapshot_schedule, dtype=np.int32),
        "grand_snapshot_metrics_are_cumulative": True,
        "hw_model": asdict(hw_model),
    }
### CELL number 30-B ##########################################################################################################################
# Publication-run overrides (multi-SNR + realistic adaptive MC stopping)
# Stop rule per SNR: 200 frame errors OR 160000 frames cap.

import os

# Option A (default): multi-SNR sweep in one job (EDIT this list as needed)
snr_sweep_global = [-5.5, -5, -4.5, -4, -3.5, -3.25, -3, -2.75, -2.5 ]

# Option B (optional): run ONE SNR per Slurm job by exporting SNR_DB
# Example: export SNR_DB=3.0
snr_env = os.environ.get("SNR_DB", "").strip()
if snr_env:
    snr_sweep_global = [float(snr_env)]

mc_cfg = AdaptiveMCConfig(
    target_frame_errors=200,
    min_frames=0,
    max_frames=160000,
)


### CELL number 31 ###
import csv
import pickle



def _diag_error_span(error_positions: np.ndarray) -> int:
    pos = np.asarray(error_positions, dtype=np.int64).reshape(-1)
    if pos.size == 0:
        return 0
    return int(pos.max() - pos.min() + 1)


def _diag_error_runs(error_positions: np.ndarray) -> int:
    pos = np.asarray(error_positions, dtype=np.int64).reshape(-1)
    if pos.size == 0:
        return 0
    pos = np.unique(np.sort(pos))
    if pos.size == 0:
        return 0
    return int(1 + np.count_nonzero(np.diff(pos) > 1))


def _diag_block_concentration(error_positions: np.ndarray, block_size: int) -> float:
    pos = np.asarray(error_positions, dtype=np.int64).reshape(-1)
    if pos.size == 0:
        return 0.0
    block = max(1, int(block_size))
    bins = pos // block
    _, counts = np.unique(bins, return_counts=True)
    return float(np.max(counts) / pos.size)


def _dist_stats(arr) -> dict:
    """Return mean/p95/p99/max for a 1D array; NaNs if empty."""
    a = np.asarray(arr)
    if a.size == 0:
        return {"mean": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan}
    a = a.astype(np.float64, copy=False)
    return {
        "mean": float(a.mean()),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }

def _cycles_to_us_arr(cycles_arr, hw_model_dict) -> np.ndarray:
    a = np.asarray(cycles_arr)
    if a.size == 0:
        return np.array([], dtype=np.float64)
    fclk = float(hw_model_dict.get("fclk_mhz", np.nan))
    if not np.isfinite(fclk) or fclk <= 0:
        return np.array([], dtype=np.float64)
    return a.astype(np.float64, copy=False) / fclk  # cycles_to_us == cycles / fclk_mhz

def save_awgn_results(
    results: Dict[float, Dict[str, Any]],
    output_dir: str,
    prefix: str = "awgn_adaptive_hw",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{prefix}_{timestamp}"

    # ---- RAW pickle (unchanged) ----
    pkl_path = os.path.join(output_dir, base_name + "_raw.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- Mean-only summary (unchanged) ----
    csv_path = os.path.join(output_dir, base_name + "_summary.csv")
    fieldnames = [
        "snr_db",
        "decoder",
        "ber",
        "fer",
        "avg_iters",
        "avg_hw_time_us",
        "avg_hw_cycles",
        "avg_hw_time_stage1_us",
        "avg_hw_time_grand_us",
        "ber_stage1",
        "fer_stage1",
    ]

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for snr in sorted(results.keys()):
            stats_all = results[snr]
            for dec_name, stats in stats_all.items():
                row = {
                    "snr_db": float(snr),
                    "decoder": str(dec_name),
                    "ber": np.nan,
                    "fer": np.nan,
                    "avg_iters": np.nan,
                    "avg_hw_time_us": np.nan,
                    "avg_hw_cycles": np.nan,
                    "avg_hw_time_stage1_us": np.nan,
                    "avg_hw_time_grand_us": np.nan,
                    "ber_stage1": np.nan,
                    "fer_stage1": np.nan,
                }

                if str(dec_name).startswith("ldpc"):
                    row["ber"] = float(stats.get("ber", np.nan))
                    row["fer"] = float(stats.get("fer", np.nan))
                    row["avg_iters"] = float(stats.get("avg_iters", np.nan))
                    row["avg_hw_cycles"] = float(stats.get("avg_hw_cycles_per_frame", np.nan))
                    row["avg_hw_time_us"] = float(stats.get("avg_hw_time_us_per_frame", np.nan))
                    row["avg_hw_time_stage1_us"] = row["avg_hw_time_us"]
                    row["avg_hw_time_grand_us"] = 0.0
                                        # For LDPC-only, "stage-1" == total
                    row["ber_stage1"] = row["ber"]
                    row["fer_stage1"] = row["fer"]
                else:
                    row["ber"] = float(stats.get("ber_after", np.nan))
                    row["fer"] = float(stats.get("fer_after", np.nan))
                    row["avg_iters"] = float(stats.get("ldpc_iters_hybrid_avg", np.nan))
                    row["avg_hw_cycles"] = float(stats.get("avg_hw_cycles_total_per_frame", np.nan))
                    row["avg_hw_time_us"] = float(stats.get("avg_hw_time_total_us_per_frame", np.nan))
                    row["avg_hw_time_stage1_us"] = float(stats.get("avg_hw_time_stage1_us_per_frame", np.nan))
                    row["avg_hw_time_grand_us"] = float(stats.get("avg_hw_time_grand_us_per_frame", np.nan))
                    row["ber_stage1"] = float(stats.get("ber_ldpc", np.nan))
                    row["fer_stage1"] = float(stats.get("fer_ldpc", np.nan))

                writer.writerow(row)

    # ---- NEW: tails + patterns summary ----
    tails_path = os.path.join(output_dir, base_name + "_summary_tails.csv")
    tails_fields = [
        "snr_db", "decoder", "n_frames",
        "ber", "fer", "avg_iters",

        # Total HW cycles/time tails
        "hw_cycles_mean", "hw_cycles_p95", "hw_cycles_p99", "hw_cycles_max",
        "hw_time_us_mean", "hw_time_us_p95", "hw_time_us_p99", "hw_time_us_max",

        # Hybrid decomposition (NaN for LDPC-only)
        "ber_stage1", "fer_stage1", "grand_invocation_rate",
        "stage1_cycles_mean", "stage1_cycles_p95", "stage1_cycles_p99", "stage1_cycles_max",
        "grand_cycles_mean", "grand_cycles_p95", "grand_cycles_p99", "grand_cycles_max",

        # GRAND pattern stats (overall; NaN for LDPC-only)
        "patterns_tested_mean", "patterns_tested_p95", "patterns_tested_p99", "patterns_tested_max",
        "patterns_evaluated_mean", "patterns_evaluated_p95", "patterns_evaluated_p99", "patterns_evaluated_max",

        # GRAND pattern stats conditional on GRAND invoked (stage1_failed==True)
        "patterns_tested_mean_if_grand", "patterns_tested_p95_if_grand", "patterns_tested_p99_if_grand", "patterns_tested_max_if_grand",
        "patterns_evaluated_mean_if_grand", "patterns_evaluated_p95_if_grand", "patterns_evaluated_p99_if_grand", "patterns_evaluated_max_if_grand",

        # Optional Receiver-3 / pre-solver tails
        "pre_solver_attempt_rate", "pre_solver_success_rate_total", "pre_solver_success_rate_if_attempted",
        "peel_candidate_mean", "peel_candidate_p95", "peel_candidate_max",
        "peel_residual_vars_mean", "peel_residual_vars_p95", "peel_residual_vars_max",

        # Optional Receiver-4 / Chase-list tails
        "chase_candidate_mean", "chase_candidate_p95", "chase_candidate_max",
        "chase_candidates_tested_mean", "chase_candidates_tested_p95", "chase_candidates_tested_max",
        "chase_ldpc_iters_mean", "chase_ldpc_iters_p95", "chase_ldpc_iters_max",

        # Optional Receiver-5 / OSD + anchored-restart tails
        "osd_candidate_mean", "osd_candidate_p95", "osd_candidate_max",
        "osd_candidates_tested_mean", "osd_candidates_tested_p95", "osd_candidates_tested_max",
        "osd_free_dim_mean", "osd_free_dim_p95", "osd_free_dim_max",
        "restart_num_runs_mean", "restart_num_runs_p95", "restart_num_runs_max",
        "restart_ldpc_iters_mean", "restart_ldpc_iters_p95", "restart_ldpc_iters_max",
        "restart_anchor_bits_mean", "restart_anchor_bits_p95", "restart_anchor_bits_max",
        "disagreement_added_mean", "disagreement_added_p95", "disagreement_added_max",
    ]

    with open(tails_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=tails_fields)
        writer.writeheader()

        for snr in sorted(results.keys()):
            stats_all = results[snr]
            for dec_name, stats in stats_all.items():
                dec_name = str(dec_name)
                hw_model_dict = stats.get("hw_model", {}) if isinstance(stats, dict) else {}

                row = {k: np.nan for k in tails_fields}
                row["snr_db"] = float(snr)
                row["decoder"] = dec_name

                if dec_name.startswith("ldpc"):
                    n_frames = int(stats.get("num_frames", 0))
                    row["n_frames"] = n_frames
                    row["ber"] = float(stats.get("ber", np.nan))
                    row["fer"] = float(stats.get("fer", np.nan))
                    row["avg_iters"] = float(stats.get("avg_iters", np.nan))

                    cyc = np.asarray(stats.get("per_frame_hw_cycles", []), dtype=np.int64)
                    t_us = np.asarray(stats.get("per_frame_hw_time_us", []), dtype=np.float64)
                    cyc_s = _dist_stats(cyc)
                    t_s = _dist_stats(t_us)

                    row["hw_cycles_mean"] = cyc_s["mean"]
                    row["hw_cycles_p95"]  = cyc_s["p95"]
                    row["hw_cycles_p99"]  = cyc_s["p99"]
                    row["hw_cycles_max"]  = cyc_s["max"]

                    row["hw_time_us_mean"] = t_s["mean"]
                    row["hw_time_us_p95"]  = t_s["p95"]
                    row["hw_time_us_p99"]  = t_s["p99"]
                    row["hw_time_us_max"]  = t_s["max"]

                    # LDPC-only: stage1 == total, grand == 0
                    row["ber_stage1"] = row["ber"]
                    row["fer_stage1"] = row["fer"]
                    row["grand_invocation_rate"] = 0.0

                    row["stage1_cycles_mean"] = row["hw_cycles_mean"]
                    row["stage1_cycles_p95"]  = row["hw_cycles_p95"]
                    row["stage1_cycles_p99"]  = row["hw_cycles_p99"]
                    row["stage1_cycles_max"]  = row["hw_cycles_max"]

                    row["grand_cycles_mean"] = 0.0
                    row["grand_cycles_p95"]  = 0.0
                    row["grand_cycles_p99"]  = 0.0
                    row["grand_cycles_max"]  = 0.0

                    row["pre_solver_attempt_rate"] = 0.0
                    row["pre_solver_success_rate_total"] = 0.0
                    row["pre_solver_success_rate_if_attempted"] = np.nan
                    row["peel_candidate_mean"] = 0.0
                    row["peel_candidate_p95"] = 0.0
                    row["peel_candidate_max"] = 0.0
                    row["peel_residual_vars_mean"] = 0.0
                    row["peel_residual_vars_p95"] = 0.0
                    row["peel_residual_vars_max"] = 0.0
                    row["chase_candidate_mean"] = 0.0
                    row["chase_candidate_p95"] = 0.0
                    row["chase_candidate_max"] = 0.0
                    row["chase_candidates_tested_mean"] = 0.0
                    row["chase_candidates_tested_p95"] = 0.0
                    row["chase_candidates_tested_max"] = 0.0
                    row["chase_ldpc_iters_mean"] = 0.0
                    row["chase_ldpc_iters_p95"] = 0.0
                    row["chase_ldpc_iters_max"] = 0.0

                else:
                    n_frames = int(stats.get("n_frames", 0))
                    row["n_frames"] = n_frames
                    row["ber"] = float(stats.get("ber_after", np.nan))
                    row["fer"] = float(stats.get("fer_after", np.nan))
                    row["avg_iters"] = float(stats.get("ldpc_iters_hybrid_avg", np.nan))

                    row["ber_stage1"] = float(stats.get("ber_ldpc", np.nan))
                    row["fer_stage1"] = float(stats.get("fer_ldpc", np.nan))

                    cyc_stage1 = np.asarray(stats.get("per_frame_hw_cycles_stage1", []), dtype=np.int64)
                    cyc_grand  = np.asarray(stats.get("per_frame_hw_cycles_grand",  []), dtype=np.int64)
                    cyc_total  = np.asarray(stats.get("per_frame_hw_cycles_total",  []), dtype=np.int64)

                    # Total tails
                    cyc_s = _dist_stats(cyc_total)
                    row["hw_cycles_mean"] = cyc_s["mean"]
                    row["hw_cycles_p95"]  = cyc_s["p95"]
                    row["hw_cycles_p99"]  = cyc_s["p99"]
                    row["hw_cycles_max"]  = cyc_s["max"]

                    t_us_total = _cycles_to_us_arr(cyc_total, hw_model_dict)
                    t_s = _dist_stats(t_us_total)
                    row["hw_time_us_mean"] = t_s["mean"]
                    row["hw_time_us_p95"]  = t_s["p95"]
                    row["hw_time_us_p99"]  = t_s["p99"]
                    row["hw_time_us_max"]  = t_s["max"]

                    # Decomposition tails
                    s1 = _dist_stats(cyc_stage1)
                    g  = _dist_stats(cyc_grand)
                    row["stage1_cycles_mean"] = s1["mean"]
                    row["stage1_cycles_p95"]  = s1["p95"]
                    row["stage1_cycles_p99"]  = s1["p99"]
                    row["stage1_cycles_max"]  = s1["max"]
                    row["grand_cycles_mean"]  = g["mean"]
                    row["grand_cycles_p95"]   = g["p95"]
                    row["grand_cycles_p99"]   = g["p99"]
                    row["grand_cycles_max"]   = g["max"]

                    # GRAND invocation + pattern stats
                    stage1_failed = np.asarray(stats.get("per_frame_stage1_failed", []), dtype=np.bool_)
                    if stage1_failed.size > 0:
                        row["grand_invocation_rate"] = float(stage1_failed.mean())
                    else:
                        row["grand_invocation_rate"] = np.nan

                    pt = np.asarray(stats.get("per_frame_patterns_tested", []), dtype=np.int64)
                    pe = np.asarray(stats.get("per_frame_patterns_evaluated", []), dtype=np.int64)

                    pt_s = _dist_stats(pt)
                    pe_s = _dist_stats(pe)
                    row["patterns_tested_mean"] = pt_s["mean"]
                    row["patterns_tested_p95"]  = pt_s["p95"]
                    row["patterns_tested_p99"]  = pt_s["p99"]
                    row["patterns_tested_max"]  = pt_s["max"]

                    row["patterns_evaluated_mean"] = pe_s["mean"]
                    row["patterns_evaluated_p95"]  = pe_s["p95"]
                    row["patterns_evaluated_p99"]  = pe_s["p99"]
                    row["patterns_evaluated_max"]  = pe_s["max"]

                    if stage1_failed.size > 0 and pt.size == stage1_failed.size:
                        pt_if = pt[stage1_failed]
                        pe_if = pe[stage1_failed]
                    else:
                        pt_if = np.array([], dtype=np.int64)
                        pe_if = np.array([], dtype=np.int64)

                    pt_if_s = _dist_stats(pt_if)
                    pe_if_s = _dist_stats(pe_if)

                    row["patterns_tested_mean_if_grand"] = pt_if_s["mean"]
                    row["patterns_tested_p95_if_grand"]  = pt_if_s["p95"]
                    row["patterns_tested_p99_if_grand"]  = pt_if_s["p99"]
                    row["patterns_tested_max_if_grand"]  = pt_if_s["max"]

                    row["patterns_evaluated_mean_if_grand"] = pe_if_s["mean"]
                    row["patterns_evaluated_p95_if_grand"]  = pe_if_s["p95"]
                    row["patterns_evaluated_p99_if_grand"]  = pe_if_s["p99"]
                    row["patterns_evaluated_max_if_grand"]  = pe_if_s["max"]

                    # Optional Receiver-3 / pre-solver tails
                    ps_attempt = np.asarray(stats.get("per_frame_pre_solver_attempted", []), dtype=np.int8)
                    ps_success = np.asarray(stats.get("per_frame_pre_solver_success", []), dtype=np.int8)
                    peel_cand = np.asarray(stats.get("per_frame_peel_candidate_size", []), dtype=np.int64)
                    peel_resv = np.asarray(stats.get("per_frame_peel_residual_vars", []), dtype=np.int64)

                    if ps_attempt.size > 0:
                        row["pre_solver_attempt_rate"] = float(ps_attempt.mean())
                    else:
                        row["pre_solver_attempt_rate"] = np.nan

                    if ps_success.size > 0:
                        row["pre_solver_success_rate_total"] = float(ps_success.mean())
                    else:
                        row["pre_solver_success_rate_total"] = np.nan

                    if ps_attempt.size > 0 and ps_success.size == ps_attempt.size and int(ps_attempt.sum()) > 0:
                        row["pre_solver_success_rate_if_attempted"] = float(ps_success[ps_attempt.astype(bool)].mean())
                    else:
                        row["pre_solver_success_rate_if_attempted"] = np.nan

                    peel_cand_s = _dist_stats(peel_cand)
                    row["peel_candidate_mean"] = peel_cand_s["mean"]
                    row["peel_candidate_p95"]  = peel_cand_s["p95"]
                    row["peel_candidate_max"]  = peel_cand_s["max"]

                    peel_resv_s = _dist_stats(peel_resv)
                    row["peel_residual_vars_mean"] = peel_resv_s["mean"]
                    row["peel_residual_vars_p95"]  = peel_resv_s["p95"]
                    row["peel_residual_vars_max"]  = peel_resv_s["max"]

                    chase_cand = np.asarray(stats.get("per_frame_chase_candidate_size", []), dtype=np.int64)
                    chase_tested = np.asarray(stats.get("per_frame_chase_candidates_tested", []), dtype=np.int64)
                    chase_iters = np.asarray(stats.get("per_frame_chase_total_ldpc_iters", []), dtype=np.int64)

                    chase_cand_s = _dist_stats(chase_cand)
                    row["chase_candidate_mean"] = chase_cand_s["mean"]
                    row["chase_candidate_p95"]  = chase_cand_s["p95"]
                    row["chase_candidate_max"]  = chase_cand_s["max"]

                    chase_tested_s = _dist_stats(chase_tested)
                    row["chase_candidates_tested_mean"] = chase_tested_s["mean"]
                    row["chase_candidates_tested_p95"]  = chase_tested_s["p95"]
                    row["chase_candidates_tested_max"]  = chase_tested_s["max"]

                    chase_iters_s = _dist_stats(chase_iters)
                    row["chase_ldpc_iters_mean"] = chase_iters_s["mean"]
                    row["chase_ldpc_iters_p95"]  = chase_iters_s["p95"]
                    row["chase_ldpc_iters_max"]  = chase_iters_s["max"]

                    osd_cand = np.asarray(stats.get("per_frame_osd_candidate_size", []), dtype=np.int64)
                    osd_tested = np.asarray(stats.get("per_frame_osd_candidates_tested", []), dtype=np.int64)
                    osd_free = np.asarray(stats.get("per_frame_osd_free_dim", []), dtype=np.int64)
                    restart_runs = np.asarray(stats.get("per_frame_restart_num_runs", []), dtype=np.int64)
                    restart_iters = np.asarray(stats.get("per_frame_restart_total_ldpc_iters", []), dtype=np.int64)
                    restart_anchor_bits = np.asarray(stats.get("per_frame_restart_anchor_bits_total", []), dtype=np.int64)
                    disagree_added = np.asarray(stats.get("per_frame_disagreement_added", []), dtype=np.int64)

                    osd_cand_s = _dist_stats(osd_cand)
                    row["osd_candidate_mean"] = osd_cand_s["mean"]
                    row["osd_candidate_p95"]  = osd_cand_s["p95"]
                    row["osd_candidate_max"]  = osd_cand_s["max"]

                    osd_tested_s = _dist_stats(osd_tested)
                    row["osd_candidates_tested_mean"] = osd_tested_s["mean"]
                    row["osd_candidates_tested_p95"]  = osd_tested_s["p95"]
                    row["osd_candidates_tested_max"]  = osd_tested_s["max"]

                    osd_free_s = _dist_stats(osd_free)
                    row["osd_free_dim_mean"] = osd_free_s["mean"]
                    row["osd_free_dim_p95"]  = osd_free_s["p95"]
                    row["osd_free_dim_max"]  = osd_free_s["max"]

                    restart_runs_s = _dist_stats(restart_runs)
                    row["restart_num_runs_mean"] = restart_runs_s["mean"]
                    row["restart_num_runs_p95"]  = restart_runs_s["p95"]
                    row["restart_num_runs_max"]  = restart_runs_s["max"]

                    restart_iters_s = _dist_stats(restart_iters)
                    row["restart_ldpc_iters_mean"] = restart_iters_s["mean"]
                    row["restart_ldpc_iters_p95"]  = restart_iters_s["p95"]
                    row["restart_ldpc_iters_max"]  = restart_iters_s["max"]

                    restart_anchor_s = _dist_stats(restart_anchor_bits)
                    row["restart_anchor_bits_mean"] = restart_anchor_s["mean"]
                    row["restart_anchor_bits_p95"]  = restart_anchor_s["p95"]
                    row["restart_anchor_bits_max"]  = restart_anchor_s["max"]

                    disagree_s = _dist_stats(disagree_added)
                    row["disagreement_added_mean"] = disagree_s["mean"]
                    row["disagreement_added_p95"]  = disagree_s["p95"]
                    row["disagreement_added_max"]  = disagree_s["max"]

                writer.writerow(row)


    # ---- NEW: hybrid diagnostics summary ----
    # Focus on what actually explains hybrid gains or failures: stage-2 invocation,
    # true-fix rate, error locality at the stage-1 output, and which cascade profile won.
    diag_iters = sorted({
        int(x)
        for stats_all in results.values()
        for stats in (stats_all.values() if isinstance(stats_all, dict) else [])
        for x in np.asarray(stats.get("grand_snapshot_schedule", []), dtype=np.int32).reshape(-1)
        if int(x) > 0
    })
    diag_fields = [
        "snr_db", "decoder", "n_frames", "ber", "fer", "ber_stage1", "fer_stage1",
        "stage2_invocation_rate", "stage2_improve_rate_if_invoked", "stage2_true_fix_rate_if_invoked",
        "avg_stage1_bit_errors_if_invoked", "p95_stage1_bit_errors_if_invoked",
        "avg_stage1_syndrome_weight_if_invoked", "p95_stage1_syndrome_weight_if_invoked",
        "avg_stage1_error_span_if_invoked", "p95_stage1_error_span_if_invoked",
        "avg_stage1_error_runs_if_invoked", "p95_stage1_error_runs_if_invoked",
        "avg_stage1_block_concentration_if_invoked", "p95_stage1_block_concentration_if_invoked",
        "avg_snapshot_attempts_if_invoked", "avg_snapshot_success_iter_if_fixed",
        "primary_success_rate_if_invoked", "fallback_success_rate_if_invoked",
        "probe_invocation_rate", "probe_success_rate_if_invoked", "probe_escalation_rate_if_invoked", "probe_syndrome_drop_mean_if_invoked",
        "ai_gate_enabled", "ai_gate_policy_mode", "ai_gate_skip_rate", "ai_gate_tiny_rate", "ai_gate_full_rate", "ai_gate_meta_rate",
        "ai_gate_first_skip_rate", "ai_gate_decision_count_mean_if_invoked", "ai_gate_escalation_rate_if_invoked",
        "ai_gate_confidence_mean_if_invoked", "ai_gate_promise_mean_if_invoked",
        "ai_gate_skip_rate_if_failed", "ai_gate_tiny_rate_if_failed", "ai_gate_full_rate_if_failed", "ai_gate_meta_rate_if_failed",
        "ai_gate_first_skip_rate_if_failed", "ai_gate_decision_count_mean_if_failed", "ai_gate_escalation_rate_if_failed",
        "ai_gate_confidence_mean_if_failed", "ai_gate_promise_mean_if_failed",
    ] + [f"snapshot_success_at_{it}" for it in diag_iters]

    diag_path = os.path.join(output_dir, base_name + "_summary_diagnostics.csv")
    with open(diag_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=diag_fields)
        writer.writeheader()
        for snr in sorted(results.keys()):
            stats_all = results[snr]
            for dec_name, stats in stats_all.items():
                row = {k: np.nan for k in diag_fields}
                row["snr_db"] = float(snr)
                row["decoder"] = str(dec_name)
                row["n_frames"] = int(stats.get("n_frames", stats.get("num_frames", 0)))
                row["ber"] = float(stats.get("ber_after", stats.get("ber", np.nan)))
                row["fer"] = float(stats.get("fer_after", stats.get("fer", np.nan)))
                row["ber_stage1"] = float(stats.get("ber_ldpc", stats.get("ber", np.nan)))
                row["fer_stage1"] = float(stats.get("fer_ldpc", stats.get("fer", np.nan)))

                failed = np.asarray(stats.get("per_frame_stage1_failed", []), dtype=np.bool_)
                invoked = np.asarray(stats.get("per_frame_stage2_invoked", stats.get("per_frame_stage1_failed", [])), dtype=np.bool_)
                be1 = np.asarray(stats.get("per_frame_bit_errors_stage1", []), dtype=np.int32)
                syn = np.asarray(stats.get("per_frame_stage1_syndrome_weight", []), dtype=np.int32)
                esp = np.asarray(stats.get("per_frame_stage1_error_span", []), dtype=np.int32)
                ern = np.asarray(stats.get("per_frame_stage1_error_runs", []), dtype=np.int32)
                bconc = np.asarray(stats.get("per_frame_stage1_block_concentration", []), dtype=np.float64)
                satt = np.asarray(stats.get("per_frame_snapshot_attempts", []), dtype=np.int32)
                ssucc = np.asarray(stats.get("per_frame_snapshot_success_iter", []), dtype=np.int32)
                simp = np.asarray(stats.get("per_frame_stage2_improved", []), dtype=np.int8)
                sfix = np.asarray(stats.get("per_frame_stage2_true_fix", []), dtype=np.int8)
                profile = np.asarray(stats.get("per_frame_stage2_success_profile", []), dtype=object)
                fallback_profile = str(stats.get("grand_fallback_profile", "") or "")
                primary_profile = str(stats.get("grand_pre_solver_mode", "") or "")
                gact = np.asarray(stats.get("per_frame_ai_gate_action", []), dtype=object)
                gfirst = np.asarray(stats.get("per_frame_ai_gate_first_action", []), dtype=object)
                gconf = np.asarray(stats.get("per_frame_ai_gate_confidence", []), dtype=np.float64)
                gprom = np.asarray(stats.get("per_frame_ai_gate_promise", []), dtype=np.float64)
                gdec = np.asarray(stats.get("per_frame_ai_gate_decision_count", []), dtype=np.int32)
                gesc = np.asarray(stats.get("per_frame_ai_gate_escalated", []), dtype=np.int8)
                row["ai_gate_enabled"] = float(bool(stats.get("grand_ai_gate_enabled", False)))
                row["ai_gate_policy_mode"] = str(stats.get("grand_ai_gate_policy_mode", "none") or "none")

                if invoked.size > 0:
                    row["stage2_invocation_rate"] = float(invoked.mean())
                if invoked.any():
                    mask = invoked.astype(bool)
                    def _safe_mean(a):
                        a = np.asarray(a)
                        return float(a.mean()) if a.size else np.nan
                    def _safe_p95(a):
                        a = np.asarray(a)
                        return float(np.percentile(a, 95)) if a.size else np.nan
                    row["stage2_improve_rate_if_invoked"] = _safe_mean(simp[mask])
                    row["stage2_true_fix_rate_if_invoked"] = _safe_mean(sfix[mask])
                    row["avg_stage1_bit_errors_if_invoked"] = _safe_mean(be1[mask])
                    row["p95_stage1_bit_errors_if_invoked"] = _safe_p95(be1[mask])
                    row["avg_stage1_syndrome_weight_if_invoked"] = _safe_mean(syn[mask])
                    row["p95_stage1_syndrome_weight_if_invoked"] = _safe_p95(syn[mask])
                    row["avg_stage1_error_span_if_invoked"] = _safe_mean(esp[mask])
                    row["p95_stage1_error_span_if_invoked"] = _safe_p95(esp[mask])
                    row["avg_stage1_error_runs_if_invoked"] = _safe_mean(ern[mask])
                    row["p95_stage1_error_runs_if_invoked"] = _safe_p95(ern[mask])
                    row["avg_stage1_block_concentration_if_invoked"] = _safe_mean(bconc[mask])
                    row["p95_stage1_block_concentration_if_invoked"] = _safe_p95(bconc[mask])
                    row["avg_snapshot_attempts_if_invoked"] = _safe_mean(satt[mask])
                    fixed_mask = mask & (sfix.astype(bool))
                    if fixed_mask.any():
                        row["avg_snapshot_success_iter_if_fixed"] = _safe_mean(ssucc[fixed_mask])
                    if profile.size == invoked.size:
                        if primary_profile:
                            row["primary_success_rate_if_invoked"] = float(np.mean(np.asarray([str(x) == primary_profile for x in profile[mask]], dtype=np.float64)))
                        if fallback_profile:
                            row["fallback_success_rate_if_invoked"] = float(np.mean(np.asarray([str(x) == fallback_profile for x in profile[mask]], dtype=np.float64)))
                    pprobe = np.asarray(stats.get("per_frame_probe_invoked", []), dtype=np.int8)
                    ppsucc = np.asarray(stats.get("per_frame_probe_success", []), dtype=np.int8)
                    ppdrop = np.asarray(stats.get("per_frame_probe_syndrome_drop", []), dtype=np.float64)
                    ppescal = np.asarray(stats.get("per_frame_probe_escalated", []), dtype=np.int8)
                    if pprobe.size == invoked.size:
                        row["probe_invocation_rate"] = _safe_mean(pprobe[mask])
                    if ppsucc.size == invoked.size and pprobe.size == invoked.size:
                        pmask = mask & (pprobe.astype(bool))
                        if pmask.any():
                            row["probe_success_rate_if_invoked"] = _safe_mean(ppsucc[pmask])
                    if ppescal.size == invoked.size and pprobe.size == invoked.size:
                        pmask = mask & (pprobe.astype(bool))
                        if pmask.any():
                            row["probe_escalation_rate_if_invoked"] = _safe_mean(ppescal[pmask])
                    if ppdrop.size == invoked.size and pprobe.size == invoked.size:
                        pmask = mask & (pprobe.astype(bool))
                        if pmask.any():
                            row["probe_syndrome_drop_mean_if_invoked"] = _safe_mean(ppdrop[pmask])
                    if gact.size == invoked.size:
                        row["ai_gate_skip_rate"] = float(np.mean(np.asarray([str(x) == "skip" for x in gact[mask]], dtype=np.float64)))
                        row["ai_gate_tiny_rate"] = float(np.mean(np.asarray([str(x) == "tiny" for x in gact[mask]], dtype=np.float64)))
                        row["ai_gate_full_rate"] = float(np.mean(np.asarray([str(x) == "full" for x in gact[mask]], dtype=np.float64)))
                        row["ai_gate_meta_rate"] = float(np.mean(np.asarray([str(x) == "meta" for x in gact[mask]], dtype=np.float64)))
                    if gfirst.size == invoked.size:
                        row["ai_gate_first_skip_rate"] = float(np.mean(np.asarray([str(x) == "skip" for x in gfirst[mask]], dtype=np.float64)))
                    if gdec.size == invoked.size:
                        row["ai_gate_decision_count_mean_if_invoked"] = _safe_mean(gdec[mask])
                    if gesc.size == invoked.size:
                        row["ai_gate_escalation_rate_if_invoked"] = _safe_mean(gesc[mask])
                    if gconf.size == invoked.size:
                        conf_vals = gconf[mask]
                        conf_vals = conf_vals[np.isfinite(conf_vals)]
                        row["ai_gate_confidence_mean_if_invoked"] = _safe_mean(conf_vals)
                    if gprom.size == invoked.size:
                        prom_vals = gprom[mask]
                        prom_vals = prom_vals[np.isfinite(prom_vals)]
                        row["ai_gate_promise_mean_if_invoked"] = _safe_mean(prom_vals)
                if failed.size > 0 and failed.any():
                    fmask = failed.astype(bool)
                    def _safe_mean_failed(a):
                        a = np.asarray(a)
                        return float(a.mean()) if a.size else np.nan
                    if gact.size == failed.size:
                        row["ai_gate_skip_rate_if_failed"] = float(np.mean(np.asarray([str(x) == "skip" for x in gact[fmask]], dtype=np.float64)))
                        row["ai_gate_tiny_rate_if_failed"] = float(np.mean(np.asarray([str(x) == "tiny" for x in gact[fmask]], dtype=np.float64)))
                        row["ai_gate_full_rate_if_failed"] = float(np.mean(np.asarray([str(x) == "full" for x in gact[fmask]], dtype=np.float64)))
                        row["ai_gate_meta_rate_if_failed"] = float(np.mean(np.asarray([str(x) == "meta" for x in gact[fmask]], dtype=np.float64)))
                    if gfirst.size == failed.size:
                        row["ai_gate_first_skip_rate_if_failed"] = float(np.mean(np.asarray([str(x) == "skip" for x in gfirst[fmask]], dtype=np.float64)))
                    if gdec.size == failed.size:
                        row["ai_gate_decision_count_mean_if_failed"] = _safe_mean_failed(gdec[fmask])
                    if gesc.size == failed.size:
                        row["ai_gate_escalation_rate_if_failed"] = _safe_mean_failed(gesc[fmask])
                    if gconf.size == failed.size:
                        conf_vals = gconf[fmask]
                        conf_vals = conf_vals[np.isfinite(conf_vals)]
                        row["ai_gate_confidence_mean_if_failed"] = _safe_mean_failed(conf_vals)
                    if gprom.size == failed.size:
                        prom_vals = gprom[fmask]
                        prom_vals = prom_vals[np.isfinite(prom_vals)]
                        row["ai_gate_promise_mean_if_failed"] = _safe_mean_failed(prom_vals)
                    if diag_iters:
                        for it in diag_iters:
                            row[f"snapshot_success_at_{it}"] = float(np.mean((ssucc[mask] == int(it)).astype(np.float64)))
                writer.writerow(row)

        print(f"[save_awgn_results] Wrote raw results : {pkl_path}")
        print(f"[save_awgn_results] Wrote mean summary: {csv_path}")
        print(f"[save_awgn_results] Wrote tails summary: {tails_path}")




import re
import zlib
import csv as _csv

def _parse_csv_float_list(s: str):
    s = (s or "").strip()
    if not s:
        return None
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out if out else None

def _stable_u32_seed_from_string(s: str) -> int:
    return int(zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF)

def _snr_sweep_from_env(default_list):
    env = os.environ.get("SNR_SWEEP", "").strip()
    parsed = _parse_csv_float_list(env)
    if parsed is not None:
        return parsed
    return list(default_list)

def _mc_cfg_from_env(default_mc_cfg: AdaptiveMCConfig) -> AdaptiveMCConfig:
    return AdaptiveMCConfig(
        target_frame_errors=_env_int("TARGET_FRAME_ERRORS", int(default_mc_cfg.target_frame_errors)),
        min_frames=int(default_mc_cfg.min_frames),
        max_frames=_env_int("MAX_FRAMES", int(default_mc_cfg.max_frames)),
    )

def _make_random_interleaver(N: int, seed: int) -> InterleaverConfig:
    rng = np.random.default_rng(int(seed))
    pattern = rng.permutation(int(N)).astype(np.int32, copy=False)
    return create_interleaver_from_pattern(pattern, name=f"randperm_seed{seed}_N{N}")

def _5g_lifting_size_sets():
    # 3GPP TS 38.212 lifting sizes grouped into 8 sets (set index 0..7)
    return [
        [2, 4, 8, 16, 32, 64, 128, 256],
        [3, 6, 12, 24, 48, 96, 192, 384],
        [5, 10, 20, 40, 80, 160, 320],
        [7, 14, 28, 56, 112, 224],
        [9, 18, 36, 72, 144, 288],
        [11, 22, 44, 88, 176, 352],
        [13, 26, 52, 104, 208],
        [15, 30, 60, 120, 240],
    ]

def _5g_set_index_from_z(Z: int) -> int:
    Z = int(Z)
    for s_idx, zs in enumerate(_5g_lifting_size_sets()):
        if Z in zs:
            return s_idx
    raise ValueError(f"Invalid 5G lifting factor Z={Z}. Not found in TS 38.212 lifting-size sets.")

def _load_5g_bg_entries(csv_path: str):
    """
    Parse 5G basegraph CSV in the same sparse format used by Sionna:
      Row index ; Column index ; Set0 ; Set1 ; ... ; Set7
    Row index can be blank, meaning "same as previous row".
    Returns: (mb, nb, entries) where entries is list of (r, c, shifts[8]).
    """
    entries = []
    cur_r = None
    max_r = -1
    max_c = -1

    with open(csv_path, "r", newline="") as f:
        reader = _csv.reader(f, delimiter=";")
        # Skip first two header lines
        next(reader, None)
        next(reader, None)

        for row in reader:
            if not row or len(row) < 3:
                continue

            r0 = row[0].strip() if len(row) > 0 else ""
            c0 = row[1].strip() if len(row) > 1 else ""
            if r0 != "":
                cur_r = int(float(r0))
            if cur_r is None:
                continue
            if c0 == "":
                continue
            c_ind = int(float(c0))

            shifts = []
            for k in range(8):
                idx = 2 + k
                tok = row[idx].strip() if idx < len(row) else ""
                shifts.append(int(float(tok)) if tok != "" else 0)

            entries.append((cur_r, c_ind, shifts))
            if cur_r > max_r:
                max_r = cur_r
            if c_ind > max_c:
                max_c = c_ind

    mb = max_r + 1
    nb = max_c + 1
    return mb, nb, entries

def build_5g_qc_code_cfg(
    bg: str,
    Z: int,
    csv_dir: str,
    interleaver_seed: int = 2025,
) -> Tuple[CodeConfig, InterleaverConfig]:
    """
    Build a 5G-style QC-LDPC Tanner graph (pure lifted basegraph).

    IMPORTANT: this is NOT full TS 38.212 rate-matching; it is the lifted BG Tanner graph.
    Encoding mode: all-zero (no generator-matrix construction).
    """
    bg = str(bg).strip().lower()
    Z = int(Z)

    if bg not in ("bg1", "bg2"):
        raise ValueError(f"LDPC_5G_BG must be bg1 or bg2, got: {bg}")

    csv_name = "5G_bg1.csv" if bg == "bg1" else "5G_bg2.csv"
    csv_path = os.path.join(str(csv_dir), csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"5G basegraph CSV not found: {csv_path}")

    set_idx = _5g_set_index_from_z(Z)
    mb, nb, entries = _load_5g_bg_entries(csv_path)

    N = nb * Z
    M = mb * Z
    K = N - M
    rate = float(K) / float(N)

    # Build checks_to_vars adjacency
    c2v_lists = [[] for _ in range(M)]
    for r, c, shifts in entries:
        p = int(shifts[set_idx]) % Z
        base_check = int(r) * Z
        base_var = int(c) * Z
        for i in range(Z):
            chk = base_check + i
            var = base_var + ((i + p) % Z)
            c2v_lists[chk].append(var)

    checks_to_vars = [np.asarray(lst, dtype=np.int32) for lst in c2v_lists]

    # Build vars_to_checks and edge positions
    v2c_lists = [[] for _ in range(N)]
    v2c_ep_lists = [[] for _ in range(N)]
    for chk in range(M):
        vs = checks_to_vars[chk]
        for local_e, v in enumerate(vs):
            v_int = int(v)
            v2c_lists[v_int].append(chk)
            v2c_ep_lists[v_int].append(local_e)

    vars_to_checks = [np.asarray(lst, dtype=np.int32) for lst in v2c_lists]
    var_to_checks_edge_pos = [np.asarray(lst, dtype=np.int32) for lst in v2c_ep_lists]

    code_name = f"5g_{bg}_Z{Z}_N{N}_K{K}_R{rate:.3f}"
    code_cfg = CodeConfig(
        code_name=code_name,
        N=int(N),
        K=int(K),
        rate=float(rate),
        H_path=None,
        checks_to_vars=checks_to_vars,
        vars_to_checks=vars_to_checks,
        var_to_checks_edge_pos=var_to_checks_edge_pos,
    )
    code_cfg.M = int(M)
    code_cfg.encoder_mode = "all_zero"

    prepare_code_for_fast_decoding(code_cfg)

    interleaver = _make_random_interleaver(N, interleaver_seed)
    return code_cfg, interleaver



# -------------------- Sionna 5G NR LDPC (38.212) --------------------
def _pcm_to_tanner_neighborhoods(pcm) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Convert a parity-check matrix to Tanner-graph neighborhoods.

    Supports:
      - scipy.sparse CSR/CSC/COO matrices
      - dense numpy arrays

    Returns:
      checks_to_vars, vars_to_checks, var_to_checks_edge_pos
    """
    # Lazy import (SciPy may not be needed for non-sparse paths)
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
        pcm_dense = np.asarray(pcm)
        if pcm_dense.ndim != 2:
            raise ValueError(f"PCM must be 2D, got shape {pcm_dense.shape}")
        M, N = pcm_dense.shape
        checks_to_vars = [np.flatnonzero(pcm_dense[c]).astype(np.int32) for c in range(M)]

    vars_to_checks_lists: List[List[int]] = [[] for _ in range(N)]
    for c, v_arr in enumerate(checks_to_vars):
        for v in v_arr:
            vars_to_checks_lists[int(v)].append(int(c))
    vars_to_checks: List[np.ndarray] = [np.asarray(lst, dtype=np.int32) for lst in vars_to_checks_lists]

    # For each VN->CN edge, store the local edge index position inside checks_to_vars[cn]
    var_to_checks_edge_pos: List[np.ndarray] = []
    for v in range(N):
        cn_list = vars_to_checks[v]
        pos = np.empty(cn_list.shape[0], dtype=np.int32)
        for i, c in enumerate(cn_list):
            # checks_to_vars[c] is small: linear search is fine
            loc = np.where(checks_to_vars[int(c)] == v)[0]
            pos[i] = int(loc[0])
        var_to_checks_edge_pos.append(pos)
    return checks_to_vars, vars_to_checks, var_to_checks_edge_pos


def build_sionna_5g_nr_code_cfg(
    k_info: int,
    n_tx: int,
    num_bits_per_symbol: int = 1,
    code_name_prefix: str = "sionna5g",
) -> Tuple[CodeConfig, InterleaverConfig]:
    """Build a 5G NR LDPC code config using Sionna's 38.212-compliant LDPC5GEncoder.

    This is the *receiver-side* LDPC graph (PCM) used for syndrome checks and GRAND membership tests.
    The transmitted codeword has length `n_tx` (rate-matched), but the decoding graph length is
    `pcm.shape[1]` (mother code incl. punctured + filler bits).
    """
    if not SIONNA_LDPC_AVAILABLE:
        raise RuntimeError(
            "Sionna 5G LDPC encoder not available. Install a compatible Sionna runtime. "
            f"Import detail: {_SIONNA_IMPORT_ERROR}"
        )

    qm = int(num_bits_per_symbol)
    enc = LDPC5GEncoder(k=k_info, n=n_tx, num_bits_per_symbol=qm)
    pcm = enc.pcm  # typically sparse
    checks_to_vars, vars_to_checks, var_to_checks_edge_pos = _pcm_to_tanner_neighborhoods(pcm)
    M, N = pcm.shape

    rate_eff = float(k_info) / float(n_tx)
    code_name = f"{code_name_prefix}_k{k_info}_n{n_tx}_qm{qm}"
    code_cfg = CodeConfig(code_name=code_name, N=int(N), K=int(k_info), rate=rate_eff, H_path=None)
    code_cfg.M = int(M)
    code_cfg.checks_to_vars = checks_to_vars
    code_cfg.vars_to_checks = vars_to_checks
    code_cfg.var_to_checks_edge_pos = var_to_checks_edge_pos
    code_cfg.encoder_mode = "all_zero"  # all-zero CW for symmetry

    # Store Sionna-specific metadata as a plain dict (picklable for joblib)
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

    # Precompute internal VN positions corresponding to transmitted bits (sanity probes + logging)
    code_cfg.sionna["tx_pos"] = _sionna5g_internal_tx_positions(code_cfg)

    prepare_code_for_fast_decoding(code_cfg)

    # IMPORTANT: do NOT apply any extra random interleaver on top of 5G rate-matching.
    interleaver = create_identity_interleaver(code_cfg.N)
    return code_cfg, interleaver

def build_gallager36_code_cfg(
    N: int,
    interleaver_seed: int = 2025,
    rng_seed_H: int = 2025,
    dv: int = 3,
    dc: int = 6,
) -> Tuple[CodeConfig, InterleaverConfig]:
    """
    Textbook-style (dv,dc) regular LDPC using a configuration-model Tanner graph.

    Encoding mode: all-zero (no generator-matrix construction).
    """
    N = int(N)
    dv = int(dv)
    dc = int(dc)
    if (N * dv) % dc != 0:
        raise ValueError(f"Need (N*dv) divisible by dc. Got N={N}, dv={dv}, dc={dc}.")

    M = (N * dv) // dc
    K = N - M
    rate = float(K) / float(N)

    rng = np.random.default_rng(int(rng_seed_H))
    total_edges = N * dv

    # Socket model: random pairing of variable sockets and check sockets
    var_sockets = np.repeat(np.arange(N, dtype=np.int32), dv)
    chk_sockets = np.repeat(np.arange(M, dtype=np.int32), dc)
    rng.shuffle(var_sockets)
    rng.shuffle(chk_sockets)

    c2v_lists = [[] for _ in range(M)]
    for e in range(total_edges):
        chk = int(chk_sockets[e])
        var = int(var_sockets[e])
        c2v_lists[chk].append(var)

    checks_to_vars = [np.asarray(lst, dtype=np.int32) for lst in c2v_lists]

    v2c_lists = [[] for _ in range(N)]
    v2c_ep_lists = [[] for _ in range(N)]
    for chk in range(M):
        vs = checks_to_vars[chk]
        for local_e, v in enumerate(vs):
            v_int = int(v)
            v2c_lists[v_int].append(chk)
            v2c_ep_lists[v_int].append(local_e)

    vars_to_checks = [np.asarray(lst, dtype=np.int32) for lst in v2c_lists]
    var_to_checks_edge_pos = [np.asarray(lst, dtype=np.int32) for lst in v2c_ep_lists]

    code_name = f"gallager36_N{N}_K{K}_R{rate:.3f}"
    code_cfg = CodeConfig(
        code_name=code_name,
        N=int(N),
        K=int(K),
        rate=float(rate),
        H_path=None,
        checks_to_vars=checks_to_vars,
        vars_to_checks=vars_to_checks,
        var_to_checks_edge_pos=var_to_checks_edge_pos,
    )
    code_cfg.M = int(M)
    code_cfg.encoder_mode = "all_zero"

    prepare_code_for_fast_decoding(code_cfg)

    interleaver = _make_random_interleaver(N, interleaver_seed)
    return code_cfg, interleaver

def _sanitize_prefix(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "awgn_adaptive_hw"


def _set_blas_threads_env_defaults():
    """Prevent BLAS/OpenMP oversubscription when we also use Numba + multiprocessing."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")


def _snr_parallel_plan(snr_sweep: List[float], total_threads: int):
    """Plan SNR-parallelism (processes over SNRs + Numba threads inside each process).

    Environment knobs (all optional):
      - PARALLEL_OVER_SNR: 1/0 (default 1)
      - SNR_PARALLEL_JOBS: number of processes (default auto)
      - SNR_THREADS_PER_WORKER: Numba threads per worker (default auto)
      - SNR_MIN_THREADS_PER_WORKER: minimum threads/worker for auto plan (default 8)
      - JOBLIB_BACKEND: loky|multiprocessing (default loky)
    """
    n_tasks = int(len(snr_sweep))
    total_threads = int(max(1, total_threads))

    if n_tasks <= 1:
        return False, 1, total_threads, "serial"

    if not JOBLIB_AVAILABLE:
        return False, 1, total_threads, "serial"

    if _env_int("PARALLEL_OVER_SNR", 1) != 1:
        return False, 1, total_threads, "serial"

    backend = (os.environ.get("JOBLIB_BACKEND", "loky") or "loky").strip().lower()
    if backend not in ("loky", "multiprocessing"):
        backend = "loky"

    # User overrides
    n_jobs_env = _env_int("SNR_PARALLEL_JOBS", 0)
    t_per_env = _env_int("SNR_THREADS_PER_WORKER", 0)
    min_threads = max(1, _env_int("SNR_MIN_THREADS_PER_WORKER", 8))

    if t_per_env > 0:
        threads_per_job = int(max(1, min(t_per_env, total_threads)))
        n_jobs = int(max(1, min(n_tasks, total_threads // threads_per_job)))
        if n_jobs_env > 0:
            n_jobs = int(max(1, min(n_tasks, n_jobs_env)))
            threads_per_job = int(max(1, total_threads // n_jobs))
    else:
        if n_jobs_env > 0:
            n_jobs = int(max(1, min(n_tasks, n_jobs_env)))
        else:
            n_jobs = int(max(1, min(n_tasks, total_threads // min_threads)))
        threads_per_job = int(max(1, total_threads // n_jobs))

    n_jobs = int(max(1, min(n_tasks, n_jobs)))
    threads_per_job = int(max(1, min(total_threads, threads_per_job)))

    # Parallel only if at least 2 processes
    use_parallel = (n_jobs > 1)
    return use_parallel, n_jobs, threads_per_job, backend


def run_awgn_sweep_for_code(
    code_cfg: CodeConfig,
    interleaver: InterleaverConfig,
    snr_sweep: List[float],
    mc_cfg_local: AdaptiveMCConfig,
    output_dir: str,
    alpha: float = 0.8,
) -> Dict[float, Dict[str, Any]]:
    """Run AWGN sweeps and save outputs via save_awgn_results.

    Parallelism:
      - per-kernel Numba parallelism (already in the decoder + GRAND kernels)
      - OPTIONAL process-level parallelism over SNRs (joblib), with Numba threads
        partitioned across workers to saturate the SLURM allocation.
    """
    _set_blas_threads_env_defaults()

    channel_name = os.environ.get("CHANNEL_NAME", "SIONNA_TDL").strip() or "SIONNA_TDL"

    # Enforce cleaned channel support (fail fast before spawning joblib workers)
    if channel_name.strip().upper() not in ("SIONNA_TDL", "TDL"):
        raise ValueError(
            f"Unsupported CHANNEL_NAME='{channel_name}'. "
            "This cleaned script supports only CHANNEL_NAME=SIONNA_TDL."
        )
    channel_name = "SIONNA_TDL"

    stage1_list_env = str(os.environ.get("STAGE1_ITERS", "4,8,15"))
    stage1_list = [int(x) for x in stage1_list_env.split(",") if x.strip()]
    stage1_list = sorted(set([it for it in stage1_list if it > 0]))


    ldpc_list_env = str(os.environ.get("LDPC_ITERS", "4,8,15,20,100"))
    ldpc_list = [int(x) for x in ldpc_list_env.split(",") if x.strip()]
    ldpc_list = sorted(set([it for it in ldpc_list if it > 0]))

    

    base_seed = _env_int("RNG_SEED_GLOBAL", 12345) + _stable_u32_seed_from_string(code_cfg.code_name)
    run_receiver1 = bool(_env_int("RUN_RECEIVER1", 1))
    run_receiver2 = bool(_env_int("RUN_RECEIVER2", 0))
    run_receiver3 = bool(_env_int("RUN_RECEIVER3", 0))
    run_receiver4 = bool(_env_int("RUN_RECEIVER4", 0))
    run_receiver5 = bool(_env_int("RUN_RECEIVER5", 0))
    run_receiver6 = bool(_env_int("RUN_RECEIVER6", 0))
    run_receiver7 = bool(_env_int("RUN_RECEIVER7", 0))
    run_receiver8 = bool(_env_int("RUN_RECEIVER8", 0))
    run_receiver9 = bool(_env_int("RUN_RECEIVER9", 0))
    pair_decoder_streams = bool(_env_int("PAIR_DECODER_STREAMS", 0))

    results: Dict[float, Dict[str, Any]] = {}

    # Total threads available (as detected in CELL 1); fallback to SLURM/OS
    total_threads = int(globals().get("NUMBA_THREADS", 0) or _detect_num_threads())
    use_parallel, n_jobs, threads_per_job, backend = _snr_parallel_plan(snr_sweep, total_threads)

    if use_parallel:
        print(f"[run_awgn_sweep_for_code] SNR-parallel ON: n_jobs={n_jobs}, threads/worker={threads_per_job}, backend={backend}")
    else:
        # Serial: give Numba the full thread budget
        if NUMBA_AVAILABLE and set_num_threads is not None:
            try:
                set_num_threads(total_threads)
            except Exception:
                pass
        print(f"[run_awgn_sweep_for_code] SNR-parallel OFF (serial); Numba threads={total_threads}")

    def _run_one_snr(snr_db: float):
        snr_db = float(snr_db)

                # In a worker process: partition Numba threads to avoid oversubscription
        if use_parallel and NUMBA_AVAILABLE and set_num_threads is not None:
            try:
                set_num_threads(int(threads_per_job))
            except Exception:
                pass

        snr_seed_common = int((base_seed + int(round(snr_db * 100.0))) & 0xFFFFFFFF)

        def _decoder_seed(family_offset: int, it: int) -> int:
            if pair_decoder_streams:
                return snr_seed_common
            return int((base_seed + int(family_offset) + int(round(snr_db * 100.0)) + int(it)) & 0xFFFFFFFF)

        # LDPC-only sim config (no snapshots needed)
        sim_cfg_ldpc = SimulationConfig(
            code=code_cfg,
            channel=ChannelConfig(name=channel_name, snr_db=snr_db),
            interleaver=interleaver,
            rng_seed_global=int(base_seed),
            snapshot_iters=[],
        )

        per_snr = {}

        # Scenario 1: Legacy LDPC-only baselines (no GRAND)
        for it in ldpc_list:
            dec_name = f"ldpc{int(it)}"
            seed = _decoder_seed(1_000, int(it))
            dec_cfg = DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)
            per_snr[dec_name] = run_ldpc_min_sum_adaptive(
                sim_cfg=sim_cfg_ldpc,
                dec_cfg=dec_cfg,
                mc_cfg=mc_cfg_local,
                rng_seed=seed,
                label=dec_name,
            )

        # Scenario 3: Complete hybrid (Receiver 1 = LLR-ranked GRAND rescue)
        if run_receiver1:
            for it in stage1_list:
                dec_name = f"hyb{int(it)}"
                snapshot_schedule = _resolve_grand_snapshot_schedule(int(it))
                seed = _decoder_seed(10_000, int(it))
                dec_cfg = DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)

                sim_cfg_hyb = SimulationConfig(
                    code=code_cfg,
                    channel=ChannelConfig(name=channel_name, snr_db=snr_db),
                    interleaver=interleaver,
                    rng_seed_global=int(base_seed),
                    snapshot_iters=snapshot_schedule,  # probe several stage-1 snapshots, not just the final one
                )

                per_snr[dec_name] = run_hybrid_ldpc_grand_adaptive(
                    sim_cfg=sim_cfg_hyb,
                    dec_cfg_stage1=dec_cfg,
                    grand_cfg=grand_cfg_awgn,
                    snapshot_iter=snapshot_schedule,
                    mc_cfg=mc_cfg_local,
                    rng_seed=seed,
                    label=dec_name,
                    grand_cfg_boost=(grand_cfg_awgn_boost if GRAND_USE_BOOST else None),
                )

        # Scenario 4: Receiver 2 (syndrome-vote + check-cover front-end)
        if run_receiver2:
            for it in stage1_list:
                dec_name = f"hybsv{int(it)}"
                snapshot_schedule = _resolve_grand_snapshot_schedule(int(it))
                seed = _decoder_seed(20_000, int(it))
                dec_cfg = DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)

                sim_cfg_hyb = SimulationConfig(
                    code=code_cfg,
                    channel=ChannelConfig(name=channel_name, snr_db=snr_db),
                    interleaver=interleaver,
                    rng_seed_global=int(base_seed),
                    snapshot_iters=snapshot_schedule,  # probe several stage-1 snapshots, not just the final one
                )

                per_snr[dec_name] = run_hybrid_ldpc_grand_adaptive(
                    sim_cfg=sim_cfg_hyb,
                    dec_cfg_stage1=dec_cfg,
                    grand_cfg=grand_cfg_awgn_sv,
                    snapshot_iter=snapshot_schedule,
                    mc_cfg=mc_cfg_local,
                    rng_seed=seed,
                    label=dec_name,
                    grand_cfg_boost=(grand_cfg_awgn_sv_boost if GRAND_SV_USE_BOOST else None),
                )

        # Scenario 5: Receiver 3+ (syndrome-vote + peel/weighted-GF(2) pre-solver + GRAND fallback)
        if run_receiver3:
            for it in stage1_list:
                dec_name = f"hybptg{int(it)}"
                snapshot_schedule = _resolve_grand_snapshot_schedule(int(it))
                seed = _decoder_seed(30_000, int(it))
                dec_cfg = DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)

                sim_cfg_hyb = SimulationConfig(
                    code=code_cfg,
                    channel=ChannelConfig(name=channel_name, snr_db=snr_db),
                    interleaver=interleaver,
                    rng_seed_global=int(base_seed),
                    snapshot_iters=snapshot_schedule,
                )

                per_snr[dec_name] = run_hybrid_ldpc_grand_adaptive(
                    sim_cfg=sim_cfg_hyb,
                    dec_cfg_stage1=dec_cfg,
                    grand_cfg=grand_cfg_awgn_ptg,
                    snapshot_iter=snapshot_schedule,
                    mc_cfg=mc_cfg_local,
                    rng_seed=seed,
                    label=dec_name,
                    grand_cfg_boost=(grand_cfg_awgn_ptg_boost if GRAND_PTG_USE_BOOST else None),
                )

        # Scenario 6: Receiver 4 (Chase-list + short-LDPC polish + peel + GRAND fallback)
        if run_receiver4:
            for it in stage1_list:
                dec_name = f"hybctg{int(it)}"
                snapshot_schedule = _resolve_grand_snapshot_schedule(int(it))
                seed = _decoder_seed(40_000, int(it))
                dec_cfg = DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)

                sim_cfg_hyb = SimulationConfig(
                    code=code_cfg,
                    channel=ChannelConfig(name=channel_name, snr_db=snr_db),
                    interleaver=interleaver,
                    rng_seed_global=int(base_seed),
                    snapshot_iters=snapshot_schedule,
                )

                per_snr[dec_name] = run_hybrid_ldpc_grand_adaptive(
                    sim_cfg=sim_cfg_hyb,
                    dec_cfg_stage1=dec_cfg,
                    grand_cfg=grand_cfg_awgn_ctg,
                    snapshot_iter=snapshot_schedule,
                    mc_cfg=mc_cfg_local,
                    rng_seed=seed,
                    label=dec_name,
                    grand_cfg_boost=(grand_cfg_awgn_ctg_boost if GRAND_CTG_USE_BOOST else None),
                )

        # Scenario 7: Receiver 5 (local OSD + anchored full-graph restarts + peel + GRAND fallback)
        if run_receiver5:
            for it in stage1_list:
                dec_name = f"hybosd{int(it)}"
                snapshot_schedule = _resolve_grand_snapshot_schedule(int(it))
                seed = _decoder_seed(50_000, int(it))
                dec_cfg = DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)

                sim_cfg_hyb = SimulationConfig(
                    code=code_cfg,
                    channel=ChannelConfig(name=channel_name, snr_db=snr_db),
                    interleaver=interleaver,
                    rng_seed_global=int(base_seed),
                    snapshot_iters=snapshot_schedule,
                )

                per_snr[dec_name] = run_hybrid_ldpc_grand_adaptive(
                    sim_cfg=sim_cfg_hyb,
                    dec_cfg_stage1=dec_cfg,
                    grand_cfg=grand_cfg_awgn_osd,
                    snapshot_iter=snapshot_schedule,
                    mc_cfg=mc_cfg_local,
                    rng_seed=seed,
                    label=dec_name,
                    grand_cfg_boost=(grand_cfg_awgn_osd_boost if GRAND_OSD_USE_BOOST else None),
                )

        # Scenario 8: Receiver 6 (soft local hypotheses + anchored restarts + peel + GRAND fallback)
        if run_receiver6:
            for it in stage1_list:
                dec_name = f"hybahr{int(it)}"
                snapshot_schedule = _resolve_grand_snapshot_schedule(int(it))
                seed = _decoder_seed(60_000, int(it))
                dec_cfg = DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)

                sim_cfg_hyb = SimulationConfig(
                    code=code_cfg,
                    channel=ChannelConfig(name=channel_name, snr_db=snr_db),
                    interleaver=interleaver,
                    rng_seed_global=int(base_seed),
                    snapshot_iters=snapshot_schedule,
                )

                per_snr[dec_name] = run_hybrid_ldpc_grand_adaptive(
                    sim_cfg=sim_cfg_hyb,
                    dec_cfg_stage1=dec_cfg,
                    grand_cfg=grand_cfg_awgn_ahr,
                    snapshot_iter=snapshot_schedule,
                    mc_cfg=mc_cfg_local,
                    rng_seed=seed,
                    label=dec_name,
                    grand_cfg_boost=(grand_cfg_awgn_ahr_boost if GRAND_AHR_USE_BOOST else None),
                )

        # Scenario 9: Receiver 7 (basis-GRAND + block-debias anchored restarts + peel + GRAND fallback)
        if run_receiver7:
            for it in stage1_list:
                dec_name = f"hybbgr{int(it)}"
                snapshot_schedule = _resolve_grand_snapshot_schedule(int(it))
                seed = _decoder_seed(70_000, int(it))
                dec_cfg = DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)

                sim_cfg_hyb = SimulationConfig(
                    code=code_cfg,
                    channel=ChannelConfig(name=channel_name, snr_db=snr_db),
                    interleaver=interleaver,
                    rng_seed_global=int(base_seed),
                    snapshot_iters=snapshot_schedule,
                )

                per_snr[dec_name] = run_hybrid_ldpc_grand_adaptive(
                    sim_cfg=sim_cfg_hyb,
                    dec_cfg_stage1=dec_cfg,
                    grand_cfg=grand_cfg_awgn_bgr,
                    snapshot_iter=snapshot_schedule,
                    mc_cfg=mc_cfg_local,
                    rng_seed=seed,
                    label=dec_name,
                    grand_cfg_boost=(grand_cfg_awgn_bgr_boost if GRAND_BGR_USE_BOOST else None),
                )

        # Scenario 10: Receiver 8 (cascade: strong AHR primary, BGR fallback on the same snapshots)
        if run_receiver8:
            for it in stage1_list:
                dec_name = f"hybmeta{int(it)}"
                snapshot_schedule = _resolve_grand_snapshot_schedule(int(it))
                seed = _decoder_seed(80_000, int(it))
                dec_cfg = DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)

                sim_cfg_hyb = SimulationConfig(
                    code=code_cfg,
                    channel=ChannelConfig(name=channel_name, snr_db=snr_db),
                    interleaver=interleaver,
                    rng_seed_global=int(base_seed),
                    snapshot_iters=snapshot_schedule,
                )

                per_snr[dec_name] = run_hybrid_ldpc_grand_adaptive(
                    sim_cfg=sim_cfg_hyb,
                    dec_cfg_stage1=dec_cfg,
                    grand_cfg=grand_cfg_awgn_meta,
                    snapshot_iter=snapshot_schedule,
                    mc_cfg=mc_cfg_local,
                    rng_seed=seed,
                    label=dec_name,
                    grand_cfg_boost=(grand_cfg_awgn_meta_boost if GRAND_AHR_USE_BOOST else None),
                    grand_cfg_fallback=(grand_cfg_awgn_bgr if GRAND_META_USE_FALLBACK else None),
                    grand_cfg_boost_fallback=(grand_cfg_awgn_bgr_boost if (GRAND_META_USE_FALLBACK and GRAND_BGR_USE_BOOST) else None),
                    fallback_label="basis_anchor",
                )

        # Scenario 11: Receiver 9 (AI-gated budgeted hybrid GRAND)
        if run_receiver9:
            for it in stage1_list:
                policy_mode = str(getattr(ai_gate_cfg_awgn_air, "policy_mode", "linear_ucb") or "linear_ucb").strip().lower()
                selection_mode = str(getattr(grand_cfg_awgn_air_full, "selection_mode", "llr") or "llr").strip().lower()
                roi_modes = ("ai_rank_roi", "airoi", "roi_rank", "receiver9_roi")
                mix_modes = ("ai_mix_roi", "aimix", "mix_roi", "receiver9_mix")
                window_modes = ("ai_window_roi", "aiwindow", "window_roi", "receiver9_window")
                graph_modes = ("ai_tanner_subgraph_roi", "aitg2", "tanner_subgraph_roi", "receiver9_tg2", "ai_tanner_roi", "aitg", "tanner_roi", "receiver9_tg")
                if policy_mode in ("distilled_tree_bandit", "tree_bandit", "dt_bandit"):
                    if selection_mode in window_modes:
                        dec_name = f"hybairdtbwin{int(it)}"
                    elif selection_mode in roi_modes:
                        dec_name = f"hybairdtbroi{int(it)}"
                    else:
                        dec_name = f"hybairdtb{int(it)}"
                elif policy_mode in ("distilled_tree", "tree", "dt"):
                    if selection_mode in window_modes:
                        dec_name = f"hybairdtwin{int(it)}"
                    elif selection_mode in roi_modes:
                        dec_name = f"hybairdtroi{int(it)}"
                    else:
                        dec_name = f"hybairdt{int(it)}"
                elif policy_mode in ("distilled_tree_roi", "tree_roi", "dt_roi"):
                    if selection_mode in window_modes:
                        dec_name = f"hybairwroi{int(it)}"
                    else:
                        dec_name = f"hybairroi{int(it)}"
                elif policy_mode in ("probe_moe_roi_fix", "probe_moe_roi", "probe_moe", "probe_fix", "probe"):
                    if selection_mode in ("ai_tanner_subgraph_roi", "aitg2", "tanner_subgraph_roi", "receiver9_tg2"):
                        dec_name = f"hybairtgsub{int(it)}"
                    elif selection_mode in graph_modes:
                        dec_name = f"hybairtg{int(it)}"
                    elif selection_mode in mix_modes:
                        dec_name = f"hybairpmix{int(it)}"
                    elif selection_mode in window_modes:
                        dec_name = f"hybairpwin{int(it)}"
                    else:
                        dec_name = f"hybairprobe{int(it)}"
                else:
                    dec_name = f"hybair{int(it)}"
                snapshot_schedule = _resolve_grand_snapshot_schedule(int(it))
                seed = _decoder_seed(90_000, int(it))
                dec_cfg = DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)

                sim_cfg_hyb = SimulationConfig(
                    code=code_cfg,
                    channel=ChannelConfig(name=channel_name, snr_db=snr_db),
                    interleaver=interleaver,
                    rng_seed_global=int(base_seed),
                    snapshot_iters=snapshot_schedule,
                )

                expected_decoder = os.environ.get("EXPECTED_HYBRID_DECODER", "").strip()
                expected_policy = os.environ.get("EXPECTED_HYBRID_POLICY", "").strip().lower()
                expected_selection = os.environ.get("EXPECTED_HYBRID_SELECTION", "").strip().lower()
                actual_selection = selection_mode
                if expected_policy and policy_mode != expected_policy:
                    raise RuntimeError(f"Receiver9 policy mismatch: expected {expected_policy}, got {policy_mode}")
                if expected_selection and actual_selection != expected_selection:
                    raise RuntimeError(f"Receiver9 selection mismatch: expected {expected_selection}, got {actual_selection}")
                if expected_decoder and dec_name != expected_decoder:
                    raise RuntimeError(f"Receiver9 decoder mismatch: expected {expected_decoder}, got {dec_name}")
                print(f"[VERIFY] Receiver9 decoder={dec_name} policy={policy_mode} selection={actual_selection} snapshots={snapshot_schedule}")

                per_snr[dec_name] = run_hybrid_ldpc_grand_adaptive(
                    sim_cfg=sim_cfg_hyb,
                    dec_cfg_stage1=dec_cfg,
                    grand_cfg=grand_cfg_awgn_air_full,
                    snapshot_iter=snapshot_schedule,
                    mc_cfg=mc_cfg_local,
                    rng_seed=seed,
                    label=dec_name,
                    grand_cfg_boost=(grand_cfg_awgn_air_full_boost if GRAND_AIR_USE_BOOST else None),
                    grand_cfg_fallback=(grand_cfg_awgn_air_fallback if GRAND_AIR_USE_FALLBACK else None),
                    grand_cfg_boost_fallback=(grand_cfg_awgn_air_fallback_boost if (GRAND_AIR_USE_FALLBACK and GRAND_BGR_USE_BOOST) else None),
                    fallback_label="basis_anchor",
                    ai_gate_cfg=ai_gate_cfg_awgn_air,
                    grand_cfg_tiny=grand_cfg_awgn_air_tiny,
                    grand_cfg_boost_tiny=None,
                )

        return snr_db, per_snr

    if use_parallel:
        tasks = Parallel(n_jobs=n_jobs, backend=backend, prefer="processes", batch_size=1)(
            delayed(_run_one_snr)(snr_db) for snr_db in snr_sweep
        )
        for snr_db, per_snr in tasks:
            results[float(snr_db)] = per_snr
    else:
        for snr_db in snr_sweep:
            snr_db_f, per_snr = _run_one_snr(snr_db)
            results[float(snr_db_f)] = per_snr

    prefix = _sanitize_prefix(f"{channel_name.lower()}_{code_cfg.code_name}_hybrid")
    save_awgn_results(results, output_dir=output_dir, prefix=prefix)
    return results


def _run_experiments_main():
    # Guarded entry point for SLURM (safe for joblib multiprocessing)
    if int(float(os.environ.get("RUN_EXPERIMENTS", "0") or "0")) != 1:
        return

    run_codes_env = os.environ.get("RUN_CODES", "").strip()
    run_codes = [c.strip().lower() for c in run_codes_env.split(",") if c.strip()] if run_codes_env else []

    if not run_codes:
        run_codes = ["sionna5g"]

    snr_sweep = _snr_sweep_from_env(snr_sweep_global)
    mc_cfg_local = _mc_cfg_from_env(mc_cfg)

    out_dir = os.environ.get("RESULTS_DIR", "./results")
    os.makedirs(out_dir, exist_ok=True)

    interleaver_seed = _env_int("INTERLEAVER_SEED", 2025)

    print(f"[RUN_EXPERIMENTS] codes={run_codes}  snr_sweep={snr_sweep}  out_dir={out_dir}")
    print(f"[RUN_EXPERIMENTS] mc_cfg={mc_cfg_local}")

    for code_token in run_codes:
        # Only keep the Sionna 5G NR LDPC path (requested).
        if code_token not in ("sionna5g", "5g_sionna", "sionna_5g"):
            print(f"[RUN_EXPERIMENTS] WARNING: unsupported RUN_CODES token '{code_token}' (only 'sionna5g' is supported) -> skipping.")
            continue

        # 5G NR LDPC from Sionna (38.212 compliant, RV=0-style rate-matching)
        k_info = _env_int("SIONNA_5G_K", 1024)
        n_tx = _env_int("SIONNA_5G_N", 2048)
        qm = _env_int("SIONNA_5G_QM", 1)
        code_cfg_local, interleaver_local = build_sionna_5g_nr_code_cfg(
            k_info=k_info,
            n_tx=n_tx,
            num_bits_per_symbol=qm,
            code_name_prefix="sionna5g",
        )

        print(f"[RUN_EXPERIMENTS] Built code: {code_cfg_local.code_name}")
        run_awgn_sweep_for_code(
            code_cfg=code_cfg_local,
            interleaver=interleaver_local,
            snr_sweep=snr_sweep,
            mc_cfg_local=mc_cfg_local,
            output_dir=out_dir,
            alpha=0.8,
        )


if __name__ == "__main__":
    # sbatch/CLI convenience:
    #   python <script.py> <output_dir> <code_token>
    # If args are provided, they override the corresponding environment variables.
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        os.environ["RESULTS_DIR"] = sys.argv[1].strip()
    if len(sys.argv) >= 3 and sys.argv[2].strip():
        os.environ["RUN_CODES"] = sys.argv[2].strip()

    # Default: run when executed as a script
    os.environ.setdefault("RUN_EXPERIMENTS", "1")
    _run_experiments_main()
