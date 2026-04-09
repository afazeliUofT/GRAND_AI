#!/bin/bash
#SBATCH --job-name=hyb_pexog_20260409
#SBATCH --account=def-rsadve
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --array=0-2%3
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

cd /home/rsadve1/scratch/GRAND_AI

module --force purge
module load StdEnv/2023 python/3.11 scipy-stack/2024a

VENV_DIR="${VENV_DIR:-$SCRATCH/venvs/grand5g}"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: expected Python venv at $VENV_DIR" >&2
  exit 2
fi
source "$VENV_DIR/bin/activate"

python - <<'PY'
import importlib, json, sys
status = {}
errors = {}
for mod in ("numpy", "scipy", "numba", "sionna"):
    try:
        importlib.import_module(mod)
        status[f"{mod}_ok"] = True
    except Exception as e:
        status[f"{mod}_ok"] = False
        errors[mod] = repr(e)
ldpc_paths = [
    "sionna.phy.fec.ldpc.encoding",
    "sionna.phy.fec.ldpc",
    "sionna.fec.ldpc.encoding",
    "sionna.fec.ldpc",
]
tdl_paths = [
    "sionna.phy.channel.tr38901.tdl",
    "sionna.phy.channel.tr38901",
    "sionna.channel.tr38901.tdl",
    "sionna.channel.tr38901",
]
status["LDPC5GEncoder_ok"] = False
status["TDL_ok"] = False
ldpc_errs = []
for path in ldpc_paths:
    try:
        mod = importlib.import_module(path)
        getattr(mod, "LDPC5GEncoder")
        status["LDPC5GEncoder_ok"] = True
        break
    except Exception as e:
        ldpc_errs.append(f"{path}: {e!r}")
if not status["LDPC5GEncoder_ok"]:
    errors["LDPC5GEncoder"] = " | ".join(ldpc_errs)
tdl_errs = []
for path in tdl_paths:
    try:
        mod = importlib.import_module(path)
        getattr(mod, "TDL")
        status["TDL_ok"] = True
        break
    except Exception as e:
        tdl_errs.append(f"{path}: {e!r}")
if not status["TDL_ok"]:
    errors["TDL"] = " | ".join(tdl_errs)
payload = {"python": sys.executable, "status": status, "errors": errors}
print(json.dumps(payload, indent=2, sort_keys=True))
needed = ["numpy_ok", "scipy_ok", "numba_ok", "sionna_ok", "LDPC5GEncoder_ok", "TDL_ok"]
if not all(status.get(k, False) for k in needed):
    raise SystemExit(3)
PY

export TMPDIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$TMPDIR/numba_cache}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-$TMPDIR/joblib}"
mkdir -p "$NUMBA_CACHE_DIR" "$JOBLIB_TEMP_FOLDER" results

export USE_GPU=0
export CUDA_VISIBLE_DEVICES="-1"
export NVIDIA_VISIBLE_DEVICES="void"
export TF_CPP_MIN_LOG_LEVEL=3
export ABSL_LOG_LEVEL=3
export GLOG_minloglevel=3
export TF_ENABLE_ONEDNN_OPTS=0
export TF_INTRA_OP=1
export TF_INTER_OP=1
export LDPC_GRAND_NUM_THREADS="${LDPC_GRAND_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-64}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LDPC_GRAND_DISABLE_NUMBA=0
export NUMBA_THREADING_LAYER=workqueue
export EVAL_SNR_WORKERS="${EVAL_SNR_WORKERS:-8}"

# Same realistic 5G receiver chain for legacy LDPC and hybrid.
# PEXOG changes:
# 1) bounded best-first queue (no heapq.nsmallest bottleneck)
# 2) first valid codeword wins in posterior order
# 3) 50K hard cap per failed frame
# 4) nonlinear distilled students + parallel SNR evaluation using 64 CPUs

export CHANNEL_NAME=SIONNA_TDL
export SIONNA_5G_K=1024
export SIONNA_5G_N=2048
export SIONNA_5G_QM=1
export SIONNA_TDL_MODEL=B
export SIONNA_TDL_DELAY_SPREAD_S=4.0e-8
export SIONNA_TDL_CARRIER_FREQUENCY_HZ=3.5e9
export SIONNA_TDL_MIN_SPEED=0.1
export SIONNA_TDL_MAX_SPEED=0.5
export SIONNA_OFDM_FFT_SIZE=512
export SIONNA_OFDM_SUBCARRIER_SPACING_HZ=15000
export SIONNA_CFO_HZ=2.0
export SIONNA_CSI_MODE=nr_imperfect
export SIONNA_CSI_PILOT_STRIDE=8
export SIONNA_CSI_SMOOTH_PILOTS=3
export SIONNA_CSI_EST_SNR_DB=30
export SIONNA_CSI_PHASE_DRIFT_STD_DEG=0.7
export SIONNA_CSI_AMP_RIPPLE_DB=0.25
export SIONNA_CSI_DELAY_BIAS_NS=3.0
export SIONNA_CSI_BLOCK_SC=8
export SIONNA_LLR_MAX=50.0

export LDPC_ALPHA=0.80
export TARGET_FRAME_ERRORS=60
export MAX_FRAMES=15000
export CALIB_TARGET_FAILED_FRAMES=1200
export CALIB_MAX_FRAMES=420000
export CALIB_NEG_RATIO=6.0
export CALIB_HARD_NEG_CAP=128

export AI_TEACHER_HIDDEN=224
export AI_STUDENT_HIDDEN=96
export AI_TEACHER_EPOCHS=56
export AI_STUDENT_EPOCHS=42
export AI_BATCH_SIZE=4096
export AI_TEACHER_LR=0.010
export AI_STUDENT_LR=0.018
export AI_DISTILL_TEMP=2.5

export GRAND_TX_INFO_POOL=72
export GRAND_PUNCTURED_INFO_POOL=36
export GRAND_TX_PARITY_POOL=20
export GRAND_SINGLE_POOL=96
export GRAND_TOP_BLOCKS=18
export GRAND_PAIR_TOP_BLOCKS=8
export GRAND_BLOCK_BEAM_WIDTH=48
export GRAND_EXPAND_TOP_K=16
export GRAND_MAX_ATOMS_PER_PATTERN=4
export GRAND_MAX_PATTERNS=50000

export GRAND_ROUND2_EXTRA_INFO=24
export GRAND_ROUND2_EXTRA_PUNCTURED=12
export GRAND_ROUND2_EXTRA_PARITY=8
export GRAND_MAX_WEIGHT=96
export GRAND_FREE_DIM_CAP=16
export GRAND_EXACT_POOL_CAP=64
export GRAND_GLOBAL_TOP_BITS=16
export GRAND_DIRECT_TOP_BITS=8
export GRAND_BLOCK_MASK_VARIANTS=4
export GRAND_PREFIX_KEEP=6
export GRAND_PROB_MIN=1.0e-5
export GRAND_PROB_MAX=0.80
export GRAND_MAX_SUPPORT_BITS=96
export GRAND_QUEUE_CAP=16384
export GRAND_BLOCK_PREFIX_SIZES="2,4,6,8,12,16"

export RNG_SEED_GLOBAL=20260409

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    export STAGE1_ITERS=8
    export CALIB_SNR_SWEEP="7,8,9,10"
    export EVAL_SNR_SWEEP="8,9,10,11,12,13,14,15"
    ITER_TAG="it08"
    ;;
  1)
    export STAGE1_ITERS=15
    export CALIB_SNR_SWEEP="7,8,9,10"
    export EVAL_SNR_SWEEP="8,9,10,11,12,13,14,15"
    ITER_TAG="it15"
    ;;
  2)
    export STAGE1_ITERS=25
    export CALIB_SNR_SWEEP="7,8,9,10"
    export EVAL_SNR_SWEEP="8,9,10,11,12,13,14,15"
    ITER_TAG="it25"
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 4
    ;;
esac

RESULTS_DIR="results/run_hybrid_ldpc_grand_pexog_20260409_${ITER_TAG}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RESULTS_DIR"

printf '[SBATCH CONFIG] task=%s it=%s eval_snr=%s calib_snr=%s\n' "$SLURM_ARRAY_TASK_ID" "$STAGE1_ITERS" "$EVAL_SNR_SWEEP" "$CALIB_SNR_SWEEP"
printf '[SBATCH CONFIG] channel=%s tdl=%s delay=%s speed=[%s,%s] cfo=%s csi=%s\n' "$CHANNEL_NAME" "$SIONNA_TDL_MODEL" "$SIONNA_TDL_DELAY_SPREAD_S" "$SIONNA_TDL_MIN_SPEED" "$SIONNA_TDL_MAX_SPEED" "$SIONNA_CFO_HZ" "$SIONNA_CSI_MODE"
printf '[SBATCH CONFIG] pexog cpus=%s snr_workers=%s max_patterns=%s queue_cap=%s\n' "$SLURM_CPUS_PER_TASK" "$EVAL_SNR_WORKERS" "$GRAND_MAX_PATTERNS" "$GRAND_QUEUE_CAP"

python -u hybrid_ldpc_grand_pexog_20260409.py "$RESULTS_DIR" \
  2> >(grep -vE 'oneDNN custom operations are on|Unable to register cuFFT factory|All log messages before absl::InitializeLog|Unable to register cuDNN factory|Unable to register cuBLAS factory|computation placer already registered' >&2)
