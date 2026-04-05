#!/bin/bash
#SBATCH --job-name=hyb_tsbcai_20260404
#SBATCH --account=def-rsadve
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --array=0-2%3
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

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
export LDPC_GRAND_NUM_THREADS="${LDPC_GRAND_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-32}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Same realistic 5G receiver chain for legacy LDPC and hybrid.
# After LDPC reaches max It and still fails, the hybrid GRAND uses the last few LDPC snapshots
# from that same completed LDPC run to choose a better starting point.
export CHANNEL_NAME=SIONNA_TDL
export SIONNA_5G_K=1024
export SIONNA_5G_N=2048
export SIONNA_5G_QM=1
export SIONNA_TDL_MODEL=B
export SIONNA_TDL_DELAY_SPREAD_S=1.0e-7
export SIONNA_TDL_CARRIER_FREQUENCY_HZ=3.5e9
export SIONNA_TDL_MIN_SPEED=0.5
export SIONNA_TDL_MAX_SPEED=3.0
export SIONNA_OFDM_FFT_SIZE=512
export SIONNA_OFDM_SUBCARRIER_SPACING_HZ=15000
export SIONNA_CFO_HZ=15.0
export SIONNA_CSI_MODE=nr_imperfect
export SIONNA_CSI_PILOT_STRIDE=8
export SIONNA_CSI_SMOOTH_PILOTS=3
export SIONNA_CSI_EST_SNR_DB=24
export SIONNA_CSI_PHASE_DRIFT_STD_DEG=0.30
export SIONNA_CSI_AMP_RIPPLE_DB=0.12
export SIONNA_CSI_DELAY_BIAS_NS=2.0
export SIONNA_CSI_BLOCK_SC=8
export SIONNA_LLR_MAX=50.0

export LDPC_ALPHA=0.80
export TARGET_FRAME_ERRORS=120
export MAX_FRAMES=180000
export CALIB_TARGET_FAILED_FRAMES=640
export CALIB_MAX_FRAMES=260000
export CALIB_NEG_RATIO=6.0
export CALIB_HARD_NEG_CAP=128
export AI_TEACHER_HIDDEN=64
export AI_TEACHER_EPOCHS=24
export AI_STUDENT_EPOCHS=20
export AI_BATCH_SIZE=4096
export AI_TEACHER_LR=0.010
export AI_STUDENT_LR=0.022
export AI_DISTILL_TEMP=2.5

# Train on the easiest late snapshots of each failed frame instead of only the final failed snapshot.
export CALIB_TX_INFO_POOL=32
export CALIB_PUNCTURED_INFO_POOL=18
export CALIB_TX_PARITY_POOL=14
export CALIB_WEIGHT_CAP=12
export CALIB_FREE_DIM_CAP=14
export CALIB_MAX_CANDIDATES=8192

# TSBCAI: trajectory-selected snapshot + block-correlated AI ranking + exact parity-constrained solve.
export GRAND_TX_INFO_POOL=26
export GRAND_TX_PARITY_POOL=12
export GRAND_PUNCTURED_INFO_POOL=18
export GRAND_ROUND2_EXTRA_INFO=12
export GRAND_ROUND2_EXTRA_PUNCTURED=8
export GRAND_ROUND2_EXTRA_PARITY=6
export GRAND_MAX_WEIGHT=20
export GRAND_FREE_DIM_CAP=20
export GRAND_MAX_PATTERNS=6144
export GRAND_FRAME_GATE_THRESHOLD=0.0
export GRAND_GATE_MAX_SYND=999
export GRAND_GATE_MAX_HARD_ONES=999
export GRAND_AI_WEIGHT=1.60
export GRAND_GAIN_WEIGHT=0.24
export GRAND_OSC_WEIGHT=0.18
export GRAND_LLR_WEIGHT=1.00
export GRAND_SW_WEIGHT=1.20
export GRAND_INFO_BONUS=0.78
export GRAND_TX_BONUS=0.16
export GRAND_PUNCTURED_INFO_BONUS=0.44
export GRAND_PARITY_PENALTY=0.18
export GRAND_BLOCK_PROB_WEIGHT=0.95
export GRAND_BLOCK_MASS_WEIGHT=0.55
export GRAND_BLOCK_COMBO_MAX=3
export GRAND_TOP_BLOCKS=10
export GRAND_BLOCK_BEAM_WIDTH=64
export GRAND_PREFIX_KEEP=8
export GRAND_BLOCK_REFINE_BITS=16
export GRAND_GLOBAL_TOP_BITS=28
export GRAND_DIRECT_TOP_BITS=8
export GRAND_EXACT_POOL_CAP=72
export GRAND_BLOCK_MASK_VARIANTS=8

export GRAND_TRAJ_DEPTH=5
export GRAND_TRAJ_TRY_TOP=2
export GRAND_TRAJ_INFO_WEIGHT=2.20
export GRAND_TRAJ_PUNCTURED_WEIGHT=1.35
export GRAND_TRAJ_PARITY_WEIGHT=0.45
export GRAND_TRAJ_SYND_WEIGHT=0.10
export GRAND_TRAJ_BEST_SCALE=1.80
export GRAND_TRAJ_SECOND_SCALE=1.25

export RNG_SEED_GLOBAL=20260404

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    export STAGE1_ITERS=8
    export CALIB_SNR_SWEEP="6.5,7.0,7.5,8.0"
    export EVAL_SNR_SWEEP="7.0,7.5,8.0,8.5,9.0"
    ITER_TAG="it08"
    ;;
  1)
    export STAGE1_ITERS=15
    export CALIB_SNR_SWEEP="7.0,7.5,8.0,8.5"
    export EVAL_SNR_SWEEP="7.5,8.0,8.5,9.0,9.5"
    ITER_TAG="it15"
    ;;
  2)
    export STAGE1_ITERS=25
    export CALIB_SNR_SWEEP="7.5,8.0,8.5,9.0"
    export EVAL_SNR_SWEEP="8.0,8.5,9.0,9.5,10.0"
    ITER_TAG="it25"
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 4
    ;;
esac

RESULTS_DIR="results/run_hybrid_ldpc_grand_tsbcai_20260404_${ITER_TAG}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RESULTS_DIR"

printf '[SBATCH CONFIG] task=%s it=%s eval_snr=%s calib_snr=%s\n' "$SLURM_ARRAY_TASK_ID" "$STAGE1_ITERS" "$EVAL_SNR_SWEEP" "$CALIB_SNR_SWEEP"
printf '[SBATCH CONFIG] channel=%s tdl=%s delay=%s speed=[%s,%s] cfo=%s csi=%s\n' \
  "$CHANNEL_NAME" "$SIONNA_TDL_MODEL" "$SIONNA_TDL_DELAY_SPREAD_S" \
  "$SIONNA_TDL_MIN_SPEED" "$SIONNA_TDL_MAX_SPEED" "$SIONNA_CFO_HZ" "$SIONNA_CSI_MODE"
printf '[SBATCH CONFIG] traj depth/try=%s/%s pools info/punc/txp=%s/%s/%s blocks=%s beam=%s exact_pool=%s max_w=%s free_dim=%s\n' \
  "$GRAND_TRAJ_DEPTH" "$GRAND_TRAJ_TRY_TOP" "$GRAND_TX_INFO_POOL" "$GRAND_PUNCTURED_INFO_POOL" "$GRAND_TX_PARITY_POOL" \
  "$GRAND_TOP_BLOCKS" "$GRAND_BLOCK_BEAM_WIDTH" "$GRAND_EXACT_POOL_CAP" "$GRAND_MAX_WEIGHT" "$GRAND_FREE_DIM_CAP"

python -u hybrid_ldpc_grand_tsbcai_20260404.py "$RESULTS_DIR" \
  2> >(grep -vE 'oneDNN custom operations are on|Unable to register cuFFT factory|All log messages before absl::InitializeLog|Unable to register cuDNN factory|Unable to register cuBLAS factory|computation placer already registered' >&2)
