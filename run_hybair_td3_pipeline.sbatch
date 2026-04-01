#!/bin/bash
#SBATCH --job-name=run_hybair_td3_pipeline
#SBATCH --account=def-rsadve
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

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
for mod in ("six", "pyparsing", "sionna"):
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
needed = ["six_ok", "pyparsing_ok", "sionna_ok", "LDPC5GEncoder_ok", "TDL_ok"]
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
export MAX_NB_THREADS="${MAX_NB_THREADS:-${SLURM_CPUS_PER_TASK:-64}}"
export LDPC_GRAND_NUM_THREADS="${LDPC_GRAND_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-64}}"
export PARALLEL_OVER_SNR=0
export PAIR_DECODER_STREAMS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Realistic but not harsh channel.
export CHANNEL_NAME=SIONNA_TDL
export SIONNA_5G_K=1024
export SIONNA_5G_N=2048
export SIONNA_5G_QM=1
export SIONNA_TDL_MODEL=A
export SIONNA_TDL_DELAY_SPREAD_S=1.0e-7
export SIONNA_TDL_CARRIER_FREQUENCY_HZ=3.5e9
export SIONNA_TDL_MIN_SPEED=0.5
export SIONNA_TDL_MAX_SPEED=2.0
export SIONNA_OFDM_FFT_SIZE=512
export SIONNA_OFDM_SUBCARRIER_SPACING_HZ=15000
export SIONNA_CFO_HZ=20.0
export SIONNA_CSI_MODE=block_ls
export SIONNA_CSI_PILOT_STRIDE=12
export SIONNA_CSI_SMOOTH_PILOTS=2
export SIONNA_CSI_EST_SNR_DB=20
export SIONNA_CSI_PHASE_DRIFT_STD_DEG=0.0
export SIONNA_CSI_AMP_RIPPLE_DB=0.0
export SIONNA_CSI_DELAY_BIAS_NS=0.0
export SIONNA_CSI_BLOCK_SC=12
export SIONNA_LLR_MAX=50.0

export FORCE_DEFAULTS=1
export LDPC_ITERS=15
export STAGE1_ITERS=15
export RUN_RECEIVER2=0
export RUN_RECEIVER3=0
export RUN_RECEIVER4=0
export RUN_RECEIVER5=0
export RUN_RECEIVER6=0
export RUN_RECEIVER7=0
export RUN_RECEIVER8=0
export RUN_RECEIVER9=1

# Receiver-9 controller and learned selector.
export GRAND_AIR_GATE_SNAPSHOT=first
export GRAND_AIR_USE_FALLBACK=0
export GRAND_AIR_USE_BOOST=1
export GRAND_AIR_POLICY_MODE=probe_moe_roi_fix
export GRAND_AIR_SELECTION_MODE=ai_teacher_distill_roi_v3
export GRAND_AIR_LLR_SOURCE=mixed
export GRAND_AIR_DYNAMIC_PER_SNAPSHOT=1
export GRAND_AIR_TINY_SNAPSHOT_CAP=2
export GRAND_AIR_FULL_SNAPSHOT_CAP=2
export GRAND_AIR_META_SNAPSHOT_CAP=0
export GRAND_AIR_COST_LAMBDA=0.22
export GRAND_AIR_UCB_ALPHA=0.16
export GRAND_AIR_SKIP_FAILURE_PENALTY=0.30
export GRAND_AIR_WARMUP_MIN_TRIALS_PER_ACTION=4
export GRAND_AIR_WARMUP_SNAPSHOT_CAP=1
export GRAND_AIR_SUPPRESS_SKIP_DURING_WARMUP=1
export GRAND_AIR_COLD_START_BONUS=0.45
export GRAND_AIR_ROI_SCORE_WEIGHT=0.55
export GRAND_AIR_ROI_DISABLE_MIN_TRIALS=4
export GRAND_AIR_ROI_DISABLE_THRESHOLD=-0.01
export GRAND_AIR_ROI_DISABLE_PENALTY=0.25
export GRAND_AIR_ROI_PROMOTE_MIN_TRIALS=3
export GRAND_AIR_ROI_PROMOTE_THRESHOLD=0.06
export GRAND_AIR_ROI_PROMOTE_WEIGHT=0.22

# Base heuristics still matter as cold-start / back-stop.
export GRAND_AIR_AI_VOTE_WEIGHT=1.00
export GRAND_AIR_AI_LLR_WEIGHT=0.82
export GRAND_AIR_AI_DISAGREEMENT_WEIGHT=1.05
export GRAND_AIR_AI_DENSITY_WEIGHT=0.72
export GRAND_AIR_AI_ROI_BLOCK_SIZE=64
export GRAND_AIR_AI_ROI_WEAK_Q=0.30
export GRAND_AIR_AI_ROI_WEAK_ABS=2.50
export GRAND_AIR_AI_ROI_DIFFUSE_UNION=224
export GRAND_AIR_AI_ROI_DIFFUSE_BLOCK=0.08
export GRAND_AIR_AI_ROI_COMPACT_BLOCK=0.10
export GRAND_AIR_AI_ROI_DIFFUSE_L_SCALE=0.74
export GRAND_AIR_AI_ROI_CONFLICT_BONUS=0.45
export GRAND_AIR_AI_WINDOW_BLOCK_SIZE=64
export GRAND_AIR_AI_WINDOW_TINY_TOP_BLOCKS=2
export GRAND_AIR_AI_WINDOW_FULL_TOP_BLOCKS=4
export GRAND_AIR_AI_WINDOW_META_TOP_BLOCKS=5
export GRAND_AIR_AI_WINDOW_NEIGHBOR_BLOCKS=1
export GRAND_AIR_AI_WINDOW_LOCAL_SEEDS=6
export GRAND_AIR_AI_WINDOW_DIFFUSE_EXTRA_BLOCKS=2
export GRAND_AIR_AI_WINDOW_COMPACT_SINGLE=0.16
export GRAND_AIR_AI_WINDOW_CONFLICT_BONUS=0.35
export GRAND_AIR_AI_WINDOW_DENSITY_BONUS=0.20
export GRAND_AIR_AI_TG_BLOCK_SIZE=64
export GRAND_AIR_AI_TG_STATIC_WEIGHT=0.30
export GRAND_AIR_AI_TG_MESSAGE_WEIGHT=1.00
export GRAND_AIR_AI_TG_CYCLE_WEIGHT=0.42
export GRAND_AIR_AI_TG_DIFFUSE_UNION=192
export GRAND_AIR_AI_TG_DIFFUSE_BLOCK=0.09
export GRAND_AIR_AI_TG_COMPACT_BLOCK=0.12
export GRAND_AIR_AI_TG_LOCAL_SHARE_DIFFUSE=0.30
export GRAND_AIR_AI_TG_LOCAL_SHARE_BALANCED=0.48
export GRAND_AIR_AI_TG_LOCAL_SHARE_COMPACT=0.66
export GRAND_AIR_AI_TG_TINY_TOP_BLOCKS=1
export GRAND_AIR_AI_TG_FULL_TOP_BLOCKS=2
export GRAND_AIR_AI_TG_BOOST_TOP_BLOCKS=3
export GRAND_AIR_AI_TG_META_TOP_BLOCKS=3
export GRAND_AIR_AI_TG_NEIGHBOR_BLOCKS=1
export GRAND_AIR_AI_TG_DIFFUSE_EXTRA_BLOCKS=1
export GRAND_AIR_AI_TG_TOP_GLOBAL_EXTRA=10
export GRAND_AIR_PROBE_MIN_DROP_FULL=0.008
export GRAND_AIR_PROBE_MIN_DROP_CONTINUE=0.003
export GRAND_AIR_PROBE_MIN_DROP_HARD=0.015
export GRAND_AIR_PROBE_NEXT_MIN_DROP=0.004
export GRAND_AIR_PROBE_LOCAL_COMPACTNESS=0.06
export GRAND_AIR_PROBE_LOCAL_BLOCK_CONCENTRATION=0.03
export GRAND_AIR_PROBE_LOCAL_CONFLICT=0.012
export GRAND_AIR_PROBE_LOCAL_PROMISE=-0.12
export GRAND_AIR_PROBE_STRONG_COMPACTNESS=0.10
export GRAND_AIR_PROBE_STRONG_BLOCK_CONCENTRATION=0.04
export GRAND_AIR_PROBE_STRONG_CONFLICT=0.016
export GRAND_AIR_PROBE_META_DROP=0.030
export GRAND_AIR_PROBE_FAILRATE_EASY=0.012
export GRAND_AIR_PROBE_FAILRATE_HARD=0.055
export GRAND_AIR_PROBE_FAILRATE_MIN_FRAMES=192
export GRAND_AIR_PROBE_FORCE_TINY_ALL_FAILED=1
export GRAND_AIR_PROBE_LATE_SNAPSHOT_CAP=2

# Distillation controls.
RESULTS_DIR_FINAL="results/run_hybair_td3_${SLURM_JOB_ID}"
CALIB_DIR="$TMPDIR/hybair_td3_calib_${SLURM_JOB_ID}"
MODEL_PATH="$RESULTS_DIR_FINAL/teacher_distill_model_td3.npz"
mkdir -p "$RESULTS_DIR_FINAL" "$CALIB_DIR"
export GRAND_AI_DISTILL_MODEL_PATH="$MODEL_PATH"
export GRAND_AI_DISTILL_MIN_SUCCESSES=24
export GRAND_AI_DISTILL_MAX_ROWS=220000
export GRAND_AI_DISTILL_NEG_RATIO=4.0
export GRAND_AI_DISTILL_ITERS=520
export GRAND_AI_DISTILL_LR=0.16
export GRAND_AI_DISTILL_REG=0.001
export GRAND_AI_DISTILL_VERBOSE=1

export EXPECTED_HYBRID_DECODER=hybairtd315
export EXPECTED_HYBRID_POLICY=probe_moe_roi_fix
export EXPECTED_HYBRID_SELECTION=ai_teacher_distill_roi_v3

run_phase() {
  local phase_name="$1"
  echo "[TD3] starting phase=${phase_name} RESULTS_DIR=${RESULTS_DIR} SNR_SWEEP=${SNR_SWEEP} MODEL=${GRAND_AI_DISTILL_MODEL_PATH}"
  python -u HW3_GRAND_AI_TD3.py "$RESULTS_DIR" sionna5g \

    2> >(grep -vE 'oneDNN custom operations are on|Unable to register cuFFT factory|All log messages before absl::InitializeLog|Unable to register cuDNN factory|Unable to register cuBLAS factory|computation placer already registered' >&2)
}

# ---------------- Calibration / teacher phase ----------------
export RUN_RECEIVER1=0
export RESULTS_DIR="$CALIB_DIR"
export SNR_SWEEP="${CALIB_SNR_SWEEP:-10.0,10.5,11.0,11.5,12.0}"
export TARGET_FRAME_ERRORS="${CALIB_TARGET_FRAME_ERRORS:-220}"
export MAX_FRAMES="${CALIB_MAX_FRAMES:-160000}"
export GRAND_RESCUE_SNAPSHOT_ITERS=4,8,12
export GRAND_RESCUE_APPEND_FINAL=0
export GRAND_AIR_TINY_MAX_PATTERNS=12000
export GRAND_AIR_MAX_PATTERNS=140000
export GRAND_AIR_BOOST_MAX_PATTERNS=200000
export GRAND_AIR_TINY_RESTART_MAX_CANDIDATES=6
export GRAND_AIR_TINY_RESTART_ITERS=6
export GRAND_AIR_RESTART_MAX_CANDIDATES=12
export GRAND_AIR_RESTART_ITERS=12
export GRAND_AIR_SAFE_ACCEPT_ENABLE=1
export GRAND_AIR_SAFE_ACCEPT_REFERENCE=final_stage1
export GRAND_AIR_SAFE_ACCEPT_SYNDROME_LAMBDA=0.03
export GRAND_AIR_SAFE_ACCEPT_COST_SLACK=0.60
export GRAND_AIR_SAFE_ACCEPT_STRONG_ABS=5.0
export GRAND_AIR_SAFE_ACCEPT_MAX_STRONG_WRONG=0
export GRAND_AI_DISTILL_COLLECT=1
export GRAND_AI_DISTILL_LOAD=0
run_phase calibration

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[TD3] WARNING: teacher-distill model was not created at $MODEL_PATH; evaluation will fall back to heuristic PMIX cold-start." >&2
  export GRAND_AI_DISTILL_LOAD=0
else
  export GRAND_AI_DISTILL_LOAD=1
fi
rm -rf "$CALIB_DIR"

# ---------------- Evaluation phase ----------------
export RUN_RECEIVER1=1
export RESULTS_DIR="$RESULTS_DIR_FINAL"
export SNR_SWEEP="${EVAL_SNR_SWEEP:-10.0,10.5,11.0,11.5,12.0,12.5}"
export TARGET_FRAME_ERRORS="${TARGET_FRAME_ERRORS:-140}"
export MAX_FRAMES="${MAX_FRAMES:-200000}"
export GRAND_RESCUE_SNAPSHOT_ITERS=4,8,12
export GRAND_RESCUE_APPEND_FINAL=0
export GRAND_AIR_TINY_MAX_PATTERNS=7000
export GRAND_AIR_MAX_PATTERNS=50000
export GRAND_AIR_BOOST_MAX_PATTERNS=70000
export GRAND_AIR_TINY_RESTART_MAX_CANDIDATES=4
export GRAND_AIR_TINY_RESTART_ITERS=4
export GRAND_AIR_RESTART_MAX_CANDIDATES=8
export GRAND_AIR_RESTART_ITERS=8
export GRAND_AIR_SAFE_ACCEPT_ENABLE=1
export GRAND_AIR_SAFE_ACCEPT_REFERENCE=final_stage1
export GRAND_AIR_SAFE_ACCEPT_SYNDROME_LAMBDA=0.03
export GRAND_AIR_SAFE_ACCEPT_COST_SLACK=0.50
export GRAND_AIR_SAFE_ACCEPT_STRONG_ABS=5.0
export GRAND_AIR_SAFE_ACCEPT_MAX_STRONG_WRONG=0
export GRAND_AI_DISTILL_COLLECT=0
run_phase evaluation

printf '[SBATCH CONFIG] EVAL_SNR_SWEEP=%s\n' "$SNR_SWEEP"
printf '[SBATCH CONFIG] LDPC_ITERS=%s STAGE1_ITERS=%s\n' "$LDPC_ITERS" "$STAGE1_ITERS"
printf '[SBATCH CONFIG] CHANNEL=%s TDL=%s DELAY=%s SPEED=[%s,%s] CFO=%s CSI_MODE=%s\n' \
  "$CHANNEL_NAME" "$SIONNA_TDL_MODEL" "$SIONNA_TDL_DELAY_SPREAD_S" \
  "$SIONNA_TDL_MIN_SPEED" "$SIONNA_TDL_MAX_SPEED" "$SIONNA_CFO_HZ" "$SIONNA_CSI_MODE"
printf '[SBATCH CONFIG] RUN_RECEIVER9=%s SNAPSHOTS=%s APPEND_FINAL=%s policy=%s selection=%s expected=%s model=%s\n' \
  "$RUN_RECEIVER9" "$GRAND_RESCUE_SNAPSHOT_ITERS" "$GRAND_RESCUE_APPEND_FINAL" \
  "$GRAND_AIR_POLICY_MODE" "$GRAND_AIR_SELECTION_MODE" "$EXPECTED_HYBRID_DECODER" "$MODEL_PATH"
