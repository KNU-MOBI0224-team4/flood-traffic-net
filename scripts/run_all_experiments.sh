#!/usr/bin/env bash
# Run all four onset-prediction models (logistic, xgboost, gru, stgcn) under a
# unified protocol and collect results into two time-feature branches.
#
# Unified across models:
#   seq_len = 12, pred_horizon = 0   (applies to stgcn/gru; tabular is single-step)
#
# Branches / variants:
#   - time-feature OFF / ON -> two summary files. Only STGCN supports time
#     features, so logistic/xgboost/gru run once and are shared by both branches.
#   - STGCN additionally sweeps Chebyshev K in {2,3} x LayerNorm in {on,off}.
#
# Usage:
#   bash scripts/run_all_experiments.sh
#
# Common overrides (env vars):
#   DEVICE=cuda:3 bash scripts/run_all_experiments.sh
#   PERCENTILES="p97 p99" bash scripts/run_all_experiments.sh
#   STGCN_EPOCHS=1 FOLDS="fold_1_train2016_2020_val2021_test2022" \
#     bash scripts/run_all_experiments.sh        # quick plumbing check

set -o pipefail
cd "$(dirname "$0")/.."

# ---- shared config ----
DATA_DIR="${DATA_DIR:-data/train_ready/d7_active_giant_full}"
BASE="${BASE_OUT_DIR:-data/outputs/run_all}"
PERCENTILES="${PERCENTILES:-p97}"
FOLDS="${FOLDS:-fold_1_train2016_2020_val2021_test2022 fold_2_train2016_2021_val2022_test2023 fold_3_train2016_2022_val2023_test2024}"
SEQ_LEN="${SEQ_LEN:-12}"
PRED_HORIZON="${PRED_HORIZON:-0}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-}"   # empty -> let each script auto-detect cuda/cpu
STGCN_TIMES="${STGCN_TIMES:-off on}"   # which time-feature branches to run for STGCN
DUMP_PREDICTIONS="${DUMP_PREDICTIONS:-0}"   # 1 -> save per-cell test predictions for case studies

# ---- per-model hyperparameters ----
STGCN_EPOCHS="${STGCN_EPOCHS:-50}"
STGCN_PATIENCE="${STGCN_PATIENCE:-7}"
STGCN_BATCH="${STGCN_BATCH:-16}"
STGCN_LR="${STGCN_LR:-1e-4}"
STGCN_POS_WEIGHT_CAP="${STGCN_POS_WEIGHT_CAP:-1.0}"
GRU_EPOCHS="${GRU_EPOCHS:-50}"
GRU_PATIENCE="${GRU_PATIENCE:-7}"
GRU_BATCH="${GRU_BATCH:-256}"
GRU_LR="${GRU_LR:-1e-3}"
GRU_POS_WEIGHT_CAP="${GRU_POS_WEIGHT_CAP:-50.0}"

RUN_DATE="$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$BASE"
LOG="$BASE/run_${RUN_DATE}.log"
FAILURES=()

DEV_ARG=()
[ -n "$DEVICE" ] && DEV_ARG=(--device "$DEVICE")

DUMP_ARG=()
[ "$DUMP_PREDICTIONS" = "1" ] && DUMP_ARG=(--dump-predictions)

run_step() {
  local label="$1"; shift
  {
    echo ""
    echo "=================================================="
    echo "[step] $label  $(date '+%F %T')"
  } | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then
    echo "[ok]   $label" | tee -a "$LOG"
  else
    local code=$?
    echo "[FAIL] $label (exit $code)" | tee -a "$LOG"
    FAILURES+=("$label")
  fi
}

{
  echo "[start] $(date '+%F %T')"
  echo "  base=$BASE"
  echo "  percentiles=$PERCENTILES"
  echo "  folds=$FOLDS"
  echo "  seq_len=$SEQ_LEN pred_horizon=$PRED_HORIZON seed=$SEED device=${DEVICE:-auto}"
} | tee "$LOG"

# ---- models without time-feature support: run once, shared by both branches ----
run_step "tabular(logistic,xgboost)" \
  python3 -u scripts/run_tabular_baselines.py \
    --data-dir "$DATA_DIR" --percentiles $PERCENTILES --folds $FOLDS \
    --models logistic xgboost "${DUMP_ARG[@]}" --out-dir "$BASE/shared/tabular"

run_step "gru" \
  python3 -u scripts/run_gru_baseline.py \
    --data-dir "$DATA_DIR" --percentiles $PERCENTILES --folds $FOLDS \
    --seq-len "$SEQ_LEN" --pred-horizon "$PRED_HORIZON" \
    --epochs "$GRU_EPOCHS" --early-stopping-patience "$GRU_PATIENCE" \
    --batch-size "$GRU_BATCH" --learning-rate "$GRU_LR" \
    --pos-weight-cap "$GRU_POS_WEIGHT_CAP" --seed "$SEED" \
    "${DEV_ARG[@]}" "${DUMP_ARG[@]}" --out-dir "$BASE/shared/gru"

# ---- STGCN: time-feature branch x Chebyshev K x LayerNorm ----
for time in $STGCN_TIMES; do
  time_arg=()
  [ "$time" = "on" ] && time_arg=(--time-features)
  for k in 2 3; do
    for ln in on off; do
      ln_arg=(--hidden-layernorm)
      [ "$ln" = "off" ] && ln_arg=(--no-hidden-layernorm)
      out="$BASE/time_${time}/stgcn_k${k}_ln${ln}"
      run_step "stgcn time=$time k=$k ln=$ln" \
        python3 -u scripts/run_stgcn.py \
          --data-dir "$DATA_DIR" --percentiles $PERCENTILES --folds $FOLDS \
          --seq-len "$SEQ_LEN" --pred-horizon "$PRED_HORIZON" \
          --cheb-k "$k" "${ln_arg[@]}" "${time_arg[@]}" \
          --epochs "$STGCN_EPOCHS" --early-stopping-patience "$STGCN_PATIENCE" \
          --batch-size "$STGCN_BATCH" --lr "$STGCN_LR" \
          --pos-weight-cap "$STGCN_POS_WEIGHT_CAP" --seed "$SEED" \
          "${DEV_ARG[@]}" "${DUMP_ARG[@]}" --out-dir "$out"
    done
  done
done

# ---- aggregate into two summary files ----
run_step "aggregate" \
  python3 scripts/aggregate_experiments.py --base "$BASE"

{
  echo ""
  echo "[done] $(date '+%F %T')"
  if [ "${#FAILURES[@]}" -gt 0 ]; then
    echo "[warn] ${#FAILURES[@]} step(s) failed:"
    printf '  - %s\n' "${FAILURES[@]}"
  else
    echo "[ok] all steps succeeded"
  fi
  echo "[summary] $BASE/summary_time_off.csv"
  echo "[summary] $BASE/summary_time_on.csv"
} | tee -a "$LOG"
