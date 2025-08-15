#!/usr/bin/env bash

# Declare an associative array mapping GPU to NUM_SPEC_TOKENS
# declare -A GPU_TO_NUM_SPEC_TOKENS=(
#   [0]=1
#   [1]=2
#   [2]=3
#   [3]=4
#   [4]=5
#   [5]=6
#   [6]=8
#   [7]=10
# )

declare -A GPU_TO_FIXED_DECODE_LENGTH=(
  [0]=2
  [1]=16
  [2]=32
  [3]=64
  [4]=128
  [5]=256
  [6]=512
  [7]=1024
)

# List of FIXED_ACCEPTANCE_RATE values to sweep
# FIXED_ACCEPTANCE_RATES=(0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00)
# FIXED_PREFILL_LENGTHS=(1 16 32 64 128 256 512 1024)
FIXED_PREFILL_LENGTHS=(256)
# FIXED_ACCEPTANCE_RATES=(0.50 0.60 0.70 0.80 0.90 1.00)
# FIXED_ACCEPTANCE_RATES=(0.55 0.65 0.75 0.85 0.95)

run_series_on_gpu() {
  local GPU=$1
  local FIXED_DECODE_LENGTH=$2
  local total=${#FIXED_PREFILL_LENGTHS[@]}
  local count=0

  # for FIXED_ACCEPTANCE_RATE in "${FIXED_ACCEPTANCE_RATES[@]}"; do
  for FIXED_PREFILL_LENGTH in "${FIXED_PREFILL_LENGTHS[@]}"; do
    count=$((count + 1))
    # Build a simple progress bar
    done_bar=$(printf "%0.s#" $(seq 1 $count))
    todo_bar=$(printf "%0.s-" $(seq 1 $((total - count))))
    echo "[GPU $GPU] Progress: [${done_bar}${todo_bar}] $count/$total (FIXED_PREFILL_LENGTH=$FIXED_PREFILL_LENGTH, FIXED_DECODE_LENGTH=$FIXED_DECODE_LENGTH)"
    LOGFILE="wrapper_gpu${GPU}_prefill${FIXED_PREFILL_LENGTH}_decode${FIXED_DECODE_LENGTH}.log"
    CUDA_VISIBLE_DEVICES=$GPU FIXED_PREFILL_LENGTH=$FIXED_PREFILL_LENGTH FIXED_DECODE_LENGTH=$FIXED_DECODE_LENGTH ./script_conda_baseline.sh > "$LOGFILE" 2>&1
  done
  echo "[GPU $GPU] All runs complete."
}

for GPU in "${!GPU_TO_FIXED_DECODE_LENGTH[@]}"; do
  FIXED_DECODE_LENGTH="${GPU_TO_FIXED_DECODE_LENGTH[$GPU]}"
  run_series_on_gpu "$GPU" "$FIXED_DECODE_LENGTH" &
done

wait  # Wait for all background jobs to finish
echo "All jobs done."