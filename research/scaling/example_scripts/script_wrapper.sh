#!/usr/bin/env bash

# Declare an associative array mapping GPU to NUM_SPEC_TOKENS
declare -A GPU_TO_NUM_SPEC_TOKENS=(
  [0]=1
  [1]=2
  [2]=3
  [3]=4
  [4]=5
  [5]=6
  [6]=8
  [7]=10
)

# List of FIXED_ACCEPTANCE_RATE values to sweep
# FIXED_ACCEPTANCE_RATES=(0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00)
FIXED_ACCEPTANCE_RATES=(0.50 0.60 0.70 0.80 0.90 1.00)
# FIXED_ACCEPTANCE_RATES=(0.55 0.65 0.75 0.85 0.95)

run_series_on_gpu() {
  local GPU=$1
  local NUM_SPEC_TOKENS=$2
  local total=${#FIXED_ACCEPTANCE_RATES[@]}
  local count=0

  for FIXED_ACCEPTANCE_RATE in "${FIXED_ACCEPTANCE_RATES[@]}"; do
    count=$((count + 1))
    # Build a simple progress bar
    done_bar=$(printf "%0.s#" $(seq 1 $count))
    todo_bar=$(printf "%0.s-" $(seq 1 $((total - count))))
    echo "[GPU $GPU] Progress: [${done_bar}${todo_bar}] $count/$total (NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS, FIXED_ACCEPTANCE_RATE=$FIXED_ACCEPTANCE_RATE)"
    LOGFILE="wrapper_gpu${GPU}_tokens${NUM_SPEC_TOKENS}_rate${FIXED_ACCEPTANCE_RATE}.log"
    CUDA_VISIBLE_DEVICES=$GPU NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS FIXED_ACCEPTANCE_RATE=$FIXED_ACCEPTANCE_RATE ./script_conda.sh > "$LOGFILE" 2>&1
  done
  echo "[GPU $GPU] All runs complete."
}

for GPU in "${!GPU_TO_NUM_SPEC_TOKENS[@]}"; do
  NUM_SPEC_TOKENS="${GPU_TO_NUM_SPEC_TOKENS[$GPU]}"
  run_series_on_gpu "$GPU" "$NUM_SPEC_TOKENS" &
done

wait  # Wait for all background jobs to finish
echo "All jobs done."