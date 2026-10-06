#!/usr/bin/env bash
# Throughput-vs-interactivity sweeps for Qwen3.8-27B with and without the dspark speculator.
cd $RUN_DIR
export PATH=$GUIDELLM_BIN:$PATH HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE
sweep() {  # <label> <port> <subset>
  local common=(--target http://localhost:$2 --model Qwen/Qwen3.8-27B --tokenizer Qwen/Qwen3.8-27B
                --dataset RedHatAI/speculator_benchmarks --subset $3 --max-tokens 1024 --repeats 1
                --out-dir $RUN_DIR/$1_$3 --label $1 --csv $RUN_DIR/$1_$3.csv --keep-going)
  python3 $RUN_DIR/throughput_interactivity.py collect "${common[@]}" --streams 1,2,4,8,16 --max-seconds 90 --warmup-seconds 30
  python3 $RUN_DIR/throughput_interactivity.py collect "${common[@]}" --streams 32,64,128 --max-seconds 120 --warmup-seconds 60
}
server() {  # <label> <port>
  for subset in HumanEval math_reasoning; do sweep $1 $2 $subset; done > $RUN_DIR/sweep_$1.log 2>&1
  echo "$(date +%T) $1 done" >> $RUN_DIR/sweeps_status.log
}
server baseline 8010 &
server dspark 8011 &
wait
echo "ALL SWEEPS DONE" >> $RUN_DIR/sweeps_status.log
