#!/usr/bin/env bash
# Third configuration: Qwen3.8-27B with its own MTP head, 2 speculative tokens (the Qwen model-card setting), pinned scheduler limits.
set -u
R=$RUN_DIR
export PATH=$VENV_BIN:$VENV_BIN:$PATH HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE
T=Qwen/Qwen3.8-27B
CMD=(vllm serve $T --served-model-name Qwen/Qwen3.8-27B --port 8012 --max-model-len 16384 --max-num-seqs 256 --max-num-batched-tokens 16384 --speculative-config '{"method":"mtp","num_speculative_tokens":2}')
echo "${CMD[*]}" > $R/serve_mtp_command.txt
CUDA_VISIBLE_DEVICES=1 "${CMD[@]}" > $R/serve_mtp.log 2>&1 &
SPID=$!; echo "server pid $SPID" > $R/mtp_status.log
n=0; until curl -sf -m 3 http://localhost:8012/health > /dev/null; do n=$((n+1)); [ $n -ge 120 ] && { echo "server failed to start" >> $R/mtp_status.log; kill $SPID; exit 1; }; kill -0 $SPID 2>/dev/null || { echo "server died" >> $R/mtp_status.log; exit 1; }; sleep 5; done
echo "$(date +%T) server healthy" >> $R/mtp_status.log
curl -s -m 120 http://localhost:8012/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen/Qwen3.8-27B","messages":[{"role":"user","content":"Write a Python function that returns the n-th Fibonacci number."}],"max_tokens":120}' > /dev/null
curl -s http://localhost:8012/metrics | grep "^vllm:spec_decode_num_drafts_total\|^vllm:spec_decode_num_accepted_tokens_total" >> $R/mtp_status.log
sweep() {  # <subset>
  local common=(--target http://localhost:8012 --model Qwen/Qwen3.8-27B --tokenizer $T
                --dataset RedHatAI/speculator_benchmarks --subset $1 --max-tokens 1024 --repeats 1
                --out-dir $R/mtp_$1 --label mtp --csv $R/mtp_$1.csv --keep-going)
  python3 $R/throughput_interactivity.py collect "${common[@]}" --streams 1,2,4,8,16 --max-seconds 90 --warmup-seconds 30
  python3 $R/throughput_interactivity.py collect "${common[@]}" --streams 32,64,128 --max-seconds 120 --warmup-seconds 60
}
for subset in HumanEval math_reasoning; do sweep $subset; done > $R/sweep_mtp.log 2>&1
echo "$(date +%T) sweep exit $?" >> $R/mtp_status.log
kill $SPID 2>/dev/null; sleep 10; kill -9 $SPID 2>/dev/null
echo "MTP DONE" >> $R/mtp_status.log
