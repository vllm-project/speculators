#!/usr/bin/env bash
# Live side-by-side demo: vLLM with and without speculative decoding.
#
# Serves the same model twice -- plain on one GPU, with a speculator on the other --
# and hands both servers to demo_race.py, which races your prompts against them.
# Both servers decode greedily, so the two answers must come out identical.
#
set -euo pipefail

usage() {
  cat <<'END'
usage: demo_side_by_side.sh --model DIR --draft DIR --method M [options]

  --model DIR     target model, run by both servers
  --draft DIR     speculator weights
  --method M      vLLM speculative method: dflash, eagle, eagle3, ngram, ...

  --gpus L,R      GPUs for the plain / speculative server  (default 0,1)
  -k N            tokens drafted per step                  (default 8)
  --max-tokens N  generation cap per prompt                (default 1000)

example:
  demo_side_by_side.sh --model ~/hf_models/Qwen/Qwen3-8B \
    --draft ~/hf_models/z-lab/Qwen3-8B-DFlash-b16 --method dflash
END
  exit 1
}

MODEL= DRAFT= METHOD= GPUS=0,1 K=8 MAX_TOKENS=1000
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL=$2; shift 2;;
    --draft) DRAFT=$2; shift 2;;
    --method) METHOD=$2; shift 2;;
    --gpus) GPUS=$2; shift 2;;
    -k) K=$2; shift 2;;
    --max-tokens) MAX_TOKENS=$2; shift 2;;
    *) usage;;
  esac
done
[[ $MODEL && $DRAFT && $METHOD ]] || usage

PLAIN_GPU=${GPUS%,*}  PLAIN_PORT=8801
SPEC_GPU=${GPUS#*,}   SPEC_PORT=8901

# The flags both servers share are what keep the comparison fair:
#   --max-num-seqs 1            one request at a time, so matrix shapes never vary
#   --no-enable-prefix-caching  no request gets a head start from an earlier one
#   --generation-config vllm    ignore the model's own sampling defaults
#   --attention-backend, VLLM_USE_FLASHINFER_SAMPLER  same kernels on both sides
start_server() {  # start_server <gpu> <port> [extra vllm args...]
  local gpu=$1 port=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu VLLM_USE_FLASHINFER_SAMPLER=0 \
    setsid vllm serve "$MODEL" \
      --served-model-name demo --port "$port" \
      --max-model-len 10240 --max-num-seqs 1 --gpu-memory-utilization 0.85 \
      --attention-backend flash_attn --generation-config vllm \
      --no-enable-prefix-caching --trust-remote-code --seed 0 \
      "$@" > /dev/null 2>&1 &
}

# Fixed ports, so refuse to start if anything already answers on them: the health
# check below would pass against that server and the demo would measure it instead.
for port in $PLAIN_PORT $SPEC_PORT; do
  curl -sf "http://127.0.0.1:$port/health" > /dev/null &&
    { echo "something is already serving on port $port; stop it first" >&2; exit 1; }
done

echo "starting two servers (the first run has to compile, so give it a minute)"
start_server "$PLAIN_GPU" "$PLAIN_PORT"
PLAIN_PID=$!
start_server "$SPEC_GPU" "$SPEC_PORT" --speculative-config \
  "{\"method\": \"$METHOD\", \"model\": \"$DRAFT\", \"num_speculative_tokens\": $K}"
SPEC_PID=$!

# setsid gave each server its own process group, so this takes its children too.
trap 'kill -- -$PLAIN_PID -$SPEC_PID 2>/dev/null' EXIT

for port in $PLAIN_PORT $SPEC_PORT; do
  echo -n "  waiting for port $port "
  until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do echo -n .; sleep 2; done
  echo " ready"
done

python demo_race.py --max-tokens "$MAX_TOKENS" \
  --left-url  "http://127.0.0.1:$PLAIN_PORT" --left-name  "WITHOUT spec-dec" \
  --right-url "http://127.0.0.1:$SPEC_PORT"  --right-name "WITH $METHOD spec-dec"
