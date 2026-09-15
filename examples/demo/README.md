# Live side-by-side demo: vLLM with vs. without speculative decoding

Your prompt, streamed to two vLLM servers at once. Left: the target model, plain decoding.
Right: the same model + a speculator, identical text but much faster.


- `demo_side_by_side.sh` serves the same model twice (plain on one GPU, with the speculator
  on the other) and hands both servers to the racer.
- `demo_race.py` warms both engines up, checks they agree token-for-token, then hands you a
  `prompt>`: whatever you type is raced live on both engines and measured. Empty line quits
  and prints a summary; the shell script then tears both servers down.


## Prerequisites
- vLLM on your `PATH` (demo developed and tested against 0.29.0)
- Separate GPUs for your model with and without speculative decoding 
- The model and the speculator. `--model` and `--draft` take either a local directory or a
  Hugging Face id; a local copy avoids a download on the first run:
  ```
  hf download Qwen/Qwen3-8B             --local-dir ~/models/Qwen3-8B
  hf download z-lab/Qwen3-8B-DFlash-b16 --local-dir ~/models/Qwen3-8B-DFlash-b16
  ```

## Run
`--model`, `--draft` and `--method` are required:
```
./demo_side_by_side.sh --gpus 0,1 --method dflash \
  --model ~/models/Qwen3-8B \
  --draft ~/models/Qwen3-8B-DFlash-b16
```

The first start compiles both servers (a minute or two); later starts hit the compile cache.
The servers' own output is discarded, so if one never reports ready, run its `vllm serve`
line by hand to debug the error.

## What makes the text identical in both deployments
Greedy decoding (temperature 0) on both sides. Pinned attention backend and sampler,
`--max-num-seqs 1`, prefix caching off. Reported metrics per race:
wall-clock, tok/s, ttft, tokens, forward passes (exact, from each server's `/metrics`),
tokens per forward pass, and what the speculator's drafts actually achieved.

