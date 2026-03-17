1. Run the small llama example:
```
python examples/data_generation_and_training/peagle_llama3_8b_example.py
```
2. Convert config:
```
python convert_peagle_config.py speculators/output/peagle_llama3_8b_sharegpt_5k/checkpoints/5
```
3. Spin up model in vllm
```
vllm serve meta-llama/Llama-3.1-8B-Instruct  --speculative-config '{"method": "eagle3", "model": "speculators/output/peagle_llama3_8b_sharegpt_5k/checkpoints/1", "num_speculative_tokens": 10, "parallel_drafting": true}' --tensor-parallel-size 2 --max-num-batched-tokens 32768 --kv-cache-dtype fp8 --async-scheduling --stream-interval 20 --max-cudagraph-capture-size 4096 --no-enable-prefix-caching --port 8042 --gpu-memory-utilization 0.9 --max-num-seqs 128 --max-model-len 32768
```
4. Bench Testing
```
vllm bench serve --backend openai-chat --base-url http://localhost:8042 --endpoint /v1/chat/completions --model meta-llama/Llama-3.1-8B-Instruct --dataset-name spec_bench --dataset-path eval_datasets/spec_bench.jsonl --spec-bench-output-len 256 --num-prompts 80 --max-concurrency 1 --temperature 0 --request-rate inf --save-result --save-detailed
```
5. Other helpful commands
```
torchrun --nnodes=1 --nproc_per_node=4 speculators/scripts/train.py --verifier-name-or-path Qwen/Qwen3-Coder-30B-A3B-Instruct --speculator-type peagle --data-path /mnt/nvme_stripe/playground/hezhao/ultrachat_33k --save-path speculators/output/qwen3_coder_30b_peagle --epochs 10 --lr 1e-4 --num-layers 4 --total-seq-len 4096 --max-seq-len 2048 --para-depths 12 --down-sample-ratio 0.7 --down-sample-ratio-min 0.2 --ptd-token-id 151620"
```