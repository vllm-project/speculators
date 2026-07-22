
 hf download   inference-optimization/gemma4-31B-responses magpie_gemma-4-31B-it.jsonl   --repo-type dataset   --local-dir ./output/dataset

 python scripts/prepare_data.py   --model google/gemma-4-31B-it   --data ./output/dataset/magpie_gemma-4-31B-it.jsonl   --output ./output/mtp_gemma-4-31B-it   --max-samples 5000   --seq-length 8192

CUDA_VISIBLE_DEVICES=0,1 python scripts/launch_vllm.py   google/gemma-4-31B-it --hidden-states-path /tmp/hidden_states_$(whoami )-- --port 8000  --tensor-parallel-size 2 --max-model-len 8192

CUDA_VISIBLE_DEVICES=2,3 python scripts/train.py \
  --verifier-name-or-path google/gemma-4-31B-it \
  --data-path ./output/mtp_gemma-4-31B-it \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./output/mtp_gemma-4-31B-it/checkpoints \
  --speculator-type mtp \
  --num-speculative-steps 3 \
  --step-weight-beta 0.6 \
  --epochs 3 \
  --lr 1e-4 \
  --total-seq-len 8192 \
  --on-missing generate \
  --on-generate delete


CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node 2 scripts/train.py  
 --verifier-name-or-path google/gemma-4-31B-it   --data-path ./output/mtp_gemma-4-31B-it   --vllm-endpoint http://localhost:8000/v1   --save-path ./output/mtp_gemma-4-31B-it/checkpoints   --speculator-type mtp   --num-speculative-steps 3   --step-weight-beta 0.6   --epochs 3   chs 3   --lr 1e-4   --total-seq-len 8192 --on-missing raise



python scripts/stitch_mtp.py   ./output/mtp_gemma-4-31B-it/checkpoints/checkpoint_best   google/gemma-4-31B-it   --output-path ./output/stitched