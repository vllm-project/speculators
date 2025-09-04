1. Change directory to research/onlineEagle3
``` bash
cd research/onlineEagle3
```

2. Download training data
``` bash
wget https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V4.3_unfiltered_cleaned_split.json
```
3. Generate t2d and d2t
``` bash
python build_t2d_from_data.py --verifier $VERIFIER_PATH --data ShareGPT_V4.3_unfiltered_cleaned_split.json --samples 50000 --drafter-vocab 32000 --out t2d.npy --seed 42
```

4. Generate the draft model eagle head

llama 70B (for server 15 only, switch out verifier path for your local model path):
``` bash
export VERIFIER_PATH=/proving-grounds/cache/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b
export CONFIG_PATH=/cache/helen/speculators/research/onlineEagle3/train/llama33_70b_head_k3.json
python create_eagle_head.py   --verifier $VERIFIER_PATH   --out meta-llama-3-70b-head-k3   --k 3   --dtype bfloat16
export EAGLE_HEAD_PATH=/cache/helen/speculators/research/onlineEagle3/meta-llama-3-70b-head-k3
```

llama 8B:
``` bash
export VERIFIER_PATH=/proving-grounds/cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
export CONFIG_PATH=/cache/helen/speculators/research/onlineEagle3/train/llama3_8_B.json
python create_eagle_head.py   --verifier $VERIFIER_PATH   --out meta-llama-3-8b-head-k3   --k 3   --dtype bfloat16
export EAGLE_HEAD_PATH=/cache/helen/speculators/research/onlineEagle3/meta-llama-3-8b-head-k3
```

5. Run training code
``` bash
PYTHONPATH=. python train/main_train_online.py --basepath $VERIFIER_PATH --configpath $CONFIG_PATH --cpdir checkpoints --data_num 68623 --epoch 5 --hf_export_every 1 --eagle-head-path "$EAGLE_HEAD_PATH" --num-speculative-tokens 3
```

6. Convert the model
``` bash
speculators convert checkpoints/epoch_4-hf/ \
--algorithm eagle3 \
--verifier $VERIFIER_PATH \
--output-path epoch_4-converted \
--algorithm-kwargs '{"norm_before_residual": true}'
```

Don't forget to update based on the actual epochs you trained on.

7. (Optional) upload model to hugging face
``` bash
hf upload nm-testing/eagle3-online-gen epoch_4-converted 
```

8. serve in vllm
HF model
``` bash
vllm serve nm-testing/eagle3-online-gen 
```

Local model
``` bash
vllm serve ../speculators/research/onlineEagle3/epoch_4-converted 
```