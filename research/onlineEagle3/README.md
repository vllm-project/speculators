1. cd research/onlineEagle3

2. Download training data
``` bash
wget https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V4.3_unfiltered_cleaned_split.json
```
3. Generate t2d and d2t
``` bash
python build_t2d_from_data.py --verifier $VERIFIER_PATH --data ShareGPT_V4.3_unfiltered_cleaned_split.json --samples 50000 --drafter-vocab 32000 --out t2d.npy --seed 42
```

4. Generate the draft model eagle head
``` bash
export VERIFIER_PATH=/proving-grounds/cache/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b [or your local verifier path]
export CONFIG_PATH=/cache/helen/speculators/research/onlineEagle3/train/llama33_70b_head_k3.json
python create_eagle_head.py   --verifier $VERIFIER_PATH   --out meta-llama-3-70b-head-k3   --k 3   --dtype bfloat16
```

5. Run training code
``` bash
PYTHONPATH=. python train/main_train_online.py \
    --basepath $VERIFIER_PATH \
    --configpath $CONFIG_PATH \
    --epoch 40 \
    --cpdir checkpoints \
    --hf_export_every 1
```

6. Convert the model
``` bash
speculators convert research/onlineEagle3/eagle3_ep40 \
  --algorithm eagle3 \
  --verifier /proving-grounds/cache/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b \
  --output-path research/onlineEagle3/eagle3_ep40-converted
```

7. (Optional) upload model to hugging face
``` bash
hf upload nm-testing/eagle3-online-gen eagle3_ep40-converted
```

8. serve in vllm
``` bash
vllm serve nm-testing/eagle3-online-gen [or your local path]
```