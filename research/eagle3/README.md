#Eagle 3
#### This code is based on the HASS code (https://github.com/HArmonizedSS/HASS) and trains models which are similar to the Eagle 3 architecture.  It implements the Train Time Test method of Eagle 3.  The original Eagle code can be found here: https://github.com/SafeAILab/EAGLE

## To Run:

The training process is broken up in to 2 steps, the first where you generate data from the large model, and the second where you actually train the drafter model. It works for Llama 3.1.8B-Instruct and Qwen 3 8B.

### Data Generation step:

1. Modify the directory names and arguments in `gen_data.sh`
2. You can get the data for ShareGPT at:  Aeala/ShareGPT_Vicuna_unfiltered on huggingface.  Ultrachat will be automatically downloaded.
3. Make sure that the system prompts and chat template demarkation in the desired file (ultrachat.py or sharegpt.py, ultrachatMistral.py or sharegptMistral.py) are correct
4. Run the script: `./gen_data.sh`
5. Run for each of: sharegpt, and ultrachat sft and gen splits.
Notes:  For llama 3.1.8B this will generate ~4TB of data on your system.  The script for training searches your data directory recursively, so the internal structure of your data directory does not matter.

### Run training
1. Modify the arguments (vocabulary size, samples, and data directory) for zipf.sh, and run to generate the restricted vocabulary mapping.  We recommend using at least 10000 examples.
2. Modify the directory names and arguments in `train.sh`
3. Run `./train.sh`

### Serve the model with vllm:
1. Convert your saved model with: `convert.sh`
2. Serve the model with:
` VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct --seed 42 -tp 1 --speculative-config '{"model": "llama_eagle3", "num_speculative_tokens": 3, "method":"eagle3", "draft_tensor_parallel_size":1}'`


## Replication
To re-create the Eagle 3 experiments for Llama 3.1.8b and Qwen 3 8b, run the following commands:  
1. Get data for sharegpt


`wget https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V4.3_unfiltered_cleaned_split.json .
`


2. Generate sharegpt data  WARNING: THIS WILL TAKE A LOT OF MEMORY ON YOUR SYSTEM


`python -m ge_data.allocation \
--outdir dataDirectory/sharegpt \
--data_path ShareGPT_V4.3_unfiltered_cleaned_split.json \
--model_path meta-llama/Llama-3.1-8B-Instruct \
--chat_template llama \
--dataset sharegpt \
--samples 68000 \
--total_gpus 8`



3. Generate ultrachat data for both sft and gen splits.
   3a.

   
`python -m ge_data.allocation \
--outdir dataDirectory/ultrachat_gen \
--data_path ShareGPT_V4.3_unfiltered_cleaned_split.json \
--model_path meta-llama/Llama-3.1-8B-Instruct \
--chat_template llama \
--dataset ultrachat \
--split gen \
--samples 250000 \
--total_gpus 8`

  3b.

  
`python -m ge_data.allocation \
--outdir dataDirectory/ultrachat_sft \
--data_path ShareGPT_V4.3_unfiltered_cleaned_split.json \
--model_path meta-llama/Llama-3.1-8B-Instruct \
--chat_template llama \
--dataset ultrachat \
--split sft \
--samples 207000 \
--total_gpus 8`

4. Calculate the most common tokens and reduce the vocabulary based on those.


`python zipf.py --dataDir dataDirectory \
	--samples 10000 \
	--vocab 128256 \
	--reduced 32000`

5. Run the training code.


`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch -m  --mixed_precision=bf16 --use_deepspeed --main_process_port  29501 --num_processes 8 train.main_train_full_gradient_calc_eagle3 \<br>
    --basepath path/to/big/model/weights \
    --tmpdir dataDirectory \
    --cpdir Eagle3 \
    --configpath train/llama3_8_B.json \
    --epoch 4 \
    --bs 1 \
    --topk_w 0 \
    --topk 1 \
    --lr 8e-5 \
    --forward_num_total 3 \`

6. Add vocabulary mappings and a config to the checkpoint weights.  
`python convert_checkpoint.py --checkpoint Eagle3/model4.safetensors --config train/llama3_8_B.json --outpath eagle3_llama `

7. Convert the saved model file to speculators format:
