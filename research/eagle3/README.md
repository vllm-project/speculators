# Eagle 3

#### This code is based on the HASS code (https://github.com/HArmonizedSS/HASS) and trains models which are similar to the Eagle 3 architecture. It implements the Train Time Test method of Eagle 3. The original Eagle code can be found here: https://github.com/SafeAILab/EAGLE

## To Run:

The training process is broken up into 2 steps: first you generate data from the large model, and second you train the drafter model. It works for Llama 3.1.8B-Instruct and Qwen 3 8B. Don't forget to change directory to research/eagle3 and install requirements.

### Data Generation Step

#### 1. Download Datasets

**For ShareGPT:**
```bash
wget https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V4.3_unfiltered_cleaned_split.json
```

**For Ultrachat:** The dataset will be automatically downloaded when you run the generation script.

#### 2. Generate Training Data

The data generation uses the `ge_data.allocation` module to process datasets in parallel across multiple GPUs. You need to run the script separately for each dataset and split combination.

**Key Parameters:**
- `--dataset`: Either `sharegpt` or `ultrachat`
- `--split`: Either `sft` or `gen` (for ultrachat only)
- `--chat_template`: Either `llama` or `qwen` (must match your model)
- `--samples`: Number of samples to process (use larger numbers for full datasets)
- `--total_gpus`: Total number of GPUs available
- `--outdir`: Output directory for generated data, ultrachat can be automatically downloaded.
- `--data_path`: Path to the downloaded dataset file
- `--model_path`: HuggingFace model path

**For Qwen models**, change `--chat_template llama` to `--chat_template qwen` and use the appropriate Qwen model path.

> 💡 **Note**: Running this takes up about ~18TB of data on the system.  It is possible to run smaller tests using just the ShareGPT dataset (simply skip the ultrachat steps).  Performance will degrade slightly, but it will only take up ~4TB and run much faster.   If you would like to change the chat template or system prompt for one of the datasets, do so in the respective files in ge_data.


#### 3. Edit gen_data.sh (Optional)

You can modify `gen_data.sh` with your specific parameters and run it instead of the individual commands:

```bash
./gen_data.sh
```

> 💡 **Note**: To reproduce the models available on https://huggingface.co/RedHatAI, use the following parameter combinations:

```python -m ge_data.allocation \
--outdir dataDirectory/sharegpt \
--data_path ShareGPT_V4.3_unfiltered_cleaned_split.json \
--model_path meta-llama/Llama-3.1-8B-Instruct \
--chat_template llama \
--dataset sharegpt \
--samples 68000 \
--total_gpus 8
```

```python -m ge_data.allocation \
--outdir dataDirectory/sharegpt \
--data_path ShareGPT_V4.3_unfiltered_cleaned_split.json \
--model_path meta-llama/Llama-3.1-8B-Instruct \
--chat_template llama \
--dataset ultrachat \
--split gen \
--samples 250000 \
--total_gpus 8
```

```python -m ge_data.allocation \
--outdir dataDirectory/sharegpt \
--data_path ShareGPT_V4.3_unfiltered_cleaned_split.json \
--model_path meta-llama/Llama-3.1-8B-Instruct \
--chat_template llama \
--dataset sharegpt \
--split sft \
--samples 207000 \
--total_gpus 8
```



### Training Step

#### 1. Generate Zipf-restricted vocabulary mapping

After data generation, create the vocabulary mapping files:

```bash
./zipf.sh
```

This will create `d2t.npy` and `t2d.npy` files needed for training.

**Configure zipf.sh parameters:**
- `--samples`: Number of examples to use for frequency analysis (recommend 10,000+)
- `--vocab`: Target model vocabulary size (128256 for Llama)
- `--reduced`: Draft model vocabulary size (32000 for Eagle3)
- `--dataDir`: Directory containing your generated data

#### 2. Run training

Update `train.sh` with the appropriate parameters:

**Required Path Updates:**
- `--basepath`: **Local path** to your downloaded base model (not HuggingFace identifier)
- `--tmpdir`: Your data directory (e.g., `dataDirectory/`)
- `--configpath`: Training config file (e.g., `train/llama3_8_B.json`)
- `--cpdir`: Where you would like to save model checkpoints

Sample parameters, including the number of epochs, learning rate, and batch size, which reproduce our results are given in `train.sh`.

**CUDA Configuration:**
The training script uses distributed training with DeepSpeed. You **must** match your GPU configuration:

- `CUDA_VISIBLE_DEVICES`: Specify available GPUs
- `--num_processes`: Must equal the number of GPUs in `CUDA_VISIBLE_DEVICES`

Launch training:
```bash
./train.sh
```

### Serving the model with vLLM

#### 1. Convert your saved model
1. Modify config path in `convert.sh`, the default is using `CONFIG_PATH="train/llama3_8_B.json"`.
2. Launch conversion by running:
   ```bash
   ./convert.sh
   ```

#### 2. Serve with vLLM
```bash
VLLM_USE_V1=1 vllm serve $PATH_TO_CONVERTED_MODEL
```
Replace `$PATH_TO_CONVERTED_MODEL` with the actual directory path where your converted model was saved.

### TODO:
1. Throw an error if you attempt to create a model that will not be supported in vLLM - with the wrong configuration of heads etc.
