#Eagle 3
#### This code is based on the HASS code (https://github.com/HArmonizedSS/HASS) and trains models which are similar to the Eagle 3 architecture.  It implements the Train Time Test method of Eagle 3.  The original Eagle code can be found here: https://github.com/SafeAILab/EAGLE

## To Run:

The training process is broken up in to 2 steps, the first where you generate data from the large model, and the second where you actually train the drafter model. It works for Llama 3.1.8B-Instruct and Qwen 3 8B.

### Data Generation Step

1. Modify the directory names and arguments in `gen_data.sh`.
2. You can get the ShareGPT dataset from [Aeala/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered) on Hugging Face. Ultrachat will be automatically downloaded.
3. Make sure the system prompts and chat template delimiters are correct in the corresponding data loader files:  
   - `ultrachat.py`, `ultrachatMistral.py`  
   - `sharegpt.py`, `sharegptMistral.py`
4. Run the script to generate forward data:
    ```bash
    ./gen_data.sh
    ```
5. Run it separately for each dataset and split:
    - ShareGPT (sft and gen)
    - Ultrachat (sft and gen)

> 💡 **Note**: For LLaMA 3.1 8B, this process may generate ~4TB of data. The training script will search the data directory recursively, so the folder structure doesn't need to be flat.

### Training Step

1. Generate Zipf-restricted vocabulary mapping:  
    After data generation, run:
    ```bash
    ./zipf.sh
    ```
    This will create `d2t.npy` and `t2d.npy` files, which are needed for training.  
    Make sure to set the correct values for:
    - number of samples
    - vocabulary size
    - data directory  
    in the `zipf.sh` script. We recommend using at least 10,000 examples for accurate frequency-based pruning.
2. Update `train.sh` with the appropriate paths to your **base model**, **data directory**, and **training configuration** for your experiment.
3. Launch training by running:
   ```bash
   ./train.sh
   ```


### Serve the model with vllm:
1. Modify config path in `convert.sh`, the default is using `CONFIG_PATH="train/llama3_8_B.json"`.
2. Launch conversion by running:
   ```bash
   ./convert.sh
   ```
2. Serve the model with: ` VLLM_USE_V1=1 vllm serve $PATH_TO_CONVERTED_MODEL`, replace $PATH_TO_CONVERTED_MODEL with the actual directory path where your converted model was saved.



### TODO:
1. Throw an error if you attempt to create a model that will not be supported in vllm - with the wrong configuration of heads etc.
