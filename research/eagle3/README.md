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
2. Serve the model with: ` VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct --seed 42 -tp 1 --speculative-config '{"model": "llama_eagle3", "num_speculative_tokens": 3, "method":"eagle3", "draft_tensor_parallel_size":1}'`



### TODO:
1. Throw an error if you attempt to create a model that will not be supported in vllm - with the wrong configuration of heads etc.
