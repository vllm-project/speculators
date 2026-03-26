# Online training

This readme walks through the process of online training an Eagle3 draft model.

## Prepare data

In a python environment with `speculators` installed, prepare the training dataset. Pass in the target model name/path, dataset name/path (you can pass in multiple datasets), and the output directory.

```
python scripts/prepare_data.py --model Qwen/Qwen3-8B --data sharegpt --output ./output
```

**Produces:**

```
./output/
    data-00000-of-00002.arrow    #  ⎤
    data-00001-of-00002.arrow    #  | Processed dataset on disk
    dataset_info.json            #  |
    state.json                   #  ⎦
    
    token_freq.pt                # Token frequencies for vocab mapping
```

## Launch vLLM

In a python environment with `vllm` installed, launch a vllm server configured for hidden states extraction. We provide a wrapper script (`scripts/launch_vllm.py`) to make this easier.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/launch_vllm.py Qwen/Qwen3-8B -- --data-parallel-size 4 --port 8000
```

Note: anything that comes after the `--` will be passed directly to vllm. The `--data-parallel-size` and `--port` are examples of optional arguments for configuring vLLM. `--tensor-parallel-size` also works as expected.

**Produces:** Model ready to serve requests on on port 8000

## Run training

In a python environment with `speculators` installed, launch the training process. `torchrun` (and the arguments to it) are used to launch a multi-gpu training job. These can be omitted if training on a single gpu.

```
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node 4 scripts/train.py --verifier-name-or-path Qwen/Qwen3-8B --data-path ./output --vllm-endpoint http://localhost:8000/v1 --save-path ./output/checkpoint --draft-model-size 32000
```

**Produces:** If `--draft-model-size` is set, vocab mappings will be generated and cached to the `--data-path` directory.

```
./output/
    data-00000-of-00002.arrow    #  ⎤
    data-00001-of-00002.arro     #  | 
    dataset_info.json            #  | From `scripts/prepare_data.py` step
    state.json                   #  |
    token_freq.pt                #  ⎦

    td2.npy                      #  ⎤ Vocab mappings
    d2t.npy                      #  ⎦
    
    checkpoints/                 # Training checkpoints (loadable by vLLM)
        0/
        1/
        ...
```
