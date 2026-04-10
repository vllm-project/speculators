# Offline Training

This readme walks through the process of offline training an Eagle3 draft model. In offline mode, hidden states are pre-generated and saved to disk, then the vLLM server is stopped before training begins. This frees all GPU memory for the training process and allows hidden states to be reused across experiments.

For online training (where hidden states are generated on-the-fly during training), see [ONLINE_TRAINING.md](ONLINE_TRAINING.md).

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

In a python environment with `vllm` installed, launch a vLLM server configured for hidden states extraction. We provide a wrapper script (`scripts/launch_vllm.py`) to make this easier. In offline mode, the server is only needed for the data generation step and will be stopped before training.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/launch_vllm.py Qwen/Qwen3-8B -- --data-parallel-size 4 --port 8000
```

Note: anything that comes after the `--` will be passed directly to vllm. The `--data-parallel-size` and `--port` are examples of optional arguments for configuring vLLM. `--tensor-parallel-size` also works as expected.

**Produces:** Model ready to serve requests on port 8000

## Generate hidden states

With the vLLM server running, generate hidden states for all samples and save them to disk.

```
python scripts/data_generation_offline2.py \
    --preprocessed-data ./output \
    --endpoint http://localhost:8000/v1 \
    --output ./output/hidden_states \
    --validate-outputs
```

Use `--concurrency` to control the number of parallel requests to the vLLM server (default: 32). Use `--max-samples` to limit the number of samples processed. The script automatically resumes from where it left off if interrupted.

**Produces:**

```
./output/
    hidden_states/
        hs_0.safetensors         #  ⎤
        hs_1.safetensors         #  | Hidden states for each sample
        ...                      #  ⎦
```

## Stop vLLM server

Stop the vLLM server to free GPU memory for training. If you launched it in the foreground, press `Ctrl+C`. If it's running in the background:

```
kill $VLLM_PID
```

## Run training

In a python environment with `speculators` installed, launch the training process. Since the vLLM server is stopped, you can use the same GPUs for training. `torchrun` (and the arguments to it) are used to launch a multi-gpu training job. These can be omitted if training on a single gpu.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 scripts/train.py \
    --verifier-name-or-path Qwen/Qwen3-8B \
    --data-path ./output \
    --hidden-states-path ./output/hidden_states \
    --save-path ./output/checkpoints \
    --draft-vocab-size 32000 \
    --on-missing raise
```

The `--on-missing raise` flag ensures training fails immediately if any hidden states are missing, rather than silently trying to generate them (which would fail since the server is stopped).

If `--draft-vocab-size` is set, vocab mappings will be generated and cached to the `--data-path` directory.

**Produces:**

```
./output/
    data-00000-of-00002.arrow    #  ⎤
    data-00001-of-00002.arrow    #  |
    dataset_info.json            #  | From `scripts/prepare_data.py` step
    state.json                   #  |
    token_freq.pt                #  ⎦

    hidden_states/               #  From `data_generation_offline2.py` step
        hs_0.safetensors         #
        hs_1.safetensors         #
        ...                      #

    d2t.npy                      #  ⎤ Vocab mappings
    t2d.npy                      #  ⎦

    checkpoints/                 # Training checkpoints (loadable by vLLM)
        0/
        1/
        ...
```

## Online vs Offline

| | Online | Offline |
|---|---|---|
| vLLM during training | Running (serves hidden states on demand) | Stopped (hidden states pre-generated) |
| GPU usage | Separate GPUs for vLLM and training | Same GPUs reused sequentially |
| Setup | Simpler (fewer steps) | More steps, but each is independent |
| Hidden states | Generated and discarded per sample | Saved to disk, reusable across experiments |

For a runnable bash script that executes the full offline workflow in one command, see [training/offline_training.sh](training/offline_training.sh).
