# Training Examples

We provide three example bash scripts that demonstrate the training workflow. Each script downloads one or two small toy datasets of precomputed hidden states (50 samples) and runs `scripts/train.py` to train a speculator model on the datasets.

## Commands

To execute the training examples:

```bash
bash examples/training/gpt_oss_20b_ultrachat.sh
bash examples/training/llama3_8b_sharegpt.sh
bash examples/training/qwen3_8b_sharegpt_ultrachat.sh
```

## Expected logs

When running, you'll see logs containing color-coded information like current epoch, loss, accuracy, conditional accuracy, learning_rate, etc. Example logs:

```
[20:17:13] INFO     train/loss_0=1.980, train/full_acc_0=0.029, train/cond_acc_0=0.029, train/loss_1=1.970, train/full_acc_1=9.45e-04, train/cond_acc_1=0.033, train/loss_2=2.015, train/full_acc_2=0.00e+00, train/cond_acc_2=0.00e+00, train/loss=5.965, epoch=0, lr=2.47e-05             trainer.py:191
```

## Output

Once training is complete, the scripts create an `./output` directory in the current working directory like the following:

```
output/
├── data/         # Downloaded HuggingFace datasets
└── checkpoints/  # Training checkpoints
```

The `./data` directory saves the huggingface datasets downloaded and the 'checkpoints' directory saves checkpoints produced by the training script. The trained speculator is saved in the speculator format and can directly be served in vLLM using `vllm serve <path_to_speculator>`. Note: this speculator is only trained on 50 samples so accuracy will not be high.
