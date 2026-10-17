# Train a Qwen3-ASR Speculator

Train a [DFlash](../algorithms/dflash.md) head for Qwen3-ASR. The verifier, including its audio encoder, remains frozen.

## Prerequisites

Follow the [environment setup](train.md#step-0-setup-your-environment), using a current [vLLM main build](https://github.com/vllm-project/vllm) and `transformers>=5.13.0`. Use [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) or [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) as the verifier.

## Step 1: Prepare Your Data

Save audio conversations with target-model assistant responses to `qwen3_asr_conversations.jsonl`:

```json
{"conversations":[{"role":"system","content":[{"type":"text","text":""}]},{"role":"user","content":[{"type":"audio","path":"/data/audio/utterance.flac"}]},{"role":"assistant","content":[{"type":"text","text":"language English<asr_text>THE TRANSCRIPT"}]}]}
```

Replace the example assistant text with the verifier's actual response, preserving its language prefix and `<asr_text>` marker. Keep the original system and user turns, and use typed content lists for every turn as shown above.

## Step 2: Train DFlash

Edit `MODEL`, `DATASET`, `MEDIA_ROOT`, and the GPU assignments in the [online DFlash example](https://github.com/vllm-project/speculators/blob/main/examples/train/dflash_qwen3_asr_online.sh), then run it from the repository root:

```bash
bash examples/train/dflash_qwen3_asr_online.sh
```

The script starts vLLM, prepares the conversations, and trains DFlash with two trainer processes on separate GPUs from the verifier. The vLLM server must be able to read the audio files under `MEDIA_ROOT`.

Checkpoints are saved to `./output/dflash_qwen3_asr/checkpoints/`. See [Train a Speculator](train.md) for offline training and other algorithms.

## Evaluate DFlash

The [LibriSpeech evaluation example](https://github.com/vllm-project/speculators/blob/main/examples/evaluate/example_qwen3_asr_librispeech.sh) evaluates [mattymchen/Qwen3-ASR-1.7B-speculator.dflash](https://huggingface.co/mattymchen/Qwen3-ASR-1.7B-speculator.dflash) using GuideLLM's native Hugging Face audio loading. It defaults to 80 test-clean clips, one GPU, and HTTP concurrency 8:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/evaluate/example_qwen3_asr_librispeech.sh
```

`acceptance.csv` reports draft acceptance; `artifacts/run_*.json` contains GuideLLM's request throughput and latency results.
