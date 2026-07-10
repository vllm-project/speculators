# Tutorials

Step-by-step tutorials to guide you through complete workflows, from data preparation to serving trained models in production.

## [Serve in vLLM](serve_vllm.md)

Deploy your trained speculator models in vLLM for production inference.

**Time required:** ~5 minutes

## [Train Eagle-3 Model Online](train_eagle3_online.md)

Learn how to train an Eagle-3 speculator using online training, where hidden states are generated on-demand during training.

**Time required:** ~30 mins

## [Train Eagle-3 Model Offline](train_eagle3_offline.md)

Learn how to train an Eagle-3 speculator using offline training with pre-generated hidden states.

**Time required:** ~3 hours

## [Train DFlash Model Online](train_dflash_online.md)

Learn how to train a DFlash speculator model with block-based token generation.

**Time required:** ~25 mins

## [Train P-Eagle Model Online](train_peagle_online.md)

Learn how to train a P-Eagle speculator model with COD sampling using online training, where hidden states are generated on-demand during training.

**Time required:** ~70 mins (Tested with A100s)

## [Train P-Eagle Model Offline](train_peagle_offline.md)

Learn how to train a P-Eagle speculator model with COD sampling using offline training with pre-generated hidden states.

**Time required:** ~50 minutes

## [Train MTP Model Online](train_mtp_online.md)

Learn how to finetune a model's native MTP head on domain-specific data using online training.

**Time required:** ~8 mins for Qwen3.5-9B on 2x H200 GPUs (varies by model size)

## [Response Regeneration](response_regeneration.md)

Regenerate dataset responses using your target model for improved drafter alignment.

**Time required:** ~10 minutes

## [Evaluating Model Performance](evaluating_performance.md)

Benchmark and evaluate your trained speculator models.
