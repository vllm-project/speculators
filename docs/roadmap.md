# Roadmap

This page outlines the current state, upcoming features, and long-term vision for Speculators.

## Current Status (v0.3.x)

Speculators is in active development with production-ready features for training and deploying speculative decoding models.

### Supported Features

✅ **Training**

- EAGLE-3 algorithm support
- DFlash algorithm support
- Online and offline training modes
- Multi-GPU distributed training with FSDP
- Vocabulary compression for draft models

✅ **Data Preparation**

- Chat template application and tokenization
- Response regeneration for data augmentation
- Turn dropout for training robustness
- Offline hidden states generation via vLLM

✅ **Model Deployment**

- vLLM integration for production serving
- Hugging Face Hub compatible model format
- Conversion tools for third-party models

✅ **Supported Architectures**

- Llama (8B, 70B)
- Qwen3 (8B, 14B, 32B)
- Qwen3 MoE (30B, 235B)
- Qwen3-VL (235B)
- gpt-oss (20B, 120B)

## In Progress

⏳ **Mistral 3 Large Support**

- Training and deployment for Mistral 3 Large 675B

⏳ **Documentation Improvements**

- Additional tutorials and examples
- Performance tuning guides
- Best practices documentation

## Upcoming Features (Q2 2026)

🔜 **Additional Algorithms**

- More speculative decoding algorithms
- Algorithm comparison benchmarks

🔜 **Training Enhancements**

- Improved memory efficiency
- Faster convergence techniques
- Advanced data augmentation strategies

🔜 **Evaluation Tools**

- Built-in benchmarking suite
- Automated acceptance rate tracking
- Performance profiling tools

## Long-term Vision

📅 **Broader Model Support**

- Additional model architectures
- Multimodal model support expansion
- Cross-architecture training

📅 **Research Integration**

- Latest speculative decoding research
- Novel acceleration techniques
- Adaptive speculation strategies

📅 **Production Features**

- Model versioning and registry
- A/B testing frameworks
- Production monitoring tools

## Release History

### v0.3.0 (December 2024)

Major release with comprehensive training pipeline and vLLM integration.

**Key Features:**

- End-to-end EAGLE-3 training
- DFlash algorithm support
- Online/offline training modes
- Multi-GPU FSDP support
- Standardized model format
- vLLM serving integration

**Models Released:**

- Llama-3.1-8B-Instruct speculator
- Llama-3.3-70B-Instruct speculator
- Qwen3 family speculators (8B, 14B, 32B)
- Qwen3 MoE speculators (30B, 235B)
- gpt-oss speculators (20B, 120B)

[View full release notes](https://blog.vllm.ai/2025/12/13/speculators-v030.html)

## Contributing to the Roadmap

We welcome community input on the roadmap!

**Ways to contribute:**

- 🐛 [Report bugs](https://github.com/vllm-project/speculators/issues/new?template=bug-report.yml)
- 💡 [Request features](https://github.com/vllm-project/speculators/discussions)
- 🗳️ Vote on existing feature requests
- 💬 Discuss ideas in `#speculators` on [Slack](https://communityinviter.com/apps/vllm-dev/join-vllm-developers-slack)

## Versioning

Speculators follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes to APIs or model format
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

## Stay Updated

- **GitHub Releases:** [github.com/vllm-project/speculators/releases](https://github.com/vllm-project/speculators/releases)
- **vLLM Blog:** [blog.vllm.ai](https://blog.vllm.ai)
- **Slack:** Join `#speculators` for announcements

______________________________________________________________________

*Last updated: April 2026*
