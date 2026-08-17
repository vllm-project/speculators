# hs_connectors

`hs_connectors` is a lightweight package that enables the transfer of hidden states between vLLM and Speculators training processes during both online and offline training. It provides a pluggable backend system where each backend registers itself via `@HiddenStatesBackend.register(name)` and exposes hooks for CLI argument injection and configuration, so that training and serving scripts can discover and configure backends without hardcoding.

The package should be installed alongside Speculators as well as within the same environment as vLLM.

## Backends

### Filesystem

The filesystem backend enables the transfer of hidden states between vLLM and Speculators on a single node with a shared filesystem. Hidden states are serialized as safetensors files with file-lock synchronization to prevent partial reads. Cached samples are stored by index and can be reused across training runs.

### Mooncake

The Mooncake backend enables multi-node training by transferring hidden states between vLLM and Speculators across nodes over TCP or RDMA, without requiring a shared filesystem. It uses a Mooncake distributed key-value store where each sample is written under a sanitized request ID, with a metadata marker written last to signal that the sample is complete and ready for consumption. This follows a similar pattern to disaggregated prefill and decode.

When using this backend, install the Mooncake transfer engine separately (`pip install mooncake-transfer-engine`).
