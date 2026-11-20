"""Dynamic hidden-state layer swapping plugin for vLLM.

Runtime control of *which* target-model hidden layers are captured during
speculator-training data generation (``method="extract_hidden_states"``),
without restarting the engine. See ``README.md`` for usage.
"""

from dynamic_hidden_states.endpoint import AuxLayerEndpoint
from dynamic_hidden_states.graph_patch import install as install_mask_patch
from dynamic_hidden_states.worker_extension import AuxLayerWorkerExtension

__all__ = ["AuxLayerEndpoint", "AuxLayerWorkerExtension", "install_mask_patch"]
