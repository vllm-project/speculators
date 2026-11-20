"""Engine-side worker RPC for live aux hidden-state layer swapping.

Mixed into every vLLM worker via ``--worker-extension-cls`` so its public
methods become string-callable through ``collective_rpc``. This lets an
external controller change *which* target-model layers emit auxiliary hidden
states during ``method="extract_hidden_states"`` data generation, without
restarting the engine.

Only the *set* of layer indices can change; the *count* is baked into fixed
proposer buffers, the dummy-proposer KV-cache shape and the on-disk
safetensors layout, so a count change is rejected (fail closed).
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import supports_eagle3

# Namespace under "vllm." so vLLM's logging config surfaces these INFO lines.
logger = init_logger(f"vllm.plugins.{__name__}")


class AuxLayerWorkerExtension:
    """Adds aux-layer RPC methods to the worker.

    ``self`` is the concrete ``Worker`` at call time (the extension class is
    injected into the worker's bases), so ``self.get_model()`` is available.
    """

    def get_aux_hidden_state_layers_rpc(self) -> tuple[int, ...]:
        """Return the layer indices currently captured on this worker."""
        container = self._aux_layer_container()
        return tuple(getattr(container, "aux_hidden_state_layers", ()))

    def set_aux_hidden_state_layers_rpc(self, layers) -> tuple[int, ...]:
        """Swap the captured aux hidden-state layers on this worker.

        Args:
            layers: New layer indices. Must match the current count.

        Returns:
            The newly applied layer indices as a tuple.

        Raises:
            TypeError: If the target model does not support EAGLE-3 capture.
            ValueError: If the new count differs from the current count.
        """
        model = self.get_model()
        if not supports_eagle3(model):
            raise TypeError(
                f"target model {type(model).__name__} does not support "
                "aux hidden-state capture (SupportsEagle3); is the engine "
                "running with method='extract_hidden_states'?"
            )

        new = tuple(int(x) for x in layers)
        container = self._aux_layer_container()
        current = tuple(getattr(container, "aux_hidden_state_layers", ()))
        if current and len(new) != len(current):
            raise ValueError(
                f"aux hidden-state layer count is fixed at {len(current)} "
                "(proposer buffers, KV-cache and on-disk shape depend on it); "
                f"got {len(new)}: {new}"
            )

        model.set_aux_hidden_state_layers(new)
        logger.info("Swapped aux hidden-state layers %s -> %s", current, new)

        # The swap takes effect on the next forward when EITHER the engine runs
        # eager, OR the masked-accumulate patch (graph_patch.install) has made
        # the selection a live buffer input. If neither holds, the compiled
        # forward has the old layer set baked in and this is a silent no-op.
        if not self._is_eager() and not self._mask_active(container):
            logger.warning(
                "Aux layer swap applied to the model object, but the engine is "
                "NOT running with enforce_eager and the masked-accumulate patch "
                "is not active: the captured layer set is baked into the "
                "torch.compile/CUDA-graph forward and the swap will NOT take "
                "effect. Enable the 'dynamic_hidden_states' general plugin "
                "(VLLM_PLUGINS=dynamic_hidden_states) or relaunch with "
                "--enforce-eager to swap at runtime."
            )
        return new

    def _is_eager(self) -> bool:
        vllm_config = getattr(self, "vllm_config", None)
        if vllm_config is None:
            return False
        return bool(vllm_config.model_config.enforce_eager)

    def _mask_active(self, container) -> bool:
        """True if the masked-accumulate patch installed a live mask buffer.

        When present, layer selection is a runtime buffer input and the swap
        takes effect under torch.compile/CUDA graphs (no --enforce-eager).
        """
        from dynamic_hidden_states.graph_patch import MASK_ATTR

        return getattr(container, MASK_ATTR, None) is not None

    def _aux_layer_container(self):
        """Resolve the ``EagleModelMixin`` holding ``aux_hidden_state_layers``.

        Mirrors the language-model indirection in
        ``SupportsEagle3.set_aux_hidden_state_layers`` so reads and writes
        target the same object.
        """
        model = self.get_model()
        parent = model
        if hasattr(model, "get_language_model"):
            parent = model.get_language_model()
        elif hasattr(model, "language_model"):
            parent = model.language_model
        return getattr(parent, "model", parent)
