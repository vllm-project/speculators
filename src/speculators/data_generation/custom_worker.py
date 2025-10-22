"""
Custom Worker Extension for hidden states capture

This is used via vLLM's worker_extension_cls mechanism.
To use: set ParallelConfig.worker_extension_cls = "custom_worker:HiddenStatesWorkerExtension"
"""
import torch
import logging
from typing import List
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.distributed import get_pp_group, get_tp_group
from vllm.sequence import IntermediateTensors
from itertools import islice

logger = logging.getLogger(__name__)


class HiddenStatesWorkerExtension:
    """Worker extension that adds hidden states capture functionality

    This extension class gets dynamically added to Worker's base classes
    via vLLM's worker_extension_cls mechanism. All methods defined here
    become available on the Worker instance in each worker process.
    """

    def _setup_hidden_states_capture(self, layer_ids: List[int]):
        """Setup model to capture auxiliary hidden states from specific layers"""
        self._layer_ids = layer_ids
        self._captured_states = None
        self._should_capture = False

        model = self.model_runner.model

        if not supports_eagle3(model):
            logger.warning(f"Model {type(model).__name__} does not support hidden state extraction")
            return

        base_model = model.model
        base_model.aux_hidden_state_layers = tuple(layer_ids)

        # Patch the forward pass to capture hidden states
        original_forward = base_model.forward
        worker_self = self  # Capture self reference for closure

        def patched_forward(input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
            # Get initial hidden states (first rank in pipeline parallel)
            if get_pp_group().is_first_rank:
                if inputs_embeds is not None:
                    hidden_states = inputs_embeds
                else:
                    hidden_states = base_model.get_input_embeddings(input_ids)
                residual = None
            else:
                assert intermediate_tensors is not None
                hidden_states = intermediate_tensors["hidden_states"]
                residual = intermediate_tensors["residual"]

            aux_hidden_states = []

            # Pass through each layer
            for idx, layer in enumerate(
                islice(base_model.layers, base_model.start_layer, base_model.end_layer)
            ):
                hidden_states, residual = layer(positions, hidden_states, residual)

                # Capture hidden states from target layers (only on TP rank 0)
                # idx is relative to start_layer, so convert to absolute layer index
                absolute_layer_idx = base_model.start_layer + idx
                if worker_self._should_capture and absolute_layer_idx in base_model.aux_hidden_state_layers:
                    if get_tp_group().rank_in_group == 0:
                        aux_hidden_states.append((hidden_states + residual).detach().clone())

            # If not last rank in pipeline parallel, return intermediate tensors
            if not get_pp_group().is_last_rank:
                return IntermediateTensors(
                    {"hidden_states": hidden_states, "residual": residual}
                )

            # Final normalization
            hidden_states, _ = base_model.norm(hidden_states, residual)

            # Store captured states (accumulate across batches)
            if worker_self._should_capture and len(aux_hidden_states) > 0:
                if get_tp_group().rank_in_group == 0:
                    if worker_self._captured_states is None:
                        worker_self._captured_states = aux_hidden_states
                    else:
                        # Concatenate with previous captures along batch dimension
                        for i, h in enumerate(aux_hidden_states):
                            worker_self._captured_states[i] = torch.cat([
                                worker_self._captured_states[i], h
                            ], dim=0)

            return hidden_states

        base_model.forward = patched_forward
        logger.info(f"Hidden states capture setup complete for layers {layer_ids}")

    def _enable_capture(self):
        """Enable hidden states capture"""
        self._should_capture = True
        self._captured_states = None

    def _disable_capture(self):
        """Disable hidden states capture and clear captured data"""
        self._should_capture = False
        self._captured_states = None

    def _get_captured_states(self):
        """Get the captured hidden states

        Returns:
            List of tensors, one per target layer, or None if no states captured
        """
        return self._captured_states
