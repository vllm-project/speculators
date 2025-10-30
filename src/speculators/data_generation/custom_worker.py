"""
Custom Worker Extension for hidden states capture

This is used via vLLM's worker_extension_cls mechanism.
To use: set ParallelConfig.worker_extension_cls =
    "custom_worker:HiddenStatesWorkerExtension"
"""

import logging
from itertools import islice
from typing import Any

import torch
from vllm.distributed import get_pp_group, get_tp_group
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.sequence import IntermediateTensors

logger = logging.getLogger(__name__)


class HiddenStatesWorkerExtension:
    """Worker extension that adds hidden states capture functionality

    This extension class gets dynamically added to Worker's base classes
    via vLLM's worker_extension_cls mechanism. All methods defined here
    become available on the Worker instance in each worker process.
    """

    # Type annotations for dynamically added attributes
    _layer_ids: list[int]
    _captured_states: list[Any] | None
    _should_capture: bool
    model_runner: Any  # vLLM ModelRunner, dynamically available via mixin

    def _create_patched_forward(self, base_model):
        """Create a patched forward function for hidden state capture"""

        def patched_forward(
            input_ids, positions, intermediate_tensors=None, inputs_embeds=None
        ):
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

            # Hoist invariant checks outside the loop for performance
            should_capture_here = (
                self._should_capture and get_tp_group().rank_in_group == 0
            )
            target_layers = (
                base_model.aux_hidden_state_layers
                if should_capture_here
                else frozenset()
            )

            # Get the total number of layers to identify the last layer
            total_layers = len(base_model.layers)

            for idx, layer in enumerate(
                islice(base_model.layers, base_model.start_layer, base_model.end_layer)
            ):
                hidden_states, residual = layer(positions, hidden_states, residual)

                # Capture hidden states from target layers (only on TP rank 0)
                # idx is relative to start_layer, so convert to absolute layer index
                absolute_layer_idx = base_model.start_layer + idx
                if absolute_layer_idx in target_layers:
                    # For intermediate layers (not the last), capture before norm for feature fusion
                    # For the last layer, we'll capture after norm below
                    if absolute_layer_idx != total_layers - 1:
                        aux_hidden_states.append((hidden_states + residual).clone())

            # If not last rank in pipeline parallel, return intermediate tensors
            if not get_pp_group().is_last_rank:
                return IntermediateTensors(
                    {"hidden_states": hidden_states, "residual": residual}
                )

            # Apply final normalization
            hidden_states, _ = base_model.norm(hidden_states, residual)

            # Capture the final normalized layer if it's in target_layers
            if self._should_capture and get_tp_group().rank_in_group == 0:
                if (total_layers - 1) in target_layers:
                    aux_hidden_states.append(hidden_states.clone())

                if len(aux_hidden_states) > 0:
                    if self._captured_states is None:
                        self._captured_states = aux_hidden_states
                    else:
                        # Concatenate with previous captures along batch dimension
                        for i, h in enumerate(aux_hidden_states):
                            self._captured_states[i] = torch.cat(
                                [self._captured_states[i], h], dim=0
                            )

            return hidden_states

        return patched_forward

    def _setup_hidden_states_capture(self, layer_ids: list[int]):
        """Setup model to capture auxiliary hidden states from specific layers"""
        self._layer_ids = layer_ids
        self._captured_states = None
        self._should_capture = False

        model = self.model_runner.model

        if not supports_eagle3(model):
            raise ValueError(
                f"Model {type(model).__name__} does not support hidden state extraction"
            )

        base_model = model.model  # type: ignore[attr-defined]
        base_model.aux_hidden_state_layers = tuple(layer_ids)

        base_model.forward = self._create_patched_forward(base_model)
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
