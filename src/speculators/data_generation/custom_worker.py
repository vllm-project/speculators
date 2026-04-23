"""Custom worker extension for hidden states capture."""

import logging
import types
from collections import defaultdict
from itertools import islice

import torch
from vllm.distributed import get_pp_group, get_tp_group
from vllm.sequence import IntermediateTensors

__all__ = ["HiddenStatesWorkerExtension"]

logger = logging.getLogger(__name__)


def _patched_forward(
    self,
    input_ids,
    positions,
    intermediate_tensors: dict[str, torch.Tensor] | None = None,
    inputs_embeds=None,
    deepstack_input_embeds: dict[str, torch.Tensor] | None = None,
    **_kwargs,
):
    """Patched forward pass that captures hidden states from specified layers.

    This function is bound to base_model instances via types.MethodType.
    It expects base_model to have an _extension attribute pointing to the
    HiddenStatesWorkerExtension instance.

    Args:
        input_ids: token id sequence (prefill path).
        positions: position ids tensor consumed by the text backbone.
        intermediate_tensors: pipeline-parallel intermediate residual/hidden.
        inputs_embeds: when the caller (VLM / thinker) has already embedded the
            tokens and scattered vision / audio features, feed that here and
            skip ``embed_input_ids``.
        deepstack_input_embeds: DeepStack visual injection dict for multimodal
            models (Qwen3VL / Qwen3-Omni). Keys are ``deepstack_input_embeds_{i}``
            where ``i`` is the absolute decoder-layer index at which the extra
            visual embedding should be additively fused into ``hidden_states``.
    """
    if get_pp_group().is_first_rank:
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    aux_hidden_states = []
    extension = self._extension  # noqa: SLF001
    # Only capture on TP rank 0 to avoid duplicates
    should_capture = get_tp_group().rank_in_group == 0
    target_layers = extension._layer_ids if should_capture else frozenset()  # noqa: SLF001

    for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
        hidden_states, residual = layer(
            hidden_states=hidden_states, positions=positions, residual=residual
        )
        absolute_layer_idx = self.start_layer + idx

        if deepstack_input_embeds:
            deepstack_key = f"deepstack_input_embeds_{absolute_layer_idx}"
            ds_embed = deepstack_input_embeds.get(deepstack_key)
            if ds_embed is not None:
                hidden_states = hidden_states + ds_embed

        # Capture intermediate layers (not the last) before norm
        if absolute_layer_idx in target_layers:
            aux_hidden_states.append((hidden_states + residual).clone())

    # Return early if not last PP rank
    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}  # type: ignore[dict-item]
        )

    hidden_states, _ = self.norm(hidden_states, residual)
    if should_capture and aux_hidden_states:
        extension._store_captured_states(aux_hidden_states)  # noqa: SLF001

    return hidden_states


def _patched_thinker_forward(self, *args, **kwargs):
    """Patched thinker forward that scatters vision/audio embeds, then delegates
    to the (already patched) text backbone so hidden-state capture on text
    decoder layers still fires.

    Supports the image / audio branches today; the video branch falls through
    to the original forward (handled upstream in vLLM).
    """
    pixel_values = kwargs.pop("pixel_values", None)
    image_grid_thw = kwargs.pop("image_grid_thw", None)
    pixel_values_videos = kwargs.pop("pixel_values_videos", None)
    video_grid_thw = kwargs.pop("video_grid_thw", None)
    second_per_grids = kwargs.pop("second_per_grids", None)
    input_features = kwargs.pop("input_features", None)
    feature_attention_mask = kwargs.pop("feature_attention_mask", None)

    input_ids = kwargs.get("input_ids")
    # If the caller already prepared ``inputs_embeds`` (nested forward, etc.),
    # don't try to re-embed / re-scatter; just delegate.
    if input_ids is None:
        return self._orig_forward(*args, **kwargs)

    # No multimodal payload → nothing to scatter; delegate to stock forward so
    # text-only prefill keeps identical numerics.
    if (
        pixel_values is None
        and pixel_values_videos is None
        and input_features is None
    ):
        return self._orig_forward(*args, **kwargs)

    inputs_embeds = self.model.get_input_embeddings()(input_ids)
    deepstack_visual_embeds = None

    # --- vision: image ---
    if pixel_values is not None and image_grid_thw is not None:
        image_payload = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        try:
            image_embeds, deepstack_visual_embeds = self._process_image_input(
                image_payload
            )
        except Exception:  # noqa: BLE001
            # Fallback: some transformers versions expose ``get_image_features``
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            deepstack_visual_embeds = None
        mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
        inputs_embeds = inputs_embeds.masked_scatter(
            mask.expand_as(inputs_embeds), image_embeds
        )

    # --- vision: video (best-effort; opt-in) ---
    if pixel_values_videos is not None and video_grid_thw is not None:
        try:
            video_embeds, _ = self._process_video_input(
                {
                    "pixel_values_videos": pixel_values_videos,
                    "video_grid_thw": video_grid_thw,
                    "second_per_grids": second_per_grids,
                }
            )
            mask = (input_ids == self.config.video_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                mask.expand_as(inputs_embeds), video_embeds
            )
        except Exception:  # noqa: BLE001
            # Defer video to the stock forward path.
            pass

    # --- audio ---
    if input_features is not None:
        try:
            audio_embeds = self.get_audio_features(
                input_features, feature_attention_mask
            )
        except Exception:  # noqa: BLE001
            audio_embeds = self.get_audio_features(input_features)
        mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
        inputs_embeds = inputs_embeds.masked_scatter(
            mask.expand_as(inputs_embeds), audio_embeds
        )

    kwargs["inputs_embeds"] = inputs_embeds
    kwargs["input_ids"] = None
    if deepstack_visual_embeds is not None:
        vision_cfg = getattr(self.config, "vision_config", None)
        deepstack_indexes = getattr(vision_cfg, "deepstack_visual_indexes", None)
        if deepstack_indexes is not None:
            kwargs["deepstack_input_embeds"] = {
                f"deepstack_input_embeds_{layer}": emb
                for layer, emb in zip(
                    deepstack_indexes,
                    deepstack_visual_embeds,
                    strict=False,
                )
            }
    return self._orig_forward(*args, **kwargs)


class HiddenStatesWorkerExtension:
    """Worker extension that adds hidden states capture functionality.

    This extension hooks into VLLM's Worker initialization by being specified
    in ParallelConfig.worker_extension_cls. It patches the model's forward pass
    to intercept and capture intermediate layer hidden states during inference.

    Key behaviors:
    - Only captures on tensor parallel (TP) rank 0 to avoid duplicate data when
      using tensor parallelism. All TP ranks compute the same hidden states, so
      capturing from rank 0 is sufficient.
    - Stores captured states in GPU memory during batch processing as lists of
      tensors, concatenating them only when retrieved via _get_captured_states().
    - Supports pipeline parallelism by handling IntermediateTensors correctly.

    Attributes:
        _layer_ids: Frozenset of layer indices for O(1) lookup during capture
        _captured_states: Accumulated hidden states per layer (GPU tensors)
        model_runner: Reference to the VLLM model runner
    """

    def _store_captured_states(self, aux_hidden_states):
        if self._captured_states is None:  # type: ignore[has-type]
            self._captured_states = [[h] for h in aux_hidden_states]
        else:
            for i, h in enumerate(aux_hidden_states):
                self._captured_states[i].append(h)

        metadata = getattr(self, "_current_request_metadata", None)
        if metadata is not None:
            # Sort by vLLM's actual batch position (vLLM reorders requests internally)
            input_batch = self.model_runner.input_batch  # type: ignore[attr-defined]
            sorted_metadata = sorted(
                metadata.items(),
                key=lambda item: input_batch.req_id_to_index.get(item[0], float("inf")),
            )
            self._request_metadata.append(sorted_metadata)  # type: ignore[has-type]

    def _setup_hidden_states_capture(self, layer_ids: list[int]):
        """Setup model to capture auxiliary hidden states from specific layers"""
        self._layer_ids = frozenset(layer_ids)  # Convert once for O(1) lookup
        self._captured_states = None  # type: ignore[assignment]

        model = self.model_runner.model  # type: ignore[attr-defined]

        # Qwen3-Omni thinker models
        if hasattr(model, "thinker"):
            thinker = model.thinker
            # Patch the thinker so it scatters vision/audio embeds and then
            # delegates to the already-patched text backbone (see below).
            if not getattr(thinker, "_orig_forward", None):
                thinker._orig_forward = thinker.forward  # noqa: SLF001
                thinker.forward = types.MethodType(_patched_thinker_forward, thinker)
            if hasattr(thinker, "get_language_model"):
                base_model = thinker.get_language_model().model
            elif hasattr(thinker, "model"):
                base_model = (
                    thinker.model.model
                    if hasattr(thinker.model, "model")
                    else thinker.model
                )
            else:
                attrs = [a for a in dir(thinker) if not a.startswith("_")]
                raise AttributeError(
                    "Could not find thinker base model with 'layers' attribute. "
                    f"Thinker type: {type(thinker).__name__}, "
                    f"Available attributes: {attrs}"
                )
        # Vision-language models
        elif hasattr(model, "get_language_model"):
            base_model = model.get_language_model().model
        # Text models
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            base_model = model.model
        else:
            attrs = [a for a in dir(model) if not a.startswith("_")]
            raise AttributeError(
                f"Could not find base model with 'layers' attribute. "
                f"Model type: {type(model).__name__}, "
                f"Available attributes: {attrs}"
            )

        base_model._extension = self  # noqa: SLF001
        base_model.forward = types.MethodType(_patched_forward, base_model)
        logger.info(f"Hidden states capture setup complete for layers {layer_ids}")

    def _set_request_metadata(self, request_metadata: dict[str, int]):
        """Set request metadata for the next forward pass.

        Args:
            request_metadata: Dict mapping request_id -> num_prefill_tokens
        """
        self._current_request_metadata = request_metadata  # type: ignore[assignment]

    def _reset_capture(self):
        """Reset captured states before starting a new batch"""
        if not hasattr(self, "_layer_ids"):
            raise RuntimeError(
                "Must call _setup_hidden_states_capture before capturing states"
            )
        self._captured_states = None  # type: ignore[assignment]
        self._request_metadata = []  # type: ignore[assignment]
        self._current_request_metadata = None  # type: ignore[assignment]

    def _get_captured_states(self):
        """Get the captured hidden states organized by request ID.

        Returns:
            Dict mapping request_id to list of tensors (one per layer),
            or None if no states captured.

        Track which tokens belong to which request across chunked prefill iterations.
        """
        if self._captured_states is None:
            return None

        # Concatenate captured states from all scheduler iterations
        concatenated_layers = [
            torch.cat(layer_tensors, dim=0) for layer_tensors in self._captured_states
        ]

        # Slice and group by request
        request_chunks: defaultdict[str, list[list[torch.Tensor]]] = defaultdict(
            lambda: [[] for _ in range(len(concatenated_layers))]
        )
        current_idx = 0

        for metadata in self._request_metadata:  # type: ignore[has-type]
            for req_id, num_tok in metadata:
                for layer_idx, layer_tensor in enumerate(concatenated_layers):
                    chunk = layer_tensor[current_idx : current_idx + num_tok].clone()
                    request_chunks[req_id][layer_idx].append(chunk)
                current_idx += num_tok

        # Concatenate chunks for each request
        result: dict[str, list[torch.Tensor]] = {
            req_id: [torch.cat(chunks, dim=0) for chunks in layer_chunks]
            for req_id, layer_chunks in request_chunks.items()
        }

        # Clear intermediate storage
        self._captured_states = None  # type: ignore[assignment]
        self._request_metadata = []  # type: ignore[assignment]
        return result
