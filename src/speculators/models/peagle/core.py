"""P-EAGLE draft model implementation with parallel multi-token prediction."""

from typing import ClassVar

import torch
from torch.nn.attention.flex_attention import create_block_mask
from transformers import PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.models.peagle.attention import create_peagle_mask_mod
from speculators.models.peagle.config import PEagleSpeculatorConfig
from speculators.models.peagle.data import generate_cod_sample_indices
from speculators.models.peagle.metrics import compute_metrics
from speculators.proposals.greedy import GreedyTokenProposalConfig


@SpeculatorModel.register("peagle")
class PEagleDraftModel(Eagle3DraftModel):
    """
    P-EAGLE (Parallel EAGLE) draft model for speculative decoding.

    P-EAGLE extends EAGLE-3 with parallel multi-token prediction using
    Conditional-On-Distribution (COD) sampling for memory-efficient training.

    Attributes:
        mask_hidden: Learnable padding parameter [1, 1, 3*hidden_size]
        fc: Projection layer from 3*hidden_size to hidden_size
        layers: Transformer decoder layers from EAGLE-3
        para_depths: Number of parallel prediction depths (groups)
    """

    # Epsilon for numerical stability in loss normalization
    LOSS_EPSILON: ClassVar[float] = 1e-5

    config_class: ClassVar[type[PEagleSpeculatorConfig]] = PEagleSpeculatorConfig
    _attn_implementation_name: ClassVar[str] = "peagle_flex_attention"
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [
        *Eagle3DraftModel._keys_to_ignore_on_load_missing,
        "mask_hidden",
    ]

    def __init__(
        self,
        config: PEagleSpeculatorConfig,
        t2d: torch.Tensor | None,
        d2t: torch.Tensor | None,
    ):
        super().__init__(config=config, t2d=t2d, d2t=d2t)

        self.para_depths = config.para_depths
        self.down_sample_ratio = config.down_sample_ratio
        self.down_sample_ratio_min = config.down_sample_ratio_min
        self.ptd_token_id = config.ptd_token_id
        self.max_seq_len = config.max_seq_len

        self.lm_head.weight.requires_grad = False
        self.lm_head.eval()

        # Learnable mask_hidden parameter for padding unsampled positions
        self.mask_hidden = torch.nn.Parameter(torch.zeros(1, 1, 3 * self.hidden_size))

    def _pad_hidden_states(self, intensors, target_len, hidden, index_list):
        """
        Pad hidden states to target length using learnable mask_hidden parameter.

        This matches the p-eagle-train paddinghidden implementation exactly.

        Args:
            intensors: Input tensor [batch, n, hidden_size]
            target_len: Target sequence length to pad to
            hidden: Learnable padding parameter (self.mask_hidden)
            index_list: List to append current length to (for tracking)

        Returns:
            Padded tensor [batch, target_len, hidden_size]
        """
        batch_size, current_len, hidden_size = intensors.shape
        index_list.append(current_len)

        # Move hidden to same device and dtype as intensors
        # mask_hidden is in ignored_params for FSDP, so it's replicated not sharded
        hidden = hidden.to(device=intensors.device, dtype=intensors.dtype)
        padding_tensor = hidden.expand(
            batch_size, target_len - current_len, hidden_size
        )

        # Concatenate
        return torch.cat((intensors, padding_tensor), dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        verifier_last_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        Forward pass for P-EAGLE model training with parallel group prediction.

        Matches p-eagle-train implementation but accepts standard EAGLE3 data format.

        Args:
            hidden_states: Verifier hidden states [batch, seq_len, 3*hidden_size]
            input_ids: Input token IDs [batch, seq_len]
            lengths: Sequence lengths for each sample in batch [batch_size]
            attention_mask: Attention mask (optional, created from sample_indices)
            position_ids: Position IDs [batch, seq_len] (optional)
            loss_mask: Loss mask for which tokens to compute loss on
                [batch, seq_len]
            verifier_last_hidden_states: Verifier final hidden states for
                targets [batch, seq_len, hidden_size]

        Returns:
            Tuple of (draft_tokens, loss, metrics)
        """
        device = hidden_states.device
        seq_length = input_ids.shape[1]

        if lengths is None:
            lengths = torch.tensor([seq_length], dtype=torch.long, device=device)
        if loss_mask is None:
            loss_mask = torch.ones_like(input_ids, dtype=torch.float32)

        # Generate COD sampling indices (moved from collate function)
        sample_indices, _ = generate_cod_sample_indices(
            seq_length=seq_length,
            loss_mask=loss_mask,
            para_num=self.para_depths,
            down_sample_ratio=self.down_sample_ratio,
            down_sample_ratio_min=self.down_sample_ratio_min,
        )

        # Use actual number of depths returned (may be less than para_depths
        # if sampling stopped early)
        para_depth = len(sample_indices)

        # Generate sample_ids to prevent cross-sample attention
        sample_ids_list = []
        cum_len = 0
        for sample_id, length in enumerate(lengths):
            sample_ids_list.append(
                torch.full((length.item(),), sample_id, dtype=torch.long, device=device)
            )
            cum_len += length.item()

        if cum_len < seq_length:
            pad_id = len(lengths) - 1 if len(lengths) > 0 else 0
            sample_ids_list.append(
                torch.full(
                    (seq_length - cum_len,), pad_id, dtype=torch.long, device=device
                )
            )

        sample_ids = torch.cat(sample_ids_list, dim=0).unsqueeze(0)  # [1, seq_len]

        # Create list of hidden states by sampling from full tensor
        hidden_states_list = [
            hidden_states[:, indices, :] for indices in sample_indices
        ]

        # Pad each sampled group to seq_length
        index = []
        padded_hidden_states = [
            self._pad_hidden_states(item, seq_length, self.mask_hidden, index)
            for item in hidden_states_list
        ]
        hidden_states_tensor = torch.cat(padded_hidden_states, dim=1)

        input_ids = input_ids.repeat(1, para_depth)
        inputs_embeds = self.embed_tokens(input_ids)

        single_pos = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = single_pos.repeat(para_depth).unsqueeze(0)
        inputs_embeds = inputs_embeds.to(hidden_states_tensor.dtype)

        # Project concatenated hidden states (3*hidden_size) -> hidden_size
        hidden_states_tensor = self.fc(hidden_states_tensor)

        layer_input = torch.cat([inputs_embeds, hidden_states_tensor], dim=-1)

        position_embeddings = self.rotary_emb(layer_input, position_ids)

        all_indices_list = []
        for depth, indices in enumerate(sample_indices):
            encoded_indices = depth * seq_length + indices
            all_indices_list.append(encoded_indices)

        all_indices = torch.cat(all_indices_list, dim=0)  # [total_sampled_length]

        layer_input = layer_input[:, all_indices, :]
        position_ids = position_ids[:, all_indices]

        position_embeddings = self.rotary_emb(layer_input, position_ids)

        mask_mod = create_peagle_mask_mod(
            all_indices=all_indices,
            seq_length=seq_length,
            para_depth=para_depth,
            sample_ids=sample_ids,
        )

        attention_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=len(all_indices),
            KV_LEN=len(all_indices),
            device=device,
        )

        layer_outputs = self.layers[0](
            layer_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states_tensor = (
            layer_outputs
            if isinstance(layer_outputs, torch.Tensor)
            else layer_outputs[0]
        )

        for decoder_layer in self.layers[1:]:
            layer_outputs = decoder_layer(
                hidden_states_tensor,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states_tensor = (
                layer_outputs
                if isinstance(layer_outputs, torch.Tensor)
                else layer_outputs[0]
            )

        logits = self.lm_head(self.norm(hidden_states_tensor))

        assert verifier_last_hidden_states is not None, (
            "verifier_last_hidden_states required for training"
        )
        with torch.no_grad():
            targets = self.verifier_lm_head(verifier_last_hidden_states)

        targets_full = targets.repeat(1, para_depth, 1)
        targets = targets_full[:, all_indices, :]

        # Use per-depth normalization, each depth should contribute equally to gradient
        with torch.no_grad():
            target_probs = torch.nn.functional.softmax(targets, dim=-1)

        # Compute loss using KL divergence (like EAGLE3)
        logits_log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
        per_token_pred_loss = (
            torch.nn.functional.kl_div(
                logits_log_softmax.reshape(-1, logits.shape[-1]),
                target_probs.reshape(-1, target_probs.shape[-1]),
                reduction="none",
                log_target=False,
            )
            .sum(dim=-1)
            .reshape(logits.shape[0], logits.shape[1])
        )  # [1, len(all_indices)]

        # Pre-compute correctness for accuracy (no gradients needed)
        with torch.no_grad():
            pred_tokens = torch.argmax(logits, dim=-1)
            target_tokens = torch.argmax(target_probs, dim=-1)
            correct = (pred_tokens == target_tokens).float()

            if loss_mask is not None:
                while loss_mask.ndim > 2:
                    loss_mask = loss_mask.squeeze(1)

        # Single loop for both loss and accuracy computation
        # Each depth contributes equally to gradient
        prediction_loss = 0.0
        total_correct = 0.0
        total_tokens = 0.0
        start_idx = 0

        for _depth, indices in enumerate(sample_indices):
            num_samples = len(indices)
            end_idx = start_idx + num_samples

            # Extract depth slice
            depth_pred_loss = per_token_pred_loss[:, start_idx:end_idx]

            if loss_mask is not None:
                depth_loss_mask = loss_mask[:, indices]
                depth_pred_loss = depth_pred_loss * depth_loss_mask
                depth_normalizer = depth_loss_mask.sum() + self.LOSS_EPSILON

                # Accuracy computation (reuse the same mask)
                depth_correct = correct[:, start_idx:end_idx]
                total_correct += (depth_correct * depth_loss_mask).sum()
                total_tokens += depth_loss_mask.sum()
            else:
                depth_normalizer = num_samples + self.LOSS_EPSILON

            # Accumulate per-depth loss
            depth_pred_loss_mean = depth_pred_loss.sum() / depth_normalizer
            prediction_loss += depth_pred_loss_mean / para_depth

            start_idx = end_idx

        # Finalize loss and accuracy
        loss = self.config.prediction_loss_weight * prediction_loss

        with torch.no_grad():
            if loss_mask is not None:
                accuracy = total_correct / (total_tokens + self.LOSS_EPSILON)
                num_tokens = total_tokens
            else:
                num_tokens = correct.numel()
                accuracy = correct.sum() / (num_tokens + self.LOSS_EPSILON)

            metrics = compute_metrics(
                loss=loss,
                prediction_loss=prediction_loss,
                accuracy=accuracy,
                num_tokens=num_tokens,
                all_indices=all_indices,
                seq_length=seq_length,
                para_depth=para_depth,
                pred_tokens=pred_tokens,
                target_tokens=target_tokens,
                device=loss.device,
                epsilon=self.LOSS_EPSILON,
            )

        if sample_indices is not None:
            first_depth_len = len(sample_indices[0])
            draft_tokens = torch.argmax(
                logits[:, :first_depth_len], dim=-1
            )  # [1, first_depth_len]
        else:
            draft_tokens = torch.argmax(logits[:, :seq_length], dim=-1)  # [1, seq_len]

        return draft_tokens, loss, metrics

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        **kwargs,
    ) -> "PEagleDraftModel":
        """
        Create P-EAGLE model from training arguments.

        Args:
            verifier_config: Verifier model configuration
            **kwargs: Training arguments with P-EAGLE-specific params
                - draft_vocab_size: Size of draft vocabulary
                - norm_before_residual: Whether to normalize before residual
                - para_depths: Number of parallel groups (default 8)
                - down_sample_ratio: COD sampling ratio (default 0.7)
                - down_sample_ratio_min: Minimum sampling ratio (default 0.2)
                - unused_token_id: Padding token ID (default 0)
                - t2d: Target-to-draft vocabulary mapping
                - d2t: Draft-to-target vocabulary mapping
                - verifier_name_or_path: Path to verifier model

        Returns:
            Initialized PEagleDraftModel
        """
        config = PEagleSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            norm_before_residual=kwargs.get("norm_before_residual", False),
            para_depths=kwargs.get("para_depths", 8),
            down_sample_ratio=kwargs.get("down_sample_ratio", 0.7),
            down_sample_ratio_min=kwargs.get("down_sample_ratio_min", 0.2),
            ptd_token_id=kwargs.get("ptd_token_id", 0),
            max_seq_len=kwargs.get("max_seq_len", 2048),
            prediction_loss_weight=kwargs.get("prediction_loss_weight", 1.0),
            speculators_config=SpeculatorsConfig(
                algorithm="peagle",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=kwargs.get("para_depths", 8),
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config, name_or_path=kwargs["verifier_name_or_path"]
                ),
            ),
        )

        return cls(config=config, t2d=kwargs.get("t2d"), d2t=kwargs.get("d2t"))

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:  # noqa: ARG004
        """
        Get training and validation kwargs for P-EAGLE.

        P-EAGLE doesn't need extra kwargs for forward() - all parameters
        are handled in the forward method (para_depths, down_sample_ratio, etc.)

        Args:
            **kwargs: Training arguments (unused)

        Returns:
            Tuple of (train_call_kwargs, val_call_kwargs) - both empty for P-EAGLE
        """
        return {}, {}
