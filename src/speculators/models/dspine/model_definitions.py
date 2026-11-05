"""Core modules for DSpine adjacent causal injection."""

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional

__all__ = [
    "AdjacentCausalInjection",
    "build_output_embedding_transfer_space",
]

_MATRIX_NDIM = 2


def _scaled_normalize(x: torch.Tensor, dimension: int, eps: float) -> torch.Tensor:
    normalized = functional.normalize(x.float(), dim=-1, eps=eps)
    return normalized.to(x.dtype) * math.sqrt(dimension)


@torch.no_grad()
def build_output_embedding_transfer_space(
    output_embeddings: torch.Tensor,
    transfer_dim: int,
    *,
    eps: float = 1e-6,
    covariance_chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the fixed whitened transfer embeddings and PCA readout initialization.

    The implementation follows Appendix A of the DSpine paper while accumulating
    the hidden-size covariance in chunks, avoiding a full float32 copy of a large
    vocabulary embedding table.
    """
    vocab_size, hidden_size = output_embeddings.shape
    if transfer_dim > hidden_size:
        raise ValueError(
            f"transfer_dim ({transfer_dim}) cannot exceed hidden_size ({hidden_size})."
        )

    output_device = output_embeddings.device
    compute_device = output_device
    if output_device.type == "cpu" and torch.cuda.is_available():
        # Draft construction happens on CPU before Trainer.setup_model(). Move
        # only the chunked PCA work to the assigned accelerator so a 4096x4096
        # eigendecomposition does not dominate startup time.
        compute_device = torch.device("cuda", torch.cuda.current_device())

    # Prepare the unified transfer space C
    mean = torch.zeros(hidden_size, dtype=torch.float32, device=compute_device)
    gram = torch.zeros(
        hidden_size, hidden_size, dtype=torch.float32, device=compute_device
    )
    for start in range(0, vocab_size, covariance_chunk_size):
        rows = functional.normalize(
            output_embeddings[start : start + covariance_chunk_size]
            .to(compute_device)
            .float(),
            dim=-1,
            eps=eps,
        )
        mean.add_(rows.sum(dim=0))
        gram.addmm_(rows.transpose(0, 1), rows)
    mean.div_(vocab_size)
    covariance = gram - vocab_size * torch.outer(mean, mean)

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    top_values = eigenvalues[-transfer_dim:].clamp_min(eps)
    basis = eigenvectors[:, -transfer_dim:]

    transfer_embeddings_chunks: list[torch.Tensor] = []
    inv_scale = top_values.rsqrt()
    for start in range(0, vocab_size, covariance_chunk_size):
        rows = functional.normalize(
            output_embeddings[start : start + covariance_chunk_size]
            .to(compute_device)
            .float(),
            dim=-1,
            eps=eps,
        )
        whitened = ((rows - mean) @ basis) * inv_scale
        transfer_embeddings_chunks.append(
            functional.normalize(whitened, dim=-1, eps=eps).to(output_device)
        )

    transfer_embeddings = torch.cat(transfer_embeddings_chunks, dim=0).to(
        output_embeddings.dtype
    )
    readout = (
        basis.transpose(0, 1)
        .contiguous()
        .to(device=output_device, dtype=output_embeddings.dtype)
    )
    return transfer_embeddings, readout


class AdjacentCausalInjection(nn.Module):
    """Layer-shared gated messages with layer-specific read and write maps."""

    def __init__(
        self,
        *,
        hidden_size: int,
        transfer_dim: int,
        message_dim: int,
        num_layers: int,
        block_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.transfer_dim = transfer_dim
        self.message_dim = message_dim
        self.num_layers = num_layers
        self.block_size = block_size
        self.eps = eps

        self.readouts = nn.ModuleList(
            nn.Linear(hidden_size, transfer_dim, bias=False) for _ in range(num_layers)
        )
        self.writes = nn.ModuleList(
            nn.Linear(message_dim, hidden_size, bias=False) for _ in range(num_layers)
        )
        self.message = nn.Linear(transfer_dim, message_dim, bias=False)
        self.receiver_gate = nn.Linear(transfer_dim, message_dim, bias=False)
        self.predecessor_gate = nn.Linear(transfer_dim, message_dim, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(message_dim))
        self.reset_identity_parameters()

    def reset_identity_parameters(self) -> None:
        """Make every injection an identity map at initialization."""
        for write in self.writes:
            nn.init.zeros_(cast("nn.Linear", write).weight)
        nn.init.zeros_(self.predecessor_gate.weight)
        nn.init.zeros_(self.gate_bias)

    @torch.no_grad()
    def initialize_readouts(self, readout: torch.Tensor) -> None:
        expected = (self.transfer_dim, self.hidden_size)
        if tuple(readout.shape) != expected:
            raise ValueError(
                f"readout must have shape {expected}, got {tuple(readout.shape)}."
            )
        for layer_readout in self.readouts:
            weight = cast("nn.Linear", layer_readout).weight
            weight.copy_(readout.to(weight))

    def project(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.readouts[layer_idx](hidden_states)
        return _scaled_normalize(projected, self.transfer_dim, self.eps)

    def token_features(
        self, transfer_embeddings: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        return transfer_embeddings[token_ids.long()].to(
            self.gate_bias.dtype
        ) * math.sqrt(self.transfer_dim)

    def _write(
        self,
        layer_idx: int,
        receiver_states: torch.Tensor,
        receiver_features: torch.Tensor,
        predecessor_features: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.sigmoid(
            self.receiver_gate(receiver_features)
            + self.predecessor_gate(predecessor_features)
            + self.gate_bias
        )
        message = self.message(predecessor_features)
        update = self.writes[layer_idx](gate * message)
        rms = receiver_states.float().square().mean(dim=-1, keepdim=True)
        rms = rms.add(self.eps).sqrt().to(receiver_states.dtype)
        return receiver_states + rms * update.to(receiver_states.dtype)

    def inject_layer(
        self,
        layer_idx: int,
        pre_injection_states: torch.Tensor,
        *,
        transfer_embeddings: torch.Tensor | None = None,
        anchor_token_ids: torch.Tensor | None = None,
        correct_predecessor_ids: torch.Tensor | None = None,
        curriculum_blocks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject every predecessor into its successor without crossing blocks."""
        original_shape = pre_injection_states.shape
        blocks = pre_injection_states.reshape(-1, self.block_size, self.hidden_size)
        features = self.project(layer_idx, blocks)
        predecessor = features[:, :-1]

        if curriculum_blocks is not None:
            if transfer_embeddings is None or correct_predecessor_ids is None:
                raise ValueError(
                    "curriculum requires transfer embeddings and predecessor IDs"
                )
            correct = self.token_features(transfer_embeddings, correct_predecessor_ids)
            predecessor = torch.where(
                curriculum_blocks.reshape(-1, 1, 1),
                correct.to(predecessor.dtype),
                predecessor,
            )

        if anchor_token_ids is not None:
            if transfer_embeddings is None:
                raise ValueError("known-anchor injection requires transfer embeddings")
            anchor_features = self.token_features(transfer_embeddings, anchor_token_ids)
            predecessor = torch.cat(
                (anchor_features.reshape(-1, 1, self.transfer_dim), predecessor[:, 1:]),
                dim=1,
            )

        successors = self._write(
            layer_idx,
            blocks[:, 1:],
            features[:, 1:],
            predecessor,
        )
        output = torch.cat([blocks[:, :1], successors], dim=1).reshape(original_shape)
        return output, self.project(layer_idx, output)

    def last_write(
        self,
        pre_injection_states: torch.Tensor,
        predecessor_features: torch.Tensor,
    ) -> torch.Tensor:
        """Redo only the final-layer write using decided predecessor features."""
        layer_idx = self.num_layers - 1
        blocks = pre_injection_states.reshape(-1, self.block_size, self.hidden_size)
        receiver_features = self.project(layer_idx, blocks[:, 1:])
        return self._write(
            layer_idx,
            blocks[:, 1:],
            receiver_features,
            predecessor_features,
        )
