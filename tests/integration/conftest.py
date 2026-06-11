"""Shared fixtures and factories for integration tests."""

import copy
from collections.abc import Callable

import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.eagle3 import Eagle3SpeculatorConfig
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.models.mtp import MTPSpeculatorConfig
from speculators.models.mtp.core import MTPDraftModel
from speculators.models.peagle.config import PEagleSpeculatorConfig
from speculators.models.peagle.core import PEagleDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.data import create_collate_fn

# ---------------------------------------------------------------------------
# Tiny verifier configs
# ---------------------------------------------------------------------------

TINY_LLAMA_CONFIG = LlamaConfig(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=16,
    max_position_embeddings=256,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    _attn_implementation="eager",  # type: ignore[call-arg]
)

TINY_QWEN3_CONFIG = Qwen3Config(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=16,
    max_position_embeddings=256,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
)

TINY_QWEN3_5_KWARGS: dict = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
    "layer_types": [
        "linear_attention",
        "full_attention",
        "linear_attention",
        "full_attention",
    ],
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 4,
    "linear_conv_kernel_dim": 4,
    "partial_rotary_factor": 0.25,
}


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def _fill_nan_weights(model):
    """Replace NaN-initialized weights with deterministic values."""
    with torch.no_grad():
        for param in model.parameters():
            if param.isnan().any():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
        for buf in model.buffers():
            if buf.is_floating_point() and buf.isnan().any():
                buf.zero_()


def make_eagle3_model(
    *,
    draft_vocab_size: int = 64,
    norm_before_residual: bool = False,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Eagle3DraftModel:
    """Create a tiny Eagle3 model with real initialized weights."""
    config = Eagle3SpeculatorConfig(
        transformer_layer_config=copy.deepcopy(TINY_LLAMA_CONFIG),
        draft_vocab_size=draft_vocab_size,
        norm_before_residual=norm_before_residual,
        embed_requires_grad=False,
        speculators_config=SpeculatorsConfig(
            algorithm="eagle3",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["LlamaForCausalLM"],
            ),
        ),
    )
    model = Eagle3DraftModel(config)
    _fill_nan_weights(model)
    return model.to(device=device, dtype=dtype)  # type: ignore[call-arg]


def make_dflash_model(
    *,
    draft_vocab_size: int = 64,
    block_size: int = 4,
    max_anchors: int = 8,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> DFlashDraftModel:
    """Create a tiny DFlash model with real initialized weights."""
    config = DFlashSpeculatorConfig(
        transformer_layer_config=copy.deepcopy(TINY_QWEN3_CONFIG),
        draft_vocab_size=draft_vocab_size,
        block_size=block_size,
        max_anchors=max_anchors,
        aux_hidden_state_layer_ids=[0, 1, 2],
        mask_token_id=0,
        speculators_config=SpeculatorsConfig(
            algorithm="dflash",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=block_size - 1)
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
    )
    model = DFlashDraftModel(config)
    _fill_nan_weights(model)
    return model.to(device=device, dtype=dtype)  # type: ignore[call-arg]


def make_peagle_model(
    *,
    draft_vocab_size: int = 64,
    num_depths: int = 4,
    down_sample_ratio: float = 0.7,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> PEagleDraftModel:
    """Create a tiny PEagle model with real initialized weights."""
    config = PEagleSpeculatorConfig(
        transformer_layer_config=copy.deepcopy(TINY_LLAMA_CONFIG),
        draft_vocab_size=draft_vocab_size,
        norm_before_residual=False,
        embed_requires_grad=True,
        num_depths=num_depths,
        down_sample_ratio=down_sample_ratio,
        down_sample_ratio_min=0.2,
        mask_token_id=0,
        speculators_config=SpeculatorsConfig(
            algorithm="peagle",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=num_depths)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["LlamaForCausalLM"],
            ),
        ),
    )
    model = PEagleDraftModel(config)
    _fill_nan_weights(model)
    return model.to(device=device, dtype=dtype)  # type: ignore[call-arg]


def make_mtp_model(
    *,
    num_speculative_steps: int = 3,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> MTPDraftModel:
    """Create a tiny MTP model mirroring Qwen3.5-0.8B architecture."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import (  # noqa: PLC0415
        Qwen3_5TextConfig,
    )

    config = MTPSpeculatorConfig(
        transformer_layer_config=Qwen3_5TextConfig(**TINY_QWEN3_5_KWARGS),
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[
                GreedyTokenProposalConfig(
                    speculative_tokens=num_speculative_steps,
                )
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["Qwen3_5ForCausalLM"],
            ),
        ),
    )
    model = MTPDraftModel(config)
    _fill_nan_weights(model)
    return model.to(device=device, dtype=dtype)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------


def make_sample(
    *,
    seq_len: int,
    hidden_size: int,
    hidden_multiplier: int = 3,
    vocab_size: int,
    loss_mask_pattern: str = "all",
    include_verifier_states: bool = True,
    boundary_token_ids: list[int] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Generate a single synthetic sample (no batch dim, seq_len on dim 0).

    Matches the per-sample format produced by BaseDataset.__getitem__:
        hidden_states: [seq_len, hidden_multiplier * hidden_size]
        input_ids: [seq_len]
        verifier_last_hidden_states: [seq_len, hidden_size]
        loss_mask: [seq_len]
        lengths: [1]
        position_ids: [seq_len]
    """
    hidden_states = torch.randn(seq_len, hidden_multiplier * hidden_size, dtype=dtype)

    input_ids = torch.randint(0, vocab_size, (seq_len,))
    if boundary_token_ids:
        for i, tok_id in enumerate(boundary_token_ids):
            if i < seq_len:
                input_ids[i] = tok_id

    if loss_mask_pattern == "all":
        loss_mask = torch.ones(seq_len, dtype=dtype)
    elif loss_mask_pattern == "none":
        loss_mask = torch.zeros(seq_len, dtype=dtype)
    elif loss_mask_pattern == "random":
        loss_mask = torch.randint(0, 2, (seq_len,)).to(dtype)
    elif loss_mask_pattern == "alternating":
        loss_mask = torch.zeros(seq_len, dtype=dtype)
        loss_mask[::2] = 1.0
    else:
        raise ValueError(f"Unknown loss_mask_pattern: {loss_mask_pattern}")

    result = {
        "hidden_states": hidden_states,
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "lengths": torch.tensor([seq_len], dtype=torch.long),
        "position_ids": torch.arange(seq_len, dtype=torch.long),
    }

    if include_verifier_states:
        result["verifier_last_hidden_states"] = torch.randn(
            seq_len, hidden_size, dtype=dtype
        )

    return result


def make_batch(
    *,
    max_len: int,
    samples: list[dict[str, torch.Tensor]],
    hidden_size: int,
    num_target_layers: int = 3,
    preprocess: Callable | None = None,
    device: str = "cuda:0",
) -> dict[str, torch.Tensor]:
    """Collate a list of samples into a single batch using the real collate_fn.

    Uses ``create_collate_fn`` from ``speculators.train.data`` to pack and pad
    samples exactly the way the training pipeline does.

    Args:
        max_len: Target sequence length to pad/truncate to.
        samples: List of per-sample dicts (from ``make_sample``).
        hidden_size: Model hidden size (needed by collate for empty batches).
        num_target_layers: Num of target layers in the speculative decode
            algorithm (e.g 3 for Eagle).
        preprocess: Optional per-sample transform (e.g. ``shift_batch_mtp``).
        device: Target device for the output tensors.

    Returns:
        Dict with keys matching model forward() signatures, all on ``device``.
    """
    collate_fn = create_collate_fn(
        max_len=max_len,
        hidden_size=hidden_size,
        num_target_layers=num_target_layers,
        preprocess=preprocess,
    )
    batch = collate_fn(samples)
    return {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HIDDEN_SIZE = TINY_LLAMA_CONFIG.hidden_size
VOCAB_SIZE = TINY_LLAMA_CONFIG.vocab_size


@pytest.fixture
def eagle3_model():
    model = make_eagle3_model()
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture
def dflash_model():
    model = make_dflash_model()
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture
def peagle_model():
    model = make_peagle_model()
    yield model
    del model
    torch.cuda.empty_cache()
