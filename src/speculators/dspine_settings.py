"""Shared DSpine checkpoint and training settings."""

from pydantic import BaseModel, Field


class DSpineSettings(BaseModel):
    """Injection, refinement, and supervision settings."""

    transfer_dim: int = Field(
        default=1024,
        ge=1,
        description="Dimension of the PCA-whitened output-embedding transfer space.",
    )
    message_dim: int = Field(
        default=512,
        ge=1,
        description="Dimension of the gated adjacent-injection message.",
    )
    transition_top_k: int = Field(
        default=16,
        ge=1,
        description="Number of unary candidates retained by transition decoding.",
    )
    distribution_loss_alpha: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Cross-entropy weight in the CE/L1 distribution objective.",
    )
    embedding_loss_beta: float = Field(
        default=0.5,
        ge=0.0,
        description="Maximum multiplier for layer-wise embedding supervision.",
    )
    embedding_loss_warmup_steps: int = Field(
        default=146,
        ge=0,
        description="Linear warmup steps for layer-wise embedding supervision.",
    )
    curriculum_probability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Initial probability of injecting correct predecessors.",
    )
    curriculum_hold_ratio: float = Field(
        default=1.0 / 6.0,
        ge=0.0,
        le=1.0,
        description="Training fraction for which curriculum probability is held.",
    )
    curriculum_end_ratio: float = Field(
        default=1.0 / 3.0,
        ge=0.0,
        le=1.0,
        description="Training fraction at which correct-predecessor injection ends.",
    )
    curriculum_layers: int = Field(
        default=3,
        ge=0,
        description="Number of shallow layers receiving curriculum substitution.",
    )
    transition_min_target_mass: float = Field(
        default=1e-4,
        ge=0.0,
        le=1.0,
        description="Minimum target mass required for the top-k L1 objective.",
    )
    transfer_space_eps: float = Field(
        default=1e-6,
        gt=0.0,
        description="Numerical floor used by PCA whitening and normalization.",
    )
