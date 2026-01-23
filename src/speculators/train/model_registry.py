"""Model registry for different speculator types."""

from typing import Any

import torch.nn as nn
from transformers import PretrainedConfig


def create_eagle3_model(
    verifier_config: PretrainedConfig,
    num_layers: int,
    norm_before_residual: bool,
    t2d: Any,
    d2t: Any,
    **kwargs,
) -> nn.Module:
    """Factory for Eagle3 models."""
    from speculators.config import SpeculatorsConfig, VerifierConfig
    from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
    from speculators.proposals.greedy import GreedyTokenProposalConfig

    draft_vocab_size = verifier_config.vocab_size
    transformer_layer_config = verifier_config

    speculator_config = Eagle3SpeculatorConfig(
        transformer_layer_config=transformer_layer_config,
        draft_vocab_size=draft_vocab_size,
        norm_before_residual=norm_before_residual,
        num_layers=num_layers,
        speculators_config=SpeculatorsConfig(
            algorithm="eagle3",
            proposal_methods=[
                GreedyTokenProposalConfig(
                    num_draft_tokens=kwargs.get("num_draft_tokens", 5),
                    use_dynamic_branching=kwargs.get("use_dynamic_branching", False),
                )
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig.from_config(
                verifier_config, name_or_path=kwargs.get("verifier_name_or_path")
            ),
        ),
    )

    return Eagle3DraftModel(config=speculator_config, t2d=t2d, d2t=d2t)


def create_eagle3_dataset(
    file_list: list[str],
    max_len: int,
    transform: Any | None = None,
    **kwargs,
) -> Any:
    """Factory for Eagle3 datasets."""
    from speculators.train.data import Eagle3SampleFileDataset

    return Eagle3SampleFileDataset(
        file_list=file_list,
        max_len=max_len,
        transform=transform,
        standardize_fn=kwargs.get("standardize_fn"),
    )


def create_dflash_model(
    verifier_config: PretrainedConfig,
    num_layers: int,
    norm_before_residual: bool,
    t2d: Any,
    d2t: Any,
    **kwargs,
) -> nn.Module:
    """Factory for DFlash models."""
    from speculators.config import SpeculatorsConfig, VerifierConfig
    from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
    from speculators.proposals.greedy import GreedyTokenProposalConfig

    draft_vocab_size = verifier_config.vocab_size
    transformer_layer_config = verifier_config

    speculator_config = DFlashSpeculatorConfig(
        transformer_layer_config=transformer_layer_config,
        draft_vocab_size=draft_vocab_size,
        num_hidden_layers=num_layers,
        block_size=kwargs.get("block_size", 8),
        target_hidden_size=kwargs.get("target_hidden_size"),
        aux_hidden_state_layer_ids=kwargs.get("aux_hidden_state_layer_ids"),
        speculators_config=SpeculatorsConfig(
            algorithm="dflash",
            proposal_methods=[
                GreedyTokenProposalConfig(
                    num_draft_tokens=kwargs.get("block_size", 8),
                    use_dynamic_branching=False,
                )
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig.from_config(
                verifier_config, name_or_path=kwargs.get("verifier_name_or_path")
            ),
        ),
    )

    return DFlashDraftModel(config=speculator_config, t2d=t2d, d2t=d2t)


def create_dflash_dataset(
    file_list: list[str],
    max_len: int,
    transform: Any | None = None,
    **kwargs,
) -> Any:
    """Factory for DFlash datasets.

    DFlash uses the same dataset structure as Eagle3 but expects data
    with 6 hidden state layers (5 concatenated + 1 verifier last hidden).
    The standardize_data_v1 function automatically concatenates all layers
    except the last one, making it compatible with both Eagle3 (4 layers)
    and DFlash (6 layers).
    """
    from speculators.train.data import Eagle3SampleFileDataset, standardize_data_v1

    # Use v1 standardization by default (works for both Eagle3 and DFlash)
    standardize_fn = kwargs.get("standardize_fn", standardize_data_v1)

    return Eagle3SampleFileDataset(
        file_list=file_list,
        max_len=max_len,
        transform=transform,
        standardize_fn=standardize_fn,
    )


def get_eagle3_trainer_kwargs(args: Any) -> tuple[dict, dict]:
    """Get trainer call kwargs for Eagle3.

    Returns:
        Tuple of (train_call_kwargs, val_call_kwargs)
    """
    train_kwargs = {
        "use_off_policy_tokens": args.use_off_policy_tokens,
        "ttt_steps": args.ttt_steps,
        "ttt_step_loss_decay": args.ttt_step_loss_decay,
    }
    val_kwargs = {
        "use_off_policy_tokens": False,
        "ttt_steps": args.ttt_steps,
        "ttt_step_loss_decay": args.ttt_step_loss_decay,
    }
    return train_kwargs, val_kwargs


def get_dflash_trainer_kwargs(args: Any) -> tuple[dict, dict]:
    """Get trainer call kwargs for DFlash.

    Returns:
        Tuple of (train_call_kwargs, val_call_kwargs)
    """
    return {
        "block_size": args.block_size,
    }, {
        "block_size": args.block_size,
    }


def get_eagle3_model_kwargs(args: Any) -> dict:
    """Get model-specific kwargs for Eagle3.

    Returns:
        Dictionary of model kwargs
    """
    return {
        "num_draft_tokens": args.ttt_steps,
        "use_dynamic_branching": False,
    }


def get_dflash_model_kwargs(args: Any) -> dict:
    """Get model-specific kwargs for DFlash.

    Returns:
        Dictionary of model kwargs
    """
    return {
        "block_size": args.block_size,
    }


model_factories = {
    "eagle3": create_eagle3_model,
    "dflash": create_dflash_model,
}

dataset_factories = {
    "eagle3": create_eagle3_dataset,
    "dflash": create_dflash_dataset,
}

trainer_kwargs_factories = {
    "eagle3": get_eagle3_trainer_kwargs,
    "dflash": get_dflash_trainer_kwargs,
}

model_kwargs_factories = {
    "eagle3": get_eagle3_model_kwargs,
    "dflash": get_dflash_model_kwargs,
}
