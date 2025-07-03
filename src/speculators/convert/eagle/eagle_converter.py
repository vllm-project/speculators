"""
Eagle checkpoint converter with loguru logging.
"""

import json
from pathlib import Path
from typing import Optional, Union

import torch
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import LlamaConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.eagle import EagleSpeculator, EagleSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


class EagleConverter:
    """Simple converter for Eagle checkpoints."""

    LAYERNORM_MAPPINGS = {
        "embed_layernorm.weight": "embedding_layernorm.weight",
        "lm_head_layernorm.weight": "pre_lm_head_layernorm.weight",
    }

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        base_model: str,
        fusion_bias: bool = False,
        layernorms: bool = False,
        validate: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Convert an Eagle checkpoint to speculators format.

        :param input_path: Path to Eagle checkpoint (local or HuggingFace ID)
        :param output_path: Where to save converted checkpoint
        :param base_model: Base model name (e.g., meta-llama/Llama-3.1-8B-Instruct)
        :param fusion_bias: Enable fusion bias
        :param layernorms: Enable extra layernorms
        :param validate: Whether to validate the converted checkpoint
        :param cache_dir: Optional cache directory for downloads
        """
        logger.info(f"Converting Eagle checkpoint: {input_path}")

        local_path = self._ensure_local(input_path, cache_dir=cache_dir)

        config_dict, weights = self._load_checkpoint(local_path)
        logger.info(f"Loaded {len(weights)} weights")

        if not fusion_bias and "fc.bias" in weights:
            logger.info("Detected fusion bias in checkpoint")
            fusion_bias = True
        if not layernorms and any(
            name in weights
            for name in ["embed_layernorm.weight", "post_embedding_layernorm.weight"]
        ):
            logger.info("Detected extra layernorms in checkpoint")
            layernorms = True

        config = self._build_config(config_dict, base_model, fusion_bias, layernorms)
        weights = self._process_weights(weights, layernorms)

        output_path = Path(output_path)
        self._save_checkpoint(output_path, config, weights)
        logger.success(f"Saved to: {output_path}")

        if validate:
            self._validate(output_path, verifier_name=base_model)

    def _ensure_local(
        self, path: Union[str, Path], cache_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Download checkpoint if it's a HuggingFace ID.

        :param path: Checkpoint path or HuggingFace ID
        :param cache_dir: Optional cache directory for downloads
        :return: Local path to checkpoint
        """
        path = Path(path) if isinstance(path, str) else path

        if path.exists():
            logger.debug(f"Using local checkpoint: {path}")
            return path

        logger.info(f"Downloading checkpoint from HuggingFace: {path}")
        try:
            local_path = snapshot_download(
                repo_id=str(path),
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.index.json"],
                cache_dir=str(cache_dir) if cache_dir else None,
            )
            logger.debug(f"Downloaded to: {local_path}")
            return Path(local_path)
        except Exception as e:
            logger.error(f"Failed to download checkpoint: {e}")
            raise FileNotFoundError(f"Checkpoint not found: {path}") from e

    def _load_checkpoint(self, path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
        """
        Load config and weights from checkpoint.

        :param path: Path to checkpoint directory
        :return: Config dict and weights dict
        """
        config_path = path / "config.json"
        if not config_path.exists():
            logger.error(f"No config.json found at {path}")
            raise FileNotFoundError(f"No config.json found at {path}")

        logger.debug(f"Loading config from: {config_path}")
        with config_path.open() as f:
            config_dict = json.load(f)

        weights = {}

        safetensors_path = path / "model.safetensors"
        if safetensors_path.exists():
            logger.debug(f"Loading safetensors weights from: {safetensors_path}")
            with safe_open(safetensors_path, framework="pt") as f:
                for key in f.keys():  # noqa: SIM118
                    weights[key] = f.get_tensor(key)
        else:
            pytorch_path = path / "pytorch_model.bin"
            if pytorch_path.exists():
                logger.debug(f"Loading PyTorch weights from: {pytorch_path}")
                weights = torch.load(pytorch_path, map_location="cpu")
            else:
                index_paths = [
                    path / "model.safetensors.index.json",
                    path / "pytorch_model.bin.index.json",
                ]
                for index_path in index_paths:
                    if index_path.exists():
                        logger.error(f"Sharded checkpoint detected: {index_path}")
                        raise NotImplementedError(
                            "Sharded checkpoints not yet supported. "
                            "Please use a single-file checkpoint."
                        )

                logger.error(f"No weights found at {path}")
                raise FileNotFoundError(f"No weights found at {path}")

        return config_dict, weights

    def _build_config(
        self,
        config_dict: dict,
        base_model: str,
        fusion_bias: bool,
        layernorms: bool,
    ) -> EagleSpeculatorConfig:
        """
        Build EagleSpeculatorConfig.

        :param config_dict: Original checkpoint config
        :param base_model: Base model name
        :param fusion_bias: Whether to enable fusion bias
        :param layernorms: Whether to enable extra layernorms
        :return: Eagle speculator config
        """
        logger.debug("Building EagleSpeculatorConfig")

        transformer_config = LlamaConfig(
            vocab_size=config_dict.get("vocab_size", 32000),
            hidden_size=config_dict.get("hidden_size", 4096),
            intermediate_size=config_dict.get("intermediate_size", 11008),
            num_hidden_layers=1,
            num_attention_heads=config_dict.get("num_attention_heads", 32),
            num_key_value_heads=config_dict.get("num_key_value_heads"),
            hidden_act=config_dict.get("hidden_act", "silu"),
            max_position_embeddings=config_dict.get("max_position_embeddings", 4096),
            initializer_range=config_dict.get("initializer_range", 0.02),
            rms_norm_eps=config_dict.get("rms_norm_eps", 1e-6),
            use_cache=config_dict.get("use_cache", True),
            pad_token_id=config_dict.get("pad_token_id"),
            bos_token_id=config_dict.get("bos_token_id", 1),
            eos_token_id=config_dict.get("eos_token_id", 2),
            tie_word_embeddings=False,
            rope_theta=config_dict.get("rope_theta", 10000.0),
            rope_scaling=config_dict.get("rope_scaling"),
            attention_bias=config_dict.get("attention_bias", False),
            attention_dropout=config_dict.get("attention_dropout", 0.0),
            mlp_bias=config_dict.get("mlp_bias", False),
        )

        verifier_config = VerifierConfig(
            name_or_path=base_model,
            architectures=config_dict.get("architectures", ["LlamaForCausalLM"]),
            vocab_size=config_dict.get("vocab_size", 32000),
            hidden_size=config_dict.get("hidden_size", 4096),
            intermediate_size=config_dict.get("intermediate_size", 11008),
            max_position_embeddings=config_dict.get("max_position_embeddings", 4096),
            bos_token_id=config_dict.get("bos_token_id", 1),
            eos_token_id=[config_dict.get("eos_token_id", 2)]
            if isinstance(config_dict.get("eos_token_id", 2), int)
            else config_dict.get("eos_token_id", [2]),
        )

        greedy_proposal = GreedyTokenProposalConfig(
            proposal_type="greedy",
            speculative_tokens=5,
        )

        speculators_config = SpeculatorsConfig(
            algorithm="eagle",
            proposal_methods=[greedy_proposal],
            default_proposal_method="greedy",
            verifier=verifier_config,
        )

        logger.debug(
            f"Config built with fusion_bias={fusion_bias}, layernorms={layernorms}"
        )

        return EagleSpeculatorConfig(
            transformer_layer_config=transformer_config,
            speculators_config=speculators_config,
            layernorms=layernorms,
            fusion_bias=fusion_bias,
        )

    def _process_weights(
        self,
        weights: dict[str, torch.Tensor],
        layernorms: bool,
    ) -> dict[str, torch.Tensor]:
        """
        Process weights, applying any necessary transformations.

        :param weights: Original checkpoint weights
        :param layernorms: Whether layernorms are enabled
        :return: Processed weights
        """
        logger.debug(f"Processing {len(weights)} weights")
        processed = {}
        skipped = []
        remapped = []

        for name, tensor in weights.items():
            result = self._process_single_weight(name, tensor, layernorms)
            if result is None:
                skipped.append(name)
            elif isinstance(result, tuple):
                new_name, new_tensor = result
                processed[new_name] = new_tensor
                remapped.append(f"{name} -> {new_name}")
            else:
                processed[name] = tensor

        if skipped:
            logger.debug(f"Skipped weights: {skipped}")
        if remapped:
            logger.debug(f"Remapped weights: {remapped}")

        return processed

    def _process_single_weight(
        self,
        name: str,
        tensor: torch.Tensor,
        layernorms: bool,
    ) -> Union[None, torch.Tensor, tuple[str, torch.Tensor]]:
        """
        Process a single weight, returning None to skip, the tensor to keep as-is,
        or a tuple of (new_name, tensor) to remap.
        """
        # Skip embed_tokens.weight as it's tied to lm_head in the model
        if name == "embed_tokens.weight":
            logger.debug("Skipping embed_tokens.weight (tied to lm_head)")
            return None

        # Handle hidden_layernorm
        if name == "hidden_layernorm.weight":
            return (
                ("transformer.input_layernorm.weight", tensor) if layernorms else None
            )

        # Handle layernorm mappings
        if layernorms and name in self.LAYERNORM_MAPPINGS:
            return (self.LAYERNORM_MAPPINGS[name], tensor)

        # Handle fc weight/bias remapping
        if name in ("fc.weight", "fc.bias"):
            new_name = name.replace("fc.", "fusion_fc.")
            return (new_name, tensor)

        # Handle transformer layer remapping
        if name.startswith("layers.0."):
            new_name = name.replace("layers.0.", "transformer.")
            return (new_name, tensor)

        # Keep weight as-is
        return tensor

    def _save_checkpoint(
        self,
        output_path: Path,
        config: EagleSpeculatorConfig,
        weights: dict[str, torch.Tensor],
    ) -> None:
        """
        Save checkpoint in speculators format.

        :param output_path: Output directory path
        :param config: Eagle speculator config
        :param weights: Model weights
        """
        output_path.mkdir(parents=True, exist_ok=True)

        config_path = output_path / "config.json"
        logger.debug(f"Saving config to: {config_path}")
        config_dict = config.to_dict()
        with config_path.open("w") as f:
            json.dump(config_dict, f, indent=2)

        weights_path = output_path / "model.safetensors"
        logger.debug(f"Saving weights to: {weights_path}")
        save_file(weights, weights_path)

    def _validate(
        self, checkpoint_path: Path, verifier_name: Optional[str] = None
    ) -> None:
        """
        Validate the converted checkpoint.

        :param checkpoint_path: Path to converted checkpoint
        :param verifier_name: Optional verifier model name for validation
        :raises Exception: If validation fails
        """
        logger.info("Validating converted checkpoint...")

        try:
            logger.debug("Loading model with EagleSpeculator.from_pretrained")
            if verifier_name:
                model = EagleSpeculator.from_pretrained(
                    checkpoint_path,
                    verifier=verifier_name,
                    verifier_attachment_mode="full",
                )
            else:
                model = EagleSpeculator.from_pretrained(checkpoint_path)
            logger.success("Model loaded successfully")

            # Test forward pass only if model is not on meta device
            device = next(model.parameters()).device
            if device.type != "meta":
                batch_size = 1
                seq_length = 10
                hidden_size = model.config.transformer_layer_config.hidden_size

                logger.debug(
                    f"Running forward pass with batch_size={batch_size}, "
                    f"seq_length={seq_length}"
                )
                input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
                hidden_states = torch.randn(batch_size, seq_length, hidden_size).to(
                    device
                )

                with torch.no_grad():
                    model(input_ids=input_ids, hidden_states=hidden_states)

                logger.success("Forward pass successful")
            else:
                logger.debug("Skipping forward pass test (model on meta device)")

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
