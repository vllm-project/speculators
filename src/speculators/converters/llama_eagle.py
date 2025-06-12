"""Converter for EAGLE and HASS models to Speculators format."""

from typing import Any, Dict, List, Optional, Tuple

import torch

from speculators.converters.base import BaseConverter
from speculators.models.llama_eagle import LlamaEagleSpeculatorConfig


class LlamaEagleConverter(BaseConverter):
    """Converter for EAGLE (v1/v2/v3) and HASS models."""
    
    # Known EAGLE/HASS model identifiers
    EAGLE_MODELS = {
        "eagle": "eagle1",  # Default EAGLE
        "eagle1": "eagle1",
        "eagle2": "eagle1",  # EAGLE2 uses same architecture as v1
        "eagle3": "eagle1",  # EAGLE3 uses same architecture as v1
        "hass": "hass",
    }
    
    def validate_source(self, config: Dict[str, Any]) -> List[str]:
        """Validate the source model configuration."""
        errors = []
        
        # Check required fields
        required_fields = ["hidden_size", "num_hidden_layers", "num_attention_heads"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate architecture
        architectures = config.get("architectures", [])
        if not architectures:
            errors.append("No architectures specified in config")
        
        # Check if it's a known EAGLE/HASS model
        model_type = config.get("model_type", "").lower()
        if model_type not in self.EAGLE_MODELS and not self._is_eagle_architecture(architectures[0]):
            errors.append(f"Unknown model type: {model_type}. Expected EAGLE or HASS variant.")
        
        return errors
    
    def map_config(
        self,
        source_config: Dict[str, Any],
        verifier_config: Dict[str, Any],
        proposal_methods: Optional[List[Dict[str, Any]]] = None,
        variant: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Map source configuration to Speculators format."""
        # Determine variant
        if variant is None:
            variant = self._detect_variant(source_config)
        
        # Create LlamaEagleSpeculatorConfig
        model_config = LlamaEagleSpeculatorConfig(
            variant=variant,
            architectures=["LlamaEagleSpeculator"],
            hidden_size=source_config["hidden_size"],
            intermediate_size=source_config.get("intermediate_size", source_config["hidden_size"] * 4),
            num_hidden_layers=source_config["num_hidden_layers"],
            num_attention_heads=source_config["num_attention_heads"],
            num_key_value_heads=source_config.get("num_key_value_heads", source_config["num_attention_heads"]),
            vocab_size=source_config.get("vocab_size", verifier_config.get("vocab_size", 32000)),
            rms_norm_eps=source_config.get("rms_norm_eps", 1e-6),
            rope_theta=source_config.get("rope_theta", 10000.0),
            max_position_embeddings=source_config.get("max_position_embeddings", 2048),
            attention_bias=source_config.get("attention_bias", False),
            attention_dropout=source_config.get("attention_dropout", 0.0),
            mlp_bias=source_config.get("mlp_bias", False),
            fusion_bias=variant == "hass",  # HASS uses fusion bias
            use_extra_layernorms=variant == "hass",  # HASS uses extra layernorms
            inputs=["input_embeddings"],  # EAGLE/HASS use embeddings as input
        )
        
        # Add speculators_config
        speculators_config = {
            "algorithm": variant,
            "proposal_methods": proposal_methods or self._get_default_proposal_methods(variant),
            "default_proposal_method": "greedy",
            "verifier": self._create_verifier_config(verifier_config),
        }
        
        # Combine into final config
        config_dict = model_config.to_dict()
        config_dict["speculators_config"] = speculators_config
        config_dict["speculators_version"] = "0.1.0"  # TODO: Get from package version
        
        return config_dict
    
    def map_weights(
        self,
        source_weights: Dict[str, torch.Tensor],
        source_config: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """Map source weights to Speculators format."""
        mapped_weights = {}
        unmapped = []
        
        # Direct mapping for most weights (EAGLE/HASS use standard naming)
        for key, tensor in source_weights.items():
            # Skip verifier weights if present
            if key.startswith("model.") and not key.startswith("model.layers."):
                unmapped.append(key)
                continue
            
            # Map fusion layer (EAGLE: fc, HASS: fc with bias)
            if key == "fc.weight":
                mapped_weights["fusion.weight"] = tensor
            elif key == "fc.bias":
                mapped_weights["fusion.bias"] = tensor
            # Map extra layernorms for HASS
            elif key.startswith("ln_") and key.endswith(".weight"):
                # ln_0.weight -> extra_layernorms.0.weight
                layer_idx = key.split("_")[1].split(".")[0]
                mapped_weights[f"extra_layernorms.{layer_idx}.weight"] = tensor
            elif key.startswith("ln_") and key.endswith(".bias"):
                # ln_0.bias -> extra_layernorms.0.bias
                layer_idx = key.split("_")[1].split(".")[0]
                mapped_weights[f"extra_layernorms.{layer_idx}.bias"] = tensor
            # Direct mapping for decoder layers
            elif key.startswith("layers.") or key == "embed_tokens.weight" or key == "lm_head.weight":
                mapped_weights[key] = tensor
            else:
                # Try direct mapping
                if self._is_valid_weight_key(key):
                    mapped_weights[key] = tensor
                else:
                    unmapped.append(key)
        
        return mapped_weights, unmapped
    
    def _detect_variant(self, config: Dict[str, Any]) -> str:
        """Detect the model variant from config."""
        model_type = config.get("model_type", "").lower()
        
        # Check explicit model type
        if model_type in self.EAGLE_MODELS:
            return self.EAGLE_MODELS[model_type]
        
        # Check architecture name
        architectures = config.get("architectures", [])
        if architectures:
            arch_lower = architectures[0].lower()
            if "hass" in arch_lower:
                return "hass"
            elif "eagle" in arch_lower:
                # Try to detect version
                if "eagle3" in arch_lower or "v3" in arch_lower:
                    return "eagle1"  # v3 uses same architecture
                elif "eagle2" in arch_lower or "v2" in arch_lower:
                    return "eagle1"  # v2 uses same architecture
                else:
                    return "eagle1"
        
        # Default to eagle1
        return "eagle1"
    
    def _is_eagle_architecture(self, architecture: str) -> bool:
        """Check if architecture name indicates EAGLE/HASS model."""
        arch_lower = architecture.lower()
        return any(name in arch_lower for name in ["eagle", "hass"])
    
    def _is_valid_weight_key(self, key: str) -> bool:
        """Check if a weight key is valid for EAGLE/HASS models."""
        valid_prefixes = [
            "layers.",
            "embed_tokens.",
            "lm_head.",
            "norm.",
            "fusion.",
            "extra_layernorms.",
        ]
        return any(key.startswith(prefix) for prefix in valid_prefixes)
    
    def _get_default_proposal_methods(self, variant: str) -> List[Dict[str, Any]]:
        """Get default proposal methods for the variant."""
        if variant in ["eagle1", "hass"]:
            # EAGLE v1 and HASS use greedy and static tree
            return [
                {
                    "proposal_type": "greedy",
                    "draft_tokens": 5,
                },
                {
                    "proposal_type": "static_tree",
                    "draft_tokens": 10,
                    "initial_branching_factor": 4,
                    "branching_factor": 2,
                    "depth": 5,
                },
            ]
        else:
            # Future variants might use dynamic trees
            return [
                {
                    "proposal_type": "greedy",
                    "draft_tokens": 5,
                },
            ]
    
    def _create_verifier_config(self, verifier_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create verifier configuration from HF config."""
        return {
            "name_or_path": verifier_config.get("_name_or_path", "unknown"),
            "architectures": verifier_config.get("architectures", ["LlamaForCausalLM"]),
            "hidden_size": verifier_config["hidden_size"],
            "vocab_size": verifier_config["vocab_size"],
            "num_attention_heads": verifier_config.get("num_attention_heads", 32),
            "num_hidden_layers": verifier_config.get("num_hidden_layers", 32),
            "num_key_value_heads": verifier_config.get("num_key_value_heads", verifier_config.get("num_attention_heads", 32)),
            "intermediate_size": verifier_config.get("intermediate_size", verifier_config["hidden_size"] * 4),
            "model_type": verifier_config.get("model_type", "llama"),
        }