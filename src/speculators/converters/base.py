"""Base converter class and utilities for model conversion."""

import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from speculators.config import SpeculatorModelConfig, SpeculatorsConfig, VerifierConfig


@dataclass
class ConversionResult:
    """Result of a model conversion operation."""
    
    success: bool
    config: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    unmapped_weights: List[str] = field(default_factory=list)
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_error(self, message: str) -> None:
        """Add an error message and mark as failed."""
        self.errors.append(message)
        self.success = False


class BaseConverter(ABC):
    """Base class for model converters."""
    
    def convert(
        self,
        source_path: str,
        output_path: str,
        verifier_model: str,
        proposal_methods: Optional[List[Dict[str, Any]]] = None,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        **kwargs
    ) -> ConversionResult:
        """
        Convert a model to Speculators format.
        
        Args:
            source_path: Path to source model directory
            output_path: Path to save converted model
            verifier_model: Name or path of the verifier model
            proposal_methods: List of proposal method configurations
            push_to_hub: Whether to push to HuggingFace Hub
            repo_id: Repository ID for pushing to hub
            **kwargs: Additional converter-specific arguments
            
        Returns:
            ConversionResult with status and any warnings/errors
        """
        result = ConversionResult(success=True)
        
        try:
            # Load source model
            print(f"Loading source model from {source_path}...")
            source_config, source_weights = self.load_source_model(source_path)
            
            # Validate source
            print("Validating source model...")
            validation_errors = self.validate_source(source_config)
            if validation_errors:
                for error in validation_errors:
                    result.add_error(error)
                return result
            
            # Load verifier config
            print(f"Loading verifier config from {verifier_model}...")
            verifier_config = self.load_verifier_config(verifier_model)
            
            # Map configuration
            print("Mapping configuration...")
            speculators_config = self.map_config(
                source_config=source_config,
                verifier_config=verifier_config,
                proposal_methods=proposal_methods,
                **kwargs
            )
            
            # Map weights
            print("Mapping weights...")
            mapped_weights, unmapped = self.map_weights(source_weights, source_config)
            result.unmapped_weights = unmapped
            
            if unmapped:
                result.add_warning(f"Found {len(unmapped)} unmapped weights")
            
            # Save converted model
            print(f"Saving converted model to {output_path}...")
            self.save_converted_model(
                output_path=output_path,
                config=speculators_config,
                weights=mapped_weights
            )
            
            result.config = speculators_config
            print("✓ Conversion completed successfully!")
            
            # Push to hub if requested
            if push_to_hub:
                if not repo_id:
                    result.add_error("repo_id is required when push_to_hub=True")
                    return result
                # TODO: Implement hub pushing
                result.add_warning("Hub pushing not yet implemented")
            
        except Exception as e:
            result.add_error(f"Conversion failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def load_source_model(self, path: str) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """Load source model configuration and weights."""
        path_obj = Path(path)
        
        # Load config
        config_path = path_obj / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Load weights
        weights = {}
        
        # Try safetensors first
        safetensors_files = list(path_obj.glob("*.safetensors"))
        if safetensors_files:
            for file in safetensors_files:
                with safe_open(file, framework="pt") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
        else:
            # Fall back to PyTorch files
            pt_files = list(path_obj.glob("*.pt")) + list(path_obj.glob("*.pth")) + list(path_obj.glob("*.bin"))
            if not pt_files:
                raise FileNotFoundError(f"No weight files found in {path}")
            
            for file in pt_files:
                state_dict = torch.load(file, map_location="cpu")
                weights.update(state_dict)
        
        return config, weights
    
    def load_verifier_config(self, verifier_model: str) -> Dict[str, Any]:
        """Load verifier model configuration."""
        # Check if it's a local path
        verifier_path = Path(verifier_model)
        if verifier_path.exists():
            config_path = verifier_path / "config.json"
            with open(config_path) as f:
                return json.load(f)
        
        # Otherwise, try to load from HuggingFace
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(verifier_model)
            return config.to_dict()
        except Exception as e:
            raise ValueError(f"Failed to load verifier config from {verifier_model}: {e}")
    
    def save_converted_model(
        self,
        output_path: str,
        config: Dict[str, Any],
        weights: Dict[str, torch.Tensor]
    ) -> None:
        """Save the converted model."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # Save weights as safetensors
        weights_path = output_dir / "model.safetensors"
        save_file(weights, weights_path)
    
    @abstractmethod
    def validate_source(self, config: Dict[str, Any]) -> List[str]:
        """Validate the source model configuration."""
        pass
    
    @abstractmethod
    def map_config(
        self,
        source_config: Dict[str, Any],
        verifier_config: Dict[str, Any],
        proposal_methods: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Map source configuration to Speculators format."""
        pass
    
    @abstractmethod
    def map_weights(
        self,
        source_weights: Dict[str, torch.Tensor],
        source_config: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """Map source weights to Speculators format."""
        pass