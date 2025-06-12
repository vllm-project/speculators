"""High-level conversion API for Speculators."""

from typing import Any, Dict, List, Literal, Optional, Union

from speculators.converters import LlamaEagleConverter
from speculators.converters.base import ConversionResult


def convert(
    source_path: str,
    output_path: str,
    verifier_model: str,
    algorithm: Literal["eagle", "eagle1", "eagle2", "eagle3", "hass"],
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
        verifier_model: Name or path of the verifier model (e.g., "meta-llama/Llama-3.1-8B")
        algorithm: The algorithm/variant to convert to
        proposal_methods: List of proposal method configurations (optional)
        push_to_hub: Whether to push to HuggingFace Hub
        repo_id: Repository ID for pushing to hub (required if push_to_hub=True)
        **kwargs: Additional algorithm-specific arguments
        
    Returns:
        ConversionResult with status and any warnings/errors
        
    Examples:
        >>> # Convert EAGLE model
        >>> result = convert(
        ...     source_path="/path/to/eagle/model",
        ...     output_path="/path/to/output",
        ...     verifier_model="meta-llama/Llama-3.1-8B-Instruct",
        ...     algorithm="eagle"
        ... )
        
        >>> # Convert HASS model with custom proposal methods
        >>> result = convert(
        ...     source_path="/path/to/hass/model",
        ...     output_path="/path/to/output",
        ...     verifier_model="meta-llama/Llama-3.1-70B",
        ...     algorithm="hass",
        ...     proposal_methods=[
        ...         {"proposal_type": "greedy", "draft_tokens": 10},
        ...         {"proposal_type": "sampling", "draft_tokens": 5, "temperature": 0.7}
        ...     ]
        ... )
    """
    # Map algorithm to converter and variant
    algorithm_lower = algorithm.lower()
    
    if algorithm_lower in ["eagle", "eagle1", "eagle2", "eagle3", "hass"]:
        converter = LlamaEagleConverter()
        # Map algorithm names to variants
        variant_map = {
            "eagle": "eagle1",
            "eagle1": "eagle1", 
            "eagle2": "eagle1",  # v2 uses same architecture
            "eagle3": "eagle3",  
            "hass": "hass", 
        }
        variant = variant_map[algorithm_lower]
        kwargs["variant"] = variant
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: eagle, eagle1, eagle2, eagle3, hass")
    
    # Run conversion
    return converter.convert(
        source_path=source_path,
        output_path=output_path,
        verifier_model=verifier_model,
        proposal_methods=proposal_methods,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        **kwargs
    )