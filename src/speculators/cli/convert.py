"""CLI command for model conversion."""

import json
from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated

from speculators import convert


def convert_command(
    source: Annotated[
        Path,
        typer.Argument(
            help="Path to the source model directory",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Argument(
            help="Path to save the converted model",
            resolve_path=True,
        ),
    ],
    verifier: Annotated[
        str,
        typer.Option(
            "--verifier",
            "-v",
            help="Name or path of the verifier model (e.g., meta-llama/Llama-3.1-8B)",
        ),
    ],
    algorithm: Annotated[
        str,
        typer.Option(
            "--algorithm",
            "-a",
            help="Algorithm to convert (eagle, eagle1, eagle2, eagle3, hass)",
        ),
    ] = "eagle",
    proposal_methods: Annotated[
        Optional[str],
        typer.Option(
            "--proposal-methods",
            "-p",
            help="JSON string of proposal methods (e.g., '[{\"proposal_type\": \"greedy\", \"draft_tokens\": 5}]')",
        ),
    ] = None,
    push_to_hub: Annotated[
        bool,
        typer.Option(
            "--push-to-hub",
            help="Push the converted model to HuggingFace Hub",
        ),
    ] = False,
    repo_id: Annotated[
        Optional[str],
        typer.Option(
            "--repo-id",
            help="Repository ID for pushing to HuggingFace Hub (required if --push-to-hub)",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite output directory if it exists",
        ),
    ] = False,
) -> None:
    """
    Convert a model to Speculators format.
    
    Examples:
        # Convert EAGLE model
        speculators convert /path/to/eagle /path/to/output \\
            --verifier meta-llama/Llama-3.1-8B-Instruct \\
            --algorithm eagle
        
        # Convert HASS model with custom proposal methods
        speculators convert /path/to/hass /path/to/output \\
            --verifier meta-llama/Llama-3.1-70B \\
            --algorithm hass \\
            --proposal-methods '[{"proposal_type": "greedy", "draft_tokens": 10}]'
    """
    # Validate inputs
    if not source.exists():
        typer.echo(f"Error: Source path does not exist: {source}", err=True)
        raise typer.Exit(1)
    
    if output.exists() and not force:
        typer.echo(
            f"Error: Output path already exists: {output}\n"
            f"Use --force to overwrite.",
            err=True,
        )
        raise typer.Exit(1)
    
    if push_to_hub and not repo_id:
        typer.echo(
            "Error: --repo-id is required when using --push-to-hub",
            err=True,
        )
        raise typer.Exit(1)
    
    # Parse proposal methods if provided
    parsed_proposal_methods = None
    if proposal_methods:
        try:
            parsed_proposal_methods = json.loads(proposal_methods)
            if not isinstance(parsed_proposal_methods, list):
                raise ValueError("Proposal methods must be a list")
        except (json.JSONDecodeError, ValueError) as e:
            typer.echo(
                f"Error: Invalid proposal methods JSON: {e}",
                err=True,
            )
            raise typer.Exit(1)
    
    # Run conversion
    typer.echo(f"Converting {algorithm} model...")
    typer.echo(f"  Source: {source}")
    typer.echo(f"  Output: {output}")
    typer.echo(f"  Verifier: {verifier}")
    
    try:
        result = convert(
            source_path=str(source),
            output_path=str(output),
            verifier_model=verifier,
            algorithm=algorithm,
            proposal_methods=parsed_proposal_methods,
            push_to_hub=push_to_hub,
            repo_id=repo_id,
        )
        
        # Display results
        if result.success:
            typer.echo("\n✓ Conversion completed successfully!")
            
            if result.warnings:
                typer.echo("\nWarnings:")
                for warning in result.warnings:
                    typer.echo(f"  ⚠ {warning}")
            
            if result.unmapped_weights:
                typer.echo(f"\nUnmapped weights ({len(result.unmapped_weights)}):")
                for weight in result.unmapped_weights[:5]:  # Show first 5
                    typer.echo(f"  - {weight}")
                if len(result.unmapped_weights) > 5:
                    typer.echo(f"  ... and {len(result.unmapped_weights) - 5} more")
            
            typer.echo(f"\nConverted model saved to: {output}")
            
            if push_to_hub:
                typer.echo(f"Model pushed to: https://huggingface.co/{repo_id}")
        else:
            typer.echo("\n✗ Conversion failed!", err=True)
            
            if result.errors:
                typer.echo("\nErrors:", err=True)
                for error in result.errors:
                    typer.echo(f"  ✗ {error}", err=True)
            
            raise typer.Exit(1)
            
    except Exception as e:
        typer.echo(f"\n✗ Unexpected error: {e}", err=True)
        raise typer.Exit(1)