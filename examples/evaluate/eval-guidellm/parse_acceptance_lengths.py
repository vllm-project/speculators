#!/usr/bin/env python3
"""
Parse vLLM server logs to extract speculative decoding acceptance rates.

This script:
1. Parses vLLM logs for SpecDecoding metrics
2. Calculates weighted per-position acceptance rates
3. Calculates conditional acceptance probabilities
"""

import argparse
import numpy as np


def parse_log(log_file):
    """
    Extract SpecDecoding metrics from vLLM server log.

    Args:
        log_file: Path to vLLM log file

    Returns:
        tokens: Array of drafted token counts per sample
        acceptances: List of acceptance rate arrays per sample
    """
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Split log by SpecDecoding metric sections
    sections = text.split("SpecDecoding metrics:")
    if len(sections) <= 1:
        raise ValueError("No SpecDecoding metrics found in log")

    tokens = []
    acceptances = []

    # Parse each section
    for section in sections[1:]:
        first_line = section.split("\n")[0]

        # Extract: "Drafted: X tokens"
        drafted = int(first_line.split("Drafted: ")[1].split()[0])
        tokens.append(drafted)

        # Extract: "Per-position acceptance rate: 0.9, 0.8, 0.7, ..."
        acceptance_str = first_line.split("Per-position acceptance rate: ")[1]
        acceptance_values = [float(x.strip(",")) for x in acceptance_str.split()]
        acceptances.append(acceptance_values)

    return np.array(tokens), acceptances


def calculate_metrics(tokens, acceptances):
    """
    Calculate weighted and conditional acceptance rates.

    Args:
        tokens: Array of drafted token counts
        acceptances: List of acceptance rate arrays

    Returns:
        weighted: Weighted average acceptance rate per position
        conditional: P(accept position i | accepted position i-1)
    """
    # Pad all acceptance arrays to same length (zero-pad shorter ones)
    max_length = max(len(acc) for acc in acceptances)
    padded_acceptances = [np.pad(acc, (0, max_length - len(acc))) for acc in acceptances]
    acceptance_array = np.array(padded_acceptances)

    # Calculate weighted average: sum(acceptance * tokens) / sum(tokens)
    weighted = np.sum(acceptance_array * tokens[:, None], axis=0) / np.sum(tokens)

    # Calculate conditional rates: P(i|i-1) = P(i) / P(i-1)
    # Prepend 1.0 for position 0 (always accepted)
    weighted_with_start = np.concatenate(([1.0], weighted))
    conditional = weighted_with_start[1:] / weighted_with_start[:-1]

    return weighted, conditional


def format_results(tokens, weighted, conditional):
    """
    Format results as human-readable text.

    Args:
        tokens: Array of drafted token counts
        weighted: Weighted acceptance rates
        conditional: Conditional acceptance rates

    Returns:
        Formatted string with analysis results
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Speculative Decoding Acceptance Analysis")
    lines.append("=" * 70)
    lines.append(f"\nTotal samples: {len(tokens)}")
    lines.append(f"Total drafted tokens: {np.sum(tokens)}")
    lines.append(f"Average drafted tokens: {np.mean(tokens):.2f}")
    lines.append(f"\nWeighted per-position acceptance rates:")
    lines.append(str(np.round(weighted, decimals=3)))
    lines.append(f"\nConditional acceptance rates:")
    lines.append(str(np.round(conditional, decimals=3)))
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Parse vLLM logs for speculative decoding acceptance rates"
    )
    parser.add_argument("log_file", help="Path to vLLM server log file")
    parser.add_argument("-o", "--output", help="Save results to file")
    args = parser.parse_args()

    # Parse log file
    tokens, acceptances = parse_log(args.log_file)

    # Calculate metrics
    weighted, conditional = calculate_metrics(tokens, acceptances)

    # Format and display results
    results = format_results(tokens, weighted, conditional)
    print(results)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(results)


if __name__ == "__main__":
    main()
