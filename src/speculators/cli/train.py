"""Train command — thin typer wrapper around TrainConfig argument resolution."""

import typer


def train_command(ctx: typer.Context) -> None:
    """Train a speculator model.

    All arguments are forwarded to the training config resolver. Pass a YAML
    config file as a positional argument, or use --key=value flags.

    \b
    Examples:
        speculators train config.yaml
        speculators train --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct
        torchrun --nproc_per_node=4 -m speculators.train config.yaml
    """
    from speculators.train.cli import main  # noqa: PLC0415
    from speculators.train.config import TrainConfig  # noqa: PLC0415

    main(TrainConfig.resolve(ctx.args))
