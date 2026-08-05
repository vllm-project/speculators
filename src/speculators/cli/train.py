"""Train command — thin typer wrapper around TrainConfig argument resolution."""

import typer


def train_command(ctx: typer.Context) -> None:
    """Train a speculator model.

    All arguments are forwarded to the training config resolver
    (use ``--help`` to see every available flag). Pass ``--config``
    with a YAML file, or use ``--key=value`` flags directly.

    \b
    Examples:
        speculators train --config config.yaml
        speculators train --verifier-name-or-path meta-llama/... \\
            --data-path ./output --speculator-type eagle3
        torchrun --nproc_per_node=4 -m speculators.train \\
            --config config.yaml
    """
    from speculators.train.cli import main  # noqa: PLC0415
    from speculators.train.config import TrainConfig  # noqa: PLC0415

    main(TrainConfig.resolve(ctx.args))
