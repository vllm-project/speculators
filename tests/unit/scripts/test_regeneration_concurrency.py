"""Native CLI parameter contract; no endpoint or model is needed."""

from unittest.mock import AsyncMock, patch

import pytest
import typer
from typer.testing import CliRunner

from speculators.cli.regenerate_responses import regenerate_responses


@pytest.mark.parametrize("concurrency", [0, -1])
def test_reject_nonpositive_concurrency(concurrency):
    app = typer.Typer()
    app.command()(regenerate_responses)
    with patch(
        "speculators.cli.regenerate_responses._run", new_callable=AsyncMock
    ) as run:
        result = CliRunner().invoke(app, ["--concurrency", str(concurrency)])
    assert result.exit_code == 2, result.output
    assert "--concurrency must be >= 1" in result.output
    run.assert_not_called()


@pytest.mark.parametrize("concurrency", [1, 64])
def test_accept_positive_concurrency(concurrency):
    app = typer.Typer()
    app.command()(regenerate_responses)
    with patch(
        "speculators.cli.regenerate_responses._run", new_callable=AsyncMock
    ) as run:
        result = CliRunner().invoke(app, ["--concurrency", str(concurrency)])
    assert result.exit_code == 0, result.output
    assert run.await_args is not None
    assert run.await_args.kwargs["concurrency"] == concurrency


@pytest.mark.parametrize("concurrency", [0, -1])
def test_nonpositive_concurrency_does_not_silently_drop_local_rows(
    tmp_path, concurrency
):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"id": "sample", "prompt": "Hello"}\n')
    output = tmp_path / "output.jsonl"
    app = typer.Typer()
    app.command()(regenerate_responses)
    # Tokenizer decoding is optional output enrichment. Keep the actual dataset
    # loader, queue, worker construction and orchestration in this regression.
    with patch(
        "speculators.cli.regenerate_responses.build_detokenizer", return_value=None
    ):
        result = CliRunner().invoke(
            app,
            [
                "--dataset",
                str(dataset),
                "--model",
                "test-model",
                "--outfile",
                str(output),
                "--concurrency",
                str(concurrency),
            ],
        )
    assert result.exit_code == 2, result.output
    assert "--concurrency must be >= 1" in result.output
    assert not output.exists()
