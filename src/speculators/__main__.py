"""CLI entrypoint for ``python -m speculators``."""

from speculators.cli import app

__all__ = ["app"]

if __name__ == "__main__":
    app()
