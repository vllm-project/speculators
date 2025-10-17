"""Logging utilities for the Speculators training module.

This module provides a logging system for training machine learning models,
supporting multiple logging backends including TensorBoard (tensorboard), Weights & Biases (wandb).

Example Usage:
    ```python
    from speculators.train.logger import setup_metric_logger

    # Setup logging with TensorBoard and wandb
    setup_metric_logger(
        loggers=["tensorboard", "wandb"],
        run_name="my_training_run",
        output_dir="logs"
    )

    # Log metrics
    import logging
    logger = logging.getLogger("speculators.metrics")

    # Log a simple metric
    logger.info({"loss": 0.5, "accuracy": 0.95}, extra={"step": 100})

    # Log nested metrics
    logger.info({
        "training": {
            "loss": 0.5,
            "accuracy": 0.95
        },
        "validation": {
            "loss": 0.6,
            "accuracy": 0.92
        }
    }, extra={"step": 100})

    # Log hyperparameters
    logger.info({
        "learning_rate": 0.001,
        "batch_size": 32,
        "model": {
            "hidden_size": 512,
            "num_layers": 6
        }
    }, extra={"hparams": True})
    ```
"""

# SPDX-License-Identifier: Apache-2.0

# Standard
from collections.abc import Mapping
from datetime import datetime, timezone
from logging.config import dictConfig
import importlib
from pathlib import Path
from typing import Any, Union
import logging
import os
import warnings


# Third Party
from rich.logging import RichHandler
import torch

### Helper functions

LogDict = Mapping[str, Union[str, int, float, "LogDict"]]


def _substitute_placeholders(
    run_name: str | None, default_template: str = "{time}"
) -> str:
    """Replace placeholders in the run name with actual values.

    This function supports dynamic run name generation by replacing placeholders
    with actual values from the environment or current time. This is particularly
    useful for distributed training scenarios where you want unique run names
    for each process.

    Supported placeholders:
        - {time}: Current local timestamp in ISO format
        - {utc_time}: Current UTC timestamp in ISO format
        - {rank}: Process rank from RANK environment variable
        - {local_rank}: Local process rank from LOCAL_RANK environment variable

    Args:
        run_name: String containing placeholders to be replaced. If None, uses default_template
        default_template: Default template to use if run_name is None

    Returns:
        String with all placeholders replaced by their values

    Example:
        ```python
        # With default template
        name = _substitute_placeholders(None)
        # Result: "2024-03-14T10:30:00_rank0"

        # With custom template
        name = _substitute_placeholders("experiment_{time}_rank{rank}")
        # Result: "experiment_2024-03-14T10:30:00_rank0"
        ```
    """
    if run_name is None:
        run_name = default_template

    substitutions = {
        "{time}": datetime.now().isoformat(timespec="seconds"),
        "{utc_time}": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "{rank}": os.environ.get("RANK", 0),
        "{local_rank}": os.environ.get("LOCAL_RANK", 0),
    }
    for placeholder_pat, value in substitutions.items():
        run_name = run_name.replace(placeholder_pat, str(value))

    return run_name


def _flatten_dict(log_dict: LogDict, sep: str = "/", prefix: str = "") -> dict:
    """Flatten a nested dictionary into a single-level dictionary.

    This function recursively traverses a nested dictionary and creates a new
    dictionary with keys that represent the path to each value in the original
    dictionary.

    Args:
        d: The dictionary to flatten
        sep: Separator to use between nested keys
        prefix: Prefix to add to all keys

    Returns:
        A flattened dictionary with keys joined by the separator
    """
    flattened = {}

    for k, v in log_dict.items():
        if isinstance(v, Mapping):
            flattened |= _flatten_dict(v, sep=sep, prefix=f"{prefix}{k}{sep}")
        else:
            flattened[prefix + k] = v

    return flattened


### Filters
class IsMappingFilter(logging.Filter):
    """Filter that only allows log records with dictionary messages.

    This filter ensures that only log records containing dictionary messages
    are processed by the handler. This is useful for metric logging where
    we want to ensure all logged messages are structured data.
    """

    def filter(self, record):
        """Check if the log record's message is a dictionary.

        Args:
            record: The log record to check

        Returns:
            bool: True if the message is a dictionary, False otherwise
        """
        return isinstance(record.msg, Mapping)


class IsRank0Filter(logging.Filter):
    """Filter that only allows log records from rank 0 in distributed training.

    This filter is useful in distributed training scenarios where you want to
    ensure that only the main process (rank 0) logs metrics to avoid duplicate
    logging. The rank can be determined from various sources in order of precedence:
    1. Explicitly provided rank value
    2. Record's rank attribute
    3. Record's message dictionary
    4. Environment variables
    5. PyTorch distributed rank

    Args:
        rank_val: Optional explicit rank value to use
        local_rank: If True, use local_rank instead of global rank
    """

    def __init__(self, rank_val: int | None = None, local_rank: bool = False):
        self.rank_val = rank_val
        if local_rank:
            self.rank_attr = "local_rank"
        else:
            self.rank_attr = "rank"

    def _get_rank(self, record):
        rank = (
            self.rank_val
            or getattr(record, self.rank_attr, None)
            or (isinstance(record.msg, Mapping) and record.msg.get(self.rank_attr))
            or os.environ.get(self.rank_attr.upper(), None)
            or (
                self.rank_attr == "rank"
                and torch.distributed.is_initialized()
                and torch.distributed.get_rank()
            )
            or 0
        )

        return int(rank)

    def filter(self, record):
        return self._get_rank(record) == 0


class FormatDictFilter(logging.Filter):
    """Reformats dictionary messages for prettier printing.

    This filter processes dictionary messages to create a more readable string
    representation. It handles different types of values appropriately:
    - Floats are formatted with 3 decimal places or scientific notation
    - Integers are formatted as decimal numbers
    - Other types are converted to their string representation

    Note: This is not a true filter, but a processing step as described in the
    Python logging cookbook: https://docs.python.org/3/howto/logging-cookbook.html#using-filters-to-impart-contextual-information
    """

    @staticmethod
    def _format_value(v):
        if isinstance(v, float):
            if abs(v) < 0.001 or abs(v) > 999:
                return f"{v:.2e}"
            return f"{v:.3f}"
        elif isinstance(v, int):
            return f"{v:d}"
        else:
            return repr(v)

    def filter(self, record):
        if not isinstance(record.msg, Mapping):
            return True
        flat_dict = _flatten_dict(record.msg)

        record.msg = ", ".join(
            f"{k}={self._format_value(v)}" for k, v in flat_dict.items()
        )

        return True


### Handlers
class TensorBoardHandler(logging.Handler):
    """Logger that writes metrics to TensorBoard.

    This handler expects a (nested) dictionary of metrics or text to be logged with string keys.
    A step can be specified by passing `extra={"step": <step>}` to the logging method.
    To log hyperparameters, pass a (nested) mapping of hyperparameters to the logging method
    and set `extra={"hparams": True}`.
    """

    def __init__(
        self,
        level: int = logging.INFO,
        run_name: str | None = None,
        log_dir: str | os.PathLike = "logs",
        **tboard_init_kwargs: Any,
    ):
        """Initialize the TensorBoard logger and check for required dependencies.

        Args:
            level: The logging level for this handler
            run_name: Name of the run, can contain placeholders
            log_dir: Directory where TensorBoard logs should be stored
        """
        super().__init__(level)

        self.tboard_init_kwargs = tboard_init_kwargs.copy()
        self.tboard_init_kwargs.setdefault(
            "log_dir", Path(log_dir) / _substitute_placeholders(run_name)
        )

        self._tboard_writer = None

    def _setup(self):
        """Create the TensorBoard log directory and initialize the writer.

        Raises:
            RuntimeError: If tensorboard package is not installed
        """

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            msg = (
                "Could not initialize TensorBoardHandler because package tensorboard could not be imported.\n"
                "Please ensure it is installed by running 'pip install tensorboard' or configure the logger to use a different backend."
            )
            raise RuntimeError(msg) from e

        os.makedirs(self.tboard_init_kwargs["log_dir"], exist_ok=True)
        self._tboard_writer = SummaryWriter(**self.tboard_init_kwargs)

    def emit(self, record: logging.LogRecord):
        """Emit a log record to TensorBoard.

        This method handles both scalar metrics and text logs, automatically
        detecting the type of data being logged.

        Args:
            record: The log record to emit
        """
        if self._tboard_writer is None:
            self._setup()

        if not isinstance(record.msg, Mapping):
            warnings.warn(
                f"TensorBoardHandler expected a mapping, got {type(record.msg)}. Skipping log. Please ensure the handler is configured correctly to filter out non-mapping objects."
            )
            return

        flat_dict = _flatten_dict(record.msg)
        step = getattr(record, "step", None)
        if getattr(record, "hparams", None):
            self._tboard_writer.add_hparams(
                flat_dict, {}, run_name=".", global_step=step
            )
            return

        for k, v in flat_dict.items():
            try:
                # Check that `v` can be converted to float
                float(v)
            except ValueError:
                # Occurs for strings that cannot be converted to floats (e.g. "3.2.3") and aren't "inf" or "nan"
                self._tboard_writer.add_text(k, v, global_step=step)
            except TypeError:
                warnings.warn(
                    f"TensorBoardHandler expected a scalar or text, got {type(v)}. Skipping log. Please ensure metric logger is only called with mappings containing scalar values or text."
                )
            else:
                self._tboard_writer.add_scalar(k, v, global_step=step)

    def flush(self):
        """Flush the TensorBoard writer."""
        if self._tboard_writer is not None:
            self._tboard_writer.flush()

    def close(self):
        """Close the TensorBoard writer and cleanup resources."""
        if self._tboard_writer is not None:
            self._tboard_writer.close()
            self._tboard_writer = None
        super().close()


class WandbHandler(logging.Handler):
    """Logger that sends metrics to Weights & Biases (wandb).

    This handler expects a (nested) dictionary of metrics or text to be logged with string keys.
    A step can be specified by passing `extra={"step": <step>}` to the logging method.
    To log hyperparameters, pass a (nested) mapping of hyperparameters to the logging method
    and set `extra={"hparams": True}`.
    """

    def __init__(
        self,
        level: int = logging.INFO,
        run_name: str | None = None,
        log_dir: str | os.PathLike = "logs",
        **init_kwargs: Any,
    ):
        """Initialize the wandb logger and check for required dependencies.

        Args:
            level: The logging level for this handler
            run_name: Name of the run, can contain placeholders
            log_dir: Directory where wandb logs should be stored
        """
        super().__init__(level)

        self.init_kwargs = init_kwargs.copy()
        self.init_kwargs.setdefault("dir", Path(log_dir))
        self.init_kwargs.setdefault("name", _substitute_placeholders(run_name))
        self.init_kwargs.setdefault("config", {})

        self._package_name = "wandb"
        self._run = None

    def _setup(self):
        try:
            wandb = importlib.import_module(self._package_name)
        except ImportError as e:
            msg = (
                f"Could not initialize {self.__class__.__name__} because package {self._package_name} could not be imported.\n"
                f"Please ensure it is installed by running 'pip install {self._package_name}' or configure the logger to use a different backend."
            )
            raise RuntimeError(msg) from e

        self._run = wandb.init(**self.init_kwargs)

    def emit(self, record: logging.LogRecord):
        if self._run is None:
            self._setup()

        if not isinstance(record.msg, Mapping):
            warnings.warn(
                f"{self.__class__.__name__} expected a mapping, got {type(record.msg)}. Skipping log. Please ensure the handler is configured correctly to filter out non-mapping objects."
            )
            return

        flat_dict = _flatten_dict(record.msg)
        step = getattr(record, "step", None)
        if getattr(record, "hparams", None):
            for k, v in flat_dict.items():
                self._run.config[k] = v
            return

        self._run.log(flat_dict, step=step)


class TrackioHandler(WandbHandler):
    """Logger that sends metrics to Trackio.

    This handler expects a (nested) dictionary of metrics or text to be logged with string keys.
    A step can be specified by passing `extra={"step": <step>}` to the logging method.
    To log hyperparameters, pass a (nested) mapping of hyperparameters to the logging method
    and set `extra={"hparams": True}`.
    """

    def __init__(
        self,
        level: int = logging.INFO,
        run_name: str | None = None,
        log_dir: str | os.PathLike = "logs",
        **init_kwargs: Any,
    ):
        """Initialize the trackio logger and check for required dependencies.

        Args:
            level: The logging level for this handler
            run_name: Name of the run, can contain placeholders
        """
        super().__init__(level)

        self.init_kwargs = init_kwargs.copy()
        self.init_kwargs.setdefault("name", _substitute_placeholders(run_name))
        self.init_kwargs.setdefault("config", {})
        self.init_kwargs.setdefault("project", "speculators")

        # Trackio doesn't support the dir keyword argument so we ignore log_dir

        self._package_name = "trackio"
        self._run = None


### Main functions


def setup_root_logger(level="INFO"):
    """Configure the root logger with rich formatting.

    This function sets up the root logger with a RichHandler for
    console output and adds the FormatDictFilter for better dictionary message
    formatting.
    """
    handler = RichHandler()
    handler.addFilter(FormatDictFilter())
    handler.addFilter(IsRank0Filter(local_rank=True))
    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]", handlers=[handler]
    )


def setup_metric_logger(loggers, run_name, output_dir):
    """Configure the metric logging system with specified backends.

    This function sets up a comprehensive logging configuration that supports
    multiple logging backends simultaneously. It configures filters, handlers,
    and loggers for structured metric logging.

    Args:
        loggers: A string or list of strings specifying which logging backends to use.
                Supported values: "tensorboard", "wandb", "trackio"
        run_name: Name for the current training run. Can include placeholders like
                 {time}, {rank}, {utc_time}, {local_rank}.
        output_dir: Directory where log files will be stored

    Example:
        ```python
        # Setup logging with multiple backends
        setup_metric_logger(
            loggers=["tensorboard", "wandb", "trackio"],
            run_name="experiment_{time}",
            output_dir="logs"
        )

        # Setup logging with a single backend
        setup_metric_logger(
            loggers="tensorboard",
            run_name="my_run",
            output_dir="logs"
        )
        ```
    """
    if isinstance(loggers, str):
        loggers = loggers.split(",")
    loggers = [logger.strip() for logger in loggers]

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "is_mapping": {
                "()": IsMappingFilter,
            },
            "is_rank0": {
                "()": IsRank0Filter,
            },
        },
        "handlers": {
            "tensorboard": {
                "()": TensorBoardHandler,
                "log_dir": output_dir,
                "run_name": run_name,
                "filters": ["is_mapping", "is_rank0"],
            },
            "wandb": {
                "()": WandbHandler,
                "log_dir": output_dir,
                "run_name": run_name,
                "filters": ["is_mapping", "is_rank0"],
            },
            "trackio": {
                "()": TrackioHandler,
                "log_dir": output_dir,
                "run_name": run_name,
                "filters": ["is_mapping", "is_rank0"],
            },
        },
        "loggers": {
            "speculators.metrics": {
                "handlers": loggers,
                "filters": ["is_mapping"],
                "level": "INFO",
                "propagate": True,
            },
            "speculators": {
                "level": "INFO",
                "propagate": True,
            },
        },
    }
    dictConfig(logging_config)
