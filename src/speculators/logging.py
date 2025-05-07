import sys

from loguru import logger

from speculators.settings import LoggingSettings, settings

__all__ = ["configure_logger", "logger"]


def configure_logger(config: LoggingSettings = settings.logging):
    """
    Configure the logger for the speculators library using the provided configuration.
    This function sets up the logger to log to the console and/or a file based on the
    provided configuration.

    Note: Environment variables take precedence over the function parameters.

    :param config: The configuration for the logger to use.
    :type config: LoggerConfig
    """

    if config.disabled:
        logger.disable("speculators")
        return

    logger.enable("speculators")

    if config.clear_loggers:
        logger.remove()

    # log as a human readable string with the time, function, level, and message
    logger.add(
        sys.stdout,
        level=config.console_log_level.upper(),
        format="{time} | {function} | {level} - {message}",
    )

    if config.log_file or config.log_file_level:
        log_file = config.log_file or "speculators.log"
        log_file_level = config.log_file_level or "INFO"
        # log as json to the file for easier parsing
        logger.add(log_file, level=log_file_level.upper(), serialize=True)


# invoke logger setup on import with default values
# enabling console logging with INFO and disabling file logging
configure_logger()
