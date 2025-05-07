"""
settings.py

This module provides configuration management for the application using Pydantic's
settings management capabilities. It defines the core settings structure,
including logging configurations, and allows settings to be loaded from environment
variables or a .env file. The settings are designed to be hierarchical and can be
customized through nested environment variables.

Core Interfaces:
- LoggingSettings: Defines logging-related configs such as log levels and file paths.
- Settings: The main settings class that aggregates all configurations and provides
    methods for generating .env files and reloading settings.
- reload_settings: A utility function to reload settings from the environment.
- print_config: A utility function to print the current configuration in .env format.

Example Usage:
```python
from speculators import settings

settings.logging.console_log_level = "DEBUG"
settings.logging.log_file = "app.log"
settings.logging.log_file_level = "INFO"
```

or utilizing environment variables:
```bash
export SPECULATORS__LOGGING__DISABLED=true
export SPECULATORS__LOGGING__CONSOLE_LOG_LEVEL=DEBUG
export SPECULATORS__LOGGING__LOG_FILE=app.log
export SPECULATORS__LOGGING__LOG_FILE_LEVEL=INFO
```
"""

import json
from collections.abc import Sequence
from typing import Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "LoggingSettings",
    "Settings",
    "print_config",
    "reload_settings",
    "settings",
]


class LoggingSettings(BaseModel):
    """
    Logging settings for the application
    """

    disabled: bool = Field(
        default=False,
        description=(
            "True to disable all logging, False (default) to enable logging. "
        ),
    )
    clear_loggers: bool = Field(
        default=True,
        description=(
            "True to clear all loggers which will remove all logging handlers that may "
            "have been added to the logger through other packages. False (default) to "
            "keep the existing loggers."
        ),
    )
    console_log_level: str = Field(
        default="WARNING",
        description=(
            "The log level for the console logger. This should be a valid log level "
            "string (e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL). "
            "DEBUG logs detailed information, useful for diagnosing problems. "
            "INFO logs general information about application execution. "
            "WARNING logs potential issues or important runtime events. "
            "ERROR logs errors that prevent some functionality from working. "
            "CRITICAL logs severe errors that may cause the application to terminate."
        ),
    )
    log_file: Optional[str] = Field(
        default=None,
        description=(
            "The path to the log file. If this is set, the logger will log to this file"
            " as well as to the console. If not set, the logger will only log to the "
            "console."
        ),
    )
    log_file_level: Optional[str] = Field(
        default=None,
        description=(
            "The log level for the file logger. This should be a valid log level string"
            " (e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL). If not set, the file logger"
            " will use the same log level as the console logger. "
            "DEBUG logs detailed information, useful for diagnosing problems. "
            "INFO logs general information about application execution. "
            "WARNING logs potential issues or important runtime events. "
            "ERROR logs errors that prevent some functionality from working. "
            "CRITICAL logs severe errors that may cause the application to terminate."
        ),
    )


class Settings(BaseSettings):
    """
    All the settings are powered by pydantic_settings and can be set through
    environment variables or .env file. The environment variables are prefixed with
    `SPECULATORS__` and nested properties are separated by `__`. For example, to set
    the `disabled` property of the `LoggingSettings` class, you can set the
    environment variable `SPECULATORS__LOGGING__DISABLED=true`. The same applies to
    all the other settings.
    """

    model_config = SettingsConfigDict(
        env_prefix="SPECULATORS__",
        env_nested_delimiter="__",
        extra="ignore",
        validate_default=True,
        env_file=".env",
    )

    # general settings
    logging: LoggingSettings = LoggingSettings()

    @model_validator(mode="after")
    @classmethod
    def set_default_source(cls, values):
        return values

    def generate_env_file(self) -> str:
        """
        Generate the .env file from the current settings
        """
        return Settings._recursive_generate_env(
            self,
            self.model_config["env_prefix"],  # type: ignore  # noqa: PGH003
            self.model_config["env_nested_delimiter"],  # type: ignore  # noqa: PGH003
        )

    @staticmethod
    def _recursive_generate_env(model: BaseModel, prefix: str, delimiter: str) -> str:
        env_file = ""
        add_models = []
        for key, value in model.model_dump().items():
            if isinstance(value, BaseModel):
                # add nested properties to be processed after the current level
                add_models.append((key, value))
                continue

            dict_values = (
                {
                    f"{prefix}{key.upper()}{delimiter}{sub_key.upper()}": sub_value
                    for sub_key, sub_value in value.items()
                }
                if isinstance(value, dict)
                else {f"{prefix}{key.upper()}": value}
            )

            for tag, sub_value in dict_values.items():
                if isinstance(sub_value, Sequence) and not isinstance(sub_value, str):
                    value_str = ",".join(f'"{item}"' for item in sub_value)
                    env_file += f"{tag}=[{value_str}]\n"
                elif isinstance(sub_value, dict):
                    value_str = json.dumps(sub_value)
                    env_file += f"{tag}={value_str}\n"
                elif not sub_value:
                    env_file += f"{tag}=\n"
                else:
                    env_file += f'{tag}="{sub_value}"\n'

        for key, value in add_models:
            env_file += Settings._recursive_generate_env(
                value, f"{prefix}{key.upper()}{delimiter}", delimiter
            )
        return env_file


settings = Settings()


def reload_settings():
    """
    Reload the settings from the environment variables
    """
    new_settings = Settings()
    settings.__dict__.update(new_settings.__dict__)


def print_config():
    """
    Print the current configuration settings
    """
    print(f"Settings: \n{settings.generate_env_file()}")  # noqa: T201
