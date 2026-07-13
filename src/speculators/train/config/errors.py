"""Configuration errors.

A single exception type for every *user-facing* configuration failure the pure
core (:meth:`TrainConfig.from_sources`) can raise -- an unknown ``--set`` key, a
draft-init conflict, and so on. Keeping it exception-based (never
``SystemExit``/``parser.error``) is what lets the core be unit-tested without
``sys.argv`` and without catching ``SystemExit``; the impure boundary
(:meth:`TrainConfig.resolve`) is the only place that turns a :class:`ConfigError`
into a clean ``SystemExit(2)``.
"""


class ConfigError(ValueError):
    """A user-facing configuration error (bad ``--set`` key, draft-init conflict).

    Subclasses ``ValueError`` so the ``resolve`` boundary's existing
    ``except (OSError, yaml.YAMLError, ValueError)`` net also catches it, and so
    callers that only know about ``ValueError`` still behave sensibly.
    """
