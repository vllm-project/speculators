from hs_connectors.transfer import (
    FileBackend,
    FileTransfer,
    HiddenStatesBackend,
    HiddenStatesTransfer,
    HttpBackend,
    HttpTransfer,
    MooncakeBackend,
    MooncakeTransfer,
)

__all__ = [
    "FileBackend",
    "FileTransfer",
    "HiddenStatesBackend",
    "HiddenStatesTransfer",
    "HttpBackend",
    "HttpTransfer",
    "MooncakeBackend",
    "MooncakeTransfer",
]

try:
    from hs_connectors.version import version as __version__
except ImportError:
    __version__ = "unknown"
