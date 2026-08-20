"""Abstraction for hidden-states transfer between vLLM and the trainer."""

from __future__ import annotations

import dataclasses
import fcntl
import os
import shutil
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from safetensors.torch import load as load_safetensors_bytes
from safetensors.torch import load_file

from hs_connectors.mooncake_store import MooncakeHiddenStatesStore, MooncakeStoreConfig

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable


def wait_for_lock(lock_path: str, timeout: float = 10.0, poll_interval: float = 0.02):
    fd = os.open(lock_path, os.O_RDWR)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for lock: {lock_path}"
                    ) from None
                time.sleep(poll_interval)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    os.remove(lock_path)


class HiddenStatesTransfer(ABC):
    """Interface for reading hidden states produced by vLLM."""

    def setup(self) -> None:  # noqa: B027
        """Lazy initialization (safe to call from dataloader worker)."""

    @abstractmethod
    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        """Return a previously cached sample, or ``None``."""

    @abstractmethod
    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        """Retrieve a freshly generated sample by its vLLM-returned handle."""

    def cache(self, handle: str, file_idx: int) -> None:  # noqa: B027
        """Persist a generated sample to the cache location."""

    def delete(self, handle: str) -> None:  # noqa: B027
        """Clean up a generated sample (e.g. delete a temp file)."""


class HiddenStatesBackend(ABC):
    """Plugin interface for hidden-states transfer backends.

    Each backend registers itself via ``@HiddenStatesBackend.register(name)``
    and implements these four static hooks so that scripts (``train.py``,
    ``launch_vllm.py``) can discover and configure backends without hardcoding.
    """

    registry: ClassVar[dict[str, type[HiddenStatesBackend]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
    ) -> Callable[[type[HiddenStatesBackend]], type[HiddenStatesBackend]]:
        def decorator(
            subclass: type[HiddenStatesBackend],
        ) -> type[HiddenStatesBackend]:
            if name in cls.registry:
                raise ValueError(f"Backend '{name}' is already registered.")
            cls.registry[name] = subclass
            return subclass

        return decorator

    @staticmethod
    @abstractmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to ``train.py``."""
        ...

    @staticmethod
    @abstractmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to ``launch_vllm.py``."""
        ...

    @staticmethod
    @abstractmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> HiddenStatesTransfer:
        """Construct a :class:`HiddenStatesTransfer` from parsed train args."""
        ...

    @staticmethod
    @abstractmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        """Construct the ``kv_transfer_config`` dict for ``vllm serve``."""
        ...


# ---------------------------------------------------------------------------
# File-based backend (shared filesystem)
# ---------------------------------------------------------------------------


def _load_hs_file(file_path: Path) -> dict[str, torch.Tensor] | None:
    lock_path = str(file_path) + ".lock"
    if Path(lock_path).exists():
        wait_for_lock(lock_path)

    if file_path.exists():
        return load_file(file_path)

    return None


class FileTransfer(HiddenStatesTransfer):
    """File-system based hidden-states transfer (shared filesystem)."""

    def __init__(self, hidden_states_path: Path):
        self.hidden_states_path = hidden_states_path

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        return _load_hs_file(path)

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        return _load_hs_file(Path(handle))

    def cache(self, handle: str, file_idx: int) -> None:
        self.hidden_states_path.mkdir(parents=True, exist_ok=True)
        target = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        shutil.move(handle, target)

    def delete(self, handle: str) -> None:
        Path(handle).unlink()


@HiddenStatesBackend.register("file")
class FileBackend(HiddenStatesBackend):
    """Shared-filesystem backend using safetensors files."""

    @staticmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--hidden-states-path",
            type=str,
            default=None,
            help=(
                "The path where cached hidden states files are stored. (Default: "
                "args.data_path / 'hidden_states')"
            ),
        )

    @staticmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--hidden-states-path",
            type=str,
            default="/tmp/hidden_states",  # noqa: S108
            help="The directory to save hidden states to. Default '/tmp/hidden_states'",
        )

    @staticmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> FileTransfer:
        hs_path = (
            Path(args.hidden_states_path)
            if args.hidden_states_path
            else Path(data_path) / "hidden_states"
        )
        return FileTransfer(hs_path)

    @staticmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        return {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": args.hidden_states_path,
            },
        }


# ---------------------------------------------------------------------------
# Mooncake-based backend (distributed store)
# ---------------------------------------------------------------------------


class MooncakeTransfer(HiddenStatesTransfer):
    """Mooncake distributed store based hidden-states transfer."""

    def __init__(self, store: MooncakeHiddenStatesStore):
        self.store = store

    def setup(self) -> None:
        if not self.store.is_setup:
            self.store.setup()

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:  # noqa: ARG002
        return None

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        return self.store.get_sample(handle)

    def delete(self, handle: str) -> None:
        self.store.delete_sample(handle)


@HiddenStatesBackend.register("mooncake")
class MooncakeBackend(HiddenStatesBackend):
    """Mooncake distributed store backend (no shared filesystem required)."""

    @staticmethod
    def _add_mooncake_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--mooncake-master",
            type=str,
            default="127.0.0.1:50051",
            help="Mooncake master server address. Used with backend=mooncake.",
        )
        parser.add_argument(
            "--mooncake-metadata-server",
            type=str,
            default="P2PHANDSHAKE",
            help=(
                "Mooncake metadata server (or P2PHANDSHAKE). "
                "Used with backend=mooncake."
            ),
        )
        parser.add_argument(
            "--mooncake-protocol",
            choices=["tcp", "rdma"],
            default="tcp",
            help="Mooncake transport protocol. Used with backend=mooncake.",
        )
        parser.add_argument(
            "--mooncake-global-segment-gib",
            type=float,
            default=4.0,
            help=(
                "Memory registered by each Mooncake client for globally visible "
                "objects, in GiB. Increase for many concurrent long sequences."
            ),
        )
        parser.add_argument(
            "--mooncake-local-buffer-gib",
            type=float,
            default=2.0,
            help="Mooncake client's local staging buffer, in GiB.",
        )

    @staticmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        MooncakeBackend._add_mooncake_args(parser)

    @staticmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        MooncakeBackend._add_mooncake_args(parser)
        parser.add_argument(
            "--mooncake-writer-threads",
            type=int,
            default=4,
            help="Number of asynchronous Mooncake writer threads in the vLLM client.",
        )

    @staticmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,  # noqa: ARG004
    ) -> MooncakeTransfer:
        local_hostname = os.environ.get(
            "MOONCAKE_LOCAL_HOSTNAME"
        ) or socket.gethostbyname(socket.gethostname())

        store = MooncakeHiddenStatesStore(
            MooncakeStoreConfig(
                local_hostname=local_hostname,
                metadata_server=args.mooncake_metadata_server,
                master_server_address=args.mooncake_master,
                global_segment_size=round(args.mooncake_global_segment_gib * 1024**3),
                local_buffer_size=round(args.mooncake_local_buffer_gib * 1024**3),
                protocol=args.mooncake_protocol,
            )
        )
        return MooncakeTransfer(store)

    @staticmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        local_hostname = os.environ.get(
            "MOONCAKE_LOCAL_HOSTNAME"
        ) or socket.gethostbyname(socket.gethostname())

        mooncake_cfg = MooncakeStoreConfig(
            local_hostname=local_hostname,
            metadata_server=args.mooncake_metadata_server,
            master_server_address=args.mooncake_master,
            global_segment_size=round(args.mooncake_global_segment_gib * 1024**3),
            local_buffer_size=round(args.mooncake_local_buffer_gib * 1024**3),
            protocol=args.mooncake_protocol,
            num_writer_threads=args.mooncake_writer_threads,
        )

        return {
            "kv_connector": "MooncakeHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_module_path": (
                "hs_connectors.mooncake_hidden_states_connector"
            ),
            "kv_connector_extra_config": {
                "mooncake": dataclasses.asdict(mooncake_cfg),
            },
        }


# ---------------------------------------------------------------------------
# HTTP backend (vLLM writes to local fast disk; trainer fetches over HTTP)
#
# Motivation: pointing ``shared_storage_path`` at a shared network filesystem
# (e.g. cephfs) makes every connector write a cross-host op that contends on
# the MDS; under saturation, batches of writes stall together with 1-3 s tail
# latency, which punches through the trainer's prefetch buffer and produces
# tail training steps. This backend instead tells vLLM to write to a local
# disk on its own node (no MDS, no cross-host write bursts) and has the
# trainer pull the file back over plain HTTP from a tiny static file server
# (``scripts/serve_hs.py``) running next to vLLM. It also works without any
# shared filesystem between the two nodes.
#
# The connector side is unchanged: it still writes
# ``{shared_storage_path}/{req_id}.safetensors`` under a ``.lock`` flock and
# returns that absolute path as the handle. We only reinterpret the handle on
# the trainer side: strip the basename, prepend ``--hs-http-base``, GET it.
# The file server blocks on the same ``.lock`` flock until the write is done,
# preserving the existing synchronization semantics.
# ---------------------------------------------------------------------------


class HttpTransfer(HiddenStatesTransfer):
    def __init__(
        self,
        hs_http_base: str,
        hidden_states_path: Path,
        timeout: float = 120.0,
    ):
        self.hs_http_base = hs_http_base.rstrip("/")
        self.hidden_states_path = hidden_states_path
        self.timeout = timeout

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        return _load_hs_file(path)

    def _url_for(self, handle: str) -> str:
        return f"{self.hs_http_base}/{os.path.basename(handle)}"

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        url = self._url_for(handle)
        try:
            req = urllib.request.Request(url, method="GET")  # noqa: S310
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                payload = resp.read()
        except urllib.error.HTTPError as e:
            if e.code == HTTPStatus.NOT_FOUND:
                return None
            raise
        if not payload:
            return None
        return load_safetensors_bytes(payload)

    def cache(self, handle: str, file_idx: int) -> None:
        raise NotImplementedError(
            "HttpTransfer.cache() is unsupported: the hidden-states file lives"
            " on the vLLM node. Use --hidden-states-backend file for"
            " on_generate=cache, or keep on_generate=delete with the http"
            " backend."
        )

    def delete(self, handle: str) -> None:
        url = self._url_for(handle)
        try:
            req = urllib.request.Request(url, method="DELETE")  # noqa: S310
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                resp.read()
        except OSError:
            pass  # best-effort cleanup; the server TTL-sweeps leftovers


@HiddenStatesBackend.register("http")
class HttpBackend(HiddenStatesBackend):
    @staticmethod
    def add_train_args(parser):
        parser.add_argument(
            "--hs-http-base",
            type=str,
            default=None,
            help="Base URL of the hidden-states file server, e.g. 'http://10.0.0.1:9010'.",
        )
        parser.add_argument(
            "--hs-http-timeout",
            type=float,
            default=120.0,
            help="Per-request timeout (seconds) for HTTP hidden-states fetches.",
        )

    @staticmethod
    def add_launch_args(parser):
        pass

    @staticmethod
    def from_train_args(args, data_path):
        if not getattr(args, "hs_http_base", None):
            raise ValueError(
                "--hs-http-base is required when --hidden-states-backend=http"
            )
        hs_path = (
            Path(args.hidden_states_path)
            if args.hidden_states_path
            else Path(data_path) / "hidden_states"
        )
        return HttpTransfer(
            hs_http_base=args.hs_http_base,
            hidden_states_path=hs_path,
            timeout=getattr(args, "hs_http_timeout", 120.0),
        )

    @staticmethod
    def build_kv_transfer_config(args):
        return {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": args.hidden_states_path,
            },
        }
