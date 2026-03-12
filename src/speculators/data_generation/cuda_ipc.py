"""PyTorch CUDA IPC helpers for cross-process tensor transfer."""

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor


class CudaIpcExporter:
    """Exports torch CUDA tensors using PyTorch's native CUDA IPC reducer args."""

    def __init__(self):
        self._refs_by_token: dict[str, dict[str, list[torch.Tensor]]] = {}

    def reset(self) -> None:
        self._refs_by_token = {}

    def release(self, capture_token: str) -> None:
        self._refs_by_token.pop(capture_token, None)

    def export_capture(
        self, capture_token: str, request_tensors: dict[str, list[torch.Tensor]]
    ) -> dict:
        ipc_requests: dict[str, list[tuple]] = {}
        refs: dict[str, list[torch.Tensor]] = {}

        for req_id, layer_tensors in request_tensors.items():
            req_args: list[tuple] = []
            req_refs: list[torch.Tensor] = []
            for tensor in layer_tensors:
                args, ref = self._export_tensor(tensor)
                req_args.append(args)
                req_refs.append(ref)
            ipc_requests[req_id] = req_args
            refs[req_id] = req_refs

        self._refs_by_token[capture_token] = refs
        return {
            "transport": "torch_cuda_ipc",
            "capture_token": capture_token,
            "requests": ipc_requests,
        }

    def _export_tensor(self, tensor: torch.Tensor) -> tuple[tuple, torch.Tensor]:
        contiguous = tensor.contiguous()
        rebuild_fn, args = reduce_tensor(contiguous)
        if rebuild_fn is not rebuild_cuda_tensor:
            raise RuntimeError(
                f"Expected CUDA IPC reducer, got {getattr(rebuild_fn, '__name__', rebuild_fn)}"
            )
        return args, contiguous


class CudaIpcImporter:
    """Reconstructs torch CUDA tensors from PyTorch CUDA IPC reducer args."""

    @staticmethod
    def open_capture(payload: dict) -> tuple[dict[str, list[torch.Tensor]], set[int]]:
        requests = payload.get("requests")
        if not isinstance(requests, dict):
            raise RuntimeError("Invalid CUDA IPC payload: missing request metadata")

        request_tensors: dict[str, list[torch.Tensor]] = {}
        imported_devices: set[int] = set()

        for req_id, layer_args in requests.items():
            if not isinstance(layer_args, list):
                raise RuntimeError(
                    "Invalid CUDA IPC payload: layer reducer args must be a list"
                )
            layer_tensors: list[torch.Tensor] = []
            for args in layer_args:
                tensor = rebuild_cuda_tensor(*args)
                layer_tensors.append(tensor)
                if tensor.device.type == "cuda" and tensor.device.index is not None:
                    imported_devices.add(tensor.device.index)
            request_tensors[req_id] = layer_tensors

        return request_tensors, imported_devices
