import torch
from transformers.utils import is_torch_npu_available


def is_npu_available() -> bool:
    """Detect Ascend NPU availability"""
    try:
        return is_torch_npu_available()
    except (ImportError, RuntimeError, ModuleNotFoundError, AttributeError):
        return False


def get_device_name(idx: int) -> str:
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        return "NO ACCELERATOR"

    mod = torch.get_device_module(acc)
    if hasattr(mod, "get_device_name"):
        return mod.get_device_name(idx)
    else:
        return f"{str(acc).upper()} DEVICE"


def mem_get_info():
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        return (0, 0)

    mod = torch.get_device_module(acc)
    if hasattr(mod, "mem_get_info"):
        return mod.mem_get_info()
    else:
        return (0, 0)


def empty_cache():
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        return

    mod = torch.get_device_module(acc)
    if hasattr(mod, "empty_cache"):
        mod.empty_cache()
    return
