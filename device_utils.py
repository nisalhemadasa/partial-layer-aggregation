"""Shared device selection without importing simulation or model modules."""

import torch


_device = None


def configure_device(requested: str = 'auto') -> torch.device:
    """
    Select the device before constructing an experiment's models and optimizers.
    :param requested: Device choice: auto, cpu, or cuda.
    :return: Resolved PyTorch device.
    """
    global _device
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Select device='cpu' or 'auto'.")
    resolved = 'cuda' if requested == 'auto' and torch.cuda.is_available() else requested
    if resolved == 'auto':
        resolved = 'cpu'
    _device = torch.device(resolved)
    print(f"Device requested={requested}, resolved={_device}; PyTorch {torch.__version__}")
    return _device


def get_device() -> torch.device:
    """
    Return the configured device, resolving auto on first use if needed.
    :return: Shared PyTorch device.
    """
    if _device is None:
        return configure_device()
    return _device
