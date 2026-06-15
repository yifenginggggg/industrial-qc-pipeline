from __future__ import annotations


def resolve_device(requested_device: str = "auto") -> str:
    import torch

    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
