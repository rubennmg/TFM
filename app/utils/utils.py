from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, device


def get_device() -> device:
    """Get the available device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_stylesheet(file_path: str) -> str:
    """Read and return the contents of a QSS file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Could not load stylesheet: {e}")
        return ""


def tensor_to_uint8_np(tensor: Tensor) -> np.ndarray:
    """Convert a torch Tensor in (C,H,W) or (H,W,C) [0..1] to uint8 HxWxC."""
    t: Tensor = tensor
    if t.ndim == 4:
        t = t.squeeze(0)
    if t.ndim == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    elif t.ndim != 3:
        raise ValueError("Tensor shape must be (C,H,W) or (H,W,C)")

    t = (t * 255).round()
    arr: np.ndarray = t.detach().cpu().numpy()
    arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
    return arr
