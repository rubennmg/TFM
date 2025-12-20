import numpy as np
import torch
from torch import Tensor, device

from core._tensor_utils import CHANNEL_DIM


def get_device() -> device:
    """Get the available device (GPU if available, else CPU).

    Returns:
        device: The available device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_uint8_np(tensor: Tensor) -> np.ndarray:
    """Convert a torch Tensor to uint8 numpy array (H, W, C) format for display.

    Args:
        tensor (Tensor): Input tensor in (C,H,W) or (H,W,C) format.

    Raises:
        ValueError: If the tensor shape is not compatible.

    Returns:
        np.ndarray: uint8 numpy array in (H, W, C) format.
    """
    t: Tensor = tensor.detach()

    if t.ndim == 3 and t.shape[CHANNEL_DIM] in (1, 3):
        # CHW -> HWC
        t = t.permute(1, 2, 0)
    elif t.ndim != 3:
        raise ValueError(f"Tensor shape must be (C,H,W) or (H,W,C), got {t.shape}")

    if t.dtype not in (torch.uint8, torch.float32, torch.float64):
        raise TypeError(f"Expected uint8 or float tensor, got {t.dtype}")

    if t.dtype == torch.uint8:
        t_u8 = t
    else:
        t_clamped = t.clamp(0.0, 1.0)
        t_u8 = torch.mul(t_clamped, 255.0).round().to(torch.uint8)

    cpu_t = t_u8.contiguous().to("cpu")

    return cpu_t.numpy()


def compute_histogram_bins_torch(
    tensor: Tensor, bins: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-channel histogram using torch on the tensor's device.

    Args:
        tensor (Tensor): Input tensor in (C,H,W) or (H,W,C) format.
        bins (int, optional): Number of histogram bins. Defaults to 256.

    Raises:
        ValueError: If the tensor shape is not compatible.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Histograms for R, G, B channels.
    """
    t: Tensor = tensor

    if t.ndim == 3 and t.shape[CHANNEL_DIM] in (1, 3):
        # CHW -> HWC
        t = t.permute(1, 2, 0)
    elif t.ndim != 3:
        raise ValueError("Tensor shape must be (C,H,W) or (H,W,C)")

    if t.dtype != torch.float32:
        raise TypeError(f"Expected float32 tensor, got {t.dtype}")

    dev = t.device
    img: Tensor = t.to(device=dev, dtype=torch.float32)

    img = img.clamp(0.0, 1.0)

    channels = img.shape[2]

    counts_per_channel: list[Tensor] = []
    for c in range(min(3, channels)):
        ch: Tensor = img[:, :, c].flatten()
        counts, _ = torch.histogram(ch, bins=bins, range=(0.0, 1.0))
        counts_per_channel.append(counts)

    if channels == 1:
        counts_per_channel = [counts_per_channel[0]] * 3
    else:
        while len(counts_per_channel) < 3:
            counts_per_channel.append(counts_per_channel[-1])

    stacked = torch.stack(counts_per_channel, dim=0).to("cpu")
    stacked = stacked.to(torch.int64)
    return stacked[0].numpy(), stacked[1].numpy(), stacked[2].numpy()


def free_cuda_cache() -> None:
    """Free the CUDA cache if a GPU is available."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
