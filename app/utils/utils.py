import numpy as np
import torch
from torch import Tensor, device


def get_device() -> device:
    """Get the available device (GPU if available, else CPU).

    Returns:
        device: The available device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_stylesheet(file_path: str) -> str:
    """Read and return the contents of a QSS file.

    Args:
        file_path (str): Path to the QSS file.

    Returns:
        str: Contents of the QSS file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Could not load stylesheet: {e}")
        return ""


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

    if t.ndim == 4:
        t = t.squeeze(0)

    if t.ndim == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    elif t.ndim != 3:
        raise ValueError(f"Tensor shape must be (C,H,W) or (H,W,C), got {t.shape}")

    if t.dtype != torch.uint8:
        t = (t * 255).clamp(0, 255).round().to(torch.uint8)
    else:
        if t.min().item() < 0 or t.max().item() > 255:
            t = t.clamp(0, 255)

    if not t.is_contiguous():
        t = t.contiguous()

    cpu_t: Tensor = t.to("cpu", non_blocking=True)
    if not cpu_t.is_contiguous():
        cpu_t = cpu_t.contiguous()

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
    if t.ndim == 4:
        t = t.squeeze(0)
    if t.ndim == 3 and t.shape[0] in (1, 3):
        # CHW -> HWC
        t = t.permute(1, 2, 0)
    elif t.ndim != 3:
        raise ValueError("Tensor shape must be (C,H,W) or (H,W,C)")

    dev = t.device
    dtype = t.dtype if t.is_floating_point() else torch.float32
    img: Tensor = t.to(device=dev, dtype=dtype)

    img = img.clamp(0, 1)

    hists: list[np.ndarray] = []
    channels = img.shape[2]
    if channels not in (1, 3):
        channels = 1 if img.shape[2] == 1 else 3

    for c in range(min(3, img.shape[2])):
        ch: Tensor = img[:, :, c]
        counts: Tensor
        counts, _ = torch.histogram(ch, bins=bins, range=(0.0, 1.0))
        hists.append(counts.to("cpu").to(torch.int64).numpy())

    if img.shape[2] == 1:
        h = hists[0]
        return h, h, h

    while len(hists) < 3:
        hists.append(hists[-1])
    return hists[0], hists[1], hists[2]
