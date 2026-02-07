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


def free_cuda_cache() -> None:
    """Free the CUDA cache if a GPU is available."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def get_device_info() -> list[str]:
    """Get detailed information about available CPU and GPU devices.

    Returns:
        list[str]: List of information lines about devices.
    """
    info_lines: list[str] = []

    info_lines.append(f"PyTorch Version: {torch.__version__}")
    info_lines.append(f"CUDA Available: {torch.cuda.is_available()}")
    info_lines.append("")

    if torch.cuda.is_available():
        info_lines.append("GPU Information")
        info_lines.append("=" * 50)
        info_lines.append(f"CUDA Version: {torch.version.cuda}")
        info_lines.append(f"cuDNN Version: {torch.backends.cudnn.version()}")
        info_lines.append(f"Number of GPUs: {torch.cuda.device_count()}")
        info_lines.append("")

        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                info_lines.append(f"GPU {i}: {props.name}")
                info_lines.append(f"  Compute Capability: {props.major}.{props.minor}")
                info_lines.append(
                    f"  Total Memory: {props.total_memory / (1024**3):.2f} GB"
                )

                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                info_lines.append(f"  Allocated Memory: {allocated:.2f} GB")
                info_lines.append(f"  Reserved Memory: {reserved:.2f} GB")
                info_lines.append("")
            except Exception as e:
                info_lines.append(f"GPU {i}: Error retrieving properties - {e}")
                info_lines.append("")
    else:
        info_lines.append("No GPU detected. Using CPU.")
        info_lines.append("")

    return info_lines
