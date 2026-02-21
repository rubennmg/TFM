from PIL import Image
import numpy as np
from pathlib import Path
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


def save_image_as_jpg(tensor: Tensor, file_path: str) -> None:
    """Save a torch Tensor as a JPG image file.

    Args:
        tensor (Tensor): Input tensor in (C,H,W) or (H,W,C) format.
        file_path (str): Path to save the JPG image.
    """
    path_obj = Path(file_path)
    if path_obj.suffix.lower() not in [".jpg", ".jpeg"]:
        path_obj = path_obj.with_suffix(".jpg")
    file_path = str(path_obj)

    np_image = tensor_to_uint8_np(tensor)

    if np_image.ndim == 2:
        image = Image.fromarray(np_image, mode="L")
    elif np_image.ndim == 3:
        if np_image.shape[2] == 1:
            image = Image.fromarray(np_image[:, :, 0], mode="L")
        elif np_image.shape[2] == 3:
            image = Image.fromarray(np_image, mode="RGB")
        elif np_image.shape[2] == 4:
            image = Image.fromarray(np_image[:, :, :3], mode="RGB")
        else:
            raise ValueError(f"Unsupported number of channels: {np_image.shape[2]}")
    else:
        raise ValueError(f"Unsupported array shape: {np_image.shape}")

    image.save(file_path, format="JPEG", quality=95)


def format_param_value(value) -> str:
    """Format a parameter value for display, handling tensors specially.

    Args:
        value: The parameter value to format.

    Returns:
        str: A concise string representation of the value.
    """
    if isinstance(value, Tensor):
        if value.ndim == 2:
            H, W = value.shape
            return f"Tensor[{H}×{W}]"
        elif value.ndim == 3:
            C, H, W = value.shape
            return f"Tensor[{C}×{H}×{W}]"
        else:
            shape_str = "×".join(str(d) for d in value.shape)
            return f"Tensor[{shape_str}]"
    elif isinstance(value, (int, float)):
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)
    elif isinstance(value, str):
        return value
    elif value is None:
        return "None"
    else:
        return str(value)


def format_params_dict(params: dict) -> str:
    """Format a dictionary of parameters for display.

    Args:
        params (dict): Dictionary of parameter names and values.

    Returns:
        str: A formatted string of parameters.
    """
    if not params:
        return "default"

    formatted = [f"{k}={format_param_value(v)}" for k, v in params.items()]
    return ", ".join(formatted)
