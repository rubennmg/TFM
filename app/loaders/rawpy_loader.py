import math
from typing import Final

import numpy as np
import rawpy
import torch
from torch import Tensor, device

from core._tensor_utils import CHANNEL_DIM, HEIGHT_DIM, WIDTH_DIM
from models.enums.color_space import ColorSpace
from models.enums.image_formats import ImageFormat
from models.enums.layouts import Layout
from models.image import Image
from models.metadata import Metadata

_COLOR_TO_INDEX: Final[dict[str, int]] = {"R": 0, "G": 1, "B": 2}


def load_rawpy(path: str, device: device, fmt: ImageFormat) -> Image:
    """Load a RAW image using rawpy from the given path and return an Image dataclass.

    Args:
        path (str): Path to the RAW image file.
        device (device): Device on which the tensor should be allocated.
        fmt (ImageFormat): Format of the image being loaded.

    Returns:
        Image: Loaded image encapsulated in an Image dataclass.
    """
    with rawpy.imread(path) as raw:
        raw_data: np.ndarray = raw.raw_image_visible.copy()
        white_level = _extract_white_level(raw)
        black_level = _extract_black_level(raw)
        bit_depth = _bit_depth_from_levels(white_level, raw_data)
        normalized_data = _normalize_raw_array(raw_data, black_level, white_level)
        bayer_pattern = _infer_bayer_pattern(raw)

    tensor: Tensor = torch.from_numpy(normalized_data)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)  # C,H,W
    else:
        if tensor.shape[2] == 4:
            tensor = tensor[:, :, :3]  # discard alpha channel

        tensor = tensor.permute(2, 0, 1).contiguous()  # C,H,W

    tensor = tensor.to(device=device)

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    color_space = (
        ColorSpace.GRAYSCALE if tensor.shape[CHANNEL_DIM] == 1 else ColorSpace.RGB
    )

    return Image(
        tensor=tensor,
        original_tensor=tensor.clone(),
        operation_result_tensor=tensor.clone(),
        path=path,
        name=path.split("/")[-1],
        image_format=fmt,
        color_space=color_space,
        metadata=Metadata(
            width=tensor.shape[WIDTH_DIM],
            height=tensor.shape[HEIGHT_DIM],
            bit_depth=bit_depth,
            bayer_pattern=bayer_pattern,
        ),
    )


def _extract_white_level(raw: rawpy.RawPy) -> float | None:  # type: ignore
    white_level = getattr(raw, "white_level", None)
    if white_level is None:
        channel_levels = getattr(raw, "camera_white_level_per_channel", None)
        if channel_levels is not None and len(channel_levels) > 0:
            white_level = float(np.max(channel_levels))
    return float(white_level) if white_level is not None else None


def _extract_black_level(raw: rawpy.RawPy) -> float | None:  # type: ignore
    channel_levels = getattr(raw, "black_level_per_channel", None)
    if channel_levels is not None and len(channel_levels) > 0:
        return float(np.mean(channel_levels))
    black_level = getattr(raw, "black_level", None)
    return float(black_level) if black_level is not None else None


def _bit_depth_from_levels(
    white_level: float | None, raw_data: np.ndarray
) -> int | None:
    if white_level is not None and white_level > 0:
        return int(math.ceil(math.log2(white_level + 1)))
    if np.issubdtype(raw_data.dtype, np.integer):
        return raw_data.dtype.itemsize * 8
    return None


def _normalize_raw_array(
    raw_data: np.ndarray, black_level: float | None, white_level: float | None
) -> np.ndarray:
    data = raw_data.astype(np.float32, copy=False)
    black = black_level if black_level is not None else 0.0
    if white_level is not None:
        max_value = white_level
    elif np.issubdtype(raw_data.dtype, np.integer):
        max_value = float(np.iinfo(raw_data.dtype).max)
    else:
        max_value = 1.0
    denom = max(1.0, max_value - black)
    np.subtract(data, black, out=data)
    np.clip(data, 0.0, denom, out=data)
    data /= denom
    return data


def _infer_bayer_pattern(raw: rawpy.RawPy) -> Layout | None:  # type: ignore
    pattern = getattr(raw, "raw_pattern", None)
    if pattern is None:
        return None

    pattern_array = np.asarray(pattern)
    if pattern_array.size != 4:
        return None

    color_desc = getattr(raw, "color_desc", "") or ""
    if isinstance(color_desc, bytes):
        color_desc = color_desc.decode("ascii", errors="ignore")

    index_map: dict[int, int] = {}
    for idx, char in enumerate(color_desc):
        canonical = _COLOR_TO_INDEX.get(char.upper())
        if canonical is not None:
            index_map[idx] = canonical

    canonical_pattern: list[int] = []
    for value in pattern_array.flatten():
        canonical_value = index_map.get(int(value))
        if canonical_value is None:
            return None
        canonical_pattern.append(canonical_value)

    canonical_tuple = tuple(canonical_pattern)
    for layout in Layout:
        if layout.value == canonical_tuple:
            return layout

    return None
