import numpy as np
import torch
from torch import Tensor, device

from enums.image_formats import ImageFormat
from enums.layouts import Layout
from models.image import Image
from models.raw_metadata import RawMetadata


def load_raw(path: str, device: device) -> Image:
    """Load a RAW image from the given path and return an Image dataclass.

    Args:
        path (str): Path to the RAW image file.
        device (device): Target device to store the loaded image tensor.

    Returns:
        Image: Loaded image dataclass.
    """
    # TODO: Implement reading actual metadata from a sidecar file associated with the RAW image.
    # Currently using default placeholder values for width, height, bit depth, and Bayer pattern.
    width = 4096
    height = 2168
    bit_depth = 12
    bayer_pattern = Layout.RGGB

    raw_data: np.ndarray = np.fromfile(path, dtype=np.uint16).reshape((height, width))
    tensor: Tensor = (
        torch.from_numpy(raw_data).to(dtype=torch.float32, device=device).unsqueeze(0)
    )
    tensor.div_(2**bit_depth - 1)

    metadata: RawMetadata = RawMetadata(
        width=width,
        height=height,
        bit_depth=bit_depth,
        bayer_pattern=bayer_pattern,
    )

    return Image(
        tensor=tensor,
        original_tensor=tensor.clone(),
        path=path,
        name=path.split("/")[-1],
        image_format=ImageFormat.RAW,
        raw_metadata=metadata,
    )
