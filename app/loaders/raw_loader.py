import numpy as np
import torch
from torch import Tensor, device

from enums.image_formats import ImageFormat
from enums.layouts import Layout
from models.image import Image
from models.metadata import Metadata

# TODO: Implement reading actual metadata from a sidecar file associated with the RAW image.
# Currently using default placeholder values for width, height, bit depth, and Bayer pattern.
METADATA: Metadata = Metadata(
    width=4096,
    height=2168,
    bit_depth=12,
    bayer_pattern=Layout.RGGB,
)

SCALE: float = 1.0 / (2**METADATA.bit_depth - 1) if METADATA.bit_depth else 1.0


def load_raw(path: str, device: device) -> Image:
    """Load a RAW image from the given path and return an Image dataclass.

    Args:
        path (str): Path to the RAW image file.
        device (device): Target device to store the loaded image tensor.

    Returns:
        Image: Loaded image dataclass.
    """
    raw_data: np.ndarray = np.fromfile(path, dtype=np.uint16).reshape(
        (METADATA.height, METADATA.width)
    )
    tensor: Tensor = (
        torch.from_numpy(raw_data).to(dtype=torch.float32).unsqueeze(0).mul_(SCALE)
    )
    tensor = tensor.to(device=device, non_blocking=True)

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    return Image(
        tensor=tensor,
        original_tensor=tensor.clone(),
        path=path,
        name=path.split("/")[-1],
        image_format=ImageFormat.RAW,
        metadata=METADATA,
    )
