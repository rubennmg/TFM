import numpy as np
import rawpy
import torch
from torch import Tensor, device

from enums.image_formats import ImageFormat
from models.image import Image


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
        raw_data: np.ndarray = raw.postprocess(output_bps=16)

    tensor: Tensor = torch.from_numpy(raw_data).to(dtype=torch.float32)
    tensor.div_(65535.0)
    tensor = tensor.to(device=device)

    return Image(
        tensor=tensor,
        original_tensor=tensor.clone(),
        path=path,
        name=path.split("/")[-1],
        image_format=fmt,
        raw_metadata=None,
    )
