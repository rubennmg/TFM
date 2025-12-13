import numpy as np
import rawpy
import torch
from torch import Tensor, device

from models.enums.image_formats import ImageFormat
from models.image import Image
from models.metadata import Metadata
from models.enums.color_space import ColorSpace

SCALE: float = 1.0 / 65535.0


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
        raw_data: np.ndarray = raw.postprocess(output_bps=16)  # always H,W,C, uint16

    tensor: Tensor = torch.from_numpy(raw_data).to(dtype=torch.float32).mul_(SCALE)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()  # B,C,H,W
    tensor = tensor.to(device=device)

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    return Image(
        tensor=tensor,
        original_tensor=tensor.clone(),
        operation_result_tensor=tensor.clone(),
        path=path,
        name=path.split("/")[-1],
        image_format=fmt,
        color_space=ColorSpace.RGB,
        metadata=Metadata(
            width=tensor.shape[3],
            height=tensor.shape[2],
            bit_depth=16,
            bayer_pattern=None,
        ),
    )
