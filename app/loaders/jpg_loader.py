import torch
from torch import Tensor, device
from torchvision.io import decode_image

from enums.image_formats import ImageFormat
from models.image import Image
from models.metadata import Metadata

SCALE: float = 1.0 / 255.0  # decode_image loads images as uint8 by default


def load_jpg(path: str, device: device) -> Image:
    """Load a JPG image from the given path and return an Image dataclass.

    Args:
        path (str): Path to the JPG image file.
        device (device): The device on which the image tensor will be loaded.

    Returns:
        Image: An Image dataclass containing the loaded image tensor and metadata.
    """
    tensor: Tensor = decode_image(path).to(dtype=torch.float32).mul_(SCALE)
    tensor = tensor.to(device=device, non_blocking=True)

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    return Image(
        tensor=tensor,
        original_tensor=tensor.clone(),
        path=path,
        name=path.split("/")[-1],
        image_format=ImageFormat.JPG,
        metadata=Metadata(
            width=tensor.shape[2],
            height=tensor.shape[1],
            bit_depth=8,
            bayer_pattern=None,
        ),
    )
