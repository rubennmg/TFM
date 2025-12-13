import torch
from torch import Tensor
from torchvision.io import decode_image

from models.enums.image_formats import ImageFormat
from loaders.base_loader import ImageLoader
from models.image import Image
from models.metadata import Metadata
from models.enums.color_space import ColorSpace

SCALE: float = 1.0 / 255.0  # decode_image loads images as uint8 by default


class JpgLoader(ImageLoader):
    """Load JPG/JPEG images.

    Args:
        ImageLoader (abc): Base ImageLoader class.
    """

    @property
    def extensions(self) -> list[str]:
        return ["jpg", "jpeg"]

    @property
    def formats(self) -> list[ImageFormat]:
        return [ImageFormat.JPG]

    def load(self, path: str, device) -> Image:
        tensor: Tensor = decode_image(path).to(dtype=torch.float32).mul_(SCALE)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device=device)

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        return Image(
            tensor=tensor,
            original_tensor=tensor.clone(),
            operation_result_tensor=tensor.clone(),
            path=path,
            name=path.split("/")[-1],
            image_format=ImageFormat.JPG,
            color_space=ColorSpace.RGB,
            metadata=Metadata(
                width=tensor.shape[3],
                height=tensor.shape[2],
                bit_depth=8,
                bayer_pattern=None,
            ),
        )
