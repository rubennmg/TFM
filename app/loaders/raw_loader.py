import numpy as np
import torch
from torch import Tensor

from models.enums.image_formats import ImageFormat
from models.enums.layouts import Layout
from loaders.base_loader import ImageLoader
from models.image import Image
from models.metadata import Metadata
from models.enums.color_space import ColorSpace

# TODO: Implement reading actual metadata from a sidecar file associated with the RAW image.
# Currently using default placeholder values for width, height, bit depth, and Bayer pattern.
METADATA: Metadata = Metadata(
    width=4096,
    height=2168,
    bit_depth=12,
    bayer_pattern=Layout.RGGB,
)

SCALE: float = 1.0 / (2**METADATA.bit_depth - 1) if METADATA.bit_depth else 1.0


class RawLoader(ImageLoader):
    """Load RAW images from binary files.

    Args:
        ImageLoader (abc): Base ImageLoader class.
    """

    @property
    def extensions(self) -> list[str]:
        return ["raw"]

    @property
    def formats(self) -> list[ImageFormat]:
        return [ImageFormat.RAW]

    def load(self, path: str, device) -> Image:
        raw_data: np.ndarray = np.fromfile(path, dtype=np.uint16).reshape(
            (METADATA.height, METADATA.width)
        )
        tensor: Tensor = (
            torch.from_numpy(raw_data).to(dtype=torch.float32).unsqueeze(0).mul_(SCALE)
        )
        tensor = tensor.unsqueeze(0)  # BxCxHxW
        tensor = tensor.to(device=device)

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        return Image(
            tensor=tensor,
            original_tensor=tensor.clone(),
            operation_result_tensor=tensor.clone(),
            path=path,
            name=path.split("/")[-1],
            image_format=ImageFormat.RAW,
            color_space=ColorSpace.GRAYSCALE,
            metadata=METADATA,
        )
