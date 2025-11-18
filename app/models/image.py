from dataclasses import dataclass

from torch import Tensor

from enums.image_formats import ImageFormat
from models.raw_metadata import RawMetadata


@dataclass
class Image:
    tensor: Tensor
    original_tensor: Tensor
    path: str
    name: str
    image_format: ImageFormat
    raw_metadata: RawMetadata | None = None
