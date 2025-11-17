from dataclasses import dataclass

from torch import Tensor


@dataclass
class ImageData:
    tensor: Tensor
    original_tensor: Tensor | None = None
    path: str | None = None
    name: str | None = None
    bit_depth: int = 12  # assuming 12-bit for RAW images
    is_raw: bool = False
