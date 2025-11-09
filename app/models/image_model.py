import torch

from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageData:
    tensor: torch.Tensor
    path: Optional[str] = None
    name: Optional[str] = None
    bit_depth: int = 16
    is_raw: bool = False
