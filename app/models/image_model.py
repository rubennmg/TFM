from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ImageData:
    tensor: torch.Tensor
    path: Optional[str] = None
    name: Optional[str] = None
    bit_depth: int = 12  # assuming 12-bit for RAW images
    is_raw: bool = False
