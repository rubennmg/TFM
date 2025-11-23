from dataclasses import dataclass

from enums.layouts import Layout
from torch import dtype, Size


@dataclass
class Metadata:
    width: int
    height: int
    channels: int
    shape: Size
    dtype: dtype
    bit_depth: int | None
    bayer_pattern: Layout | None
