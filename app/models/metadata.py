from dataclasses import dataclass

from models.enums.layouts import Layout


@dataclass
class Metadata:
    width: int
    height: int
    bit_depth: int | None
    bayer_pattern: Layout | None
