from dataclasses import dataclass

from enums.layouts import Layout


@dataclass
class Metadata:
    width: int
    height: int
    bit_depth: int | None
    bayer_pattern: Layout | None
