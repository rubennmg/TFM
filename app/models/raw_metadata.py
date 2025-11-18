from dataclasses import dataclass

from enums.layouts import Layout


@dataclass
class RawMetadata:
    width: int
    height: int
    bit_depth: int
    bayer_pattern: Layout
