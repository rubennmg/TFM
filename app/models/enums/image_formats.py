from enum import Enum, auto


class ImageFormat(Enum):
    """Possible image file formats for loaded images."""

    RAW = auto()
    ARW = auto()
    DNG = auto()
    JPG = auto()
    PKL_GZ = auto()
