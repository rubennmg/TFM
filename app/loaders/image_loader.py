from typing import Callable

from torch import device

from enums.image_formats import ImageFormat
from loaders.arw_loader import load_arw
from loaders.dng_loader import load_dng
from loaders.jpg_loader import load_jpg
from loaders.raw_loader import load_raw
from models.image import Image

LOADERS: dict[ImageFormat, Callable[[str, device], Image]] = {
    ImageFormat.RAW: load_raw,
    ImageFormat.ARW: load_arw,
    ImageFormat.DNG: load_dng,
    ImageFormat.JPG: load_jpg,
}

FORMATS_MAP: dict[str, ImageFormat] = {
    "raw": ImageFormat.RAW,
    "arw": ImageFormat.ARW,
    "dng": ImageFormat.DNG,
    "jpg": ImageFormat.JPG,
}


def load_image(path: str, device: device) -> Image:
    """Load an image from the given path and return an Image dataclass.
    Automatically selects the appropiate loader.

    Args:
        path (str): Path to the iamage file.
        device (device): Target device to store the loaded image tensor.

    Raises:
        ValueError: If the image format is not supported.

    Returns:
        Image: Loaded image dataclass.
    """
    fmt_str: str = path.split(".")[-1].lower()
    fmt: ImageFormat | None = FORMATS_MAP.get(fmt_str)

    if fmt is None or fmt not in LOADERS:
        raise ValueError(f"Unsupported image format: {fmt_str}")
    return LOADERS[fmt](path, device)
