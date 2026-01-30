from torch import device

from loaders.arw_loader import ArwLoader
from loaders.base_loader import ImageLoader
from loaders.dng_loader import DngLoader
from loaders.jpg_loader import JpgLoader
from loaders.pkl_gz_loader import PklGzLoader
from loaders.raw_loader import RawLoader
from models.image import Image


def _build_extension_map(loaders: list[ImageLoader]) -> dict[str, ImageLoader]:
    ext_map: dict[str, ImageLoader] = {}
    for loader in loaders:
        for ext in loader.extensions:
            ext_map[ext.lower()] = loader
    return ext_map


_LOADERS: list[ImageLoader] = [
    RawLoader(),
    ArwLoader(),
    DngLoader(),
    JpgLoader(),
    PklGzLoader(),
]
_EXTENSION_MAP: dict[str, ImageLoader] = _build_extension_map(_LOADERS)


def load_image(path: str, device: device) -> Image:
    """Load an image from the given path and return an Image dataclass.
    Automatically selects the appropriate loader via file extension.

    Args:
        path (str): Path to the image file.
        device (TorchDevice): Target device to store the loaded image tensor.

    Raises:
        ValueError: If the image format/extension is not supported.

    Returns:
        Image: Loaded image dataclass.
    """
    fmt_str: str = path.split(".")[-1].lower()
    loader: ImageLoader | None = _EXTENSION_MAP.get(fmt_str)

    if loader is None:
        raise ValueError(f"Unsupported image format: {fmt_str}")

    return loader.load(path, device)


def get_supported_extensions() -> list[str]:
    """Get a list of all supported file extensions across all loaders.

    Returns:
        list[str]: List of supported file extensions.
    """
    return list(_EXTENSION_MAP.keys())
