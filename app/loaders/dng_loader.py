from torch import device

from enums.image_formats import ImageFormat
from loaders.base_loader import ImageLoader
from loaders.rawpy_loader import load_rawpy
from models.image import Image


class DngLoader(ImageLoader):
    """Load DNG images.

    Args:
        ImageLoader (abc): Base ImageLoader class.
    """

    @property
    def extensions(self) -> list[str]:
        return ["dng"]

    @property
    def formats(self) -> list[ImageFormat]:
        return [ImageFormat.DNG]

    def load(self, path: str, device: device) -> Image:
        return load_rawpy(path, device, ImageFormat.DNG)
