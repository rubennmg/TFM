from torch import device

from enums.image_formats import ImageFormat
from loaders.base_loader import ImageLoader
from loaders.rawpy_loader import load_rawpy
from models.image import Image


class ArwLoader(ImageLoader):
    """Load ARW Images.

    Args:
        ImageLoader (abc): Base ImageLoader class.
    """

    @property
    def extensions(self) -> list[str]:
        return ["arw"]

    @property
    def formats(self) -> list[ImageFormat]:
        return [ImageFormat.ARW]

    def load(self, path: str, device: device) -> Image:
        return load_rawpy(path, device, ImageFormat.ARW)
