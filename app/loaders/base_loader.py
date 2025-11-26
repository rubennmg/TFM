from abc import ABC, abstractmethod

from typing import Sequence
from torch import device

from enums.image_formats import ImageFormat
from models.image import Image


class ImageLoader(ABC):
    """Abstract base class for image loaders.

    Concrete loaders should declare the file extensions they support
    and the corresponding image formats, and implement the load method
    returning a populated `Image` dataclass.
    """

    @property
    @abstractmethod
    def extensions(self) -> Sequence[str]:
        """List of lowercase file extensions supported by this loader (e.g., ["jpg"])."""

    @property
    @abstractmethod
    def formats(self) -> Sequence[ImageFormat]:
        """List of ImageFormat values handled by this loader."""

    @abstractmethod
    def load(self, path: str, device: device) -> Image:
        """Load an image given a path and target device, returning an `Image`."""
