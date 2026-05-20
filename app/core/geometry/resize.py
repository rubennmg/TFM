from torch import Tensor

from core.image_operation import ImageOperation
from core.registry import register_operation
from kornia.geometry import transform as Kg_t


@register_operation
class Resize(ImageOperation):
    """Class to resize image tensors.
    Uses Kornia's Resize implementation.

    See: https://kornia.readthedocs.io/en/stable/geometry.transform.html#kornia.geometry.transform.resize

    Args:
        ImageOperation (ImageOperation): Base class for image operations.

    Raises:
        TypeError: If size is not a tuple of two integers.
        ValueError: If size values are not positive integers.
    """

    def __init__(self, size: tuple[int, int]):
        """Class constructor.
        Args:
            size (tuple[int, int]): Desired output size as (height, width).
        """
        if not (
            isinstance(size, tuple)
            and len(size) == 2
            and all(isinstance(s, int) for s in size)
        ):
            raise TypeError(f"Size must be a tuple of two integers, got {size}")
        if not all(s > 0 for s in size):
            raise ValueError("Size values must be positive integers")

        self.size = size

    def apply(self, x: Tensor) -> Tensor:
        """Apply the resizing to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (C, H, W).

        Returns:
            Tensor: Resized image tensor of shape (C, new_H, new_W).
        """
        return Kg_t.resize(x, size=self.size)
