from torch import Tensor

from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class HsvToRgb(ImageOperation):
    """Class to convert HSV image tensors to RGB color space.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor."""
        raise NotImplementedError("HSV to RGB conversion is not yet implemented.")

    def apply(self, x: Tensor) -> Tensor:
        """Apply the HSV to RGB conversion to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W) in HSV format.

        Returns:
            Tensor: Converted image tensor of shape (B, C, H, W) in RGB format.
        """
        raise NotImplementedError("HSV to RGB conversion is not implemented yet.")
