from torch import Tensor

from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class RealToRGB8(ImageOperation):
    """Class to convert real-valued image tensors to RGB8 format.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor."""
        raise NotImplementedError("Real to RGB8 conversion is not yet implemented.")

    def apply(self, x: Tensor) -> Tensor:
        """Convert real-valued image tensor to RGB8 format.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Image tensor in RGB8 format of shape (B, C, H, W).
        """

        raise NotImplementedError("Real to RGB8 conversion is not yet implemented.")
