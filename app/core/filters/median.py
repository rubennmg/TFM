from torch import Tensor

from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class MedianFilter(ImageOperation):
    """Class to apply Median Filter to image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor."""
        raise NotImplementedError("Median Filter is not yet implemented.")

    def apply(self, x: Tensor) -> Tensor:
        """Apply the Median Filter to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Filtered image tensor of shape (B, C, H, W).
        """
        raise NotImplementedError("Median Filter is not implemented yet.")
