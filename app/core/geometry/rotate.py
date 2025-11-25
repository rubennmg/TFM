from torch import Tensor

from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class Rotate(ImageOperation):
    """Class to rotate image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor."""
        raise NotImplementedError("Rotation operation is not yet implemented.")

    def apply(self, x: Tensor) -> Tensor:
        """Apply the rotation to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Rotated image tensor of shape (B, C, H, W).
        """

        raise NotImplementedError("Rotation operation is not yet implemented.")
