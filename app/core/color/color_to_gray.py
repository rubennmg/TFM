from torch import Tensor

from core.image_operation import ImageOperation


class ColorToGray(ImageOperation):
    """Class to convert color image tensors to grayscale.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor."""
        raise NotImplementedError(
            "Color to Grayscale conversion is not yet implemented."
        )

    def apply(self, x: Tensor) -> Tensor:
        """Apply the color to grayscale conversion to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, 3, H, W) in color format.

        Returns:
            Tensor: Converted image tensor of shape (B, 1, H, W) in grayscale format.
        """
        raise NotImplementedError(
            "Color to Grayscale conversion is not implemented yet."
        )
