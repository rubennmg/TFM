from torch import Tensor

from core.image_operation import ImageOperation


class GrayToColor(ImageOperation):
    """Class to convert grayscale image tensors to color.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor."""
        raise NotImplementedError(
            "Grayscale to Color conversion is not yet implemented."
        )

    def apply(self, x: Tensor) -> Tensor:
        """Apply the grayscale to color conversion to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, 1, H, W) in grayscale format.

        Returns:
            Tensor: Converted image tensor of shape (B, 3, H, W) in color format.
        """
        raise NotImplementedError(
            "Grayscale to Color conversion is not implemented yet."
        )
