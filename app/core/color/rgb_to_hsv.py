from torch import Tensor

from core.image_operation import ImageOperation


class RgbToHsv(ImageOperation):
    """Class to convert RGB image tensors to HSV color space.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor."""
        raise NotImplementedError("RGB to HSV conversion is not yet implemented.")

    def apply(self, x: Tensor) -> Tensor:
        """Apply the RGB to HSV conversion to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W) in RGB format.

        Returns:
            Tensor: Converted image tensor of shape (B, C, H, W) in HSV format.
        """
        raise NotImplementedError("RGB to HSV conversion is not implemented yet.")
