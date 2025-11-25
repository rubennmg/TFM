from torch import Tensor

from core.image_operation import ImageOperation


class HistogramEqualization(ImageOperation):
    """Class to apply Histogram Equalization to image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor."""
        raise NotImplementedError(
            "Histogram Equalization operation is not yet implemented."
        )

    def apply(self, x: Tensor) -> Tensor:
        """Apply Histogram Equalization to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Image tensor after applying Histogram Equalization of shape (B, C, H, W).
        """

        raise NotImplementedError(
            "Histogram Equalization operation is not yet implemented."
        )
