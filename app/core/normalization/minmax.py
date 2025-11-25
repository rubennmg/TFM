from torch import Tensor

from core.image_operation import ImageOperation


class MinMaxNormalization(ImageOperation):
    """Class to apply Min-Max Normalization to image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor."""
        raise NotImplementedError("Min-Max Normalization is not yet implemented.")

    def apply(self, x: Tensor) -> Tensor:
        """Apply Min-Max Normalization to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Normalized image tensor of shape (B, C, H, W).
        """
        raise NotImplementedError("Min-Max Normalization is not implemented yet.")
