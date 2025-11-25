from torch import Tensor

from core.image_operation import ImageOperation


class UnsharpMasking(ImageOperation):
    """Class to apply Unsharp Masking to image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor."""
        raise NotImplementedError("Unsharp Masking is not yet implemented.")

    def apply(self, x: Tensor) -> Tensor:
        """Apply Unsharp Masking to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Filtered image tensor of shape (B, C, H, W).
        """
        raise NotImplementedError("Unsharp Masking is not implemented yet.")
