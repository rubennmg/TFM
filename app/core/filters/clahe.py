from torch import Tensor

from core.image_operation import ImageOperation


class CLAHE(ImageOperation):
    """Class to apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor."""
        raise NotImplementedError("CLAHE operation is not yet implemented.")

    def apply(self, x: Tensor) -> Tensor:
        """Apply CLAHE to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Image tensor after applying CLAHE of shape (B, C, H, W).
        """

        raise NotImplementedError("CLAHE operation is not yet implemented.")
