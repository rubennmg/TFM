from torch import Tensor

from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class GaussianFilter(ImageOperation):
    """Class to apply Gaussian filter to image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor.

        Args:
            kernel_size (int): Size of the Gaussian kernel. Default is 5.
            sigma (float): Standard deviation of the Gaussian kernel. Default is 1.0.
        """
        raise NotImplementedError("Gaussian filter is not yet implemented.")

    def apply(self, x: Tensor) -> Tensor:
        """Apply the Gaussian filter to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Filtered image tensor of shape (B, C, H, W).
        """
        raise NotImplementedError("Gaussian filter is not implemented yet.")
