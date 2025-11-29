from torch import Tensor

from torchvision.transforms import GaussianBlur

from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class GaussianFilter(ImageOperation):
    """Class to apply Gaussian filter to image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    target_tensor = "original_tensor"
    updates_debayer_state = False

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        """Class constructor.

        Args:
            kernel_size (int): Size of the Gaussian kernel. Default is 5.
            sigma (float): Standard deviation of the Gaussian kernel. Default is 1.0.
        """
        if kernel_size % 2 == 0 or kernel_size <= 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if sigma <= 0:
            raise ValueError("sigma must be a positive float.")

        self.blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def apply(self, x: Tensor) -> Tensor:
        """Apply the Gaussian filter to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Filtered image tensor of shape (B, C, H, W).
        """
        return self.blur(x)
