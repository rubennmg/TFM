import math
from torch import Tensor
from torchvision import transforms as T

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class GaussianFilter(ImageOperation):
    """Class to apply Gaussian filter to image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self, sigma: float = 1.0):
        """Class constructor.

        Args:
            sigma (float): Standard deviation of the Gaussian kernel. Default is 1.0.
        """
        if sigma <= 0:
            raise ValueError("sigma must be a positive float.")

        kernel_size = self._calculate_kernel_size(sigma)

        self.blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def apply(self, x: Tensor) -> Tensor:
        """Apply the Gaussian filter to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (C, H, W).

        Returns:
            Tensor: Filtered image tensor of shape (C, H, W).
        """
        T_u.assert_real_valued_tensor(x)

        return self.blur(x)

    def _calculate_kernel_size(self, sigma: float, k: float = 3.0) -> int:
        """
        Calculate the kernel size based on the standard deviation and a scaling factor.

        Args:
            sigma (float): Standard deviation of the Gaussian kernel.
            k (float): Truncation factor (default 3 → ~99.7% coverage)

        Returns:
            int: Calculated kernel size, which is an odd integer.
        """

        return 2 * math.ceil(k * sigma) + 1
