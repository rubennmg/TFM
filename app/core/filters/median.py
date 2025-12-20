from kornia import filters as K_e
from torch import Tensor

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class MedianFilter(ImageOperation):
    """Class to apply Median Filter to image tensors.
    Uses Kornia's median_blur implementation.

    See: https://kornia.readthedocs.io/en/latest/filters.html#kornia.filters.median_blur

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self, kernel_size: int = 3):
        """Class constructor.

        Args:
            kernel_size (int): Size of the median filter kernel. Defaults to 3.

        Raises:
            ValueError: If kernel_size is not a positive odd integer.
        """
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kernel size must be a positive odd integer.")

        self.kernel_size = (kernel_size, kernel_size)

    def apply(self, x: Tensor) -> Tensor:
        """Apply the Median Filter to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (C, H, W).

        Returns:
            Tensor: Filtered image tensor of shape (C, H, W).
        """
        T_u.assert_real_valued_tensor(x)

        out = K_e.median_blur(x.unsqueeze(0), self.kernel_size)

        return out.squeeze(0)
