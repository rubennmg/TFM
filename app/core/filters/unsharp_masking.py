from kornia import filters as K_f
from torch import Tensor

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class UnsharpMasking(ImageOperation):
    """Class to apply Unsharp Masking to image tensors.
    Uses Kornia's unsharp_mask implementation.

    See: https://kornia.readthedocs.io/en/latest/filters.html#kornia.filters.unsharp_mask

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self, kernel_size: float = 3, sigma: float = 1.5):
        """Class constructor.

        Args:
            kernel_size (float): Size of the kernel. Default is 3.
            sigma (float): Standard deviation of the kernel. Default is 1.5.
        """
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("kernel_size must be a positive odd integer.")
        if sigma <= 0.0:
            raise ValueError("sigma must be a positive float.")

        self.kernel_size = (int(kernel_size), int(kernel_size))
        self.sigma = (sigma, sigma)

    def apply(self, x: Tensor) -> Tensor:
        """Apply Unsharp Masking to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (C, H, W).

        Returns:
            Tensor: Filtered image tensor of shape (C, H, W).
        """
        T_u.assert_real_valued_tensor(x)

        out = K_f.unsharp_mask(
            x.unsqueeze(0), kernel_size=self.kernel_size, sigma=self.sigma
        )

        return out.squeeze(0)
