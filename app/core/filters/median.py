from torch import Tensor
from torchvision.transforms.v2 import functional as F

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class MedianFilter(ImageOperation):
    """Class to apply Median Filter to image tensors.

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

        self.kernel_size = kernel_size

    def apply(self, x: Tensor) -> Tensor:
        """Apply the Median Filter to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Filtered image tensor of shape (B, C, H, W).
        """
        T_u.assert_real_valued_tensor(x)

        b, c, h, w = x.shape
        k = self.kernel_size

        pad = k // 2
        x_padded = F.pad(x, [pad, pad, pad, pad])

        patches = x_padded.unfold(2, k, 1).unfold(3, k, 1)
        patches = patches.contiguous().view(b, c, h, w, k * k)

        median = patches.median(dim=-1).values

        return median
