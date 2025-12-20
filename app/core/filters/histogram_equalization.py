import torch
from torch import Tensor
from torchvision.transforms.v2 import functional as F

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class HistogramEqualization(ImageOperation):
    """Class to apply Histogram Equalization to image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor."""
        pass

    def apply(self, x: Tensor) -> Tensor:
        """Apply Histogram Equalization to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, 1, H, W) or (B, 3, H, W).

        Returns:
            Tensor: Image tensor after applying Histogram Equalization of shape (B, 1, H, W) or (B, 3, H, W).
        """
        if x.shape[1] == 1:
            T_u.assert_grayscale_image_tensor(x)
        elif x.shape[1] == 3:
            T_u.assert_color_image_tensor(x)
        else:
            raise ValueError(
                f"Expected image with 1 or 3 channels, got {x.shape[1]} channels."
            )

        T_u.assert_integer_valued_tensor(x)

        if x.dtype != torch.uint8:
            raise ValueError(
                f"Histogram Equalization requires uint8 images, got {x.dtype}."
            )

        return F.equalize(x)
