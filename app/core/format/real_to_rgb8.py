import torch
from torch import Tensor
from torchvision.transforms import functional as F

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class RealToRGB8(ImageOperation):
    """Class to convert real-valued image tensors to RGB8 format.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor.

        This operation does not require any parameters.
        """
        pass

    def apply(self, x: Tensor) -> Tensor:
        """Convert real-valued image tensor to RGB8 format.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Image tensor in RGB8 format of shape (B, C, H, W).
        """
        T_u.assert_real_valued_tensor(x)

        x_rgb8 = F.convert_image_dtype(x, dtype=torch.uint8)

        T_u.assert_integer_valued_tensor(x_rgb8)

        return x_rgb8
