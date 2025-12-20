import torch
from torch import Tensor
from torchvision.transforms.v2 import functional as F

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class RGB8ToReal(ImageOperation):
    """Class to convert RGB8 image tensors to real-valued format.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor

        This operation does not require any parameters.
        """
        pass

    def apply(self, x: Tensor) -> Tensor:
        """Convert RGB8 image tensor to real-valued format.

        Args:
            x (Tensor): Input image tensor of shape (C, H, W).

        Returns:
            Tensor: Image tensor in real-valued format of shape (C, H, W).
        """
        T_u.assert_integer_valued_tensor(x)

        x_real = F.convert_image_dtype(x, dtype=torch.float32)

        T_u.assert_real_valued_tensor(x_real)

        return x_real
