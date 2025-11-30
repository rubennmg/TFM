import torch
from torch import Tensor

from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class RGB8ToReal(ImageOperation):
    """Class to convert RGB8 image tensors to real-valued format.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    target_tensor = "tensor"
    updates_debayer_state = False

    def __init__(self):
        """Class constructor

        This operation does not require any parameters.
        """
        pass

    def apply(self, x: Tensor) -> Tensor:
        """Convert RGB8 image tensor to real-valued format.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Image tensor in real-valued format of shape (B, C, H, W).
        """
        if x.dtype != torch.uint8:
            raise ValueError("Input tensor must be of type uint8.")

        x_real = x.to(torch.float32).div(255.0)

        return x_real
