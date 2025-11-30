import torch
from torch import Tensor

from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class RealToRGB8(ImageOperation):
    """Class to convert real-valued image tensors to RGB8 format.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    target_tensor = "tensor"
    updates_debayer_state = False

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
        if x.dtype != torch.float32 and x.dtype != torch.float64:
            raise ValueError(
                "Input tensor must be of real-valued type (float32 or float64)."
            )

        x_clamped = x.clamp(0.0, 1.0)
        x_rgb8 = torch.mul(x_clamped, 255.0).round().to(torch.uint8)

        return x_rgb8
